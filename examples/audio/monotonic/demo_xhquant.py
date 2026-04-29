import argparse
import copy
import importlib
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from export_onnx import export_onnx, read_text, simplify_onnx
from model_utils import extract_inputs, load_auto_model

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config, get_root_logger, xhquant_init
from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

REPO_ROOT = THIS_DIR.parents[2]
DEFAULT_MODEL_DIR = Path("/data02/datasets/funasr/Monotonic")
DEFAULT_AUDIO_PATH = DEFAULT_MODEL_DIR / "example" / "asr_example.wav"
DEFAULT_TEXT_PATH = DEFAULT_MODEL_DIR / "example" / "text.txt"
DEFAULT_ONNX_PATH = THIS_DIR / "monotonic_timestamp.onnx"
DEFAULT_SIMPLIFIED_ONNX_PATH = THIS_DIR / "monotonic_timestamp_simplified.onnx"
DEFAULT_WORK_DIR = REPO_ROOT / "work_dirs" / "monotonic"
def inspect_onnx(model_path: Path) -> None:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type}, shape={inp.shape}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type}, shape={out.shape}")
def pad_speech(speech: torch.Tensor, fixed_frames: int) -> torch.Tensor:
    if speech.shape[1] > fixed_frames:
        raise ValueError(f"Speech frame length {speech.shape[1]} exceeds fixed_frames {fixed_frames}")
    if speech.shape[1] == fixed_frames:
        return speech
    padded = torch.zeros((speech.shape[0], fixed_frames, speech.shape[2]), dtype=speech.dtype)
    padded[:, : speech.shape[1], :] = speech
    return padded


def postprocess_timestamp(us_alphas, us_peaks, encoder_out_lens, token_list):
    from funasr.utils import postprocess_utils
    from funasr.utils.timestamp_tools import ts_prediction_lfr6_standard

    valid_length = int(encoder_out_lens[0]) * 3
    alpha = us_alphas[0][:valid_length]
    peak = us_peaks[0][:valid_length]
    _, timestamp = ts_prediction_lfr6_standard(alpha, peak, copy.copy(token_list))
    text_postprocessed, time_stamp_postprocessed, _ = postprocess_utils.sentence_postprocess(token_list, timestamp)
    return {"text": text_postprocessed, "timestamp": time_stamp_postprocessed}


class TorchTimestampWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model.cpu().float().eval()

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor, token_num: torch.Tensor):
        with torch.no_grad():
            encoder_out, encoder_out_lens = self.model.encode(speech, speech_lengths)
            _, _, us_alphas, us_peaks = self.model.calc_predictor_timestamp(
                encoder_out,
                encoder_out_lens,
                token_num=token_num.to(dtype=encoder_out_lens.dtype),
            )
        return us_alphas.cpu(), us_peaks.cpu(), encoder_out_lens.to(dtype=torch.int32).cpu()


class OnnxTimestampWrapper(nn.Module):
    def __init__(self, onnx_path: Path):
        super().__init__()
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.input_names = [item.name for item in self.session.get_inputs()]

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor, token_num: torch.Tensor):
        ort_inputs = {"speech": speech.numpy().astype(np.float32), "token_num": token_num.numpy().astype(np.int32)}
        if "speech_lengths" in self.input_names:
            ort_inputs["speech_lengths"] = speech_lengths.numpy().astype(np.int32)
        outputs = self.session.run(None, ort_inputs)
        return tuple(torch.from_numpy(item) for item in outputs)


class HmonnxTimestampWrapper(nn.Module):
    def __init__(self, hmonnx_path: Path, exec_device: str, save_golden_dir: Path | None = None):
        super().__init__()
        self.session = HMONNXInference(str(hmonnx_path))
        self.session.exec_device = exec_device
        self.session.to(exec_device)
        if save_golden_dir is not None:
            self.session.save_golden = True
            self.session.save_golden_dir = str(save_golden_dir)
        self.input_infos = list(self.session.inputs)

    def _to_session_tensor(self, tensor: torch.Tensor, index: int) -> torch.Tensor:
        info = self.input_infos[index]
        return tensor.to(device=self.session.exec_device, dtype=info.dtype)

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor, token_num: torch.Tensor):
        input_tensors = [self._to_session_tensor(speech, 0)]
        if len(self.input_infos) == 3:
            input_tensors.append(self._to_session_tensor(speech_lengths, 1))
            input_tensors.append(self._to_session_tensor(token_num, 2))
        else:
            input_tensors.append(self._to_session_tensor(token_num, 1))
        outputs = self.session.forward(*input_tensors)
        if not isinstance(outputs, (list, tuple)):
            raise TypeError(f"Unexpected HMONNX outputs type: {type(outputs)}")
        return tuple(item.detach().float().cpu() for item in outputs)


def run_hmonnx_with_fallback(
    hmonnx_path: Path,
    exec_device: str,
    speech: torch.Tensor,
    speech_lengths: torch.Tensor,
    token_num: torch.Tensor,
    logger,
    save_golden_dir: Path | None = None,
):
    hm_runner = HmonnxTimestampWrapper(hmonnx_path, exec_device, save_golden_dir=save_golden_dir)
    try:
        return hm_runner.forward(speech, speech_lengths, token_num), exec_device
    except ValueError as exc:
        if "[NonFinite]" not in str(exc) or exec_device == "cpu":
            raise
        logger.warning("HMONNX inference on %s hit NonFinite. Falling back to CPU for validation/golden export.", exec_device)
        hm_runner = HmonnxTimestampWrapper(hmonnx_path, "cpu", save_golden_dir=save_golden_dir)
        return hm_runner.forward(speech, speech_lengths, token_num), "cpu"


def compare_results(lhs, rhs) -> bool:
    return lhs["text"] == rhs["text"] and lhs["timestamp"] == rhs["timestamp"]


def max_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.max(np.abs(lhs - rhs)))


def ensure_onnx_artifacts(args, logger):
    export_onnx(
        model_dir=args.model_dir,
        model_revision=args.model_revision,
        audio_path=args.audio,
        text=args.text,
        onnx_path=args.onnx_path,
        opset_version=14,
    )
    simplify_onnx(args.onnx_path, args.simplified_onnx_path)
    logger.info(f"Refreshed ONNX artifacts: {args.simplified_onnx_path}")


def convert_onnx_to_hmonnx_no_simplify(
    onnx_path: str,
    dummy_inputs,
    target_device,
    out_hmonnx_path: str,
    quant_config,
    input_names,
    output_names,
):
    ptq_module = importlib.import_module("xhquant.api.ptq_export_hmonnx")
    original_to_frontend_graph = ptq_module.to_frontend_graph

    def patched_to_frontend_graph(input_model, frontend_type, example_input, enable_fuse=True, **kwargs):
        kwargs.setdefault("simplify", False)
        return original_to_frontend_graph(input_model, frontend_type, example_input, enable_fuse, **kwargs)

    ptq_module.to_frontend_graph = patched_to_frontend_graph
    try:
        convert_onnx_to_hmonnx(
            onnx_path,
            dummy_inputs,
            target_device,
            out_hmonnx_path,
            quant_config=quant_config,
            input_names=input_names,
            output_names=output_names,
        )
    finally:
        ptq_module.to_frontend_graph = original_to_frontend_graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-revision", type=str, default="v2.0.4")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text-path", type=Path, default=DEFAULT_TEXT_PATH)
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--simplified-onnx-path", type=Path, default=DEFAULT_SIMPLIFIED_ONNX_PATH)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--quant-type", type=str, default="w8a8_sefp")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-golden", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--inspect-onnx", action="store_true")
    parser.add_argument("--reuse-onnx", action="store_true")
    args = parser.parse_args()

    args.text = read_text(args.text, args.text_path)

    xhquant_init(None, debug=False)
    logger = get_root_logger()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_onnx:
        if not args.onnx_path.exists():
            raise FileNotFoundError(f"Missing ONNX model: {args.onnx_path}")
    else:
        ensure_onnx_artifacts(args, logger)

    if args.inspect_onnx:
        inspect_onnx(args.simplified_onnx_path)

    auto_model = load_auto_model(args.model_dir, args.model_revision)
    speech, speech_lengths, token_num, token_list = extract_inputs(auto_model, args.audio, args.text)
    fixed_frames = int(speech.shape[1])
    padded_speech = pad_speech(speech, fixed_frames)
    logger.info(f"Fixed speech shape: {list(padded_speech.shape)}")
    logger.info(f"Token count (+eos): {int(token_num[0])}")

    target_device = DeviceType.XH2a
    quant_scheme = QuantScheme(target_device=target_device, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_path = args.work_dir / "hmonnx" / f"{args.onnx_path.stem}_{fixed_frames}f_{target_device}_{args.quant_type}.onnx"
    hmonnx_path.parent.mkdir(parents=True, exist_ok=True)

    if not hmonnx_path.exists():
        logger.info(f"Converting ONNX to hmonnx: {hmonnx_path}")
        session = ort.InferenceSession(str(args.simplified_onnx_path), providers=["CPUExecutionProvider"])
        input_names = [item.name for item in session.get_inputs()]
        dummy_inputs = (padded_speech, speech_lengths, token_num) if len(input_names) == 3 else (padded_speech, token_num)
        convert_onnx_to_hmonnx_no_simplify(
            str(args.simplified_onnx_path),
            dummy_inputs,
            target_device,
            str(hmonnx_path),
            quant_config,
            input_names,
            ["us_alphas", "us_peaks", "encoder_out_lens"],
        )

    if args.no_eval and not args.save_golden:
        return

    fp_runner = TorchTimestampWrapper(auto_model.model)
    onnx_runner = OnnxTimestampWrapper(args.simplified_onnx_path)
    golden_dir = args.work_dir / "hmonnx" / f"{args.onnx_path.stem}_{fixed_frames}f_golden" if args.save_golden else None

    fp_outputs = fp_runner.forward(padded_speech, speech_lengths, token_num)
    onnx_outputs = onnx_runner.forward(padded_speech, speech_lengths, token_num)
    hm_outputs, hm_exec_device = run_hmonnx_with_fallback(
        hmonnx_path,
        args.device,
        padded_speech,
        speech_lengths,
        token_num,
        logger,
        save_golden_dir=golden_dir,
    )

    fp_result = postprocess_timestamp(*fp_outputs, token_list)
    onnx_result = postprocess_timestamp(*onnx_outputs, token_list)
    hm_result = postprocess_timestamp(*hm_outputs, token_list)

    if not args.no_eval:
        logger.info(f"FP text: {fp_result['text']}")
        logger.info(f"ONNX text: {onnx_result['text']}")
        logger.info(f"HMONNX text: {hm_result['text']}")
        logger.info(f"FP timestamp: {fp_result['timestamp']}")
        logger.info(f"ONNX timestamp: {onnx_result['timestamp']}")
        logger.info(f"HMONNX timestamp: {hm_result['timestamp']}")
        logger.info(f"HMONNX exec device: {hm_exec_device}")
        logger.info(
            "Equal: fp-vs-onnx=%s, fp-vs-hmonnx=%s",
            compare_results(fp_result, onnx_result),
            compare_results(fp_result, hm_result),
        )
        logger.info(
            "Max abs diff: us_alphas fp-vs-onnx=%s, fp-vs-hmonnx=%s",
            max_abs_diff(fp_outputs[0].numpy(), onnx_outputs[0].numpy()),
            max_abs_diff(fp_outputs[0].numpy(), hm_outputs[0].numpy()),
        )
        logger.info(
            "Max abs diff: us_peaks fp-vs-onnx=%s, fp-vs-hmonnx=%s",
            max_abs_diff(fp_outputs[1].numpy(), onnx_outputs[1].numpy()),
            max_abs_diff(fp_outputs[1].numpy(), hm_outputs[1].numpy()),
        )

    if args.save_golden:
        logger.info(f"Golden saved to {golden_dir}")


if __name__ == "__main__":
    main()