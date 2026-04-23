import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config, get_root_logger, xhquant_init
from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
DEFAULT_MODEL_DIR = Path("/data02/datasets/funasr/FSMN")
DEFAULT_AUDIO_PATH = DEFAULT_MODEL_DIR / "example" / "vad_example.wav"
DEFAULT_ONNX_PATH = THIS_DIR / "fsmn_vad.onnx"
DEFAULT_SIMPLIFIED_ONNX_PATH = THIS_DIR / "fsmn_vad_simplified.onnx"
DEFAULT_WORK_DIR = REPO_ROOT / "work_dirs" / "fsmn"
DEFAULT_CHUNK_SIZE = 120000
INPUT_NAMES = ["speech", "in_cache0", "in_cache1", "in_cache2", "in_cache3"]
OUTPUT_NAMES = ["logits", "out_cache0", "out_cache1", "out_cache2", "out_cache3"]


def load_auto_model(model_dir: Path, model_revision: str):
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    vad_pipeline = pipeline(
        task=Tasks.voice_activity_detection,
        model=str(model_dir),
        model_revision=model_revision,
    )
    return vad_pipeline.model.model


def prepare_audio_and_feats(auto_model, audio_path: Path, fs: int):
    from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video

    frontend = auto_model.kwargs["frontend"]
    cfg = {"is_final": True, "is_streaming_input": False}
    audio_sample = load_audio_text_image_video(
        str(audio_path),
        fs=frontend.fs,
        audio_fs=fs,
        data_type="sound",
        tokenizer=auto_model.kwargs.get("tokenizer"),
        cache=cfg,
    )
    speech, speech_lengths = extract_fbank(
        audio_sample,
        data_type="sound",
        frontend=frontend,
        cache={},
        is_final=True,
    )
    return audio_sample, speech.float(), speech_lengths


def build_zero_caches(auto_model, dtype=torch.float32):
    encoder_conf = auto_model.kwargs["encoder_conf"]
    cache_frames = encoder_conf["lorder"] + encoder_conf["rorder"] - 1
    proj_dim = encoder_conf["proj_dim"]
    layers = encoder_conf["fsmn_layers"]
    return tuple(torch.zeros(1, proj_dim, cache_frames, 1, dtype=dtype) for _ in range(layers))


def inspect_onnx(model_path: Path) -> None:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type}, shape={inp.shape}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type}, shape={out.shape}")


def simplify_onnx_if_needed(logger, onnx_path: Path, simplified_onnx_path: Path):
    if simplified_onnx_path.exists():
        return

    import onnxsim

    logger.info("Simplifying ONNX model...")
    model = onnx.load(str(onnx_path))
    simplified_model, check = onnxsim.simplify(model)
    onnx.save(simplified_model, str(simplified_onnx_path))
    logger.info(f"Simplified ONNX saved to {simplified_onnx_path}, check={check}")


class TorchEncoderWrapper(nn.Module):
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self._device_anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.chunk_logits = []
        self.encoder = encoder

    def reset(self):
        self.chunk_logits = []

    def forward(self, feats: torch.Tensor, cache=None):
        logits = self.encoder(feats, cache=cache)
        self.chunk_logits.append(logits.detach().cpu().float())
        return logits


class OnnxEncoderWrapper(nn.Module):
    def __init__(self, onnx_path: Path, zero_caches):
        super().__init__()
        self._device_anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.chunk_logits = []
        self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.zero_caches = tuple(cache.detach().cpu().numpy() for cache in zero_caches)

    def reset(self):
        self.chunk_logits = []

    def forward(self, feats: torch.Tensor, cache=None):
        cache = {} if cache is None else cache
        ort_inputs = {"speech": feats.detach().cpu().float().numpy()}
        for idx in range(4):
            cache_key = f"cache_layer_{idx}"
            ort_inputs[f"in_cache{idx}"] = (
                cache[cache_key].detach().cpu().float().numpy()
                if cache_key in cache
                else self.zero_caches[idx]
            )
        outputs = self.session.run(None, ort_inputs)
        logits = torch.from_numpy(outputs[0]).float()
        for idx, output in enumerate(outputs[1:]):
            cache[f"cache_layer_{idx}"] = torch.from_numpy(output).float()
        self.chunk_logits.append(logits.clone())
        return logits


class HmonnxEncoderWrapper(nn.Module):
    def __init__(self, hmonnx_path: Path, exec_device: str, save_golden_dir: Path | None = None):
        super().__init__()
        self._device_anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.chunk_logits = []
        self.session = HMONNXInference(str(hmonnx_path))
        self.session.exec_device = exec_device
        self.session.to(exec_device)
        if save_golden_dir is not None:
            self.session.save_golden = True
            self.session.save_golden_dir = str(save_golden_dir)
        self.input_infos = list(self.session.inputs)

    def reset(self):
        self.chunk_logits = []

    def _to_session_tensor(self, tensor: torch.Tensor, input_index: int) -> torch.Tensor:
        info = self.input_infos[input_index]
        return tensor.to(device=self.session.exec_device, dtype=info.dtype)

    def forward(self, feats: torch.Tensor, cache=None):
        cache = {} if cache is None else cache
        input_tensors = [self._to_session_tensor(feats.detach().cpu().float(), 0)]
        for idx in range(4):
            cache_key = f"cache_layer_{idx}"
            cache_tensor = cache.get(cache_key)
            if cache_tensor is None:
                info = self.input_infos[idx + 1]
                cache_tensor = torch.zeros(tuple(info.shape), dtype=torch.float32)
            input_tensors.append(self._to_session_tensor(cache_tensor.detach().cpu().float(), idx + 1))

        outputs = self.session.forward(*input_tensors)
        if not isinstance(outputs, (tuple, list)):
            outputs = (outputs,)

        logits = outputs[0].detach().float().cpu()
        for idx, output in enumerate(outputs[1:]):
            cache[f"cache_layer_{idx}"] = output.detach().float().cpu()
        self.chunk_logits.append(logits.clone())
        return logits


def run_vad(auto_model, audio_path: Path, chunk_size: int, encoder_wrapper: nn.Module | None = None):
    model = auto_model.model
    original_encoder = model.encoder
    if encoder_wrapper is not None:
        encoder_wrapper.reset()
        model.encoder = encoder_wrapper
    try:
        result = auto_model.generate(input=str(audio_path), chunk_size=chunk_size)
        records = [] if encoder_wrapper is None else [tensor.numpy() for tensor in encoder_wrapper.chunk_logits]
    finally:
        model.encoder = original_encoder
    return result, records


def max_abs_diff(lhs: np.ndarray, rhs: np.ndarray) -> float:
    return float(np.max(np.abs(lhs - rhs)))


def summarize_diffs(lhs_records, rhs_records) -> tuple[float, float]:
    if len(lhs_records) != len(rhs_records):
        raise ValueError(f"Chunk count mismatch: {len(lhs_records)} vs {len(rhs_records)}")

    chunk_diffs = [max_abs_diff(lhs, rhs) for lhs, rhs in zip(lhs_records, rhs_records, strict=True)]
    flat_lhs = np.concatenate([item.reshape(-1, item.shape[-1]) for item in lhs_records], axis=0)
    flat_rhs = np.concatenate([item.reshape(-1, item.shape[-1]) for item in rhs_records], axis=0)
    return max(chunk_diffs), max_abs_diff(flat_lhs, flat_rhs)


def extract_segments(result) -> list:
    if not result:
        return []
    return result[0].get("value", [])


def ensure_hmonnx_input_compatibility(audio_num_samples: int, fs: int, chunk_size: int):
    duration_ms = audio_num_samples * 1000.0 / fs
    if duration_ms > chunk_size:
        raise ValueError(
            f"Audio duration {duration_ms:.1f} ms exceeds chunk_size {chunk_size} ms. "
            "Increase --chunk-size so the sample runs as a single chunk before HMONNX verification."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-revision", type=str, default="v2.0.4")
    parser.add_argument("--input", type=Path, default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--simplified-onnx-path", type=Path, default=DEFAULT_SIMPLIFIED_ONNX_PATH)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--quant-type", type=str, default="w8a8_sefp")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save-golden", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--inspect-onnx", action="store_true")
    args = parser.parse_args()

    xhquant_init(None, debug=False)
    logger = get_root_logger()
    args.work_dir.mkdir(parents=True, exist_ok=True)

    if not args.onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {args.onnx_path}")

    if args.inspect_onnx:
        inspect_onnx(args.onnx_path)

    simplify_onnx_if_needed(logger, args.onnx_path, args.simplified_onnx_path)

    auto_model = load_auto_model(args.model_dir, args.model_revision)
    audio_sample, speech, _ = prepare_audio_and_feats(auto_model, args.input, args.fs)
    ensure_hmonnx_input_compatibility(int(audio_sample.shape[0]), args.fs, args.chunk_size)
    fixed_frames = int(speech.shape[1])
    zero_caches = build_zero_caches(auto_model)

    logger.info(f"Audio: {args.input}")
    logger.info(f"Fixed feature length: {fixed_frames}")

    target_device = DeviceType.XH2a
    quant_scheme = QuantScheme(target_device=target_device, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_path = args.work_dir / "hmonnx" / f"{args.onnx_path.stem}_{fixed_frames}f_{target_device}_{args.quant_type}.onnx"
    hmonnx_path.parent.mkdir(parents=True, exist_ok=True)

    if not hmonnx_path.exists():
        logger.info(f"Converting ONNX to hmonnx: {hmonnx_path}")
        convert_onnx_to_hmonnx(
            str(args.simplified_onnx_path),
            (speech, *zero_caches),
            target_device,
            str(hmonnx_path),
            quant_config=quant_config,
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
        )

    if args.no_eval and not args.save_golden:
        return

    fp_result, fp_records = run_vad(auto_model, args.input, args.chunk_size, TorchEncoderWrapper(auto_model.model.encoder))
    onnx_result, onnx_records = run_vad(auto_model, args.input, args.chunk_size, OnnxEncoderWrapper(args.simplified_onnx_path, zero_caches))

    golden_dir = args.work_dir / "hmonnx" / f"{args.onnx_path.stem}_{fixed_frames}f_golden" if args.save_golden else None
    hmonnx_result, hmonnx_records = run_vad(
        auto_model,
        args.input,
        args.chunk_size,
        HmonnxEncoderWrapper(hmonnx_path, args.device, save_golden_dir=golden_dir),
    )

    if not args.no_eval:
        onnx_chunk_diff, onnx_full_diff = summarize_diffs(fp_records, onnx_records)
        hmonnx_chunk_diff, hmonnx_full_diff = summarize_diffs(fp_records, hmonnx_records)

        fp_segments = extract_segments(fp_result)
        onnx_segments = extract_segments(onnx_result)
        hmonnx_segments = extract_segments(hmonnx_result)

        logger.info(f"FP segments: {fp_segments}")
        logger.info(f"ONNX segments: {onnx_segments}")
        logger.info(f"HMONNX segments: {hmonnx_segments}")
        logger.info(
            f"Segments equal: fp-vs-onnx={fp_segments == onnx_segments}, "
            f"fp-vs-hmonnx={fp_segments == hmonnx_segments}"
        )
        logger.info(f"Logits max abs diff: fp-vs-onnx chunk={onnx_chunk_diff} full={onnx_full_diff}")
        logger.info(f"Logits max abs diff: fp-vs-hmonnx chunk={hmonnx_chunk_diff} full={hmonnx_full_diff}")

    if args.save_golden:
        logger.info(f"Golden saved to {golden_dir}")


if __name__ == "__main__":
    main()