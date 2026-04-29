import argparse
import shutil
from pathlib import Path

import onnx
import onnxruntime as ort
import torch
from torch import nn

from model_utils import build_export_encoder, extract_inputs, load_auto_model, read_text

DEFAULT_MODEL_DIR = Path("/data02/datasets/funasr/Monotonic")
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO_PATH = DEFAULT_MODEL_DIR / "example" / "asr_example.wav"
DEFAULT_TEXT_PATH = DEFAULT_MODEL_DIR / "example" / "text.txt"
DEFAULT_ONNX_PATH = THIS_DIR / "monotonic_timestamp.onnx"
DEFAULT_SIMPLIFIED_ONNX_PATH = THIS_DIR / "monotonic_timestamp_simplified.onnx"


def lift_tensor_constants_to_initializers(onnx_model: onnx.ModelProto) -> int:
    from onnx import numpy_helper

    existing_initializers = {init.name for init in onnx_model.graph.initializer}
    kept_nodes = []
    lifted = 0
    for node in onnx_model.graph.node:
        if node.op_type != "Constant" or len(node.output) != 1:
            kept_nodes.append(node)
            continue

        value_attr = next((attr for attr in node.attribute if attr.name == "value" and attr.HasField("t")), None)
        if value_attr is None:
            kept_nodes.append(node)
            continue

        output_name = node.output[0]
        if output_name not in existing_initializers:
            tensor = numpy_helper.from_array(numpy_helper.to_array(value_attr.t), name=output_name)
            onnx_model.graph.initializer.append(tensor)
            existing_initializers.add(output_name)
        lifted += 1

    del onnx_model.graph.node[:]
    onnx_model.graph.node.extend(kept_nodes)
    return lifted


def rewrite_square_pow_nodes(onnx_path: Path) -> None:
    import numpy as np
    from onnx import numpy_helper

    onnx_model = onnx.load(str(onnx_path))
    lifted = lift_tensor_constants_to_initializers(onnx_model)
    constant_values = {init.name: numpy_helper.to_array(init) for init in onnx_model.graph.initializer}

    rewritten = 0
    for node in onnx_model.graph.node:
        if node.op_type != "Pow" or len(node.input) != 2:
            continue
        exponent_values = constant_values.get(node.input[1])
        if exponent_values is None:
            continue
        exponent_array = np.asarray(exponent_values)
        if exponent_array.size != 1 or not np.isclose(float(exponent_array.reshape(-1)[0]), 2.0):
            continue
        squared_input = node.input[0]
        del node.input[:]
        node.input.extend([squared_input, squared_input])
        node.op_type = "Mul"
        rewritten += 1

    onnx.save(onnx_model, str(onnx_path))
    if lifted:
        print(f"Lifted {lifted} Constant nodes to initializers in {onnx_path}")
    if rewritten:
        print(f"Rewrote {rewritten} Pow(x, 2) nodes as Mul(x, x) in {onnx_path}")


def inspect_onnx(model_path: Path) -> None:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type}, shape={inp.shape}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type}, shape={out.shape}")
class MonotonicTimestampWrapper(nn.Module):
    def __init__(self, model: nn.Module, export_encoder: nn.Module):
        super().__init__()
        self.model = model.cpu().float().eval()
        self.export_encoder = export_encoder.cpu().float().eval()
        self.predictor = StaticTimestampPredictorExport(
            self.model.predictor,
            encoder_frames=int(self.export_encoder.get_predictor_mask().shape[-1]),
        )

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor, token_num: torch.Tensor):
        encoder_out, encoder_out_lens = self.export_encoder(speech, speech_lengths)
        us_alphas, us_peaks = self.predictor(
            encoder_out,
            token_num=token_num,
        )
        return us_alphas, us_peaks, encoder_out_lens.to(dtype=torch.int32)


class StaticTimestampPredictorExport(nn.Module):
    def __init__(self, predictor: nn.Module, encoder_frames: int):
        super().__init__()
        self.predictor = predictor
        self.encoder_frames = encoder_frames
        self.upsample_times = int(predictor.upsample_times)
        self.upsampled_frames = encoder_frames * self.upsample_times
        self.smooth_factor2 = float(predictor.smooth_factor2)
        self.noise_threshold2 = float(predictor.noise_threshold2)
        self.threshold = float(predictor.threshold - 1e-4)
        self.use_cif1_cnn = bool(predictor.use_cif1_cnn)
        if hasattr(self.predictor, "blstm") and hasattr(self.predictor.blstm, "fixed_seq_len"):
            self.predictor.blstm.fixed_seq_len = self.upsampled_frames
        self.register_buffer(
            "upsample_mask",
            torch.ones((1, self.upsampled_frames), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer("initial_integrate", torch.zeros((1,), dtype=torch.float32), persistent=False)

    def forward(self, hidden: torch.Tensor, token_num: torch.Tensor):
        context = hidden.transpose(1, 2)
        queries = self.predictor.pad(context)
        output = torch.relu(self.predictor.cif_conv1d(queries))
        upsample_input = output if self.use_cif1_cnn else context
        output2 = self.predictor.upsample_cnn(upsample_input)
        output2 = output2.transpose(1, 2)
        if hasattr(self.predictor, "blstm"):
            output2, (_, _) = self.predictor.blstm(output2)
        alphas2 = torch.sigmoid(self.predictor.cif_output2(output2))
        alphas2 = torch.nn.functional.relu(alphas2 * self.smooth_factor2 - self.noise_threshold2)
        alphas2 = alphas2.squeeze(-1) * self.upsample_mask
        token_num = token_num.float()
        predicted_token_num = alphas2.sum(-1)
        alphas2 = alphas2 * (token_num / predicted_token_num).reshape(1, 1)
        us_peaks = self._compute_us_peaks(alphas2)
        return alphas2, us_peaks

    def _compute_us_peaks(self, alphas: torch.Tensor) -> torch.Tensor:
        integrate = self.initial_integrate
        fires = []
        for index in range(self.upsampled_frames):
            integrate = integrate + alphas[:, index]
            fires.append(integrate)
            fire_place = integrate >= self.threshold
            integrate = torch.where(fire_place, integrate - self.threshold, integrate)
        return torch.stack(fires, dim=1)


def export_onnx(
    model_dir: Path,
    model_revision: str,
    audio_path: Path,
    text: str,
    onnx_path: Path,
    opset_version: int,
) -> tuple[int, int]:
    auto_model = load_auto_model(model_dir, model_revision)
    speech, speech_lengths, token_num, _ = extract_inputs(auto_model, audio_path, text)

    fixed_frames = int(speech.shape[1])
    feature_dim = int(speech.shape[2])
    export_encoder = build_export_encoder(auto_model.model, max_seq_len=fixed_frames, feats_dim=feature_dim)
    export_model = MonotonicTimestampWrapper(auto_model.model, export_encoder)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        (speech, speech_lengths, token_num),
        str(onnx_path),
        verbose=False,
        do_constant_folding=True,
        opset_version=opset_version,
        input_names=["speech", "speech_lengths", "token_num"],
        output_names=["us_alphas", "us_peaks", "encoder_out_lens"],
    )

    rewrite_square_pow_nodes(onnx_path)
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"Exported ONNX model to {onnx_path}")
    print(f"Fixed speech shape for {audio_path}: [1, {fixed_frames}, {feature_dim}]")
    print(f"Fixed token_num for {audio_path}: {int(token_num[0])}")
    inspect_onnx(onnx_path)
    return fixed_frames, feature_dim


def simplify_onnx(onnx_path: Path, simplified_onnx_path: Path) -> None:
    simplified_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, simplified_onnx_path)
    print(f"Copied ONNX to {simplified_onnx_path} without external simplification")
    inspect_onnx(simplified_onnx_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-revision", type=str, default="v2.0.4")
    parser.add_argument("--audio", type=Path, default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--text-path", type=Path, default=DEFAULT_TEXT_PATH)
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--simplified-onnx-path", type=Path, default=DEFAULT_SIMPLIFIED_ONNX_PATH)
    parser.add_argument("--opset-version", type=int, default=14)
    parser.add_argument("--step", type=str, default="all", choices=["export", "simplify", "all"])
    args = parser.parse_args()

    text = read_text(args.text, args.text_path)
    if args.step in ["export", "all"]:
        export_onnx(
            model_dir=args.model_dir,
            model_revision=args.model_revision,
            audio_path=args.audio,
            text=text,
            onnx_path=args.onnx_path,
            opset_version=args.opset_version,
        )

    if args.step in ["simplify", "all"]:
        simplify_onnx(args.onnx_path, args.simplified_onnx_path)


if __name__ == "__main__":
    main()