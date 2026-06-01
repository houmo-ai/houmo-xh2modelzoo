import argparse
from pathlib import Path

import onnx
import onnxruntime as ort
import torch

DEFAULT_MODEL_DIR = Path("/data02/datasets/funasr/FSMN")
THIS_DIR = Path(__file__).resolve().parent
DEFAULT_AUDIO_PATH = DEFAULT_MODEL_DIR / "example" / "vad_example.wav"
DEFAULT_ONNX_PATH = THIS_DIR / "fsmn_vad.onnx"
DEFAULT_SIMPLIFIED_ONNX_PATH = THIS_DIR / "fsmn_vad_simplified.onnx"


def inspect_onnx(model_path: Path) -> None:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type}, shape={inp.shape}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type}, shape={out.shape}")


def load_auto_model(model_dir: Path, model_revision: str):
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    vad_pipeline = pipeline(
        task=Tasks.voice_activity_detection,
        model=str(model_dir),
        model_revision=model_revision,
    )
    return vad_pipeline.model.model


def extract_feature_length(auto_model, audio_path: Path, fs: int) -> int:
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
    speech, _ = extract_fbank(
        audio_sample,
        data_type="sound",
        frontend=frontend,
        cache={},
        is_final=True,
    )
    return int(speech.shape[1])


def export_onnx(
    model_dir: Path,
    model_revision: str,
    audio_path: Path,
    onnx_path: Path,
    opset_version: int,
    fs: int,
) -> int:
    auto_model = load_auto_model(model_dir, model_revision)
    fixed_frames = extract_feature_length(auto_model, audio_path, fs)
    export_kwargs = dict(auto_model.kwargs)
    export_kwargs.pop("model", None)

    export_model = auto_model.model.export(type="onnx", **export_kwargs).cpu().float().eval()
    dummy_inputs = export_model.export_dummy_inputs(frame=fixed_frames)

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        dummy_inputs,
        str(onnx_path),
        verbose=False,
        do_constant_folding=True,
        opset_version=opset_version,
        input_names=export_model.export_input_names(),
        output_names=export_model.export_output_names(),
        dynamic_axes=export_model.export_dynamic_axes(),
    )

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    print(f"Exported ONNX model to {onnx_path}")
    print(f"Fixed feature length for {audio_path}: {fixed_frames}")
    inspect_onnx(onnx_path)
    return fixed_frames


def simplify_onnx(onnx_path: Path, simplified_onnx_path: Path) -> None:
    import onnxsim

    model = onnx.load(str(onnx_path))
    simplified_model, check = onnxsim.simplify(model)
    simplified_onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(simplified_model, str(simplified_onnx_path))
    print(f"Simplified ONNX saved to {simplified_onnx_path}, check={check}")
    inspect_onnx(simplified_onnx_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-revision", type=str, default="v2.0.4")
    parser.add_argument("--input", type=Path, default=DEFAULT_AUDIO_PATH)
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--simplified-onnx-path", type=Path, default=DEFAULT_SIMPLIFIED_ONNX_PATH)
    parser.add_argument("--opset-version", type=int, default=14)
    parser.add_argument("--step", type=str, default="all", choices=["export", "simplify", "all"])
    args = parser.parse_args()

    if args.step in ["export", "all"]:
        export_onnx(
            model_dir=args.model_dir,
            model_revision=args.model_revision,
            audio_path=args.input,
            onnx_path=args.onnx_path,
            opset_version=args.opset_version,
            fs=args.fs,
        )

    if args.step in ["simplify", "all"]:
        simplify_onnx(args.onnx_path, args.simplified_onnx_path)


if __name__ == "__main__":
    main()