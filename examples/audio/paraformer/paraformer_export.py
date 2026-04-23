import argparse
from pathlib import Path

from funasr import AutoModel

from export_utils import default_manifest_path, write_export_manifest
from mask_safe_export import configure_static_export_from_audio, install_mask_safe_export


def _check_conda_env(expected: str, strict: bool) -> None:
    if not expected:
        return
    import os

    current = os.environ.get("CONDA_DEFAULT_ENV")
    if not current:
        message = f"CONDA_DEFAULT_ENV is not set; expected '{expected}'."
        if strict:
            raise RuntimeError(message)
        print(f"[env] {message}")
        return
    if current != expected:
        message = f"expected conda env '{expected}', but found '{current}'."
        if strict:
            raise RuntimeError(message)
        print(f"[env] {message}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data02/datasets/funasr/Paraformer")
    parser.add_argument("--model-revision", default="v2.0.4")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--input", default="/data02/datasets/funasr/Paraformer/example/asr_example.wav")
    parser.add_argument("--skip-generate", action="store_true")
    parser.add_argument("--export-type", default="onnx")
    parser.add_argument("--output-dir", default="examples/audio/paraformer/exports/mask_safe")
    parser.add_argument("--export-manifest", default=str(default_manifest_path()))
    parser.add_argument("--no-export-manifest", action="store_true")
    parser.add_argument("--use-default-export", action="store_true")
    parser.add_argument("--static-shape-export", action="store_true")
    parser.add_argument("--expected-conda-env", default="xhquant")
    parser.add_argument("--strict-conda-env", action="store_true")
    parser.add_argument("--no-conda-check", action="store_true")
    args = parser.parse_args()

    if not args.no_conda_check:
        _check_conda_env(args.expected_conda_env, args.strict_conda_env)

    if not args.use_default_export:
        install_mask_safe_export()
        print("[run] enabled local mask-safe export hook")

    model = AutoModel(model=args.model, model_revision=args.model_revision, device=args.device, disable_update=True)
    if args.static_shape_export and not args.use_default_export:
        configure_static_export_from_audio(model, args.input)
        print("[run] configured static-shape export from input sample")
    if not args.skip_generate:
        print("[run] generate:", model.generate(input=args.input, chunk_size=[0, 10, 5], encoder_chunk_look_back=4, decoder_chunk_look_back=1))
    result = model.export(type=args.export_type, output_dir=args.output_dir)
    print("[run] export result:", result)

    if not args.no_export_manifest:
        manifest = write_export_manifest(result, Path(args.export_manifest), export_dir_hint=args.output_dir)
        print("[run] export manifest:", manifest)


if __name__ == "__main__":
    main()