import argparse
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Callable, Optional

from sanm_fp16_rotation_plugin import apply_sanm_fp16_rotation
from export_utils import default_manifest_path, write_export_manifest


_EXPORT_META_MODULE = "funasr.models.seaco_paraformer.export_meta"
_MISSING = object()


def _resolve_path(path_str: str) -> Path:
    return Path(os.path.expandvars(path_str)).expanduser()


def _ensure_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"{label} is not a file: {path}")


def _check_conda_env(expected: str, strict: bool) -> None:
    if not expected:
        return
    current = os.environ.get("CONDA_DEFAULT_ENV")
    if not current:
        msg = f"CONDA_DEFAULT_ENV is not set; expected '{expected}'."
        if strict:
            raise RuntimeError(msg)
        print(f"[env] {msg}")
        return
    if current != expected:
        msg = f"expected conda env '{expected}', but found '{current}'."
        if strict:
            raise RuntimeError(msg)
        print(f"[env] {msg}")


def _set_eval_mode(model) -> None:
    if hasattr(model, "eval"):
        model.eval()
        return
    if hasattr(model, "model") and hasattr(model.model, "eval"):
        model.model.eval()


def patch_export_meta(
    custom_path: str,
    *,
    require: bool = True,
    verbose: bool = True,
) -> Optional[Callable[[], None]]:
    target_module = _EXPORT_META_MODULE
    path = _resolve_path(custom_path)
    if not path.exists():
        msg = f"export_meta override not found: {path}"
        if require:
            raise FileNotFoundError(msg)
        if verbose:
            print(f"[patch] {msg}. Using upstream export_meta.")
        return None
    if not path.is_file():
        raise FileNotFoundError(f"export_meta override is not a file: {path}")

    spec = importlib.util.spec_from_file_location(target_module, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load export_meta override: {path}")

    prev_module = sys.modules.get(target_module)
    target_mod = importlib.util.module_from_spec(spec)
    sys.modules[target_module] = target_mod
    spec.loader.exec_module(target_mod)

    prev_parent = None
    prev_attr = _MISSING
    try:
        parent = importlib.import_module("funasr.models.seaco_paraformer")
        prev_parent = parent
        prev_attr = getattr(parent, "export_meta", _MISSING)
        setattr(parent, "export_meta", sys.modules[target_module])
    except Exception:
        prev_parent = None

    if verbose:
        print(f"[patch] using custom export_meta: {path}")

    def _restore() -> None:
        if prev_module is None:
            sys.modules.pop(target_module, None)
        else:
            sys.modules[target_module] = prev_module

        if prev_parent is not None:
            if prev_attr is _MISSING:
                if hasattr(prev_parent, "export_meta"):
                    delattr(prev_parent, "export_meta")
            else:
                setattr(prev_parent, "export_meta", prev_attr)

        if verbose:
            print("[patch] export_meta restored")

    return _restore


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(Path(__file__).with_name("asr_example_hotword.wav")),
        help="input wav path for a sanity run",
    )
    parser.add_argument(
        "--model",
        default="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--export-meta",
        default=str(Path(__file__).with_name("export_meta_seacoparaformer.py")),
        help="path to export_meta override file",
    )
    parser.add_argument(
        "--no-patch-export-meta",
        action="store_true",
        help="skip runtime export_meta patch",
    )
    parser.add_argument(
        "--require-export-meta",
        action="store_true",
        help="fail if export_meta override file is missing",
    )
    parser.add_argument(
        "--expected-conda-env",
        default="xhquant",
        help="warn if not running in this conda env (set empty to disable)",
    )
    parser.add_argument(
        "--strict-conda-env",
        action="store_true",
        help="fail if the expected conda env is not active",
    )
    parser.add_argument(
        "--no-conda-check",
        action="store_true",
        help="skip conda environment checks",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="skip generate() sanity runs",
    )
    parser.add_argument(
        "--skip-rotation",
        action="store_true",
        help="skip apply_sanm_fp16_rotation",
    )
    parser.add_argument(
        "--export-type",
        default="onnx",
        help="export type passed to model.export",
    )
    parser.add_argument(
        "--export-dir",
        default="",
        help="optional export directory (passed to model.export if supported)",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="enable quantized export",
    )
    parser.add_argument(
        "--export-manifest",
        default=str(default_manifest_path()),
        help="path to write export manifest json",
    )
    parser.add_argument(
        "--no-export-manifest",
        action="store_true",
        help="skip writing export manifest json",
    )
    args = parser.parse_args()

    if args.no_patch_export_meta and args.require_export_meta:
        raise ValueError("cannot combine --no-patch-export-meta with --require-export-meta")

    if not args.no_conda_check:
        _check_conda_env(args.expected_conda_env, args.strict_conda_env)

    input_path = _resolve_path(args.input)
    if not args.skip_generate:
        _ensure_file(input_path, "input wav")

    try:
        from funasr import AutoModel
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Failed to import funasr. Install it in your active environment first."
        ) from exc

    restore_patch = None
    if not args.no_patch_export_meta:
        restore_patch = patch_export_meta(
            args.export_meta,
            require=args.require_export_meta,
            verbose=True,
        )

    try:
        model = AutoModel(model=args.model, device=args.device)
        _set_eval_mode(model)

        if not args.skip_generate:
            res = model.generate(input=str(input_path))
            print("[run] generate result (before rotation):", res)

        if not args.skip_rotation:
            apply_sanm_fp16_rotation(
                model,
                step_identity=True,
                step_rmsnorm=True,
                step_fuse=True,
                step_hadamard=True,
                seed=0,
                step_scale=True,
                random_sign=True,
                output_linear_use_rt=False,
            )

            if not args.skip_generate:
                res = model.generate(input=str(input_path))
                print("[run] generate result (after rotation):", res)

        export_kwargs = {"quantize": args.quantize, "type": args.export_type}
        if args.export_dir:
            export_kwargs["output_dir"] = args.export_dir

        res = model.export(**export_kwargs)
        print("[run] export result:", res)

        if not args.no_export_manifest:
            manifest_path = _resolve_path(args.export_manifest)
            manifest = write_export_manifest(res, manifest_path, export_dir_hint=args.export_dir)
            print(f"[run] export manifest written: {manifest_path}")
            if not manifest.get("encoder_onnx") or not manifest.get("decoder_onnx"):
                print("[warn] manifest is missing encoder/decoder paths; pass --export-dir or set paths manually")
    finally:
        if restore_patch is not None:
            restore_patch()


if __name__ == "__main__":
    main()
