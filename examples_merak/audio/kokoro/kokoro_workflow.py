from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG = "configs_merak/workflows/xh2a/other_models/kokoro/kokoro_xh2a_bucketed_w16.yaml"
DEFAULT_OUTPUT = "work_dirs/kokoro_merak/bucketed_w16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Kokoro static graph workflow")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--onnx-only", action="store_true")
    parser.add_argument(
        "--token-bucket",
        dest="token_buckets",
        type=int,
        action="append",
        help="Select one paired route by T capacity; repeat for more routes",
    )
    parser.add_argument(
        "--audio-seconds-bucket",
        dest="audio_seconds_buckets",
        type=int,
        action="append",
        help="Select one paired route by audio capacity; repeat for more routes",
    )
    parser.add_argument("--lstm-variants", choices=("native", "decomposed"), nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow
    from xhmodel_merak.xh_other_model.models.kokoro.buckets import BUCKET_ROUTES

    output_dir = Path(args.output_dir).expanduser().resolve()
    _remove_output_dir_if_needed(output_dir, overwrite=args.overwrite)
    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        debug=args.debug,
    )
    overrides = {}
    if args.onnx_only:
        overrides["export.convert_hmonnx"] = False
    if args.token_buckets:
        overrides["export.token_buckets"] = args.token_buckets
    if args.audio_seconds_buckets:
        overrides["export.audio_seconds_buckets"] = args.audio_seconds_buckets
    if args.token_buckets and not args.audio_seconds_buckets:
        selected = set(args.token_buckets)
        overrides["export.audio_seconds_buckets"] = [
            route.audio_seconds for route in BUCKET_ROUTES if route.token_max_length in selected
        ]
    if args.audio_seconds_buckets and not args.token_buckets:
        selected = set(args.audio_seconds_buckets)
        overrides["export.token_buckets"] = [
            route.token_max_length for route in BUCKET_ROUTES if route.audio_seconds in selected
        ]
    if args.lstm_variants:
        overrides["export.lstm_variants"] = args.lstm_variants
    config_overrides = overrides or None
    quant_result = workflow.quant(
        output_dir=str(output_dir / "quant"),
        device=args.device,
        config_overrides=config_overrides,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=str(output_dir),
        device=args.device,
        config_overrides=config_overrides,
    )
    print(output_dir / "export_meta_info.json")
    if args.dump_golden:
        if args.onnx_only:
            raise ValueError("--dump-golden cannot be combined with --onnx-only")
        print(
            workflow.dump_golden(
                export_result=export_result,
                device=args.device,
                input_messages=None,
            )
        )


def _remove_output_dir_if_needed(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not overwrite:
        raise FileExistsError(f"output directory already exists: {output_dir}; pass --overwrite to replace it")
    cwd = Path.cwd().resolve()
    if output_dir in {Path("/").resolve(), cwd, cwd.parent}:
        raise ValueError(f"refusing to recursively remove unsafe output path: {output_dir}")
    shutil.rmtree(output_dir)


if __name__ == "__main__":
    main()
