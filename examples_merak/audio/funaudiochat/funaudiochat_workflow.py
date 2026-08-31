from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


DEFAULT_CONFIG = "configs_merak/workflows/xh2a/other_models/funaudiochat/funaudiochat.yaml"
DEFAULT_OUTPUT = "work_dirs/funaudiochat_merak/export_xh2a_w8a8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FunAudioChat HMONNX graphs with the Merak workflow")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--audio", required=True, help="Reference audio used to build the static audio-encoder graph")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--system-prompt")
    parser.add_argument("--audio-duration-seconds", type=float)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    overrides = {"export.runtime.audio": str(Path(args.audio).expanduser().resolve())}
    if args.system_prompt is not None:
        overrides["export.runtime.system_prompt"] = args.system_prompt
    if args.audio_duration_seconds is not None:
        overrides["export.audio_duration_seconds"] = args.audio_duration_seconds

    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        debug=args.debug,
    )
    quant_result = workflow.quant(
        output_dir=str(output_dir / "quant"),
        device=args.device,
        config_overrides=overrides,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=str(output_dir),
        device=args.device,
        config_overrides=overrides,
    )
    print(Path(export_result.work_dir) / "export_meta_info.json")


if __name__ == "__main__":
    main()
