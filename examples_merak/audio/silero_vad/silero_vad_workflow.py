from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG = "configs_merak/workflows/xh2a/other_models/silero_vad/silero_vad_xh2a_w16.yaml"
DEFAULT_OUTPUT = "work_dirs/silero_vad_merak/export_xh2a_w16a16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export official Silero VAD 8/16 kHz graphs with Merak")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--golden-audio")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _remove_output_dir_if_needed(path: Path, overwrite: bool) -> None:
    if overwrite and path.exists():
        shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    output_dir = Path(args.output_dir).expanduser().resolve()
    _remove_output_dir_if_needed(output_dir, args.overwrite)
    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        debug=args.debug,
    )
    quant_result = workflow.quant(
        output_dir=str(output_dir / "quant"),
        device=args.device,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=str(output_dir),
        device=args.device,
    )
    print(output_dir / "export_meta_info.json")
    if args.dump_golden:
        print(
            workflow.dump_golden(
                export_result=export_result,
                device=args.device,
                input_messages=({"audio": args.golden_audio} if args.golden_audio else None),
            )
        )


if __name__ == "__main__":
    main()
