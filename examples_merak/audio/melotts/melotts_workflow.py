from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG = "configs_merak/workflows/xh2a/other_models/melotts/melotts_xh2a_w16_l32_t64.yaml"
DEFAULT_OUTPUT = "work_dirs/melotts_merak/export_xh2a_w16_l32_t64"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export strict MeloTTS encoder/decoder graphs with Merak")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)
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
                input_messages=None,
            )
        )


if __name__ == "__main__":
    main()
