# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG_PATH = "configs_merak/workflows/xh2a/other_models/wan2_2/i2v_A14B/wan2_2_i2v_A14B.yaml"
DEFAULT_OUTPUT_DIR = "work_dirs/wan2_2_i2v-A14B_merak"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Wan2.2 Merak export workflow.")
    parser.add_argument("--model-dir", required=True, help="Local Wan2.2 checkpoint directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--task", default=None)
    parser.add_argument("--components", nargs="+", default=None)
    parser.add_argument("--quant-type", default=None)
    parser.add_argument("--torch-dtype", default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--size", nargs=2, type=int, default=None, metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--frame-num", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument(
        "--use-resolved-float-loader",
        action="store_true",
        help="Load resolved split/merged safetensors; required for <model-dir>/merged LoRA weights.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    values = {
        "export.wan2_2.task": args.task,
        "export.wan2_2.components": args.components,
        "export.quant_scheme.quant_type": args.quant_type,
        "export.wan2_2.torch_dtype": args.torch_dtype,
        "export.wan2_2.prompt": args.prompt,
        "export.wan2_2.size": args.size,
        "export.wan2_2.frame_num": args.frame_num,
        "export.wan2_2.sample_steps": args.sample_steps,
        "export.wan2_2.use_resolved_float_loader": True if args.use_resolved_float_loader else None,
    }
    overrides.update({key: value for key, value in values.items() if value is not None})
    return overrides


def main() -> None:
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        seed=args.seed,
        debug=args.debug,
    )
    overrides = build_overrides(args)
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
    print(f"Export metadata: {output_dir / 'export_meta_info.json'}")
    if args.dump_golden:
        golden_meta = workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages={"components": overrides.get("export.wan2_2.components")} if args.components else None,
        )
        print(f"Golden metadata: {golden_meta}")


if __name__ == "__main__":
    main()
