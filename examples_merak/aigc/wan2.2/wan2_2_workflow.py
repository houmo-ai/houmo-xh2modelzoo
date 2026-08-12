# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

import argparse
import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_DIR = (SCRIPT_DIR / "../../../configs_merak/workflows/xh2a/other_models/wan2_2").resolve()
WORKFLOW_CONFIGS = {
    "i2v-A14B": CONFIG_DIR / "i2v_A14B/wan2_2_i2v_A14B.yaml",
    "t2v-A14B": CONFIG_DIR / "t2v_A14B/wan2_2_t2v_A14B.yaml",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Wan2.2 Merak export workflow.")
    parser.add_argument("--model-dir", required=True, help="Local Wan2.2 checkpoint directory.")
    parser.add_argument(
        "--task",
        choices=WORKFLOW_CONFIGS,
        default="i2v-A14B",
        help="Select the matching I2V/T2V workflow YAML when --config-path is not specified.",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Explicit workflow YAML; overrides the YAML selected by --task.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Export directory; defaults to work_dirs/wan2_<task>.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--components", nargs="+", default=None)
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
    parser.add_argument(
        "--release-dit-fp16-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Override whether each DiT module's FP16 weights are released "
            "immediately after PTQ; when omitted, use the workflow YAML value."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    values = {
        "export.wan2_2.components": args.components,
        "export.wan2_2.torch_dtype": args.torch_dtype,
        "export.wan2_2.prompt": args.prompt,
        "export.wan2_2.size": args.size,
        "export.wan2_2.frame_num": args.frame_num,
        "export.wan2_2.sample_steps": args.sample_steps,
        "export.wan2_2.use_resolved_float_loader": True if args.use_resolved_float_loader else None,
        "export.wan2_2.release_dit_fp16_weights": args.release_dit_fp16_weights,
    }
    overrides.update({key: value for key, value in values.items() if value is not None})
    return overrides


def resolve_output_dir(args: argparse.Namespace) -> Path:
    output_dir = args.output_dir or Path("work_dirs") / f"wan2_{args.task}"
    return Path(output_dir).expanduser().resolve()


def main() -> None:
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    config_path = Path(args.config_path).expanduser().resolve() if args.config_path else WORKFLOW_CONFIGS[args.task]
    if not config_path.is_file():
        raise FileNotFoundError(f"Wan2.2 workflow config not found: {config_path}")

    output_dir = resolve_output_dir(args)
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=str(config_path),
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
