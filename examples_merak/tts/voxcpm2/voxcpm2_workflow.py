import argparse


DEFAULT_CONFIG = "configs_merak/workflows/xh2a/other_models/voxcpm2/default/voxcpm2_xh2a.yaml"
DEFAULT_OUTPUT = "work_dirs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VoxCPM2 Merak workflow.")
    parser.add_argument("--model-dir", required=True, help="VoxCPM2 local model directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG, help="Workflow YAML path.")
    parser.add_argument(
        "--export-output-dir",
        default=DEFAULT_OUTPUT,
        help="Parent directory for the automatically named HM-standard release directory.",
    )
    parser.add_argument("--quant-output-dir", default="work_dirs/voxcpm2_quant")
    parser.add_argument(
        "--device",
        default="cuda",
        help="Execution device used by every exporter: cpu, cuda, or cuda:N.",
    )
    parser.add_argument("--components", default=None, help="Comma-separated component list override.")
    parser.add_argument("--quant-type", default=None, help="Override component quant types where applicable.")
    parser.add_argument(
        "--dump-golden",
        action="store_true",
        help="Generate step_0 golden from the released HMONNX after export.",
    )
    parser.add_argument("--release-date", default=None, help="HM release date in YYYYMMDD; defaults to today.")
    parser.add_argument("--release-prefix", default=None, help="Explicit lowercase HM release prefix override.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the selected export/release output.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _build_config_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.components:
        overrides["export.components"] = [
            item.strip()
            for item in args.components.split(",")
            if item.strip()
        ]
    if args.quant_type:
        for name in (
            "lm",
            "locenc",
            "locdit",
            "audiovae_decoder_stream",
            "audiovae_decoder_full",
            "audiovae_decoder_stateful",
        ):
            overrides[f"export.quant_types.{name}"] = args.quant_type
    return overrides


def main() -> None:
    args = parse_args()
    try:
        from xhmodel_merak.workflows import AutoWorkflow

        workflow = AutoWorkflow.from_config(
            model_dir=args.model_dir,
            config_path=args.config_path,
            debug=args.debug,
        )
    except ModuleNotFoundError:
        from xhmodel_merak.xh_other_model.workflows import AutoOtherModelWorkflow

        workflow = AutoOtherModelWorkflow.from_config(
            model_dir=args.model_dir,
            config_path=args.config_path,
            debug=args.debug,
        )
    overrides = _build_config_overrides(args)
    quant_result = workflow.quant(
        output_dir=args.quant_output_dir,
        device=args.device,
        config_overrides=overrides,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=args.export_output_dir,
        device=args.device,
        config_overrides=overrides,
        release_date=args.release_date,
        release_prefix=args.release_prefix,
        overwrite=args.overwrite,
    )
    print(f"release_dir: {export_result.work_dir}")
    if args.dump_golden:
        golden_meta = workflow.dump_golden(export_result, args.device)
        print(f"golden_meta: {golden_meta}")


if __name__ == "__main__":
    main()
