import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG = "configs_merak/workflows/xh2a/other_models/voxcpm2/default/voxcpm2_xh2a.yaml"
DEFAULT_OUTPUT = "work_dirs/hmquant_xh2_voxcpm2_wmix_amix_256_1k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the VoxCPM2 Merak workflow.")
    parser.add_argument("--model-dir", required=True, help="VoxCPM2 local model directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG, help="Workflow YAML path.")
    parser.add_argument(
        "--export-output-dir",
        default=DEFAULT_OUTPUT,
        help="Final VoxCPM2 artifact directory.",
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
        "--cal-wav",
        default=None,
        help="Real calibration audio for LocEnc and AudioVAE Encoder.",
    )
    parser.add_argument(
        "--dump-golden",
        action="store_true",
        help="Generate step_0 golden from the released HMONNX after export.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the export output directory before running.",
    )
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
    if args.cal_wav:
        overrides["export.locenc.cal_wav"] = args.cal_wav
        overrides["export.audiovae_encoder.audio"] = args.cal_wav
    return overrides


def _remove_output_dir_if_needed(output_dir: str, overwrite: bool) -> None:
    path = Path(output_dir).expanduser()
    if overwrite and path.exists():
        shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        debug=args.debug,
    )
    overrides = _build_config_overrides(args)
    _remove_output_dir_if_needed(args.export_output_dir, args.overwrite)
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
    )
    print(f"release_dir: {export_result.work_dir}")
    if args.dump_golden:
        golden_meta = workflow.dump_golden(export_result, args.device)
        print(f"golden_meta: {golden_meta}")


if __name__ == "__main__":
    main()
