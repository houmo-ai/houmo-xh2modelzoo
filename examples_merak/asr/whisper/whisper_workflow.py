import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG_PATH = "configs_merak/workflows/xh2a/other_models/whisper/whisper.yaml"
DEFAULT_EXPORT_OUTPUT_DIR = "work_dirs/whisper_export"


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Whisper Merak workflow (encoder + prefill + decoder).",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Whisper HF model directory (e.g. openai/whisper-large-v3-turbo).",
    )
    parser.add_argument(
        "--config-path",
        default=DEFAULT_CONFIG_PATH,
        help=f"Workflow YAML path. Default: {DEFAULT_CONFIG_PATH}",
    )
    parser.add_argument(
        "--export-output-dir",
        default=DEFAULT_EXPORT_OUTPUT_DIR,
        help=f"Export output directory. Default: {DEFAULT_EXPORT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device label passed to workflow export/golden. Default: cuda",
    )
    parser.add_argument(
        "--quant-type",
        default=None,
        help="Override encoder/prefill/decoder quant_type in the YAML.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the export output directory before running.",
    )
    parser.add_argument(
        "--dump-golden",
        action="store_true",
        help="Generate golden data after export.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Pass debug=True to the workflow.",
    )
    return parser.parse_args()


def _build_config_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.quant_type is not None:
        overrides["export.components.encoder.quant_type"] = args.quant_type
        overrides["export.components.prefill.quant_type"] = args.quant_type
        overrides["export.components.decoder.quant_type"] = args.quant_type
    return overrides


def main() -> None:
    args = parse_args()

    from xhmodel_merak.workflows import AutoWorkflow

    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        debug=args.debug,
    )

    config_overrides = _build_config_overrides(args)
    quant_result = workflow.quant(
        output_dir="work_dirs/whisper_quant",
        device=args.device,
        config_overrides=config_overrides,
    )
    print(f"quant_result: {quant_result}")

    _remove_output_dir_if_needed(args.export_output_dir, args.overwrite)
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=args.export_output_dir,
        device=args.device,
        config_overrides=config_overrides,
    )
    print(f"export_result: {export_result}")
    if args.dump_golden:
        golden_dir = workflow.dump_golden(
            export_result=export_result,
            device=args.device,
        )
        print(f"golden_dir: {golden_dir}")


if __name__ == "__main__":
    main()
