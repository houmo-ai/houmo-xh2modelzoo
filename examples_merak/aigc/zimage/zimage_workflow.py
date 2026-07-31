import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG_PATH = "configs_merak/workflows/xh2a/other_models/zimage/zimage.yaml"
DEFAULT_OUTPUT_DIR = "work_dirs/zimage_export"


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the ZImage Merak workflow.")
    parser.add_argument("--model-dir", required=True, help="ZImage HF / diffusers model directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--components", nargs="+", default=None)
    parser.add_argument("--quant-type", default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--input-sequence-length", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _build_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.components is not None:
        overrides["export.zimage.components"] = args.components
    if args.quant_type is not None:
        overrides["export.zimage.quant_type"] = args.quant_type
        overrides["export.zimage.text_encoder.quant_type"] = args.quant_type
        overrides["export.zimage.vae.quant_type"] = args.quant_type
        overrides["export.zimage.dit.quant_type"] = args.quant_type
    if args.context_length is not None:
        overrides["export.zimage.context_length"] = args.context_length
    if args.input_sequence_length is not None:
        overrides["export.zimage.input_sequence_length"] = args.input_sequence_length
    return overrides


def main() -> None:
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    _remove_output_dir_if_needed(args.output_dir, args.overwrite)
    workflow = AutoWorkflow.from_config(model_dir=args.model_dir, config_path=args.config_path, debug=args.debug)
    overrides = _build_overrides(args)
    quant_result = workflow.quant(output_dir=args.output_dir, device=args.device, config_overrides=overrides)
    export_result = workflow.export(quant_result=quant_result, output_dir=args.output_dir, device=args.device, config_overrides=overrides)
    print(f"export_result: {export_result}")
    if args.dump_golden:
        print(workflow.dump_golden(export_result=export_result, device=args.device))


if __name__ == "__main__":
    main()
