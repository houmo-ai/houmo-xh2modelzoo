import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG_PATH = "configs_merak/workflows/xh2a/other_models/hy_mt2/hy_mt2.yaml"
DEFAULT_QUANT_OUTPUT_DIR = "work_dirs/hy_mt2_quant"
DEFAULT_EXPORT_OUTPUT_DIR = "work_dirs/hy_mt2_export"


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Hy-MT2 Merak workflow.")
    parser.add_argument("--model-dir", required=True, help="Hy-MT2 HF model directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--quant-output-dir", default=DEFAULT_QUANT_OUTPUT_DIR)
    parser.add_argument("--export-output-dir", default=DEFAULT_EXPORT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--input-sequence-length", type=int, default=None)
    parser.add_argument("--quant-type", default=None)
    parser.add_argument("--quant-weight", default=None)
    parser.add_argument("--mix-search", default=None)
    parser.add_argument("--num-logits-to-keep", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _build_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    values = {
        "export.hy_mt2.context_length": args.context_length,
        "export.hy_mt2.input_sequence_length": args.input_sequence_length,
        "export.hy_mt2.quant_type": args.quant_type,
        "export.hy_mt2.quant_weight": args.quant_weight,
        "export.hy_mt2.mix_search": args.mix_search,
        "export.hy_mt2.num_logits_to_keep": args.num_logits_to_keep,
    }
    overrides.update({key: value for key, value in values.items() if value is not None})
    return overrides


def main() -> None:
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    _remove_output_dir_if_needed(args.export_output_dir, args.overwrite)
    workflow = AutoWorkflow.from_config(model_dir=args.model_dir, config_path=args.config_path, debug=args.debug)
    overrides = _build_overrides(args)
    quant_result = workflow.quant(args.quant_output_dir, args.device, overrides)
    export_result = workflow.export(quant_result, args.export_output_dir, args.device, overrides)
    print(f"export_result: {export_result}")
    if args.dump_golden:
        golden_dir = workflow.dump_golden(export_result, args.device)
        print(f"golden_dir: {golden_dir}")


if __name__ == "__main__":
    main()
