import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG = (
    "configs_merak/workflows/xh2a/llm_models/unlimited_ocr/base/"
    "unlimited_ocr_base_xh2a_w8a8.yaml"
)


def _remove_output_dir_if_needed(output_dir: str, overwrite: bool) -> None:
    path = Path(output_dir)
    if overwrite and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Unlimited-OCR base/no-crop Merak workflow.",
    )
    parser.add_argument("--model-dir", required=True, help="Unlimited-OCR HF model directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG, help="Workflow YAML path.")
    parser.add_argument(
        "--quant-output-dir",
        default="work_dirs/unlimited_ocr_quant",
        help="Workflow quant output directory. Quant is skipped by current configs.",
    )
    parser.add_argument(
        "--export-output-dir",
        default="work_dirs/unlimited_ocr_workflow_export",
        help="HMONNX export output directory.",
    )
    parser.add_argument("--device", default="cuda", help="Device for export and golden generation.")
    parser.add_argument("--dump-golden", action="store_true", help="Generate aligned golden data after export.")
    parser.add_argument("--image", default="data/images/unlimited_ocr_demo.jpg", help="Golden image path.")
    parser.add_argument("--prompt", default="<image>\\nFree OCR. ", help="Golden OCR prompt.")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing export directory.")
    parser.add_argument("--debug", action="store_true", help="Enable workflow debug logging.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        debug=args.debug,
    )
    quant_result = workflow.quant(
        output_dir=args.quant_output_dir,
        device=args.device,
    )
    print(f"quant_result: {quant_result}")

    _remove_output_dir_if_needed(args.export_output_dir, args.overwrite)
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=args.export_output_dir,
        device=args.device,
    )
    print(f"export_result: {export_result}")

    if args.dump_golden:
        meta_file = workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages={"image": args.image, "prompt": args.prompt},
        )
        print(f"golden_meta_file: {meta_file}")


if __name__ == "__main__":
    main()