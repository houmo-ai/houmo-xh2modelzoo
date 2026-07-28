import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG = (
    "configs_merak/workflows/xh2a/llm_models/"
    "qwen3_vl_embedding/2b/"
    "qwen3_vl_embedding_2b_xh2a_w8a8.yaml"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the Qwen3-VL-Embedding Merak workflow."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--quant-output-dir",
        default="work_dirs/qwen3_vl_embedding_quant",
    )
    parser.add_argument(
        "--export-output-dir",
        default="work_dirs/qwen3_vl_embedding_export",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--prefill-length", type=int, default=None)
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--quant-type", default=None)
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument(
        "--prompt",
        default=None,
        help="Optional text used to generate golden data",
    )
    parser.add_argument(
        "--image",
        default=None,
        help="Optional image used to generate golden data",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _build_golden_input(
    prompt: str | None,
    image: str | None,
) -> dict[str, str] | None:
    golden_input = {}
    if prompt is not None:
        golden_input["text"] = prompt
    if image is not None:
        image_path = Path(image)
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Golden image not found: {image_path}"
            )
        golden_input["image"] = str(image_path)
    return golden_input or None


def _remove_output_dir_if_needed(
    output_dir: str,
    overwrite: bool,
) -> None:
    path = Path(output_dir).resolve()
    if overwrite and path.exists():
        protected_paths = {
            Path("/"),
            Path.cwd().resolve(),
            Path.home().resolve(),
        }
        if path in protected_paths:
            raise ValueError(
                f"Refusing to remove protected directory: {path}"
            )
        shutil.rmtree(path)


def main():
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    golden_input = _build_golden_input(
        args.prompt,
        args.image,
    )
    overrides = {}
    if args.context_length is not None:
        overrides[
            "export.model.context_max_length"
        ] = args.context_length
    if args.prefill_length is not None:
        overrides[
            "export.model.prefill_chunk_length"
        ] = args.prefill_length
    if args.image_size is not None:
        overrides[
            "export.model.visual_config.max_size_h"
        ] = args.image_size
        overrides[
            "export.model.visual_config.max_size_w"
        ] = args.image_size
    if args.quant_type is not None:
        overrides[
            "export.model.quant_scheme.quant_type"
        ] = args.quant_type
        overrides[
            "export.model.visual_config.quant_scheme.quant_type"
        ] = args.quant_type

    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
    )
    quant_result = workflow.quant(
        args.quant_output_dir,
        args.device,
        overrides,
    )
    _remove_output_dir_if_needed(
        args.export_output_dir,
        args.overwrite,
    )
    export_result = workflow.export(
        quant_result,
        args.export_output_dir,
        args.device,
        overrides,
    )
    print(f"export_result: {export_result}")
    if args.dump_golden:
        golden_dir = workflow.dump_golden(
            export_result,
            args.device,
            golden_input,
        )
        print(f"golden_dir: {golden_dir}")


if __name__ == "__main__":
    main()
