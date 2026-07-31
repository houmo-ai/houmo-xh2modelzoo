import argparse
import shutil
from pathlib import Path


DEFAULT_MODEL_DIR = "/data01/datasets/Laguna-S-2.1"
DEFAULT_CONFIG_PATH = "configs_merak/workflows/xh2a/llm_models/laguna/s_2_1/laguna_s_2_1_xh2a_w4a8.yaml"
AUTOROUND_CONFIG_PATH = (
    "configs_merak/workflows/xh2a/llm_models/laguna/s_2_1/"
    "laguna_s_2_1_autoround_expert_w4_rest_w8_g64_xh2a_w4a8.yaml"
)


def _remove_output_dir_if_needed(output_dir: str, overwrite: bool) -> None:
    path = Path(output_dir)
    if overwrite and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Laguna-S-2.1 from floating-point HF weights to W4A8 HMONNX.")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="Downloaded Laguna HF model directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH, help="Merak workflow YAML path.")
    parser.add_argument("--export-output-dir", default="work_dirs/laguna_s_2_1_export")
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--golden-device-map",
        default=None,
        help="Comma-separated devices for HMONNX golden auto-offload, for example cuda:0,cuda:1.",
    )
    parser.add_argument("--context-max-length", "--context-length", type=int, default=None)
    parser.add_argument("--prefill-chunk-length", type=int, default=None)
    parser.add_argument("--quant-type", default=None, help="Override export-time HMONNX quantization type.")
    layer_group = parser.add_mutually_exclusive_group()
    layer_group.add_argument("--only-first-block", action="store_true", help="Export only layer 0 for debugging.")
    layer_group.add_argument("--max-layers", type=int, default=None, help="Export the first N decoder layers.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--prompt", default="Briefly explain what makes Laguna-S-2.1 useful for coding tasks.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _export_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.context_max_length is not None:
        overrides["export.model.context_max_length"] = args.context_max_length
    if args.prefill_chunk_length is not None:
        overrides["export.model.prefill_chunk_length"] = args.prefill_chunk_length
    if args.quant_type is not None:
        overrides["export.model.quant_scheme.quant_type"] = args.quant_type
        overrides["export.model.quant_scheme.nodes.lm_head.quant_type"] = args.quant_type
    if args.only_first_block:
        overrides["export.model.only_first_block"] = True
    if args.max_layers is not None:
        if args.max_layers <= 0:
            raise ValueError("--max-layers must be greater than zero")
        overrides["export.model.max_layers"] = args.max_layers
    return overrides


def _configure_cuda_device(device: str) -> None:
    import torch

    parsed_device = torch.device(device)
    if parsed_device.type == "cuda" and parsed_device.index is not None:
        torch.cuda.set_device(parsed_device.index)


def main() -> None:
    args = parse_args()
    _configure_cuda_device(args.device)

    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        debug=args.debug,
    )

    quant_result = workflow.quant(
        output_dir=args.export_output_dir,
        device=args.device,
    )
    print(f"quant_result: {quant_result}")

    _remove_output_dir_if_needed(args.export_output_dir, args.overwrite)
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=args.export_output_dir,
        device=args.device,
        config_overrides=_export_overrides(args),
    )
    print(f"export_result: {export_result}")

    if args.dump_golden:
        golden_meta = workflow.dump_golden(
            export_result=export_result,
            device=args.golden_device_map or args.device,
            input_messages={"text": args.prompt},
        )
        print(f"golden_meta: {golden_meta}")


if __name__ == "__main__":
    main()
