"""Export MiniCPM5-15B-A2.5B to Merak HMONNX."""

import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG_PATH = "configs_merak/workflows/xh2a/llm_models/minicpm5/15b_a2_5b/minicpm5_15b_a2_5b_xh2a_w8a8.yaml"


def _remove_output_dir_if_needed(output_dir: str, overwrite: bool) -> None:
    path = Path(output_dir)
    if overwrite and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export MiniCPM5-15B-A2.5B to HMONNX.")
    parser.add_argument("--model-dir", required=True, help="MiniCPM5 HF model directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH, help="Merak workflow YAML path.")
    parser.add_argument("--quant-output-dir", default="work_dirs/minicpm5_15b_a2_5b_quant")
    parser.add_argument("--export-output-dir", default="work_dirs/minicpm5_15b_a2_5b_export")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-max-length", "--context-length", type=int, default=None)
    parser.add_argument("--prefill-chunk-length", type=int, default=None)
    parser.add_argument(
        "--quant-type",
        default=None,
        help="Override export.model.quant_scheme.quant_type; lm_head keeps its configured scheme.",
    )
    parser.add_argument("--only-first-block", action="store_true")
    parser.add_argument("--max-layers", type=int, default=None)
    parser.add_argument("--num-logits-to-keep", type=int, choices=(0, 1), default=None)
    parser.add_argument(
        "--export-from-quanted-model",
        action="store_true",
        help="Treat --model-dir as an existing GPTQ checkpoint and skip quantization.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--prompt", default="请用中文简单介绍 MiniCPM5。")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.context_max_length is not None:
        if args.context_max_length <= 0:
            raise ValueError("--context-max-length must be positive")
        overrides["export.model.context_max_length"] = args.context_max_length
    if args.prefill_chunk_length is not None:
        if args.prefill_chunk_length <= 0:
            raise ValueError("--prefill-chunk-length must be positive")
        overrides["export.model.prefill_chunk_length"] = args.prefill_chunk_length
    if args.quant_type is not None:
        overrides["export.model.quant_scheme.quant_type"] = args.quant_type
    if args.only_first_block:
        overrides["export.model.only_first_block"] = True
    if args.max_layers is not None:
        if args.max_layers <= 0:
            raise ValueError("--max-layers must be positive")
        overrides["export.model.max_layers"] = args.max_layers
    if args.num_logits_to_keep is not None:
        overrides["export.model.num_logits_to_keep"] = args.num_logits_to_keep
    return overrides


def main() -> None:
    args = parse_args()
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        debug=args.debug,
    )
    quant_overrides = {"quant": None} if args.export_from_quanted_model else None
    quant_result = workflow.quant(
        output_dir=args.quant_output_dir,
        device=args.device,
        config_overrides=quant_overrides,
    )
    print(f"quant_result: {quant_result}")

    _remove_output_dir_if_needed(args.export_output_dir, args.overwrite)
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=args.export_output_dir,
        device=args.device,
        config_overrides=_overrides(args),
    )
    print(f"export_result: {export_result}")

    if args.dump_golden:
        golden_meta = workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages={"text": args.prompt},
        )
        print(f"golden_meta: {golden_meta}")


if __name__ == "__main__":
    main()
