#!/usr/bin/env python3
"""Run the YAML-based DeepSeek-V4 Flash HMONNX export workflow."""

from __future__ import annotations

import argparse

from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Existing GPTQModel/AutoRound checkpoint directory")
    parser.add_argument("--config-path", required=True, help="DeepSeek-V4 workflow YAML")
    parser.add_argument("--export-output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--context-length", type=int)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--model-name")
    parser.add_argument("--quant-type", choices=("w8a8h1_sefp", "w8a16h1_sefp"))
    parser.add_argument("--low-memory", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--golden-device-map", nargs="+")
    parser.add_argument(
        "--golden-prompt",
        default="17 multiplied by 3 equals what? Answer with only the result.",
    )
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    return parser


def main(args: argparse.Namespace) -> None:
    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        seed=args.seed,
        debug=args.debug,
    )
    overrides = {}
    if args.context_length is not None:
        overrides["export.model.context_max_length"] = args.context_length
        overrides["export.model.max_pe_length"] = args.context_length
    if args.max_layers is not None:
        overrides["export.model.max_layers"] = args.max_layers
    if args.model_name is not None:
        overrides["export.model.model_name"] = args.model_name
    if args.quant_type is not None:
        overrides["export.model.quant_scheme.quant_type"] = args.quant_type
    if args.low_memory is not None:
        overrides["runtime.low_memory"] = args.low_memory

    quant_result = workflow.quant(
        output_dir=f"{args.export_output_dir}_quant",
        device=args.device,
        config_overrides=overrides,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=args.export_output_dir,
        device=args.device,
        config_overrides=overrides,
    )
    print(f"export_result: {export_result}")

    if args.dump_golden:
        golden_meta = workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages=args.golden_prompt,
            device_map=args.golden_device_map,
        )
        print(f"golden_meta_info: {golden_meta}")


if __name__ == "__main__":
    main(build_parser().parse_args())
