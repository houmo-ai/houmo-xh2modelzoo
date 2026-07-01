"""Standard Qwen3.5/Qwen3.6 Merak workflow example.

Run this file from the repository root.  Model shape, quantization, visual
size, MTP/DFlash, and GDR options stay in YAML or ``CONFIG_OVERRIDES``; the
workflow API only needs paths plus ``QuantResult``.
"""

import argparse
import shutil
from pathlib import Path

def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Qwen3.5/Qwen3.6 Merak quant/export workflow.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="HF model directory.",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Workflow YAML path, choose one in ./configs",
    )
    parser.add_argument(
        "--quant-output-dir",
        default="work_dirs/qwen3.5_quant",
        help="Quantization output directory. Default: work_dirs/qwen3.5_quant",
    )
    parser.add_argument(
        "--export-output-dir",
        default="work_dirs/qwen3.5_export",
        help="Export output directory. Default: work_dirs/qwen3.5_export",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for quant/export/golden/quick test. Default: cuda",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing export output directories before running.",
    )
    parser.add_argument(
        "--dump-golden",
        action="store_true",
        help="Dump golden data after export.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick HMONNX test after export.",
    )
    parser.add_argument(
        "--export-from-quanted-model",
        action="store_true",
        help="if --model-dir is a quanted model, set this param to True",
    )
    parser.add_argument(
        "--bits",
        type=int,
        default=None,
        help="quantization bits, set this param to override config.yaml",
    )
    parser.add_argument(
        "--max-size-h",
        type=int,
        default=None,
        help="ViT input height, set this param to override config.yaml",
    )
    parser.add_argument(
        "--max-size-w",
        type=int,
        default=None,
        help="ViT input width, set this param to override config.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 初始化工作流
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        hf_model_dir=args.model_dir,
        config_path=args.config_path,
    )

    # quant
    if args.export_from_quanted_model:
        quant_result = workflow.quant(
            output_dir=args.quant_output_dir,
            device=args.device,
            # 从已量化的 HF 模型导出，需要跳过量化阶段
            config_overrides={"quant": None},
        )
    else:
        config_overrides = {}
        if args.bits:
            config_overrides["quant.bits"] = args.bits
        quant_result = workflow.quant(
            output_dir=args.quant_output_dir,
            device=args.device,
            config_overrides=config_overrides,
        )
    print(f"quant_result: {quant_result}")

    # export
    _remove_output_dir_if_needed(args.export_output_dir, args.overwrite)
    config_overrides = {}
    if args.max_size_h:
        config_overrides["export.model.visual_config.max_size_h"] = args.max_size_h
    if args.max_size_w:
        config_overrides["export.model.visual_config.max_size_w"] = args.max_size_w
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=args.export_output_dir,
        device=args.device,
        config_overrides=config_overrides,
    )
    print(f"export_result: {export_result}")

    # dump golden
    if args.dump_golden:
        workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages={
                "text": "描述这张图片",
                "image": "data/images/qwen2_vl_demo.jpeg"
            },
        )

    # test hmonnx generation
    if args.quick_test:
        from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import (  # noqa: E402
            print_quick_test_result,
            quick_test_hmonnx,
        )

        quick_result = quick_test_hmonnx(
            export_result,
            prompt="用中文简单介绍 Qwen3.5。",
            device=args.device,
            max_new_tokens=64,
            do_sample=False,
        )
        print_quick_test_result(quick_result)


if __name__ == "__main__":
    main()
