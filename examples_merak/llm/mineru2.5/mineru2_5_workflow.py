"""
MinerU-2.5 config.yaml中的配置已经过人工调优，不建议override
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
        description="Run the MinerU-2.5 Merak quant/export workflow.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="HF model directory.",
    )
    parser.add_argument(
        "--config-path",
        required=True,
        help="Workflow YAML path, choose one in configs_merak/workflows/xh2a/llm_models/mineru2_5",
    )
    parser.add_argument(
        "--export-from-quanted-model",
        action="store_true",
        help="if --model-dir is a quanted model, set this param to True",
    )
    parser.add_argument(
        "--quant-output-dir",
        default="work_dirs/mineru-2.5_quant",
        help="Quantization output directory. Default: work_dirs/mineru-2.5_quant",
    )
    parser.add_argument(
        "--export-output-dir",
        default="work_dirs/mineru-2.5_export",
        help="Export output directory. Default: work_dirs/mineru-2.5_export",
    )
    parser.add_argument(
        "--dump-golden",
        action="store_true",
        help="Dump golden data after export.",
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
        quant_result = workflow.quant(
            output_dir=args.quant_output_dir,
            device=args.device,
        )
    print(f"quant_result: {quant_result}")

    # export
    _remove_output_dir_if_needed(args.export_output_dir, args.overwrite)
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=args.export_output_dir,
        device=args.device,
    )
    print(f"export_result: {export_result}")

    if args.dump_golden:
        """
        如无必要，MinerU-2.5不建议dump-golden。
        默认ViT尺寸较大，会导致hmonnx输入长度超过默认prefill_chunk_length报错。
        修改prefill_chunk_length可解决。
        """
        workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages={
                "image": "./data/images/qwen2_vl_demo.jpeg",
                "text": "描述这张图片",
            },
        )


if __name__ == "__main__":
    main()
