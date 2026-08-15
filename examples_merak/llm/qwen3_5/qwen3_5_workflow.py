"""Standard Qwen3.5/Qwen3.6 Merak workflow example.

Run this file from the repository root.  Model shape, quantization, visual
size, MTP/DFlash, FlashAttention, and GDR defaults stay in YAML.  The CLI only
adds explicit run-local overrides.
"""

import argparse
import shutil
from pathlib import Path


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def _normalize_model_name(model_name: str) -> str:
    return model_name.strip().lower().replace(".", "_").replace("-", "_")


def _apply_model_name_override(config_overrides: dict[str, object], model_name: str | None) -> None:
    if model_name:
        config_overrides["export.model.model_name"] = _normalize_model_name(model_name)


def _add_bool_override_args(
    parser: argparse.ArgumentParser,
    *,
    dest: str,
    enable_flag: str,
    disable_flag: str,
    help_name: str,
) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        enable_flag,
        dest=dest,
        action="store_true",
        default=None,
        help=f"Enable {help_name} for this export run.",
    )
    group.add_argument(
        disable_flag,
        dest=dest,
        action="store_false",
        default=None,
        help=f"Disable {help_name} for this export run.",
    )


def _config_path_exists(data: dict, path: str) -> bool:
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _add_context_length_overrides(
    config_overrides: dict[str, object],
    context_max_length: int | None,
) -> None:
    if context_max_length is None:
        return
    if context_max_length <= 0:
        raise ValueError(f"--context-max-length must be positive, got {context_max_length}")

    # Draft cache capacities are derived from this target-model value by the
    # model config; the CLI intentionally exposes only one context length.
    config_overrides["export.model.context_max_length"] = context_max_length


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
        "--model-name",
        default=None,
        help=(
            "Override export.model.model_name from the workflow YAML. "
            "Use this for fine-tuned checkpoints that share the YAML architecture; "
            "'.' and '-' are normalized to '_'. "
            "Example: Qwen3.6-27B-mode1-llm-only."
        ),
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
        "--golden-device-map",
        nargs="+",
        default=None,
        help=(
            "Explicit HMONNX device map used only by --dump-golden. More "
            "than one CUDA entry enables HMONNXInferenceV2 auto-offload; "
            "ordinary golden generation remains single-device."
        ),
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick HMONNX test after export.",
    )
    parser.add_argument(
        "--resource-tight-mode",
        action="store_true",
        help=(
            "Release the speculative prefill runtime before allocating decode "
            "caches. Use this for single-GPU MTP/DFlash validation with long contexts."
        ),
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
    parser.add_argument(
        "--context-max-length",
        "--context-length",
        type=int,
        default=None,
        help=("LLM max context length for export; MTP/DFlash cache capacities derive from the same value."),
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=None,
        help=(
            "Override export.model.num_draft_tokens for this run. DFlash "
            "configs default to 9; use 15 only for an explicit experiment."
        ),
    )
    _add_bool_override_args(
        parser,
        dest="flash_attention",
        enable_flag="--enable-flash-attention",
        disable_flag="--disable-flash-attention",
        help_name="FlashAttention",
    )
    _add_bool_override_args(
        parser,
        dest="fuse_gdr_ops",
        enable_flag="--enable-fuse-gdr-ops",
        disable_flag="--disable-fuse-gdr-ops",
        help_name="fuse_gdr_ops",
    )
    _add_bool_override_args(
        parser,
        dest="fuse_gdr_block_recurrent_ops",
        enable_flag="--enable-fuse-gdr-block-recurrent-ops",
        disable_flag="--disable-fuse-gdr-block-recurrent-ops",
        help_name="fuse_gdr_block_recurrent_ops",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 初始化工作流
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
    )
    if args.resource_tight_mode and not _config_path_exists(
        workflow.workflow_config.data,
        "export.model.spec_decode_mode",
    ):
        raise ValueError("--resource-tight-mode requires an MTP or DFlash workflow config")

    # quant
    if args.export_from_quanted_model:
        config_overrides = {"quant": None}
        _apply_model_name_override(config_overrides, args.model_name)
        quant_result = workflow.quant(
            output_dir=args.quant_output_dir,
            device=args.device,
            # 从已量化的 HF 模型导出，需要跳过量化阶段
            config_overrides=config_overrides,
        )
    else:
        config_overrides = {}
        if args.bits:
            config_overrides["quant.bits"] = args.bits
        _apply_model_name_override(config_overrides, args.model_name)
        quant_result = workflow.quant(
            output_dir=args.quant_output_dir,
            device=args.device,
            config_overrides=config_overrides,
        )
    print(f"quant_result: {quant_result}")

    # export
    _remove_output_dir_if_needed(args.export_output_dir, args.overwrite)
    config_overrides = {}
    _apply_model_name_override(config_overrides, args.model_name)
    if args.max_size_h:
        config_overrides["export.model.visual_config.max_size_h"] = args.max_size_h
    if args.max_size_w:
        config_overrides["export.model.visual_config.max_size_w"] = args.max_size_w
    _add_context_length_overrides(
        config_overrides,
        args.context_max_length,
    )
    if args.num_draft_tokens is not None:
        if args.num_draft_tokens <= 0:
            raise ValueError(f"--num-draft-tokens must be positive, got {args.num_draft_tokens}")
        if not _config_path_exists(
            workflow.workflow_config.data,
            "export.model.num_draft_tokens",
        ):
            raise ValueError("--num-draft-tokens requires an MTP or DFlash workflow config")
        config_overrides["export.model.num_draft_tokens"] = args.num_draft_tokens
    if args.flash_attention is not None:
        config_overrides["export.model.flash_attention.enable"] = args.flash_attention
    if args.fuse_gdr_ops is not None:
        config_overrides["export.model.fuse_gdr_ops"] = args.fuse_gdr_ops
    if args.fuse_gdr_block_recurrent_ops is not None:
        config_overrides["export.model.fuse_gdr_block_recurrent_ops"] = args.fuse_gdr_block_recurrent_ops
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
                "image": "data/images/qwen2_vl_demo.jpeg",
            },
            device_map=args.golden_device_map,
            resource_tight_mode=args.resource_tight_mode,
        )

    # test hmonnx generation
    if args.quick_test:
        from xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation import (  # noqa: E402
            print_quick_test_result,
            quick_test_hmonnx,
        )

        quick_test_kwargs = {}
        if args.resource_tight_mode:
            quick_test_kwargs.update(
                resource_tight_mode=True,
                disable_auto_offload=True,
            )
        quick_result = quick_test_hmonnx(
            export_result,
            prompt="用中文简单说明这个模型可以做什么。",
            device=args.device,
            max_new_tokens=64,
            do_sample=False,
            **quick_test_kwargs,
        )
        print_quick_test_result(quick_result)


if __name__ == "__main__":
    main()
