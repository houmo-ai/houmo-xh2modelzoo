"""Run the MiniCPM-V-4.5 Merak export workflow."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG = "configs_merak/workflows/xh2a/llm_models/minicpm_v_4_5/8b/minicpm_v_4_5_xh2a_w8a8.yaml"
DEFAULT_W4A8_CONFIG = "configs_merak/workflows/xh2a/llm_models/minicpm_v_4_5/8b/minicpm_v_4_5_xh2a_w4a8_gptq.yaml"


def _default_export_dir() -> str:
    """Release-convention default export dir:
    ``hmquant_<xh>_<modelcope>_<wmix_amix>_<prefill>_<context>_<capacity>_<date>``.
    """
    import time

    date = time.strftime("%Y%m%d")
    return f"work_dirs/hmquant_xh2_minicpm_v_4_5_w8a8_256_8k_cap1600_{date}"


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MiniCPM-V-4.5 Merak workflow.")
    parser.add_argument("--model-dir", required=True, help="MiniCPM-V-4.5 HF model directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG)
    parser.add_argument("--quant-output-dir", default="work_dirs/minicpm_v_4_5_merak_quant")
    parser.add_argument("--export-output-dir", default=_default_export_dir())
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-capacity", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--prefill-length", type=int, default=None)
    parser.add_argument("--llm-quant-type", default=None)
    parser.add_argument("--vision-quant-type", default=None)
    parser.add_argument(
        "--calib-jsonl", default=None, help="GPTQ calibration jsonl (overrides quant.calibration_jsonl)"
    )
    parser.add_argument("--gptq-bits", type=int, default=None, help="GPTQ bits (overrides quant.bits)")
    parser.add_argument("--group-size", type=int, default=None, help="GPTQ group size (overrides quant.group_size)")
    parser.add_argument(
        "--nsamples", type=int, default=None, help="GPTQ calibration sample count (overrides quant.nsamples)"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--golden-image", default=None)
    parser.add_argument("--golden-prompt", default="请描述这张图片。")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {}
    if args.token_capacity is not None:
        # patch_capacity 是静态图容量，不能通过 max_size_w/h 间接推导，
        # 否则非平方容量会被 isqrt 静默截断。
        capacity = int(args.token_capacity)
        if capacity <= 0:
            raise ValueError(f"--token-capacity must be positive, got {capacity}")
        result["export.model.visual_config.patch_capacity"] = capacity
    if args.context_length is not None:
        result["export.model.context_max_length"] = args.context_length
    if args.prefill_length is not None:
        result["export.model.prefill_chunk_length"] = args.prefill_length
    if args.llm_quant_type is not None:
        result["export.model.quant_scheme.quant_type"] = args.llm_quant_type
    if args.vision_quant_type is not None:
        result["export.model.visual_config.quant_scheme.quant_type"] = args.vision_quant_type
    if args.calib_jsonl is not None:
        result["quant.calibration_jsonl"] = args.calib_jsonl
    if args.gptq_bits is not None:
        result["quant.bits"] = args.gptq_bits
    if args.group_size is not None:
        result["quant.group_size"] = args.group_size
    if args.nsamples is not None:
        result["quant.nsamples"] = args.nsamples
    return result


def main() -> None:
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        debug=args.debug,
    )
    overrides = _overrides(args)
    quant_result = workflow.quant(
        output_dir=args.quant_output_dir,
        device=args.device,
        config_overrides=overrides,
    )
    export_dir = Path(args.export_output_dir)
    _remove_output_dir_if_needed(str(export_dir), args.overwrite)
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=str(export_dir),
        device=args.device,
        config_overrides=overrides,
    )
    print(f"export_meta: {Path(export_result.work_dir) / 'golden_meta_info.json'}")
    if args.dump_golden:
        golden_input = None
        if args.golden_image is not None:
            golden_input = {"image": args.golden_image, "text": args.golden_prompt}
        golden_dir = workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages=golden_input,
        )
        print(f"golden_dir: {golden_dir}")


if __name__ == "__main__":
    main()
