"""Run the MiniCPM-V-4.6 Merak export workflow."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG = "configs_merak/workflows/xh2a/llm_models/minicpm_v_4_6/0_8b/minicpm_v_4_6_xh2a_w8a8.yaml"


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MiniCPM-V-4.6 Merak workflow.")
    parser.add_argument("--model-dir", required=True, help="MiniCPM-V-4.6 HF model directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG)
    parser.add_argument("--quant-output-dir", default="work_dirs/minicpm_v_4_6_merak_quant")
    parser.add_argument("--export-output-dir", default="work_dirs/minicpm_v_4_6_merak_export")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--token-capacity", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=None)
    parser.add_argument("--prefill-length", type=int, default=None)
    parser.add_argument("--quant-type", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--golden-image", default=None)
    parser.add_argument("--golden-prompt", default="请描述这张图片。")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _overrides(args: argparse.Namespace) -> dict[str, object]:
    result: dict[str, object] = {}
    if args.token_capacity is not None:
        result["export.vision.token_capacity"] = args.token_capacity
    if args.context_length is not None:
        result["export.model.context_max_length"] = args.context_length
    if args.prefill_length is not None:
        result["export.model.prefill_chunk_length"] = args.prefill_length
    if args.quant_type is not None:
        for component in ("vision_4x", "vision_16x", "llm"):
            result[f"export.components.{component}.quant_type"] = args.quant_type
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
    print(f"export_meta: {Path(export_result.work_dir) / 'export_meta_info.json'}")
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
