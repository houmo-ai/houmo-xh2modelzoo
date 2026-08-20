from __future__ import annotations

import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG = "configs_merak/workflows/xh2a/llm_models/minicpm_o_4_5/minicpm_o_4_5_xh2a_w8a8_gptq.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MiniCPM-o-4.5 Merak workflow")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--quant-output-dir",
        default=None,
        help="Quantization output directory. Default: work_dirs/<config_stem>_quant",
    )
    parser.add_argument(
        "--export-output-dir",
        default=None,
        help="Export output directory. Default: work_dirs/<config_stem>_export",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--golden-mode", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--golden-video", type=Path)
    parser.add_argument("--golden-question", default="请简短描述视频内容。")
    parser.add_argument("--golden-max-new-tokens", type=int, default=32)
    return parser.parse_args()


def build_golden_request(args: argparse.Namespace) -> dict[str, object]:
    if args.golden_mode == "synthetic":
        return {"mode": "synthetic"}
    if args.golden_video is None:
        raise ValueError("--golden-video is required with --golden-mode real")
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import normalize_minicpmo_video

    contents = normalize_minicpmo_video(args.golden_video, include_audio=True, stack_frames=1)
    return {
        "mode": "real",
        "messages": [{"role": "user", "content": [*contents, args.golden_question]}],
        "chat_kwargs": {
            "omni_mode": True,
            "max_new_tokens": args.golden_max_new_tokens,
            "generate_audio": False,
            "use_tts_template": False,
            "max_slice_nums": 1,
        },
    }


def main() -> None:
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    config_stem = Path(args.config_path).stem
    if args.quant_output_dir is None:
        args.quant_output_dir = f"work_dirs/{config_stem}_quant"
    if args.export_output_dir is None:
        args.export_output_dir = f"work_dirs/{config_stem}_export"

    if args.overwrite:
        shutil.rmtree(args.export_output_dir, ignore_errors=True)
    workflow = AutoWorkflow.from_config(model_dir=args.model_dir, config_path=args.config_path)
    quant_result = workflow.quant(output_dir=args.quant_output_dir, device=args.device)
    export_result = workflow.export(quant_result=quant_result, output_dir=args.export_output_dir, device=args.device)
    print(f"export_result: {export_result}")
    if args.dump_golden:
        print(workflow.dump_golden(export_result, device=args.device, input_messages=build_golden_request(args)))


if __name__ == "__main__":
    main()
