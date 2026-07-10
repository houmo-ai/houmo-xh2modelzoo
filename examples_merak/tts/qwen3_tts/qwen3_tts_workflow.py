import argparse
import shutil
from pathlib import Path


DEFAULTS = {
    "0_6B_base": {
        "config": "configs_merak/workflows/xh2a/other_models/qwen3_tts/0_6b_base/qwen3_tts_12hz_0_6b_base.yaml",
        "output": "work_dirs/Qwen3-TTS-12Hz-0.6B-Base_XH2a",
    },
    "0_6B_customvoice": {
        "config": "configs_merak/workflows/xh2a/other_models/qwen3_tts/0_6b_customvoice/qwen3_tts_12hz_0_6b_customvoice.yaml",
        "output": "work_dirs/Qwen3-TTS-12Hz-0.6B-CustomVoice_XH2a",
    },
    "1_7B_voicedesign": {
        "config": "configs_merak/workflows/xh2a/other_models/qwen3_tts/1_7b_voicedesign/qwen3_tts_12hz_1_7b_voicedesign.yaml",
        "output": "work_dirs/Qwen3-TTS-12Hz-1.7B-VoiceDesign_XH2a",
    },
}


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Qwen3-TTS Merak workflow.")
    parser.add_argument(
        "--variant",
        choices=sorted(DEFAULTS),
        default="0_6B_customvoice",
        help="Qwen3-TTS variant to export.",
    )
    parser.add_argument("--model-dir", required=True, help="HF model directory.")
    parser.add_argument("--config-path", default=None, help="Override workflow YAML path.")
    parser.add_argument("--export-output-dir", default=None, help="Override export output directory.")
    parser.add_argument("--quant-output-dir", default="work_dirs/qwen3_tts_quant")
    parser.add_argument("--device", default="cuda", help="Device passed to workflow export.")
    parser.add_argument("--target-device", default=None, help="Override export.target_device.")
    parser.add_argument("--quant-type", default=None, help="Override all component quant types.")
    parser.add_argument("--components", default=None, help="Comma-separated component list override.")
    parser.add_argument("--dump-golden", action="store_true", help="Generate golden data after export.")
    parser.add_argument("--overwrite", action="store_true", help="Remove export output dir first.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _build_config_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.target_device is not None:
        overrides["export.target_device"] = args.target_device
    if args.components:
        overrides["export.components"] = [item.strip() for item in args.components.split(",") if item.strip()]
    if args.quant_type is not None:
        for name in (
            "talker",
            "code_predictor",
            "text_projection",
            "speech_tokenizer",
            "base_frontend",
            "stateful_decoder",
        ):
            overrides[f"export.quant_types.{name}"] = args.quant_type
    return overrides


def main() -> None:
    args = parse_args()
    defaults = DEFAULTS[args.variant]
    config_path = args.config_path or defaults["config"]
    output_dir = args.export_output_dir or defaults["output"]
    _remove_output_dir_if_needed(output_dir, args.overwrite)

    from xhmodel_merak.workflows import AutoWorkflow

    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=config_path,
        debug=args.debug,
    )
    overrides = _build_config_overrides(args)
    quant_result = workflow.quant(
        output_dir=args.quant_output_dir,
        device=args.device,
        config_overrides=overrides,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=output_dir,
        device=args.device,
        config_overrides=overrides,
    )
    print(f"export_result: {export_result}")
    if args.dump_golden:
        golden_dir = workflow.dump_golden(
            export_result=export_result,
            device=args.device,
        )
        print(f"golden_dir: {golden_dir}")


if __name__ == "__main__":
    main()
