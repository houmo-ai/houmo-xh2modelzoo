import argparse
import re
import shutil
from pathlib import Path


DEFAULT_CONFIG_PATH = (
    "configs_merak/workflows/xh2a/other_models/lingbot_video/dense_1_3b/lingbot_video_dense_1_3b_w8a8.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the LingBot Video Merak workflow.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--quant-output-dir", default="work_dirs/lingbot_video_quant")
    parser.add_argument("--export-output-dir", default="work_dirs/lingbot_video_w8a8")
    parser.add_argument(
        "--export-components",
        nargs="+",
        default=None,
        choices=("text_encoder", "visual_encoder", "transformer", "vae_encoder", "vae_decoder"),
        help="Export only the selected components instead of the full pipeline.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quant-type", default=None)
    parser.add_argument("--mode", choices=("t2i", "t2v", "ti2v"), default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def _config_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    for name, value in (
        ("mode", getattr(args, "mode", None)),
        ("height", getattr(args, "height", None)),
        ("width", getattr(args, "width", None)),
        ("num_frames", getattr(args, "num_frames", None)),
        ("num_inference_steps", getattr(args, "steps", None)),
    ):
        if value is not None:
            overrides[f"export.geometry.{name}"] = value
    if args.quant_type is None:
        return overrides
    components = (
        "text_encoder",
        "visual_encoder",
        "transformer",
        "vae_encoder",
        "vae_decoder",
    )
    overrides.update({f"export.components.{component}.quant_type": args.quant_type for component in components})
    activation_match = re.search(r"a(\d+)", args.quant_type.lower())
    if activation_match:
        activation_bits = int(activation_match.group(1))
        for component in ("text_encoder", "visual_encoder", "transformer"):
            for schema in ("act_scheme", "act_schema_2"):
                overrides[f"export.components.{component}.ops.MatMul.{schema}.bits"] = activation_bits
        for operand in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits"):
            overrides[f"export.components.transformer.flash_attention.{operand}"] = activation_bits
    return overrides


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    _remove_output_dir_if_needed(args.export_output_dir, args.overwrite)

    from xhmodel_merak.workflows import AutoWorkflow

    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        debug=args.debug,
    )
    overrides = _config_overrides(args)
    if args.export_components is not None:
        overrides["export.components.text_encoder.enabled"] = "text_encoder" in args.export_components
        overrides["export.components.visual_encoder.enabled"] = "visual_encoder" in args.export_components
        overrides["export.components.transformer.enabled"] = "transformer" in args.export_components
        overrides["export.components.vae_encoder.enabled"] = "vae_encoder" in args.export_components
        overrides["export.components.vae_decoder.enabled"] = "vae_decoder" in args.export_components
    quant_result = workflow.quant(
        output_dir=args.quant_output_dir,
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
        golden_dir = workflow.dump_golden(export_result, device=args.device)
        print(f"golden_dir: {golden_dir}")


if __name__ == "__main__":
    main()
