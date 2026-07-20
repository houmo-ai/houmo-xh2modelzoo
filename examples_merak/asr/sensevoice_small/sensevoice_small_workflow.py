import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG_PATH = "configs_merak/workflows/xh2a/other_models/sensevoice_small/sensevoice_small_xh2a.yaml"
DEFAULT_OUTPUT_DIR = "work_dirs/sensevoice_small_merak/export_xh2a_w8a8h1_sefp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SenseVoiceSmall Merak export workflow.")
    parser.add_argument("--model-dir", required=True, help="Local FunASR SenseVoiceSmall model directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda:0", help="Device used for multi-sample PTQ and golden data.")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--quant-type", default=None)
    parser.add_argument("--calib-metric", choices=["minmax", "mse", "kl"], default=None)
    parser.add_argument("--calib-samples", type=int, default=None)
    parser.add_argument("--calib-file", default=None, help="Use a local calibration .pth instead of HF data.")
    parser.add_argument("--hf-dataset", default=None)
    parser.add_argument("--hf-config", default=None)
    parser.add_argument("--hf-split", default=None)
    parser.add_argument("--hf-audio-field", default=None)
    parser.add_argument("--hf-streaming", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--onnx-only", action="store_true", help="Skip HMONNX conversion.")
    parser.add_argument(
        "--dynamic-onnx",
        action="store_true",
        help="Export dynamic ONNX; only valid together with --onnx-only.",
    )
    parser.add_argument("--no-simplify", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--golden-audio", default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def build_overrides(args: argparse.Namespace) -> dict[str, object]:
    if args.dynamic_onnx and not args.onnx_only:
        raise ValueError("--dynamic-onnx requires --onnx-only because HMONNX export needs a static input shape")
    if args.onnx_only and args.dump_golden:
        raise ValueError("--dump-golden requires HMONNX export and cannot be combined with --onnx-only")
    overrides: dict[str, object] = {}
    values = {
        "export.onnx.max_seq_len": args.max_seq_len,
        "export.hmonnx.quant_type": args.quant_type,
        "export.hmonnx.calib_metric": args.calib_metric,
        "export.hmonnx.calibration.samples": args.calib_samples,
        "export.hmonnx.calibration.hf_dataset": args.hf_dataset,
        "export.hmonnx.calibration.hf_config": args.hf_config,
        "export.hmonnx.calibration.hf_split": args.hf_split,
        "export.hmonnx.calibration.hf_audio_field": args.hf_audio_field,
        "export.hmonnx.calibration.hf_streaming": args.hf_streaming,
    }
    overrides.update({key: value for key, value in values.items() if value is not None})
    if args.calib_file:
        overrides["export.hmonnx.calibration.source"] = "file"
        overrides["export.hmonnx.calibration.file"] = args.calib_file
    if args.onnx_only:
        overrides["export.hmonnx.enabled"] = False
    if args.dynamic_onnx:
        overrides["export.onnx.static"] = False
    if args.no_simplify:
        overrides["export.onnx.simplify"] = False
    return overrides


def main() -> None:
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.overwrite and output_dir.exists():
        shutil.rmtree(output_dir)

    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        seed=args.seed,
        debug=args.debug,
    )
    overrides = build_overrides(args)
    quant_result = workflow.quant(
        output_dir=str(output_dir / "quant"),
        device=args.device,
        config_overrides=overrides,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=str(output_dir),
        device=args.device,
        config_overrides=overrides,
    )
    print(f"Export metadata: {output_dir / 'export_meta_info.json'}")
    if args.dump_golden:
        input_messages = {"audio": args.golden_audio} if args.golden_audio else None
        golden_meta = workflow.dump_golden(
            export_result=export_result,
            device=args.device,
            input_messages=input_messages,
        )
        print(f"Golden metadata: {golden_meta}")


if __name__ == "__main__":
    main()
