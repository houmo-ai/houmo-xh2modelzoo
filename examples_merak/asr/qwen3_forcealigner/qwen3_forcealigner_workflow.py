import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG = "configs_merak/workflows/xh2a/other_models/qwen3_forcealigner/0_6b/qwen3_forcealigner.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Qwen3-ForceAligner Merak workflow.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--config-path", default=DEFAULT_CONFIG)
    parser.add_argument("--quant-output-dir", default="work_dirs/qwen3_forcealigner_quant")
    parser.add_argument("--export-output-dir", default="work_dirs/qwen3_forcealigner_export")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-audio-length", type=int, default=None)
    parser.add_argument("--sequence-length", type=int, default=None)
    parser.add_argument("--quant-type", default=None)
    parser.add_argument("--dump-golden", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    from xhmodel_merak.workflows import AutoWorkflow

    overrides = {}
    if args.max_audio_length is not None:
        overrides["export.audio.max_audio_length"] = args.max_audio_length
    if args.sequence_length is not None:
        overrides["export.prefill.sequence_length"] = args.sequence_length
    if args.quant_type is not None:
        overrides["export.audio.quant_type"] = args.quant_type
        overrides["export.prefill.quant_type"] = args.quant_type

    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        debug=args.debug,
    )
    quant_result = workflow.quant(args.quant_output_dir, args.device, overrides)
    export_dir = Path(args.export_output_dir)
    if args.overwrite and export_dir.exists():
        shutil.rmtree(export_dir)
    export_result = workflow.export(quant_result, str(export_dir), args.device, overrides)
    print(f"export_result: {export_result}")
    if args.dump_golden:
        golden_dir = workflow.dump_golden(export_result, args.device)
        print(f"golden_dir: {golden_dir}")


if __name__ == "__main__":
    main()
