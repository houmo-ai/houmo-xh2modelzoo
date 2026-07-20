import argparse
import json
import os


os.environ.setdefault("USE_TRITON_MATMUL", "1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a migrated SenseVoiceSmall HMONNX.")
    parser.add_argument("--export-dir", required=True, help="Directory containing export_meta_info.json.")
    parser.add_argument("--assets-dir", default="")
    parser.add_argument("--tokens", default="")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest-jsonl", default="")
    source.add_argument("--wav-scp", default="")
    source.add_argument("--hf-dataset", default="")
    parser.add_argument("--text", default="", help="Required with --wav-scp.")
    parser.add_argument("--hf-config", default="")
    parser.add_argument("--hf-split", default="test")
    parser.add_argument("--hf-streaming", action="store_true")
    parser.add_argument("--hf-audio-field", default="audio")
    parser.add_argument("--hf-text-field", default="")
    parser.add_argument("--hf-text-path", default="")
    parser.add_argument("--keep-rich-tags", action="store_true")
    parser.add_argument("--report", default="work_dirs/sensevoice_small_merak/report/quant_report.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from xhmodel_merak.xh_other_model.models.sensevoice_small.evaluation import evaluate_export
    from xhquant.xhonnxruntime import config as runtime_config

    runtime_config.disable_progress = True
    runtime_config.verbose_progress = False
    report = evaluate_export(
        export_dir=args.export_dir,
        backend="hmonnx",
        report_path=args.report,
        device=args.device,
        assets_dir=args.assets_dir,
        tokens=args.tokens,
        manifest_jsonl=args.manifest_jsonl,
        wav_scp=args.wav_scp,
        text=args.text,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        hf_streaming=args.hf_streaming,
        hf_audio_field=args.hf_audio_field,
        hf_text_field=args.hf_text_field,
        hf_text_path=args.hf_text_path,
        limit=args.limit,
        strip_tags=not args.keep_rich_tags,
        fast=args.fast,
    )
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
