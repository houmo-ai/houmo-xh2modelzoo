from __future__ import annotations

import argparse
import json
from pathlib import Path

from xhmodel_merak.xh_other_model.models.silero_vad.runtime import (
    HmonnxRunner,
    OrtRunner,
    decision_stats,
    error_stats,
    load_audio,
    load_export_meta,
    run_audio,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Silero static ONNX and HMONNX on real audio")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--audio")
    parser.add_argument("--seconds", type=float, default=3.0)
    parser.add_argument(
        "--output",
        default="work_dirs/silero_vad_merak/real_audio_eval.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir).expanduser().resolve()
    source = model_dir / "src/silero_vad/data/silero_vad.onnx"
    if not source.is_file():
        source = model_dir / "silero_vad.onnx"
    audio = Path(args.audio).expanduser().resolve() if args.audio else model_dir / "tests/data/test.wav"
    root, meta = load_export_meta(args.export_dir)
    result: dict[str, object] = {
        "audio": str(audio),
        "source_model": str(source),
        "target_device": meta["target_device"],
    }
    for rate, component in meta["components"].items():
        sample_rate = int(rate)
        waveform = load_audio(audio, sample_rate, args.seconds)
        source_values = run_audio(OrtRunner(source, source_graph=True), waveform, sample_rate)
        static_values = run_audio(
            OrtRunner(root / component["onnx_file"]),
            waveform,
            sample_rate,
        )
        hmonnx_values = run_audio(
            HmonnxRunner(root / component["hmonnx_file"]),
            waveform,
            sample_rate,
        )
        result[rate] = {
            "frames": int(source_values.size),
            "static_vs_source": {
                "probability": error_stats(source_values, static_values),
                "decision": decision_stats(source_values, static_values),
            },
            "hmonnx_vs_source": {
                "probability": error_stats(source_values, hmonnx_values),
                "decision": decision_stats(source_values, hmonnx_values),
            },
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
