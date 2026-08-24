from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

from xhmodel_merak.xh_other_model.models.kokoro.assets import resolve_model_assets
from xhmodel_merak.xh_other_model.models.kokoro.host import (
    DEFAULT_INPUT_IDS,
    SAMPLE_RATE,
    load_voice_style,
)
from xhmodel_merak.xh_other_model.models.kokoro.runtime import OrtRunner
from xhmodel_merak.xh_other_model.models.kokoro.single_graph import SINGLE_GRAPH_ROLE


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kokoro's end-to-end static ONNX")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-wav", default="work_dirs/kokoro_single_graph.wav")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()
    if args.speed <= 0:
        raise ValueError("--speed must be positive")

    export_dir = Path(args.export_dir).expanduser().resolve()
    meta = json.loads((export_dir / "export_meta_info.json").read_text(encoding="utf-8"))
    if meta.get("graph_mode") != "single_graph":
        raise ValueError(f"{export_dir} is not a Kokoro single-graph export")
    text_max_length = int(meta["text_max_length"])
    frame_max_length = int(meta["frame_max_length"])
    tokens = np.asarray(DEFAULT_INPUT_IDS, dtype=np.int32)
    if tokens.size > text_max_length:
        raise ValueError(f"token length {tokens.size} exceeds T bucket {text_max_length}")
    input_ids = np.zeros((1, text_max_length), dtype=np.int32)
    input_ids[0, : tokens.size] = tokens
    assets = resolve_model_assets(args.model_dir)
    style = load_voice_style(assets.voice, phoneme_count=tokens.size - 2).numpy()
    component = meta["components"][SINGLE_GRAPH_ROLE]
    output = OrtRunner(export_dir / component["onnx_file"]).run(
        {
            "input_ids": input_ids,
            "style": style.astype(np.float32, copy=False),
            "speed": np.asarray([args.speed], dtype=np.float32),
            "valid_len": np.asarray([tokens.size], dtype=np.int32),
        }
    )
    valid_frames = int(np.asarray(output["valid_frames"]).reshape(-1)[0])
    if valid_frames > frame_max_length:
        raise ValueError(
            f"duration produced F={valid_frames}, exceeding bucket F={frame_max_length}; retry a larger F bucket"
        )
    valid_samples = valid_frames * int(meta["samples_per_frame"])
    waveform = np.asarray(output["waveform"], dtype=np.float32).reshape(-1)[:valid_samples]
    destination = Path(args.output_wav).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(waveform, -1, 1) * 32767).astype("<i2")
    with wave.open(str(destination), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm.tobytes())
    print(
        json.dumps(
            {
                "output_wav": str(destination),
                "duration": np.asarray(output["duration"])[0, : tokens.size].tolist(),
                "frames": valid_frames,
                "samples": valid_samples,
                "seconds": valid_samples / SAMPLE_RATE,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
