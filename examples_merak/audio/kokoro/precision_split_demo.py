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
from xhmodel_merak.xh_other_model.models.kokoro.precision_split import (
    KokoroPrecisionSplitRuntime,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Kokoro's primary precision-split pipeline")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--output-wav", default="work_dirs/kokoro_precision_split.wav")
    parser.add_argument("--backend", choices=("ort", "hmonnx"), default="hmonnx")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--speed", type=float, default=1.0)
    args = parser.parse_args()

    export_dir = Path(args.export_dir).expanduser().resolve()
    assets = resolve_model_assets(args.model_dir)
    tokens = np.asarray(DEFAULT_INPUT_IDS, dtype=np.int32)
    style = load_voice_style(assets.voice, phoneme_count=tokens.size - 2).numpy()
    runtime = KokoroPrecisionSplitRuntime.from_export(
        export_dir,
        backend=args.backend,
        device=args.device,
    )
    waveform, synthesis = runtime.synthesize(tokens, style, speed=args.speed)

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
                "backend": args.backend,
                **synthesis,
                "seconds": waveform.size / SAMPLE_RATE,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
