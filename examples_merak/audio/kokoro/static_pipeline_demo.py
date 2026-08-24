from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

from xhmodel_merak.xh_other_model.models.kokoro.assets import resolve_model_assets
from xhmodel_merak.xh_other_model.models.kokoro.graph import GRAPH_ROLES
from xhmodel_merak.xh_other_model.models.kokoro.host import (
    DEFAULT_INPUT_IDS,
    SAMPLE_RATE,
    load_voice_style,
)
from xhmodel_merak.xh_other_model.models.kokoro.runtime import KokoroStaticRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the exported Kokoro static pipeline")
    parser.add_argument("--model-dir", required=True, help="source asset directory")
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--backend", choices=("ort", "hmonnx"), default="ort")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ort-fallback",
        action="append",
        default=[],
        choices=GRAPH_ROLES,
        help="run this role with its static ONNX/ORT graph; repeat for multiple roles",
    )
    parser.add_argument("--output-wav", default="work_dirs/kokoro_static.wav")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    assets = resolve_model_assets(args.model_dir)
    tokens = np.asarray(DEFAULT_INPUT_IDS, dtype=np.int32)
    style = load_voice_style(assets.voice, phoneme_count=tokens.size - 2)
    runtime = KokoroStaticRuntime.from_export(
        args.export_dir,
        backend=args.backend,
        device=args.device,
        ort_fallback_roles=args.ort_fallback,
    )
    waveform, metadata = runtime.synthesize(
        tokens,
        style,
        speed=args.speed,
        seed=args.seed,
    )
    output = Path(args.output_wav).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.round(np.clip(waveform, -1, 1) * 32767).astype("<i2")
    with wave.open(str(output), "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(SAMPLE_RATE)
        target.writeframes(pcm.tobytes())
    print(
        json.dumps(
            {
                **metadata,
                "backend": args.backend,
                "ort_fallback": args.ort_fallback,
                "output_wav": str(output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
