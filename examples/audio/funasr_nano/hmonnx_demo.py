"""Greedy HMONNX demo for FunASR-Nano exported modules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from xh_model_zoo.xh_llm.models.funasr_nano import FunASRNanoHMONNXModel


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--work-dir", default="work_dirs/funasr_nano_xh2a")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--language", default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    model = FunASRNanoHMONNXModel(str(Path(args.work_dir).expanduser().resolve()))
    model.to(args.device)
    audio, _ = librosa.load(args.audio, sr=16000, mono=True)
    result = model.generate(audio, language=args.language, max_new_tokens=args.max_new_tokens)[0]
    print(result["text"])


if __name__ == "__main__":
    main()
