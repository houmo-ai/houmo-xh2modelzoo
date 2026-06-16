"""Compare FunASR-Nano floating-point inference with the HMONNX wrapper."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import librosa

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from funasr import AutoModel
from xh_model_zoo.xh_llm.models.funasr_nano import FunASRNanoHMONNXModel


def _cer(ref: str, hyp: str) -> float:
    ref_chars = list(ref)
    hyp_chars = list(hyp)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    dp = list(range(len(hyp_chars) + 1))
    for i, rc in enumerate(ref_chars, start=1):
        prev, dp[0] = dp[0], i
        for j, hc in enumerate(hyp_chars, start=1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (rc != hc))
            prev = cur
    return dp[-1] / len(ref_chars)


def _read_audio_list(audio: str) -> List[str]:
    path = Path(audio).expanduser()
    if path.suffix in {".scp", ".txt"}:
        items = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            parts = line.split(maxsplit=1)
            items.append(parts[-1])
        return items
    return [str(path)]


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", default="/data01/datasets/Funasr/Fun-ASR-Nano-2512")
    parser.add_argument("--work-dir", default="work_dirs/funasr_nano_xh2a")
    parser.add_argument("--audio", required=True, help="Audio path or wav.scp/txt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    fp_model = AutoModel(model=args.model_dir, device=args.device, trust_remote_code=True, disable_update=True)
    hm_model = FunASRNanoHMONNXModel(str(Path(args.work_dir).expanduser().resolve()))
    hm_model.to(args.device)

    total_cer = 0.0
    count = 0
    for audio_path in _read_audio_list(args.audio):
        fp_res = fp_model.generate(input=audio_path)[0]
        fp_text = fp_res.get("text", "")
        wav, _ = librosa.load(audio_path, sr=16000, mono=True)
        hm_res = hm_model.generate(wav, max_new_tokens=args.max_new_tokens)[0]
        hm_text = hm_res.get("text", "")
        cer = _cer(fp_text, hm_text)
        total_cer += cer
        count += 1
        print(f"[{count}] {audio_path}")
        print(f"  FP      : {fp_text}")
        print(f"  HMONNX  : {hm_text}")
        print(f"  CER     : {cer:.4f}")

    if count:
        print(f"Average CER vs FP: {total_cer / count:.4f}")


if __name__ == "__main__":
    main()
