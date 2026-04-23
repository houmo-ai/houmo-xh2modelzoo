"""
F5-TTS 浮点模型推理测试
========================

使用 F5-TTS 原生 API 加载模型，推理中英文测试样例，
生成 golden reference 音频和 mel 频谱图。

用法:
    python f5tts_float_infer.py \
        --ckpt /data01/nfs_shared/ASR_TTS/F5TTS_base/F5TTS_Base/model_1200000.safetensors \
        --vocab /data01/nfs_shared/ASR_TTS/F5TTS_base/F5TTS_Base/vocab.txt \
        --out-dir work_dirs/f5tts/float_golden
"""

import argparse
import json
from pathlib import Path

import numpy as np

from f5tts_common import (
    F5TTS_SRC,
    MODEL_CKPT,
    REF_EN_WAV,
    REF_ZH_WAV,
    VOCAB_PATH,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=str, default=MODEL_CKPT)
    p.add_argument("--vocab", type=str, default=VOCAB_PATH)
    p.add_argument("--out-dir", type=str, default="work_dirs/f5tts/float_golden")
    p.add_argument("--device", type=str, default="cuda" if _cuda_ok() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _cuda_ok() -> bool:
    import torch
    return torch.cuda.is_available()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 加载 F5-TTS 原生 API
    import sys
    src = str(Path(F5TTS_SRC).parent)
    if src not in sys.path:
        sys.path.insert(0, src)

    from f5_tts.api import F5TTS

    f5tts = F5TTS(
        model="F5TTS_v1_Base",
        ckpt_file=args.ckpt,
        vocab_file=args.vocab,
        device=args.device,
    )

    # 测试样例（跳过 Whisper 转录，直接硬编码 ref_text 避免下载大模型）
    test_cases = [
        (
            "en",
            REF_EN_WAV,
            "Some call me nature, others call me mother nature.",
            "I don't really care what you call me. I've been a silent spectator, "
            "watching species evolve, empires rise and fall.",
        ),
        (
            "zh",
            REF_ZH_WAV,
            "对，这就是我，万人敬仰的太乙真人。",
            "这是我的一段测试语音，用来验证模型推理是否正常工作。",
        ),
    ]

    for lang, ref_wav, ref_text, gen_text in test_cases:
        tag = f"float_{lang}"
        print(f"\n{'='*60}")
        print(f"[{lang}] ref_text: {ref_text}")
        print(f"[{lang}] gen_text: {gen_text}")
        print(f"{'='*60}")

        wav, sr, spec = f5tts.infer(
            ref_file=ref_wav,
            ref_text=ref_text,
            gen_text=gen_text,
            seed=args.seed,
            nfe_step=32,
            cfg_strength=2.0,
        )

        wav_path = out_dir / f"{tag}.wav"
        import soundfile as sf
        sf.write(str(wav_path), wav, sr)
        print(f"saved: {wav_path}")

        # 保存 spec 图（如有）
        try:
            spec_path = out_dir / f"{tag}_spec.png"
            from f5_tts.infer.utils_infer import save_spectrogram
            save_spectrogram(spec, str(spec_path))
            print(f"saved: {spec_path}")
        except Exception:
            pass

    # 元信息
    meta = {
        "seed": args.seed,
        "ckpt": args.ckpt,
        "vocab": args.vocab,
        "device": args.device,
        "test_cases": [
            {"lang": lang, "ref_wav": ref, "ref_text": rt, "gen_text": gt}
            for lang, ref, rt, gt in test_cases
        ],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(f"\nmeta saved: {out_dir / 'meta.json'}")


if __name__ == "__main__":
    main()
