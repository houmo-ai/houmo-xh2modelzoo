"""
F5-TTS 已有音频的 ASR 评测
===========================

对已生成的 wav 文件跑 Whisper ASR，计算 WER/CER。
不重新生成音频，仅做后处理评测。

用法:
    python f5tts_asr_eval.py \
        --wav-dir work_dirs/f5tts_v1/eval/float \
        --whisper-model base \
        --out work_dirs/f5tts_v1/eval/float_asr_results.csv
"""

import argparse
import csv
import struct
import re
import unicodedata
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd

# ── 数据路径（与 f5tts_eval.py 一致） ────────────────────────
COSYVOICE_PREFIX = "You are a helpful assistant.<|endofprompt|>"
DATA_DIR = Path("/data01/home/axel/workspace/repo/xh2modelzoo/examples/audio/Cosyvoice3/data")
ZH_DATA_DIR = DATA_DIR / "data_zero_shot_zh"
EN_DATA_DIR = DATA_DIR / "data_zero_shot"
ZH_PUNC_RE = re.compile(r"[，。！？；：、,.!?;:\"'“”‘’（）()《》〈〉【】\[\]{}…—\-\s]+")
_OPENCC_T2S = None


def _get_opencc_t2s():
    global _OPENCC_T2S
    if _OPENCC_T2S is None:
        try:
            from opencc import OpenCC  # type: ignore
            _OPENCC_T2S = OpenCC("t2s")
        except Exception:
            _OPENCC_T2S = False
    return _OPENCC_T2S


def _normalize_zh_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    converter = _get_opencc_t2s()
    if converter:
        normalized = converter.convert(normalized)
    normalized = ZH_PUNC_RE.sub("", normalized)
    return normalized


def _load_ref_text_map(n_samples: int) -> dict:
    """重建 utt → gen_text 映射（与 f5tts_eval.py 相同逻辑）。"""
    n_total = n_samples * 2

    def _wav_duration(audio_bytes: bytes) -> float:
        byte_rate = struct.unpack_from("<I", audio_bytes, 28)[0]
        idx = audio_bytes.find(b"data")
        if idx < 0:
            return len(audio_bytes) / max(byte_rate, 1)
        data_size = struct.unpack_from("<I", audio_bytes, idx + 4)[0]
        return data_size / max(byte_rate, 1)

    def _load_samples(parquet_dir: Path, n: int, min_sec: float = 2.0, max_sec: float = 12.0):
        files = sorted(glob(str(parquet_dir / "*.parquet")))
        samples = []
        for fp in files:
            if len(samples) >= n:
                break
            df = pd.read_parquet(fp, columns=["utt", "text", "audio_data"])
            for _, row in df.iterrows():
                if len(samples) >= n:
                    break
                dur = _wav_duration(row["audio_data"])
                if not (min_sec <= dur <= max_sec):
                    continue
                text = row["text"]
                if COSYVOICE_PREFIX in text:
                    text = text.split(COSYVOICE_PREFIX, 1)[1].strip()
                if not text:
                    continue
                samples.append({"utt": row["utt"], "text": text})
        return samples

    en_all = _load_samples(EN_DATA_DIR, n_total)
    zh_all = _load_samples(ZH_DATA_DIR, n_total)

    en_refs, en_gens = en_all[:n_samples], en_all[n_samples:]
    zh_refs, zh_gens = zh_all[:n_samples], zh_all[n_samples:]
    if len(en_gens) < n_samples:
        en_gens = en_refs
    if len(zh_gens) < n_samples:
        zh_gens = zh_refs

    ref_map = {}
    for i in range(min(n_samples, len(en_refs), len(en_gens))):
        ref_map[("en", en_refs[i]["utt"])] = en_gens[i]["text"]
    for i in range(min(n_samples, len(zh_refs), len(zh_gens))):
        ref_map[("zh", zh_refs[i]["utt"])] = zh_gens[i]["text"]
    return ref_map


def _edit_distance(hyp: list, ref: list) -> int:
    n, m = len(ref), len(hyp)
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        curr = [i] + [0] * m
        for j in range(1, m + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[m]


def _compute_wer(hyp: str, ref: str, lang: str) -> float:
    if lang == "zh":
        h, r = list(_normalize_zh_text(hyp)), list(_normalize_zh_text(ref))
    else:
        h, r = hyp.lower().split(), ref.lower().split()
    if not r:
        return 0.0
    return _edit_distance(h, r) / len(r)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-dir", type=str, default=None,
                        help="单模型目录，目录结构应为 <wav-dir>/<lang>/*.wav")
    parser.add_argument(
        "--model-dir",
        type=str,
        nargs="*",
        default=None,
        help="多模型对比输入，格式: model_name=wav_dir，可传多个",
    )
    parser.add_argument("--whisper-model", type=str, default="base")
    parser.add_argument("--n-samples", type=int, default=30)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    model_dirs = {}
    if args.model_dir:
        for item in args.model_dir:
            if "=" not in item:
                raise ValueError(f"--model-dir 参数格式错误: {item}，应为 model_name=wav_dir")
            model_name, wav_dir = item.split("=", 1)
            model_name = model_name.strip()
            wav_dir = wav_dir.strip()
            if not model_name or not wav_dir:
                raise ValueError(f"--model-dir 参数格式错误: {item}，应为 model_name=wav_dir")
            model_dirs[model_name] = Path(wav_dir)
    elif args.wav_dir:
        wav_dir = Path(args.wav_dir)
        model_dirs[wav_dir.name] = wav_dir
    else:
        raise ValueError("必须提供 --wav-dir 或 --model-dir")

    first_dir = next(iter(model_dirs.values()))
    out_path = Path(args.out) if args.out else first_dir / "asr_results.csv"

    # ── 加载参考文本 ──
    print("加载参考文本...")
    ref_map = _load_ref_text_map(args.n_samples)

    # ── 加载 Whisper ──
    import whisper
    print(f"加载 Whisper ({args.whisper_model})...")
    model = whisper.load_model(args.whisper_model, device=args.device)

    # ── 评测 ──
    results = []
    for model_name, wav_root in model_dirs.items():
        print(f"\n====== 模型: {model_name} ({wav_root}) ======")
        for lang in ["en", "zh"]:
            lang_dir = wav_root / lang
            if not lang_dir.exists():
                print(f"  跳过 {lang}: 目录不存在")
                continue

            wavs = sorted(lang_dir.glob("*.wav"))
            print(f"\n{lang.upper()}: {len(wavs)} 个 wav 文件")

            for wav_path in wavs:
                utt = wav_path.stem
                key = (lang, utt)

                if key not in ref_map:
                    print(f"  [warn] {utt} 无参考文本，跳过")
                    continue

                ref_text = ref_map[key]

                # ASR 转录
                result = model.transcribe(str(wav_path))
                hyp_text = result["text"].strip()

                # 计算 WER/CER
                wer = _compute_wer(hyp_text, ref_text, lang)

                results.append({
                    "model": model_name,
                    "utt": utt,
                    "lang": lang,
                    "ref_text": ref_text[:100],
                    "hyp_text": hyp_text[:100],
                    "wer": round(wer, 4),
                })
                metric = "CER" if lang == "zh" else "WER"
                print(f"  {utt}: {metric}={wer:.4f}")

    # ── 保存结果 ──
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # ── 汇总 ──
    print("\n" + "=" * 78)
    print(f"{'模型':<14} {'语言':<6} {'N':>3} {'WER/CER 平均':>12} {'最小':>8} {'最大':>8}")
    print("-" * 78)
    for model_name in model_dirs.keys():
        for lang in ["en", "zh"]:
            wers = [r["wer"] for r in results if r["lang"] == lang and r["model"] == model_name]
            if wers:
                print(f"{model_name:<14} {lang:<6} {len(wers):>3} {np.mean(wers):>11.4f} "
                      f"{np.min(wers):>8.4f} {np.max(wers):>8.4f}")
    print("=" * 78)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
