#!/usr/bin/env python3
"""CosyVoice3 HMONNX evaluation.

Builds CosyVoice3HMONNXInference from export_meta_info.json and runs
batch zero-shot TTS over a CV3-Eval scp directory.

Usage:
  python cosyvoice3_eval.py \
    --work-dir work_dirs/cosyvoice3_new2/export \
    --data-dir /path/to/CV3-Eval/data/zero_shot/zh \
    --output-dir /path/to/output \
    --max-samples 10
"""

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import soundfile as sf
import torch
from tqdm import tqdm


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from hmonnx_utils import build_cosyvoice3_model  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _read_scp(path):
    result = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def load_eval_data(data_dir, max_samples=None):
    data_dir = Path(data_dir).resolve()
    text_dict = _read_scp(data_dir / "text")
    prompt_text_dict = _read_scp(data_dir / "prompt_text")
    prompt_wav_dict = _read_scp(data_dir / "prompt_wav.scp")

    cv3_eval_base = data_dir.parents[2]
    for utt, wav_path in prompt_wav_dict.items():
        if not os.path.isabs(wav_path):
            prompt_wav_dict[utt] = str(cv3_eval_base / wav_path)

    utts = list(text_dict.keys())
    if max_samples:
        utts = utts[:max_samples]
    return utts, text_dict, prompt_text_dict, prompt_wav_dict


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def worker(model, utts, text_dict, prompt_text_dict, prompt_wav_dict, output_dir, seed=1986):
    torch.manual_seed(seed)

    results = []
    for idx, utt in enumerate(tqdm(utts, desc="推理进度")):
        wav_path_out = os.path.join(output_dir, f"{utt}.wav")
        if os.path.exists(wav_path_out):
            logging.info(f"{wav_path_out} 已存在，跳过")
            continue
        if utt not in prompt_text_dict or utt not in text_dict:
            logging.warning(f"utt {utt} 缺少prompt_text或target_text，跳过")
            continue

        text = text_dict[utt]
        prompt_text = prompt_text_dict[utt]
        prompt_wav = prompt_wav_dict[utt]

        logging.info(f"synthesis text {text}")
        start = time.time()
        wavs, sr = model.generate(
            text=text,
            prompt_wav=prompt_wav,
            prompt_text=prompt_text,
        )
        elapsed = time.time() - start

        os.makedirs(output_dir, exist_ok=True)
        sf.write(wav_path_out, wavs[0].squeeze(0), sr)
        results.append({"utt": utt, "text": text, "output": wav_path_out, "elapsed": round(elapsed, 2)})
        print(f"  [{idx + 1}/{len(utts)}] {utt}: {wav_path_out} ({elapsed:.2f}s)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CosyVoice3 HMONNX eval (post-migration)")
    parser.add_argument("--work-dir", required=True, help="Export output dir (e.g. work_dirs/cosyvoice3_new2/export)")
    parser.add_argument("--data-dir", required=True, help="Eval data dir with text/prompt_text/prompt_wav.scp")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output dir for wav files (default: <work-dir>/eval_out)"
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1986)
    cli_args = parser.parse_args()

    work_dir = Path(cli_args.work_dir)
    output_dir = cli_args.output_dir or str(work_dir / "eval_out")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[resolve] work_dir: {work_dir}")
    model = build_cosyvoice3_model(work_dir, cli_args.device)

    utts, text_dict, prompt_text_dict, prompt_wav_dict = load_eval_data(cli_args.data_dir, cli_args.max_samples)

    print(f"\n开始推理 (前{len(utts)}条)")
    print(f"  输出目录: {output_dir}")
    print(f"  LLM:    {work_dir / 'LLM'}")
    print()

    results = worker(model, utts, text_dict, prompt_text_dict, prompt_wav_dict, output_dir, seed=cli_args.seed)

    report_path = os.path.join(output_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    wavs = sorted(glob.glob(f"{output_dir}/*.wav"))
    print(f"\n完成! 产出 {len(wavs)} 条音频:")
    for w in wavs:
        size = os.path.getsize(w) // 1024
        print(f"  {os.path.basename(w)} ({size}KB)")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
