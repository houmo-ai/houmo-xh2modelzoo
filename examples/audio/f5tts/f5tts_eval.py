"""
F5-TTS 量化评估脚本
======================

从数据集加载中英文样本，对比 PyTorch baseline / HMONNX
的合成质量。指标包括 mel 余弦相似度和 WER/CER。

用法:
    python f5tts_eval.py \
        --hmonnx work_dirs/f5tts_v1/static_mask/export_xh2a/hmonnx/f5tts_dit_XH2a.onnx \
        --n-samples 30 \
        --out-dir work_dirs/f5tts_v1/eval
"""

import argparse
import csv
import json
import os
import re
import shutil
import soundfile as sf
import struct
import subprocess
import sys
import time
import unicodedata
from glob import glob
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from f5tts_common import (
    F5TTS_SRC,
    MODEL_CKPT,
    STATIC_N,
    STATIC_NT,
    TARGET_SR,
    VOCAB_PATH,
    extract_mel_spec,
    load_audio,
    load_vocab,
    load_vocos_vocoder,
    ode_sample,
    pinyin_to_tokens,
    text_to_pinyin,
)

# ============================================================
# 常量
# ============================================================

COSYVOICE_PREFIX = "You are a helpful assistant.<|endofprompt|>"

DATA_DIR = Path(
    "/data01/home/axel/workspace/repo/xh2modelzoo/examples/audio/Cosyvoice3/data"
)
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
    """中文 CER 归一化：NFKC、繁转简（可用时）、去标点和空白。"""
    normalized = unicodedata.normalize("NFKC", text or "")
    converter = _get_opencc_t2s()
    if converter:
        normalized = converter.convert(normalized)
    normalized = ZH_PUNC_RE.sub("", normalized)
    return normalized


# ============================================================
# 数据加载
# ============================================================

def _wav_duration(audio_bytes: bytes) -> float:
    """解析 WAV header 获取真实音频时长（秒）。"""
    byte_rate = struct.unpack_from("<I", audio_bytes, 28)[0]
    idx = audio_bytes.find(b"data")
    if idx < 0:
        return len(audio_bytes) / max(byte_rate, 1)
    data_size = struct.unpack_from("<I", audio_bytes, idx + 4)[0]
    return data_size / max(byte_rate, 1)


def _clean_text(raw: str) -> str:
    """去除 CosyVoice3 prompt 前缀，返回纯文本。"""
    if COSYVOICE_PREFIX in raw:
        return raw.split(COSYVOICE_PREFIX, 1)[1].strip()
    return raw.strip()


def _load_samples(
    parquet_dir: str,
    n: int,
    min_sec: float = 2.0,
    max_sec: float = 12.0,
) -> List[Dict]:
    """从目录下所有 parquet 文件加载满足时长范围的样本。"""
    files = sorted(glob(str(Path(parquet_dir) / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"未找到 parquet 文件: {parquet_dir}")

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
            text = _clean_text(row["text"])
            if not text:
                continue
            samples.append({
                "utt": row["utt"],
                "audio_bytes": row["audio_data"],
                "text": text,
                "dur": round(dur, 2),
            })
    return samples


# ============================================================
# PyTorch baseline（F5TTS 官方 API）
# ============================================================

def _load_f5tts_api(ckpt_path: str, vocab_path: str, device: str):
    """加载 F5TTS 原生 API，用于生成 baseline 音频。"""
    import sys
    src = str(Path(F5TTS_SRC).parent)
    if src not in sys.path:
        sys.path.insert(0, src)
    from f5_tts.api import F5TTS
    return F5TTS(model="F5TTS_v1_Base", ckpt_file=ckpt_path,
                 vocab_file=vocab_path, device=device)


# ============================================================
# 量化模型封装
# ============================================================

def _make_ort_fn(onnx_path: str, device: str):
    """Float ONNX (ORT) 推理函数。"""
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_names = {x.name for x in sess.get_inputs()}
    need_lengths = "input_lengths" in input_names

    def fn(x, cond, text, t, input_lengths=None):
        if input_lengths is None:
            input_lengths = torch.tensor([x.shape[1]], device=x.device, dtype=torch.int32)
        feeds = {
            "x": x.cpu().numpy().astype(np.float32),
            "cond": cond.cpu().numpy().astype(np.float32),
            "text": text.cpu().numpy().astype(np.int64),
            "time": t.cpu().numpy().astype(np.float32),
        }
        if need_lengths:
            feeds["input_lengths"] = input_lengths.cpu().numpy().astype(np.int32)
        out = sess.run(None, feeds)[0]
        return torch.from_numpy(out).to(device)

    return fn


def _make_hmonnx_fn(hmonnx_path: str, device: str):
    """HMONNX 推理函数。"""
    from xhquant.api import HMONNXGoldenInference
    sess = HMONNXGoldenInference(hmonnx_path)
    sess.exec_device = torch.device(device)
    state = {"try_with_lengths": True}

    def fn(x, cond, text, t, input_lengths=None):
        if input_lengths is None:
            input_lengths = torch.tensor([x.shape[1]], device=x.device, dtype=torch.int32)
        t_in = t.to(device).half()
        if t_in.dim() == 1:
            t_in = t_in.unsqueeze(1)
        if state["try_with_lengths"]:
            try:
                out = sess(
                    x.to(device).half(),
                    cond.to(device).half(),
                    text.to(device).int(),
                    t_in,
                    input_lengths.to(device).int(),
                )
            except TypeError:
                state["try_with_lengths"] = False
                out = sess(
                    x.to(device).half(),
                    cond.to(device).half(),
                    text.to(device).int(),
                    t_in,
                )
        else:
            out = sess(
                x.to(device).half(),
                cond.to(device).half(),
                text.to(device).int(),
                t_in,
            )
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.float()

    return fn


# ============================================================
# 评估指标
# ============================================================

def _mel_cos_sim(mel1: torch.Tensor, mel2: torch.Tensor) -> float:
    """两个 mel 的余弦相似度（对齐到最短长度）。"""
    min_len = min(mel1.shape[1], mel2.shape[1])
    a = mel1[:, :min_len].flatten().float()
    b = mel2[:, :min_len].flatten().float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-10))


def _edit_distance(hyp: list, ref: list) -> int:
    """标准编辑距离。"""
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
    """EN: 词错误率; ZH: 字符错误率。"""
    if lang == "zh":
        h, r = list(_normalize_zh_text(hyp)), list(_normalize_zh_text(ref))
    else:
        h, r = hyp.lower().split(), ref.lower().split()
    if not r:
        return 0.0
    return _edit_distance(h, r) / len(r)


def _transcribe_whisper(audio_path: str, model) -> str:
    """用预加载的 Whisper 模型转录。"""
    result = model.transcribe(audio_path)
    return result["text"].strip()


# ============================================================
# 推理管线
# ============================================================

def _prepare_ref(ref_wav_path: str):
    """加载参考音频 → cond_mel + N_ref，供所有 model 复用。"""
    audio = load_audio(ref_wav_path)
    cond_mel = extract_mel_spec(audio)
    return cond_mel, cond_mel.shape[1]


def _preprocess_ref_text(ref_text: str) -> str:
    """对齐官方 F5TTS API 的 ref_text 预处理。"""
    # 官方: 确保 ref_text 以 ". " 或 "。" 结尾
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "
    # 官方: 最后一个字符是 ASCII 单字节时追加空格
    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text += " "
    return ref_text


def _synthesize(
    model_fn,
    cond_mel: torch.Tensor,
    N_ref: int,
    ref_text: str,
    gen_text: str,
    vocab_map: dict,
    seed: int,
    device: str,
    nfe_steps: int,
    cfg_strength: float,
) -> torch.Tensor:
    """ODE 采样，返回完整 mel_out (含 ref 部分)。"""
    ref_text = _preprocess_ref_text(ref_text)
    combined = ref_text + gen_text
    tokens = pinyin_to_tokens(text_to_pinyin([combined]), vocab_map, STATIC_NT)

    # Duration estimation — 内联计算，避免 estimate_duration 重复加空格
    local_speed = 0.3 if len(gen_text.encode("utf-8")) < 10 else 1.0
    ref_bytes = max(len(ref_text.encode("utf-8")), 1)
    gen_bytes = len(gen_text.encode("utf-8"))
    duration = min(N_ref + int(N_ref / ref_bytes * gen_bytes / local_speed), STATIC_N)

    return ode_sample(
        model_fn, cond_mel, tokens, duration,
        steps=nfe_steps, cfg_strength=cfg_strength,
        seed=seed, device=device,
    )


# ============================================================
# CLI
# ============================================================

def _parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--pytorch-ckpt", type=str, default=MODEL_CKPT,
                   help="PyTorch 原始 safetensors 路径（baseline）")
    p.add_argument("--hmonnx", type=str, required=True,
                   help="HMONNX 模型路径（默认与 pytorch baseline 对比）")
    p.add_argument("--vocab", type=str, default=VOCAB_PATH)
    p.add_argument("--zh-data", type=str, default=str(ZH_DATA_DIR))
    p.add_argument("--en-data", type=str, default=str(EN_DATA_DIR))
    p.add_argument("--n-samples", type=int, default=30)
    p.add_argument("--nfe-steps", type=int, default=32)
    p.add_argument("--cfg-strength", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out-dir", type=str, default="work_dirs/f5tts_v1/eval")
    p.add_argument("--skip-wer", action="store_true")
    p.add_argument("--whisper-model", type=str, default="base")
    p.add_argument("--min-sec", type=float, default=2.0)
    p.add_argument("--max-sec", type=float, default=12.0)
    p.add_argument("--gpus", type=int, default=1,
                   help="自动拉起的并行 GPU 进程数（>1 时自动按样本分片启动多个子进程）")
    return p.parse_args()


def _launch_multi_gpu_workers(args):
    """单命令自动拉起多 GPU 分片进程，并合并结果 CSV。"""
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible_env.strip():
        visible_gpus = [x.strip() for x in visible_env.split(",") if x.strip()]
    else:
        visible_gpus = [str(i) for i in range(torch.cuda.device_count())]

    if not visible_gpus:
        raise RuntimeError("未检测到可用 GPU，请检查 CUDA 环境。")
    if args.gpus > len(visible_gpus):
        raise ValueError(
            f"--gpus={args.gpus} 超过可见 GPU 数量={len(visible_gpus)}，"
            f"当前 CUDA_VISIBLE_DEVICES={visible_env or '<all>'}"
        )

    workers = []
    shard_out_dirs = []
    script_path = str(Path(__file__).resolve())

    def _add_opt(cmd, key, value):
        if value is None:
            return
        cmd.extend([f"--{key}", str(value)])

    for shard_id in range(args.gpus):
        gpu = visible_gpus[shard_id]
        shard_out_dir = f"{args.out_dir}_shard{shard_id}"
        shard_out_dirs.append(Path(shard_out_dir))

        cmd = [sys.executable, script_path]
        _add_opt(cmd, "pytorch-ckpt", args.pytorch_ckpt)
        _add_opt(cmd, "hmonnx", args.hmonnx)
        _add_opt(cmd, "vocab", args.vocab)
        _add_opt(cmd, "zh-data", args.zh_data)
        _add_opt(cmd, "en-data", args.en_data)
        _add_opt(cmd, "n-samples", args.n_samples)
        _add_opt(cmd, "nfe-steps", args.nfe_steps)
        _add_opt(cmd, "cfg-strength", args.cfg_strength)
        _add_opt(cmd, "seed", args.seed)
        _add_opt(cmd, "device", args.device)
        _add_opt(cmd, "out-dir", shard_out_dir)
        _add_opt(cmd, "whisper-model", args.whisper_model)
        _add_opt(cmd, "min-sec", args.min_sec)
        _add_opt(cmd, "max-sec", args.max_sec)
        _add_opt(cmd, "gpus", 1)
        if args.skip_wer:
            cmd.append("--skip-wer")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["F5TTS_EVAL_NUM_SHARDS"] = str(args.gpus)
        env["F5TTS_EVAL_SHARD_ID"] = str(shard_id)
        print(f"[multi-gpu] 启动 shard {shard_id}/{args.gpus} on GPU {gpu}")
        workers.append(subprocess.Popen(cmd, env=env))

    ret_codes = [p.wait() for p in workers]
    if any(code != 0 for code in ret_codes):
        raise RuntimeError(f"存在分片进程失败，退出码: {ret_codes}")

    merged_rows = []
    all_keys = set()
    for out_dir in shard_out_dirs:
        # 合并各 shard 的音频目录到主 out-dir，保持与单卡一致目录结构。
        for sub in out_dir.iterdir():
            if not sub.is_dir():
                continue
            if sub.name in {"refs"}:
                continue
            target_sub = Path(args.out_dir) / sub.name
            target_sub.mkdir(parents=True, exist_ok=True)
            for src_file in sub.rglob("*"):
                if not src_file.is_file():
                    continue
                rel = src_file.relative_to(sub)
                dst_file = target_sub / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                if not dst_file.exists():
                    shutil.copy2(src_file, dst_file)

        csv_path = out_dir / "eval_results.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                merged_rows.append(row)
                all_keys.update(row.keys())

    merge_out_dir = Path(args.out_dir)
    merge_out_dir.mkdir(parents=True, exist_ok=True)
    merged_csv_path = merge_out_dir / "eval_results.csv"
    if merged_rows:
        keys = sorted(all_keys)
        with open(merged_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(merged_rows)
        print(f"[multi-gpu] 合并结果: {merged_csv_path}")

        # 生成与单卡一致的汇总 json
        def _to_float(row, key):
            try:
                return float(row[key])
            except Exception:
                return None

        summary = {}
        model_names = sorted({r.get("model", "") for r in merged_rows if r.get("model")})
        for model_name in model_names:
            for lang in ["en", "zh"]:
                rows = [r for r in merged_rows if r.get("model") == model_name and r.get("lang") == lang]
                if not rows:
                    continue
                entry = {"n": len(rows)}
                durs = [v for v in (_to_float(r, "gen_dur") for r in rows) if v is not None]
                if durs:
                    entry["gen_dur_avg"] = round(float(np.mean(durs)), 2)
                cos_sims = [v for v in (_to_float(r, "mel_cos_sim") for r in rows) if v is not None]
                if cos_sims:
                    entry["mel_cos_sim_avg"] = round(float(np.mean(cos_sims)), 4)
                    entry["mel_cos_sim_min"] = round(float(np.min(cos_sims)), 4)
                wers = [v for v in (_to_float(r, "wer") for r in rows) if v is not None and v >= 0]
                if wers:
                    wer_key = "cer_avg" if lang == "zh" else "wer_avg"
                    entry[wer_key] = round(float(np.mean(wers)), 4)
                    entry[f"{wer_key.replace('avg', 'min')}"] = round(float(np.min(wers)), 4)
                summary[f"{model_name}/{lang}"] = entry

        summary_path = merge_out_dir / "eval_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"[multi-gpu] 汇总结果: {summary_path}")
    else:
        print("[multi-gpu] 未找到可合并的分片结果。")


# ============================================================
# 主流程
# ============================================================

def main():
    args = _parse_args()
    # 仅用于子进程分片，用户侧不暴露。
    internal_num_shards = int(os.environ.get("F5TTS_EVAL_NUM_SHARDS", "1"))
    internal_shard_id = int(os.environ.get("F5TTS_EVAL_SHARD_ID", "0"))

    if args.gpus < 1:
        raise ValueError(f"--gpus 必须 >= 1，当前: {args.gpus}")
    if args.gpus > 1:
        _launch_multi_gpu_workers(args)
        return

    if internal_num_shards < 1:
        raise ValueError(f"internal_num_shards 必须 >= 1，当前: {internal_num_shards}")
    if not (0 <= internal_shard_id < internal_num_shards):
        raise ValueError(
            f"internal_shard_id 越界: {internal_shard_id}, "
            f"需满足 0 <= internal_shard_id < internal_num_shards({internal_num_shards})"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = args.device
    vocab_map, _ = load_vocab(args.vocab)
    existing_rows = {}
    existing_csv = out_dir / "eval_results.csv"
    if existing_csv.exists():
        try:
            with open(existing_csv, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row.get("model"), row.get("lang"), row.get("utt"))
                    if all(key):
                        existing_rows[key] = row
            print(f"[resume] 发现历史结果: {existing_csv} ({len(existing_rows)} 条)")
        except Exception as e:
            print(f"[resume] 读取历史结果失败，继续全量执行: {e}")

    # ----------------------------------------------------------
    # 1. 加载数据
    # ----------------------------------------------------------
    n_total = args.n_samples * 2  # 前半做 ref, 后半提供 gen_text
    print(f"[1/5] 加载数据 (EN + ZH 各 {args.n_samples} 条)...")

    en_all = _load_samples(args.en_data, n_total, args.min_sec, args.max_sec)
    zh_all = _load_samples(args.zh_data, n_total, args.min_sec, args.max_sec)

    en_refs, en_gens = en_all[:args.n_samples], en_all[args.n_samples:]
    zh_refs, zh_gens = zh_all[:args.n_samples], zh_all[args.n_samples:]
    if len(en_gens) < args.n_samples:
        en_gens = en_refs
    if len(zh_gens) < args.n_samples:
        zh_gens = zh_refs

    tasks = []
    for i in range(args.n_samples):
        tasks.append(("en", en_refs[i], en_gens[i]["text"]))
    for i in range(args.n_samples):
        tasks.append(("zh", zh_refs[i], zh_gens[i]["text"]))

    total_tasks = len(tasks)
    sharded_tasks = [task for idx, task in enumerate(tasks) if idx % internal_num_shards == internal_shard_id]
    if not sharded_tasks:
        print(
            f"[warn] 当前分片无任务: shard_id={internal_shard_id}, "
            f"num_shards={internal_num_shards}, total_tasks={total_tasks}"
        )
    tasks = sharded_tasks

    # 持久化 ref 音频
    ref_dir = out_dir / "refs"
    ref_dir.mkdir(parents=True, exist_ok=True)
    for _, ref, _ in tasks:
        wav_path = ref_dir / f"{ref['utt']}.wav"
        if not wav_path.exists():
            wav_path.write_bytes(ref["audio_bytes"])
        ref["wav_path"] = str(wav_path)

    print(
        f"  EN: {len(en_refs)} refs, ZH: {len(zh_refs)} refs, "
        f"总任务: {total_tasks}, 分片后任务: {len(tasks)} "
        f"(shard {internal_shard_id}/{internal_num_shards})"
    )

    # ----------------------------------------------------------
    # 2. 准备模型
    # ----------------------------------------------------------
    print("[2/5] 加载模型...")

    print("  加载 F5TTS 原生 API (baseline)...")
    f5tts_api = _load_f5tts_api(args.pytorch_ckpt, args.vocab, device)
    print("  加载 HMONNX...")
    models = {"hmonnx": _make_hmonnx_fn(args.hmonnx, device)}

    all_names = ["pytorch", "hmonnx"]
    whisper_model = None
    if not args.skip_wer:
        try:
            import whisper
            print(f"  加载 Whisper ({args.whisper_model})...")
            whisper_model = whisper.load_model(args.whisper_model)
        except ImportError:
            print("  [warn] whisper 未安装，跳过 WER 计算")

    vocoder = load_vocos_vocoder(device)

    # ----------------------------------------------------------
    # 3. 生成 + 评估
    # ----------------------------------------------------------
    results = []

    print(f"[3/5] 生成 + 评估 ({len(tasks)} 条 x {len(all_names)} 模型)...")
    for lang, ref, gen_text in tqdm(tasks, desc="评估进度"):
        utt = ref["utt"]
        ref_text = ref["text"]

        # -- PyTorch baseline (F5TTS 官方 API) --
        wav_dir = out_dir / "pytorch" / lang
        wav_dir.mkdir(parents=True, exist_ok=True)
        wav_path = wav_dir / f"{utt}.wav"
        cache_key = ("pytorch", lang, utt)
        cached_row = existing_rows.get(cache_key)
        if cached_row is not None and wav_path.exists():
            cached = dict(cached_row)
            if "wer" in cached or args.skip_wer:
                results.append(cached)
                continue

        row = {
            "utt": utt, "lang": lang, "model": "pytorch",
            "ref_dur": ref["dur"], "gen_text": gen_text[:80],
        }

        if wav_path.exists():
            info = sf.info(str(wav_path))
            row["gen_dur"] = round(info.duration, 2)
            row["gen_time"] = 0.0
        else:
            t0 = time.time()
            wav_out, sr_out, _ = f5tts_api.infer(
                ref_file=ref["wav_path"],
                ref_text=ref_text,
                gen_text=gen_text,
                seed=args.seed,
                nfe_step=args.nfe_steps,
                cfg_strength=args.cfg_strength,
            )
            gen_time = time.time() - t0
            sf.write(str(wav_path), wav_out, sr_out)
            row["gen_dur"] = round(len(wav_out) / sr_out, 2)
            row["gen_time"] = round(gen_time, 1)

        if whisper_model is not None:
            try:
                hyp = _transcribe_whisper(str(wav_path), whisper_model)
                row["wer"] = round(_compute_wer(hyp, gen_text, lang), 4)
                row["asr_text"] = hyp[:80]
            except Exception as e:
                row["wer"] = -1.0
                row["asr_text"] = f"ASR_ERROR: {e}"

        results.append(row)

        # -- HMONNX 模型 (ODE 采样管线) --
        cond_mel, N_ref = _prepare_ref(ref["wav_path"])

        for model_name, model_fn in models.items():
            wav_dir = out_dir / model_name / lang
            wav_dir.mkdir(parents=True, exist_ok=True)
            wav_path = wav_dir / f"{utt}.wav"
            cache_key = (model_name, lang, utt)
            cached_row = existing_rows.get(cache_key)
            if cached_row is not None and wav_path.exists():
                cached = dict(cached_row)
                if "wer" in cached or args.skip_wer:
                    results.append(cached)
                    continue

            row = {
                "utt": utt, "lang": lang, "model": model_name,
                "ref_dur": ref["dur"], "gen_text": gen_text[:80],
            }

            mel = None
            if wav_path.exists():
                info = sf.info(str(wav_path))
                row["gen_dur"] = round(info.duration, 2)
                row["gen_time"] = 0.0
            else:
                t0 = time.time()
                mel = _synthesize(
                    model_fn, cond_mel, N_ref, ref_text, gen_text,
                    vocab_map, args.seed, device,
                    args.nfe_steps, args.cfg_strength,
                )
                gen_time = time.time() - t0

                gen_mel = mel[:, N_ref:, :].permute(0, 2, 1)
                wav = vocoder.decode(gen_mel)
                wav_np = wav.squeeze().cpu().numpy()
                sf.write(str(wav_path), wav_np, TARGET_SR)

                row["gen_dur"] = round(len(wav_np) / TARGET_SR, 2)
                row["gen_time"] = round(gen_time, 1)

            # Mel cos_sim vs pytorch baseline
            pytorch_wav = out_dir / "pytorch" / lang / f"{utt}.wav"
            if pytorch_wav.exists() and mel is not None:
                base_audio = load_audio(str(pytorch_wav), preprocess=False)
                base_mel = extract_mel_spec(base_audio)
                row["mel_cos_sim"] = round(_mel_cos_sim(base_mel, mel.cpu()), 6)

            # WER
            if whisper_model is not None:
                try:
                    hyp = _transcribe_whisper(str(wav_path), whisper_model)
                    row["wer"] = round(_compute_wer(hyp, gen_text, lang), 4)
                    row["asr_text"] = hyp[:80]
                except Exception as e:
                    row["wer"] = -1.0
                    row["asr_text"] = f"ASR_ERROR: {e}"

            results.append(row)

    # ----------------------------------------------------------
    # 4. 汇总
    # ----------------------------------------------------------
    print("\n[4/5] 汇总结果...")

    csv_path = out_dir / "eval_results.csv"
    all_keys = sorted({k for r in results for k in r})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"  详细结果: {csv_path}")

    summary = {}
    model_names = sorted({r["model"] for r in results if "model" in r})
    for model_name in model_names:
        for lang in ["en", "zh"]:
            key = f"{model_name}/{lang}"
            rows = [r for r in results if r["model"] == model_name and r["lang"] == lang]
            if not rows:
                continue
            entry = {"n": len(rows)}
            durs = [r["gen_dur"] for r in rows if "gen_dur" in r]
            if durs:
                entry["gen_dur_avg"] = round(float(np.mean(durs)), 2)
            cos_sims = [r["mel_cos_sim"] for r in rows if "mel_cos_sim" in r]
            if cos_sims:
                entry["mel_cos_sim_avg"] = round(float(np.mean(cos_sims)), 4)
                entry["mel_cos_sim_min"] = round(float(np.min(cos_sims)), 4)
            wers = [r["wer"] for r in rows if r.get("wer", -1) >= 0]
            if wers:
                wer_key = "cer_avg" if lang == "zh" else "wer_avg"
                entry[wer_key] = round(float(np.mean(wers)), 4)
                entry[f"{wer_key.replace('avg', 'min')}"] = round(float(np.min(wers)), 4)
            summary[key] = entry

    summary_path = out_dir / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"  汇总结果: {summary_path}")

    # 打印汇总表
    print("\n" + "=" * 70)
    print(f"{'模型/语言':<16} {'N':>3} {'平均时长':>8} {'Mel CosSim':>10} {'WER/CER':>10}")
    print("-" * 70)
    for key, entry in summary.items():
        cos = entry.get("mel_cos_sim_avg", "-")
        wer_key = "cer_avg" if "zh" in key else "wer_avg"
        wer = entry.get(wer_key, "-")
        dur = f"{entry['gen_dur_avg']:.1f}s" if "gen_dur_avg" in entry else "-"
        print(f"{key:<16} {entry['n']:>3} {dur:>8} "
              f"{str(cos):>10} {str(wer):>10}")
    print("=" * 70)

    print(f"\n评估完成! 音频文件在 {out_dir}/<model>/<lang>/")


if __name__ == "__main__":
    main()
