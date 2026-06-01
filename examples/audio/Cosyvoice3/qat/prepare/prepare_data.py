"""
CosyVoice3 QAT 训练数据准备
============================

统一入口，支持 4 种数据集: librispeech / thchs30 / zero_shot / libritts。

所有数据集统一输出 parquet 格式:
    utt, text, audio_data (WAV @24kHz),
    spk_embedding (192-dim float32), speech_token (int32)

两种运行模式:
  1. 编排模式（默认）: 扫描/解压数据 + 启动多 GPU worker
  2. Worker 模式 (--worker): 处理分配的样本列表，增量保存 parquet

用法:
    # LibriSpeech (本地 tar.gz)
    python prepare_data.py --dataset librispeech \\
        --train_tgz ./LibriSpeech/train-clean-100.tar.gz \\
        --dev_tgz ./LibriSpeech/dev-clean.tar.gz \\
        --output_dir ./data_librispeech --gpu_ids 0,1,2,3

    # THCHS-30 中文 (本地 tgz)
    python prepare_data.py --dataset thchs30 \\
        --tgz_path ./data_thchs30.tgz \\
        --output_dir ./data_thchs30 --gpu_ids 0,1

    # CV3-Eval 多语言 (Kaldi 格式目录)
    python prepare_data.py --dataset zero_shot \\
        --data_root /path/to/CV3-Eval/data/zero_shot \\
        --cv3_eval_root /path/to/CV3-Eval \\
        --output_dir ./data_zero_shot --gpu_ids 2,3

    # LibriTTS (HuggingFace 下载，单进程)
    python prepare_data.py --dataset libritts \\
        --output_dir ./data_libritts --max_train 5000
"""

import argparse
import glob
import io
import os
import pickle
import subprocess
import sys
import tarfile
import time

import numpy as np
import onnxruntime as ort
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper
from tqdm import tqdm

# ================================================================
#  全局配置
# ================================================================

INSTRUCT = "You are a helpful assistant.<|endofprompt|>"
SAMPLE_RATE = 24000
FEAT_SAMPLE_RATE = 16000


# ================================================================
#  共享特征提取器
# ================================================================

def create_embedding_extractor(onnx_path):
    """CAMPPlus 说话人嵌入提取器（CPU）。"""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def extract(waveform_16k):
        feat = kaldi.fbank(
            waveform_16k.unsqueeze(0), num_mel_bins=80,
            dither=0, sample_frequency=FEAT_SAMPLE_RATE,
        )
        feat = feat - feat.mean(dim=0, keepdim=True)
        inp = np.ascontiguousarray(feat.unsqueeze(0).numpy(), dtype=np.float32)
        return session.run(None, {input_name: inp})[0].flatten().astype(np.float32)

    return extract


def create_speech_token_extractor(onnx_path):
    """Speech token 提取器（优先 GPU）。"""
    session = ort.InferenceSession(
        onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    feat_input = session.get_inputs()[0].name
    len_input = session.get_inputs()[1].name

    def extract(waveform_16k):
        feat = whisper.log_mel_spectrogram(waveform_16k, n_mels=128)
        feat_np = np.ascontiguousarray(feat.unsqueeze(0).numpy(), dtype=np.float32)
        length_np = np.array([feat_np.shape[2]], dtype=np.int32)
        return session.run(
            None, {feat_input: feat_np, len_input: length_np}
        )[0].flatten().astype(np.int32)

    return extract


# ================================================================
#  共享音频工具
# ================================================================

def encode_wav_bytes(waveform, sr):
    buf = io.BytesIO()
    torchaudio.save(buf, waveform.unsqueeze(0), sr, format="wav")
    return buf.getvalue()


def resample(waveform, orig_sr, target_sr):
    if orig_sr == target_sr:
        return waveform
    return torchaudio.transforms.Resample(orig_sr, target_sr)(waveform)


def process_one_audio(wav_path, text, extract_emb, extract_tok):
    """统一音频处理: 加载 → 重采样 → 特征提取 → parquet 行。

    Parameters
    ----------
    wav_path : str 或 None (None 时跳过)
    text : str (已含 INSTRUCT 前缀)

    Returns dict 或 None
    """
    if wav_path is None or not os.path.exists(wav_path):
        return None
    try:
        waveform, sr = torchaudio.load(wav_path)
        waveform = waveform.squeeze(0)
        if waveform.dim() == 0:
            return None
        duration = len(waveform) / sr
        if duration < 1.0 or duration > 30.0:
            return None

        waveform_16k = resample(waveform, sr, FEAT_SAMPLE_RATE)
        waveform_24k = resample(waveform, sr, SAMPLE_RATE)

        embedding = extract_emb(waveform_16k)
        speech_token = extract_tok(waveform_16k)
        audio_bytes = encode_wav_bytes(waveform_24k, SAMPLE_RATE)

        return {
            "utt": os.path.splitext(os.path.basename(wav_path))[0],
            "text": text,
            "audio_data": audio_bytes,
            "spk_embedding": embedding.tobytes(),
            "speech_token": speech_token.tobytes(),
        }
    except Exception:
        return None


# ================================================================
#  共享 Parquet 写入 + 编排框架
# ================================================================

def save_chunk(buffer, output_dir, prefix, chunk_idx):
    if not buffer:
        return None
    os.makedirs(output_dir, exist_ok=True)
    table = pa.table({
        "utt":           [s["utt"] for s in buffer],
        "text":          [s["text"] for s in buffer],
        "audio_data":    [s["audio_data"] for s in buffer],
        "spk_embedding": [s["spk_embedding"] for s in buffer],
        "speech_token":  [s["speech_token"] for s in buffer],
    })
    pq_path = os.path.join(output_dir, f"{prefix}_{chunk_idx:04d}.parquet")
    pq.write_table(table, pq_path)
    return pq_path


def split_list(lst, num_chunks):
    chunks = [[] for _ in range(num_chunks)]
    for i, item in enumerate(lst):
        chunks[i % num_chunks].append(item)
    return chunks


def merge_list_files(output_dir, prefix, absolute=False):
    parquets = sorted(glob.glob(os.path.join(output_dir, f"{prefix}_*.parquet")))
    suffix = "_abs" if absolute else ""
    list_path = os.path.join(output_dir, f"{prefix}{suffix}.list")
    with open(list_path, "w") as f:
        for p in parquets:
            f.write(f"{os.path.abspath(p)}\n" if absolute else f"{p}\n")
    print(f"  {list_path}: {len(parquets)} parquet files")
    return list_path


# ================================================================
#  数据集扫描器 — 各数据集返回统一格式 [(utt, text, wav_path), ...]
# ================================================================

# ---- LibriSpeech (本地 tar.gz) ----

def _read_librispeech_transcripts(tar_path):
    transcripts = {}
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith(".trans.txt"):
                f = tar.extractfile(member)
                if f is None:
                    continue
                for line in f.read().decode("utf-8").strip().split("\n"):
                    parts = line.strip().split(maxsplit=1)
                    if len(parts) == 2:
                        transcripts[parts[0]] = parts[1]
    return transcripts


def scan_librispeech(tar_path, tmp_dir, max_n=-1):
    """解压 tar.gz → 返回 [(utt, text, wav_path)]。"""
    extract_dir = os.path.join(
        tmp_dir, os.path.splitext(os.path.basename(tar_path))[0])

    if not (os.path.exists(extract_dir) and os.listdir(extract_dir)):
        os.makedirs(extract_dir, exist_ok=True)
        print(f"  解压 {os.path.basename(tar_path)} ...")
        sys.stdout.flush()
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        print("  解压完成")
        sys.stdout.flush()

    # 缓存 transcripts
    trans_path = os.path.join(tmp_dir, "transcripts.pt")
    if not os.path.exists(trans_path):
        transcripts = _read_librispeech_transcripts(tar_path)
        with open(trans_path, "wb") as f:
            pickle.dump(transcripts, f)
    else:
        with open(trans_path, "rb") as f:
            transcripts = pickle.load(f)

    flac_files = sorted(glob.glob(os.path.join(extract_dir, "**/*.flac"),
                                   recursive=True))
    if 0 < max_n < len(flac_files):
        flac_files = flac_files[:max_n]

    samples = []
    for fp in flac_files:
        utt = os.path.splitext(os.path.basename(fp))[0]
        if utt in transcripts:
            samples.append((utt, INSTRUCT + transcripts[utt], fp))
    return samples


# ---- THCHS-30 (本地 tgz, WAV @16kHz) ----

def scan_thchs30(tgz_path, tmp_dir, split_name="train", max_n=-1):
    """解压 tgz → 读 .wav.trn → 返回 [(utt, text, wav_path)]。"""
    extract_dir = os.path.join(tmp_dir, "data_thchs30")
    if not (os.path.exists(extract_dir) and os.listdir(extract_dir)):
        os.makedirs(extract_dir, exist_ok=True)
        print(f"  解压 {os.path.basename(tgz_path)} ...")
        sys.stdout.flush()
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(extract_dir)
        print("  解压完成")
        sys.stdout.flush()

    # 缓存全部转录
    trans_path = os.path.join(tmp_dir, "all_transcripts.pt")
    if not os.path.exists(trans_path):
        data_dir = os.path.join(extract_dir, "data")
        transcripts = {}
        for trn_path in glob.glob(os.path.join(data_dir, "*.wav.trn")):
            utt = os.path.basename(trn_path).replace(".wav.trn", "")
            with open(trn_path, "r", encoding="utf-8") as f:
                lines = f.read().strip().split("\n")
            if lines and lines[0].strip():
                transcripts[utt] = lines[0].strip()
        with open(trans_path, "wb") as f:
            pickle.dump(transcripts, f)
    else:
        with open(trans_path, "rb") as f:
            transcripts = pickle.load(f)

    split_dir = os.path.join(extract_dir, split_name)
    if not os.path.isdir(split_dir):
        return []

    wav_files = [f for f in sorted(glob.glob(os.path.join(split_dir, "*.wav")))
                 if not f.endswith(".trn")]
    if 0 < max_n < len(wav_files):
        wav_files = wav_files[:max_n]

    samples = []
    for fp in wav_files:
        utt = os.path.splitext(os.path.basename(fp))[0]
        if utt in transcripts:
            samples.append((utt, INSTRUCT + transcripts[utt], fp))
    return samples


# ---- CV3-Eval zero-shot (Kaldi 格式) ----

def _read_kaldi_scp(filepath, base_dir):
    result = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = os.path.normpath(
                    os.path.join(base_dir, parts[1]))
    return result


def _read_kaldi_text(filepath):
    result = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                result[parts[0]] = parts[1]
    return result


def scan_zero_shot(data_root, cv3_eval_root, langs=None):
    """扫描 Kaldi 格式语言目录 → 返回 [(utt, text, wav_path)]。"""
    DEFAULT_LANGS = [
        "en", "zh", "de", "es", "fr", "it", "ja", "ko", "ru",
        "hard_en", "hard_zh",
    ]
    samples = []
    for lang in (langs or DEFAULT_LANGS):
        lang_dir = os.path.join(data_root, lang)
        if not os.path.isdir(lang_dir):
            continue
        pt_path = os.path.join(lang_dir, "prompt_text")
        scp_path = os.path.join(lang_dir, "prompt_wav.scp")
        if not os.path.exists(pt_path) or not os.path.exists(scp_path):
            continue
        texts = _read_kaldi_text(pt_path)
        wav_paths = _read_kaldi_scp(scp_path, cv3_eval_root)
        for utt_id, prompt_text in texts.items():
            if utt_id in wav_paths:
                samples.append((
                    f"{lang}_{utt_id}",
                    INSTRUCT + prompt_text,
                    wav_paths[utt_id],
                ))
        print(f"  {lang}: {sum(1 for s in samples if s[0].startswith(lang))} samples")
        sys.stdout.flush()
    return samples


# ---- LibriTTS (HuggingFace, 单进程) ----

def scan_libritts_hf(max_train=5000, max_dev=500):
    """从 HuggingFace 下载 → 返回 [(utt, text, waveform_24k_tensor)]。"""
    from datasets import load_dataset

    splits_config = [
        ("train", "clean", "train.clean.100", max_train),
        ("dev",   "clean", "dev.clean",       max_dev),
    ]
    all_samples = {}
    for split_name, config, hf_split, max_n in splits_config:
        ds = load_dataset("mythicinfinity/libritts", config,
                          split=hf_split, streaming=True)
        samples = []
        for idx, item in enumerate(tqdm(ds, desc=hf_split)):
            if 0 < max_n <= idx:
                break
            duration = len(item["audio"]["array"]) / item["audio"]["sampling_rate"]
            if duration < 1.0 or duration > 30.0:
                continue
            text = INSTRUCT + item["text_normalized"]
            waveform = torch.tensor(item["audio"]["array"], dtype=torch.float32)
            samples.append((item["id"], text, waveform))
        all_samples[split_name] = samples
    return all_samples


# ================================================================
#  Worker 模式: 处理 (utt, text, wav_path) 列表
# ================================================================

def run_worker(args):
    """Worker: 加载样本 pickle → 提取特征 → 增量保存 parquet。"""
    print(f"[GPU{args.worker_id}] 加载特征提取器 ...")
    sys.stdout.flush()

    extract_emb = create_embedding_extractor(
        os.path.join(args.model_dir, "campplus.onnx"))
    extract_tok = create_speech_token_extractor(
        os.path.join(args.model_dir, "speech_tokenizer_v3.onnx"))

    with open(args.sample_list, "rb") as f:
        samples = pickle.load(f)
    print(f"[GPU{args.worker_id}] {len(samples)} samples")
    sys.stdout.flush()

    buffer = []
    chunk_idx = 0
    total_ok = total_skip = 0
    start = time.time()

    for utt, text, wav_path in tqdm(samples, desc=f"GPU{args.worker_id}"):
        result = process_one_audio(wav_path, text, extract_emb, extract_tok)
        if result is not None:
            result["utt"] = utt  # 使用扫描器的 utt (可能含 lang_ 前缀)
            buffer.append(result)
            total_ok += 1
        else:
            total_skip += 1

        if len(buffer) >= args.num_utts_per_parquet:
            save_chunk(buffer, args.output_dir, args.prefix, chunk_idx)
            chunk_idx += 1
            elapsed = time.time() - start
            speed = total_ok / elapsed if elapsed > 0 else 0
            print(f"[GPU{args.worker_id}] saved chunk {chunk_idx} "
                  f"({total_ok} ok, {total_skip} skip, {speed:.1f} s/s)")
            sys.stdout.flush()
            buffer = []

    if buffer:
        save_chunk(buffer, args.output_dir, args.prefix, chunk_idx)
    elapsed = time.time() - start
    print(f"[GPU{args.worker_id}] done: {total_ok} ok, {total_skip} skip, "
          f"{elapsed:.0f}s, {total_ok/elapsed:.1f} s/s")
    sys.stdout.flush()


# ================================================================
#  编排模式: 分配 + 启动多 GPU worker
# ================================================================

def launch_workers(gpu_ids, samples, output_dir, prefix, model_dir, num_per_file):
    """将 samples 分配到多个 GPU worker。"""
    chunks = split_list(samples, len(gpu_ids))
    print(f"  {len(samples)} samples → {len(gpu_ids)} GPUs:")
    for gid, chunk in zip(gpu_ids, chunks):
        print(f"    GPU {gid}: {len(chunk)}")
    sys.stdout.flush()

    # 写分块 pickle
    chunk_paths = []
    for i, chunk in enumerate(chunks):
        path = f"/tmp/prepare_chunk_{prefix}_{i:03d}.pkl"
        with open(path, "wb") as f:
            pickle.dump(chunk, f)
        chunk_paths.append(path)

    # 启动 worker
    this_script = os.path.abspath(__file__)
    procs = []
    for i, (gid, cpath) in enumerate(zip(gpu_ids, chunk_paths)):
        cmd = [
            sys.executable, "-u", this_script,
            "--worker",
            "--dataset", os.environ.get("_PREPARE_DATASET", "unknown"),
            "--sample_list", cpath,
            "--output_dir", output_dir,
            "--prefix", f"{prefix}_w{i}",
            "--worker_id", str(gid),
            "--model_dir", model_dir,
            "--num_utts_per_parquet", str(num_per_file),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gid)
        env["PYTHONUNBUFFERED"] = "1"
        env["_PREPARE_DATASET"] = os.environ.get("_PREPARE_DATASET", "")

        proc = subprocess.Popen(cmd, env=env)
        procs.append(proc)

    for proc in procs:
        proc.wait()
    print("  all workers done")
    sys.stdout.flush()


# ================================================================
#  LibriTTS 单进程模式 (HuggingFace streaming, 无 worker)
# ================================================================

def run_libritts(args):
    """LibriTTS: HF 下载 → 单进程处理 → 保存 parquet。"""
    campplus_path = os.path.join(args.model_dir, "campplus.onnx")
    tokenizer_path = os.path.join(args.model_dir, "speech_tokenizer_v3.onnx")

    extract_emb = create_embedding_extractor(campplus_path)
    extract_tok = create_speech_token_extractor(tokenizer_path)

    all_splits = scan_libritts_hf(args.max_train, args.max_dev)

    for split_name, raw_samples in all_splits.items():
        print(f"\n  Processing {split_name}: {len(raw_samples)} samples ...")
        processed = []
        for utt, text, waveform in tqdm(raw_samples, desc=split_name):
            waveform_16k = resample(waveform, SAMPLE_RATE, FEAT_SAMPLE_RATE)
            try:
                emb = extract_emb(waveform_16k)
                tok = extract_tok(waveform_16k)
                audio_bytes = encode_wav_bytes(waveform, SAMPLE_RATE)
                processed.append({
                    "utt": utt, "text": text,
                    "audio_data": audio_bytes,
                    "spk_embedding": emb.tobytes(),
                    "speech_token": tok.tobytes(),
                })
            except Exception:
                continue

        # 保存
        out_dir = os.path.join(args.output_dir, split_name)
        for i in range(0, len(processed), args.num_utts_per_parquet):
            chunk = processed[i:i + args.num_utts_per_parquet]
            save_chunk(chunk, out_dir, split_name,
                       i // args.num_utts_per_parquet)
        merge_list_files(out_dir, split_name)

    print(f"\n  LibriTTS done: {args.output_dir}")


# ================================================================
#  CLI
# ================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="CosyVoice3 QAT 数据准备 (统一入口)")

    # 数据集选择
    p.add_argument("--dataset", required=True,
                   choices=["librispeech", "thchs30", "zero_shot", "libritts"],
                   help="数据集名称")

    # 通用参数
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_dir",
                   default="/data01/nfs_shared/ASR_TTS/CosyVoice3-0.5B-2512")
    p.add_argument("--num_utts_per_parquet", type=int, default=1000)
    p.add_argument("--gpu_ids", default="0,1,2,3",
                   help="GPU ID 列表 (逗号分隔)")

    # Worker 模式
    p.add_argument("--worker", action="store_true")
    p.add_argument("--sample_list", default="")
    p.add_argument("--prefix", default="train")
    p.add_argument("--worker_id", type=int, default=0)

    # LibriSpeech 专属
    p.add_argument("--train_tgz", default="")
    p.add_argument("--dev_tgz", default="")
    p.add_argument("--max_train", type=int, default=5000)
    p.add_argument("--max_dev", type=int, default=500)
    p.add_argument("--tmp_dir", default="/tmp/prepare_data")

    # THCHS-30 专属
    p.add_argument("--tgz_path", default="")
    p.add_argument("--splits", default="train,dev")

    # Zero-shot 专属
    p.add_argument("--data_root", default="")
    p.add_argument("--cv3_eval_root", default="")
    p.add_argument("--langs", default="")

    return p.parse_args()


def main():
    args = parse_args()

    # 传递 dataset 类型给 worker 子进程
    os.environ["_PREPARE_DATASET"] = args.dataset

    if args.worker:
        run_worker(args)
        return

    gpu_ids = [int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()]
    dataset = args.dataset
    output_dir = args.output_dir
    model_dir = args.model_dir

    print(f"\n{'=' * 60}")
    print(f"  CosyVoice3 数据准备 — {dataset.upper()}")
    print(f"  output_dir = {output_dir}")
    print(f"  GPUs       = {gpu_ids}")
    print(f"{'=' * 60}")

    # ---- LibriTTS: 单进程, 走特殊路径 ----
    if dataset == "libritts":
        run_libritts(args)
        return

    os.makedirs(output_dir, exist_ok=True)

    # ---- 多数据集扫描 → 统一格式 [(utt, text, wav_path)] ----
    all_splits = {}

    if dataset == "librispeech":
        for split_name, tar_path, max_n in [
            ("train", args.train_tgz, args.max_train),
            ("dev",   args.dev_tgz,   args.max_dev),
        ]:
            if not tar_path or not os.path.exists(tar_path):
                continue
            tmp = os.path.join(args.tmp_dir, f"{dataset}_{split_name}")
            samples = scan_librispeech(tar_path, tmp, max_n)
            all_splits[split_name] = samples
            print(f"  {split_name}: {len(samples)} samples")

    elif dataset == "thchs30":
        if not args.tgz_path:
            print("[ERROR] --tgz_path required")
            return
        tmp = os.path.join(args.tmp_dir, "thchs30")
        for split_name in args.splits.split(","):
            split_name = split_name.strip()
            samples = scan_thchs30(args.tgz_path, tmp, split_name, -1)
            all_splits[split_name] = samples
            print(f"  {split_name}: {len(samples)} samples")

    elif dataset == "zero_shot":
        if not args.data_root or not args.cv3_eval_root:
            print("[ERROR] --data_root and --cv3_eval_root required")
            return
        langs = [l.strip() for l in args.langs.split(",") if l.strip()] or None
        samples = scan_zero_shot(args.data_root, args.cv3_eval_root, langs)
        all_splits["train"] = samples
        print(f"  total: {len(samples)} samples")

    # ---- 启动 worker ----
    for split_name, samples in all_splits.items():
        if not samples:
            continue
        print(f"\n  {split_name}: launching {len(gpu_ids)} workers ...")
        sys.stdout.flush()
        launch_workers(gpu_ids, samples, output_dir, split_name,
                       model_dir, args.num_utts_per_parquet)
        merge_list_files(output_dir, split_name)

    print(f"\n{'=' * 60}")
    print(f"  Done: {output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
