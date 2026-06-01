#!/usr/bin/env python3
"""
HMSW-3885 最终对比脚本：sherpa-onnx / ORT FP32 / HMONNX w16a16 / HMONNX w8a8
已修正全部根因：
  1. chunk_shift=32 (decode_chunk_len) 替代错误的 shift=45
  2. high_freq=-400 匹配 sherpa-onnx 特征提取
  3. tail_pad=45 帧尾部静音填充
  4. BPE byte fallback 解码
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import kaldi_native_fbank as knf
from scipy.signal import resample_poly


# ──────────────────────── 音频 / Fbank ────────────────────────
def load_audio_mono_16k(wav_path, target_sr=16000):
    waveform, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sr != target_sr:
        gcd = np.gcd(sr, target_sr)
        waveform = resample_poly(waveform, up=target_sr // gcd, down=sr // gcd).astype(np.float32)
    return waveform.astype(np.float32), target_sr


def extract_fbank(waveform, sample_rate=16000, num_mel_bins=80):
    """匹配 sherpa-onnx 的 fbank 参数: dither=0, snip_edges=False, high_freq=-400"""
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = num_mel_bins
    opts.mel_opts.low_freq = 20
    opts.mel_opts.high_freq = -400  # sherpa-onnx default: Nyquist - 400
    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sample_rate, waveform.tolist())
    fbank.input_finished()
    return np.array([fbank.get_frame(i) for i in range(fbank.num_frames_ready)], dtype=np.float32)


# ──────────────────────── Tokens & BPE 解码 ────────────────────────
def load_tokens(tokens_path):
    tok = {}
    with open(tokens_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                tok[int(parts[1])] = parts[0]
    return tok


def decode_bpe_byte_fallback(token_ids, token_map):
    """BPE byte fallback 解码: 连续 <0xHH> 批量 UTF-8 解码"""
    pieces = [token_map.get(tid, "") for tid in token_ids]
    raw = "".join(pieces).replace("\u2581", " ")
    result, byte_buf = [], []
    i = 0
    while i < len(raw):
        m = re.match(r"<0x([0-9A-Fa-f]{2})>", raw[i:])
        if m:
            byte_buf.append(int(m.group(1), 16))
            i += len(m.group(0))
        else:
            if byte_buf:
                try:
                    result.append(bytes(byte_buf).decode("utf-8"))
                except UnicodeDecodeError:
                    result.append(bytes(byte_buf).decode("utf-8", "replace"))
                byte_buf = []
            result.append(raw[i])
            i += 1
    if byte_buf:
        try:
            result.append(bytes(byte_buf).decode("utf-8"))
        except UnicodeDecodeError:
            result.append(bytes(byte_buf).decode("utf-8", "replace"))
    return "".join(result).strip()


# ──────────────────────── CTC 解码 ────────────────────────
def ctc_greedy_decode(log_probs, blank_id=0):
    ids = np.argmax(log_probs, axis=-1)
    result, prev = [], -1
    for idx in ids:
        if idx != blank_id and idx != prev:
            result.append(int(idx))
        prev = idx
    return result


# ──────────────────────── 流式推理 (通用) ────────────────────────
CHUNK_LENGTH = 45
CHUNK_SHIFT = 32
TAIL_PAD_FRAMES = 45


def streaming_infer_onnx(sess, fbank_feat):
    """ORT FP32 流式推理，正确参数"""
    input_info = sess.get_inputs()
    output_info = sess.get_outputs()
    output_names = [o.name for o in output_info]

    states = {}
    for inp in input_info:
        if inp.name == "x":
            continue
        shape = [1 if isinstance(d, str) else d for d in inp.shape]
        dtype = np.int64 if inp.type == "tensor(int64)" else np.float32
        states[inp.name] = np.zeros(shape, dtype=dtype)

    out_to_in = {}
    for oname in output_names:
        if oname == "log_probs":
            continue
        in_name = oname[4:] if oname.startswith("new_") else oname
        if in_name in states:
            out_to_in[oname] = in_name

    # tail padding
    pad = np.zeros((TAIL_PAD_FRAMES, 80), dtype=np.float32)
    fbank_padded = np.concatenate([fbank_feat, pad], axis=0)

    T = fbank_padded.shape[0]
    all_lp, offset = [], 0
    while offset + CHUNK_LENGTH <= T:
        chunk = fbank_padded[offset: offset + CHUNK_LENGTH]
        feed = {"x": chunk[np.newaxis, :, :]}
        feed.update(states)
        outputs = sess.run(output_names, feed)
        out_dict = dict(zip(output_names, outputs))
        lp = out_dict["log_probs"]
        all_lp.append(lp[0] if lp.ndim == 3 else lp)
        for oname, iname in out_to_in.items():
            states[iname] = out_dict[oname]
        offset += CHUNK_SHIFT
    if not all_lp:
        return np.zeros((0, 2000), dtype=np.float32)
    return np.concatenate(all_lp, axis=0)


def streaming_infer_hmonnx(model, fbank_feat, device="cuda"):
    """HMONNX 流式推理，正确参数"""
    input_names = model.get_input_names()
    output_names = model.get_output_names()

    states = {}
    for name in input_names:
        if name == "x":
            continue
        info = model.get_input(name)
        shape = list(info.shape)
        dtype = info.dtype
        if dtype in (torch.int64, torch.int32):
            states[name] = torch.zeros(shape, dtype=torch.int32, device=device)
        else:
            states[name] = torch.zeros(shape, dtype=torch.float16, device=device)

    out_to_in = {}
    for oname in output_names:
        if oname == "log_probs":
            continue
        in_name = oname[4:] if oname.startswith("new_") else oname
        if in_name in states:
            out_to_in[oname] = in_name

    # tail padding
    pad = np.zeros((TAIL_PAD_FRAMES, 80), dtype=np.float32)
    fbank_padded = np.concatenate([fbank_feat, pad], axis=0)

    T = fbank_padded.shape[0]
    all_lp, offset = [], 0
    while offset + CHUNK_LENGTH <= T:
        chunk = fbank_padded[offset: offset + CHUNK_LENGTH]
        x = torch.from_numpy(chunk[np.newaxis, :, :]).half().to(device)
        feed = {"x": x}
        feed.update(states)
        out_list = model.run(feed)
        out_dict = dict(zip(output_names, out_list))
        lp = out_dict["log_probs"]
        if isinstance(lp, torch.Tensor):
            lp_np = lp.float().cpu().numpy()
            all_lp.append(lp_np[0] if lp_np.ndim == 3 else lp_np)
        else:
            all_lp.append(lp[0] if lp.ndim == 3 else lp)
        for oname, iname in out_to_in.items():
            states[iname] = out_dict[oname]
        offset += CHUNK_SHIFT
    if not all_lp:
        return np.zeros((0, 2000), dtype=np.float32)
    return np.concatenate(all_lp, axis=0)


# ──────────────────────── sherpa-onnx 基线 ────────────────────────
def sherpa_infer(model_dir, wav_path, onnx_name):
    import sherpa_onnx
    recognizer = sherpa_onnx.OnlineRecognizer.from_zipformer2_ctc(
        tokens=str(Path(model_dir) / "tokens.txt"),
        model=str(Path(model_dir) / onnx_name),
        num_threads=4,
    )
    stream = recognizer.create_stream()
    audio, sr = sf.read(wav_path, dtype="float32")
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sr != 16000:
        gcd = np.gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // gcd, sr // gcd).astype(np.float32)
    stream.accept_waveform(16000, audio)
    stream.accept_waveform(16000, np.zeros(int(0.3 * 16000), dtype=np.float32))
    stream.input_finished()
    while recognizer.is_ready(stream):
        recognizer.decode_stream(stream)
    return recognizer.get_result(stream)


# ──────────────────────── CER ────────────────────────
def edit_distance(ref, hyp):
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = (
                dp[i - 1][j - 1]
                if ref[i - 1] == hyp[j - 1]
                else 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
            )
    return dp[n][m]


def compute_cer(ref_text, hyp_text):
    rc = list(ref_text.replace(" ", ""))
    hc = list(hyp_text.replace(" ", ""))
    if not rc:
        return (0.0, 0, 0)
    d = edit_distance(rc, hc)
    return (d / len(rc), d, len(rc))


def cosine_sim(a, b):
    minlen = min(a.shape[0], b.shape[0])
    a_f = a[:minlen].flatten().astype(np.float64)
    b_f = b[:minlen].flatten().astype(np.float64)
    na, nb = np.linalg.norm(a_f), np.linalg.norm(b_f)
    return float(np.dot(a_f, b_f) / (na * nb)) if na > 0 and nb > 0 else 0.0


# ──────────────────────── Main ────────────────────────
def main():
    parser = argparse.ArgumentParser(description="HMSW-3885: 最终全方案精度对比 (修正版)")
    parser.add_argument("--model-dir", type=str,
                        default="./models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13")
    parser.add_argument("--hmonnx-w16a16", type=str,
                        default="./work_dirs/w16a16/ctc-epoch-20-avg-1-chunk-16-left-128_sim_XH2a.onnx")
    parser.add_argument("--hmonnx-w8a8", type=str,
                        default="./work_dirs/w8a8/ctc-epoch-20-avg-1-chunk-16-left-128_sim_XH2a.onnx")
    parser.add_argument("--data-dir", type=str, default="./data/cn")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--report", type=str, default="./work_dirs/final_report_v2.json")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    onnx_path = str(model_dir / "ctc-epoch-20-avg-1-chunk-16-left-128_sim.onnx")
    tokens_path = str(model_dir / "tokens.txt")
    tokens = load_tokens(tokens_path)

    refs = {
        "0.wav": "对我做了介绍那么我想说的是大家如果对我的研究感兴趣",
        "1.wav": "重点想谈三个问题首先就是这一轮全球金融动荡的表现",
        "8k.wav": "深度的分析这一次全球金融动荡背后的根源",
    }

    # 加载模型
    onnx_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    from xhquant.api import HMONNXInference
    hm_w16 = HMONNXInference(args.hmonnx_w16a16)
    hm_w16.to(args.device)
    hm_w8 = HMONNXInference(args.hmonnx_w8a8)
    hm_w8.to(args.device)

    print("=" * 100)
    print("HMSW-3885: Zipformer CTC 流式 ASR — 最终修正版全方案对比")
    print(f"  chunk_length={CHUNK_LENGTH}, chunk_shift={CHUNK_SHIFT}, tail_pad={TAIL_PAD_FRAMES}")
    print(f"  fbank: dither=0, snip_edges=False, high_freq=-400, num_bins=80")
    print("=" * 100)

    agg = {k: {"dist": 0, "chars": 0} for k in ["sherpa", "ort_fp32", "w16a16", "w8a8"]}
    results = []

    for wav_name, ref_text in refs.items():
        wav_path = str(data_dir / wav_name)
        waveform, sr = load_audio_mono_16k(wav_path)
        fbank_feat = extract_fbank(waveform, sample_rate=sr)

        # sherpa-onnx
        sherpa_text = sherpa_infer(str(model_dir), wav_path, "ctc-epoch-20-avg-1-chunk-16-left-128_sim.onnx")

        # ORT FP32
        ort_lp = streaming_infer_onnx(onnx_sess, fbank_feat)
        ort_text = decode_bpe_byte_fallback(ctc_greedy_decode(ort_lp), tokens)

        # HMONNX w16a16
        w16_lp = streaming_infer_hmonnx(hm_w16, fbank_feat, args.device)
        w16_text = decode_bpe_byte_fallback(ctc_greedy_decode(w16_lp), tokens)

        # HMONNX w8a8
        w8_lp = streaming_infer_hmonnx(hm_w8, fbank_feat, args.device)
        w8_text = decode_bpe_byte_fallback(ctc_greedy_decode(w8_lp), tokens)

        # logprobs similarity
        cos_w16 = cosine_sim(ort_lp, w16_lp)
        cos_w8 = cosine_sim(ort_lp, w8_lp)

        # CER
        ref_chars = list(ref_text.replace(" ", ""))
        n = len(ref_chars)

        rec = {"wav": wav_name, "ref": ref_text}
        print(f"\n{'─' * 100}")
        print(f" {wav_name}  (ref: {ref_text})")
        print(f"{'─' * 100}")

        for label, hyp in [("sherpa", sherpa_text), ("ort_fp32", ort_text),
                            ("w16a16", w16_text), ("w8a8", w8_text)]:
            d = edit_distance(ref_chars, list(hyp.replace(" ", "")))
            cer = d / n if n else 0
            agg[label]["dist"] += d
            agg[label]["chars"] += n
            rec[label] = {"text": hyp, "cer": round(cer, 4), "dist": d}
            print(f"  {label:10s}: {hyp:50s}  CER={cer:.2%}")

        rec["cosine_w16"] = round(cos_w16, 6)
        rec["cosine_w8"] = round(cos_w8, 6)
        print(f"  logprobs cosine: w16a16={cos_w16:.6f}, w8a8={cos_w8:.6f}")
        results.append(rec)

    # Summary
    print(f"\n{'=' * 100}")
    print("Corpus CER 汇总")
    print(f"{'=' * 100}")
    summary = {}
    for label in ["sherpa", "ort_fp32", "w16a16", "w8a8"]:
        d, c = agg[label]["dist"], agg[label]["chars"]
        cer = d / c if c else 0
        summary[label] = {"corpus_cer": round(cer, 4), "total_dist": d, "total_chars": c}
        print(f"  {label:10s}: Corpus CER = {cer:.2%}  ({d}/{c})")

    delta_w16 = summary["w16a16"]["corpus_cer"] - summary["ort_fp32"]["corpus_cer"]
    delta_w8 = summary["w8a8"]["corpus_cer"] - summary["ort_fp32"]["corpus_cer"]
    print(f"\n  Δ CER (w16a16 vs ORT FP32): {delta_w16:+.2%}")
    print(f"  Δ CER (w8a8 vs ORT FP32):   {delta_w8:+.2%}")

    report = {"summary": summary, "details": results,
              "config": {"chunk_length": CHUNK_LENGTH, "chunk_shift": CHUNK_SHIFT,
                         "tail_pad": TAIL_PAD_FRAMES, "high_freq": -400}}
    report_path = args.report
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已保存到: {report_path}")


if __name__ == "__main__":
    main()
