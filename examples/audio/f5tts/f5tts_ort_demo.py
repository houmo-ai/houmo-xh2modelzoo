"""
F5-TTS Float ONNX 推理 Demo (ORT)
=====================================

用 ONNX Runtime 加载 float ONNX 模型，跑与 hmonnx_demo 完全相同的
ODE 采样管线。用于隔离问题：推理管线 vs 量化。

用法:
    python f5tts_ort_demo.py \
        --onnx work_dirs/f5tts_v1/export_fp32/onnx/f5tts_dit.onnx \
        --ref-audio <ref.wav> \
        --ref-text "Some reference text." \
        --gen-text "Text to generate." \
        --output ort_output.wav
"""

import argparse
import re
import time as _time

import numpy as np
import torch

from f5tts_common import (
    HOP_LENGTH,
    TARGET_SR,
    VOCAB_PATH,
    extract_mel_spec,
    infer_chunked_wave_official,
    load_vocab,
    load_vocos_vocoder,
)


def _preprocess_ref_text(ref_text: str) -> str:
    """对齐官方 F5TTS API 的 ref_text 预处理：确保以句子终止符+空格结尾。"""
    if not ref_text.endswith(". ") and not ref_text.endswith("。"):
        if ref_text.endswith("."):
            ref_text += " "
        else:
            ref_text += ". "
    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text += " "
    return ref_text


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--onnx", type=str, required=True, help="float ONNX 模型路径")
    p.add_argument("--vocab", type=str, default=VOCAB_PATH)
    p.add_argument("--ref-audio", type=str, required=True)
    p.add_argument("--ref-text", type=str, required=True)
    p.add_argument("--gen-text", type=str, required=True)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--nfe-steps", type=int, default=32)
    p.add_argument("--cfg-strength", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--target-rms", type=float, default=0.1)
    p.add_argument(
        "--no-preprocess-ref-audio",
        action="store_true",
        default=False,
        help="关闭参考音频预处理（不做静音裁剪/12s 限制），避免参考时长被压缩",
    )
    p.add_argument("--cross-fade-duration", type=float, default=0.15)
    p.add_argument("--segment-max-chars", type=int, default=100, help="外层文本分段最大字符数（对齐 00_infer_cli.py）")
    p.add_argument("--output", type=str, default="ort_output.wav")
    return p.parse_args()


def _split_text_like_official(text: str, max_chars: int = 80):
    """对齐 00_infer_cli.py 的文本分段逻辑。"""
    chunks = []
    current_chunk = ""
    is_english = bool(re.search(r"[a-zA-Z]", text)) and not bool(re.search(r"[\u4e00-\u9fff]", text))

    if is_english:
        sentences = re.split(r"([.!?])", text)
        if len(sentences) == 1:
            sentences = [text, ""]
        for i in range(0, len(sentences) - 1, 2):
            sentence = sentences[i].strip()
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else ""
            if not sentence:
                continue
            if len(current_chunk + sentence + punctuation) <= max_chars:
                current_chunk += sentence + punctuation + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                remaining_text = sentence
                while len(remaining_text) > 0:
                    if len(remaining_text) <= max_chars:
                        current_chunk = remaining_text + punctuation + " "
                        remaining_text = ""
                    else:
                        split_pos = max_chars
                        for j in range(max_chars - 1, max(0, max_chars - 20), -1):
                            if remaining_text[j] in [" ", ","]:
                                split_pos = j + 1
                                break
                        if split_pos < max_chars:
                            part = remaining_text[:split_pos]
                            chunks.append(part.strip())
                            remaining_text = remaining_text[split_pos:]
                        else:
                            part = remaining_text[:max_chars]
                            last_space = part.rfind(" ")
                            if last_space > 0:
                                part = part[:last_space]
                                remaining_text = remaining_text[last_space + 1:]
                            else:
                                remaining_text = remaining_text[max_chars:]
                            chunks.append(part.strip())
    else:
        segments = re.split(r"([。；，、。,；！？!?:])", text)
        for i in range(0, len(segments) - 1, 2):
            segment = segments[i].strip()
            punctuation = segments[i + 1] if i + 1 < len(segments) else ""
            if not segment:
                continue
            if len(current_chunk + segment + punctuation) <= max_chars:
                current_chunk += segment + punctuation
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                remaining_text = segment
                while len(remaining_text) > 0:
                    if len(remaining_text) <= max_chars:
                        current_chunk = remaining_text + punctuation
                        remaining_text = ""
                    else:
                        split_pos = max_chars
                        for j in range(max_chars - 1, max(0, max_chars - 20), -1):
                            if remaining_text[j] in ["，", "、", "；", " ", ",", ";"]:
                                split_pos = j + 1
                                break
                        if split_pos < max_chars:
                            part = remaining_text[:split_pos]
                            chunks.append(part.strip())
                            remaining_text = remaining_text[split_pos:]
                        else:
                            part = remaining_text[:max_chars]
                            chunks.append(part.strip())
                            remaining_text = remaining_text[max_chars:]
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    # -- 1. 加载 float ONNX --
    print("[1/6] 加载 float ONNX 模型...")
    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])

    def ort_model_fn(x, cond, text, time, input_lengths=None):
        """与 hmonnx_demo 中 hmonnx_model_fn 相同的调用约定。"""
        if input_lengths is None:
            input_lengths = torch.tensor([x.shape[1]], device=x.device, dtype=torch.int32)
        out = sess.run(None, {
            "x": x.cpu().numpy().astype(np.float32),
            "cond": cond.cpu().numpy().astype(np.float32),
            "text": text.cpu().numpy().astype(np.int64),
            "time": time.cpu().numpy().astype(np.float32),
            "input_lengths": input_lengths.cpu().numpy().astype(np.int32),
        })[0]
        return torch.from_numpy(out).to(device)

    # -- 2. 预处理参考音频 --
    print("[2/6] 预处理参考音频...")
    from f5tts_common import load_audio
    audio, ref_rms = load_audio(
        args.ref_audio,
        target_rms=args.target_rms,
        preprocess=not args.no_preprocess_ref_audio,
        return_ref_rms=True,
    )
    cond_mel = extract_mel_spec(audio)
    ref_audio_len = audio.shape[-1] // HOP_LENGTH
    print(f"  ref mel: {cond_mel.shape}")

    # -- 3. 文本处理 --
    print("[3/6] 文本准备...")
    vocab_map, _ = load_vocab(args.vocab)
    ref_text = _preprocess_ref_text(args.ref_text)
    print(f"  ref_text_len(bytes): {len(ref_text.encode('utf-8'))}")
    print(f"  gen_text_len(bytes): {len(args.gen_text.encode('utf-8'))}")

    # -- 4. 加载 vocoder --
    print("[4/6] 加载 vocoder...")
    vocoder = load_vocos_vocoder(str(device))

    # -- 5. 官方 chunk + cross-fade 推理 --
    print(f"[5/6] 官方 chunk + cross-fade 推理 ({args.nfe_steps} 步, cfg={args.cfg_strength})...")
    t0 = _time.time()
    text_segments = _split_text_like_official(args.gen_text, max_chars=args.segment_max_chars)
    print(f"  外层文本分段: {len(text_segments)} (segment_max_chars={args.segment_max_chars})")
    all_waves = []
    all_chunk_metas = []
    all_max_chars = []
    for idx, segment in enumerate(text_segments):
        print(f"  segment[{idx}]: {segment}")
        wav_seg, chunk_metas, _, max_chars = infer_chunked_wave_official(
            model_fn=ort_model_fn,
            vocoder=vocoder,
            cond_mel=cond_mel,
            ref_audio_len=ref_audio_len,
            ref_audio_samples=audio.shape[-1],
            ref_text=ref_text,
            gen_text=segment,
            vocab_map=vocab_map,
            nfe_steps=args.nfe_steps,
            cfg_strength=args.cfg_strength,
            seed=args.seed,
            device=str(device),
            target_rms=args.target_rms,
            ref_rms=ref_rms,
            speed=args.speed,
            cross_fade_duration=args.cross_fade_duration,
        )
        all_waves.append(wav_seg)
        all_chunk_metas.append(chunk_metas)
        all_max_chars.append(max_chars)
    wav_np = np.concatenate(all_waves, axis=0) if all_waves else np.zeros(0, dtype=np.float32)
    elapsed = _time.time() - t0
    print(f"  max_chars={all_max_chars}, segments={len(text_segments)}")
    if all_chunk_metas:
        print(f"  first_segment_chunk[0]: {all_chunk_metas[0][0] if all_chunk_metas[0] else None}")
        print(f"  last_segment_chunk[-1]: {all_chunk_metas[-1][-1] if all_chunk_metas[-1] else None}")
    print(f"  总耗时: {elapsed:.1f}s")

    # -- 6. 保存音频 --
    print("[6/6] 保存音频...")

    import soundfile as sf
    sf.write(args.output, wav_np, TARGET_SR)
    print(f"\n  输出: {args.output} ({len(wav_np) / TARGET_SR:.1f}s)")


if __name__ == "__main__":
    main()
