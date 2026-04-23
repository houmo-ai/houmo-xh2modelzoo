"""
F5-TTS HMONNX 端到端推理 Demo
================================

加载量化后的 HMONNX 模型，运行完整 TTS 推理管线：
文本 → 拼音 token → ODE 采样(32步) → mel → vocoder → 音频

用法:
    python f5tts_hmonnx_demo.py \
        --hmonnx work_dirs/f5tts/export_xh2a/hmonnx/f5tts_dit_XH2a.onnx \
        --vocab /data01/nfs_shared/ASR_TTS/F5TTS_base/F5TTS_Base/vocab.txt \
        --ref-audio <ref.wav> \
        --ref-text "Some reference text." \
        --gen-text "Text to generate." \
        --seed 42 \
        --output output.wav
"""

import argparse

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
    p.add_argument("--hmonnx", type=str, required=True, help="HMONNX 模型路径")
    p.add_argument("--vocab", type=str, default=VOCAB_PATH)
    p.add_argument("--ref-audio", type=str, required=True, help="参考音频")
    p.add_argument("--ref-text", type=str, required=True, help="参考音频对应文本")
    p.add_argument("--gen-text", type=str, required=True, help="待生成文本")
    p.add_argument("--nfe-steps", type=int, default=32)
    p.add_argument("--cfg-strength", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--target-rms", type=float, default=0.1)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--cross-fade-duration", type=float, default=0.15)
    p.add_argument(
        "--force-sentence-chunks",
        action="store_true",
        default=False,
        help="按标点强制分句后逐句生成（默认关闭，保持官方分块逻辑）",
    )
    p.add_argument("--output", type=str, default="hmonnx_output.wav")
    p.add_argument("--save-mel", type=str, default=None, help="保存中间 mel 为 .npy")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device)

    # -- 1. 加载 HMONNX --
    print("[1/6] 加载 HMONNX 模型...")
    from xhquant.api import HMONNXGoldenInference
    session = HMONNXGoldenInference(args.hmonnx)
    session.exec_device = device

    def hmonnx_model_fn(x, cond, text, time, input_lengths=None):
        # HMONNX 期望: x/cond/time=float16, text/int长度为int32, time shape=(1,1)
        if input_lengths is None:
            input_lengths = torch.tensor([x.shape[1]], device=x.device, dtype=torch.int32)
        t_in = time.to(device).half()
        if t_in.dim() == 1:
            t_in = t_in.unsqueeze(1)
        out = session(x.to(device).half(), cond.to(device).half(),
                      text.to(device).int(), t_in, input_lengths.to(device).int())
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out.float()

    # -- 2. 预处理参考音频 --
    print("[2/6] 预处理参考音频...")
    from f5tts_common import load_audio
    audio, ref_rms = load_audio(
        args.ref_audio,
        target_rms=args.target_rms,
        return_ref_rms=True,
    )
    cond_mel = extract_mel_spec(audio)  # (1, N_ref, 100)
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
    import time as _time
    t0 = _time.time()
    wav_np, chunk_metas, merged_chunk_mel, max_chars = infer_chunked_wave_official(
        model_fn=hmonnx_model_fn,
        vocoder=vocoder,
        cond_mel=cond_mel,
        ref_audio_len=ref_audio_len,
        ref_audio_samples=audio.shape[-1],
        ref_text=ref_text,
        gen_text=args.gen_text,
        vocab_map=vocab_map,
        nfe_steps=args.nfe_steps,
        cfg_strength=args.cfg_strength,
        seed=args.seed, device=str(device),
        target_rms=args.target_rms,
        ref_rms=ref_rms,
        speed=args.speed,
        cross_fade_duration=args.cross_fade_duration,
        force_sentence_chunks=args.force_sentence_chunks,
        return_chunk_mels=args.save_mel is not None,
    )
    elapsed = _time.time() - t0
    print(f"  max_chars={max_chars}, chunks={len(chunk_metas)}")
    if chunk_metas:
        print(f"  chunk[0]: {chunk_metas[0]}")
        print(f"  chunk[-1]: {chunk_metas[-1]}")
    print(f"  总耗时: {elapsed:.1f}s")

    # 保存中间 mel
    if args.save_mel:
        if merged_chunk_mel is not None:
            np.save(args.save_mel, merged_chunk_mel)
        else:
            np.save(args.save_mel, np.zeros((1, 100, 0), dtype=np.float32))
        print(f"  mel saved: {args.save_mel}")

    # -- 6. 保存音频 --
    print("[6/6] 保存音频...")

    import soundfile as sf
    sf.write(args.output, wav_np, TARGET_SR)
    print(f"\n✓ 输出: {args.output} ({len(wav_np) / TARGET_SR:.1f}s)")


if __name__ == "__main__":
    main()
