"""
F5-TTS Base 适配共享工具集
============================

提供 mel 频谱提取、文本 token 化、DiT 模型加载、ODE 采样循环等
所有适配脚本共用的基础函数。

依赖: torch, torchaudio, safetensors, f5_tts (源码)
"""

import sys
import os
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# 路径与常量
# ============================================================

TARGET_SR = 24_000
N_MEL_CHANNELS = 100
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_FFT = 1024

STATIC_N =2048     # ONNX 固定 mel 帧数 (~22s @ 24kHz / hop=256)
STATIC_NT = 256     # ONNX 固定 text token 数
DEFAULT_NFE = 32
DEFAULT_CFG = 2.0

MODEL_CKPT = "/data01/nfs_shared/ASR_TTS/TTS-Model-20260410/F5-TTS/F5TTS_v1_Base/model_1250000.safetensors"
VOCAB_PATH = "/data01/nfs_shared/ASR_TTS/TTS-Model-20260410/F5-TTS/F5TTS_v1_Base/vocab.txt"
F5TTS_SRC = "/data01/home/axel/workspace/repo/develop/F5-TTS/src"

REF_EN_WAV = os.path.join(F5TTS_SRC, "f5_tts/infer/examples/basic/basic_ref_en.wav")
REF_ZH_WAV = os.path.join(F5TTS_SRC, "f5_tts/infer/examples/basic/basic_ref_zh.wav")


# ============================================================
# Vocab 加载
# ============================================================

def load_vocab(vocab_path: str) -> Tuple[Dict[str, int], int]:
    """加载字符级 vocab 文件，返回 {char: idx} 映射和 vocab_size。"""
    vocab_char_map = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            char = line.rstrip("\r\n")
            vocab_char_map[char] = idx
    return vocab_char_map, len(vocab_char_map)


# ============================================================
# 文本处理: 拼音转换 + token 化
# ============================================================

def _ensure_f5tts_importable():
    """确保 f5_tts 包可导入。"""
    src = Path(F5TTS_SRC).parent
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def text_to_pinyin(text_list: List[str]) -> List[List[str]]:
    """将文本列表转换为拼音 token 列表。"""
    _ensure_f5tts_importable()
    from f5_tts.model.utils import convert_char_to_pinyin
    return convert_char_to_pinyin(text_list)


def pinyin_to_tokens(
    pinyin_list: List[List[str]],
    vocab_char_map: Dict[str, int],
    max_len: int = STATIC_NT,
) -> torch.Tensor:
    """拼音列表 → padded token tensor, shape (1, max_len), pad_value=-1。"""
    _ensure_f5tts_importable()
    from f5_tts.model.utils import list_str_to_idx
    tokens = list_str_to_idx(pinyin_list, vocab_char_map)
    if tokens.shape[1] < max_len:
        tokens = F.pad(tokens, (0, max_len - tokens.shape[1]), value=-1)
    else:
        tokens = tokens[:, :max_len]
    return tokens


# ============================================================
# 音频 I/O
# ============================================================

def _trim_silence_edges(wav: torch.Tensor, threshold_db: float = -40.0) -> torch.Tensor:
    """裁剪音频首尾静音，对齐官方 preprocess_ref_audio_text 中的 remove_silence_edges。"""
    energy = wav.pow(2).squeeze(0)
    threshold = 10 ** (threshold_db / 10)
    nonzero = torch.where(energy > threshold)[0]
    if len(nonzero) == 0:
        return wav
    start = nonzero[0].item()
    end = nonzero[-1].item() + 1
    return wav[:, start:end]


def _clip_audio_to_max(wav: torch.Tensor, max_seconds: float = 12.0, sr: int = TARGET_SR) -> torch.Tensor:
    """裁剪音频到最大时长（对齐官方 12s 限制）。"""
    max_samples = int(max_seconds * sr)
    if wav.shape[-1] > max_samples:
        wav = wav[:, :max_samples]
    return wav


def _split_on_long_silence(
    wav: torch.Tensor, sr: int = TARGET_SR,
    min_silence_ms: int = 1000, threshold_db: float = -50,
) -> torch.Tensor:
    """按长静音分割并拼接，去除超长静音段（对齐官方 split_on_silence 逻辑）。"""
    frame_len = int(sr * min_silence_ms / 1000)
    if wav.shape[-1] < frame_len:
        return wav
    energy = wav.pow(2).unfold(1, frame_len, frame_len).mean(dim=-1).squeeze(0)
    threshold = 10 ** (threshold_db / 10)
    is_loud = energy > threshold

    parts = []
    silence_pad = int(sr * 0.05)  # 50ms 静音填充，对齐官方
    prev_end = 0
    for i in range(len(is_loud)):
        start = i * frame_len
        end = min(start + frame_len, wav.shape[-1])
        if is_loud[i]:
            if prev_end == 0 and start > 0:
                prev_end = start
            parts.append(wav[:, prev_end:end])
            prev_end = end
        else:
            if prev_end > 0 and start - prev_end > frame_len:
                parts.append(torch.zeros(1, min(silence_pad, frame_len)))
            prev_end = end

    if not parts:
        return wav
    result = torch.cat(parts, dim=-1)
    result = torch.cat([result, torch.zeros(1, silence_pad)], dim=-1)
    return result


def load_audio(
    path: str,
    target_sr: int = TARGET_SR,
    target_rms: float = 0.1,
    preprocess: bool = True,
    return_ref_rms: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, float]]:
    """
    加载音频 → mono → resample → 静音裁剪 → RMS 归一化 → (1, n_samples)。

    preprocess=True 时对齐官方 preprocess_ref_audio_text:
      1. 按长静音分割并拼接
      2. 裁剪到 12s
      3. 裁剪首尾静音
      4. 末尾补 50ms 静音
    """
    import torchaudio

    wav_path = path
    use_official_preprocess = False

    # 优先复用官方预处理，确保与 F5-TTS 浮点推理链路一致。
    if preprocess:
        try:
            _ensure_f5tts_importable()
            from f5_tts.infer.utils_infer import preprocess_ref_audio_text

            wav_path, _ = preprocess_ref_audio_text(
                path,
                ref_text=".",
                show_info=lambda *_args, **_kwargs: None,
            )
            use_official_preprocess = True
        except Exception:
            # 官方路径不可用时回退本地实现，避免工具脚本直接中断。
            use_official_preprocess = False

    wav, sr = torchaudio.load(wav_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    if preprocess:
        # 已走官方预处理时，这里只做后续统一归一化。
        if not use_official_preprocess:
            wav = _split_on_long_silence(wav, sr)
            wav = _clip_audio_to_max(wav, max_seconds=12.0, sr=sr)
            wav = _trim_silence_edges(wav, threshold_db=-40.0)
            # 末尾补 50ms 静音（对齐官方）
            wav = torch.cat([wav, torch.zeros(1, int(sr * 0.05))], dim=-1)

    # 对齐官方：RMS 在重采样前统计，并在解码后做回放。
    ref_rms = torch.sqrt(torch.mean(torch.square(wav))).clamp(min=1e-12)

    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)

    if ref_rms < target_rms:
        wav = wav * target_rms / ref_rms

    if return_ref_rms:
        return wav, float(ref_rms.item())
    return wav


def save_audio(waveform: torch.Tensor, path: str, sr: int = TARGET_SR):
    """保存波形到文件。"""
    import torchaudio
    torchaudio.save(path, waveform.cpu(), sr)


# ============================================================
# Mel 频谱提取
# ============================================================

def extract_mel_spec(
    audio: torch.Tensor,
    n_mel: int = N_MEL_CHANNELS,
    hop: int = HOP_LENGTH,
    win: int = WIN_LENGTH,
    n_fft: int = N_FFT,
    sr: int = TARGET_SR,
) -> torch.Tensor:
    """
    从原始波形提取 log mel 频谱。

    audio: (1, n_samples) @ sr Hz
    → mel: (1, N_frames, n_mel)
    """
    _ensure_f5tts_importable()
    from f5_tts.model.modules import MelSpec
    mel_spec = MelSpec(
        n_fft=n_fft,
        hop_length=hop,
        win_length=win,
        n_mel_channels=n_mel,
        target_sample_rate=sr,
        mel_spec_type="vocos",
    )
    mel = mel_spec(audio)          # (1, n_mel, N_frames)
    return mel.permute(0, 2, 1)    # (1, N_frames, n_mel)


# ============================================================
# DiT 模型加载
# ============================================================

def load_f5tts_dit(
    ckpt_path: str = MODEL_CKPT,
    vocab_path: str = VOCAB_PATH,
    device: str = "cpu",
) -> nn.Module:
    """
    加载 F5TTS_Base 的 DiT backbone（提取 EMA 权重）。

    返回 eval 模式的 DiT 模型。
    """
    _ensure_f5tts_importable()
    from f5_tts.model.backbones.dit import DiT
    from safetensors.torch import load_file

    _, vocab_size = load_vocab(vocab_path)

    model = DiT(
        dim=1024,
        depth=22,
        heads=16,
        dim_head=64,
        ff_mult=2,
        mel_dim=N_MEL_CHANNELS,
        text_num_embeds=vocab_size,
        text_dim=512,
        text_mask_padding=True,
        conv_layers=4,
        pe_attn_head=None,
        qk_norm=None,
        attn_backend="torch",
        attn_mask_enabled=True,
    )

    checkpoint = load_file(ckpt_path, device=device)
    model_state = {
        k.replace("ema_model.", ""): v
        for k, v in checkpoint.items()
        if k not in ("initted", "step")
    }
    # safetensors keys: ema_model.transformer.xxx → 去掉 ema_model. 后仍有 transformer. 前缀
    # DiT 模型 state_dict 期望无 transformer. 前缀
    if any(k.startswith("transformer.") for k in model_state):
        model_state = {
            k.replace("transformer.", "", 1): v
            for k, v in model_state.items()
        }
    for key in (
        "mel_spec.mel_stft.mel_scale.fb",
        "mel_spec.mel_stft.spectrogram.window",
    ):
        model_state.pop(key, None)

    model.load_state_dict(model_state, strict=False)
    del checkpoint
    torch.cuda.empty_cache()

    return model.eval().to(device)


# ============================================================
# LayerNorm 缩放 (FP16 稳定性)
# ============================================================

class ScaledLayerNorm(nn.Module):
    """输入 ÷ scale 后过原始 LayerNorm，防止 FP16 溢出。"""

    def __init__(self, original_ln: nn.LayerNorm, scale: float = 32.0):
        super().__init__()
        self.ln = original_ln
        self.register_buffer("inv_scale", torch.tensor(1.0 / scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x * self.inv_scale)


def replace_layernorm_with_scaled(model: nn.Module, scale: float = 32.0):
    """递归替换所有 nn.LayerNorm 为 ScaledLayerNorm。"""
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm):
            setattr(model, name, ScaledLayerNorm(module, scale))
        else:
            replace_layernorm_with_scaled(module, scale)


# ============================================================
# ODE 采样循环
# ============================================================

def ode_sample(
    model_fn: Callable,
    cond_mel: torch.Tensor,
    text_tokens: torch.Tensor,
    duration: int,
    steps: int = DEFAULT_NFE,
    cfg_strength: float = DEFAULT_CFG,
    seed: int = 42,
    device: str = "cpu",
    static_n: int = STATIC_N,
) -> torch.Tensor:
    """
    Flow Matching ODE 采样（Euler 积分 + CFG）。

    model_fn:    callable(x, cond, text, time[, input_lengths]) → velocity (1, static_n, 100)
    cond_mel:    (1, N_ref, 100) 参考音频 mel
    text_tokens: (1, NT) 文本 token (-1 为 pad)
    duration:    生成目标帧数

    返回: (1, duration, 100) 生成的 mel 频谱
    """
    torch.manual_seed(seed)

    # -- 对齐官方 CFM.sample 的时长下限保护 --
    # 1) 至少覆盖有效文本 token 数 + 1
    # 2) 至少覆盖参考 mel 长度 + 1，确保一定会生成非参考部分
    valid_text_tokens = int((text_tokens != -1).sum().item())
    min_duration = max(valid_text_tokens + 1, cond_mel.shape[1] + 1)
    duration = max(duration, min_duration)
    duration = min(duration, static_n)

    # -- 条件准备 --
    step_cond = F.pad(cond_mel, (0, 0, 0, static_n - cond_mel.shape[1]), value=0.0)
    zeros_cond = torch.zeros_like(step_cond)
    # 对齐官方 drop_text=True: text 全 -1 → +1 后全 0 → embedding index 0 (filler)
    uncond_tokens = torch.full_like(text_tokens, -1)
    input_lengths = torch.tensor([duration], device=device, dtype=torch.int32)
    frame_ids = torch.arange(static_n, device=device, dtype=torch.int32).unsqueeze(0)
    audio_mask = (frame_ids < input_lengths.unsqueeze(1)).unsqueeze(-1)

    # -- 初始噪声 --
    y = torch.randn(1, duration, N_MEL_CHANNELS, device=device, dtype=step_cond.dtype)
    y = F.pad(y, (0, 0, 0, static_n - duration), value=0.0)
    y = torch.where(audio_mask, y, torch.zeros_like(y))

    # -- 时间步调度 --
    _ensure_f5tts_importable()
    from f5_tts.model.utils import get_epss_timesteps
    t = get_epss_timesteps(steps, device=device, dtype=step_cond.dtype)
    # F5TTS API 默认 sway_sampling_coef=-1
    t = t + (-1) * (torch.cos(torch.pi / 2 * t) - 1 + t)

    # -- Euler 积分 --
    for i in range(len(t) - 1):
        t_now = t[i].unsqueeze(0)
        dt = t[i + 1] - t[i]

        try:
            vel_cond = model_fn(y, step_cond, text_tokens, t_now, input_lengths)
        except TypeError:
            vel_cond = model_fn(y, step_cond, text_tokens, t_now)
        if cfg_strength > 1e-5:
            try:
                vel_uncond = model_fn(y, zeros_cond, uncond_tokens, t_now, input_lengths)
            except TypeError:
                vel_uncond = model_fn(y, zeros_cond, uncond_tokens, t_now)
            vel = vel_cond + (vel_cond - vel_uncond) * cfg_strength
        else:
            vel = vel_cond

        vel = torch.where(audio_mask, vel, torch.zeros_like(vel))
        y = y + dt * vel
        y = torch.where(audio_mask, y, torch.zeros_like(y))

    # -- 截取有效区域，条件部分用原始 mel 覆盖 --
    out = y[:, :duration, :]
    ref_len = cond_mel.shape[1]
    out[:, :ref_len, :] = cond_mel[:, :min(ref_len, duration), :]
    return out


# ============================================================
# 时长估计
# ============================================================

def estimate_duration(
    ref_mel_len: int,
    ref_text: str,
    gen_text: str,
    speed: float = 1.0,
) -> int:
    """根据参考音频时长和文本长度估计生成帧数（对齐 F5TTS API）。"""
    # F5TTS API: ref_text 末尾加空格
    if len(ref_text[-1].encode("utf-8")) == 1:
        ref_text = ref_text + " "
    # F5TTS API: 短文本降速
    if len(gen_text.encode("utf-8")) < 10:
        speed = 0.3
    ref_bytes = max(len(ref_text.encode("utf-8")), 1)
    gen_bytes = len(gen_text.encode("utf-8"))
    return ref_mel_len + int(ref_mel_len / ref_bytes * gen_bytes / speed)


def estimate_duration_robust(
    ref_mel_len: int,
    ref_text: str,
    gen_text: str,
    speed: float = 1.0,
) -> Tuple[int, int, int]:
    """
    更稳健的时长估计（尤其适配中英混合/跨语种）：
      1) byte 比例估计（兼容官方）
      2) token 比例估计（避免 byte 比例低估英文尾句）
      3) 使用融合策略，避免 token 估计过大导致句中长静音

    返回:
      (duration, duration_by_bytes, duration_by_tokens)
    """
    # 与官方一致：英文短句降速
    local_speed = speed
    if len(gen_text.encode("utf-8")) < 10:
        local_speed = 0.3

    # byte 比例
    ref_bytes = max(len(ref_text.encode("utf-8")), 1)
    gen_bytes = len(gen_text.encode("utf-8"))
    duration_by_bytes = ref_mel_len + int(ref_mel_len / ref_bytes * gen_bytes / local_speed)

    # token 比例（按 convert_char_to_pinyin 后的离散长度）
    ref_tokens = max(len(text_to_pinyin([ref_text])[0]), 1)
    gen_tokens = len(text_to_pinyin([gen_text])[0])
    duration_by_tokens = ref_mel_len + int(ref_mel_len / ref_tokens * gen_tokens / local_speed)

    # 融合策略：
    # - token <= byte：使用 byte（与官方一致）
    # - token > byte：只吸收 token 额外部分的一部分，防止过估计引入长静音
    if duration_by_tokens <= duration_by_bytes:
        duration = duration_by_bytes
    else:
        extra = duration_by_tokens - duration_by_bytes
        duration = duration_by_bytes + int(extra * 0.65)
        # 至少给一小段安全余量，减少末词被截断风险（约 0.34s）
        duration = max(duration, duration_by_bytes + 32)

    return duration, duration_by_bytes, duration_by_tokens


def chunk_text_official(text: str, max_chars: int = 135) -> List[str]:
    """对齐官方 utils_infer.chunk_text 的分块逻辑。"""
    chunks = []
    current_chunk = ""
    sentences = re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", text)

    for sentence in sentences:
        if not sentence:
            continue
        if len(current_chunk.encode("utf-8")) + len(sentence.encode("utf-8")) <= max_chars:
            current_chunk += sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + " " if sentence and len(sentence[-1].encode("utf-8")) == 1 else sentence

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def make_gen_batches_official(
    ref_text: str,
    gen_text: str,
    ref_audio_samples: int,
    speed: float = 1.0,
    sr: int = TARGET_SR,
    force_sentence_chunks: bool = False,
) -> Tuple[List[str], int]:
    """
    对齐官方 infer_process 的 max_chars 估算 + 文本分块。
    返回 (gen_text_batches, max_chars)。
    """
    ref_sec = max(ref_audio_samples / sr, 1e-6)
    max_chars = int(len(ref_text.encode("utf-8")) / ref_sec * (22 - ref_sec) * speed)
    max_chars = max(max_chars, 8)
    if force_sentence_chunks:
        # 强制按句切分（适配中英文逗号/句号），每句独立走一次采样后再 cross-fade。
        batches = [s.strip() for s in re.split(r"(?<=[;:,.!?])\s+|(?<=[；：，。！？])", gen_text) if s.strip()]
    else:
        batches = chunk_text_official(gen_text, max_chars=max_chars)
    return batches, max_chars


def combine_waves_with_cross_fade(
    generated_waves: List[np.ndarray],
    cross_fade_duration: float = 0.15,
    sr: int = TARGET_SR,
) -> np.ndarray:
    """对齐官方 infer_batch_process 的 cross-fade 拼接逻辑。"""
    if not generated_waves:
        return np.zeros(0, dtype=np.float32)
    if len(generated_waves) == 1 or cross_fade_duration <= 0:
        return np.concatenate(generated_waves, axis=0).astype(np.float32, copy=False)

    final_wave = generated_waves[0]
    for i in range(1, len(generated_waves)):
        prev_wave = final_wave
        next_wave = generated_waves[i]

        cross_fade_samples = int(cross_fade_duration * sr)
        cross_fade_samples = min(cross_fade_samples, len(prev_wave), len(next_wave))
        if cross_fade_samples <= 0:
            final_wave = np.concatenate([prev_wave, next_wave], axis=0)
            continue

        prev_overlap = prev_wave[-cross_fade_samples:]
        next_overlap = next_wave[:cross_fade_samples]
        fade_out = np.linspace(1, 0, cross_fade_samples, dtype=prev_overlap.dtype)
        fade_in = np.linspace(0, 1, cross_fade_samples, dtype=prev_overlap.dtype)
        cross_faded_overlap = prev_overlap * fade_out + next_overlap * fade_in
        final_wave = np.concatenate(
            [prev_wave[:-cross_fade_samples], cross_faded_overlap, next_wave[cross_fade_samples:]],
            axis=0,
        )

    return final_wave.astype(np.float32, copy=False)


def infer_chunked_wave_official(
    model_fn: Callable,
    vocoder,
    cond_mel: torch.Tensor,
    ref_audio_len: int,
    ref_audio_samples: int,
    ref_text: str,
    gen_text: str,
    vocab_map: Dict[str, int],
    nfe_steps: int,
    cfg_strength: float,
    seed: int,
    device: str,
    target_rms: float,
    ref_rms: float,
    speed: float = 1.0,
    cross_fade_duration: float = 0.15,
    force_sentence_chunks: bool = False,
    static_n: int = STATIC_N,
    static_nt: int = STATIC_NT,
    return_chunk_mels: bool = False,
) -> Tuple[np.ndarray, List[Dict[str, int]], Optional[np.ndarray], int]:
    """
    对齐官方 chunk + cross-fade 的推理管线，底层模型前向由 model_fn 提供。
    返回:
      final_wave, chunk_metas, merged_chunk_mel(optional), max_chars
    """
    gen_batches, max_chars = make_gen_batches_official(
        ref_text=ref_text,
        gen_text=gen_text,
        ref_audio_samples=ref_audio_samples,
        speed=speed,
        sr=TARGET_SR,
        force_sentence_chunks=force_sentence_chunks,
    )
    if not gen_batches:
        return np.zeros(0, dtype=np.float32), [], None, max_chars

    chunk_waves: List[np.ndarray] = []
    chunk_metas: List[Dict[str, int]] = []
    chunk_mels: List[np.ndarray] = []

    for i, text_chunk in enumerate(gen_batches):
        local_speed = speed
        if len(text_chunk.encode("utf-8")) < 10:
            local_speed = 0.3

        tokens = pinyin_to_tokens(text_to_pinyin([ref_text + text_chunk]), vocab_map, static_nt)
        ref_bytes = max(len(ref_text.encode("utf-8")), 1)
        gen_bytes = len(text_chunk.encode("utf-8"))
        duration = ref_audio_len + int(ref_audio_len / ref_bytes * gen_bytes / local_speed)
        duration = min(duration, static_n)

        mel_out = ode_sample(
            model_fn,
            cond_mel,
            tokens,
            duration,
            steps=nfe_steps,
            cfg_strength=cfg_strength,
            seed=seed,
            device=device,
            static_n=static_n,
        )
        gen_mel = mel_out[:, ref_audio_len:, :].permute(0, 2, 1)
        wav = vocoder.decode(gen_mel).squeeze().detach().cpu().numpy()
        if ref_rms < target_rms:
            wav = wav * (ref_rms / target_rms)

        chunk_waves.append(wav.astype(np.float32, copy=False))
        chunk_metas.append(
            {
                "index": i,
                "text_bytes": gen_bytes,
                "duration": int(duration),
                "actual_duration": int(mel_out.shape[1]),
            }
        )
        if return_chunk_mels:
            chunk_mels.append(gen_mel.detach().cpu().numpy())

    final_wave = combine_waves_with_cross_fade(
        chunk_waves,
        cross_fade_duration=cross_fade_duration,
        sr=TARGET_SR,
    )

    merged_chunk_mel = None
    if return_chunk_mels and chunk_mels:
        merged_chunk_mel = np.concatenate(chunk_mels, axis=2)

    return final_wave, chunk_metas, merged_chunk_mel, max_chars


# ============================================================
# Vocoder 加载
# ============================================================

def load_vocos_vocoder(device: str = "cpu"):
    """加载 vocos vocoder（mel → 波形）。"""
    _ensure_f5tts_importable()
    from f5_tts.infer.utils_infer import load_vocoder
    return load_vocoder("vocos", False, None, device)
