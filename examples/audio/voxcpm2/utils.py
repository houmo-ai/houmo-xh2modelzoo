"""VoxCPM2 导出/推理共用的工具函数。

与 qwen3_asr 的 utils 定位一致:
- 固定长度输入构造(用于 calibration 和 golden)
- HF 配置文件拷贝、meta_info 写入
- AudioVAE 预处理工具
- prefill 序列长度、KV cache shape 等推导
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Optional

import librosa
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 用于校准的默认 prompt 音频,用户可通过 --audio 覆盖
DEFAULT_AUDIO_PATH = Path("/LibriSpeech/test-clean/2094/142345/2094-142345-0008.flac")

# VoxCPM2 音频采样率(由 config.json 的 audio_vae_config.sample_rate 决定)
VOXCPM2_ENCODE_SAMPLE_RATE = 16000

# 复用 qwen3_asr 的 HF 配置文件清单
HF_CONFIG_FILES = [
    "added_tokens.json",
    "chat_template.json",
    "chat_template.jinja",
    "config.json",
    "configuration.json",
    "generation_config.json",
    "merges.txt",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
]


# ---------------------------------------------------------------------------
# 参数校验
# ---------------------------------------------------------------------------

def validate_prefill_length(length: int) -> None:
    """校验 prefill 序列长度参数。"""
    if length <= 0:
        raise ValueError(f"prefill_length must be positive, but got {length}.")


def validate_cache_length(length: int) -> None:
    """校验 decode KV cache 长度参数。"""
    if length <= 0:
        raise ValueError(f"cache_length must be positive, but got {length}.")


def normalize_audio_path(audio_path: str | Path) -> Path:
    """解析音频路径,并确保能找到该文件。"""
    path = Path(audio_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Calibration audio not found: {path}")
    return path


# ---------------------------------------------------------------------------
# prefill 长度推导
# ---------------------------------------------------------------------------

def compute_max_prompt_audio_patches(prefill_length: int, reserved_text_tokens: int = 16) -> int:
    """根据 prefill 总长度,推算 prompt 音频最多能放多少 patch。

    prefill = text_tokens + ref_audio_patches(optional) + prompt_audio_patches + special_tokens
    这里做一个保守估算,给调用方作参考,实际构造时以实际 shape 为准。
    """
    return max(0, prefill_length - reserved_text_tokens)


# ---------------------------------------------------------------------------
# AudioVAE 固定长度输入构造
# ---------------------------------------------------------------------------

def load_and_pad_audio(
    audio_path: Path,
    target_samples: int,
    sample_rate: int = VOXCPM2_ENCODE_SAMPLE_RATE,
) -> np.ndarray:
    """读取音频并裁剪/补零到固定长度。"""
    audio, _ = librosa.load(str(audio_path), sr=sample_rate, mono=True)
    if len(audio) >= target_samples:
        return audio[:target_samples].astype(np.float32, copy=False)
    padded = np.pad(audio, (0, target_samples - len(audio)))
    return padded.astype(np.float32, copy=False)


def compute_audiovae_encoder_input_length(
    num_patches: int,
    patch_size: int,
    chunk_size: int,
) -> int:
    """AudioVAE encoder 的定长输入长度(样本数)。

    VAE encode 的 hop_length = chunk_size,所以
        num_patches * patch_size * chunk_size 个样本 → num_patches * patch_size 个 latent 时间步
    """
    return num_patches * patch_size * chunk_size


def compute_audiovae_decoder_latent_length(
    num_patches: int,
    patch_size: int,
) -> int:
    """AudioVAE decoder 输入的 latent 时间步数。

    decoder 输入 shape = [B, latent_dim, num_patches * patch_size]
    """
    return num_patches * patch_size


def compute_audiovae_decoder_output_samples(
    num_patches: int,
    patch_size: int,
    decoder_rates: list[int],
) -> int:
    """AudioVAE decoder 输出的波形样本数。"""
    upscale = int(math.prod(decoder_rates))
    return num_patches * patch_size * upscale


def build_fixed_vae_encoder_input(
    audio_path: Path,
    num_patches: int,
    patch_size: int,
    chunk_size: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """构造 AudioVAE encoder 的定长 calibration 输入 [1, 1, L]。"""
    target_samples = compute_audiovae_encoder_input_length(num_patches, patch_size, chunk_size)
    audio_np = load_and_pad_audio(audio_path, target_samples)
    tensor = torch.from_numpy(audio_np).to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(0)
    return tensor  # [1, 1, L]


def build_fixed_vae_decoder_input(
    num_patches: int,
    patch_size: int,
    latent_dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """构造 AudioVAE decoder 的定长 calibration 输入。

    Returns:
        z: [1, latent_dim, num_patches * patch_size]
        sr_cond: [1] int32,对应 out_sample_rate=48000 → bucket 索引 3
    """
    T = compute_audiovae_decoder_latent_length(num_patches, patch_size)
    g = torch.Generator(device="cpu").manual_seed(seed)
    z = torch.randn((1, latent_dim, T), generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    sr_cond = torch.tensor([48000], dtype=torch.int32, device=device)
    return z, sr_cond


# ---------------------------------------------------------------------------
# weight_norm 展开(为导出)
# ---------------------------------------------------------------------------

def remove_weight_norm_recursively(module: nn.Module) -> nn.Module:
    """递归移除所有 weight_norm 的 reparametrization,保留权重数值等价。

    这是无损的:weight_norm 只是把 weight 表示为 (g, v) 的 parametrize,移除后
    module 上直接挂一个 weight 参数,推理等价。
    """
    from torch.nn.utils import remove_weight_norm

    for m in module.modules():
        # 只有真正被 weight_norm 过的 module 才有 'weight_v' 属性
        if hasattr(m, "weight_v") and hasattr(m, "weight_g"):
            try:
                remove_weight_norm(m)
            except Exception:
                pass
    return module


# ---------------------------------------------------------------------------
# VoxCPM2 prompt 构造(host 侧,和导出路径无关,但 export_lm 用它构造 calibration)
# ---------------------------------------------------------------------------

def build_calibration_prompt_text() -> str:
    """构造一段用于 calibration 的默认 prompt 文本(英文短句,避免 tokenizer 方言差异)。"""
    return "Hello, this is a calibration prompt for VoxCPM2 export."


def build_calibration_target_text() -> str:
    """构造一段用于 calibration 的 target 文本。"""
    return "The quick brown fox jumps over the lazy dog."


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------

def copy_hf_config_files(
    config_dir: Path,
    output_dir: Path,
    logger=None,
    filenames: Optional[list[str]] = None,
) -> list[Path]:
    """复制导出和 demo 需要的 HF 配置文件,缺失文件仅记录 warning。"""
    config_dir = Path(config_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    copied_files: list[Path] = []
    for cfg_file in (filenames or HF_CONFIG_FILES):
        src = config_dir / cfg_file
        if not src.exists():
            if logger is not None:
                logger.warning("Skip missing config file: %s", src)
            continue
        dst = output_dir / cfg_file
        shutil.copyfile(src, dst)
        copied_files.append(dst)
    return copied_files


def write_json_file(json_path: Path, payload: Any) -> None:
    """原子写入 JSON,避免导出中断留下半成品。"""
    json_path = Path(json_path)
    json_path.parent.mkdir(exist_ok=True, parents=True)
    tmp_path = json_path.with_suffix(f"{json_path.suffix}.tmp")
    # 兼容 ConfigDict 等非标准对象
    def _default(obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        if isinstance(obj, (Path,)):
            return str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, torch.Size):
            return list(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, ensure_ascii=False, default=_default)
        f.write("\n")
    tmp_path.replace(json_path)


def load_json_if_exists(json_path: Path) -> Optional[dict]:
    """存在则读取 JSON,不存在则返回 None。"""
    json_path = Path(json_path)
    if not json_path.exists():
        return None
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# 诊断/日志辅助
# ---------------------------------------------------------------------------

def log_tensor_stats(name: str, tensor: torch.Tensor, logger) -> None:
    """打印张量关键统计信息,用于排查导出前后数值偏差。"""
    if logger is None:
        return
    t = tensor.detach().float().cpu()
    logger.info(
        "%s: shape=%s dtype=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
        name,
        tuple(tensor.shape),
        tensor.dtype,
        float(t.min()),
        float(t.max()),
        float(t.mean()),
        float(t.std()),
    )