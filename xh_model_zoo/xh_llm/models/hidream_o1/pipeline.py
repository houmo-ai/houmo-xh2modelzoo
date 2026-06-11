import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

from xhquant.api import HMONNXGoldenInference


PATCH_SIZE = 32


def hidream_o1_source_dir() -> Path:
    return Path(__file__).resolve().parent


def ensure_hidream_o1_imports() -> None:
    src = str(hidream_o1_source_dir())
    if src not in sys.path:
        sys.path.insert(0, src)


def add_special_tokens(tokenizer) -> None:
    tokenizer.boi_token = "<|boi_token|>"
    tokenizer.bor_token = "<|bor_token|>"
    tokenizer.eor_token = "<|eor_token|>"
    tokenizer.bot_token = "<|bot_token|>"
    tokenizer.tms_token = "<|tms_token|>"


def get_tokenizer(processor):
    from transformers import PreTrainedTokenizerBase

    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


def patchify_rgb_noise(noise: torch.Tensor, patch_size: int = PATCH_SIZE) -> torch.Tensor:
    batch, channels, height, width = noise.shape
    return (
        noise.reshape(batch, channels, height // patch_size, patch_size, width // patch_size, patch_size)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, -1, channels * patch_size * patch_size)
    )


def unpatchify_rgb_patches(
    patches: torch.Tensor,
    height: int,
    width: int,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    batch = patches.shape[0]
    return (
        patches.reshape(batch, height // patch_size, width // patch_size, 3, patch_size, patch_size)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(batch, 3, height, width)
    )


def build_t2i_sample_inputs(
    model: nn.Module,
    processor,
    prompt: str,
    height: int,
    width: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
    timestep: float = 0.001,
    text_seq_len: int = 512,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    ensure_hidream_o1_imports()
    from models.pipeline import build_t2i_text_sample  # pyright: ignore[reportMissingImports]

    tokenizer = get_tokenizer(processor)
    add_special_tokens(tokenizer)
    sample = build_t2i_text_sample(prompt, height, width, tokenizer, processor, model.config)
    sample = {key: (value.to(device) if torch.is_tensor(value) else value) for key, value in sample.items()}

    original_text_seq_len = int(sample["input_ids"].shape[-1])
    if text_seq_len <= 0:
        text_seq_len = original_text_seq_len
    if text_seq_len < original_text_seq_len:
        raise ValueError(
            f"text_seq_len={text_seq_len} must be >= prompt text length {original_text_seq_len}; "
            "increase --text-seq-len or shorten the prompt."
        )
    image_seq_len = int(sample["token_types"].shape[-1]) - original_text_seq_len
    if text_seq_len != original_text_seq_len:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id
        if pad_token_id is None:
            pad_token_id = 0

        old_input_ids = sample["input_ids"]
        old_position_ids = sample["position_ids"]
        batch_size = int(old_input_ids.shape[0])
        text_without_tms_len = original_text_seq_len - 1
        pad_start = text_without_tms_len
        pad_end = text_seq_len - 1

        input_ids = torch.full((batch_size, text_seq_len), pad_token_id, dtype=old_input_ids.dtype, device=device)
        input_ids[:, :text_without_tms_len] = old_input_ids[:, :text_without_tms_len]
        input_ids[:, text_seq_len - 1] = old_input_ids[:, original_text_seq_len - 1]
        sample["input_ids"] = input_ids

        new_total_seq_len = text_seq_len + image_seq_len
        token_types = torch.zeros((batch_size, new_total_seq_len), dtype=sample["token_types"].dtype, device=device)
        token_types[:, text_seq_len - 1 :] = 1
        sample["token_types"] = token_types
        vinput_mask = torch.zeros((batch_size, new_total_seq_len), dtype=torch.bool, device=device)
        vinput_mask[:, text_seq_len:] = True
        sample["vinput_mask"] = vinput_mask
        sample["pad_positions"] = torch.arange(pad_start, pad_end, dtype=torch.long, device=device)

        new_position_ids = torch.zeros(
            (*old_position_ids.shape[:-1], new_total_seq_len),
            dtype=old_position_ids.dtype,
            device=device,
        )
        new_position_ids[..., :text_without_tms_len] = old_position_ids[..., :text_without_tms_len]
        pad_positions = torch.arange(pad_start, pad_end, dtype=old_position_ids.dtype, device=device)
        if pad_positions.numel() > 0:
            new_position_ids[..., pad_start:pad_end] = pad_positions.view(*([1] * (new_position_ids.ndim - 1)), -1)
        # Keep RoPE positions of all real non-pad tokens identical to the
        # unpadded prompt path. Padding changes physical sequence indices only;
        # it must not shift tms/image position_ids, otherwise rotary embeddings
        # change and padded output no longer matches the original prompt.
        new_position_ids[..., text_seq_len - 1] = old_position_ids[..., original_text_seq_len - 1]
        new_position_ids[..., text_seq_len:] = old_position_ids[..., original_text_seq_len:]
        sample["position_ids"] = new_position_ids
    else:
        sample["pad_positions"] = torch.empty((0,), dtype=torch.long, device=device)

    generator = torch.Generator("cpu").manual_seed(seed + 1)
    noise = torch.randn((1, 3, height, width), generator=generator, dtype=torch.float32)
    vinputs = patchify_rgb_noise(noise).to(device=device, dtype=dtype)
    sample["timestep"] = torch.tensor([timestep], device=device, dtype=torch.float32)
    fixed_timesteps = torch.tensor(
        [
            0.001,
            0.007,
            0.014,
            0.021,
            0.029,
            0.036,
            0.044,
            0.052,
            0.060,
            0.069,
            0.078,
            0.087,
            0.096,
            0.105,
            0.115,
            0.126,
            0.136,
            0.147,
            0.159,
            0.170,
            0.182,
            0.195,
            0.208,
            0.222,
            0.236,
            0.251,
            0.266,
            0.282,
            0.298,
            0.316,
            0.334,
            0.353,
            0.373,
            0.393,
            0.415,
            0.438,
            0.462,
            0.487,
            0.514,
            0.542,
            0.572,
            0.604,
            0.637,
            0.672,
            0.710,
            0.751,
            0.794,
            0.840,
            0.889,
            0.943,
        ],
        dtype=torch.float32,
        device=device,
    )
    timestep_index = torch.argmin(torch.abs(fixed_timesteps - sample["timestep"].reshape(-1, 1)), dim=1)
    sample["timestep_index"] = timestep_index.to(dtype=torch.int32)
    total_seq_len = int(sample["token_types"].shape[-1])
    token_types = sample["token_types"].to(device=device)
    min_val = torch.finfo(dtype).min
    causal_mask = torch.triu(
        torch.full((total_seq_len, total_seq_len), min_val, device=device, dtype=dtype),
        diagonal=1,
    )
    attention_mask = causal_mask.unsqueeze(0).unsqueeze(0).repeat(token_types.shape[0], 1, 1, 1)
    # token_types > 0 are denoise/generation positions: the original HiDream
    # path allows them to attend to the whole text+image sequence.
    gen_positions = token_types.bool()
    attention_mask[:, 0][gen_positions] = 0
    pad_positions = sample["pad_positions"]
    if pad_positions.numel() > 0:
        attention_mask[:, :, :, pad_positions] = min_val
        attention_mask[:, :, pad_positions, :] = min_val
        batch_indices = torch.arange(token_types.shape[0], device=device)
        for pad_pos in pad_positions.tolist():
            attention_mask[batch_indices, 0, pad_pos, pad_pos] = 0
    sample["attention_mask"] = attention_mask
    return sample, vinputs


@torch.no_grad()
def build_rotary_inputs(
    rotary_emb: nn.Module,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rotary_cos, rotary_sin = rotary_emb(inputs_embeds, position_ids)
    return rotary_cos.to(dtype=dtype), rotary_sin.to(dtype=dtype)


@torch.no_grad()
def build_timestep_embeddings(
    timestep_embedder: nn.Module,
    timesteps: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    return timestep_embedder(timesteps).to(dtype=dtype)


class HiDreamO1DenoiseExportWrapper(nn.Module):
    """Single-step text-to-image denoise graph wrapper for HiDream-O1 Pixel-DiT."""

    def __init__(self, model: nn.Module, txt_seq_len: int):
        super().__init__()
        self.model = model
        self.txt_seq_len = int(txt_seq_len)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        vinputs: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_embeddings=(rotary_cos, rotary_sin),
            vinputs=vinputs,
            timestep=timestep,
            token_types=None,
            use_flash_attn=False,
        )
        return outputs.x_pred[:, self.txt_seq_len :, :]


class HiDreamO1HMONNXDenoiseInference(nn.Module):
    def __init__(
        self,
        hmonnx_path: str | Path,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.runtime = HMONNXGoldenInference(str(hmonnx_path))
        self._device = (
            torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._dtype = dtype
        self.runtime.exec_device = self._device

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def to(self, *args, **kwargs):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self._device = torch.device(device)
            self.runtime.exec_device = self._device
        if dtype is not None:
            self._dtype = dtype
        return self

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        rotary_cos: torch.Tensor,
        rotary_sin: torch.Tensor,
        vinputs: torch.Tensor,
        timestep: torch.Tensor,
        token_types: torch.Tensor,
    ) -> torch.Tensor:
        out = self.runtime(
            inputs_embeds.to(device=self.device, dtype=self.dtype),
            attention_mask.to(device=self.device, dtype=self.dtype),
            rotary_cos.to(device=self.device, dtype=self.dtype),
            rotary_sin.to(device=self.device, dtype=self.dtype),
            vinputs.to(device=self.device, dtype=self.dtype),
            timestep.to(device=self.device, dtype=torch.int32),
            token_types.to(device=self.device, dtype=torch.int32),
        )
        x_pred = out[0] if isinstance(out, tuple) else out
        return x_pred.to(device=self.device, dtype=self.dtype)
