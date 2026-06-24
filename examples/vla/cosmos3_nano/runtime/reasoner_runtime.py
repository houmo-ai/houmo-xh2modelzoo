# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Fixed-shape Cosmos3-Nano reasoner HMONNX runtime.

This module owns host-side graph orchestration only. It intentionally leaves
tokenization, sampling, EOS handling, and image preprocessing outside the
runtime until those paths are validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from xhquant.api import HMONNXInference, xhquant_init

from export.common_quant.attention.export_transformer_attention_core import make_causal_mask


IMAGE_TOKEN_ID = 151655


def as_tuple(value: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def require_paths(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing HMONNX files:\n" + "\n".join(missing))


def fuse_text_image(text_embeds: torch.Tensor, image_embeds: torch.Tensor, image_mask: torch.Tensor) -> torch.Tensor:
    fused = text_embeds.clone()
    image_token_count = int(image_mask.sum().item())
    if image_token_count != image_embeds.shape[0]:
        raise ValueError(f"image token count {image_token_count} != image embeddings {image_embeds.shape[0]}")
    fused[image_mask] = image_embeds.to(device=fused.device, dtype=fused.dtype)
    return fused


@dataclass(frozen=True)
class ReasonerHmonnxPaths:
    text: Path
    text_seq1: Path
    vision_patch: Path
    vision_core: Path
    prefill_kv: Path
    decode: Path
    logits_prefill: Path
    logits_seq1: Path

    def all(self) -> list[Path]:
        return [
            self.text,
            self.text_seq1,
            self.vision_patch,
            self.vision_core,
            self.prefill_kv,
            self.decode,
            self.logits_prefill,
            self.logits_seq1,
        ]

    def as_dict(self) -> dict[str, str]:
        return {
            "text": str(self.text),
            "text_seq1": str(self.text_seq1),
            "vision_patch": str(self.vision_patch),
            "vision_core": str(self.vision_core),
            "prefill_kv": str(self.prefill_kv),
            "decode": str(self.decode),
            "logits_prefill": str(self.logits_prefill),
            "logits_seq1": str(self.logits_seq1),
        }


@dataclass(frozen=True)
class ReasonerRuntimeInputs:
    input_ids: torch.Tensor
    image_mask: torch.Tensor
    position_ids: torch.Tensor
    prefill_attention_mask: torch.Tensor
    pixel_values: torch.Tensor
    grid_thw: torch.Tensor
    current_token_id: torch.Tensor
    decode_position_ids: torch.Tensor
    decode_attention_mask: torch.Tensor


@dataclass(frozen=True)
class ReasonerRuntimeOutput:
    text_embeds: torch.Tensor
    image_embeds_full: torch.Tensor
    image_embeds: torch.Tensor
    fused_embeds: torch.Tensor
    prefill_hidden: torch.Tensor
    prefill_kv: tuple[torch.Tensor, ...]
    prefill_logits: torch.Tensor
    current_embed: torch.Tensor
    decode_hidden: torch.Tensor
    decode_logits: torch.Tensor


def make_reasoner_inputs(
    *,
    batch: int = 1,
    seq: int = 8,
    image_tokens: int = 4,
    height: int = 224,
    width: int = 224,
    seed: int = 7,
    mask_value: float = -10000.0,
    current_token_id: int = 24,
) -> ReasonerRuntimeInputs:
    prefix = [151643, 785]
    suffix = [279, 151645]
    expected_seq = len(prefix) + image_tokens + len(suffix)
    if batch != 1:
        raise ValueError("Reasoner smoke runtime currently expects batch=1")
    if seq != expected_seq:
        raise ValueError(f"seq must equal image_tokens + 4 for this prompt template, got seq={seq}, image_tokens={image_tokens}")
    if height % 16 != 0 or width % 16 != 0:
        raise ValueError("height and width must be divisible by 16")

    input_ids = torch.tensor([prefix + [IMAGE_TOKEN_ID] * image_tokens + suffix], dtype=torch.long)
    image_mask = input_ids == IMAGE_TOKEN_ID
    position_ids = torch.arange(seq, dtype=torch.int32).reshape(1, seq)
    prefill_attention_mask = make_causal_mask(batch, seq, mask_value, device=torch.device("cpu"), dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pixel_values = torch.rand((2, 3, height, width), generator=generator, dtype=torch.float32)
    grid_thw = torch.tensor([[1, height // 16, width // 16]], dtype=torch.int64)
    decode_position_ids = torch.full((batch, 1), seq, dtype=torch.int32)
    decode_attention_mask = torch.zeros((batch, 1, 1, seq + 1), dtype=torch.float32)
    current_token_tensor = torch.tensor([[current_token_id]], dtype=torch.long)
    return ReasonerRuntimeInputs(
        input_ids=input_ids,
        image_mask=image_mask,
        position_ids=position_ids,
        prefill_attention_mask=prefill_attention_mask,
        pixel_values=pixel_values,
        grid_thw=grid_thw,
        current_token_id=current_token_tensor,
        decode_position_ids=decode_position_ids,
        decode_attention_mask=decode_attention_mask,
    )


class ReasonerHmonnxRuntime:
    def __init__(
        self,
        paths: ReasonerHmonnxPaths,
        *,
        device: str = "cuda",
        input_dtype: str = "float16",
        debug: bool = False,
        init_xhquant: bool = True,
    ) -> None:
        require_paths(paths.all())
        if init_xhquant:
            xhquant_init(None, debug=debug)
        self.paths = paths
        self.device = torch.device(device if torch.cuda.is_available() or not device.startswith("cuda") else "cpu")
        self.hmonnx_dtype = torch.float16 if input_dtype == "float16" else torch.float32

        self.text_session = HMONNXInference(str(paths.text)).to(self.device)
        self.text_seq1_session = HMONNXInference(str(paths.text_seq1)).to(self.device)
        self.patch_session = HMONNXInference(str(paths.vision_patch)).to(self.device)
        self.core_session = HMONNXInference(str(paths.vision_core)).to(self.device)
        self.prefill_kv_session = HMONNXInference(str(paths.prefill_kv)).to(self.device)
        self.decode_session = HMONNXInference(str(paths.decode)).to(self.device)
        self.logits_prefill_session = HMONNXInference(str(paths.logits_prefill)).to(self.device)
        self.logits_seq1_session = HMONNXInference(str(paths.logits_seq1)).to(self.device)

    def encode_text(self, input_ids: torch.Tensor) -> torch.Tensor:
        return as_tuple(self.text_session.forward(input_ids.to(device=self.device, dtype=torch.int32)))[0].detach().cpu()

    def encode_current_token(self, token_id: torch.Tensor) -> torch.Tensor:
        return as_tuple(self.text_seq1_session.forward(token_id.to(device=self.device, dtype=torch.int32)))[0]

    def encode_vision(self, pixel_values: torch.Tensor, image_tokens: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        patch = as_tuple(self.patch_session.forward(pixel_values.to(device=self.device, dtype=self.hmonnx_dtype)))[0]
        image_full = as_tuple(self.core_session.forward(patch))[0].detach().cpu()
        image = image_full if image_tokens is None else image_full[:image_tokens]
        return image_full, image

    def prefill(self, fused_embeds: torch.Tensor, position_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
        outputs = as_tuple(
            self.prefill_kv_session.forward(
                fused_embeds.to(device=self.device, dtype=self.hmonnx_dtype),
                position_ids.to(device=self.device, dtype=torch.int32),
                attention_mask.to(device=self.device, dtype=self.hmonnx_dtype),
            )
        )
        hidden_device = outputs[0]
        logits = as_tuple(self.logits_prefill_session.forward(hidden_device.to(device=self.device, dtype=self.hmonnx_dtype)))[0].detach().cpu()
        return hidden_device.detach().cpu(), tuple(item.detach().cpu() for item in outputs[1:]), logits

    def decode_step(
        self,
        token_id: torch.Tensor,
        kv: tuple[torch.Tensor, ...],
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current_embed = self.encode_current_token(token_id)
        kv_device = tuple(item.to(device=self.device, dtype=self.hmonnx_dtype) for item in kv)
        outputs = as_tuple(
            self.decode_session.forward(
                current_embed.to(device=self.device, dtype=self.hmonnx_dtype),
                position_ids.to(device=self.device, dtype=torch.int32),
                attention_mask.to(device=self.device, dtype=self.hmonnx_dtype),
                *kv_device,
            )
        )
        hidden_device = outputs[0]
        logits = as_tuple(self.logits_seq1_session.forward(hidden_device.to(device=self.device, dtype=self.hmonnx_dtype)))[0].detach().cpu()
        return current_embed.detach().cpu(), hidden_device.detach().cpu(), logits

    def run_one_step(self, inputs: ReasonerRuntimeInputs, *, image_tokens: int) -> ReasonerRuntimeOutput:
        text_embeds = self.encode_text(inputs.input_ids)
        image_full, image = self.encode_vision(inputs.pixel_values, image_tokens=image_tokens)
        fused = fuse_text_image(text_embeds, image, inputs.image_mask)
        prefill_hidden, prefill_kv, prefill_logits = self.prefill(
            fused,
            inputs.position_ids,
            inputs.prefill_attention_mask,
        )
        current_embed, decode_hidden, decode_logits = self.decode_step(
            inputs.current_token_id,
            prefill_kv,
            inputs.decode_position_ids,
            inputs.decode_attention_mask,
        )
        return ReasonerRuntimeOutput(
            text_embeds=text_embeds,
            image_embeds_full=image_full,
            image_embeds=image,
            fused_embeds=fused,
            prefill_hidden=prefill_hidden,
            prefill_kv=prefill_kv,
            prefill_logits=prefill_logits,
            current_embed=current_embed,
            decode_hidden=decode_hidden,
            decode_logits=decode_logits,
        )
