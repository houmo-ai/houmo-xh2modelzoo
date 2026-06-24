# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export one Cosmos3-Nano reasoner text MLP block from safetensors."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys

_COSMOS3_ROOT = Path(__file__).resolve().parents[3]
if str(_COSMOS3_ROOT) not in sys.path:
    sys.path.insert(0, str(_COSMOS3_ROOT))

from export.common_quant.linear.export_transformer_linear import load_json, load_tensor
from export.common_quant.utils import tensor_summary, torch_dtype


__all__ = [
    "TransformerTextMLPWrapper",
    "load_text_mlp",
    "tensor_summary",
    "torch_dtype",
]


DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "transformer_text_mlp_layer0_smoke"


class TransformerTextMLPWrapper(nn.Module):
    def __init__(
        self,
        norm_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        up_weight: torch.Tensor,
        down_weight: torch.Tensor,
        eps: float,
    ):
        super().__init__()
        self.eps = float(eps)
        self.norm_weight = nn.Parameter(norm_weight.float().clone())
        self.gate_weight = nn.Parameter(gate_weight.float().clone())
        self.up_weight = nn.Parameter(up_weight.float().clone())
        self.down_weight = nn.Parameter(down_weight.float().clone())

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        normed = hidden_states.float() * torch.rsqrt(variance + self.eps)
        normed = normed.to(hidden_states.dtype) * self.norm_weight.to(hidden_states.dtype)
        gate = F.linear(normed, self.gate_weight)
        up = F.linear(normed, self.up_weight)
        return F.linear(F.silu(gate) * up, self.down_weight)


def _pick_key(weight_map: dict[str, str], candidates: list[str], *, description: str) -> str:
    for key in candidates:
        if key in weight_map:
            return key
    raise KeyError(f"Could not find {description} weight key in index. Tried: {candidates}")


def load_text_mlp(model_dir: Path, layer: int) -> tuple[TransformerTextMLPWrapper, dict[str, Any]]:
    transformer_dir = model_dir / "transformer"
    config = load_json(transformer_dir / "config.json")
    index = load_json(transformer_dir / "diffusion_pytorch_model.safetensors.index.json")
    weight_map = index["weight_map"]
    prefix = f"layers.{layer}"

    norm_key = _pick_key(
        weight_map,
        [
            f"{prefix}.post_attention_layernorm.weight",
            f"{prefix}.post_attention_layernorm_moe.weight",
            f"{prefix}.text_mlp.post_attention_layernorm.weight",
            f"{prefix}.mlp.layernorm.weight",
            f"{prefix}.mlp.input_layernorm.weight",
            f"{prefix}.input_layernorm.weight",
            f"{prefix}.layernorm.weight",
        ],
        description="norm",
    )
    gate_key = _pick_key(
        weight_map,
        [
            f"{prefix}.mlp.gate_proj.weight",
            f"{prefix}.text_mlp.gate_proj.weight",
            f"{prefix}.mlp.gate.weight",
            f"{prefix}.mlp.w1.weight",
            f"{prefix}.mlp.fc1.weight",
        ],
        description="gate",
    )
    up_key = _pick_key(
        weight_map,
        [
            f"{prefix}.mlp.up_proj.weight",
            f"{prefix}.text_mlp.up_proj.weight",
            f"{prefix}.mlp.w3.weight",
            f"{prefix}.mlp.up.weight",
        ],
        description="up",
    )
    down_key = _pick_key(
        weight_map,
        [
            f"{prefix}.mlp.down_proj.weight",
            f"{prefix}.text_mlp.down_proj.weight",
            f"{prefix}.mlp.w2.weight",
            f"{prefix}.mlp.down.weight",
        ],
        description="down",
    )

    tensors = {
        "norm": load_tensor(transformer_dir, weight_map, norm_key),
        "gate": load_tensor(transformer_dir, weight_map, gate_key),
        "up": load_tensor(transformer_dir, weight_map, up_key),
        "down": load_tensor(transformer_dir, weight_map, down_key),
    }
    wrapper = TransformerTextMLPWrapper(
        tensors["norm"],
        tensors["gate"],
        tensors["up"],
        tensors["down"],
        eps=float(config.get("rms_norm_eps", 1e-6)),
    )
    keys = {
        "norm": norm_key,
        "gate": gate_key,
        "up": up_key,
        "down": down_key,
    }
    meta = {
        "keys": keys,
        "shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "dtypes": {name: str(tensor.dtype) for name, tensor in tensors.items()},
        "rms_norm_eps": float(config.get("rms_norm_eps", 1e-6)),
        "hidden_size": int(config.get("hidden_size", tensors["norm"].numel())),
        "intermediate_size": int(config.get("intermediate_size", tensors["up"].shape[0])),
    }
    return wrapper, meta
