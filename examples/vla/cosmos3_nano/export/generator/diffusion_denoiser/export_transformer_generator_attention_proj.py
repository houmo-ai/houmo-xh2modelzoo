# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export one Cosmos3-Nano generator added-attention projection probe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch


_COSMOS3_ROOT = Path(__file__).resolve().parents[3]
if str(_COSMOS3_ROOT) not in sys.path:
    sys.path.insert(0, str(_COSMOS3_ROOT))

from export.common_quant.attention.export_transformer_attention_proj import (  # noqa: E402
    TransformerAttentionProjWrapper,
    tensor_summary,
    torch_dtype,
)
from export.common_quant.linear.export_transformer_linear import DEFAULT_MODEL, load_json, load_tensor  # noqa: E402


DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "transformer_generator_attention_proj_layer0_smoke"


def load_generator_attention_proj(model_dir: Path, layer: int, include_qk_norm: bool) -> tuple[TransformerAttentionProjWrapper, dict[str, Any]]:
    transformer_dir = model_dir / "transformer"
    config = load_json(transformer_dir / "config.json")
    index = load_json(transformer_dir / "diffusion_pytorch_model.safetensors.index.json")
    weight_map = index["weight_map"]
    prefix = f"layers.{layer}"
    keys = {
        "input_norm": f"{prefix}.input_layernorm_moe_gen.weight",
        "q": f"{prefix}.self_attn.add_q_proj.weight",
        "k": f"{prefix}.self_attn.add_k_proj.weight",
        "v": f"{prefix}.self_attn.add_v_proj.weight",
        "o": f"{prefix}.self_attn.to_add_out.weight",
        "norm_q": f"{prefix}.self_attn.norm_added_q.weight",
        "norm_k": f"{prefix}.self_attn.norm_added_k.weight",
    }
    tensors = {name: load_tensor(transformer_dir, weight_map, key) for name, key in keys.items()}
    num_attention_heads = int(config["num_attention_heads"])
    num_key_value_heads = int(config["num_key_value_heads"])
    head_dim = int(config["head_dim"])
    wrapper = TransformerAttentionProjWrapper(
        tensors["input_norm"],
        tensors["q"],
        tensors["k"],
        tensors["v"],
        tensors["o"],
        tensors["norm_q"],
        tensors["norm_k"],
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        eps=float(config.get("rms_norm_eps", 1e-6)),
        include_qk_norm=include_qk_norm,
    )
    meta = {
        "keys": keys,
        "shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "dtypes": {name: str(tensor.dtype) for name, tensor in tensors.items()},
        "rms_norm_eps": float(config.get("rms_norm_eps", 1e-6)),
        "hidden_size": int(config["hidden_size"]),
        "num_attention_heads": num_attention_heads,
        "num_key_value_heads": num_key_value_heads,
        "head_dim": head_dim,
        "include_qk_norm": include_qk_norm,
    }
    return wrapper, meta


