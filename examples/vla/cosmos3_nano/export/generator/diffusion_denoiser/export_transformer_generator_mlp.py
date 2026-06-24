# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export one Cosmos3-Nano generator MoE MLP block from safetensors."""

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

from export.common_quant.linear.export_transformer_linear import DEFAULT_MODEL, load_json, load_tensor  # noqa: E402
from export.common_quant.mlp.export_transformer_text_mlp import (  # noqa: E402
    TransformerTextMLPWrapper,
    tensor_summary,
    torch_dtype,
)


DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "transformer_generator_mlp_layer0_smoke"


def load_generator_mlp(model_dir: Path, layer: int) -> tuple[TransformerTextMLPWrapper, dict[str, Any]]:
    transformer_dir = model_dir / "transformer"
    config = load_json(transformer_dir / "config.json")
    index = load_json(transformer_dir / "diffusion_pytorch_model.safetensors.index.json")
    weight_map = index["weight_map"]
    prefix = f"layers.{layer}"
    keys = {
        "norm": f"{prefix}.post_attention_layernorm_moe_gen.weight",
        "gate": f"{prefix}.mlp_moe_gen.gate_proj.weight",
        "up": f"{prefix}.mlp_moe_gen.up_proj.weight",
        "down": f"{prefix}.mlp_moe_gen.down_proj.weight",
    }
    tensors = {name: load_tensor(transformer_dir, weight_map, key) for name, key in keys.items()}
    wrapper = TransformerTextMLPWrapper(
        tensors["norm"],
        tensors["gate"],
        tensors["up"],
        tensors["down"],
        eps=float(config.get("rms_norm_eps", 1e-6)),
    )
    meta = {
        "keys": keys,
        "shapes": {name: list(tensor.shape) for name, tensor in tensors.items()},
        "dtypes": {name: str(tensor.dtype) for name, tensor in tensors.items()},
        "rms_norm_eps": float(config.get("rms_norm_eps", 1e-6)),
        "hidden_size": int(config.get("hidden_size", tensors["norm"].numel())),
        "intermediate_size": int(config.get("intermediate_size", tensors["gate"].shape[0])),
    }
    return wrapper, meta


