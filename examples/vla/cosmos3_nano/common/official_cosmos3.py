# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Helpers for loading the official Cosmos3-Nano implementation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch


try:
    from common.paths import default_diffusers_cosmos3_src, default_official_transformer
except ModuleNotFoundError:
    from paths import default_diffusers_cosmos3_src, default_official_transformer


_OFFICIAL_SRC = default_diffusers_cosmos3_src()
DEFAULT_OFFICIAL_TRANSFORMER = default_official_transformer()


def ensure_official_cosmos3_importable() -> None:
    """Put the downloaded official diffusers-cosmos3 package on sys.path."""
    if not _OFFICIAL_SRC.exists():
        raise FileNotFoundError(
            f"Official Cosmos source is missing: {_OFFICIAL_SRC}. "
            "Set COSMOS_FRAMEWORK_ROOT or clone https://github.com/NVIDIA/cosmos-framework.git under <workspace>/packages first."
        )
    src = str(_OFFICIAL_SRC)
    if src not in sys.path:
        sys.path.insert(0, src)


def patch_default_rope_registry() -> None:
    """Install a compatibility shim for official Cosmos rope_type='default'."""
    import transformers.modeling_rope_utils as rope_utils

    def default_rope(config: Any, device: torch.device | None = None, seq_len: int | None = None, layer_type: str | None = None):
        del seq_len, layer_type
        if isinstance(config, dict):
            head_dim = int(config["head_dim"])
            theta = float(config["rope_theta"])
        else:
            head_dim = int(config.head_dim)
            theta = float(config.rope_theta)
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
        return inv_freq, 1.0

    rope_utils.ROPE_INIT_FUNCTIONS["default"] = default_rope


def import_official_transformer_class():
    ensure_official_cosmos3_importable()
    patch_default_rope_registry()
    import diffusers_cosmos3  # noqa: F401
    from diffusers_cosmos3 import Cosmos3OmniTransformer

    return Cosmos3OmniTransformer


def load_official_transformer(
    model_dir: Path = DEFAULT_OFFICIAL_TRANSFORMER,
    *,
    torch_dtype: torch.dtype = torch.float32,
    low_cpu_mem_usage: bool = True,
):
    Cosmos3OmniTransformer = import_official_transformer_class()
    model = Cosmos3OmniTransformer.from_pretrained(
        str(model_dir),
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )
    return model.eval()
