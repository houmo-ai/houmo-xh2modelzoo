# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano forward dynamics real-packed denoiser boundary."""

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

from common.paths import default_model_root  # noqa: E402

from common.official_cosmos3 import DEFAULT_OFFICIAL_TRANSFORMER, load_official_transformer  # noqa: E402
from export.common_quant.attention.export_transformer_rope import torch_dtype  # noqa: E402
from runtime.forward_dynamics_runtime import (  # noqa: E402
    OfficialCosmos3ForwardDynamicsBoundaryWrapper,
    load_forward_dynamics_action_chunk,
    make_forward_dynamics_real_packed_boundary_inputs,
)


__all__ = [
    "DEFAULT_MODEL_ROOT",
    "DEFAULT_ACTION_JSON",
    "DEFAULT_OUTPUT_ROOT",
    "graph_suffix",
]


def graph_suffix(*parts: int) -> str:
    if not parts:
        return ""
    return "_" + "_".join(f"{int(item)}" for item in parts)


DEFAULT_MODEL_ROOT = default_model_root()
DEFAULT_ACTION_JSON = DEFAULT_MODEL_ROOT / "assets" / "example_action_fd_agibotworld_action_chunks.json"
DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "forward_dynamics_real_packed_boundary_2layers_smoke"
