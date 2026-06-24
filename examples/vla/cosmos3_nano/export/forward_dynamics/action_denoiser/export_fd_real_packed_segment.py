# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano forward dynamics segmented real-packed denoiser graphs."""

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

from common.official_cosmos3 import DEFAULT_OFFICIAL_TRANSFORMER, load_official_transformer  # noqa: E402
from export.common_quant.attention.export_transformer_rope import torch_dtype  # noqa: E402
from export.forward_dynamics.action_denoiser.export_fd_real_packed_boundary import (  # noqa: E402
    DEFAULT_ACTION_JSON,
    DEFAULT_MODEL_ROOT,
    graph_suffix,
)
from runtime.forward_dynamics_runtime import (  # noqa: E402
    OfficialCosmos3ForwardDynamicsInitSegmentWrapper,
    OfficialCosmos3ForwardDynamicsStackSegmentWrapper,
    load_forward_dynamics_action_chunk,
    make_forward_dynamics_real_packed_boundary_inputs,
)


DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "forward_dynamics_segments_smoke"


