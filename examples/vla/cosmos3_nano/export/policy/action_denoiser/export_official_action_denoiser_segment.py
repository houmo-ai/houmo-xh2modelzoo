# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano policy action denoiser segmented backbone graphs."""

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
from runtime.policy_runtime import (  # noqa: E402
    OfficialCosmos3ActionDenoiserInitSegmentWrapper,
    OfficialCosmos3ActionDenoiserStackSegmentWrapper,
    make_policy_real_packed_action_boundary_inputs,
)

from export.common_quant.attention.export_transformer_rope import torch_dtype  # noqa: E402
from export.policy.action_denoiser.export_official_action_denoiser_boundary import (  # noqa: E402
    DEFAULT_MODEL_ROOT,
    graph_suffix,
)


DEFAULT_OUTPUT_ROOT = _COSMOS3_ROOT / "data" / "policy_action_segments_smoke"


