# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any


def enable_normalize_force_fp32(quant_config: Any) -> Any:
    quant_config.ops_cfg["Normalize"] = {"force_fp32": True}
    return quant_config
