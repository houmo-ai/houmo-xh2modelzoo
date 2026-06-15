# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Shared default paths for VLA-JEPA examples.

The defaults are derived from the repository layout so the scripts are not tied
to one user's home directory. Environment variables can override the common
roots when running on another machine.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


VLA_JEPA_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VLA_JEPA_ROOT.parents[2]
WORKSPACE_ROOT = Path(os.environ.get("VLA_JEPA_WORKSPACE", REPO_ROOT.parents[1])).expanduser()

DEFAULT_MODEL = str(
    Path(os.environ.get("VLA_JEPA_MODEL", WORKSPACE_ROOT / "models/lerobot/VLA-JEPA-LIBERO")).expanduser()
)
DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("VLA_JEPA_OUTPUT_ROOT", WORKSPACE_ROOT / "outputs/vla_jepa")
).expanduser()
DEFAULT_LOG_ROOT = Path(
    os.environ.get("VLA_JEPA_LOG_ROOT", WORKSPACE_ROOT / "logs/vla_jepa_libero10_20eps")
).expanduser()
DEFAULT_DOCS_ROOT = Path(os.environ.get("VLA_JEPA_DOCS_ROOT", WORKSPACE_ROOT / "docs_vla")).expanduser()
DEFAULT_LIBERO_CONFIG_PATH = Path(
    os.environ.get("LIBERO_CONFIG_PATH", WORKSPACE_ROOT / ".libero")
).expanduser()
DEFAULT_PYTHON = os.environ.get("VLA_JEPA_PYTHON", sys.executable)


def output_path(*parts: str) -> Path:
    return DEFAULT_OUTPUT_ROOT.joinpath(*parts)


def output_str(*parts: str) -> str:
    return str(output_path(*parts))


def docs_str(*parts: str) -> str:
    return str(DEFAULT_DOCS_ROOT.joinpath(*parts))


def set_default_libero_config_path() -> None:
    os.environ.setdefault("LIBERO_CONFIG_PATH", str(DEFAULT_LIBERO_CONFIG_PATH))


DEFAULT_ACTION_HEAD_HMONNX_NAME = "vla_jepa_action_head_step_w8a8_sefp.hmonnx.onnx"
DEFAULT_VISUAL_ENCODER_HMONNX_NAME = "vla_jepa_qwen_visual_encoder_w8a8_sefp.hmonnx.onnx"
DEFAULT_CONTEXT_GRAPH_HMONNX_NAME = (
    "vla_jepa_context_graph_wrapper_float16_w8a16_sefp_linear_w16a16_sefp."
    "fused_rmsnorm.hmonnx.onnx"
)

DEFAULT_ACTION_HEAD_HMONNX = output_str("action_head", DEFAULT_ACTION_HEAD_HMONNX_NAME)
DEFAULT_STANDARD_ACTION_HEAD_HMONNX = output_str("standard_export", "action_head", DEFAULT_ACTION_HEAD_HMONNX_NAME)
DEFAULT_VISUAL_ENCODER_HMONNX = output_str(
    "context_encoder", "qwen_visual_encoder", DEFAULT_VISUAL_ENCODER_HMONNX_NAME
)
DEFAULT_CONTEXT_GRAPH_HMONNX = output_str(
    "standard_export_ctx256", "context", DEFAULT_CONTEXT_GRAPH_HMONNX_NAME
)
