# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Shared path defaults for Cosmos3-Nano tools."""

from __future__ import annotations

import os
from pathlib import Path


COSMOS3_NANO_MODEL_ROOT_ENV = "COSMOS3_NANO_MODEL_ROOT"
COSMOS_FRAMEWORK_ROOT_ENV = "COSMOS_FRAMEWORK_ROOT"

DEFAULT_MODEL_ROOT = Path(__file__).resolve().parents[1] / "data" / "Cosmos3-nano"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def workspace_root() -> Path:
    return project_root().parents[4]


def env_path(name: str, fallback: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else fallback


def default_model_root() -> Path:
    return env_path(COSMOS3_NANO_MODEL_ROOT_ENV, DEFAULT_MODEL_ROOT)


def default_official_transformer() -> Path:
    return default_model_root() / "transformer"


def default_sound_tokenizer_dir() -> Path:
    return default_model_root() / "sound_tokenizer"


def default_cosmos_framework_root() -> Path:
    return env_path(COSMOS_FRAMEWORK_ROOT_ENV, workspace_root() / "packages" / "cosmos-framework")


def default_diffusers_cosmos3_src() -> Path:
    return default_cosmos_framework_root() / "packages" / "diffusers-cosmos3"
