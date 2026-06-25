# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Shared test fixtures for the streaming pipeline tests."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# Ensure tests can import modules both as package modules and by the historical
# top-level names used in this directory's tests.
_STREAM_DIR = Path(__file__).resolve().parent.parent
_TESTS_DIR = _STREAM_DIR / "tests"
_REPO_ROOT = _STREAM_DIR.parents[3]
for _path in (str(_TESTS_DIR), str(_REPO_ROOT), str(_STREAM_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)


def pytest_configure(config):
    import importlib

    namespace_paths = {
        "examples": _REPO_ROOT / "examples",
        "examples.llm": _REPO_ROOT / "examples" / "llm",
        "examples.llm.qwen3omni": _REPO_ROOT / "examples" / "llm" / "qwen3omni",
        "examples.llm.qwen3omni.stream": _STREAM_DIR,
    }
    for module_name, module_path in namespace_paths.items():
        module = sys.modules.get(module_name)
        if module is None:
            module = types.ModuleType(module_name)
            sys.modules[module_name] = module
        module.__path__ = [str(module_path)]

    package = "examples.llm.qwen3omni.stream"
    for name in (
        "events",
        "code2wav_stage",
        "connectors",
        "thinker_stage",
        "talker_stage",
        "orchestrator",
        "_utils",
        "qwen3_omni_hmonnx_generate_stream",
    ):
        sys.modules.setdefault(name, importlib.import_module(f"{package}.{name}"))

    sys.modules["conftest"] = sys.modules[__name__]


# ---------------------------------------------------------------------------
# Fake HMONNX session
# ---------------------------------------------------------------------------


class FakeHMONNXSession:
    """Minimal mock for HMONNXInference sessions.

    Returns deterministic outputs based on input shapes.
    """

    def __init__(self, output_shapes=None, return_logits=True, return_hidden=True):
        self.inputs = [SimpleNamespace(name=f"input_{i}") for i in range(10)]
        self.output_shapes = output_shapes
        self.return_logits = return_logits
        self.return_hidden = return_hidden
        self.call_count = 0

    def forward(self, *args, **kwargs):
        self.call_count += 1
        # Determine seq_len from the first tensor argument
        seq_len = 1
        hidden_size = 64
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dim() >= 2:
                seq_len = max(seq_len, int(arg.shape[1]))
                hidden_size = int(arg.shape[-1])
                break

        vocab_size = 128
        outputs = []
        if self.return_logits:
            logits = torch.randn(1, seq_len, vocab_size, dtype=torch.float32)
            outputs.append(logits)
        if self.return_hidden:
            hidden = torch.randn(1, seq_len, hidden_size, dtype=torch.float16)
            outputs.append(hidden)

        if len(outputs) == 1:
            return outputs[0]
        return outputs


# ---------------------------------------------------------------------------
# Fake Code2Wav
# ---------------------------------------------------------------------------


class FakeCode2Wav:
    """Mock for code2wav module."""

    def __init__(self, total_upsample=600):
        self.total_upsample = total_upsample
        self.hmonnx = FakeHMONNXSession(return_logits=False, return_hidden=False)
        self.hmonnx_max_code_len = 512

    def forward(self, codes):
        if isinstance(codes, torch.Tensor):
            code_len = int(codes.shape[-1])
        else:
            code_len = 1
        wav_len = code_len * self.total_upsample
        return torch.randn(1, wav_len, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_kv_cache_info():
    return {
        "shape": [1, 4, 2048, 128],
        "num_decoder_layers": 4,
    }


@pytest.fixture
def fake_predictor_kv_cache_info():
    return {
        "shape": [1, 2, 256, 128],
        "num_decoder_layers": 2,
    }


@pytest.fixture
def hidden_size():
    return 64


@pytest.fixture
def vocab_size():
    return 128


@pytest.fixture
def fake_token_embedding(vocab_size, hidden_size):
    emb = torch.nn.Embedding(vocab_size, hidden_size)
    emb.eval()
    return emb
