# Copyright 2026 HOUMO AI
#
# File: test_minicpm_v_4_5_vision_pos_embed.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for resampler pos-embed construction beyond the base grid.

The official MiniCPM-V-4.5 Resampler dynamically extends its 2D sincos cache
when a slice grid exceeds the default 70x70 cache (``_adjust_pos_cache``).
The runtime must reproduce this: OCRBench and other real datasets contain
images whose dynamic slices produce grids wider/taller than the base cache.
"""

from __future__ import annotations

import pytest
import torch

from xhmodel_merak.xh_llm.models.minicpm_v_4_5.vision import (
    build_resampler_pos_embed,
    build_resampler_pos_embed_cache,
)


@pytest.fixture(scope="module")
def base_cache() -> torch.Tensor:
    return build_resampler_pos_embed_cache(embed_dim=4096, max_side=70)


def test_pos_embed_within_base_grid_shape():
    cache = build_resampler_pos_embed_cache(embed_dim=4096, max_side=70)
    out = build_resampler_pos_embed((32, 32), 1600, cache, torch.float16)
    assert tuple(out.shape) == (1, 1600, 4096)
    assert out.dtype == torch.float16
    assert bool(torch.isfinite(out).all())


def test_pos_embed_wider_than_base_cache():
    cache = build_resampler_pos_embed_cache(embed_dim=4096, max_side=70)
    out = build_resampler_pos_embed((79, 13), 1600, cache, torch.float32)
    assert tuple(out.shape) == (1, 1600, 4096)
    valid = out[0, : 79 * 13, :]
    assert bool(torch.isfinite(valid).all())
    assert not bool(torch.equal(valid, torch.zeros_like(valid)))


def test_pos_embed_taller_than_base_cache():
    cache = build_resampler_pos_embed_cache(embed_dim=4096, max_side=70)
    out = build_resampler_pos_embed((13, 79), 1600, cache, torch.float32)
    assert tuple(out.shape) == (1, 1600, 4096)
    valid = out[0, : 13 * 79, :]
    assert bool(torch.isfinite(valid).all())
    assert not bool(torch.equal(valid, torch.zeros_like(valid)))


def test_pos_embed_matches_direct_sincos_for_oversized_grid():
    cache = build_resampler_pos_embed_cache(embed_dim=256, max_side=70)
    target = (79, 13)
    out = build_resampler_pos_embed(target, 1600, cache, torch.float32)
    direct = build_resampler_pos_embed_cache(embed_dim=256, max_side=79)
    reference = build_resampler_pos_embed(target, 1600, direct, torch.float32)
    assert torch.allclose(out, reference, atol=1e-5, rtol=1e-5)


def test_pos_embed_oversized_grid_within_capacity_raises():
    cache = build_resampler_pos_embed_cache(embed_dim=4096, max_side=70)
    with pytest.raises(ValueError):
        build_resampler_pos_embed((90, 90), 1600, cache, torch.float32)
