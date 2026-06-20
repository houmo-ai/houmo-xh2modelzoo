from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn

from xhmodel_merak.xh_llm.models.gemma4.gemma4_visual_model import (
    _make_vision_attn_traceable as _make_legacy_vision_attn_traceable,
)
from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_vision_model import (
    _make_vision_attn_traceable as _make_series_vision_attn_traceable,
)


class _IdentityRope(nn.Module):
    def forward(self, x, *position_embeddings):  # noqa: D401 - tiny test helper
        return x


class _RecordingMaskedAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, mask):
        self.calls += 1
        return x + mask + mask


def _build_fake_attention() -> nn.Module:
    hidden_size = 2
    attn = nn.Module()
    attn.config = SimpleNamespace(num_attention_heads=1, num_key_value_heads=1)
    attn.head_dim = hidden_size
    attn.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    attn.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    attn.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
    attn.o_proj = nn.Identity()
    attn.q_norm = nn.Identity()
    attn.k_norm = nn.Identity()
    attn.v_norm = nn.Identity()
    return attn


def test_gemma4_series_visual_attention_uses_xhquant_masked_add_for_attention_mask():
    """visual/video_visual attention masks must lower to xhquant MaskAdd.

    The compiler relies on MaskAdd's double-add cleanup semantics instead of a
    plain torch Add before Softmax.
    """

    hidden_size = 2
    attn = _build_fake_attention()

    _make_series_vision_attn_traceable(attn)
    recorder = _RecordingMaskedAdd()
    attn.masked_add = recorder
    attn._fused_multidim_rope = _IdentityRope()

    hidden_states = torch.randn(1, 3, hidden_size)
    attention_mask = torch.full((1, 1, 1, 3), -1000.0)
    position_embeddings = tuple(torch.ones(1, 3, 1, 1) for _ in range(4))

    attn(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask)

    assert recorder.calls == 1


def test_legacy_gemma4_visual_attention_uses_xhquant_masked_add_for_attention_mask():
    """The E4B legacy video/image wrapper imports the legacy Gemma4 helper."""

    hidden_size = 2
    attn = _build_fake_attention()

    _make_legacy_vision_attn_traceable(attn)
    recorder = _RecordingMaskedAdd()
    attn.masked_add = recorder
    attn._fused_multidim_rope = _IdentityRope()

    hidden_states = torch.randn(1, 3, hidden_size)
    attention_mask = torch.full((1, 1, 1, 3), -1000.0)
    position_embeddings = tuple(torch.ones(1, 3, hidden_size) for _ in range(2))

    attn(hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask)

    assert recorder.calls == 1
