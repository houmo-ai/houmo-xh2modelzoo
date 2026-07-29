"""DFlash cache and attention export contracts.

The context graphs share one persistent cache with draft_decode.  Only the
FlashAttention-enabled draft_decode path may use the non-causal padding-full
cross-attention operator; the legacy explicit softmax path remains available.
"""

from __future__ import annotations

import inspect

import torch

from xhmodel_merak.xh_llm.models.qwen3_5 import _dflash_model_impl as impl
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_dflash_model import (
    _build_dflash_export_adapter,
)
from xhquant.core import CacheTensor


class _Core(torch.nn.Module):
    def forward_context(self, *args):
        return args

    def forward_decode(self, *args):
        return args[0]


def test_context_export_adapter_keeps_shared_cache_inputs():
    adapter = _build_dflash_export_adapter(
        _Core(),
        mode="context",
        num_hidden_layers=2,
    )

    assert list(inspect.signature(adapter.forward).parameters) == [
        "target_hidden",
        "past_seq_length",
        "current_input_length",
        "past_key_cache_0",
        "past_key_cache_1",
        "past_value_cache_0",
        "past_value_cache_1",
    ]


def test_flash_decode_export_adapter_omits_legacy_dense_mask():
    adapter = _build_dflash_export_adapter(
        _Core(),
        mode="decode",
        num_hidden_layers=2,
        use_flash_attention=True,
    )

    assert list(inspect.signature(adapter.forward).parameters) == [
        "noise_embedding",
        "past_seq_length",
        "current_input_length",
        "past_key_cache_0",
        "past_key_cache_1",
        "past_value_cache_0",
        "past_value_cache_1",
    ]


def test_nonflash_decode_export_adapter_keeps_legacy_dense_mask():
    adapter = _build_dflash_export_adapter(
        _Core(),
        mode="decode",
        num_hidden_layers=1,
        use_flash_attention=False,
    )

    assert list(inspect.signature(adapter.forward).parameters) == [
        "noise_embedding",
        "past_seq_length",
        "current_input_length",
        "attn_mask",
        "past_key_cache_0",
        "past_value_cache_0",
    ]


def _attention(*, flash_attention: bool) -> impl.DFlashCrossAttention:
    return impl.DFlashCrossAttention(
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        rms_norm_eps=1e-6,
        input_sequence_length=3,
        max_pe_length=16,
        rope_theta=10_000.0,
        use_cache=True,
        flash_attention={
            "enable": flash_attention,
            "q_bits": 8,
            "k_bits": 8,
            "v_bits": 8,
            "s_bits": 8,
            "p_bits": 8,
        },
    )


def _decode_inputs():
    return {
        "hidden_states": torch.randn(1, 3, 4),
        "past_seq_length": torch.tensor([5], dtype=torch.int64),
        "current_input_length": torch.tensor([3], dtype=torch.int64),
        "target_key_cache": CacheTensor(torch.zeros(1, 1, 16, 2)),
        "target_value_cache": CacheTensor(torch.zeros(1, 1, 16, 2)),
        "attn_mask": torch.zeros(1, 16),
    }


def test_flash_decode_is_padding_full_cross_attention(monkeypatch):
    captured: dict[str, object] = {}

    class _CapturedFlashAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["constructor"] = kwargs

        def forward(self, query, key, value, **kwargs):
            captured["forward"] = kwargs
            return query

    monkeypatch.setattr(impl.xhnn, "FlashAttention", _CapturedFlashAttention)
    attention = _attention(flash_attention=True)

    output = attention.forward_decode(**_decode_inputs())

    assert output.shape == (1, 3, 4)
    assert captured["constructor"]["is_causal"] is False
    assert torch.equal(
        captured["forward"]["current_input_length"],
        torch.tensor([3]),
    )
    assert torch.equal(
        captured["forward"]["kv_valid_length"],
        torch.tensor([8]),
    )


def test_non_flash_decode_keeps_explicit_softmax_path(monkeypatch):
    def _unexpected_flash_attention(*args, **kwargs):
        raise AssertionError("non-FA DFlash must not construct FlashAttention")

    monkeypatch.setattr(impl.xhnn, "FlashAttention", _unexpected_flash_attention)
    attention = _attention(flash_attention=False)

    output = attention.forward_decode(**_decode_inputs())

    assert output.shape == (1, 3, 4)


def test_dflash_layers_do_not_carry_page_conversion_tags():
    model = impl.DFlashModel(
        mode="decode",
        hidden_size=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        intermediate_size=8,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
        num_target_layers=1,
        target_layer_ids=[0],
        vocab_size=16,
        input_sequence_length=3,
        max_pe_length=16,
        max_sequence_length=16,
        flash_attention={"enable": False},
    )

    assert not hasattr(model, "layer_tags")
    assert all(layer.self_attn.use_cache for layer in model.layers)
