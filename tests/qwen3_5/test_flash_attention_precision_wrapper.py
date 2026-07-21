import copy
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from xhmodel_merak.xh_llm.models.qwen3_5 import (
    _hybrid_gated_delta_net as shared_impl,
)
from xhmodel_merak.xh_llm.models.qwen3_5 import _llm_model_impl as impl
from xhmodel_merak.xh_llm.models.qwen3_5_moe import _moe_model as moe_impl
from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig


CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b"
QWEN_CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs_merak/workflows/xh2a/llm_models"


class _Cfg(dict):
    __getattr__ = dict.__getitem__


class _ReferenceFlashAttention(torch.nn.Module):
    """Return the canonical FlashAttention layout [B, H, S, D]."""

    def __init__(self, *, scale, num_heads, num_kv_heads, **kwargs):
        super().__init__()
        del num_heads, kwargs
        self.scale = scale
        self.num_kv_heads = num_kv_heads

    def forward(self, query, key, value, **kwargs):
        del kwargs
        groups = query.shape[1] // self.num_kv_heads
        key = torch.repeat_interleave(key, groups, dim=1)
        value = torch.repeat_interleave(value, groups, dim=1)
        probabilities = torch.softmax(torch.matmul(query * self.scale, key.transpose(2, 3)), dim=-1)
        return torch.matmul(probabilities, value)


class _SoftmaxIgnoringPast(torch.nn.Module):
    def forward(self, scores, past_seq_length=None):
        del past_seq_length
        return torch.softmax(scores, dim=-1)


def _build_attention(monkeypatch, bits, *, enable=True):
    captured = {}

    class _CapturedFlashAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["args"] = args
            captured.update(kwargs)

    monkeypatch.setattr(shared_impl, "FlashAttention", _CapturedFlashAttention)
    attention = impl._Qwen3_5TextAttention.__new__(impl._Qwen3_5TextAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        num_key_value_heads=2,
        num_attention_heads=4,
        partial_rotary_factor=0.5,
    )
    attention.head_dim = 8
    attention.q_proj = torch.nn.Linear(1, 1, bias=False)
    cfg = _Cfg(
        enable_rope=False,
        use_cache=False,
        flash_attention={"enable": enable, **bits},
    )
    attention._setup(cfg)
    return attention, captured


def _build_moe_attention(monkeypatch, bits, *, enable=True):
    captured = {}

    class _CapturedFlashAttention(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            captured["args"] = args
            captured.update(kwargs)

    monkeypatch.setattr(shared_impl, "FlashAttention", _CapturedFlashAttention)
    attention = moe_impl._Qwen3_5MoeAttention.__new__(moe_impl._Qwen3_5MoeAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        num_key_value_heads=2,
        num_attention_heads=4,
        partial_rotary_factor=0.5,
    )
    attention.head_dim = 8
    attention.q_proj = torch.nn.Linear(1, 1, bias=False)
    cfg = _Cfg(enable_rope=False, use_cache=False, flash_attention={"enable": enable, **bits})
    attention._setup(cfg)
    return attention, captured


def _build_layout_attention(monkeypatch):
    monkeypatch.setattr(shared_impl, "FlashAttention", _ReferenceFlashAttention)
    attention = impl._Qwen3_5TextAttention.__new__(impl._Qwen3_5TextAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(
        num_key_value_heads=1,
        num_attention_heads=2,
        partial_rotary_factor=0.5,
    )
    attention.head_dim = 2
    attention.q_proj = torch.nn.Linear(4, 8, bias=False)
    attention.k_proj = torch.nn.Linear(4, 2, bias=False)
    attention.v_proj = torch.nn.Linear(4, 2, bias=False)
    attention.o_proj = torch.nn.Linear(4, 4, bias=False)
    attention.q_norm = torch.nn.Identity()
    attention.k_norm = torch.nn.Identity()
    attention._setup(
        _Cfg(
            enable_rope=False,
            use_cache=False,
            flash_attention={"enable": False},
        )
    )
    attention.masked_softmax = _SoftmaxIgnoringPast()
    return attention


@pytest.mark.parametrize("value", [8, 16])
def test_flash_attention_passes_all_five_precision_bits(monkeypatch, value):
    attention, captured = _build_attention(
        monkeypatch,
        {"q_bits": value, "k_bits": value, "v_bits": value, "s_bits": value, "p_bits": value},
    )

    assert attention.flash_q_bits == value
    assert attention.flash_k_bits == value
    assert attention.flash_v_bits == value
    assert attention.flash_s_bits == value
    assert attention.flash_p_bits == value
    assert captured["q_bits"] == value
    assert captured["k_bits"] == value
    assert captured["v_bits"] == value
    assert captured["s_bits"] == value
    assert captured["p_bits"] == value


def test_flash_attention_uses_keyword_only_public_api(monkeypatch):
    _, captured = _build_attention(monkeypatch, {})

    assert captured["args"] == ()
    assert captured["num_heads"] == 4
    assert captured["num_kv_heads"] == 2
    assert captured["is_causal"] is True


def test_flash_and_legacy_attention_preserve_same_bshd_wrapper_layout(monkeypatch):
    torch.manual_seed(0)
    legacy = _build_layout_attention(monkeypatch)
    flash = copy.deepcopy(legacy)
    flash.use_flash_attention = True
    flash.flash_attn = _ReferenceFlashAttention(
        num_heads=flash.num_heads,
        num_kv_heads=flash.num_key_value_heads,
        scale=1 / flash.head_dim**0.5,
    )
    hidden_states = torch.randn(1, 3, 4)

    legacy_output = legacy(hidden_states)[0]
    flash_output = flash(hidden_states)[0]

    assert legacy_output.shape == flash_output.shape == (1, 3, 4)
    torch.testing.assert_close(flash_output, legacy_output)


@pytest.mark.parametrize("field", ["q_bits", "k_bits", "v_bits", "s_bits", "p_bits"])
def test_flash_attention_rejects_invalid_precision_bit(monkeypatch, field):
    bits = {"q_bits": 8, "k_bits": 8, "v_bits": 8, "s_bits": 8, "p_bits": 8}
    bits[field] = 12

    with pytest.raises(ValueError, match=rf"{field}=12"):
        _build_attention(monkeypatch, bits)


@pytest.mark.parametrize(
    "bits",
    [
        {"q_bits": 8, "k_bits": 8, "v_bits": 8, "s_bits": 16, "p_bits": 8},
        {"q_bits": 8, "k_bits": 8, "v_bits": 8, "s_bits": 8, "p_bits": 16},
        {"q_bits": 16, "k_bits": 8, "v_bits": 16, "s_bits": 8, "p_bits": 16},
    ],
)
def test_flash_attention_keeps_s_and_p_precision_independent(monkeypatch, bits):
    attention, captured = _build_attention(monkeypatch, bits)

    assert attention.flash_s_bits == bits["s_bits"]
    assert attention.flash_p_bits == bits["p_bits"]
    assert captured["s_bits"] == bits["s_bits"]
    assert captured["p_bits"] == bits["p_bits"]


@pytest.mark.parametrize("values", product((8, 16), repeat=5))
def test_flash_attention_preserves_every_five_bit_combination(monkeypatch, values):
    fields = ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")
    bits = dict(zip(fields, values, strict=True))

    attention, captured = _build_attention(monkeypatch, bits)

    assert tuple(getattr(attention, f"flash_{field}") for field in fields) == values
    assert tuple(captured[field] for field in fields) == values


@pytest.mark.parametrize("values", product((8, 16), repeat=5))
def test_moe_flash_attention_preserves_every_five_bit_combination(monkeypatch, values):
    fields = ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")
    bits = dict(zip(fields, values, strict=True))
    _, captured = _build_moe_attention(monkeypatch, bits)
    assert tuple(captured[field] for field in fields) == values


def test_moe_flash_attention_rejects_invalid_precision(monkeypatch):
    with pytest.raises(ValueError, match="k_bits.*12"):
        _build_moe_attention(
            monkeypatch,
            {"q_bits": 8, "k_bits": 12, "v_bits": 8, "s_bits": 8, "p_bits": 8},
        )


def test_moe_flash_attention_defaults_all_precision_fields(monkeypatch):
    _, captured = _build_moe_attention(monkeypatch, {})
    assert tuple(captured[field] for field in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")) == (8,) * 5


def test_all16_config_resolves_to_five_bits(monkeypatch):
    workflow_cfg = WorkflowConfig.from_file(str(CONFIG_DIR / "qwen3_6_35b_a3b_full_fa_all16.yaml"))
    flash_cfg = workflow_cfg.build_export_dict()["model"]["flash_attention"]
    attention, captured = _build_attention(monkeypatch, flash_cfg)

    fields = ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")
    assert tuple(getattr(attention, f"flash_{field}") for field in fields) == (16,) * 5
    assert tuple(captured[field] for field in fields) == (16,) * 5


def test_all_qwen35_workflows_declare_five_precision_fields():
    fields = {"q_bits", "k_bits", "v_bits", "s_bits", "p_bits"}
    paths = sorted(QWEN_CONFIG_ROOT.glob("qwen3_5*/**/*.yaml"))
    flash_paths = []
    for path in paths:
        flash_cfg = WorkflowConfig.from_file(str(path)).build_export_dict()["model"].get("flash_attention")
        if flash_cfg is None:
            continue
        flash_paths.append(path)
        assert fields <= flash_cfg.keys(), path
    assert flash_paths
