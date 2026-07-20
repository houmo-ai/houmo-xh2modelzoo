from __future__ import annotations

import sys
from types import ModuleType

import torch
import torch.nn as nn
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeMLP,
    Qwen3_5MoeSparseMoeBlock,
)

from xhmodel_merak.xh_llm._dequant_converter import restore_autoround_qwen3_5_moe_sparse_block
from xhmodel_merak.xh_llm.base_model import XHBaseModel


class LinearQwen3_5MoeSparseMoeBlock(nn.Module):  # noqa: N801
    def __init__(self, config: Qwen3_5MoeTextConfig):
        super().__init__()
        original = Qwen3_5MoeSparseMoeBlock(config)
        self.gate = original.gate
        self.experts = nn.ModuleList(
            [Qwen3_5MoeMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(config.num_experts)]
        )
        self.shared_expert = original.shared_expert
        self.shared_expert_gate = original.shared_expert_gate


def _make_text_config() -> Qwen3_5MoeTextConfig:
    return Qwen3_5MoeTextConfig(
        hidden_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        moe_intermediate_size=3,
        shared_expert_intermediate_size=3,
        num_experts=2,
        num_experts_per_tok=1,
        layer_types=["full_attention"],
    )


def test_restore_autoround_qwen3_5_moe_sparse_block_packs_expert_weights():
    config = _make_text_config()
    fused_block = LinearQwen3_5MoeSparseMoeBlock(config).eval()

    with torch.no_grad():
        for expert_idx, expert in enumerate(fused_block.experts):
            expert.gate_proj.weight.fill_(expert_idx + 1)
            expert.up_proj.weight.fill_(expert_idx + 11)
            expert.down_proj.weight.fill_(expert_idx + 21)

    restored = restore_autoround_qwen3_5_moe_sparse_block(fused_block)

    assert type(restored) is Qwen3_5MoeSparseMoeBlock
    assert restored.training is False
    assert restored.gate is fused_block.gate
    assert restored.shared_expert is fused_block.shared_expert
    assert restored.shared_expert_gate is fused_block.shared_expert_gate
    for expert_idx, expert in enumerate(fused_block.experts):
        expected_gate_up = torch.cat((expert.gate_proj.weight, expert.up_proj.weight), dim=0)
        torch.testing.assert_close(restored.experts.gate_up_proj[expert_idx], expected_gate_up)
        torch.testing.assert_close(restored.experts.down_proj[expert_idx], expert.down_proj.weight)
    assert restored(torch.randn(1, 2, config.hidden_size)).shape == (1, 2, config.hidden_size)


def test_dequantize_autoround_hf_model_restores_fused_qwen3_5_moe_block(monkeypatch):
    config = _make_text_config()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.layer = nn.ModuleDict({"mlp": LinearQwen3_5MoeSparseMoeBlock(config)})
            self.quantization_method = "autoround"
            self._is_hf_initialized = True

    fake_weight_handler = ModuleType("auto_round.utils.weight_handler")
    fake_weight_handler.check_and_mark_quantized_module = lambda model: []
    fake_weight_handler.convert_module_to_hp_if_necessary = lambda model, **kwargs: model
    monkeypatch.setitem(sys.modules, "auto_round.utils.weight_handler", fake_weight_handler)

    model = FakeModel().eval()
    out = XHBaseModel._dequantize_autoround_hf_model(model)

    assert out is model
    assert type(model.layer["mlp"]) is Qwen3_5MoeSparseMoeBlock
    assert model.layer["mlp"].training is False
    assert model.quantization_method is None
    assert model._is_hf_initialized is False


def test_dequantize_autoround_supports_isolated_qwen3_5_moe_block(monkeypatch):
    config = _make_text_config()
    fused_block = LinearQwen3_5MoeSparseMoeBlock(config).eval()
    fake_weight_handler = ModuleType("auto_round.utils.weight_handler")
    fake_weight_handler.check_and_mark_quantized_module = lambda model: []
    fake_weight_handler.convert_module_to_hp_if_necessary = lambda model, **kwargs: model
    monkeypatch.setitem(sys.modules, "auto_round.utils.weight_handler", fake_weight_handler)

    restored = XHBaseModel.dequantize_hf_model(
        fused_block,
        {"quant_method": "auto-round"},
    )

    assert type(restored) is Qwen3_5MoeSparseMoeBlock
    assert restored.training is False
