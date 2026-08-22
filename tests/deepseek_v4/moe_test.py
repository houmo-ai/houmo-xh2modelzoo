from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

import xhquant.nn as xhnn
from xhmodel_merak.xh_llm.models.deepseek_v4.moe import (
    DeepSeekV4Router,
    ExactClampedSwiGLURoutedExperts,
)


def test_score_router_matches_sqrt_softplus_topk_with_correction() -> None:
    generator = torch.Generator().manual_seed(83)
    router = DeepSeekV4Router(
        hidden_size=4,
        num_experts=5,
        top_k=2,
        routed_scaling_factor=1.5,
        is_hash=False,
    )
    router.weight.data.copy_(torch.randn(5, 4, generator=generator, dtype=torch.float16))
    router.correction_bias.copy_(torch.randn(5, generator=generator, dtype=torch.float16))
    hidden = torch.randn(1, 3, 4, generator=generator, dtype=torch.float16)

    actual = router(hidden)
    scores = torch.sqrt(F.softplus(F.linear(hidden, router.weight)))
    expected_indices = torch.topk(
        scores + router.correction_bias,
        2,
        dim=-1,
        sorted=True,
    ).indices

    assert torch.equal(actual.indices, expected_indices)
    assert actual.weights.shape == (1, 3, 5)
    torch.testing.assert_close(actual.weights, scores)


def test_hash_router_uses_tid2eid_but_learned_weights() -> None:
    generator = torch.Generator().manual_seed(89)
    router = DeepSeekV4Router(
        hidden_size=4,
        num_experts=5,
        top_k=2,
        routed_scaling_factor=1.5,
        is_hash=True,
        vocab_size=7,
    )
    router.weight.data.copy_(torch.randn(5, 4, generator=generator, dtype=torch.float16))
    router.tid2eid.copy_(torch.tensor([[0, 1], [2, 4], [3, 1], [0, 4], [2, 3], [1, 4], [3, 0]]))
    hidden = torch.randn(1, 3, 4, generator=generator, dtype=torch.float16)
    # Deployment ABI is INT32; the hash lookup must consume it directly
    # without an exported Cast adapter.
    input_ids = torch.tensor([[1, 5, 2]], dtype=torch.int32)

    actual = router(hidden, input_ids)
    expected_indices = router.tid2eid[input_ids]
    scores = torch.sqrt(F.softplus(F.linear(hidden, router.weight)))

    assert torch.equal(actual.indices, expected_indices)
    assert actual.weights.shape == (1, 3, 5)
    torch.testing.assert_close(actual.weights, scores)


def test_single_moe_block_uses_zero_up_shift_exactly() -> None:
    generator = torch.Generator().manual_seed(97)
    experts = ExactClampedSwiGLURoutedExperts(
        num_experts=3,
        hidden_size=4,
        intermediate_size=5,
        top_k=2,
        limit=1.25,
        fast_mode=False,
    )
    gate = torch.randn(3, 5, 4, generator=generator)
    up = torch.randn(3, 5, 4, generator=generator)
    down = torch.randn(3, 4, 5, generator=generator)
    experts.bind_stacked_weights(gate, up, down)
    assert experts.block.up_shift == 0.0
    assert not hasattr(experts, "shifted")
    assert not hasattr(experts, "shift")
    hidden = torch.randn(1, 3, 4, generator=generator)
    selected = torch.tensor([[[0, 2], [1, 0], [2, 1]]])
    routing = torch.tensor([[[0.9, 0.2, 0.6], [0.75, 0.4, 0.75], [0.8, 0.3, 1.2]]])

    actual = experts(hidden, routing, selected)
    expected = torch.zeros_like(hidden)
    for token in range(hidden.shape[1]):
        for slot in range(selected.shape[-1]):
            expert = selected[0, token, slot].item()
            x = hidden[0, token]
            gate_value = F.linear(x, gate[expert]).clamp(max=1.25)
            up_value = F.linear(x, up[expert]).clamp(min=-1.25, max=1.25)
            value = F.linear(F.silu(gate_value) * up_value, down[expert])
            selected_weights = routing[0, token, selected[0, token]]
            selected_weights = selected_weights / selected_weights.sum()
            expected[0, token] += value * selected_weights[slot]

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


class _DefusedExpert(nn.Module):
    def __init__(self, offset: int) -> None:
        super().__init__()
        for name in ("gate_proj", "up_proj"):
            linear = nn.Linear(3, 2, bias=False, dtype=torch.float16)
            with torch.no_grad():
                linear.weight.copy_(torch.arange(6, dtype=torch.float16).view(2, 3) + offset)
            linear.register_buffer(
                "quant_weight",
                torch.tensor(
                    [[-8 + offset, -1, 0], [1, 6, 7 - offset]],
                    dtype=torch.int8,
                ),
            )
            setattr(self, name, linear)
        self.down_proj = nn.Linear(2, 3, bias=False, dtype=torch.float16)
        with torch.no_grad():
            self.down_proj.weight.copy_(torch.arange(6, dtype=torch.float16).view(3, 2) + offset)
        self.down_proj.register_buffer(
            "quant_weight",
            torch.tensor(
                [[-8 + offset, -1], [0, 1], [6, 7 - offset]],
                dtype=torch.int8,
            ),
        )


class _DefusedExpertsWrapper(nn.Module):
    """Match GPTQModel's numeric children plus non-expert act_fn layout."""

    def __init__(self) -> None:
        super().__init__()
        self.act_fn = nn.SiLU()
        self.add_module("0", _DefusedExpert(0))
        self.add_module("1", _DefusedExpert(1))


def test_from_hf_preserves_routed_expert_w4_integer_weights() -> None:
    source = nn.ModuleList([_DefusedExpert(0), _DefusedExpert(1)])
    expected = {
        name: torch.stack([getattr(expert, name).quant_weight.clone() for expert in source])
        for name in ("gate_proj", "up_proj", "down_proj")
    }

    packed = ExactClampedSwiGLURoutedExperts.from_hf(
        source,
        top_k=2,
        limit=1.25,
        fast_mode=False,
    )

    assert packed.block.up_shift == 0.0
    for name, value in expected.items():
        actual = getattr(packed.block, f"expert_{name}_quant_weight")
        assert actual.dtype == torch.int8
        torch.testing.assert_close(actual, value)
    for expert in source:
        for name in ("gate_proj", "up_proj", "down_proj"):
            assert getattr(expert, name).weight.device.type == "meta"
            assert getattr(expert, name).quant_weight.device.type == "meta"

    decode = ExactClampedSwiGLURoutedExperts.from_hf(
        source,
        top_k=2,
        limit=1.25,
        fast_mode=False,
    )
    for name in ("gate_proj", "up_proj", "down_proj"):
        weight_name = f"expert_{name}_weight"
        quant_weight_name = f"expert_{name}_quant_weight"
        assert getattr(decode.block, weight_name) is getattr(packed.block, weight_name)
        assert getattr(decode.block, quant_weight_name) is getattr(
            packed.block,
            quant_weight_name,
        )


def test_from_hf_accepts_gptqmodel_numeric_expert_children() -> None:
    source = _DefusedExpertsWrapper()

    packed = ExactClampedSwiGLURoutedExperts.from_hf(
        source,
        top_k=2,
        limit=1.25,
        fast_mode=False,
    )

    assert packed.num_experts == 2
    assert packed.block.expert_gate_proj_weight.shape == (2, 2, 3)
    assert packed.block.expert_gate_proj_quant_weight.min().item() == -8
    assert packed.block.expert_gate_proj_quant_weight.max().item() == 7


def _gptq_packed_weight(offset: int) -> xhnn.GPTQPackedWeight:
    qweight = torch.zeros(8, 64, dtype=torch.int32)
    qweight.view(-1)[0] = offset
    return xhnn.GPTQPackedWeight(
        qweight,
        torch.zeros(1, 8, dtype=torch.int32),
        torch.full((1, 64), 0.25 + offset, dtype=torch.float16),
        torch.zeros(64, dtype=torch.int32),
        bits=4,
        group_size=64,
        in_features=64,
        out_features=64,
        pack_dtype_bits=32,
        qzero_format=2,
        sym=True,
        desc_act=False,
    )


class _PackedExpert(nn.Module):
    def __init__(self, offset: int) -> None:
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            setattr(self, name, xhnn.GPTQPackedLinear(_gptq_packed_weight(offset)))


def test_from_hf_preserves_and_shares_gptq_packed_routed_expert_state() -> None:
    source = nn.ModuleList([_PackedExpert(0), _PackedExpert(1)])
    source_gate = source[0].gate_proj.packed_weight
    source_words = source_gate.qweight

    prefill = ExactClampedSwiGLURoutedExperts.from_hf(
        source,
        top_k=2,
        limit=1.25,
        fast_mode=False,
    )

    assert isinstance(prefill.block, xhnn.GPTQPackedMoeBlock)
    assert prefill.block.up_shift == 0.0
    assert prefill.block.expert_gate_proj_weight is None
    assert prefill.block.expert_gate_proj_packed[0] is source_gate
    assert prefill.block.expert_gate_proj_packed[0].qweight is source_words

    decode = ExactClampedSwiGLURoutedExperts.from_hf(
        source,
        top_k=2,
        limit=1.25,
        fast_mode=False,
    )
    assert isinstance(decode.block, xhnn.GPTQPackedMoeBlock)
    assert decode.block.expert_gate_proj_packed[0] is (prefill.block.expert_gate_proj_packed[0])
    assert decode.block.expert_up_proj_packed[1] is (prefill.block.expert_up_proj_packed[1])
