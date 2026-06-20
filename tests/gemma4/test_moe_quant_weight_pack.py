import torch
from torch import nn

from xhquant.nn.modules.moeblock import MoeBlock
from xhmodel_merak.xh_llm.models.gemma4_series._llm_model_impl import _pack_defused_experts_to_moeblock


class _Expert(nn.Module):
    def __init__(self, offset: int):
        super().__init__()
        for name in ("gate_proj", "up_proj", "down_proj"):
            linear = nn.Linear(3, 2, bias=False)
            with torch.no_grad():
                linear.weight.copy_(torch.arange(6, dtype=torch.float16).view(2, 3) + offset)
            # GPTQModel dequantized Linear keeps 4-bit signed values in int8 quant_weight.
            linear.quant_weight = torch.tensor(
                [[-8 + offset, -1, 0], [1, 6, 7 - offset]],
                dtype=torch.int8,
            )
            setattr(self, name, linear)


class _Experts(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([_Expert(0), _Expert(1)])


def test_pack_defused_experts_preserves_gptq_4bit_quant_weight():
    moe_block = MoeBlock("gelu", 2, normalize_routing_weights=True)
    experts = _Experts()

    assert _pack_defused_experts_to_moeblock(moe_block, experts)

    for name in ("gate_proj", "up_proj", "down_proj"):
        packed_weight = getattr(moe_block, f"expert_{name}_weight")
        packed_quant_weight = getattr(moe_block, f"expert_{name}_quant_weight")

        assert tuple(packed_weight.shape) == (2, 2, 3)
        assert tuple(packed_quant_weight.shape) == (2, 2, 3)
        assert packed_quant_weight.dtype == torch.int8
        assert int(packed_quant_weight.min()) == -8
        assert int(packed_quant_weight.max()) == 7

    # The source dense weights are moved to meta to keep large MoE loads memory-safe.
    assert experts.experts[0].gate_proj.weight.device.type == "meta"
    assert experts.experts[0].gate_proj.quant_weight.device.type == "meta"
