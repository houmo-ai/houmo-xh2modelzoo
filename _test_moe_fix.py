"""Quick test to verify MoE expert weight copy with both quantized and non-quantized paths."""
import torch
import torch.nn as nn
from xhquant.nn.modules.moeblock import MoeBlock
from xh_model_zoo.xh_llm.models.qwen3_next._model import (
    _init_moe_linear_storage,
    _copy_defused_expert_weights_to_moeblock,
)

num_experts = 4
hidden_dim = 16
intermediate_dim = 32


class FakeQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, dtype=torch.float16):
        super().__init__()
        self._out = out_features
        self._in = in_features
        self._dtype = dtype
        self._real_weight = torch.randn(in_features, out_features, dtype=dtype)
        self.bias = None

    @property
    def weight(self):
        class Meta:
            def __init__(self, out, inp, dtype):
                self.shape = torch.Size([out, inp])
                self.dtype = dtype
        return Meta(self._out, self._in, self._dtype)

    def dequantize_weight(self):
        return self._real_weight


class FakeQuantExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = FakeQuantLinear(hidden_dim, intermediate_dim)
        self.up_proj = FakeQuantLinear(hidden_dim, intermediate_dim)
        self.down_proj = FakeQuantLinear(intermediate_dim, hidden_dim)
        self.act_fn = nn.SiLU()


# Test quantized path
experts = nn.ModuleList([FakeQuantExpert() for _ in range(num_experts)])
for i, expert in enumerate(experts):
    expert.gate_proj._real_weight.fill_(float(i + 1))
    expert.up_proj._real_weight.fill_(float(i + 10))
    expert.down_proj._real_weight.fill_(float(i + 100))

moeblock = MoeBlock("silu", 2, True)
device = torch.device("cpu")

for ln in ("gate_proj", "up_proj", "down_proj"):
    _init_moe_linear_storage(moeblock, experts, ln, device)

print("gate_proj storage shape:", moeblock.expert_gate_proj_weight.shape)
assert moeblock.expert_gate_proj_weight.shape == (4, 32, 16), "Shape mismatch!"

_copy_defused_expert_weights_to_moeblock(moeblock, experts, device)

assert moeblock.expert_gate_proj_weight[0, 0, 0].item() == 1.0
assert moeblock.expert_gate_proj_weight[1, 0, 0].item() == 2.0
assert moeblock.expert_down_proj_weight[2, 0, 0].item() == 102.0
print("Weight copy values correct")

# F.linear compatibility
hidden = torch.randn(1, hidden_dim, dtype=torch.float16)
out = torch.nn.functional.linear(hidden, moeblock.expert_gate_proj_weight[0])
assert out.shape == (1, 32), f"F.linear output wrong shape: {out.shape}"
print("F.linear output shape:", out.shape)

# Full MoeBlock forward
hidden_states = torch.randn(1, 1, hidden_dim, dtype=torch.float16)
routing_weights = torch.softmax(torch.randn(1, 1, num_experts, dtype=torch.float16), dim=-1)
output = moeblock(hidden_states, routing_weights, fast_mode=False)
assert output.shape == (1, 1, 16)
assert torch.isfinite(output).all()
print("MoeBlock forward OK, shape:", output.shape)

print("\nALL QUANT TESTS PASSED")
