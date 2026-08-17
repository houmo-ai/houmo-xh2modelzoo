"""Compact shared-latent Q/K/V projection for DeepSeek-V4 attention."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from ._trace import is_fx_proxy
from .attention import InterleavedPartialRope
from .norm import build_xh_rms_norm, rms_norm_from_hf


class ProjectedQKV(NamedTuple):
    q_residual: Tensor
    query: Tensor
    latent_kv: Tensor


class StaticGroupedLinear(nn.Module):
    """Express V4's block-diagonal grouped projection as ordinary linears.

    XH2a has no quantization wrapper for Transformers'
    ``DeepseekV4GroupedLinear`` class.  Keeping one ``nn.Linear`` per group
    avoids a huge block-diagonal weight expansion and lets each slice follow
    the standard W8/W4 graph path.  Existing integer ``quant_weight`` buffers
    are sliced with the dense weights, so GPTQModel and AutoRound artifacts
    retain their exact integers.
    """

    def __init__(self, projections: list[nn.Linear]) -> None:
        super().__init__()
        if not projections:
            raise ValueError("at least one grouped projection is required")
        self.projections = nn.ModuleList(projections)
        self.num_groups = len(projections)
        self.in_features = int(projections[0].in_features)
        self.out_features_per_group = int(projections[0].out_features)

    @staticmethod
    def _linear_slice(source: nn.Module, start: int, end: int) -> nn.Linear:
        weight = source.weight[start:end]
        bias = None if source.bias is None else source.bias[start:end]
        linear = nn.Linear(
            int(weight.shape[1]),
            int(weight.shape[0]),
            bias=bias is not None,
            device="meta",
            dtype=weight.dtype,
        )
        linear.weight = nn.Parameter(weight, requires_grad=source.weight.requires_grad)
        if bias is not None:
            linear.bias = nn.Parameter(bias, requires_grad=source.bias.requires_grad)
        quant_weight = getattr(source, "quant_weight", None)
        if quant_weight is not None:
            persistent = "quant_weight" not in source._non_persistent_buffers_set
            linear.register_buffer(
                "quant_weight",
                quant_weight[start:end],
                persistent=persistent,
            )
        return linear

    @classmethod
    def from_hf(cls, source: nn.Module, *, num_groups: int) -> "StaticGroupedLinear":
        num_groups = int(num_groups)
        if num_groups <= 0 or source.weight.ndim != 2:
            raise ValueError("grouped source must have a rank-2 weight and positive group count")
        if int(source.weight.shape[0]) % num_groups:
            raise ValueError("grouped output rows must divide evenly across groups")
        width = int(source.weight.shape[0]) // num_groups
        return cls([cls._linear_slice(source, group * width, (group + 1) * width) for group in range(num_groups)])

    def forward(self, x: Tensor) -> Tensor:
        if not is_fx_proxy(x) and (x.ndim < 2 or x.shape[-2] != self.num_groups or x.shape[-1] != self.in_features):
            raise ValueError(f"grouped input must end in [{self.num_groups},{self.in_features}]")
        return torch.stack(
            tuple(projection(x[..., group, :]) for group, projection in enumerate(self.projections)),
            dim=-2,
        )


class DeepSeekV4QKVProjection(nn.Module):
    """Project Q and the one shared 512-wide KV latent without head expansion."""

    def __init__(
        self,
        *,
        hidden_size: int = 4096,
        q_lora_rank: int = 1024,
        num_heads: int = 64,
        head_dim: int = 512,
        rope_dim: int = 64,
        rms_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.q_lora_rank = int(q_lora_rank)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.rope_dim = int(rope_dim)
        self.partial_rope = InterleavedPartialRope(self.rope_dim)
        self.q_a_proj = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False, dtype=torch.float16)
        self.q_a_norm: nn.Module = build_xh_rms_norm(
            self.q_lora_rank,
            rms_norm_eps,
            weight=torch.ones(self.q_lora_rank, dtype=torch.float16),
        )
        self.q_b_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.head_dim,
            bias=False,
            dtype=torch.float16,
        )
        self.q_b_norm: nn.Module = build_xh_rms_norm(
            self.head_dim,
            rms_norm_eps,
        )
        self.kv_proj = nn.Linear(self.hidden_size, self.head_dim, bias=False, dtype=torch.float16)
        self.kv_norm: nn.Module = build_xh_rms_norm(
            self.head_dim,
            rms_norm_eps,
            weight=torch.ones(self.head_dim, dtype=torch.float16),
        )

    @classmethod
    def from_hf(cls, attention: nn.Module) -> "DeepSeekV4QKVProjection":
        config = attention.config
        module = cls(
            hidden_size=config.hidden_size,
            q_lora_rank=config.q_lora_rank,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            rope_dim=config.qk_rope_head_dim,
            rms_norm_eps=config.rms_norm_eps,
        )
        module.q_a_proj = attention.q_a_proj
        module.q_a_norm = rms_norm_from_hf(attention.q_a_norm)
        module.q_b_proj = attention.q_b_proj
        module.q_b_norm = rms_norm_from_hf(
            attention.q_b_norm,
            hidden_size=module.head_dim,
        )
        module.kv_proj = attention.kv_proj
        module.kv_norm = rms_norm_from_hf(attention.kv_norm)
        return module

    def forward(
        self,
        hidden_states: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) -> ProjectedQKV:
        if not is_fx_proxy(hidden_states) and (hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size):
            raise ValueError(f"hidden_states must be [B,P,{self.hidden_size}]")
        q_residual = self.q_a_norm(self.q_a_proj(hidden_states))
        query = self.q_b_proj(q_residual).reshape(
            hidden_states.shape[0],
            hidden_states.shape[1],
            self.num_heads,
            self.head_dim,
        )
        query = self.q_b_norm(query)
        query = self.partial_rope(
            query,
            cos,
            sin,
        )

        latent = self.kv_norm(self.kv_proj(hidden_states))
        latent = self.partial_rope(
            latent.unsqueeze(2),
            cos,
            sin,
        ).squeeze(2)
        return ProjectedQKV(q_residual, query, latent)


class BoundGroupedOutputProjection(nn.Module):
    """Use checkpoint grouped o_a/o_b modules after inverse partial RoPE."""

    def __init__(
        self,
        o_a_proj: nn.Module,
        o_b_proj: nn.Module,
        *,
        num_heads: int,
        head_dim: int,
        num_groups: int,
        rope_dim: int,
    ) -> None:
        super().__init__()
        self.o_a_proj = StaticGroupedLinear.from_hf(
            o_a_proj,
            num_groups=num_groups,
        )
        self.o_b_proj = o_b_proj
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.num_groups = int(num_groups)
        self.rope_dim = int(rope_dim)
        self.partial_rope = InterleavedPartialRope(self.rope_dim)
        if self.num_heads % self.num_groups:
            raise ValueError("num_heads must be divisible by num_groups")

    @classmethod
    def from_hf(cls, attention: nn.Module) -> "BoundGroupedOutputProjection":
        config = attention.config
        return cls(
            attention.o_a_proj,
            attention.o_b_proj,
            num_heads=config.num_attention_heads,
            head_dim=config.head_dim,
            num_groups=config.o_groups,
            rope_dim=config.qk_rope_head_dim,
        )

    def forward(self, attention_output: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        output = self.partial_rope(
            attention_output,
            cos,
            sin,
            inverse=True,
        )
        grouped = output.reshape(
            output.shape[0],
            output.shape[1],
            self.num_groups,
            -1,
        )
        return self.o_b_proj(self.o_a_proj(grouped).flatten(2))


__all__ = [
    "BoundGroupedOutputProjection",
    "DeepSeekV4QKVProjection",
    "ProjectedQKV",
    "StaticGroupedLinear",
]
