"""Static DeepSeek-V4 routing and exact clamped-SwiGLU MoE."""

from __future__ import annotations

from typing import NamedTuple
from weakref import WeakKeyDictionary

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from xhquant import nn as xhnn
from xhquant.nn.modules.moeblock import MoeBlock

from ._trace import is_fx_proxy


FP16_MIN = -torch.finfo(torch.float16).max


class RoutedTokens(NamedTuple):
    weights: Tensor
    indices: Tensor


class StackedLinearState(NamedTuple):
    weight: Tensor
    quant_weight: Tensor | None


_STACKED_EXPERT_CACHE: WeakKeyDictionary[
    nn.Module,
    tuple[StackedLinearState, StackedLinearState, StackedLinearState],
] = WeakKeyDictionary()


def _move_parameter_to_meta(module: nn.Module, name: str) -> None:
    parameter = getattr(module, name)
    setattr(
        module,
        name,
        nn.Parameter(
            torch.empty_like(parameter.data, device="meta"),
            requires_grad=parameter.requires_grad,
        ),
    )


def _move_tensor_attribute_to_meta(module: nn.Module, name: str) -> None:
    value = getattr(module, name)
    meta_value = torch.empty_like(value, device="meta")
    if name in module._buffers:
        module._buffers[name] = meta_value
    else:
        setattr(module, name, meta_value)


def _stack_defused_linear(
    experts: list[nn.Module],
    linear_name: str,
) -> StackedLinearState:
    """Stack a dense/dequantized expert projection for the legacy load mode."""

    first = getattr(experts[0], linear_name)
    if isinstance(first, xhnn.GPTQPackedLinear):
        raise TypeError("GPTQPackedLinear projections must use GPTQPackedMoeBlock")
    weight_shape = tuple(first.weight.shape)
    packed_weight = torch.empty(
        len(experts),
        *weight_shape,
        device=first.weight.device,
        dtype=first.weight.dtype,
    )
    first_quant_weight = getattr(first, "quant_weight", None)
    has_quant_weight = torch.is_tensor(first_quant_weight)
    packed_quant_weight = None
    if has_quant_weight:
        if tuple(first_quant_weight.shape) != weight_shape:
            raise ValueError(
                f"{linear_name}.quant_weight must match weight shape {weight_shape}, "
                f"got {tuple(first_quant_weight.shape)}"
            )
        packed_quant_weight = torch.empty(
            len(experts),
            *weight_shape,
            device=first_quant_weight.device,
            dtype=first_quant_weight.dtype,
        )
    with torch.no_grad():
        for expert_index, expert in enumerate(experts):
            linear = getattr(expert, linear_name)
            if tuple(linear.weight.shape) != weight_shape:
                raise ValueError(
                    f"expert {expert_index} {linear_name}.weight has shape "
                    f"{tuple(linear.weight.shape)}, expected {weight_shape}"
                )
            packed_weight[expert_index].copy_(
                linear.weight.data.to(
                    device=packed_weight.device,
                    dtype=packed_weight.dtype,
                )
            )

            quant_weight = getattr(linear, "quant_weight", None)
            if has_quant_weight:
                if not torch.is_tensor(quant_weight):
                    raise ValueError(f"expert {expert_index} {linear_name} is missing quant_weight")
                if tuple(quant_weight.shape) != weight_shape:
                    raise ValueError(
                        f"expert {expert_index} {linear_name}.quant_weight has shape "
                        f"{tuple(quant_weight.shape)}, expected {weight_shape}"
                    )
                packed_quant_weight[expert_index].copy_(
                    quant_weight.data.to(
                        device=packed_quant_weight.device,
                        dtype=packed_quant_weight.dtype,
                    )
                )
            elif torch.is_tensor(quant_weight):
                raise ValueError(
                    f"expert {expert_index} {linear_name} unexpectedly has quant_weight; "
                    "all routed experts must use the same weight representation"
                )

            _move_parameter_to_meta(linear, "weight")
            if torch.is_tensor(quant_weight):
                _move_tensor_attribute_to_meta(linear, "quant_weight")

    return StackedLinearState(
        packed_weight,
        packed_quant_weight,
    )


class DeepSeekV4Router(nn.Module):
    """Hash or score router with V4's sqrt-softplus selection contract.

    The second MoeBlock input stays on the long-lived XH ABI: all expert
    scores ``[B,P,E]``. The separately selected IDs ``[B,P,K]`` encode V4's
    correction-bias/hash routing; MoeBlock gathers and normalizes those rows.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        routed_scaling_factor: float,
        is_hash: bool,
        vocab_size: int = 0,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.routed_scaling_factor = float(routed_scaling_factor)
        self.is_hash = bool(is_hash)
        self.proj = nn.Linear(self.hidden_size, self.num_experts, bias=False, dtype=torch.float16)
        self.register_buffer(
            "correction_bias",
            None if self.is_hash else torch.zeros(self.num_experts, dtype=torch.float16),
        )
        self.register_buffer(
            "tid2eid",
            torch.zeros(int(vocab_size), self.top_k, dtype=torch.int64) if self.is_hash else None,
        )
        # Use the deployable Gather operator directly.  PyTorch advanced
        # indexing lowers through an otherwise redundant input_ids -> INT32
        # Cast even when input_ids is already INT32 in the HMONNX ABI.
        self.hash_gather = xhnn.Gather(axis=0) if self.is_hash else None

    @property
    def weight(self) -> nn.Parameter:
        return self.proj.weight

    @classmethod
    def from_hf(cls, router: nn.Module, config: object) -> "DeepSeekV4Router":
        module = cls(
            hidden_size=router.hidden_dim,
            num_experts=router.num_experts,
            top_k=router.top_k,
            routed_scaling_factor=router.routed_scaling_factor,
            is_hash=hasattr(router, "tid2eid"),
            vocab_size=getattr(config, "vocab_size", 0),
        )
        module.proj.weight = router.weight
        quant_weight = getattr(router, "quant_weight", None)
        if torch.is_tensor(quant_weight):
            module.proj.register_buffer("quant_weight", quant_weight)
        if module.is_hash:
            module.tid2eid = router.tid2eid
        else:
            module.correction_bias = router.e_score_correction_bias
        return module

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor | None = None,
    ) -> RoutedTokens:
        if not is_fx_proxy(hidden_states) and (hidden_states.ndim != 3 or hidden_states.shape[-1] != self.hidden_size):
            raise ValueError(f"hidden_states must be [B,P,{self.hidden_size}]")
        logits = self.proj(hidden_states)
        scores = torch.sqrt(F.softplus(logits))
        if self.is_hash:
            if input_ids is None or (
                not is_fx_proxy(input_ids) and tuple(input_ids.shape) != tuple(hidden_states.shape[:2])
            ):
                raise ValueError("hash routing requires input_ids with shape [B,P]")
            assert self.hash_gather is not None
            indices = self.hash_gather(self.tid2eid, input_ids)
        else:
            indices = torch.topk(
                scores + self.correction_bias,
                self.top_k,
                dim=-1,
                sorted=True,
            )[1]
        return RoutedTokens(scores, indices)


class DeepSeekV4SharedExpert(nn.Module):
    """The always-on expert with the checkpoint's exact clamp semantics."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        limit: float,
    ) -> None:
        super().__init__()
        self.limit = float(limit)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float16)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=torch.float16)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=torch.float16)

    @classmethod
    def from_hf(cls, shared_expert: nn.Module) -> "DeepSeekV4SharedExpert":
        module = cls(
            hidden_size=shared_expert.hidden_size,
            intermediate_size=shared_expert.intermediate_size,
            limit=shared_expert.limit,
        )
        module.gate_proj = shared_expert.gate_proj
        module.up_proj = shared_expert.up_proj
        module.down_proj = shared_expert.down_proj
        return module

    def forward(self, hidden_states: Tensor) -> Tensor:
        gate = self.gate_proj(hidden_states).clamp(
            min=FP16_MIN,
            max=self.limit,
        )
        up = self.up_proj(hidden_states).clamp(
            min=-self.limit,
            max=self.limit,
        )
        return self.down_proj(F.silu(gate) * up)


class ExactClampedSwiGLURoutedExperts(nn.Module):
    """DeepSeek-V4 routed experts using one MoeBlock with ``up_shift=0``."""

    def __init__(
        self,
        *,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        top_k: int,
        limit: float,
        fast_mode: bool = True,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.hidden_size = int(hidden_size)
        self.intermediate_size = int(intermediate_size)
        self.fast_mode = bool(fast_mode)
        self.block = MoeBlock(
            activation_type="silu",
            k=int(top_k),
            normalize_routing_weights=True,
            limit=float(limit),
            up_shift=0.0,
            topk_outside=True,
        )

    def bind_stacked_weights(
        self,
        gate_weight: Tensor,
        up_weight: Tensor,
        down_weight: Tensor,
    ) -> None:
        expected_gate = (
            self.num_experts,
            self.intermediate_size,
            self.hidden_size,
        )
        expected_down = (
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
        )
        if tuple(gate_weight.shape) != expected_gate:
            raise ValueError(f"gate weight must have shape {expected_gate}")
        if tuple(up_weight.shape) != expected_gate:
            raise ValueError(f"up weight must have shape {expected_gate}")
        if tuple(down_weight.shape) != expected_down:
            raise ValueError(f"down weight must have shape {expected_down}")

        self.block._buffers["expert_gate_proj_weight"] = gate_weight
        self.block._buffers["expert_up_proj_weight"] = up_weight
        self.block._buffers["expert_down_proj_weight"] = down_weight
        self.block.expert_gate_proj_bias = None
        self.block.expert_up_proj_bias = None
        self.block.expert_down_proj_bias = None

    def bind_stacked_quant_weights(
        self,
        gate_quant_weight: Tensor,
        up_quant_weight: Tensor,
        down_quant_weight: Tensor,
    ) -> None:
        expected_gate = (
            self.num_experts,
            self.intermediate_size,
            self.hidden_size,
        )
        expected_down = (
            self.num_experts,
            self.hidden_size,
            self.intermediate_size,
        )
        for name, value, expected in (
            ("gate_proj", gate_quant_weight, expected_gate),
            ("up_proj", up_quant_weight, expected_gate),
            ("down_proj", down_quant_weight, expected_down),
        ):
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} quant_weight must have shape {expected}, got {tuple(value.shape)}")
            buffer_name = f"expert_{name}_quant_weight"
            if buffer_name in self.block._buffers:
                self.block._buffers[buffer_name] = value
            else:
                self.block.register_buffer(buffer_name, value)

    @classmethod
    def from_hf(
        cls,
        experts: nn.Module,
        *,
        top_k: int,
        limit: float,
        fast_mode: bool = True,
    ) -> "ExactClampedSwiGLURoutedExperts":
        if hasattr(experts, "gate_up_proj"):
            num_experts = experts.num_experts
            hidden_size = experts.hidden_dim
            intermediate_size = experts.intermediate_dim
            module = cls(
                num_experts=num_experts,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                top_k=top_k,
                limit=limit,
                fast_mode=fast_mode,
            )
            gate_up = experts.gate_up_proj
            module.bind_stacked_weights(
                gate_up[:, :intermediate_size, :],
                gate_up[:, intermediate_size:, :],
                experts.down_proj,
            )
            return module

        if isinstance(experts, list):
            expert_list = list(experts)
        elif isinstance(experts, nn.Module):
            # GPTQModel Defuser keeps the original DeepseekV4Experts wrapper
            # and registers defused experts as numeric children alongside
            # ``act_fn``. ModuleList is the simpler special case of this ABI.
            expert_list = [child for name, child in experts.named_children() if name.isdigit()]
        else:
            expert_list = []

        if expert_list:
            first = expert_list[0]
            if not all(hasattr(first, projection) for projection in ("gate_proj", "up_proj", "down_proj")):
                raise TypeError(f"unsupported defused expert type: {type(first)}")
            packed_linears = [
                getattr(expert, projection)
                for expert in expert_list
                for projection in ("gate_proj", "up_proj", "down_proj")
            ]
            packed_flags = [isinstance(linear, xhnn.GPTQPackedLinear) for linear in packed_linears]
            if any(packed_flags) and not all(packed_flags):
                raise TypeError("routed experts cannot mix packed and dense projections")

            hidden_size = int(first.gate_proj.in_features)
            intermediate_size = int(first.gate_proj.out_features)
            module = cls(
                num_experts=len(expert_list),
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                top_k=top_k,
                limit=limit,
                fast_mode=fast_mode,
            )
            if all(packed_flags):
                if any(linear.bias is not None for linear in packed_linears):
                    raise NotImplementedError("DeepSeek-V4 packed routed experts require bias-free projections")
                module.block = xhnn.GPTQPackedMoeBlock(
                    activation_type="silu",
                    k=int(top_k),
                    normalize_routing_weights=True,
                    gate_proj=[expert.gate_proj.packed_weight for expert in expert_list],
                    up_proj=[expert.up_proj.packed_weight for expert in expert_list],
                    down_proj=[expert.down_proj.packed_weight for expert in expert_list],
                    limit=float(limit),
                    up_shift=0.0,
                    topk_outside=True,
                )
                return module

            stacked_cache = _STACKED_EXPERT_CACHE.get(experts) if isinstance(experts, nn.Module) else None
            if stacked_cache is not None:
                gate_state, up_state, down_state = stacked_cache
                module.bind_stacked_weights(
                    gate_state.weight,
                    up_state.weight,
                    down_state.weight,
                )
                packed_quant_weights = (
                    gate_state.quant_weight,
                    up_state.quant_weight,
                    down_state.quant_weight,
                )
                if all(value is not None for value in packed_quant_weights):
                    module.bind_stacked_quant_weights(*packed_quant_weights)
                elif any(value is not None for value in packed_quant_weights):
                    raise ValueError("cached routed expert projections have incomplete quant_weight tensors")
                return module

            gate_state = _stack_defused_linear(
                expert_list,
                "gate_proj",
            )
            up_state = _stack_defused_linear(
                expert_list,
                "up_proj",
            )
            down_state = _stack_defused_linear(
                expert_list,
                "down_proj",
            )
            module.bind_stacked_weights(
                gate_state.weight,
                up_state.weight,
                down_state.weight,
            )
            packed_quant_weights = (
                gate_state.quant_weight,
                up_state.quant_weight,
                down_state.quant_weight,
            )
            if all(value is not None for value in packed_quant_weights):
                module.bind_stacked_quant_weights(*packed_quant_weights)
            elif any(value is not None for value in packed_quant_weights):
                raise ValueError(
                    "gate/up/down routed expert projections must all either provide quant_weight or omit it"
                )
            if isinstance(experts, nn.Module):
                # Prefill and decode are independent static graphs but their
                # immutable stacked expert parameters must share storage. The
                # first conversion consumes the defused source linears by
                # moving them to meta; cache the stacked tensors by source
                # module for the second conversion instead of copying many GiB.
                _STACKED_EXPERT_CACHE[experts] = (
                    StackedLinearState(
                        module.block.expert_gate_proj_weight,
                        getattr(module.block, "expert_gate_proj_quant_weight", None),
                    ),
                    StackedLinearState(
                        module.block.expert_up_proj_weight,
                        getattr(module.block, "expert_up_proj_quant_weight", None),
                    ),
                    StackedLinearState(
                        module.block.expert_down_proj_weight,
                        getattr(module.block, "expert_down_proj_quant_weight", None),
                    ),
                )
            return module
        raise TypeError(f"unsupported experts container: {type(experts)}")

    def forward(
        self,
        hidden_states: Tensor,
        routing_weights: Tensor,
        selected_experts: Tensor,
    ) -> Tensor:
        return self.block(
            hidden_states,
            routing_weights,
            selected_experts,
            fast_mode=self.fast_mode,
        )


class DeepSeekV4MoE(nn.Module):
    """Router + 256 routed experts + one always-on shared expert."""

    def __init__(
        self,
        router: DeepSeekV4Router,
        routed_experts: ExactClampedSwiGLURoutedExperts,
        shared_expert: DeepSeekV4SharedExpert,
    ) -> None:
        super().__init__()
        self.router = router
        self.routed_experts = routed_experts
        self.shared_expert = shared_expert

    @classmethod
    def from_hf(
        cls,
        moe: nn.Module,
        config: object,
        *,
        fast_mode: bool = True,
    ) -> "DeepSeekV4MoE":
        router = DeepSeekV4Router.from_hf(moe.gate, config)
        routed = ExactClampedSwiGLURoutedExperts.from_hf(
            moe.experts,
            top_k=moe.gate.top_k,
            limit=moe.shared_experts.limit,
            fast_mode=fast_mode,
        )
        shared = DeepSeekV4SharedExpert.from_hf(moe.shared_experts)
        return cls(router, routed, shared)

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor | None = None,
    ) -> Tensor:
        routing = self.router(hidden_states, input_ids)
        routed = self.routed_experts(
            hidden_states,
            routing.weights,
            routing.indices,
        )
        routed = routed * self.router.routed_scaling_factor
        return routed + self.shared_expert(hidden_states)


class DeepSeekV4MoEPlaceHolder(nn.Module):
    """Weightless main-graph boundary used by streamed MoE export.

    The real module is exported separately from the same two-input/one-output
    contract and is inlined into the final HMONNX file. Keeping ``input_ids``
    in the signature is required by the first three hash-routed blocks even
    though the placeholder itself only propagates shape metadata.
    """

    PLACEHOLDER_TYPE_NAME = "DeepseekV4SparseMoeBlock"

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor | None = None,
    ) -> Tensor:
        del input_ids
        return hidden_states


__all__ = [
    "DeepSeekV4MoE",
    "DeepSeekV4MoEPlaceHolder",
    "DeepSeekV4Router",
    "DeepSeekV4SharedExpert",
    "ExactClampedSwiGLURoutedExperts",
    "RoutedTokens",
]
