import math
from functools import partial
from typing import Any, Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers.activations import ACT2FN
from transformers.cache_utils import (
    Cache,
    DynamicCache,
    SlidingWindowCache,
    StaticCache,
)
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
from transformers.utils import (
    is_torchdynamo_compiling,
    logging,
)

from .configuration_ipt import IPTConfig


logger = logging.get_logger(__name__)


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """
    Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
    `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full(
            (sequence_length, target_length),
            fill_value=min_dtype,
            dtype=dtype,
            device=device,
        )
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(padding_mask, min_dtype)

    return causal_mask


def print_input_shapes(
    input_ids: Any = None,
    attention_mask: Any = None,
    position_ids: Any = None,
    routing_ids: Any = None,
    tag="",
) -> None:
    """
    Prints shapes for the given arguments. Handles:
      - torch.Tensor
      - numpy.ndarray
      - list/tuple/dict containing tensors/arrays
      - None
    """

    def shape_of(x):
        # Lazy imports so the function is standalone
        try:
            import torch
        except Exception:
            torch = None
        try:
            import numpy as np
        except Exception:
            np = None

        if x is None:
            return "None"

        if torch is not None and isinstance(x, torch.Tensor):
            return f"{tuple(x.shape)}"

        if np is not None and isinstance(x, np.ndarray):
            return f"{tuple(x.shape)} (numpy)"

        if isinstance(x, (list, tuple)):
            if not x:
                return "[]"
            parts = []
            for i, xi in enumerate(x):
                if torch is not None and isinstance(xi, torch.Tensor):
                    parts.append(f"{i}:{tuple(xi.shape)}")
                elif np is not None and isinstance(xi, np.ndarray):
                    parts.append(f"{i}:{tuple(xi.shape)}(np)")
                else:
                    parts.append(f"{i}:{type(xi).__name__}")
            return f"[{', '.join(parts)}]"

        if isinstance(x, dict):
            parts = []
            for k, v in x.items():
                if torch is not None and isinstance(v, torch.Tensor):
                    parts.append(f"{k}:{tuple(v.shape)}")
                elif np is not None and isinstance(v, np.ndarray):
                    parts.append(f"{k}:{tuple(v.shape)}(np)")
                else:
                    parts.append(f"{k}:{type(v).__name__}")
            return "{" + ", ".join(parts) + "}"

        return f"{type(x).__name__}"

    print(f"[{tag}] input_ids     :", shape_of(input_ids))
    print(f"[{tag}] attention_mask:", shape_of(attention_mask))
    print(f"[{tag}] position_ids  :", shape_of(position_ids))
    print(f"[{tag}] routing_ids   :", shape_of(routing_ids))


class IPTRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        IPTRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.float32))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        output = self.weight * hidden_states
        return output.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class IPTRotaryEmbedding(nn.Module):
    def __init__(self, config, device=None):
        super().__init__()
        self.rope_kwargs = {}
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.float(), sin.float()  # float32 output


class IPTMLP(nn.Module):
    def __init__(self, config, hidden_size=None, intermediate_size=None):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size if hidden_size is None else hidden_size
        self.intermediate_size = config.intermediate_size if intermediate_size is None else intermediate_size

        self.clamp_input_value = 0
        if hasattr(config, "clamp_training"):
            clamp_training_cfg = config.clamp_training
            self.clamp_input_value = clamp_training_cfg["clamp_input_value"]

        self.fc1 = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=False)
        self.fc2 = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.clamp_input_value > 0:
            x = torch.clamp_(x, -self.clamp_input_value, self.clamp_input_value)
        intermediate_parallel = self.fc1(x)

        intermediate_parallel1, intermediate_parallel2 = torch.chunk(intermediate_parallel, 2, dim=-1)
        intermediate_parallel1 = intermediate_parallel1.squeeze(-1)
        intermediate_parallel2 = intermediate_parallel2.squeeze(-1)
        intermediate_parallel1 = self.act_fn(intermediate_parallel1)
        intermediate_parallel = intermediate_parallel1 * intermediate_parallel2

        if self.clamp_input_value > 0:
            intermediate_parallel = torch.clamp_(intermediate_parallel, -self.clamp_input_value, self.clamp_input_value)
        output = self.fc2(intermediate_parallel)

        return output


class IPTRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_groups = config.num_groups
        self.num_experts = config.num_routed_experts - config.num_shared_experts

        self.gating = nn.Linear(config.hidden_size, self.num_groups * self.num_experts, bias=False, dtype=torch.float32)

        self.enable_noaux_tc = True
        if self.enable_noaux_tc:
            num_all_routed_experts_shape = self.num_groups * self.num_experts
            self.register_buffer(
                "e_score_correction_bias", torch.zeros(num_all_routed_experts_shape, dtype=torch.float32)
            )

        self.calc_denominator_cross_groups = config.calc_denominator_cross_groups
        self.routed_scaling_factor = config.routed_scaling_factor

    def routing(self, logits, routing_map: Optional[torch.Tensor] = None):
        num_tokens, num_groups, num_experts_per_group = logits.shape
        scores = logits.sigmoid()  # [num_tokens, groups, num_routed_experts_per_group]
        # breakpoint()
        # assert routing_map is not None
        if routing_map is not None:
            # 复用routing map，直接取相应位置的值
            try:
                roll_topk_indices = routing_map.view(
                    num_tokens, num_groups, self.top_k
                ).long()  # TODO cx note: review required
                scores_for_choice = scores.view(num_tokens, -1) + self.e_score_correction_bias.unsqueeze(
                    0
                )  # [num_tokens, groups * num_routed_experts_per_group]
                scores_for_choice = scores_for_choice.view_as(
                    scores
                )  # [num_tokens, groups, num_routed_experts_per_group]
                _, topk_indices = torch.topk(
                    scores_for_choice, k=self.top_k, dim=-1, sorted=False
                )  # [num_tokens, groups, topk]
                topk_indices = torch.where(roll_topk_indices >= 0, roll_topk_indices, topk_indices)
            except RuntimeError as re:
                raise re
        else:
            scores_for_choice = scores.view(num_tokens, -1) + self.e_score_correction_bias.unsqueeze(
                0
            )  # [num_tokens, groups * num_routed_experts_per_group]
            scores_for_choice = scores_for_choice.view_as(scores)  # [num_tokens, groups, num_routed_experts_per_group]
            _, topk_indices = torch.topk(
                scores_for_choice, k=self.top_k, dim=-1, sorted=False
            )  # [num_tokens, groups, topk]
        topk_probs = scores.gather(-1, topk_indices)
        if self.top_k > 1:
            if self.calc_denominator_cross_groups:
                denominator = topk_probs.view(topk_probs.size(0), -1)
                denominator = denominator.sum(dim=-1, keepdim=True) + 1e-20
                denominator = denominator.unsqueeze(-1)
            else:
                denominator = topk_probs.sum(dim=-1, keepdim=True) + 1e-20
            topk_probs = topk_probs / denominator
        topk_probs = topk_probs * self.routed_scaling_factor

        topk_mask = torch.zeros(logits.shape, dtype=torch.int32, device=logits.device).scatter(-1, topk_indices, 1)
        tokens_per_expert = topk_mask.sum(dim=0)

        head_incre = (
            torch.arange(num_groups, dtype=topk_indices.dtype, device=topk_indices.device) * num_experts_per_group
        ).view(1, -1, 1)
        topk_indices = (topk_indices + head_incre).view(num_tokens, -1)
        topk_probs = topk_probs.view(num_tokens, -1)
        tokens_per_expert = tokens_per_expert.view(-1)
        tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

        return topk_probs.to(torch.float32), topk_indices, tokens_per_expert

    def forward(self, hidden_states, routing_map: Optional[torch.Tensor] = None):
        hidden_states = hidden_states.float()
        logits = self.gating(hidden_states)
        logits = logits.view(-1, self.num_groups, self.num_experts)
        scores, indices, tokens_per_expert = self.routing(logits, routing_map)
        return scores, indices, tokens_per_expert


class IPTGroupedRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_groups = config.num_groups
        self.num_experts = config.num_routed_experts
        self.num_shared_experts = config.num_shared_experts

        self.gating = nn.Linear(config.hidden_size, self.num_groups * self.num_experts, bias=False, dtype=torch.float32)

        self.enable_noaux_tc = True
        if self.enable_noaux_tc:
            num_all_routed_experts_shape = (self.num_groups * (self.num_experts - self.num_shared_experts),)
            self.register_buffer(
                "e_score_correction_bias", torch.zeros(num_all_routed_experts_shape, dtype=torch.float32)
            )

        self.calc_denominator_cross_groups = config.calc_denominator_cross_groups
        self.routed_scaling_factor = config.routed_scaling_factor

    def routing(self, logits):
        num_tokens, num_groups, num_experts_per_group = logits.shape
        scores = logits[:, :, self.num_shared_experts :].sigmoid()  # [num_tokens, groups, num_routed_experts_per_group]

        scores_for_choice = scores.view(num_tokens, -1) + self.e_score_correction_bias.unsqueeze(
            0
        )  # [num_tokens, groups * num_routed_experts_per_group]
        scores_for_choice = scores_for_choice.view_as(scores)  # [num_tokens, groups, num_routed_experts_per_group]
        _, topk_indices = torch.topk(
            scores_for_choice, k=self.top_k, dim=-1, sorted=False
        )  # [num_tokens, groups, topk]
        topk_probs = scores.gather(-1, topk_indices)
        if self.top_k > 1:
            if self.calc_denominator_cross_groups:
                denominator = topk_probs.view(topk_probs.size(0), -1)
                denominator = denominator.sum(dim=-1, keepdim=True) + 1e-20
                denominator = denominator.unsqueeze(-1)
            else:
                denominator = topk_probs.sum(dim=-1, keepdim=True) + 1e-20
            topk_probs = topk_probs / denominator
        topk_probs = topk_probs * self.routed_scaling_factor
        shared_probs = torch.ones_like(
            topk_probs[:, :, : self.num_shared_experts]
        )  # [num_tokens, groups, num_shared_experts]
        topk_probs = torch.cat([shared_probs, topk_probs], dim=-1)  # [num_tokens, groups, num_shared_experts + topk]

        topk_indices = topk_indices + self.num_shared_experts
        shared_indices = torch.arange(0, self.num_shared_experts, dtype=topk_indices.dtype, device=topk_indices.device)
        shared_indices = shared_indices.unsqueeze(0).unsqueeze(0).repeat(*topk_indices.shape[:2], 1)
        topk_indices = torch.cat([shared_indices, topk_indices], dim=-1)

        topk_mask = torch.zeros(logits.shape, dtype=torch.int32, device=logits.device).scatter(-1, topk_indices, 1)
        tokens_per_expert = topk_mask.sum(dim=0)

        head_incre = (
            torch.arange(num_groups, dtype=topk_indices.dtype, device=topk_indices.device) * num_experts_per_group
        ).view(1, -1, 1)
        topk_indices = (topk_indices + head_incre).view(num_tokens, -1)
        topk_probs = topk_probs.view(num_tokens, -1)
        tokens_per_expert = tokens_per_expert.view(-1)
        tokens_per_expert = tokens_per_expert.cpu().to(torch.long)

        return topk_probs.to(torch.float32), topk_indices, tokens_per_expert

    def forward(self, hidden_states):
        hidden_states = hidden_states.float()
        logits = self.gating(hidden_states)
        logits = logits.view(-1, self.num_groups, self.num_experts)
        scores, indices, tokens_per_expert = self.routing(logits)

        return scores, indices, tokens_per_expert


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def build_activation(activation_func: str = "geglu"):
    def glu_gelu_gpu(x):
        x = torch.chunk(x, 2, dim=-1)
        return F.gelu(x[0]) * x[1]

    activation = glu_gelu_gpu

    if activation is None:
        raise NotImplementedError(f"{activation_func} not supported yet")
    return activation


class GroupedMoEMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = (config.num_routed_experts - config.num_shared_experts) * config.num_groups
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size

        self.fc1_output_size = self.intermediate_size
        if config.activation_func == "geglu":
            self.fc1_output_size = self.fc1_output_size * 2
        self.fc1_weights = nn.Parameter(torch.empty(self.num_experts, self.fc1_output_size, self.hidden_size))
        self.fc2_weights = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, self.intermediate_size))

        self.activation = build_activation(config.activation_func)
        self.reset_parameters()

    def _get_init_method(self):
        emb_dim = self.hidden_size
        mlp_hid_dim = self.config.moe_intermediate_size * 2 * (self.config.num_experts_per_tok) * self.config.num_groups
        if self.config.activation_func == "geglu":
            mlp_hid_dim = mlp_hid_dim // 2

        fc2_init_std = math.sqrt(self.config.weight_init_factor * 2.0 / float(emb_dim + mlp_hid_dim))
        init_method = partial(nn.init.normal_, mean=0, std=fc2_init_std)
        if getattr(self.config, "fixed_init_std", False):
            init_method = partial(nn.init.normal_, mean=0, std=self.config.fixed_init_std)
        return init_method

    def reset_parameters(self) -> None:
        init_method = self._get_init_method()
        init_method(self.fc1_weights)
        init_method(self.fc2_weights)

    def forward(self, x):
        raise NotImplementedError("The MoE computation is performed in the parent `IPTMoE` module.")

    def extra_repr(self) -> str:
        return f"hidden_size={self.hidden_size}, intermediate_size={self.intermediate_size}, num_experts={self.num_experts}"


class IPTMoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.grouped_gemm = config.grouped_gemm
        self.clamp_input_value = 0
        if hasattr(config, "clamp_training"):
            clamp_training_cfg = config.clamp_training
            self.clamp_input_value = clamp_training_cfg["clamp_input_value"]

        if self.grouped_gemm:
            self.router = IPTRouter(config)
            self.routed_experts = GroupedMoEMLP(config)
        else:
            self.num_experts = (config.num_routed_experts - config.num_shared_experts) * config.num_groups
            self.router = IPTRouter(config)
            self.routed_experts = nn.ModuleList(
                [IPTMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(self.num_experts)]
            )
        self.shared_experts = None
        if config.num_shared_experts > 0:
            self.shared_experts = IPTMLP(
                config, intermediate_size=config.moe_intermediate_size * config.num_shared_experts * config.num_groups
            )

    def grouped_moe(self, hidden_states, topk_indices, topk_weights, tokens_per_expert):
        import grouped_gemm.ops as ops

        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        topk_indices = topk_indices.to(torch.int32)
        permuted_hidden_states, row_id_map = ops.permute(hidden_states, topk_indices)
        permuted_hidden_states = torch.clamp_(permuted_hidden_states, -self.clamp_input_value, self.clamp_input_value)
        fc1_output = ops.gmm(
            permuted_hidden_states,
            self.routed_experts.fc1_weights,
            tokens_per_expert,
            trans_b=True,
        )

        intermediate_parallel = self.routed_experts.activation(fc1_output)
        intermediate_parallel = torch.clamp_(intermediate_parallel, -self.clamp_input_value, self.clamp_input_value)
        fc2_output = ops.gmm(
            intermediate_parallel,
            self.routed_experts.fc2_weights,
            tokens_per_expert,
            trans_b=True,
        )

        output = ops.unpermute(fc2_output, row_id_map, topk_weights)
        return output

    def moe(self, hidden_states, topk_indices, topk_weights):
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        final_hidden_states = torch.zeros_like(hidden_states, dtype=topk_weights.dtype)
        expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 0, 1)

        for expert_idx in range(self.num_experts):
            expert = self.routed_experts[expert_idx]
            mask = expert_mask[expert_idx]
            token_indices, weight_indices = torch.where(mask)

            if token_indices.numel() > 0:
                expert_weights = topk_weights[token_indices, weight_indices]
                expert_input = hidden_states[token_indices]
                expert_output = expert(expert_input)
                weighted_output = expert_output * expert_weights.unsqueeze(-1)
                final_hidden_states.index_add_(0, token_indices, weighted_output)
        return final_hidden_states.type(hidden_states.dtype)

    def forward(self, hidden_states, routing_map: Optional[torch.Tensor] = None):
        orig_shape = hidden_states.shape
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if routing_map is not None:
            # print('0000000000000000000000000000')
            # 检查一下routing map的尺寸是否相符
            rp_batch_size, rp_sequence, rp_expert_num = routing_map.shape

            assert rp_batch_size == batch_size, (
                f"[spark_routing_model] Shape mismatch: routing_map.shape={routing_map.shape} but hidden_states.shape={hidden_states.shape}"
            )
            assert rp_sequence == sequence_length, (
                f"[spark_routing_model] Shape mismatch: routing_map.shape={routing_map.shape} but hidden_states.shape={hidden_states.shape}"
            )
            assert rp_expert_num == self.config.num_experts_per_tok, (
                f"[spark_routing_model] Expert number mismatch: rp_expert_num={rp_expert_num} but top_k={self.config.num_experts_per_tok}"
            )

        if self.shared_experts:
            shared_output = self.shared_experts(hidden_states)

        if self.grouped_gemm:
            probs, indices, tokens_per_expert = self.router(hidden_states, routing_map)
            hidden_states = self.grouped_moe(hidden_states, indices, probs, tokens_per_expert).view(*orig_shape)
        else:
            probs, indices, tokens_per_expert = self.router(hidden_states, routing_map)
            hidden_states = self.moe(hidden_states, indices, probs).view(*orig_shape)

        if self.shared_experts:
            hidden_states = hidden_states + shared_output

        return hidden_states


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class IPTAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: IPTConfig, layer_idx: Optional[int] = None):
        super().__init__()

        self.config = config
        self.config._attn_implementation = "flash_attention_2"
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.clamp_input_value = 0
        if hasattr(config, "clamp_training"):
            clamp_training_cfg = config.clamp_training
            self.clamp_input_value = clamp_training_cfg["clamp_input_value"]

        self.rotary_emb = IPTRotaryEmbedding(config=config)

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        self.q_k_v_proj = nn.Linear(
            self.hidden_size,
            self.num_heads * self.head_dim + self.num_key_value_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.out_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)
        self.scaling = self.head_dim ** (-0.5)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.clamp_input_value > 0:
            hidden_states = torch.clamp_(hidden_states, -self.clamp_input_value, self.clamp_input_value)
        q_k_v = self.q_k_v_proj(hidden_states)

        if self.num_key_value_heads is not None and self.num_key_value_heads != self.num_heads:
            query_states = q_k_v[..., : self.num_heads * self.head_dim]
            key_states, value_states = torch.chunk(q_k_v[..., self.num_heads * self.head_dim :], 2, dim=-1)
        else:
            query_states, key_states, value_states = torch.chunk(q_k_v, 3, dim=-1)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        # sin,cos float32 output, dense use bfloat16
        cos = cos.to(query_states.dtype)
        sin = sin.to(query_states.dtype)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {
                "sin": sin,
                "cos": cos,
                "cache_position": cache_position,
            }  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        if (
            self.config.use_sliding_window and getattr(self.config, "sliding_window", None) is not None
            # and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,
            position_ids=position_ids,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.clamp_input_value > 0:
            attn_output = torch.clamp_(attn_output, -self.clamp_input_value, self.clamp_input_value)
        attn_output = self.out_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


def yarn_get_mscale(scale=1, mscale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class IPTMLAttention(nn.Module):
    def __init__(self, config: IPTConfig, layer_idx: Optional[int] = None):
        super().__init__()

        self.config = config
        self.config._attn_implementation = "flash_attention_2"
        self.layer_idx = layer_idx
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.attention_dropout = config.attention_dropout
        self.num_heads = config.num_attention_heads
        self.rope_theta = config.rope_theta
        self.apply_q_lora = config.apply_q_lora
        self.q_lora_rank = config.q_lora_rank
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.kv_lora_rank = config.kv_lora_rank
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self.clamp_input_value = 0
        if hasattr(config, "clamp_training"):
            clamp_training_cfg = config.clamp_training
            self.clamp_input_value = clamp_training_cfg["clamp_input_value"]

        self.is_causal = True
        if not self.apply_q_lora:
            self.q_up_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_down_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
            self.q_down_layernorm = IPTRMSNorm(config.q_lora_rank, eps=config.layernorm_epsilon)
            self.q_up_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.kv_down_proj_with_mqa = nn.Linear(
            config.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_down_layernorm = IPTRMSNorm(self.kv_lora_rank, eps=config.layernorm_epsilon)
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )

        self.scaling = self.qk_head_dim ** (-0.5)
        if self.config.rope_scaling and "yarn" in self.config.rope_scaling.get("rope_type", "default"):
            mscale_all_dim = self.config.rope_scaling.get("mscale_all_dim", 0)
            scaling_factor = self.config.rope_scaling["factor"]
            if mscale_all_dim:
                mscale = yarn_get_mscale(scaling_factor, mscale_all_dim)
                self.scaling = self.scaling * mscale * mscale
        self.rotary_emb = IPTRotaryEmbedding(config=config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
        batch_size, seq_length = hidden_states.shape[:-1]
        query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
        key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

        if self.clamp_input_value > 0:
            hidden_states = torch.clamp_(hidden_states, -self.clamp_input_value, self.clamp_input_value)

        if not self.apply_q_lora:
            q_states = self.q_up_proj(hidden_states)
        else:
            q_states = self.q_down_layernorm(self.q_down_proj(hidden_states))
            if self.clamp_input_value > 0:
                q_states = torch.clamp_(q_states, -self.clamp_input_value, self.clamp_input_value)
            q_states = self.q_up_proj(q_states)
        q_states = q_states.view(query_shape).transpose(1, 2)
        q_nope, q_pe = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        compressed_kv = self.kv_down_proj_with_mqa(hidden_states)
        kv_a, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

        kv_a = self.kv_down_layernorm(kv_a)
        if self.clamp_input_value > 0:
            kv_a = torch.clamp_(kv_a, -self.clamp_input_value, self.clamp_input_value)
        kv_a = self.kv_up_proj(kv_a).view(key_shape).transpose(1, 2)
        k_nope, value_states = torch.split(kv_a, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings

        # MLA use float32 rope embedding
        origin_dtype = q_pe.dtype
        q_pe = q_pe.float()
        k_pe = k_pe.float()
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin)
        q_pe = q_pe.to(origin_dtype)
        k_pe = k_pe.to(origin_dtype)

        k_pe = k_pe.expand(*k_nope.shape[:-1], -1)

        query_states = torch.cat((q_nope, q_pe), dim=-1)
        key_states = torch.cat((k_nope, k_pe), dim=-1)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

        if (
            self.config.use_sliding_window and getattr(self.config, "sliding_window", None) is not None
            # and self.layer_idx >= self.config.max_window_layers
        ):
            sliding_window = self.config.sliding_window
        else:
            sliding_window = None

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=sliding_window,
            position_ids=position_ids,
            **kwargs,
        )

        if self.config._attn_implementation == "flash_attention_2" and self.qk_head_dim != self.v_head_dim:
            attn_output = attn_output[:, :, :, : self.v_head_dim]

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        if self.clamp_input_value > 0:
            attn_output = torch.clamp_(attn_output, -self.clamp_input_value, self.clamp_input_value)

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value


class IPTDecoderLayer(nn.Module):
    def __init__(self, config: IPTConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        if config.sliding_window and config._attn_implementation != "flash_attention_2":
            logger.warning_once(
                f"Sliding Window Attention is enabled but not implemented for `{config._attn_implementation}`; "
                "unexpected results may be encountered."
            )

        if config.apply_mla:
            self.attention = IPTMLAttention(config, layer_idx)
        else:
            self.attention = IPTAttention(config, layer_idx)

        self.apply_gmoe = config.apply_gmoe
        self.use_moe = False
        if layer_idx >= config.skip_first_n_layers:
            self.use_moe = True
        if self.apply_gmoe and self.use_moe:
            self.mlp = IPTMoE(config)
        else:
            self.mlp = IPTMLP(config)
        self.layer_norm = IPTRMSNorm(config.hidden_size, eps=config.layernorm_epsilon)
        self.final_layer_norm = IPTRMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        routing_map: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
            cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
                Indices depicting the position of the input sequence tokens in the sequence
            position_embeddings (`Tuple[torch.FloatTensor, torch.FloatTensor]`, *optional*):
                Tuple containing the cosine and sine positional embeddings of shape `(batch_size, seq_len, head_dim)`,
                with `head_dim` being the embedding dimension of each attention head.
            kwargs (`dict`, *optional*):
                Arbitrary kwargs to be ignored, used for FSDP and other methods that injects code
                into the model
        """
        orig_dtype = hidden_states.dtype
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, past_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

        hidden_states = (residual.float() + hidden_states.float()).to(orig_dtype)

        # Fully Connected
        residual = hidden_states

        hidden_states = self.final_layer_norm(hidden_states)
        # hidden_states = self.mlp(hidden_states)
        # MLP forward with optional routing_map
        if hasattr(self.mlp, "forward") and "routing_map" in self.mlp.forward.__code__.co_varnames:
            hidden_states = self.mlp(hidden_states, routing_map=routing_map)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = (residual.float() + hidden_states.float()).to(orig_dtype)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)
        outputs += (past_key_value,)

        return outputs


class IPTDecoder(nn.Module):
    def __init__(self, config: IPTConfig):
        super().__init__()
        self.layers = nn.ModuleList(
            [IPTDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        for i, layer in enumerate(self.layers):
            layer.layer_indice = i
        self.layernorm = IPTRMSNorm(config.hidden_size, eps=config.layernorm_epsilon)

    def forward(
        self,
        hidden_states: torch.LongTensor = None,
        causal_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        routing_maps=None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer, routing_map in zip(self.layers, routing_maps):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                routing_map=routing_map,
                **kwargs,
            )
            # print("layer_outputs: ", layer_outputs)
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        return hidden_states, next_cache, all_hidden_states, all_self_attns


class ExEmbedding(nn.Module):
    def __init__(self, config, padding_idx):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx)

    def forward(self, input_):
        return self.word_embeddings(input_)


class IPTPreTrainedModel(PreTrainedModel):
    config_class = IPTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["IPTDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class IPTModel(IPTPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`IPTDecoderLayer`]

    Args:
        config: IPTConfig
    """

    def __init__(self, config: IPTConfig):
        super().__init__(config)

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.gradient_checkpointing = False

        self.embedding = ExEmbedding(config, self.padding_idx)
        self.transformer = IPTDecoder(config)
        self.rotary_emb = IPTRotaryEmbedding(config=config)

        self.post_init()

    def get_input_embeddings(self):
        return self.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.embedding.word_embeddings = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        routing_maps=None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        # breakpoint()
        if routing_maps is not None:
            routing_maps = routing_maps.permute(2, 0, 1, 3)
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = self._update_causal_mask(
            attention_mask,
            inputs_embeds,
            cache_position,
            past_key_values,
            output_attentions,
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if routing_maps is None:
            routing_maps = [None] * self.config.num_hidden_layers

        hidden_states, next_cache, all_hidden_states, all_self_attns = self.transformer(
            hidden_states,
            causal_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            routing_maps=routing_maps,
            **kwargs,
        )
        # breakpoint()
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()

        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: IPTConfig,
        past_key_values: Cache,
    ):
        """
        Creates a causal 4D mask of shape `(batch_size, 1, query_length, key_value_length)` from a 2D mask of shape
        `(batch_size, key_value_length)`, or if the input `attention_mask` is already 4D, do nothing.

        Args:
            attention_mask (`torch.Tensor`):
                A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
            sequence_length (`int`):
                The sequence length being processed.
            target_length (`int`):
                The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
            dtype (`torch.dtype`):
                The dtype to use for the 4D attention mask.
            device (`torch.device`):
                The device to plcae the 4D attention mask on.
            cache_position (`torch.Tensor`):
                Indices depicting the position of the input sequence tokens in the sequence.
            batch_size (`torch.Tensor`):
                Batch size.
            config (`IPT2Config`):
                The model's configuration class
            past_key_values (`Cache`):
                The cache class that is being used currently to generate
        """
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            if config.sliding_window is not None:
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)
            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        return causal_mask

    def _fuse_experts(self):
        for layer in tqdm(self.transformer.layers, desc="Fusing experts"):
            if isinstance(layer.mlp, IPTMoE) and not layer.mlp.grouped_gemm:
                grouped_experts = GroupedMoEMLP(layer.mlp.config)

                fc1_weights = torch.stack([expert.fc1.weight.T for expert in layer.mlp.routed_experts])
                fc2_weights = torch.stack([expert.fc2.weight.T for expert in layer.mlp.routed_experts])

                grouped_experts.fc1_weights.data = fc1_weights
                grouped_experts.fc2_weights.data = fc2_weights

                layer.mlp.routed_experts = grouped_experts
                layer.mlp.grouped_gemm = True

    def _unfuse_experts(self):
        for layer in tqdm(self.transformer.layers, desc="Unfusing experts"):
            if isinstance(layer.mlp, IPTMoE) and layer.mlp.grouped_gemm:
                grouped_experts = layer.mlp.routed_experts
                config = layer.mlp.config
                num_groups = config.num_groups
                num_experts_per_group = config.num_routed_experts
                num_shared_experts_per_group = config.num_shared_experts
                num_routed_experts_per_group = config.num_routed_experts - config.num_shared_experts
                num_routed_experts = num_routed_experts_per_group * num_groups

                experts = nn.ModuleList(
                    [IPTMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(num_routed_experts)]
                )

                experts_fc1_weights = layer.mlp.routed_experts.fc1_weights.permute(0, 2, 1)
                experts_fc2_weights = layer.mlp.routed_experts.fc2_weights.permute(0, 2, 1)
                for idx in range(num_routed_experts):
                    group_id = idx // num_routed_experts_per_group
                    expert_offset = num_shared_experts_per_group + (idx % num_routed_experts_per_group)
                    global_idx = group_id * num_experts_per_group + expert_offset

                    experts[idx].fc1.weight.data.copy_(experts_fc1_weights[global_idx].data)
                    experts[idx].fc2.weight.data.copy_(experts_fc2_weights[global_idx].data)

                shared_indices = (
                    torch.arange(num_groups * num_experts_per_group)
                    .view(num_groups, -1)[:, :num_shared_experts_per_group]
                    .reshape(-1)
                )
                shared_fc1_weights = experts_fc1_weights[shared_indices].reshape(-1, config.hidden_size)
                shared_fc2_weights = (
                    experts_fc2_weights[shared_indices].permute(1, 0, 2).reshape(config.hidden_size, -1)
                )

                shared_experts = IPTMLP(
                    config, intermediate_size=config.moe_intermediate_size * num_shared_experts_per_group * num_groups
                )
                shared_experts.fc1.weight.data.copy_(shared_fc1_weights.data)
                shared_experts.fc2.weight.data.copy_(shared_fc2_weights.data)

                router = IPTRouter(config)

                grouped_gating_weight = layer.mlp.router.gating.weight
                grouped_gating_weight = grouped_gating_weight.view(num_groups, -1, config.hidden_size)
                grouped_gating_weight = (
                    grouped_gating_weight[:, num_shared_experts_per_group:, :]
                    .contiguous()
                    .reshape(-1, config.hidden_size)
                )
                router.gating.weight.data.copy_(grouped_gating_weight.data)
                router.e_score_correction_bias.data.copy_(layer.mlp.router.e_score_correction_bias.data)

                layer.mlp.routed_experts = experts
                layer.mlp.grouped_gemm = False
                layer.mlp.shared_experts = shared_experts
                layer.mlp.router = router

    def _unfuse_padding_experts(self):
        print("unfuse padding here")
        for layer in tqdm(self.transformer.layers, desc="Unfusing experts"):
            if isinstance(layer.mlp, IPTMoE) and layer.mlp.grouped_gemm:
                grouped_experts = layer.mlp.routed_experts
                config = layer.mlp.config
                num_groups = config.num_groups
                num_experts_per_group = config.num_routed_experts
                num_shared_experts_per_group = config.num_shared_experts
                num_routed_experts_per_group = config.num_routed_experts - config.num_shared_experts
                num_padded_experts_per_group = 1
                num_routed_experts = (num_routed_experts_per_group + num_padded_experts_per_group) * num_groups

                experts = nn.ModuleList(
                    [IPTMLP(config, intermediate_size=config.moe_intermediate_size) for _ in range(num_routed_experts)]
                )

                experts_fc1_weights = layer.mlp.routed_experts.fc1_weights.permute(0, 2, 1)
                experts_fc2_weights = layer.mlp.routed_experts.fc2_weights.permute(0, 2, 1)
                for idx in range(num_routed_experts):
                    group_id = idx // (num_routed_experts_per_group + num_padded_experts_per_group)
                    expert_offset = num_shared_experts_per_group + idx % (
                        num_routed_experts_per_group + num_padded_experts_per_group
                    )
                    global_idx = (
                        group_id * (num_routed_experts_per_group + num_padded_experts_per_group) + expert_offset
                    )
                    if (idx + 1) % (num_routed_experts_per_group + num_padded_experts_per_group) == 0:
                        # print(f'current padding {idx}, {global_idx}')
                        experts[idx].fc1.weight.data.fill_(0.0)
                        experts[idx].fc2.weight.data.fill_(0.0)
                    else:
                        experts[idx].fc1.weight.data.copy_(experts_fc1_weights[global_idx].data)
                        experts[idx].fc2.weight.data.copy_(experts_fc2_weights[global_idx].data)

                shared_indices = (
                    torch.arange(num_groups * num_experts_per_group)
                    .view(num_groups, -1)[:, :num_shared_experts_per_group]
                    .reshape(-1)
                )
                shared_fc1_weights = experts_fc1_weights[shared_indices].reshape(-1, config.hidden_size)
                shared_fc2_weights = (
                    experts_fc2_weights[shared_indices].permute(1, 0, 2).reshape(config.hidden_size, -1)
                )

                shared_experts = IPTMLP(
                    config, intermediate_size=config.moe_intermediate_size * num_shared_experts_per_group * num_groups
                )
                shared_experts.fc1.weight.data.copy_(shared_fc1_weights.data)
                shared_experts.fc2.weight.data.copy_(shared_fc2_weights.data)

                config.num_routed_experts += num_padded_experts_per_group  # pading experts
                router = IPTRouter(config)
                config.num_routed_experts -= num_padded_experts_per_group  # pading experts

                grouped_gating_weight = layer.mlp.router.gating.weight
                grouped_gating_weight = grouped_gating_weight.view(num_groups, -1, config.hidden_size)
                # padd experts
                num_local_experts_pergroup = grouped_gating_weight.shape[1] - num_shared_experts_per_group
                num_pad_experts = num_padded_experts_per_group
                grouped_gating_weight = F.pad(grouped_gating_weight, (0, 0, 0, num_pad_experts, 0, 0), value=0)
                grouped_gating_weight = (
                    grouped_gating_weight[:, num_shared_experts_per_group:, :]
                    .contiguous()
                    .reshape(-1, config.hidden_size)
                )
                router.gating.weight.data.copy_(grouped_gating_weight.data)
                e_score_correction_bias = layer.mlp.router.e_score_correction_bias
                e_score_correction_bias = e_score_correction_bias.view(-1, num_local_experts_pergroup)
                e_score_correction_bias = (
                    F.pad(e_score_correction_bias, (0, num_pad_experts, 0, 0), value=0).contiguous().reshape(-1)
                )
                router.e_score_correction_bias.data.copy_(e_score_correction_bias)

                layer.mlp.routed_experts = experts
                layer.mlp.grouped_gemm = False
                layer.mlp.shared_experts = shared_experts
                layer.mlp.router = router


class IPTForCausalLM(IPTPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        print("attn type: ", config._attn_implementation)
        self.model = IPTModel(config)
        self.vocab_size = config.vocab_size
        self.clamp_input_value = 0
        if hasattr(config, "clamp_training"):
            clamp_training_cfg = config.clamp_training
            self.clamp_input_value = clamp_training_cfg["embed_state"]["clamp_input_value"]

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embedding.word_embeddings

    def set_input_embeddings(self, value):
        self.model.embedding.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: int = 0,
        routing_ids=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            num_logits_to_keep (`int`, *optional*):
                Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # print_input_shapes(input_ids, attention_mask, position_ids, routing_ids, tag='[cx_debug][spark_routing_model]')
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            routing_maps=routing_ids,
        )

        hidden_states = outputs.last_hidden_state

        if labels is None and not is_torchdynamo_compiling():
            logger.warning_once(
                "Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)"
            )

        if self.clamp_input_value > 0:
            hidden_states = torch.clamp_(hidden_states, -self.clamp_input_value, self.clamp_input_value)

        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            # Upcast to float if we need to compute the loss to avoid potential precision issues
            logits = logits.float()
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        num_logits_to_keep=None,
        **kwargs,
    ):
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {
                "input_ids": input_ids.clone(memory_format=torch.contiguous_format),
                "inputs_embeds": None,
            }

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        if num_logits_to_keep is not None:
            model_inputs["num_logits_to_keep"] = num_logits_to_keep

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def fuse_experts(self):
        import importlib.util

        if importlib.util.find_spec("grouped_gemm") is None:
            raise ImportError(
                "Please install grouped_gemm to use grouped_gemm=True. "
                "You can install it with `pip install git+https://git-in.iflytek.com/RS_RDG_AI_Public_Group/grouped-gemm@v1.1.4i`."
            )
        self.model._fuse_experts()

    def unfuse_experts(self, use_padding_expert=False):
        if not use_padding_expert:
            self.model._unfuse_experts()
        else:
            self.model._unfuse_padding_experts()
