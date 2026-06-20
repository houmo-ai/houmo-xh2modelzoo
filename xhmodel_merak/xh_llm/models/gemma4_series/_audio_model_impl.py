import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4AudioAttention,
    Gemma4AudioLightConv1d,
    Gemma4AudioModel,
)
from xhquant.nn import MaskedAdd
from xhquant.utils.registry import DynamicModule

from ...register import XHLLM_TRACEABLE_MODULES


class _Gemma4AudioDynamicModule(DynamicModule):
    def _setup(self, cfg=None):
        return None


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4AudioAttention: "Gemma4AudioAttention"})
class _Gemma4AudioAttention(_Gemma4AudioDynamicModule):
    def _convert_to_block(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        hidden_states = F.pad(hidden_states, (0, 0, 0, 0, 0, pad))
        return hidden_states.reshape(batch_size, num_blocks, self.chunk_size, num_heads, head_dim).contiguous()

    def _extract_block_context(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, num_heads, head_dim = hidden_states.shape
        num_blocks = (seq_len + self.chunk_size - 1) // self.chunk_size
        pad = num_blocks * self.chunk_size - seq_len
        hidden_states = F.pad(
            hidden_states,
            (0, 0, 0, 0, self.max_past_horizon, self.max_future_horizon + self.chunk_size - 1 + pad),
        )
        block_starts = torch.arange(num_blocks, device=hidden_states.device, dtype=torch.long) * self.chunk_size
        offsets = torch.arange(self.context_size, device=hidden_states.device, dtype=torch.long)
        gather_indices = (block_starts[:, None] + offsets[None, :]).reshape(-1)
        hidden_states = torch.index_select(hidden_states, 1, gather_indices)
        return hidden_states.reshape(batch_size, num_blocks, self.context_size, num_heads, head_dim).contiguous()

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_heads, num_blocks, block_size, position_length = x.shape
        context_size = self.context_size
        x = F.pad(x, (0, context_size + 1 - position_length))
        x = x.view(batch_size, num_heads, num_blocks, block_size * (context_size + 1))
        x = x[..., : block_size * context_size]
        return x.view(batch_size, num_heads, num_blocks, block_size, context_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: torch.Tensor,
        attention_mask: torch.BoolTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_length, _ = hidden_states.shape
        hidden_shape = (batch_size, seq_length, self.num_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).float().view(hidden_shape)
        key_states = self.k_proj(hidden_states).float().view(hidden_shape)
        value_states = self.v_proj(hidden_states).float().view(hidden_shape)

        query_states = query_states * self.q_per_dim_scale.to(query_states)
        key_states = key_states * self.k_scale

        query_states = self._convert_to_block(query_states)
        key_states = self._extract_block_context(key_states)
        value_states = self._extract_block_context(value_states)
        num_blocks = query_states.shape[1]

        relative_key_states = self.relative_k_proj(position_embeddings)
        relative_key_states = relative_key_states.view(-1, self.num_heads, self.head_dim)
        relative_key_states = relative_key_states.to(dtype=query_states.dtype)

        queries = query_states.permute(0, 3, 1, 2, 4)
        matrix_ac = queries @ key_states.permute(0, 3, 1, 4, 2)

        queries_flat = queries.reshape(batch_size, self.num_heads, -1, self.head_dim)
        matrix_bd = queries_flat @ relative_key_states.permute(1, 2, 0)
        matrix_bd = matrix_bd.reshape(batch_size, self.num_heads, num_blocks, self.chunk_size, -1)
        matrix_bd = self._rel_shift(matrix_bd)

        attn_weights = matrix_ac + matrix_bd
        attn_weights = attn_weights / self.softcap
        attn_weights = torch.tanh(attn_weights)
        attn_weights = attn_weights * self.softcap

        if attention_mask is not None:
            attn_weights = self.masked_add(attn_weights, attention_mask.to(attn_weights.dtype))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_states.dtype)
        attn_output = attn_weights @ value_states.permute(0, 3, 1, 2, 4)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, num_blocks * self.chunk_size, -1)
        attn_output = attn_output[:, :seq_length].contiguous()
        attn_output = self.post(attn_output.to(dtype=self.post.linear.weight.dtype))

        return attn_output, attn_weights

    def _setup(self, cfg=None):
        self.attention_logits_soft_cap = self.config.attention_logit_cap
        self.head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.num_heads = self.config.num_attention_heads
        self.q_scale = (self.head_dim**-0.5) / math.log(2)
        self.k_scale = math.log(1 + math.e) / math.log(2)
        q_per_dim_scale = (F.softplus(self.per_dim_scale.detach()) * self.q_scale).to(self.per_dim_scale.dtype)
        if "q_per_dim_scale" in self._buffers:
            self.q_per_dim_scale = q_per_dim_scale
        else:
            self.register_buffer("q_per_dim_scale", q_per_dim_scale)
        self.chunk_size = self.config.attention_chunk_size
        self.max_past_horizon = self.config.attention_context_left - 1
        self.max_future_horizon = self.config.attention_context_right
        self.context_size = self.chunk_size + self.max_past_horizon + self.max_future_horizon
        self.masked_add = MaskedAdd()


@XHLLM_TRACEABLE_MODULES.register_module(
    {Gemma4AudioLightConv1d: "Gemma4AudioLightConv1d"}
)
class _Gemma4AudioLightConv1d(_Gemma4AudioDynamicModule):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states

        hidden_states = self.pre_layer_norm(hidden_states)
        if self.linear_start_use_clipped_linears:
            hidden_states = torch.clamp(
                hidden_states,
                self.linear_start_input_min,
                self.linear_start_input_max,
            )

        value = self.linear_start_value(hidden_states)
        gate = self.linear_start_gate(hidden_states)
        if self.linear_start_use_clipped_linears:
            value = torch.clamp(
                value,
                self.linear_start_output_min,
                self.linear_start_output_max,
            )
            gate = torch.clamp(
                gate,
                self.linear_start_output_min,
                self.linear_start_output_max,
            )
        hidden_states = value * torch.sigmoid(gate)

        hidden_states = self.depthwise_conv1d(
            hidden_states.transpose(1, 2)
        ).transpose(1, 2)

        gradient_clipping = min(
            self.gradient_clipping,
            torch.finfo(self.linear_start_value.weight.dtype).max,
        )
        hidden_states = torch.clamp(
            hidden_states,
            -gradient_clipping,
            gradient_clipping,
        )
        hidden_states = self.conv_norm(hidden_states)

        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_end(hidden_states)
        hidden_states += residual
        return hidden_states

    def _setup(self, cfg=None):
        start = self.linear_start
        linear = start.linear
        hidden_size = linear.out_features // 2
        has_bias = linear.bias is not None
        self.linear_start_value = nn.Linear(
            linear.in_features,
            hidden_size,
            bias=has_bias,
        )
        self.linear_start_gate = nn.Linear(
            linear.in_features,
            hidden_size,
            bias=has_bias,
        )
        with torch.no_grad():
            self.linear_start_value.weight.copy_(linear.weight[:hidden_size])
            self.linear_start_gate.weight.copy_(linear.weight[hidden_size:])
            if has_bias:
                self.linear_start_value.bias.copy_(linear.bias[:hidden_size])
                self.linear_start_gate.bias.copy_(linear.bias[hidden_size:])
        self.linear_start_value = self.linear_start_value.to(
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        self.linear_start_gate = self.linear_start_gate.to(
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        self.linear_start_use_clipped_linears = bool(
            getattr(start, "use_clipped_linears", False)
        )
        if self.linear_start_use_clipped_linears:
            self.register_buffer(
                "linear_start_input_min",
                start.input_min.detach().clone(),
            )
            self.register_buffer(
                "linear_start_input_max",
                start.input_max.detach().clone(),
            )
            self.register_buffer(
                "linear_start_output_min",
                start.output_min.detach().clone(),
            )
            self.register_buffer(
                "linear_start_output_max",
                start.output_max.detach().clone(),
            )
        return None


@XHLLM_TRACEABLE_MODULES.register_module({Gemma4AudioModel: "Gemma4AudioModel"})
class _Gemma4AudioModel(_Gemma4AudioDynamicModule):
    def _build_sliding_attention_mask(self, valid_positions: torch.Tensor) -> torch.Tensor:
        seq_len = valid_positions.shape[-1]
        device = valid_positions.device
        left_window_size = self.config.attention_context_left - 1
        right_window_size = self.config.attention_context_right

        query_idx = torch.arange(seq_len, device=device, dtype=torch.int32).view(1, 1, seq_len, 1)
        key_idx = torch.arange(seq_len, device=device, dtype=torch.int32).view(1, 1, 1, seq_len)
        dist = query_idx - key_idx
        window_mask = ((dist >= 0) & (dist < left_window_size)) | ((dist < 0) & (-dist < right_window_size))
        valid_mask = valid_positions[:, None, :, None] & valid_positions[:, None, None, :]
        return valid_mask & window_mask

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        audio_attention_mask: torch.Tensor | None = None,
    ):
        hidden_states, output_mask = self.subsample_conv_projection(input_features, attention_mask)
        position_embeddings = self.rel_pos_enc(hidden_states)

        if audio_attention_mask is None:
            # Compatibility path for direct HF-style calls. Exported graphs pass
            # this as an explicit additive mask input to keep bool mask building
            # out of the NPU graph.
            attention_mask = self._convert_4d_mask_to_blocked_5d(
                self._build_sliding_attention_mask(output_mask)
            )
            neg = torch.tensor(
                torch.finfo(hidden_states.dtype).min,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            attention_mask = torch.zeros(
                attention_mask.shape,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            ).masked_fill(~attention_mask, neg)
        else:
            attention_mask = audio_attention_mask

        for encoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = encoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
            )

        hidden_states = self.output_proj(hidden_states)
        return hidden_states, output_mask


def register_wrap_modules(hf_model=None):
    return None


register_wrap_cls = register_wrap_modules
