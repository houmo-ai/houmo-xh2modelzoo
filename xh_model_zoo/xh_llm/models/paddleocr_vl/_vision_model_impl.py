# -*- coding: utf-8 -*-
# Copyright 2025 PaddlePaddle Authors. All rights reserved.
# Copyright 2025 HOUMO AI. All rights reserved.
#
# Modifications:
# - Portions of this file have been modified by HOUMO AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File: _vision_model_impl.py
# Description:
#   PaddleOCR-VL vision-tower model adapted for the xh2 model zoo (xh2modelzoo).

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling
from xhquant import nn as xhnn
from xhquant.api import ConfigDict
from xhquant.utils.registry import DynamicModule

from ..builder import XHLLM_TRACEABLE_MODULES
from .modeling_paddleocr_vl import (
    SiglipAttention,
    SiglipEncoder,
    SiglipEncoderLayer,
    SiglipMLP,
    SiglipVisionEmbeddings,
    SiglipVisionTransformer,
    rotate_half,
)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        SiglipAttention: "SiglipAttention",
    }
)
class _SiglipAttention(DynamicModule):
    """Optimized SiglipAttention with torch.fx compatible forward pass

    Design: Inspired by Qwen3-VL but maintains cu_seqlens support
    Key simplifications:
    - rope_emb is precomputed (no height/width position ID calculation here)
    - cu_seqlens handling deferred to calling code
    - Pure eager attention for torch.fx traceability
    """

    def _setup(self, cfg: ConfigDict):
        """Setup: apply padding for optimization (V and output only, not Q/K to preserve RoPE compatibility)"""
        # Get head dimension
        embed_dim = self.embed_dim
        num_heads = self.num_heads
        head_dim = embed_dim // num_heads

        self.scale = head_dim**-0.5

        # Apply head dimension padding for 64-bit alignment optimization
        # BUT: Only apply to V and output projection, NOT to Q/K to preserve RoPE compatibility
        if head_dim % 64 != 0:
            new_head_dim = math.ceil(head_dim / 64) * 64
            padding_size = new_head_dim - head_dim

            device = self.v_proj.weight.device
            weight_dtype = self.v_proj.weight.dtype
            bias_dtype = self.v_proj.bias.dtype

            # Expand V projection with full padding
            v_weight = self.v_proj.weight.data.transpose(0, 1).reshape(
                embed_dim, num_heads, head_dim
            )
            padded_v_weight = torch.zeros(
                (embed_dim, num_heads, head_dim + padding_size),
                device=device,
                dtype=weight_dtype,
            )
            padded_v_weight[:, :, :head_dim] = v_weight
            padded_v_weight = padded_v_weight.reshape(
                embed_dim, num_heads * new_head_dim
            ).transpose(0, 1)

            v_bias = self.v_proj.bias.data.reshape(num_heads, head_dim)
            padded_v_bias = torch.zeros(
                (num_heads, head_dim + padding_size),
                device=device,
                dtype=bias_dtype,
            )
            padded_v_bias[:, :head_dim] = v_bias
            padded_v_bias = padded_v_bias.reshape(num_heads * new_head_dim)

            expanded_v = nn.Linear(embed_dim, num_heads * new_head_dim, bias=True)
            expanded_v = expanded_v.to(device=device, dtype=weight_dtype)
            expanded_v.weight.data.copy_(padded_v_weight)
            expanded_v.bias.data.copy_(padded_v_bias.to(weight_dtype))
            self.v_proj = expanded_v

            # Expand output projection
            padded_proj_weight = torch.zeros(
                (self.out_proj.out_features, num_heads, new_head_dim),
                device=device,
                dtype=weight_dtype,
            )
            padded_proj_weight[:, :, :head_dim] = self.out_proj.weight.data.reshape(
                self.out_proj.out_features, num_heads, head_dim
            )
            padded_proj_weight = padded_proj_weight.reshape(
                self.out_proj.out_features, num_heads * new_head_dim
            )
            proj_bias = self.out_proj.bias.data.clone()

            expanded_proj = nn.Linear(
                num_heads * new_head_dim, self.out_proj.out_features, bias=True
            )
            expanded_proj = expanded_proj.to(device=device, dtype=weight_dtype)
            expanded_proj.weight.data.copy_(padded_proj_weight)
            expanded_proj.bias.data.copy_(proj_bias.to(weight_dtype))
            self.out_proj = expanded_proj

            # Note: Q/K are NOT expanded to maintain RoPE compatibility
            # The attention computation will handle different Q/K vs V dimensions

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        cu_seqlens: Optional[List[torch.Tensor]] = None,
        rope_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for torch.fx traceability

        Args:
            hidden_states: [batch, seq_len, embed_dim]
            attention_mask: Optional attention mask
            output_attentions: Whether to output attention weights
            cu_seqlens: Cumulative sequence lengths (passed through, not used)
            rope_emb: Optional (cos, sin) for rotary position embeddings
                     If None, RoPE is not applied (for non-vision or non-positional cases)

        Note: RoPE application is conditional on rope_emb being not None (Python-level check OK for torch.fx)
        """
        batch_size, seq_length, embed_dim = hidden_states.shape

        # Project Q, K, V
        queries = self.q_proj(hidden_states)
        keys = self.k_proj(hidden_states)
        values = self.v_proj(hidden_states)

        # Get original and potentially padded dimensions
        qk_head_dim = queries.shape[-1] // self.num_heads
        v_head_dim = values.shape[-1] // self.num_heads

        # Reshape for attention: [batch, seq, embed_dim] -> [batch, heads, seq, head_dim]
        queries = queries.view(
            batch_size, seq_length, self.num_heads, qk_head_dim
        ).transpose(1, 2)
        keys = keys.view(batch_size, seq_length, self.num_heads, qk_head_dim).transpose(
            1, 2
        )
        values = values.view(
            batch_size, seq_length, self.num_heads, v_head_dim
        ).transpose(1, 2)

        # Apply RoPE if provided (Python-level None check is OK for torch.fx)
        if rope_emb is not None:
            cos, sin = rope_emb
            cos = cos.unsqueeze(0)
            sin = sin.unsqueeze(0)
            # Apply rotary embeddings
            queries = (queries * cos) + (rotate_half(queries) * sin)
            keys = (keys * cos) + (rotate_half(keys) * sin)

        # Scaled dot-product attention
        keys_t = keys.transpose(-2, -1)
        queries_scaled = queries * self.scale

        # Compute attention weights
        attn_weights = torch.matmul(queries_scaled, keys_t)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            queries.dtype
        )

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, values)

        # Reshape back: [batch, heads, seq, head_dim] -> [batch, seq, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_length, -1)

        # Output projection
        attn_output = self.out_proj(attn_output)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs

    @staticmethod
    def _rotate_half(x):
        """Rotate half the hidden dims of the input."""
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        SiglipMLP: "SiglipMLP",
    }
)
class _SiglipMLP(DynamicModule):
    """Optimized SiglipMLP with simplified forward pass"""

    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        """No special setup needed for MLP"""
        pass

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Simple feed-forward pass"""
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        SiglipEncoderLayer: "SiglipEncoderLayer",
    }
)
class _SiglipEncoderLayer(DynamicModule):
    """Optimized SiglipEncoderLayer with simplified forward pass"""

    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        """No special setup needed for encoder layer"""
        pass

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
        cu_seqlens: Optional[List[torch.Tensor]] = None,
        rope_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.FloatTensor]:
        """Pre-norm residual connections for torch.fx compatibility"""
        # Self-attention with residual
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_outputs = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            cu_seqlens=cu_seqlens,
            rope_emb=rope_emb,
        )
        hidden_states = residual + attn_outputs[0]

        # MLP with residual
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (attn_outputs[1],)

        return outputs


# @XHLLM_TRACEABLE_MODULES.register_module(
#     {
#         SiglipEncoder: "SiglipEncoder",
#     }
# )
# class _SiglipEncoder(DynamicModule):
#     """Optimized SiglipEncoder with torch.fx compatible forward pass

#     Key Design:
#     - Does NOT compute RoPE - receives pre-computed rope_emb from caller
#     - Simple layer loop with no tensor operations
#     - 100% torch.fx compatible
#     """

#     @torch.no_grad()
#     def _setup(self, cfg: ConfigDict):
#         """Setup encoder-specific configurations"""
#         pass

#     def forward(
#         self,
#         inputs_embeds: torch.Tensor,
#         attention_mask: Optional[torch.Tensor] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         cu_seqlens: Optional[List[torch.Tensor]] = None,
#         image_grid_thw: Optional[List] = None,
#         height_position_ids: Optional[torch.Tensor] = None,
#         width_position_ids: Optional[torch.Tensor] = None,
#         **kwargs,
#     ) -> BaseModelOutput:
#         """Forward pass for torch.fx compatibility

#         CRITICAL: rope_emb is pre-computed by _SiglipVisionTransformer
#         This wrapper just does layer iteration - no complex logic
#         """
#         output_attentions = (
#             output_attentions
#             if output_attentions is not None
#             else self.config.output_attentions
#         )
#         output_hidden_states = (
#             output_hidden_states
#             if output_hidden_states is not None
#             else self.config.output_hidden_states
#         )

#         encoder_states = () if output_hidden_states else None
#         all_attentions = () if output_attentions else None

#         hidden_states = inputs_embeds
#         attention_mask = (
#             attention_mask.to(inputs_embeds.dtype)
#             if attention_mask is not None
#             else None
#         )

#         # Get rope_emb from kwargs (pre-computed by caller)
#         rope_emb = kwargs.get("rope_emb", None)

#         for encoder_layer in self.layers:
#             if output_hidden_states:
#                 encoder_states = encoder_states + (hidden_states,)

#             if self.gradient_checkpointing and self.training:
#                 layer_outputs = self._gradient_checkpointing_func(
#                     encoder_layer.__call__,
#                     hidden_states,
#                     attention_mask,
#                     output_attentions,
#                     cu_seqlens,
#                     rope_emb,
#                 )
#             else:
#                 layer_outputs = encoder_layer(
#                     hidden_states,
#                     attention_mask,
#                     output_attentions=output_attentions,
#                     cu_seqlens=cu_seqlens,
#                     rope_emb=rope_emb,
#                 )

#             hidden_states = layer_outputs[0]

#             if output_attentions:
#                 all_attentions = all_attentions + (layer_outputs[1],)

#         if output_hidden_states:
#             encoder_states = encoder_states + (hidden_states,)

#         return BaseModelOutput(
#             last_hidden_state=hidden_states,
#             hidden_states=encoder_states,
#             attentions=all_attentions,
#         )


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        SiglipVisionEmbeddings: "SiglipVisionEmbeddings",
    }
)
class _SiglipVisionEmbeddings(DynamicModule):
    """Optimized SiglipVisionEmbeddings with simplified forward pass"""

    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        """Setup vision embeddings"""
        # Cache dimension information for efficient forward pass
        self.embed_dim = self.config.hidden_size
        self.image_size = self.config.image_size
        self.patch_size = self.config.patch_size
        self.num_patches = (self.image_size // self.patch_size) ** 2

    def forward(
        self,
        pixel_values: torch.Tensor,
        interpolate_pos_encoding: Optional[bool] = False,
        position_ids: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[
            List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]
        ] = None,
    ) -> torch.Tensor:
        """Embedding forward pass with support for 5D input.

        Handles two 5D formats:
        - [B, num_patches, C, patch_h, patch_w]: per-patch input (native HF format)
        - [B, 1, C, H, W]: full-image input (ONNX export format)

        For full-image input (sequence_len == 1), Conv2d produces a multi-patch
        feature map that is flattened to [B, num_patches, embed_dim].
        For per-patch input, Conv2d produces a 1×1 feature per patch.
        """
        if pixel_values.dim() == 5:
            from einops import rearrange

            batch_size, sequence_len, channel, height, width = pixel_values.shape
            target_dtype = self.patch_embedding.weight.dtype

            # Flatten temporal/sequence dimension for patch embedding
            pixel_values_flat = rearrange(pixel_values, "b l c h w -> (b l) c h w")
            patch_embeds = self.patch_embedding(
                pixel_values_flat.to(dtype=target_dtype)
            )
            # patch_embeds: [(B*L), embed_dim, out_h, out_w]

            # Two cases based on whether input is full-image or per-patch:
            out_h = patch_embeds.shape[2]
            out_w = patch_embeds.shape[3]

            if sequence_len == 1:
                # Full-image input: [B, 1, C, H, W]
                # Conv2d output: [B, embed_dim, H/patch, W/patch]
                # -> flatten spatial dims and transpose to [B, num_patches, embed_dim]
                embeddings = patch_embeds.flatten(2).transpose(1, 2)
            else:
                # Per-patch input: [B, N, C, 14, 14]
                # Conv2d output per patch: [B*N, embed_dim, 1, 1]
                # -> squeeze spatial dims and rearrange to [B, N, embed_dim]
                embeddings = patch_embeds.flatten(-2).squeeze(-1)
                embeddings = rearrange(
                    embeddings, "(b l) d -> b l d", b=batch_size, l=sequence_len
                )

            # Add position embeddings
            if interpolate_pos_encoding and image_grid_thw is not None:
                # Interpolate 2D position encoding for each image's grid
                flatten_image_grid_thw = self.flatten_list(image_grid_thw)
                assert batch_size == 1
                start = 0
                tmp_embeddings = []
                embeddings_squeezed = embeddings.squeeze(0)

                for image_grid in image_grid_thw:
                    t, h, w = image_grid
                    end = start + t * h * w
                    image_embeddings = embeddings_squeezed[start:end, :]

                    # Interpolate position encoding for this image size
                    position_embedding = (
                        self.interpolate_pos_encoding(image_embeddings, h, w, True)
                        .squeeze(0)
                        .repeat(t, 1)
                    )

                    image_embeddings = image_embeddings + position_embedding
                    tmp_embeddings.append(image_embeddings)
                    start = end

                embeddings = torch.concat(tmp_embeddings, dim=0).unsqueeze(0)
            else:
                # Fallback: use packing position embedding
                if position_ids is None:
                    position_ids = torch.arange(
                        embeddings.shape[1], device=pixel_values.device
                    ).unsqueeze(0)
                embeddings = embeddings + self.packing_position_embedding(position_ids)

            return embeddings
        else:
            # Handle 4D input (batch, channel, height, width) - standard image format
            patch_embeds = self.patch_embedding(pixel_values)
            patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

            # Position embedding - use provided position_ids or default
            if position_ids is None:
                position_ids = self.position_ids

            embeddings = patch_embeds + self.position_embedding(position_ids)
            return embeddings

    @staticmethod
    def flatten_list(image_grid_thw):
        """Flatten nested list structure"""
        tmp_image_grid_thw = []
        for image_grid in image_grid_thw:
            if isinstance(image_grid, list):
                tmp_image_grid_thw.extend(image_grid)
            else:
                tmp_image_grid_thw.append(image_grid)
        return tmp_image_grid_thw

    def interpolate_pos_encoding(
        self,
        embeddings: torch.Tensor,
        height: int,
        width: int,
        is_after_patchify: bool = False,
    ) -> torch.Tensor:
        """Interpolate position embeddings to match image dimensions"""
        num_positions = self.position_embedding.weight.shape[0]
        patch_pos_embed = self.position_embedding.weight.unsqueeze(0)
        dim = embeddings.shape[-1]

        if is_after_patchify:
            new_height = height
            new_width = width
        else:
            new_height = height // self.patch_size
            new_width = width // self.patch_size

        sqrt_num_positions = int(num_positions**0.5)
        patch_pos_embed = patch_pos_embed.reshape(
            1, sqrt_num_positions, sqrt_num_positions, dim
        )
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = F.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
        return patch_pos_embed


@XHLLM_TRACEABLE_MODULES.register_module(
    {
        SiglipVisionTransformer: "SiglipVisionTransformer",
    }
)
class _SiglipVisionTransformer(DynamicModule):
    """Optimized SiglipVisionTransformer with simplified forward pass

    Note on return types for torch.fx compatibility:
    - When sample_indices is provided: Returns list of tensors (for Projector to process)
    - Otherwise: Returns BaseModelOutputWithPooling with single tensor

    This design maintains compatibility with downstream Projector.forward() which expects:
    - list/tuple when processing multi-sample outputs
    - single Tensor otherwise
    """

    @torch.no_grad()
    def _setup(self, cfg: ConfigDict):
        """Setup vision transformer"""
        self.embed_dim = self.config.hidden_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = False,
        attention_mask: Optional[torch.Tensor] = None,
        sample_indices: Optional[torch.Tensor] = None,
        image_indices: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        height_position_ids: Optional[torch.Tensor] = None,
        width_position_ids: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[List[torch.Tensor]] = None,
        padding_mask: Optional[torch.Tensor] = None,
        vision_return_embed_list: Optional[bool] = False,
        image_grid_thw: Optional[
            List[Union[Tuple[int, int, int], List[Tuple[int, int, int]]]]
        ] = None,
        return_pooler_output: Optional[bool] = True,
        use_rope: Optional[bool] = False,
        window_size: Optional[bool] = -1,
    ) -> BaseModelOutputWithPooling:
        """Forward pass with proper position embeddings and RoPE.

        For ONNX export with full-image input [B, 1, C, H, W]:
        - Auto-computes image_grid_thw from the input shape
        - Enables interpolated position embeddings
        - Enables RoPE in encoder attention
        This ensures the ONNX graph captures the correct positional computation.
        """
        # Auto-detect full-image input and enable positional features
        if pixel_values.dim() == 5 and pixel_values.shape[1] == 1:
            _, _, _, H, W = pixel_values.shape
            patch_size = self.config.patch_size
            h = H // patch_size
            w = W // patch_size
            image_grid_thw = [(1, h, w)]
            interpolate_pos_encoding = True
            use_rope = True
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        # Get embeddings
        hidden_states = self.embeddings(
            pixel_values,
            interpolate_pos_encoding=interpolate_pos_encoding,
            position_ids=position_ids,
            image_grid_thw=image_grid_thw,
        )

        # Pass through encoder with full parameters
        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cu_seqlens=cu_seqlens,
            image_grid_thw=image_grid_thw,
            height_position_ids=height_position_ids,
            width_position_ids=width_position_ids,
            use_rope=use_rope,
            window_size=window_size,
            vision_or_text="vision",
        )

        # Extract last hidden state
        last_hidden_state = encoder_outputs.last_hidden_state

        # Apply post-layer norm
        last_hidden_state = self.post_layernorm(last_hidden_state)

        # Handle different output paths based on sample_indices/cu_seqlens
        # Note: These conditions are Python-level (None checks), OK for torch.fx
        # The actual tensor-value based splits are delegated to calling code
        # Important: Always return BaseModelOutputWithPooling, but last_hidden_state
        # can be either Tensor or List[Tensor] - Projector.forward() handles both
        if sample_indices is not None:
            # Return list of hidden states split by sample_indices in last_hidden_state field
            last_hidden_state = self._split_by_sample_indices(
                last_hidden_state, sample_indices
            )
        elif cu_seqlens is not None and return_pooler_output is False:
            # Return list of hidden states split by cu_seqlens in last_hidden_state field
            last_hidden_state = self._split_by_cu_seqlens(last_hidden_state, cu_seqlens)

        # Compute pooler output only for tensor path (not for list)
        pooled_output = None
        if (
            self.use_head
            and return_pooler_output
            and isinstance(last_hidden_state, torch.Tensor)
        ):
            pooled_output = self.head(last_hidden_state)

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def _split_by_sample_indices(
        self, hidden_state: torch.Tensor, sample_indices: torch.Tensor
    ) -> List[torch.Tensor]:
        """Split hidden states by sample_indices.

        Note: This method contains tensor-value conditionals that cannot be traced by torch.fx.
        However, since it's called based on sample_indices being not None (Python-level check),
        it won't affect torch.fx tracing of the main forward path.
        """
        sample_hidden_state_list = []
        hidden_state = hidden_state.squeeze(0)
        sample_index = sample_indices

        # Get unique indices - this is dynamic based on sample_index values
        unique_sample_index = torch.unique(sample_index).sort().values.unbind(0)
        unique_sample_index = list(unique_sample_index)

        # Filter out -1 indices if present
        if len(unique_sample_index) > 0 and unique_sample_index[0] == -1:
            unique_sample_index = unique_sample_index[1:]

        # Split by each unique sample index
        for sample_idx in unique_sample_index:
            token_indices = (sample_index == sample_idx).nonzero().flatten()
            sample_hidden_state = hidden_state[token_indices]
            sample_hidden_state_list.append(sample_hidden_state)

        return sample_hidden_state_list

    def _split_by_cu_seqlens(
        self, hidden_state: torch.Tensor, cu_seqlens: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Split hidden states by cumulative sequence lengths.

        Note: This method contains loops over tensor dimensions that cannot be traced by torch.fx.
        However, since it's called based on cu_seqlens being not None (Python-level check),
        it won't affect torch.fx tracing of the main forward path.
        """
        sample_hidden_state = []
        for i in range(cu_seqlens.shape[0] - 1):
            start = cu_seqlens[i]
            end = cu_seqlens[i + 1]
            tensor = hidden_state[:, start:end, :].squeeze(0)
            sample_hidden_state.append(tensor)
        return sample_hidden_state


def register_wrap_cls(hf_model):
    pass
