# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""Export Cosmos3-Nano transformer attention core probe."""

from __future__ import annotations

import torch
import torch.nn as nn


__all__ = ["TransformerAttentionCoreWrapper", "make_causal_mask"]


def make_causal_mask(
    batch: int,
    seq: int,
    mask_value: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    mask = torch.full((seq, seq), fill_value=mask_value, dtype=dtype, device=device)
    causal = torch.triu(mask, diagonal=1)
    return causal.view(1, 1, seq, seq).expand(batch, 1, seq, seq)


class TransformerAttentionCoreWrapper(nn.Module):
    def __init__(self, scaling: float):
        super().__init__()
        self.scaling = float(scaling)

    @staticmethod
    def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
        if n_rep == 1:
            return hidden_states
        batch, num_key_value_heads, seq, head_dim = hidden_states.shape
        expanded = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, seq, head_dim)
        return expanded.reshape(batch, num_key_value_heads * n_rep, seq, head_dim)

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        num_key_value_groups = query_states.shape[1] // key_states.shape[1]
        key_states = self.repeat_kv(key_states, num_key_value_groups)
        value_states = self.repeat_kv(value_states, num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output.reshape(query_states.shape[0], query_states.shape[2], -1)
