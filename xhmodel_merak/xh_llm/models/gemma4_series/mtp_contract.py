"""Lightweight shared constants for the Gemma4 MTP draft ABI."""

from __future__ import annotations

from typing import Any


MTP_DRAFT_INPUT_NAMES = (
    "inputs_embeds",
    "past_seq_length",
    "current_input_length",
    "sliding_attention_mask",
    "shared_key_cache_sliding",
    "shared_value_cache_sliding",
    "shared_key_cache_full",
    "shared_value_cache_full",
)
MTP_SHARED_KV_INPUT_NAMES = MTP_DRAFT_INPUT_NAMES[-4:]

# ``exact_range`` is the general read-only cache ABI: it lowers the assistant
# attention to non-causal Padding Full Cross-Attention and supplies each
# query's absolute [start, end) KV range. It supports arbitrary query widths
# and expresses a sliding window exactly.
#
# ``causal`` is the optimized Gemma4 MTP ABI. The Q-only assistant always has
# M=1 and never writes draft KV. Feeding attention valid length N-1 while the
# target-owned cache contains N entries gives the same visibility through a
# causal PageAttention mask, without materializing a per-query range tensor.
READONLY_ATTENTION_LOWERINGS = frozenset({"exact_range", "causal"})


def normalize_readonly_attention_lowering(value: Any = None) -> str:
    """Resolve the optional read-only attention lowering to a valid ABI.

    Choose ``exact_range`` for the general non-causal cross-attention form and
    ``causal`` only for Gemma4's read-only, single-query MTP specialization.
    """

    lowering = str(value or "exact_range").strip().lower()
    if lowering not in READONLY_ATTENTION_LOWERINGS:
        raise ValueError("Gemma4 MTP readonly_attention_lowering must be 'exact_range' or 'causal'")
    return lowering
