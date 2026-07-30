"""Lightweight shared constants for the Gemma4 MTP draft ABI."""

from __future__ import annotations


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

# FlashAttention does not consume the legacy dense sliding mask.  It reads the
# same target-owned K/V tensors plus the physical origin and valid width of the
# compact sliding cache.  Their half-open absolute interval is
# [kv_window_start_abs, kv_window_start_abs + kv_valid_length).
MTP_FLASH_DRAFT_INPUT_NAMES = (
    *MTP_DRAFT_INPUT_NAMES[:3],
    *MTP_SHARED_KV_INPUT_NAMES,
    "kv_window_start_abs",
    "kv_valid_length",
)


def resolve_readonly_sliding_cache_range(
    target_visible_length: int,
    sliding_window: int,
) -> tuple[int, int]:
    """Return the compact target-KV ``(absolute_start, valid_width)``.

    Gemma4's assistant reads the target cache without appending draft K/V.
    Consequently every proposal in one speculative round sees the same
    committed target prefix ``[0, target_visible_length)``.  Sliding attention
    stores only its suffix, represented without moving cache data as
    ``[absolute_start, absolute_start + valid_width)``.
    """

    target_visible_length = int(target_visible_length)
    sliding_window = int(sliding_window)
    if target_visible_length <= 0:
        raise ValueError("target_visible_length must be positive")
    if sliding_window <= 0:
        raise ValueError("sliding_window must be positive")
    absolute_start = max(0, target_visible_length - sliding_window)
    return absolute_start, target_visible_length - absolute_start
