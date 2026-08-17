"""Static row-count rules shared by DeepSeek-V4 SWA graph components."""

from __future__ import annotations


def _validate_layout(
    input_sequence_length: int,
    window_size: int,
    alignment: int,
) -> tuple[int, int, int]:
    values = (
        int(input_sequence_length),
        int(window_size),
        int(alignment),
    )
    if min(values) <= 0:
        raise ValueError("SWA dimensions must be positive")
    return values


def _align_up(value: int, alignment: int) -> int:
    return ((int(value) + int(alignment) - 1) // int(alignment)) * int(alignment)


def aligned_swa_attention_length(
    input_sequence_length: int,
    window_size: int,
    *,
    alignment: int = 16,
) -> int:
    """Rows emitted by sliding ``xh.LLMCache`` for one static invocation."""

    input_sequence_length, window_size, alignment = _validate_layout(
        input_sequence_length,
        window_size,
        alignment,
    )
    return _align_up(window_size + input_sequence_length - 1, alignment)


def aligned_swa_backing_length(
    input_sequence_length: int,
    window_size: int,
    *,
    alignment: int = 16,
) -> int:
    """Rows reserved by the persistent ``window + static input`` backing."""

    input_sequence_length, window_size, alignment = _validate_layout(
        input_sequence_length,
        window_size,
        alignment,
    )
    return _align_up(window_size + input_sequence_length, alignment)


__all__ = ["aligned_swa_attention_length", "aligned_swa_backing_length"]
