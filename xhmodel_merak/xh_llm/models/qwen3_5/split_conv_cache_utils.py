"""Shared split q/k/v conv-cache helpers for Qwen3.5 dense and MoE exports.

The Merak split conv-cache contract keeps one logical cache tuple per linear
attention layer internally, but preserves the flat HMONNX runtime/export
signature: ``q0, k0, v0, q1, k1, v1, ...``.
"""

from __future__ import annotations

from typing import Optional


def _is_nested_split_conv_cache(cache_list) -> bool:
    return isinstance(cache_list, (list, tuple)) and len(cache_list) > 0 and isinstance(cache_list[0], (list, tuple))


def _is_flat_split_conv_cache(cache_list) -> bool:
    return (
        isinstance(cache_list, (list, tuple)) and len(cache_list) > 0 and not isinstance(cache_list[0], (list, tuple))
    )


def _regroup_flat_split_conv_cache(cache_list):
    if (
        cache_list is None
        or not isinstance(cache_list, (list, tuple))
        or len(cache_list) == 0
        or _is_nested_split_conv_cache(cache_list)
    ):
        return cache_list
    if len(cache_list) % 3 != 0:
        raise RuntimeError(f"Expected split conv cache length to be divisible by 3, got {len(cache_list)}")
    return [tuple(cache_list[idx : idx + 3]) for idx in range(0, len(cache_list), 3)]


def _looks_like_flat_split_conv_cache(cache_list) -> bool:
    if (
        not isinstance(cache_list, (list, tuple))
        or _is_nested_split_conv_cache(cache_list)
        or len(cache_list) == 0
        or len(cache_list) % 3 != 0
    ):
        return False
    try:
        q_cache, k_cache, v_cache = cache_list[0], cache_list[1], cache_list[2]
        q_width = q_cache.shape[1]
        k_width = k_cache.shape[1]
        v_width = v_cache.shape[1]
        return q_width == k_width and v_width == q_width * 2
    except Exception:
        return False


def _layers_use_split_conv_cache(layers) -> bool:
    try:
        for decoder_layer in layers:
            linear_attn = getattr(decoder_layer, "linear_attn", None)
            if linear_attn is not None and hasattr(linear_attn, "in_proj_q"):
                return True
    except Exception:
        return False
    return False


def _flatten_split_conv_cache_outputs(cache_list):
    if cache_list is None or not isinstance(cache_list, (list, tuple)) or not _is_nested_split_conv_cache(cache_list):
        return cache_list
    flat_cache = []
    for cache_tuple in cache_list:
        if len(cache_tuple) % 3 != 0:
            raise RuntimeError(f"Expected q/k/v conv cache tuple length to be divisible by 3, got {len(cache_tuple)}")
        flat_cache.extend(cache_tuple)
    return flat_cache


def _flatten_merged_conv_cache_outputs(cache_list):
    if cache_list is None or not isinstance(cache_list, (list, tuple)):
        return cache_list

    flat_cache = []
    changed = False
    for cache in cache_list:
        if isinstance(cache, (list, tuple)):
            flat_cache.extend(cache)
            changed = True
        else:
            flat_cache.append(cache)
    return flat_cache if changed else cache_list


def _get_linear_layer_conv_cache(cache_list, layer_idx: int, split_conv_cache: bool):
    if cache_list is None:
        return None
    if not split_conv_cache or _is_nested_split_conv_cache(cache_list):
        return cache_list[layer_idx]
    base_idx = layer_idx * 3
    return (cache_list[base_idx], cache_list[base_idx + 1], cache_list[base_idx + 2])


def _select_linear_attn_conv_cache(cache_list, cache_idx: int, split_conv_cache: bool):
    if cache_list is None:
        return None
    if not split_conv_cache:
        return cache_list[cache_idx]
    if _is_nested_split_conv_cache(cache_list):
        return cache_list[cache_idx]
    base_idx = cache_idx * 3
    return (
        cache_list[base_idx],
        cache_list[base_idx + 1],
        cache_list[base_idx + 2],
    )


def _is_grouped_split_conv_cache(cache_list) -> bool:
    return _is_nested_split_conv_cache(cache_list)
