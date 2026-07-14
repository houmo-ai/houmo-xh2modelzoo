"""Shared HMONNX linear-cache parsing for Qwen hybrid-attention models.

The Qwen3.5 dense/MoE and Qwen3-Next target graphs intentionally share the
same cache ABI.  In particular, speculative target graphs emit one snapshot
per verify token, ordered as ``C * L * T`` for convolution caches and
``L * T`` for recurrent states.  This module keeps that ABI in one place so a
new model family cannot silently drift from the exporter.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .split_conv_cache_utils import _regroup_flat_split_conv_cache


_CONV_OUTPUT_RE = re.compile(r"^conv_cache_out_(?:(q|k|v)_)?(\d+)(?:_(\d+))?$")
_RECURRENT_OUTPUT_RE = re.compile(r"^recurrent_state_out_(\d+)(?:_(\d+))?$")


def model_config_prefill_recurrent_state_uses_cache(model_config: Any) -> bool:
    """Return the exported prefill recurrent-state mutation contract."""

    explicit = (
        model_config.get("prefill_recurrent_state_uses_cache")
        if isinstance(model_config, Mapping)
        else getattr(model_config, "prefill_recurrent_state_uses_cache", None)
    )
    # The exporter derives this field from the concrete GDRChunkScan
    # ``state_is_cache`` capability.  ``fuse_gdr_ops`` alone is insufficient:
    # legacy/custom fused scans may still return a recurrent-state output.
    # Old metadata without the explicit capability must therefore use the
    # conservative output-returning ABI.
    return bool(explicit) if explicit is not None else False


def get_spec_decode_verify_steps(meta_info: Any) -> int:
    """Resolve target verify length from the shared ``spec_decode`` metadata."""

    spec_decode = getattr(meta_info, "spec_decode", None)
    if spec_decode is None:
        return 1
    if isinstance(spec_decode, dict):
        mode = spec_decode.get("mode")
        num_draft_tokens = spec_decode.get("num_draft_tokens")
    else:
        mode = getattr(spec_decode, "mode", None)
        num_draft_tokens = getattr(spec_decode, "num_draft_tokens", None)
    if mode in {"mtp", "dflash"} and num_draft_tokens is not None:
        return int(num_draft_tokens) + 1
    return 1


def _graph_output_names(runtime: Any, output_count: int) -> list[str] | None:
    """Read output names from the active graph/session when the runtime exposes them."""

    active_model = getattr(
        runtime,
        "prefill_model" if runtime.is_prefill() else "decode_model",
        None,
    )
    if active_model is None:
        return None
    candidates: list[Any] = [getattr(active_model, "_onnx_graph", None)]
    session = getattr(active_model, "hmonnx_session", None)
    candidates.extend(
        [
            getattr(session, "onnx_graph", None),
            getattr(session, "graph", None),
            getattr(session, "graph_module", None),
        ]
    )
    for candidate in candidates:
        graph = getattr(candidate, "graph", candidate)
        graph_outputs = getattr(graph, "output", None)
        if graph_outputs is None:
            continue
        names = [str(getattr(item, "name", item)) for item in graph_outputs]
        if len(names) == output_count:
            return names

    for owner in (session, active_model):
        get_names = getattr(owner, "get_output_names", None)
        if callable(get_names):
            names = [str(name) for name in get_names()]
            if len(names) == output_count:
                return names
        names = getattr(owner, "output_names", None)
        if isinstance(names, Sequence) and not isinstance(names, (str, bytes)):
            names = [str(name) for name in names]
            if len(names) == output_count:
                return names
    return None


def _select_named_cache_outputs(
    output_names: Sequence[str],
    outputs: Sequence[Any],
    *,
    linear_layers: int,
    split_conv_cache: bool,
    verify_steps: int,
    allow_missing_recurrent: bool,
    model_label: str,
) -> tuple[list[Any], list[Any]] | None:
    conv: dict[tuple[int, str | None, int], Any] = {}
    recurrent: dict[tuple[int, int], Any] = {}
    found_cache_name = False
    for name, value in zip(output_names, outputs, strict=True):
        conv_match = _CONV_OUTPUT_RE.fullmatch(name)
        if conv_match:
            found_cache_name = True
            branch, layer, step = conv_match.groups()
            conv[(int(layer), branch, int(step) if step is not None else 0)] = value
            continue
        recurrent_match = _RECURRENT_OUTPUT_RE.fullmatch(name)
        if recurrent_match:
            found_cache_name = True
            layer, step = recurrent_match.groups()
            recurrent[(int(layer), int(step) if step is not None else 0)] = value

    if not found_cache_name:
        return None

    final_step = verify_steps - 1
    branches: tuple[str | None, ...] = ("q", "k", "v") if split_conv_cache else (None,)
    selected_conv: list[Any] = []
    missing_conv: list[str] = []
    for layer in range(linear_layers):
        for branch in branches:
            key = (layer, branch, final_step)
            if key not in conv:
                suffix = f"{branch}_{layer}_{final_step}" if branch else f"{layer}_{final_step}"
                missing_conv.append(f"conv_cache_out_{suffix}")
            else:
                selected_conv.append(conv[key])
    if missing_conv:
        raise RuntimeError(
            f"{model_label} active HMONNX metadata is missing final-step convolution cache outputs: {missing_conv}"
        )

    selected_recurrent: list[Any] = []
    missing_recurrent: list[str] = []
    for layer in range(linear_layers):
        key = (layer, final_step)
        if key not in recurrent:
            missing_recurrent.append(f"recurrent_state_out_{layer}_{final_step}")
        else:
            selected_recurrent.append(recurrent[key])
    if missing_recurrent and recurrent:
        raise RuntimeError(
            f"{model_label} active HMONNX metadata contains only a partial recurrent "
            f"state output section; missing: {missing_recurrent}"
        )
    if missing_recurrent and not allow_missing_recurrent:
        raise RuntimeError(
            f"{model_label} active HMONNX metadata is missing final-step recurrent state outputs: {missing_recurrent}"
        )
    if missing_recurrent:
        selected_recurrent = []
    return selected_conv, selected_recurrent


def _select_positional_cache_outputs(
    linear_outputs: Sequence[Any],
    *,
    linear_layers: int,
    split_conv_cache: bool,
    verify_steps: int,
    allow_missing_recurrent: bool,
    model_label: str,
) -> tuple[list[Any], list[Any]]:
    conv_branches = 3 if split_conv_cache else 1
    conv_count = linear_layers * conv_branches * verify_steps
    recurrent_count = linear_layers * verify_steps
    required = conv_count + (0 if allow_missing_recurrent else recurrent_count)
    if len(linear_outputs) < required:
        raise RuntimeError(
            f"{model_label} HMONNX returned {len(linear_outputs)} auxiliary outputs; "
            f"expected at least {required} ({conv_count} conv + "
            f"{0 if allow_missing_recurrent else recurrent_count} recurrent across "
            f"{verify_steps} step(s))"
        )

    raw_conv = list(linear_outputs[:conv_count])
    # A fused prefill graph may mutate recurrent CacheTensor inputs and export
    # no recurrent outputs.  When names are unavailable, only consume a full
    # recurrent section; trailing hidden outputs must not be mistaken for it.
    raw_recurrent = [] if allow_missing_recurrent else list(linear_outputs[conv_count : conv_count + recurrent_count])

    final_step = verify_steps - 1
    selected_conv: list[Any] = []
    for layer in range(linear_layers):
        for branch in range(conv_branches):
            offset = (layer * conv_branches + branch) * verify_steps
            selected_conv.append(raw_conv[offset + final_step])
    selected_recurrent = (
        [raw_recurrent[layer * verify_steps + final_step] for layer in range(linear_layers)] if raw_recurrent else []
    )
    return selected_conv, selected_recurrent


def commit_hybrid_cache_outputs(
    runtime: Any,
    outputs: Sequence[Any],
    *,
    model_label: str,
) -> tuple[Any, list[Any], list[Any]]:
    """Parse active graph outputs and commit only verify step ``T - 1``."""

    logits, *linear_outputs = outputs
    past_conv_caches = runtime._kvcache_mixin.past_conv_caches
    past_recurrent_states = runtime._kvcache_mixin.past_recurrent_states
    split_conv_cache = bool(runtime._kvcache_mixin.split_conv_cache)
    verify_steps = 1 if runtime.is_prefill() else get_spec_decode_verify_steps(runtime.meta_info)
    allow_missing_recurrent = runtime.is_prefill() and model_config_prefill_recurrent_state_uses_cache(
        runtime.meta_info.model_config
    )
    output_names = _graph_output_names(runtime, len(outputs))
    selected = None
    if output_names is not None:
        selected = _select_named_cache_outputs(
            output_names,
            outputs,
            linear_layers=len(past_conv_caches),
            split_conv_cache=split_conv_cache,
            verify_steps=verify_steps,
            allow_missing_recurrent=allow_missing_recurrent,
            model_label=model_label,
        )
    if selected is None:
        selected = _select_positional_cache_outputs(
            linear_outputs,
            linear_layers=len(past_conv_caches),
            split_conv_cache=split_conv_cache,
            verify_steps=verify_steps,
            allow_missing_recurrent=allow_missing_recurrent,
            model_label=model_label,
        )
    conv_outputs, recurrent_outputs = selected

    if split_conv_cache:
        grouped = _regroup_flat_split_conv_cache(conv_outputs)
        for (past_q, past_k, past_v), (out_q, out_k, out_v) in zip(past_conv_caches, grouped, strict=True):
            past_q[:] = out_q[:]
            past_k[:] = out_k[:]
            past_v[:] = out_v[:]
    else:
        for past, out in zip(past_conv_caches, conv_outputs, strict=True):
            past[:] = out[:]
    if recurrent_outputs:
        for past, out in zip(past_recurrent_states, recurrent_outputs, strict=True):
            past[:] = out[:]
    return logits, conv_outputs, recurrent_outputs


__all__ = [
    "commit_hybrid_cache_outputs",
    "get_spec_decode_verify_steps",
    "model_config_prefill_recurrent_state_uses_cache",
]
