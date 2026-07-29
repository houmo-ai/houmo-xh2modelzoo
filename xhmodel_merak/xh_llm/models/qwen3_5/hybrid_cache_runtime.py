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
from dataclasses import dataclass
from typing import Any

import torch

from .split_conv_cache_utils import _regroup_flat_split_conv_cache


_CONV_OUTPUT_RE = re.compile(r"^conv_cache_out_(?:(q|k|v)_)?(\d+)(?:_(\d+))?$")
_RECURRENT_OUTPUT_RE = re.compile(r"^recurrent_state_out_(\d+)(?:_(\d+))?$")


@dataclass
class HybridCacheTransaction:
    """Uncommitted per-step linear-state outputs from one target verify."""

    outputs: tuple[Any, ...]
    output_names: tuple[str, ...] | None
    verify_steps: int


def normalize_hybrid_hmonnx_args(args: Sequence[Any]) -> list[Any]:
    """Flatten the hybrid-cache ABI and cast ONNX integer inputs to int32.

    Transformers may wrap ``past_key_values`` in more than one tuple layer.
    HMONNX graphs expose those caches as flat tensor inputs, so normalization
    must recurse instead of assuming the legacy one-level tuple layout.
    """

    normalized: list[Any] = []

    def append_flat(value: Any) -> None:
        if isinstance(value, (list, tuple)):
            for nested in value:
                append_flat(nested)
            return
        if isinstance(value, torch.Tensor) and value.dtype == torch.int64:
            value = value.to(torch.int32)
        normalized.append(value)

    append_flat(args)
    return normalized


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
    session = getattr(active_model, "hmonnx_session", None)

    # Runtime/session APIs are the authoritative output ABI.  They must be
    # consulted before graph introspection because HMONNX sessions expose
    # unknown attributes as dynamic ONNX operators; probing ``graph`` or
    # ``output`` on that namespace emits false registration errors on every
    # decode step.
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

    candidates: list[Any] = [getattr(active_model, "_onnx_graph", None)]
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
        if graph_outputs is None or callable(graph_outputs):
            continue
        names = [str(getattr(item, "name", item)) for item in graph_outputs]
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
    selected_step: int | None = None,
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

    final_step = verify_steps - 1 if selected_step is None else selected_step
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
    selected_step: int | None = None,
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

    final_step = verify_steps - 1 if selected_step is None else selected_step
    selected_conv: list[Any] = []
    for layer in range(linear_layers):
        for branch in range(conv_branches):
            offset = (layer * conv_branches + branch) * verify_steps
            selected_conv.append(raw_conv[offset + final_step])
    selected_recurrent = (
        [raw_recurrent[layer * verify_steps + final_step] for layer in range(linear_layers)] if raw_recurrent else []
    )
    return selected_conv, selected_recurrent


def _select_runtime_cache_outputs(
    runtime: Any,
    outputs: Sequence[Any],
    *,
    output_names: Sequence[str] | None,
    verify_steps: int,
    selected_step: int | None,
    model_label: str,
) -> tuple[list[Any], list[Any]]:
    linear_outputs = list(outputs[1:])
    past_conv_caches = runtime._kvcache_mixin.past_conv_caches
    split_conv_cache = bool(runtime._kvcache_mixin.split_conv_cache)
    allow_missing_recurrent = runtime.is_prefill() and model_config_prefill_recurrent_state_uses_cache(
        runtime.meta_info.model_config
    )
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
            selected_step=selected_step,
        )
    if selected is None:
        selected = _select_positional_cache_outputs(
            linear_outputs,
            linear_layers=len(past_conv_caches),
            split_conv_cache=split_conv_cache,
            verify_steps=verify_steps,
            allow_missing_recurrent=allow_missing_recurrent,
            model_label=model_label,
            selected_step=selected_step,
        )
    return selected


def _commit_selected_cache_outputs(
    runtime: Any,
    conv_outputs: Sequence[Any],
    recurrent_outputs: Sequence[Any],
) -> None:
    past_conv_caches = runtime._kvcache_mixin.past_conv_caches
    past_recurrent_states = runtime._kvcache_mixin.past_recurrent_states
    if runtime._kvcache_mixin.split_conv_cache:
        grouped = _regroup_flat_split_conv_cache(conv_outputs)
        for (past_q, past_k, past_v), (out_q, out_k, out_v) in zip(
            past_conv_caches,
            grouped,
            strict=True,
        ):
            past_q[:] = out_q[:]
            past_k[:] = out_k[:]
            past_v[:] = out_v[:]
    else:
        for past, out in zip(past_conv_caches, conv_outputs, strict=True):
            past[:] = out[:]
    if recurrent_outputs:
        for past, out in zip(
            past_recurrent_states,
            recurrent_outputs,
            strict=True,
        ):
            past[:] = out[:]


def begin_hybrid_cache_transaction(runtime: Any) -> None:
    """Defer Qwen verify-state mutation until acceptance is known."""

    if getattr(runtime, "_hybrid_cache_transaction", None) is not None:
        raise RuntimeError("Qwen hybrid-cache transaction is already pending")
    runtime._defer_hybrid_cache_commit = True


def abort_hybrid_cache_transaction(runtime: Any) -> None:
    runtime._defer_hybrid_cache_commit = False
    runtime._hybrid_cache_transaction = None


def begin_hybrid_cache_output_passthrough(
    runtime: Any,
    *,
    selected_step: int = 0,
) -> None:
    """Commit one known step while preserving the graph's raw outputs.

    Merak speculative decoding needs the target hidden-state output even on
    ordinary prefill/decode calls.  The legacy wrapper returned only logits
    and selected cache tensors, which discarded that hidden state.  Decode
    graphs are exported at the full verify width, so a one-token non-verify
    call must also commit step zero rather than the padded final step.
    """

    if getattr(runtime, "_hybrid_cache_transaction", None) is not None or bool(
        getattr(runtime, "_defer_hybrid_cache_commit", False)
    ):
        raise RuntimeError(
            "Qwen hybrid-cache output passthrough conflicts with a transaction"
        )
    if bool(getattr(runtime, "_hybrid_cache_output_passthrough", False)):
        raise RuntimeError(
            "Qwen hybrid-cache output passthrough is already active"
        )
    selected_step = int(selected_step)
    if selected_step < 0:
        raise ValueError("selected_step must be non-negative")
    runtime._hybrid_cache_output_passthrough = True
    runtime._hybrid_cache_passthrough_step = selected_step


def end_hybrid_cache_output_passthrough(runtime: Any) -> None:
    runtime._hybrid_cache_output_passthrough = False
    runtime._hybrid_cache_passthrough_step = None


def detach_hybrid_cache_transaction(runtime: Any) -> HybridCacheTransaction:
    """Move an uncommitted verify result out of the serial runtime."""

    transaction = getattr(runtime, "_hybrid_cache_transaction", None)
    if not isinstance(transaction, HybridCacheTransaction):
        raise RuntimeError("Qwen hybrid-cache transaction is not pending")
    runtime._hybrid_cache_transaction = None
    runtime._defer_hybrid_cache_commit = False
    return transaction


def commit_hybrid_cache_transaction(
    runtime: Any,
    accepted_steps: int,
    *,
    model_label: str,
    transaction: HybridCacheTransaction | None = None,
) -> tuple[list[Any], list[Any]]:
    """Commit snapshot ``accepted_steps - 1`` without cloning all states.

    ``accepted_steps`` includes the verifier's mandatory recovery token.
    Therefore zero accepted *draft* tokens is ``accepted_steps == 1`` and
    commits snapshot zero.  A value of zero means that the verify operation
    itself failed; callers must abort that transaction instead of committing
    a nonexistent state.
    """

    attached_transaction = transaction is None
    if transaction is None:
        transaction = getattr(runtime, "_hybrid_cache_transaction", None)
    if not isinstance(transaction, HybridCacheTransaction):
        raise RuntimeError("Qwen hybrid-cache transaction is not pending")
    accepted_steps = int(accepted_steps)
    if accepted_steps < 1 or accepted_steps > transaction.verify_steps:
        raise ValueError(
            "accepted_steps must be in [1, verify_steps], got "
            f"{accepted_steps} for {transaction.verify_steps}"
        )
    selected = _select_runtime_cache_outputs(
        runtime,
        transaction.outputs,
        output_names=transaction.output_names,
        verify_steps=transaction.verify_steps,
        selected_step=accepted_steps - 1,
        model_label=model_label,
    )
    _commit_selected_cache_outputs(runtime, *selected)
    if attached_transaction:
        runtime._hybrid_cache_transaction = None
    runtime._defer_hybrid_cache_commit = False
    return selected


def commit_hybrid_cache_outputs(
    runtime: Any,
    outputs: Sequence[Any],
    *,
    model_label: str,
) -> tuple[Any, ...]:
    """Parse active graph outputs and commit only verify step ``T - 1``."""

    logits = outputs[0]
    verify_steps = 1 if runtime.is_prefill() else get_spec_decode_verify_steps(runtime.meta_info)
    output_names = _graph_output_names(runtime, len(outputs))
    passthrough = bool(
        getattr(runtime, "_hybrid_cache_output_passthrough", False)
    )
    passthrough_step = getattr(
        runtime,
        "_hybrid_cache_passthrough_step",
        None,
    )
    if (
        not runtime.is_prefill()
        and bool(getattr(runtime, "_defer_hybrid_cache_commit", False))
    ):
        if passthrough:
            raise RuntimeError(
                "Qwen hybrid-cache passthrough and transaction overlap"
            )
        if getattr(runtime, "_hybrid_cache_transaction", None) is not None:
            raise RuntimeError("Qwen hybrid-cache transaction is already pending")
        transaction = HybridCacheTransaction(
            outputs=tuple(outputs),
            output_names=(
                tuple(output_names) if output_names is not None else None
            ),
            verify_steps=verify_steps,
        )
        runtime._hybrid_cache_transaction = transaction
        runtime._defer_hybrid_cache_commit = False
        return tuple(outputs)

    selected = _select_runtime_cache_outputs(
        runtime,
        outputs,
        output_names=output_names,
        verify_steps=verify_steps,
        selected_step=(
            int(passthrough_step)
            if passthrough_step is not None and passthrough
            else None
        ),
        model_label=model_label,
    )
    conv_outputs, recurrent_outputs = selected
    _commit_selected_cache_outputs(runtime, conv_outputs, recurrent_outputs)
    if passthrough:
        return tuple(outputs)
    return logits, conv_outputs, recurrent_outputs


__all__ = [
    "commit_hybrid_cache_outputs",
    "begin_hybrid_cache_transaction",
    "abort_hybrid_cache_transaction",
    "begin_hybrid_cache_output_passthrough",
    "end_hybrid_cache_output_passthrough",
    "detach_hybrid_cache_transaction",
    "commit_hybrid_cache_transaction",
    "HybridCacheTransaction",
    "get_spec_decode_verify_steps",
    "model_config_prefill_recurrent_state_uses_cache",
]
