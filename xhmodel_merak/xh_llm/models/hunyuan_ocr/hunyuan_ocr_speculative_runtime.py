# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""DFlash speculative decoding runtime for HunyuanOCR-1.5."""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .dflash_draft import (
    HunyuanOCRDraftCacheController,
    build_dflash_noise_embedding,
    dflash_graph_io_contract,
)


_DEPLOYMENT_DTYPE = torch.float16
_ATTENTION_MASK_KEEP = 0.0
_ATTENTION_MASK_BLOCK = -10000.0
_V2_ENABLED_VALUES = frozenset({"1", "true", "yes", "on"})

FALLBACK_DISABLED = "disabled"
FALLBACK_ARTIFACTS_MISSING = "artifacts_missing"
FALLBACK_CAPACITY_SHORTFALL = "capacity_shortfall"
FALLBACK_DRAFT_EXECUTION_FAILED = "draft_execution_failed"

STOP_EOS = "eos"
STOP_MAX_LENGTH = "max_length"


class HunyuanOCRSpeculativeRuntimeError(RuntimeError):
    """Raised when the speculative runtime cannot honor its contract."""


@dataclass
class HunyuanOCRDraftGraphs:
    """Bind the three exported draft graphs and their shared cache tensors."""

    context: Any
    context_decode: Any
    decode: Any
    cache_input_names: tuple[str, ...]
    cache_output_names: tuple[str, ...]
    cache_tensors: list[Any]
    hidden_width: int
    draft_hidden_size: int
    capacity: int

    def reset_caches(self) -> None:
        for cache in self.cache_tensors:
            cache.data.zero_()


@dataclass
class SpeculativeStats:
    """Accumulate request-local speculative runtime statistics."""

    num_draft_tokens: int
    block_size: int
    blocks: int = 0
    proposed_draft_tokens: int = 0
    accepted_draft_tokens: int = 0
    target_verify_calls: int = 0
    target_decode_calls: int = 0
    draft_context_calls: int = 0
    draft_decode_calls: int = 0
    draft_context_decode_calls: int = 0
    truncated_tokens: int = 0
    fallback_reason: str | None = None
    fallback_block_index: int | None = None
    stop_reason: str | None = None
    accept_length_histogram: list[int] = field(default_factory=list)
    block_token_counts: list[int] = field(default_factory=list)
    timing_seconds: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.accept_length_histogram:
            self.accept_length_histogram = [0] * (self.num_draft_tokens + 1)

    def record_block(self, accepted_draft_count: int, proposed: int) -> None:
        self.blocks += 1
        self.proposed_draft_tokens += proposed
        self.accepted_draft_tokens += accepted_draft_count
        self.accept_length_histogram[accepted_draft_count] += 1
        self.block_token_counts.append(accepted_draft_count + 1)

    def add_time(self, name: str, seconds: float) -> None:
        self.timing_seconds[name] = self.timing_seconds.get(name, 0.0) + seconds

    def as_summary(self, *, enabled: bool) -> dict[str, Any]:
        denominator = self.blocks * self.num_draft_tokens
        return {
            "enabled": enabled,
            "num_draft_tokens": self.num_draft_tokens,
            "block_size": self.block_size,
            "blocks": self.blocks,
            "proposed_draft_tokens": self.proposed_draft_tokens,
            "accepted_draft_tokens": self.accepted_draft_tokens,
            "acceptance_rate": (self.accepted_draft_tokens / denominator) if denominator else 0.0,
            "mean_accepted_draft_tokens": (
                self.accepted_draft_tokens / self.blocks if self.blocks else 0.0
            ),
            "mean_tokens_per_block": (
                (self.accepted_draft_tokens + self.blocks) / self.blocks if self.blocks else 0.0
            ),
            "accept_length_histogram": list(self.accept_length_histogram),
            "block_token_counts": list(self.block_token_counts),
            "target_verify_calls": self.target_verify_calls,
            "target_decode_calls": self.target_decode_calls,
            "draft_context_calls": self.draft_context_calls,
            "draft_decode_calls": self.draft_decode_calls,
            "draft_context_decode_calls": self.draft_context_decode_calls,
            "truncated_tokens": self.truncated_tokens,
            "fallback_reason": self.fallback_reason,
            "fallback_block_index": self.fallback_block_index,
            "stop_reason": self.stop_reason,
        }


def resolve_draft_runtime_contract(spec_decode: Mapping[str, Any]) -> dict[str, Any]:
    """Read DFlash runtime constants from exported metadata."""

    draft = spec_decode.get("draft")
    if not isinstance(draft, Mapping):
        raise HunyuanOCRSpeculativeRuntimeError(
            "HunyuanOCR speculative runtime requires spec_decode.draft metadata"
        )
    cache = draft.get("cache")
    if not isinstance(cache, Mapping):
        raise HunyuanOCRSpeculativeRuntimeError(
            "HunyuanOCR speculative runtime requires spec_decode.draft.cache metadata"
        )
    output_head = draft.get("output_head")
    if not isinstance(output_head, Mapping):
        raise HunyuanOCRSpeculativeRuntimeError(
            "HunyuanOCR speculative runtime requires spec_decode.draft.output_head metadata"
        )
    generation_eos = draft.get("generation_eos_token_id")
    if type(generation_eos) is int:
        generation_eos_token_ids = (generation_eos,)
    elif isinstance(generation_eos, Sequence) and generation_eos:
        generation_eos_token_ids = tuple(int(token_id) for token_id in generation_eos)
    else:
        raise HunyuanOCRSpeculativeRuntimeError(
            "spec_decode.draft.generation_eos_token_id must be an int or a non-empty list"
        )
    block_size = int(draft["block_size"])
    num_draft_tokens = int(draft["num_draft_tokens"])
    if num_draft_tokens != block_size - 1:
        raise HunyuanOCRSpeculativeRuntimeError(
            "spec_decode.draft.num_draft_tokens must equal block_size - 1: "
            f"num_draft_tokens={num_draft_tokens}, block_size={block_size}"
        )
    source_shape = [int(value) for value in output_head["source_shape"]]
    if len(source_shape) != 2:
        raise HunyuanOCRSpeculativeRuntimeError(
            f"spec_decode.draft.output_head.source_shape must be rank 2, got {source_shape}"
        )
    return {
        "block_size": block_size,
        "num_draft_tokens": num_draft_tokens,
        "mask_token_id": int(draft["mask_token_id"]),
        "generation_eos_token_ids": generation_eos_token_ids,
        "capacity": int(cache["capacity"]),
        "cache_shape": [int(value) for value in cache["shape"]],
        "cache_input_names": tuple(str(name) for name in cache["input_names"]),
        "cache_output_names": tuple(str(name) for name in cache["output_names"]),
        "num_hidden_layers": len(cache["input_names"]) // 2,
        "draft_hidden_size": source_shape[1],
        "vocab_size": source_shape[0],
        "hidden_width": int(spec_decode["target_hidden_concat_size"]),
    }


def _default_draft_session_factory(
    device: torch.device | str,
    *,
    enable_cuda_graph: bool = False,
) -> Callable[[str], Any]:
    from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel

    enabled = os.environ.get(HMONNXModel.ENV_ENABLE_INFERENCE_V2, "").strip().lower()
    if enabled in _V2_ENABLED_VALUES:
        from xhquant.xhonnxruntime.hmonnx_inference_v2 import (
            HMONNXInferenceConfig,
            HMONNXInferenceV2,
        )

        def open_session(path: str) -> Any:
            session_config = HMONNXInferenceConfig()
            session_config.exec_devices = [str(device)]
            session_config.enable_cuda_graph = bool(enable_cuda_graph)
            session = HMONNXInferenceV2(path, session_config)
            session.initialize()
            return session

        return open_session

    from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

    def open_legacy_session(path: str) -> Any:
        return HMONNXInference(path).to(str(device))

    return open_legacy_session


def load_draft_graphs(
    meta_info: Any,
    contract: Mapping[str, Any],
    *,
    device: torch.device | str,
    enable_cuda_graph: bool = False,
    session_factory: Callable[[str], Any] | None = None,
    cache_factory: Callable[[torch.Tensor], Any] | None = None,
) -> HunyuanOCRDraftGraphs:
    """Open the three draft sessions and allocate their shared caches."""

    open_session = session_factory or _default_draft_session_factory(
        device,
        enable_cuda_graph=enable_cuda_graph,
    )
    if cache_factory is None:
        from xhquant.core import CacheTensor

        cache_factory = CacheTensor

    sessions = {}
    for mode in ("context", "context_decode", "decode"):
        path = getattr(meta_info, f"dflash_{mode}_hmonnx", None)
        if path is None:
            raise HunyuanOCRSpeculativeRuntimeError(
                f"HunyuanOCR speculative runtime requires dflash_{mode}_hmonnx metadata"
            )
        sessions[mode] = open_session(str(path))

    num_hidden_layers = int(contract["num_hidden_layers"])
    for mode, session in sessions.items():
        expected = dflash_graph_io_contract(mode=mode, num_hidden_layers=num_hidden_layers)
        actual_inputs = list(session.get_input_names())
        actual_outputs = list(session.get_output_names())
        if actual_inputs != expected["input_names"] or actual_outputs != expected["output_names"]:
            raise HunyuanOCRSpeculativeRuntimeError(
                f"HunyuanOCR draft {mode} graph IO does not match the exported contract: "
                f"expected_inputs={expected['input_names']}, got_inputs={actual_inputs}, "
                f"expected_outputs={expected['output_names']}, got_outputs={actual_outputs}"
            )

    cache_tensors = [
        cache_factory(
            torch.zeros(tuple(contract["cache_shape"]), dtype=_DEPLOYMENT_DTYPE, device=device)
        )
        for _ in contract["cache_input_names"]
    ]
    return HunyuanOCRDraftGraphs(
        context=sessions["context"],
        context_decode=sessions["context_decode"],
        decode=sessions["decode"],
        cache_input_names=tuple(contract["cache_input_names"]),
        cache_output_names=tuple(contract["cache_output_names"]),
        cache_tensors=cache_tensors,
        hidden_width=int(contract["hidden_width"]),
        draft_hidden_size=int(contract["draft_hidden_size"]),
        capacity=int(contract["capacity"]),
    )


class HunyuanOCRSpeculativeDecoder:
    """Drive draft proposal, target verification, and dual-cache commits."""

    def __init__(
        self,
        *,
        graphs: HunyuanOCRDraftGraphs,
        draft_cache: HunyuanOCRDraftCacheController,
        embedding_weight: torch.Tensor,
        mask_token_id: int,
        generation_eos_token_ids: Sequence[int],
        block_size: int,
        num_draft_tokens: int,
        run_target_verify: Callable[..., Any],
        commit_verify_prefix: Callable[..., int],
        discard_verify_result: Callable[..., None],
        run_decode_step: Callable[[int], torch.Tensor],
        target_logical_past_length: Callable[[], int],
    ) -> None:
        if embedding_weight.dtype != _DEPLOYMENT_DTYPE:
            raise HunyuanOCRSpeculativeRuntimeError(
                f"draft embedding weight must use float16 deployment dtype, got {embedding_weight.dtype}"
            )
        if not 1 <= num_draft_tokens <= block_size - 1:
            raise HunyuanOCRSpeculativeRuntimeError(
                f"num_draft_tokens must satisfy 1 <= k <= {block_size - 1}, got {num_draft_tokens}"
            )
        self._graphs = graphs
        self._draft_cache = draft_cache
        self._embedding_weight = embedding_weight
        self._mask_token_id = int(mask_token_id)
        self._generation_eos_token_ids = tuple(int(token_id) for token_id in generation_eos_token_ids)
        self._block_size = int(block_size)
        self._num_draft_tokens = int(num_draft_tokens)
        self._run_target_verify = run_target_verify
        self._commit_verify_prefix = commit_verify_prefix
        self._discard_verify_result = discard_verify_result
        self._run_decode_step = run_decode_step
        self._target_logical_past_length = target_logical_past_length
        self.stats = SpeculativeStats(
            num_draft_tokens=self._num_draft_tokens,
            block_size=self._block_size,
        )

    @property
    def block_size(self) -> int:
        return self._block_size

    @property
    def num_draft_tokens(self) -> int:
        return self._num_draft_tokens

    def reset(self) -> None:
        self._draft_cache.reset()
        self._graphs.reset_caches()
        self.stats = SpeculativeStats(
            num_draft_tokens=self._num_draft_tokens,
            block_size=self._block_size,
        )

    def append_context(self, target_hidden: torch.Tensor) -> int:
        span = self._validate_hidden_span(target_hidden)
        session = self._graphs.context
        padded = self._pad_hidden(target_hidden, self._static_hidden_length(session, span))
        past_seq_length = self._draft_cache.begin_context(current_input_length=span)
        self._run_context_graph(
            session,
            padded,
            past_seq_length=past_seq_length,
            current_input_length=span,
        )
        self.stats.draft_context_calls += 1
        return self._draft_cache.commit_context(current_input_length=span)

    def _validate_hidden_span(self, target_hidden: torch.Tensor) -> int:
        if target_hidden.ndim != 3 or target_hidden.shape[0] != 1:
            raise HunyuanOCRSpeculativeRuntimeError(
                f"draft context hidden must have shape [1, sequence, hidden], got {tuple(target_hidden.shape)}"
            )
        if int(target_hidden.shape[2]) != self._graphs.hidden_width:
            raise HunyuanOCRSpeculativeRuntimeError(
                "draft context hidden width does not match metadata: "
                f"expected={self._graphs.hidden_width}, got={int(target_hidden.shape[2])}"
            )
        span = int(target_hidden.shape[1])
        if span <= 0:
            raise HunyuanOCRSpeculativeRuntimeError(
                "draft context requires at least one hidden position"
            )
        return span

    @staticmethod
    def _static_hidden_length(session: Any, span: int) -> int:
        static_length = int(session.get_input("target_hidden").shape[1])
        if span > static_length:
            raise HunyuanOCRSpeculativeRuntimeError(
                "draft context span exceeds the exported static length: "
                f"span={span}, static_length={static_length}"
            )
        return static_length

    def _run_context_graph(
        self,
        session: Any,
        padded_hidden: torch.Tensor,
        *,
        past_seq_length: int,
        current_input_length: int,
    ) -> None:
        started = time.perf_counter()
        outputs = self._run_draft_session(
            session,
            {
                "target_hidden": padded_hidden,
                "past_seq_length": self._scalar(past_seq_length),
                "current_input_length": self._scalar(current_input_length),
            },
        )
        self.stats.add_time("draft", time.perf_counter() - started)
        self._rebind_caches(outputs)

    def propose(self, current_token_id: int) -> tuple[list[int], int]:
        transaction_id = self._draft_cache.begin_decode()
        past_seq_length = self._draft_cache.committed_length
        noise_embedding = build_dflash_noise_embedding(
            current_token_id=int(current_token_id),
            embedding_weight=self._embedding_weight,
            mask_token_id=self._mask_token_id,
            block_size=self._block_size,
        )
        attn_mask = torch.full(
            (1, self._graphs.capacity),
            _ATTENTION_MASK_BLOCK,
            dtype=_DEPLOYMENT_DTYPE,
            device=noise_embedding.device,
        )
        attn_mask[:, : past_seq_length + self._block_size] = _ATTENTION_MASK_KEEP
        started = time.perf_counter()
        try:
            outputs = self._run_draft_session(
                self._graphs.decode,
                {
                    "noise_embedding": noise_embedding.to(_DEPLOYMENT_DTYPE),
                    "past_seq_length": self._scalar(past_seq_length),
                    "current_input_length": self._scalar(self._block_size),
                    "attn_mask": attn_mask,
                },
            )
        except Exception:
            self._draft_cache.mark_execution_failed(transaction_id=transaction_id)
            raise
        self.stats.add_time("draft", time.perf_counter() - started)
        self.stats.draft_decode_calls += 1
        draft_logits = self._restore_batch_axis(outputs["draft_logits"])
        if draft_logits.ndim != 3 or tuple(draft_logits.shape[:2]) != (1, self._block_size - 1):
            raise HunyuanOCRSpeculativeRuntimeError(
                "draft decode must emit candidate logits as [1, block_size - 1, vocab], "
                f"got {tuple(draft_logits.shape)}"
            )
        candidates = draft_logits[0, : self._num_draft_tokens].argmax(dim=-1)
        return [int(token_id) for token_id in candidates.tolist()], transaction_id

    def narrow_and_commit(
        self,
        *,
        transaction_id: int,
        accepted_draft_count: int,
        target_hidden: torch.Tensor,
    ) -> int:
        span = self._validate_hidden_span(target_hidden)
        accepted_prefix_length = self._draft_cache.begin_context_decode(
            accepted_draft_count=accepted_draft_count,
            transaction_id=transaction_id,
        )
        if span != accepted_prefix_length:
            raise HunyuanOCRSpeculativeRuntimeError(
                "draft context_decode hidden span must equal accepted_draft_count + 1: "
                f"span={span}, accepted_prefix_length={accepted_prefix_length}"
            )
        session = self._graphs.context_decode
        padded = self._pad_hidden(target_hidden, self._static_hidden_length(session, span))
        self._run_context_graph(
            session,
            padded,
            past_seq_length=self._draft_cache.committed_length,
            current_input_length=span,
        )
        self.stats.draft_context_decode_calls += 1
        return self._draft_cache.commit_context_decode(transaction_id=transaction_id)

    def _run_draft_session(self, session: Any, feed: Mapping[str, Any]) -> dict[str, Any]:
        full_feed = dict(feed)
        for name, cache in zip(
            self._graphs.cache_input_names,
            self._graphs.cache_tensors,
            strict=True,
        ):
            full_feed[name] = cache
        outputs = session.run(full_feed)
        return dict(zip(session.get_output_names(), outputs, strict=True))

    def _rebind_caches(self, outputs: Mapping[str, Any]) -> None:
        cache_factory = type(self._graphs.cache_tensors[0])
        self._graphs.cache_tensors = [
            cache_factory(outputs[name]) for name in self._graphs.cache_output_names
        ]

    @staticmethod
    def _restore_batch_axis(logits: torch.Tensor) -> torch.Tensor:
        if logits.ndim == 2:
            return logits.unsqueeze(0)
        return logits

    @staticmethod
    def _pad_hidden(hidden: torch.Tensor, static_length: int) -> torch.Tensor:
        hidden = hidden.to(_DEPLOYMENT_DTYPE)
        current = int(hidden.shape[1])
        if current == static_length:
            return hidden.contiguous()
        padding = torch.zeros(
            (1, static_length - current, int(hidden.shape[2])),
            dtype=hidden.dtype,
            device=hidden.device,
        )
        return torch.cat((hidden, padding), dim=1).contiguous()

    def _scalar(self, value: int) -> torch.Tensor:
        return torch.tensor(
            [int(value)],
            dtype=torch.int32,
            device=self._embedding_weight.device,
        )

    def has_draft_capacity(self) -> bool:
        return self._draft_cache.committed_length + self._block_size <= self._draft_cache.capacity

    def assert_lengths_agree(self, *, draft_length: int) -> None:
        target_length = int(self._target_logical_past_length())
        if draft_length != target_length:
            raise HunyuanOCRSpeculativeRuntimeError(
                "HunyuanOCR DFlash dual-cache length divergence: "
                f"draft_committed_length={draft_length}, target_logical_past_length={target_length}"
            )

    def run(
        self,
        *,
        first_token_id: int,
        max_new_tokens: int,
        on_committed_tokens: Callable[[Sequence[int]], None] | None = None,
    ) -> list[int]:
        if max_new_tokens <= 0:
            self.stats.stop_reason = STOP_MAX_LENGTH
            return []
        produced = [int(first_token_id)]
        self._emit_committed_tokens(on_committed_tokens, produced)
        if produced[0] in self._generation_eos_token_ids:
            self.stats.stop_reason = STOP_EOS
            return produced
        if len(produced) >= max_new_tokens:
            self.stats.stop_reason = STOP_MAX_LENGTH
            return produced

        current_token_id = produced[0]
        while True:
            if not self.has_draft_capacity():
                self._record_fallback(FALLBACK_CAPACITY_SHORTFALL)
                return self._finish_with_plain_ar(
                    produced,
                    current_token_id,
                    max_new_tokens,
                    on_committed_tokens=on_committed_tokens,
                )
            try:
                draft_token_ids, transaction_id = self.propose(current_token_id)
            except HunyuanOCRSpeculativeRuntimeError:
                raise
            except Exception:
                self._record_fallback(FALLBACK_DRAFT_EXECUTION_FAILED)
                return self._finish_with_plain_ar(
                    produced,
                    current_token_id,
                    max_new_tokens,
                    on_committed_tokens=on_committed_tokens,
                )

            past_length_before_block = int(self._target_logical_past_length())
            started = time.perf_counter()
            verify_result = self._run_target_verify(
                current_token_id=current_token_id,
                draft_token_ids=draft_token_ids,
            )
            self.stats.add_time("verify", time.perf_counter() - started)
            self.stats.target_verify_calls += 1
            if verify_result.requires_commit is not True:
                self._draft_cache.discard_decode(transaction_id=transaction_id)
                self._discard_verify_result(transaction_id=verify_result.transaction_id)
                raise HunyuanOCRSpeculativeRuntimeError(
                    "HunyuanOCR speculative verify unexpectedly returned a decode fallback"
                )

            predicted = self._greedy_predictions(verify_result, len(draft_token_ids))
            matched_count = 0
            for index, draft_token_id in enumerate(draft_token_ids):
                if draft_token_id != predicted[index]:
                    break
                matched_count += 1

            accepted_count = self._truncate_accepted_count(draft_token_ids, matched_count)
            try:
                draft_length = self.narrow_and_commit(
                    transaction_id=transaction_id,
                    accepted_draft_count=accepted_count,
                    target_hidden=verify_result.valid_target_hidden[:, : accepted_count + 1],
                )
            except Exception:
                self._discard_draft_transaction(transaction_id)
                self._discard_verify_transaction(verify_result.transaction_id)
                self._record_fallback(FALLBACK_DRAFT_EXECUTION_FAILED)
                return self._finish_with_plain_ar(
                    produced,
                    current_token_id,
                    max_new_tokens,
                    on_committed_tokens=on_committed_tokens,
                )

            committed_length = self._commit_verify_prefix(
                transaction_id=verify_result.transaction_id,
                accepted_draft_count=accepted_count,
            )
            if committed_length != past_length_before_block + accepted_count + 1:
                raise HunyuanOCRSpeculativeRuntimeError(
                    "HunyuanOCR verify commit produced an inconsistent accept count: "
                    f"committed_length={committed_length}, past_length={past_length_before_block}, "
                    f"accepted_count={accepted_count}"
                )
            self.assert_lengths_agree(draft_length=draft_length)
            self.stats.record_block(accepted_count, len(draft_token_ids))

            produced_length_before_block = len(produced)
            accepted_tokens = list(draft_token_ids[:accepted_count])
            produced.extend(accepted_tokens)
            if any(token_id in self._generation_eos_token_ids for token_id in accepted_tokens):
                self._truncate_to_limit(produced, max_new_tokens)
                visible_count = max(0, max_new_tokens - produced_length_before_block)
                self._emit_committed_tokens(on_committed_tokens, accepted_tokens[:visible_count])
                self.stats.stop_reason = STOP_EOS
                return produced

            next_token_id = predicted[accepted_count]
            produced.append(next_token_id)
            visible_block = [*accepted_tokens, next_token_id]
            if next_token_id in self._generation_eos_token_ids:
                self._truncate_to_limit(produced, max_new_tokens)
                visible_count = max(0, max_new_tokens - produced_length_before_block)
                self._emit_committed_tokens(on_committed_tokens, visible_block[:visible_count])
                self.stats.stop_reason = STOP_EOS
                return produced
            if len(produced) >= max_new_tokens:
                self._truncate_to_limit(produced, max_new_tokens)
                visible_count = max(0, max_new_tokens - produced_length_before_block)
                self._emit_committed_tokens(on_committed_tokens, visible_block[:visible_count])
                self.stats.stop_reason = STOP_MAX_LENGTH
                return produced

            self._emit_committed_tokens(on_committed_tokens, visible_block)
            current_token_id = next_token_id

    def _truncate_accepted_count(self, draft_token_ids: Sequence[int], accepted_count: int) -> int:
        for index, token_id in enumerate(draft_token_ids[:accepted_count]):
            if token_id in self._generation_eos_token_ids:
                return index + 1
        return accepted_count

    def _discard_draft_transaction(self, transaction_id: int) -> None:
        try:
            self._draft_cache.discard_decode(transaction_id=transaction_id)
        except (RuntimeError, ValueError):
            pass

    def _discard_verify_transaction(self, transaction_id: int) -> None:
        try:
            self._discard_verify_result(transaction_id=transaction_id)
        except (RuntimeError, ValueError):
            pass

    @staticmethod
    def _emit_committed_tokens(
        callback: Callable[[Sequence[int]], None] | None,
        token_ids: Sequence[int],
    ) -> None:
        if callback is not None and token_ids:
            callback(tuple(int(token_id) for token_id in token_ids))

    @staticmethod
    def _greedy_predictions(verify_result: Any, draft_count: int) -> list[int]:
        logits = verify_result.valid_logits
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise HunyuanOCRSpeculativeRuntimeError(
                f"verify logits must have shape [1, sequence, vocab], got {tuple(logits.shape)}"
            )
        if int(logits.shape[1]) != draft_count + 1:
            raise HunyuanOCRSpeculativeRuntimeError(
                "verify logits length must equal 1 + number of draft tokens: "
                f"expected={draft_count + 1}, got={int(logits.shape[1])}"
            )
        return [int(token_id) for token_id in logits[0].argmax(dim=-1).tolist()]

    def _record_fallback(self, reason: str) -> None:
        if self.stats.fallback_reason is None:
            self.stats.fallback_reason = reason
            self.stats.fallback_block_index = self.stats.blocks

    def _truncate_to_limit(self, produced: list[int], max_new_tokens: int) -> None:
        if len(produced) > max_new_tokens:
            self.stats.truncated_tokens += len(produced) - max_new_tokens
            del produced[max_new_tokens:]

    def _finish_with_plain_ar(
        self,
        produced: list[int],
        current_token_id: int,
        max_new_tokens: int,
        *,
        on_committed_tokens: Callable[[Sequence[int]], None] | None = None,
    ) -> list[int]:
        while len(produced) < max_new_tokens:
            started = time.perf_counter()
            logits = self._run_decode_step(current_token_id)
            self.stats.add_time("decode", time.perf_counter() - started)
            self.stats.target_decode_calls += 1
            next_token_id = int(logits[0, -1].argmax(dim=-1).item())
            produced.append(next_token_id)
            self._emit_committed_tokens(on_committed_tokens, [next_token_id])
            if next_token_id in self._generation_eos_token_ids:
                self.stats.stop_reason = STOP_EOS
                return produced
            current_token_id = next_token_id
        self.stats.stop_reason = STOP_MAX_LENGTH
        return produced


__all__ = [
    "FALLBACK_ARTIFACTS_MISSING",
    "FALLBACK_CAPACITY_SHORTFALL",
    "FALLBACK_DISABLED",
    "FALLBACK_DRAFT_EXECUTION_FAILED",
    "HunyuanOCRDraftGraphs",
    "HunyuanOCRSpeculativeDecoder",
    "HunyuanOCRSpeculativeRuntimeError",
    "STOP_EOS",
    "STOP_MAX_LENGTH",
    "SpeculativeStats",
    "load_draft_graphs",
    "resolve_draft_runtime_contract",
]