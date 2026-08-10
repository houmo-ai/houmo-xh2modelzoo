# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class TargetVerifyResult:
    transaction_id: int
    logits: Tensor
    target_hidden: Tensor
    valid_length: int
    max_draft_tokens: int
    requires_commit: bool = True

    @property
    def valid_logits(self) -> Tensor:
        return self.logits[:, : self.valid_length]

    @property
    def valid_target_hidden(self) -> Tensor:
        return self.target_hidden[:, : self.valid_length]


@dataclass(frozen=True)
class _PendingVerifyTransaction:
    transaction_id: int
    past_seq_length: int
    current_input_length: int
    draft_token_ids: tuple[int, ...]


def commit_verify_prefix_length(
    *,
    past_seq_length: int,
    current_input_length: int,
    accepted_draft_count: int,
    max_sequence_length: int,
) -> int:
    """Return the logical target-cache length after a verified prefix commit."""

    values = {
        "past_seq_length": past_seq_length,
        "current_input_length": current_input_length,
        "accepted_draft_count": accepted_draft_count,
        "max_sequence_length": max_sequence_length,
    }
    for field, value in values.items():
        if type(value) is not int:
            raise ValueError(f"{field} must be an integer, got {type(value).__name__}")
    if past_seq_length < 0:
        raise ValueError(f"past_seq_length must be non-negative, got {past_seq_length}")
    if current_input_length <= 0:
        raise ValueError(f"current_input_length must be positive, got {current_input_length}")
    if accepted_draft_count < 0 or accepted_draft_count > current_input_length - 1:
        raise ValueError(
            "accepted_draft_count must satisfy 0 <= accepted_draft_count < current_input_length, "
            f"got accepted_draft_count={accepted_draft_count}, current_input_length={current_input_length}"
        )
    if max_sequence_length <= 0:
        raise ValueError(f"max_sequence_length must be positive, got {max_sequence_length}")
    if past_seq_length + current_input_length > max_sequence_length:
        raise ValueError(
            "Verify input exceeds max_sequence_length: "
            f"past_seq_length={past_seq_length}, current_input_length={current_input_length}, "
            f"max_sequence_length={max_sequence_length}"
        )

    committed_length = past_seq_length + accepted_draft_count + 1
    if committed_length > max_sequence_length:
        raise ValueError(
            "Committed verify prefix exceeds max_sequence_length: "
            f"committed_length={committed_length}, max_sequence_length={max_sequence_length}"
        )
    return committed_length


class HunyuanOCRTargetVerifyController:
    """Own request-local target verify transaction state without owning KV storage."""

    def __init__(
        self,
        *,
        max_sequence_length: int,
        verify_input_length: int,
        pad_token_id: int = 0,
        generation_eos_token_ids: Sequence[int] = (),
        logits_width: int | None = None,
        target_hidden_width: int | None = None,
        output_dtype: torch.dtype | None = None,
    ) -> None:
        if type(max_sequence_length) is not int or max_sequence_length <= 0:
            raise ValueError(f"max_sequence_length must be a positive integer, got {max_sequence_length!r}")
        if type(verify_input_length) is not int or verify_input_length <= 1:
            raise ValueError(
                f"verify_input_length must be an integer greater than 1, got {verify_input_length!r}"
            )
        if type(pad_token_id) is not int:
            raise ValueError(f"pad_token_id must be an integer, got {type(pad_token_id).__name__}")
        self.max_sequence_length = max_sequence_length
        self.verify_input_length = verify_input_length
        self.max_draft_tokens = verify_input_length - 1
        self.pad_token_id = pad_token_id
        self.generation_eos_token_ids = frozenset(generation_eos_token_ids)
        if any(type(token_id) is not int for token_id in self.generation_eos_token_ids):
            raise ValueError("generation_eos_token_ids must contain only integers")
        for name, width in (("logits_width", logits_width), ("target_hidden_width", target_hidden_width)):
            if width is not None and (type(width) is not int or width <= 0):
                raise ValueError(f"{name} must be a positive integer when provided, got {width!r}")
        if output_dtype is not None and not isinstance(output_dtype, torch.dtype):
            raise ValueError(f"output_dtype must be a torch.dtype when provided, got {output_dtype!r}")
        self.logits_width = logits_width
        self.target_hidden_width = target_hidden_width
        self.output_dtype = output_dtype
        self._logical_past_length = 0
        self._rope_delta = 0
        self._pending: _PendingVerifyTransaction | None = None
        self._poisoned = False
        self._next_transaction_id = 1

    @property
    def logical_past_length(self) -> int:
        return self._logical_past_length

    @property
    def has_pending_transaction(self) -> bool:
        return self._pending is not None

    def require_idle(self) -> None:
        self._ensure_available()
        if self._pending is not None:
            raise RuntimeError("verify_transaction_pending: resolve the current transaction before continuing")

    def reset_generation_state(self) -> None:
        self._logical_past_length = 0
        self._rope_delta = 0
        self._pending = None
        self._poisoned = False

    def restore_request_state(self, *, past_seq_length: int, rope_delta: int) -> None:
        self._ensure_available()
        if self._pending is not None:
            raise RuntimeError(
                "verify_transaction_pending: cannot restore request state while a transaction is pending"
            )
        if type(past_seq_length) is not int or past_seq_length < 0:
            raise ValueError(f"past_seq_length must be a non-negative integer, got {past_seq_length!r}")
        if type(rope_delta) is not int:
            raise ValueError(f"rope_delta must be an integer, got {type(rope_delta).__name__}")
        if past_seq_length > self.max_sequence_length:
            raise ValueError(
                f"past_seq_length exceeds max_sequence_length: {past_seq_length} > {self.max_sequence_length}"
            )
        self._logical_past_length = past_seq_length
        self._rope_delta = rope_delta

    def run_target_verify(
        self,
        *,
        current_token_id: int,
        draft_token_ids: Sequence[int],
        executor: Callable[..., tuple[Tensor, Tensor]],
    ) -> TargetVerifyResult:
        self._ensure_available()
        if self._pending is not None:
            raise RuntimeError("verify_transaction_pending: resolve the current transaction before verify")
        if type(current_token_id) is not int:
            raise ValueError(f"verify_input_invalid: current_token_id must be an integer, got {current_token_id!r}")
        if not isinstance(draft_token_ids, Sequence) or isinstance(draft_token_ids, (str, bytes)):
            raise ValueError("verify_input_invalid: draft_token_ids must be an integer sequence")
        draft_tokens = list(draft_token_ids)
        if any(type(token_id) is not int for token_id in draft_tokens):
            raise ValueError("verify_input_invalid: draft_token_ids must contain only integers")
        if len(draft_tokens) > self.max_draft_tokens:
            raise ValueError(
                "verify_input_invalid: draft token count exceeds the graph contract: "
                f"drafts={len(draft_tokens)}, max_draft_tokens={self.max_draft_tokens}"
            )
        current_input_length = len(draft_tokens) + 1
        if self._logical_past_length + current_input_length > self.max_sequence_length:
            raise ValueError(
                "verify_input_invalid: past_seq_length + current_input_length exceeds max_sequence_length: "
                f"{self._logical_past_length} + {current_input_length} > {self.max_sequence_length}"
            )

        token_values = [current_token_id, *draft_tokens]
        token_values.extend([self.pad_token_id] * (self.verify_input_length - current_input_length))
        input_token_ids = torch.tensor([token_values], dtype=torch.long)
        decode_start = self._logical_past_length + self._rope_delta
        if decode_start < 0:
            raise ValueError(
                "verify_input_invalid: past_seq_length + rope_delta must be non-negative, "
                f"got {self._logical_past_length} + {self._rope_delta}"
            )
        valid_positions = torch.arange(decode_start, decode_start + current_input_length, dtype=torch.long)
        padded_positions = torch.zeros(self.verify_input_length, dtype=torch.long)
        padded_positions[:current_input_length] = valid_positions
        position_ids = padded_positions.view(1, 1, -1).expand(4, 1, -1).clone()

        try:
            output = executor(
                input_token_ids=input_token_ids,
                position_ids=position_ids,
                past_seq_length=self._logical_past_length,
                current_input_length=current_input_length,
            )
        except Exception as error:
            self._poisoned = True
            raise RuntimeError(f"verify_execution_failed: {error}") from error

        try:
            logits, target_hidden = self._validate_output(output)
        except Exception as error:
            self._poisoned = True
            raise RuntimeError(f"verify_output_contract_mismatch: {error}") from error

        transaction_id = self._next_transaction_id
        self._next_transaction_id += 1
        self._pending = _PendingVerifyTransaction(
            transaction_id=transaction_id,
            past_seq_length=self._logical_past_length,
            current_input_length=current_input_length,
            draft_token_ids=tuple(draft_tokens),
        )
        return TargetVerifyResult(
            transaction_id=transaction_id,
            logits=logits,
            target_hidden=target_hidden,
            valid_length=current_input_length,
            max_draft_tokens=self.max_draft_tokens,
            requires_commit=True,
        )

    def run_decode_fallback(self, executor: Callable[[], tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]:
        self.require_idle()
        try:
            output = executor()
        except Exception as error:
            self._poisoned = True
            raise RuntimeError(f"verify_execution_failed: {error}") from error
        try:
            return self._validate_output(output, sequence_length=1)
        except Exception as error:
            self._poisoned = True
            raise RuntimeError(f"verify_output_contract_mismatch: {error}") from error

    def commit_verify_prefix(self, *, transaction_id: int, accepted_draft_count: int) -> int:
        pending = self._require_pending(transaction_id)
        commit_verify_prefix_length(
            past_seq_length=pending.past_seq_length,
            current_input_length=pending.current_input_length,
            accepted_draft_count=accepted_draft_count,
            max_sequence_length=self.max_sequence_length,
        )
        accepted_draft_count = self._truncate_accepted_count_at_eos(pending, accepted_draft_count)
        committed_length = commit_verify_prefix_length(
            past_seq_length=pending.past_seq_length,
            current_input_length=pending.current_input_length,
            accepted_draft_count=accepted_draft_count,
            max_sequence_length=self.max_sequence_length,
        )
        self._logical_past_length = committed_length
        self._pending = None
        return committed_length

    def _truncate_accepted_count_at_eos(
        self,
        pending: _PendingVerifyTransaction,
        accepted_draft_count: int,
    ) -> int:
        if type(accepted_draft_count) is not int:
            return accepted_draft_count
        accepted_tokens = pending.draft_token_ids[:accepted_draft_count]
        for index, token_id in enumerate(accepted_tokens):
            if token_id in self.generation_eos_token_ids:
                return index + 1
        return accepted_draft_count

    def discard_verify_result(self, *, transaction_id: int) -> None:
        self._require_pending(transaction_id)
        self._pending = None

    def _ensure_available(self) -> None:
        if self._poisoned:
            raise RuntimeError("verify_request_poisoned: reset and prefill the request before continuing")

    def _require_pending(self, transaction_id: int) -> _PendingVerifyTransaction:
        self._ensure_available()
        if type(transaction_id) is not int:
            raise ValueError(f"transaction_id must be an integer, got {type(transaction_id).__name__}")
        if self._pending is None or self._pending.transaction_id != transaction_id:
            expected = self._pending.transaction_id if self._pending is not None else None
            raise ValueError(
                "transaction_id does not match the pending verify transaction: "
                f"got={transaction_id}, expected={expected}"
            )
        return self._pending

    def _validate_output(
        self,
        output: Any,
        *,
        sequence_length: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise TypeError("verify executor must return (logits, target_hidden)")
        logits, target_hidden = output
        expected_sequence_length = self.verify_input_length if sequence_length is None else sequence_length
        for name, tensor in (("logits", logits), ("target_hidden", target_hidden)):
            if not isinstance(tensor, Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if tensor.ndim != 3 or tensor.shape[0] != 1 or tensor.shape[1] != expected_sequence_length:
                raise ValueError(
                    f"{name} must have shape [1, {expected_sequence_length}, width], got {tuple(tensor.shape)}"
                )
            if tensor.shape[2] <= 0:
                raise ValueError(f"{name} width must be positive, got {tensor.shape[2]}")
            expected_width = self.logits_width if name == "logits" else self.target_hidden_width
            if expected_width is not None and tensor.shape[2] != expected_width:
                raise ValueError(f"{name} width must be {expected_width}, got {tensor.shape[2]}")
            if self.output_dtype is not None and tensor.dtype != self.output_dtype:
                raise ValueError(f"{name} dtype must be {self.output_dtype}, got {tensor.dtype}")
        return logits, target_hidden


__all__ = [
    "HunyuanOCRTargetVerifyController",
    "TargetVerifyResult",
    "commit_verify_prefix_length",
]