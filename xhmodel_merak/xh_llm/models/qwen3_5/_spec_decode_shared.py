from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch


@dataclass
class SpecDecodeVerifyResult:
    initial_seq_len: int
    verify_token_ids: List[int]
    predicted_token_ids: List[int]
    verify_hidden: Optional[torch.Tensor]
    raw_result: Any


def run_spec_decode_loop(
    *,
    max_new_tokens: int,
    initial_token_id: int,
    initial_past_seq_len: int,
    initial_last_hidden: Optional[torch.Tensor],
    initial_mtp_past_seq_len: int,
    eos_token_id: Optional[int],
    num_drafts: int,
    run_draft: Callable[[int, Optional[torch.Tensor], int, int, int], List[int]],
    verify_round: Callable[[int, List[int], int], SpecDecodeVerifyResult],
    apply_verify_result: Callable[[SpecDecodeVerifyResult, int], None],
    post_verify: Callable[
        [SpecDecodeVerifyResult, int, int, int, int],
        Tuple[Optional[torch.Tensor], int],
    ],
    on_token: Optional[Callable[[int], None]] = None,
) -> Tuple[List[int], Dict[str, Union[int, float, List[int]]]]:
    if max_new_tokens <= 0:
        return [], {
            "num_rounds": 0,
            "output_tokens": 0,
            "draft_tokens_total": 0,
            "accepted_drafts_total": 0,
            "avg_accepted_per_round": 0.0,
            "draft_capacity_per_round": num_drafts,
            "accepted_drafts_per_round": [],
        }

    generated_ids: List[int] = []

    def emit(token_id: int) -> None:
        generated_ids.append(token_id)
        if on_token is not None:
            on_token(token_id)

    current_token_id = initial_token_id
    past_seq_len = initial_past_seq_len
    last_hidden = initial_last_hidden
    mtp_past_seq_len = initial_mtp_past_seq_len

    emit(current_token_id)
    if len(generated_ids) >= max_new_tokens:
        return generated_ids, {
            "num_rounds": 0,
            "output_tokens": len(generated_ids),
            "draft_tokens_total": 0,
            "accepted_drafts_total": 0,
            "avg_accepted_per_round": 0.0,
            "draft_capacity_per_round": num_drafts,
            "accepted_drafts_per_round": [],
        }
    if eos_token_id is not None and current_token_id == eos_token_id:
        return generated_ids, {
            "num_rounds": 0,
            "output_tokens": len(generated_ids),
            "draft_tokens_total": 0,
            "accepted_drafts_total": 0,
            "avg_accepted_per_round": 0.0,
            "draft_capacity_per_round": num_drafts,
            "accepted_drafts_per_round": [],
        }

    total_rounds = 0
    total_draft_tokens = 0
    total_accepted_tokens = 0
    accepted_drafts_per_round: List[int] = []

    while len(generated_ids) < max_new_tokens:
        total_rounds += 1
        draft_token_ids = run_draft(
            current_token_id,
            last_hidden,
            past_seq_len,
            mtp_past_seq_len,
            num_drafts,
        )
        total_draft_tokens += len(draft_token_ids)

        verify_result = verify_round(current_token_id, draft_token_ids, past_seq_len)
        accepted_count = 0
        for predicted_token_id, draft_token_id in zip(
            verify_result.predicted_token_ids, draft_token_ids, strict=False
        ):
            if predicted_token_id != draft_token_id:
                break
            accepted_count += 1

        total_accepted_tokens += accepted_count
        accepted_drafts_per_round.append(accepted_count)
        # ``accepted_steps`` counts target-executed cache snapshots, not only
        # accepted draft tokens.  Even when no draft token matches, the
        # verifier contributes its recovery token, so snapshot zero must be
        # committed and this value is always at least one.
        accepted_steps = accepted_count + 1
        apply_verify_result(verify_result, accepted_steps)
        past_seq_len = verify_result.initial_seq_len + accepted_steps

        eos_hit = False
        for token_id in draft_token_ids[:accepted_count]:
            emit(token_id)
            if eos_token_id is not None and token_id == eos_token_id:
                eos_hit = True
                break
            if len(generated_ids) >= max_new_tokens:
                break
        if eos_hit or len(generated_ids) >= max_new_tokens:
            break

        if accepted_count < len(draft_token_ids):
            next_token_id = verify_result.predicted_token_ids[accepted_count]
        else:
            bonus_index = len(draft_token_ids)
            if bonus_index >= len(verify_result.predicted_token_ids):
                raise RuntimeError(
                    "Verify round did not return the bonus prediction required for speculative decoding."
                )
            next_token_id = verify_result.predicted_token_ids[bonus_index]

        last_hidden, mtp_past_seq_len = post_verify(
            verify_result,
            accepted_count,
            accepted_steps,
            next_token_id,
            mtp_past_seq_len,
        )
        current_token_id = next_token_id
        if eos_token_id is not None and current_token_id == eos_token_id:
            break
        emit(current_token_id)

    avg_accepted = (
        total_accepted_tokens / total_rounds if total_rounds > 0 else 0.0
    )
    stats: Dict[str, Union[int, float, List[int]]] = {
        "num_rounds": total_rounds,
        "output_tokens": len(generated_ids),
        "draft_tokens_total": total_draft_tokens,
        "accepted_drafts_total": total_accepted_tokens,
        "avg_accepted_per_round": avg_accepted,
        "draft_capacity_per_round": num_drafts,
        "accepted_drafts_per_round": accepted_drafts_per_round,
    }
    return generated_ids, stats
