from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from xh_model_zoo.xh_llm.models.qwen3_5.qwen3_5_onnx_model import (
    _apply_presence_penalty,
    _apply_repetition_penalty,
    _clone_cache_value,
)
from xh_model_zoo.xh_llm.models.qwen3_5_moe.inference import (
    Qwen3_5MoeInference,
    _build_runtime_linear_attn_mask,
)
from xh_model_zoo.xh_llm.models.qwen3_5_moe.qwen3_5_moe_spec_decode_inference import (
    Qwen3_5MoeSpecDecodeInference,
)


def postprocess_chat_output(text: str, enable_thinking: bool) -> str:
    output = text.strip()
    if not enable_thinking:
        output = re.sub(r"<think>[\s\S]*?</think>", "", output)
        output = output.replace("<think>", "").replace("</think>", "")
    return output.strip()


def encode_chat_prompt(tokenizer, prompt: str, enable_thinking: bool) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return tokenizer([text], return_tensors="pt").input_ids


def _select_greedy_token_with_penalties(
    logits: torch.Tensor,
    history_token_ids: List[int],
    repetition_penalty: float,
    presence_penalty: float,
) -> tuple[int, torch.Tensor]:
    logits = _apply_repetition_penalty(logits, history_token_ids, repetition_penalty)
    logits = _apply_presence_penalty(logits, history_token_ids, presence_penalty)
    token_id = int(torch.argmax(logits, dim=-1).item())
    return token_id, logits


def _get_max_context_tokens(meta_path: str) -> Optional[int]:
    meta = json.load(open(meta_path, "r"))
    wrap_cfg = meta.get("wrap_cfg", {})
    value = wrap_cfg.get("context_length") or meta.get("max_context_tokens")
    return int(value) if value is not None else None


def _select_last_valid_logits(logits: torch.Tensor, valid_len: int) -> torch.Tensor:
    if logits.dim() == 2:
        return logits
    return logits[:, valid_len - 1, :]


def _load_baseline_runtime(meta_path: str, device: str, exec_device: str) -> Qwen3_5MoeSpecDecodeInference:
    return _load_spec_runtime(meta_path, device, exec_device)


def _load_spec_runtime(meta_path: str, device: str, exec_device: str) -> Qwen3_5MoeSpecDecodeInference:
    return Qwen3_5MoeSpecDecodeInference(meta_path, device=device, execution_device=exec_device)


def _release_runtime(runtime) -> None:
    for attr in (
        "prefill_session",
        "decode_session",
        "_draft_prefill_session",
        "_draft_context_session",
        "_draft_decode_session",
    ):
        if hasattr(runtime, attr):
            setattr(runtime, attr, None)
    del runtime
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@torch.no_grad()
def run_moe_target_baseline_from_spec(
    meta_path: str,
    prompt: str,
    max_new_tokens: int,
    device: str,
    exec_device: str,
    enable_thinking: bool = False,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
) -> Dict[str, Any]:
    runtime = _load_baseline_runtime(meta_path, device, exec_device)
    tokenizer = runtime.tokenizer
    input_ids = encode_chat_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
    max_context_tokens = _get_max_context_tokens(meta_path)
    if max_context_tokens is not None and input_ids.shape[1] > max_context_tokens:
        input_ids = input_ids[:, -max_context_tokens:]

    prefill_chunk_len = int(runtime.prefill_input_sequence_length)
    total_prompt_len = int(input_ids.shape[1])
    prefill_calls = 0
    last_prefill_logits = None
    past_seq_len = 0

    runtime.set_phase_prefill(True)
    try:
        for start in range(0, total_prompt_len, prefill_chunk_len):
            end = min(start + prefill_chunk_len, total_prompt_len)
            chunk_ids = input_ids[:, start:end]
            valid_len = int(chunk_ids.shape[1])
            (
                inputs_embeds,
                time_pos,
                hight_pos,
                width_pos,
                past_seq_t,
                seq_len_t,
                past_key_caches,
                past_value_caches,
                past_conv_caches,
                past_recurrent_states,
            ) = runtime.prepare_inputs(
                {"input_ids": chunk_ids, "past_seq_length": past_seq_len},
                prefill_chunk_len,
            )
            lin_mask = _build_runtime_linear_attn_mask(
                valid_len,
                prefill_chunk_len,
                inputs_embeds.dtype,
                runtime.execution_device,
            )
            logits = runtime._forward(
                inputs_embeds,
                time_pos,
                hight_pos,
                width_pos,
                past_seq_t,
                seq_len_t,
                lin_mask,
                past_key_caches,
                past_value_caches,
                past_conv_caches,
                past_recurrent_states,
            )
            last_prefill_logits = _select_last_valid_logits(logits, valid_len)
            prefill_calls += 1
            past_seq_len += valid_len

        if last_prefill_logits is None:
            return {
                "text": "",
                "token_ids": [],
                "target_prefill_calls": prefill_calls,
                "target_decoder_calls": 0,
                "output_tokens": 0,
            }

        history_token_ids = input_ids[0].tolist()
        next_token_id, _ = _select_greedy_token_with_penalties(
            last_prefill_logits,
            history_token_ids,
            repetition_penalty,
            presence_penalty,
        )
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None and next_token_id == eos_token_id:
            return {
                "text": "",
                "token_ids": [],
                "target_prefill_calls": prefill_calls,
                "target_decoder_calls": 0,
                "output_tokens": 0,
            }

        generated_ids: List[int] = [next_token_id]
        history_token_ids.append(next_token_id)
        decoder_calls = 0
        runtime.set_phase_prefill(False)
        dev = runtime._device
        verify_len = runtime.spec_decode_verify_length
        current_token = next_token_id
        while len(generated_ids) < max_new_tokens:
            decoder_calls += 1
            tok_emb = runtime._embed_token_ids(
                torch.tensor([[current_token]], dtype=torch.long, device=dev)
            )
            if verify_len > 1:
                pad = torch.zeros(
                    1,
                    verify_len - 1,
                    tok_emb.shape[2],
                    dtype=tok_emb.dtype,
                    device=dev,
                )
                tok_emb = torch.cat([tok_emb, pad], dim=1)
            time_pos = torch.arange(
                past_seq_len, past_seq_len + verify_len, dtype=torch.int32, device=dev
            ).unsqueeze(0)
            hight_pos = time_pos
            width_pos = time_pos
            past_seq_t = torch.tensor([past_seq_len], dtype=torch.int32, device=dev)
            seq_len_t = torch.ones(1, dtype=torch.int32, device=dev)
            lin_mask = _build_runtime_linear_attn_mask(
                1, verify_len, tok_emb.dtype, runtime.execution_device
            )
            decode_logits, _, _ = runtime._forward_with_hidden(
                tok_emb,
                time_pos,
                hight_pos,
                width_pos,
                past_seq_t,
                seq_len_t,
                lin_mask,
                runtime.past_key_caches,
                runtime.past_value_caches,
                runtime.past_conv_caches,
                runtime.past_recurrent_states,
            )
            current_token, _ = _select_greedy_token_with_penalties(
                decode_logits[:, 0, :],
                history_token_ids,
                repetition_penalty,
                presence_penalty,
            )
            if eos_token_id is not None and current_token == eos_token_id:
                break
            generated_ids.append(current_token)
            history_token_ids.append(current_token)
            past_seq_len += 1

        text = postprocess_chat_output(
            tokenizer.decode(generated_ids, skip_special_tokens=True),
            enable_thinking=enable_thinking,
        )
        return {
            "text": text,
            "token_ids": generated_ids,
            "target_prefill_calls": prefill_calls,
            "target_decoder_calls": decoder_calls,
            "output_tokens": len(generated_ids),
        }
    finally:
        _release_runtime(runtime)


@torch.no_grad()
def reset_moe_spec_runtime(runtime: Qwen3_5MoeSpecDecodeInference) -> None:
    runtime.set_phase_prefill(True)
    if runtime.spec_decode_mode == "mtp":
        runtime._reset_mtp_cache()
    else:
        runtime._dflash_cache_state = None


@torch.no_grad()
def run_moe_spec_metrics_with_runtime(
    *,
    meta_path: str,
    runtime: Qwen3_5MoeSpecDecodeInference,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool = False,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
) -> Dict[str, Any]:
    reset_moe_spec_runtime(runtime)
    tokenizer = runtime.tokenizer
    baseline = run_moe_target_baseline_from_spec(
        meta_path,
        prompt,
        max_new_tokens,
        str(runtime.device),
        str(runtime.execution_device),
        enable_thinking=enable_thinking,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
    )
    reference_token_ids = list(baseline.get("token_ids", []))[:max_new_tokens]

    if not reference_token_ids:
        draft_counts = (
            {
                "mtp_prefill_calls": 0,
                "mtp_decode_calls": 0,
            }
            if runtime.spec_decode_mode == "mtp"
            else {
                "dflash_prefill_calls": 0,
                "dflash_decode_calls": 0,
            }
        )
        return {
            "baseline": {
                "text": baseline["text"],
                "target_prefill_calls": int(baseline["target_prefill_calls"]),
                "target_decoder_calls": int(baseline["target_decoder_calls"]),
                "output_tokens": int(baseline["output_tokens"]),
            },
            "speculative": {
                "text": baseline["text"],
                "output_tokens": 0,
                "target_prefill_calls": int(baseline["target_prefill_calls"]),
                "target_decoder_calls": 0,
                "num_rounds": 0,
                "draft_capacity_per_round": 0,
                "accepted_drafts_per_round": [],
                "round_acceptance_rates": [],
                "accepted_drafts_total": 0,
                "draft_tokens_total": 0,
                "overall_acceptance_rate": 0.0,
                "avg_accepted_per_round": 0.0,
                **draft_counts,
            },
        }

    reset_moe_spec_runtime(runtime)
    input_ids = encode_chat_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
    max_context_tokens = runtime.meta_info.get("wrap_cfg", {}).get("context_length") or runtime.meta_info.get(
        "max_context_tokens"
    )
    max_context_tokens = int(max_context_tokens) if max_context_tokens is not None else None
    if max_context_tokens is not None and input_ids.shape[1] > max_context_tokens:
        input_ids = input_ids[:, -max_context_tokens:]

    prefill_chunk_len = int(runtime.prefill_input_sequence_length)
    total_prompt_len = int(input_ids.shape[1])
    last_prefill_logits = None
    last_hidden = None
    past_seq_len = 0
    target_prefill_calls = 0
    mtp_prefill_calls = 0
    mtp_decode_calls = 0
    dflash_prefill_calls = 0
    dflash_decode_calls = 0
    mtp_prefill_seq_len = 0
    mtp_pending_hidden = None

    runtime.set_phase_prefill(True)
    try:
        if runtime.spec_decode_mode == "mtp":
            runtime._reset_mtp_cache()
        for start in range(0, total_prompt_len, prefill_chunk_len):
            end = min(start + prefill_chunk_len, total_prompt_len)
            chunk_ids = input_ids[:, start:end]
            valid_len = int(chunk_ids.shape[1])
            (
                inputs_embeds,
                time_pos,
                hight_pos,
                width_pos,
                past_seq_t,
                seq_len_t,
                past_key_caches,
                past_value_caches,
                past_conv_caches,
                past_recurrent_states,
            ) = runtime.prepare_inputs(
                {"input_ids": chunk_ids, "past_seq_length": past_seq_len},
                prefill_chunk_len,
            )
            lin_mask = _build_runtime_linear_attn_mask(
                valid_len,
                prefill_chunk_len,
                inputs_embeds.dtype,
                runtime.execution_device,
            )
            logits, hidden_all, _ = runtime._forward_with_hidden(
                inputs_embeds,
                time_pos,
                hight_pos,
                width_pos,
                past_seq_t,
                seq_len_t,
                lin_mask,
                past_key_caches,
                past_value_caches,
                past_conv_caches,
                past_recurrent_states,
            )
            target_prefill_calls += 1
            last_prefill_logits = _select_last_valid_logits(logits, valid_len)
            if hidden_all is not None:
                hidden_all = hidden_all[:, :valid_len, :]
                last_hidden = runtime._select_hidden_step(hidden_all, valid_len - 1)
                if runtime.spec_decode_mode == "dflash":
                    runtime._append_dflash_context(hidden_all, past_seq_len)
                    dflash_prefill_calls += 1
                else:
                    hidden_parts = []
                    token_parts = []
                    if mtp_pending_hidden is not None:
                        hidden_parts.append(mtp_pending_hidden)
                        token_parts.append(chunk_ids[:, :1])
                    if valid_len > 1:
                        hidden_parts.append(hidden_all[:, : valid_len - 1, :])
                        token_parts.append(chunk_ids[:, 1:valid_len])
                    if hidden_parts:
                        mtp_hidden = torch.cat(hidden_parts, dim=1)
                        mtp_tokens = torch.cat(token_parts, dim=1)
                        runtime._prefill_mtp_chunk(
                            runtime._embed_token_ids(mtp_tokens.to(runtime.execution_device)),
                            mtp_hidden.to(runtime.execution_device),
                            mtp_prefill_seq_len,
                        )
                        mtp_prefill_calls += 1
                        mtp_prefill_seq_len += int(mtp_hidden.shape[1])
                    mtp_pending_hidden = hidden_all[:, valid_len - 1 : valid_len, :]
            past_seq_len += valid_len

        if last_prefill_logits is None:
            raise RuntimeError("Prefill did not produce logits.")

        history_token_ids = input_ids[0].tolist()
        current_token = reference_token_ids[0]
        history_token_ids.append(current_token)
        generated_ids: List[int] = [current_token]

        runtime.set_phase_prefill(False)
        mtp_past_seq_len = mtp_prefill_seq_len
        num_drafts = (
            runtime.spec_decode_block_size - 1
            if runtime.spec_decode_mode == "dflash"
            else runtime.spec_decode_block_size
        )
        target_decoder_calls = 0
        total_rounds = 0
        total_accepted_tokens = 0
        accepted_drafts_per_round: List[int] = []
        reference_limit = len(reference_token_ids)
        reference_index = 1

        while reference_index < reference_limit:
            total_rounds += 1

            if runtime.spec_decode_mode == "dflash":
                dflash_decode_calls += 1
                draft_tokens = runtime._run_draft_dflash(current_token, past_seq_len)
                mtp_snapshot = None
            else:
                mtp_snapshot = (
                    _clone_cache_value(runtime._mtp_k_cache),
                    _clone_cache_value(runtime._mtp_v_cache),
                )
                draft_tokens = []
                local_hidden = last_hidden
                for k in range(num_drafts):
                    if local_hidden is None:
                        break
                    tok_id = current_token if k == 0 else draft_tokens[-1]
                    tok_emb = runtime._embed_token_ids(
                        torch.tensor([[tok_id]], dtype=torch.long, device=runtime.execution_device)
                    )
                    d_logits, local_hidden = runtime._mtp_decode_step(
                        tok_emb,
                        local_hidden.to(runtime.execution_device),
                        mtp_past_seq_len + k,
                    )
                    mtp_decode_calls += 1
                    draft_tokens.append(int(d_logits[0, 0, :].argmax(dim=-1).item()))

            if not draft_tokens:
                target_decoder_calls += 1
                (
                    inputs_embeds,
                    time_pos,
                    hight_pos,
                    width_pos,
                    past_seq_t,
                    seq_len_t,
                    past_key_caches,
                    past_value_caches,
                    past_conv_caches,
                    past_recurrent_states,
                ) = runtime.prepare_inputs(
                    {"input_ids": torch.tensor([[current_token]]), "past_seq_length": past_seq_len},
                    1,
                )
                lin_mask = _build_runtime_linear_attn_mask(
                    1, 1, inputs_embeds.dtype, runtime.execution_device
                )
                logits, hidden, _ = runtime._forward_with_hidden(
                    inputs_embeds,
                    time_pos,
                    hight_pos,
                    width_pos,
                    past_seq_t,
                    seq_len_t,
                    lin_mask,
                    past_key_caches,
                    past_value_caches,
                    past_conv_caches,
                    past_recurrent_states,
                )
                next_tok = reference_token_ids[reference_index]
                generated_ids.append(next_tok)
                history_token_ids.append(next_tok)
                past_seq_len += 1
                current_token = next_tok
                reference_index += 1
                last_hidden = runtime._select_hidden_step(hidden, 0)
                if runtime.spec_decode_mode == "dflash":
                    runtime._append_dflash_context(
                        hidden[:, :1, :] if hidden is not None else None,
                        past_seq_len - 1,
                    )
                    dflash_prefill_calls += 1
                else:
                    mtp_past_seq_len += 1
                accepted_drafts_per_round.append(0)
                continue

            verify_ids = [current_token] + draft_tokens
            actual_vlen = len(verify_ids)
            verify_tokens_t = torch.tensor([verify_ids], dtype=torch.long)
            (
                inputs_embeds,
                time_pos,
                hight_pos,
                width_pos,
                past_seq_t,
                _seq_len_t,
                past_key_caches,
                past_value_caches,
                past_conv_caches,
                past_recurrent_states,
            ) = runtime.prepare_inputs(
                {"input_ids": verify_tokens_t, "past_seq_length": past_seq_len},
                runtime.spec_decode_verify_length,
            )
            lin_mask = _build_runtime_linear_attn_mask(
                actual_vlen,
                runtime.spec_decode_verify_length,
                inputs_embeds.dtype,
                runtime.execution_device,
            )
            target_decoder_calls += 1
            initial_seq_len = past_seq_len
            verify_logits, verify_hidden, verify_outputs = runtime._forward_with_hidden(
                inputs_embeds,
                time_pos,
                hight_pos,
                width_pos,
                past_seq_t,
                torch.tensor([actual_vlen], dtype=torch.int32, device=runtime.execution_device),
                lin_mask,
                past_key_caches,
                past_value_caches,
                past_conv_caches,
                past_recurrent_states,
                update_linear_cache=False,
            )

            accepted = 0
            for k, d_tok in enumerate(draft_tokens):
                if reference_index + k >= reference_limit:
                    break
                if d_tok != reference_token_ids[reference_index + k]:
                    break
                accepted += 1

            accepted_steps = accepted + 1
            runtime._apply_verify_linear_cache_outputs(
                past_conv_caches,
                past_recurrent_states,
                verify_outputs,
                accepted_steps=accepted_steps,
            )
            for _ in range(accepted):
                token_val = reference_token_ids[reference_index]
                generated_ids.append(token_val)
                history_token_ids.append(token_val)
                reference_index += 1
            accepted_drafts_per_round.append(accepted)
            total_accepted_tokens += accepted
            past_seq_len += accepted_steps
            if reference_index >= reference_limit:
                break
            new_tok = reference_token_ids[reference_index]
            current_token = new_tok

            last_hidden = runtime._select_hidden_step(verify_hidden, accepted)
            accepted_hidden = (
                verify_hidden[:, :accepted_steps, :]
                if verify_hidden is not None
                else None
            )
            if runtime.spec_decode_mode == "dflash":
                runtime._append_dflash_context(accepted_hidden, initial_seq_len)
                dflash_prefill_calls += 1
            else:
                if mtp_snapshot is not None:
                    runtime._mtp_k_cache.data = mtp_snapshot[0].data.clone()
                    runtime._mtp_v_cache.data = mtp_snapshot[1].data.clone()
                if accepted_hidden is not None:
                    mtp_initial_seq_len = mtp_past_seq_len
                    for step_idx in range(accepted_steps):
                        next_token_for_cache = (
                            verify_ids[step_idx + 1]
                            if step_idx < accepted_steps - 1
                            else new_tok
                        )
                        tok_emb = runtime._embed_token_ids(
                            torch.tensor([[next_token_for_cache]], dtype=torch.long, device=runtime.execution_device)
                        )
                        runtime._mtp_decode_step(
                            tok_emb,
                            accepted_hidden[:, step_idx : step_idx + 1, :],
                            mtp_initial_seq_len + step_idx,
                        )
                        mtp_decode_calls += 1
                    mtp_past_seq_len = mtp_initial_seq_len + accepted_steps

            generated_ids.append(new_tok)
            history_token_ids.append(new_tok)
            reference_index += 1

        text = postprocess_chat_output(
            tokenizer.decode(generated_ids, skip_special_tokens=True),
            enable_thinking=enable_thinking,
        )
        draft_capacity = max(num_drafts, 1)
        spec_output_tokens = len(generated_ids)
        return {
            "baseline": {
                "text": baseline["text"],
                "target_prefill_calls": int(baseline["target_prefill_calls"]),
                "target_decoder_calls": int(baseline["target_decoder_calls"]),
                "output_tokens": int(baseline["output_tokens"]),
            },
            "speculative": {
                "text": text,
                "output_tokens": spec_output_tokens,
                "target_prefill_calls": target_prefill_calls,
                "target_decoder_calls": target_decoder_calls,
                "num_rounds": total_rounds,
                "draft_capacity_per_round": draft_capacity,
                "accepted_drafts_per_round": accepted_drafts_per_round,
                "round_acceptance_rates": [
                    x / draft_capacity for x in accepted_drafts_per_round
                ],
                "accepted_drafts_total": total_accepted_tokens,
                "draft_tokens_total": total_rounds * draft_capacity,
                "overall_acceptance_rate": (
                    total_accepted_tokens / (total_rounds * draft_capacity)
                    if total_rounds
                    else 0.0
                ),
                "avg_accepted_per_round": (
                    total_accepted_tokens / total_rounds if total_rounds else 0.0
                ),
                **(
                    {
                        "mtp_prefill_calls": mtp_prefill_calls,
                        "mtp_decode_calls": mtp_decode_calls,
                    }
                    if runtime.spec_decode_mode == "mtp"
                    else {
                        "dflash_prefill_calls": dflash_prefill_calls,
                        "dflash_decode_calls": dflash_decode_calls,
                    }
                ),
            },
        }
    finally:
        reset_moe_spec_runtime(runtime)
 

@torch.no_grad()
def run_moe_spec_metrics(
    *,
    meta_path: str,
    prompt: str,
    max_new_tokens: int,
    device: str,
    exec_device: str,
    enable_thinking: bool = False,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
) -> Dict[str, Any]:
    runtime = _load_spec_runtime(meta_path, device, exec_device)
    try:
        return run_moe_spec_metrics_with_runtime(
            meta_path=meta_path,
            runtime=runtime,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            enable_thinking=enable_thinking,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )
    finally:
        _release_runtime(runtime)
