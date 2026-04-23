from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from xhquant.api import CacheTensor

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _runtime import load_token_embedding, parse_dtype, resolve_path
from qwen3_5_mtp_benchmark import (
    DTYPE_MAP as FLOAT_DTYPE_MAP,
    ForcedDecodeContext,
    install_forced_decode_patch,
    PreNormCapture,
    TEST_PROMPTS as MTP_TEST_PROMPTS,
    baseline_decode,
    build_mtp_head,
    generate_drafts,
    rollback_deltanet_states,
    trim_full_attention_kv,
)
from xh_model_zoo.xh_llm.models.qwen3_5 import Qwen3_5ONNXModel, Qwen3_5SpecDecodeONNXModel
from xh_model_zoo.xh_llm.models.qwen3_5._mtp_model import MTPModelXH2a
from xh_model_zoo.xh_llm.models.qwen3_5.qwen3_5_onnx_model import (
    _alloc_cache_inputs,
    _apply_presence_penalty,
    _apply_repetition_penalty,
    _as_cache_value,
    _build_linear_attn_mask,
    _clone_cache_value,
    _ensure_logits_shape,
    _is_kv_cache_name,
    _sample_next_token,
    _select_last_valid_logits,
)
from xh_model_zoo.xh_llm.models.qwen3_5_moe import load_moe_inference
from xh_model_zoo.xh_llm.models.qwen3_5_moe.qwen3_5_moe_spec_decode_inference import (
    Qwen3_5MoeSpecDecodeInference,
    _build_runtime_linear_attn_mask as build_moe_linear_attn_mask,
)

try:
    import dflash.qwen3_5_transformers_benchmark as dflash_benchmark
    from dflash.model import DFlashDraftModel
    from dflash.benchmark import load_and_process_dataset
    from dflash.qwen3_5_transformers_benchmark import (
        generate_baseline_qwen3_5 as dflash_baseline_decode,
        generate_dflash_qwen3_5_forced,
    )
except ImportError:
    dflash_benchmark = None
    DFlashDraftModel = None
    load_and_process_dataset = None
    dflash_baseline_decode = None
    generate_dflash_qwen3_5_forced = None


def make_messages(prompt: str, system_prompt: str = "") -> List[Dict[str, str]]:
    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def to_float_list(values: List[float]) -> List[float]:
    return [round(float(v), 4) for v in values]


def _select_greedy_token_with_penalties(
    logits: torch.Tensor,
    history_token_ids: List[int],
    repetition_penalty: float,
    presence_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = _apply_repetition_penalty(logits, history_token_ids, repetition_penalty)
    logits = _apply_presence_penalty(logits, history_token_ids, presence_penalty)
    next_token = torch.argmax(logits, dim=-1).reshape(1, 1)
    return next_token, logits


def encode_chat_prompt(tokenizer, prompt: str, enable_thinking: bool = False, system_prompt: str = "") -> torch.Tensor:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    text = tokenizer.apply_chat_template(
        make_messages(prompt, system_prompt=system_prompt),
        **kwargs,
    )
    return tokenizer([text], return_tensors="pt").input_ids


def thinking_mode_name(enable_thinking: bool | None) -> str:
    if enable_thinking is True:
        return "thinking"
    if enable_thinking is False:
        return "non-thinking"
    return "auto"


def resolve_enable_thinking(args: argparse.Namespace) -> bool | None:
    if getattr(args, "enable_thinking", False):
        return True
    if getattr(args, "disable_thinking", False):
        return False
    return None


def add_thinking_args(parser: argparse.ArgumentParser) -> None:
    thinking_group = parser.add_mutually_exclusive_group()
    thinking_group.add_argument("--enable-thinking", action="store_true")
    thinking_group.add_argument("--disable-thinking", action="store_true")


def resolve_dflash_block_size(draft_model_path: str, block_size: int | None) -> int:
    if block_size is not None:
        return int(block_size)
    config_path = resolve_path(REPO_ROOT, draft_model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"DFlash config not found: {config_path}")
    config = json.loads(config_path.read_text())
    if "block_size" not in config:
        raise KeyError(f"block_size missing from DFlash config: {config_path}")
    return int(config["block_size"])


def _get_hf_input_device(model) -> torch.device:
    embed_tokens = model.get_input_embeddings()
    if embed_tokens is not None and hasattr(embed_tokens, "weight"):
        return embed_tokens.weight.device
    return getattr(model, "device", torch.device("cpu"))


def _get_hf_output_device(model) -> torch.device:
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None and hasattr(lm_head, "weight"):
        return lm_head.weight.device
    return _get_hf_input_device(model)


def _is_multi_gpu_dispatched(model) -> bool:
    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return False

    cuda_devices: set[str] = set()
    for mapped_device in device_map.values():
        if isinstance(mapped_device, torch.device):
            mapped_name = str(mapped_device)
        elif isinstance(mapped_device, int):
            mapped_name = f"cuda:{mapped_device}"
        else:
            mapped_name = str(mapped_device)
            if mapped_name.isdigit():
                mapped_name = f"cuda:{mapped_name}"
        if mapped_name.startswith("cuda"):
            cuda_devices.add(mapped_name)
    return len(cuda_devices) > 1


def _stack_dflash_hidden(capture, device: torch.device) -> torch.Tensor:
    missing = [layer_id for layer_id in capture.layer_ids if layer_id not in capture.storage]
    if missing:
        raise RuntimeError(f"Missing captured hidden states for layers: {missing}")
    return torch.cat([capture.storage[layer_id].to(device) for layer_id in capture.layer_ids], dim=-1)


class _DFlashDecodeMetrics:
    def __init__(self, *, text: str, num_output_tokens: int, acceptance_lengths: List[int]):
        self.text = text
        self.num_output_tokens = num_output_tokens
        self.acceptance_lengths = acceptance_lengths


@torch.inference_mode()
def _generate_dflash_qwen3_5_forced_multigpu(
    draft_model,
    target,
    tokenizer,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
    enable_thinking: bool,
    block_size: int | None = None,
) -> _DFlashDecodeMetrics:
    if dflash_benchmark is None:
        raise RuntimeError("dflash benchmark helpers are not importable.")

    text = dflash_benchmark._apply_chat_template(tokenizer, prompt, enable_thinking)
    input_device = _get_hf_input_device(target)
    output_device = _get_hf_output_device(target)
    input_ids = tokenizer.encode(text, return_tensors="pt").to(input_device)
    block_size = block_size or draft_model.block_size
    mask_id = draft_model.mask_token_id
    capture = dflash_benchmark.LayerCapture(target.model.layers, draft_model.target_layer_ids)
    draft_cache = dflash_benchmark.DynamicCache()
    target_cache = dflash_benchmark.Qwen3_5DynamicCache(target.config)
    position_ids = torch.arange(
        input_ids.shape[1] + max_new_tokens + block_size + 4,
        device=input_device,
    ).unsqueeze(0)

    dflash_benchmark.install_forced_decode_patch()
    try:
        capture.clear()
        output = target(
            input_ids,
            position_ids=position_ids[:, : input_ids.shape[1]],
            past_key_values=target_cache,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )
        next_tok = dflash_benchmark.sample(output.logits, temperature).to(input_device)
        target_hidden = _stack_dflash_hidden(capture, input_device)

        generated: List[int] = []
        acceptance_lengths: List[int] = []
        committed_len = input_ids.shape[1]

        while len(generated) < max_new_tokens:
            round_committed_len = committed_len
            block = torch.full((1, block_size), mask_id, dtype=torch.long, device=input_device)
            block[:, 0] = next_tok[:, 0]
            noise_embedding = target.model.embed_tokens(block)
            draft_pos = position_ids[:, draft_cache.get_seq_length() : committed_len + block_size]
            draft_hidden = draft_model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=draft_pos,
                past_key_values=draft_cache,
                use_cache=True,
                is_causal=False,
            )
            draft_hidden = draft_hidden.to(output_device)
            draft_logits = target.lm_head(draft_hidden[:, 1 - block_size :, :])
            draft_tokens = dflash_benchmark.sample(draft_logits, temperature).to(input_device)
            draft_cache.crop(round_committed_len)

            verify_tokens = torch.cat([next_tok, draft_tokens], dim=1)
            with dflash_benchmark.ForcedDecodeContext():
                capture.clear()
                output = target(
                    verify_tokens,
                    position_ids=position_ids[:, committed_len : committed_len + block_size],
                    past_key_values=target_cache,
                    use_cache=True,
                    logits_to_keep=0,
                    output_hidden_states=True,
                )

            logits = output.logits
            accepted_drafts = 0
            for draft_idx in range(block_size - 1):
                posterior = dflash_benchmark.sample(logits[:, draft_idx : draft_idx + 1, :], temperature)
                if int(posterior[0, 0].item()) == int(draft_tokens[0, draft_idx].item()):
                    accepted_drafts += 1
                else:
                    break

            committed_tokens = [int(next_tok[0, 0].item())] + [
                int(draft_tokens[0, draft_idx].item()) for draft_idx in range(accepted_drafts)
            ]
            if len(generated) + len(committed_tokens) > max_new_tokens:
                committed_tokens = committed_tokens[: max_new_tokens - len(generated)]
            generated.extend(committed_tokens)
            committed_this_round = len(committed_tokens)
            acceptance_lengths.append(committed_this_round)
            target_hidden = _stack_dflash_hidden(capture, input_device)[:, :committed_this_round, :]

            if generated and generated[-1] == tokenizer.eos_token_id:
                break

            replacement = dflash_benchmark.sample(
                logits[:, accepted_drafts : accepted_drafts + 1, :],
                temperature,
            ).to(input_device)
            if accepted_drafts < block_size - 1:
                dflash_benchmark.rollback_deltanet_states(target_cache, rollback_idx=accepted_drafts)
                dflash_benchmark.trim_full_attention_kv(
                    target_cache,
                    n_trim=(block_size - 1) - accepted_drafts,
                )

            committed_len += committed_this_round
            next_tok = replacement

        return _DFlashDecodeMetrics(
            text=tokenizer.decode(generated, skip_special_tokens=True),
            num_output_tokens=len(generated),
            acceptance_lengths=acceptance_lengths,
        )
    finally:
        capture.close()
        dflash_benchmark.remove_forced_decode_patch()


def build_suite_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        return {
            "num_cases": 0,
            "text_match_cases": 0,
            "avg_baseline_decoder_calls": 0.0,
            "avg_spec_decoder_calls": 0.0,
            "avg_overall_acceptance_rate": 0.0,
            "avg_overall_acceptance_rate_text_matched": 0.0,
        }

    matched = [r for r in results if r.get("text_match")]
    return {
        "num_cases": len(results),
        "text_match_cases": len(matched),
        "avg_baseline_decoder_calls": round(
            sum(int(r["baseline"]["target_decoder_calls"]) for r in results) / len(results), 4
        ),
        "avg_spec_decoder_calls": round(
            sum(int(r["speculative"]["target_decoder_calls"]) for r in results) / len(results), 4
        ),
        "avg_overall_acceptance_rate": round(
            sum(float(r["speculative"]["overall_acceptance_rate"]) for r in results) / len(results), 4
        ),
        "avg_overall_acceptance_rate_text_matched": round(
            (
                sum(float(r["speculative"]["overall_acceptance_rate"]) for r in matched) / len(matched)
                if matched
                else 0.0
            ),
            4,
        ),
    }


def load_dense_runtime_from_meta(
    *,
    meta_path: str,
    dtype: str,
    device: str,
    exec_device: str,
    auto_offload_max_memory=None,
    prefill_auto_offload_max_memory=None,
    decode_auto_offload_max_memory=None,
) -> tuple[Qwen3_5ONNXModel, AutoTokenizer, dict]:
    meta_file = Path(meta_path).resolve()
    model_dir = meta_file.parent
    meta_info = json.loads(meta_file.read_text(encoding="utf-8"))

    prefill_onnx = resolve_path(model_dir, meta_info.get("prefill_onnx") or meta_info["prefill_onnx_file"])
    decode_onnx = resolve_path(model_dir, meta_info.get("decode_onnx") or meta_info["decode_onnx_file"])
    hf_model_config_dir = resolve_path(model_dir, meta_info["hf_config"])
    token_embedding_file = resolve_path(model_dir, meta_info["token_embedding_file"])

    tokenizer = AutoTokenizer.from_pretrained(str(hf_model_config_dir))
    token_embedding = load_token_embedding(token_embedding_file).to(dtype=parse_dtype(dtype))
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = 0

    spec_decode = meta_info.get("spec_decode")
    if spec_decode and spec_decode.get("mode") in ("mtp", "dflash"):
        draft_prefill = spec_decode.get("draft_prefill_onnx")
        draft_context = spec_decode.get("draft_context_onnx")
        draft_decode = spec_decode.get("draft_decode_onnx") or spec_decode.get("draft_onnx")
        if draft_decode is None:
            raise ValueError(f"{meta_path} missing draft decode path.")
        runtime: Qwen3_5ONNXModel = Qwen3_5SpecDecodeONNXModel(
            prefill={"onnx": str(prefill_onnx)},
            decode={"onnx": str(decode_onnx)},
            draft={
                "prefill": {"onnx": str(resolve_path(model_dir, draft_prefill))} if draft_prefill else None,
                "context": {"onnx": str(resolve_path(model_dir, draft_context))} if draft_context else None,
                "decode": {"onnx": str(resolve_path(model_dir, draft_decode))},
            },
            spec_decode_mode=spec_decode["mode"],
            block_size=int(spec_decode.get("block_size", 4)),
            hidden_output_name=spec_decode.get("hidden_output_name", "pre_norm_hidden"),
            max_context_tokens=meta_info.get("max_context_tokens"),
            auto_offload=True,
            auto_offload_max_memory=auto_offload_max_memory,
            prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
            decode_auto_offload_max_memory=decode_auto_offload_max_memory,
            pad_token_id=pad_token_id,
        )
    else:
        runtime = Qwen3_5ONNXModel(
            prefill={"onnx": str(prefill_onnx)},
            decode={"onnx": str(decode_onnx)},
            max_context_tokens=meta_info.get("max_context_tokens"),
            auto_offload=True,
            auto_offload_max_memory=auto_offload_max_memory,
            prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
            decode_auto_offload_max_memory=decode_auto_offload_max_memory,
            pad_token_id=pad_token_id,
        )

    runtime.set_input_embeddings(token_embedding)
    runtime.to(torch.device(device))
    runtime.set_exec_device(torch.device(exec_device))
    runtime.to(parse_dtype(dtype))
    return runtime, tokenizer, meta_info


def finalise_spec_metrics(
    *,
    prompt: str,
    draft_mode: str,
    model_name: str,
    baseline_text: str,
    baseline_decoder_calls: int,
    baseline_prefill_calls: int,
    baseline_output_tokens: int | None = None,
    spec_text: str,
    spec_output_tokens: int,
    target_prefill_calls: int,
    target_decoder_calls: int,
    accepted_drafts_per_round: List[int],
    draft_capacity: int,
    extra_counts: Dict[str, int],
) -> Dict[str, Any]:
    total_accepted = sum(accepted_drafts_per_round)
    total_rounds = len(accepted_drafts_per_round)
    total_drafts = total_rounds * draft_capacity
    return {
        "prompt": prompt,
        "model": model_name,
        "draft_mode": draft_mode,
        "verify_mode": "forced",
        "baseline": {
            "text": baseline_text,
            "output_tokens": (
                int(baseline_output_tokens)
                if baseline_output_tokens is not None
                else max(baseline_decoder_calls + 1, 0)
            ),
            "target_prefill_calls": baseline_prefill_calls,
            "target_decoder_calls": baseline_decoder_calls,
        },
        "speculative": {
            "text": spec_text,
            "output_tokens": spec_output_tokens,
            "target_prefill_calls": target_prefill_calls,
            "target_decoder_calls": target_decoder_calls,
            **extra_counts,
            "num_rounds": total_rounds,
            "draft_capacity_per_round": draft_capacity,
            "accepted_drafts_per_round": accepted_drafts_per_round,
            "round_acceptance_rates": to_float_list(
                [(v / draft_capacity) if draft_capacity else 0.0 for v in accepted_drafts_per_round]
            ),
            "accepted_drafts_total": total_accepted,
            "draft_tokens_total": total_drafts,
            "overall_acceptance_rate": round((total_accepted / total_drafts) if total_drafts else 0.0, 4),
            "avg_accepted_per_round": round((total_accepted / total_rounds) if total_rounds else 0.0, 4),
        },
        "text_match": baseline_text == spec_text,
    }


def release_moe_runtime(runtime: Qwen3_5MoeSpecDecodeInference) -> None:
    runtime.prefill_session = None
    runtime.decode_session = None
    if hasattr(runtime, "_draft_prefill_session"):
        runtime._draft_prefill_session = None
    if hasattr(runtime, "_draft_context_session"):
        runtime._draft_context_session = None
    if hasattr(runtime, "_draft_decode_session"):
        runtime._draft_decode_session = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def release_dense_runtime(runtime: Qwen3_5ONNXModel) -> None:
    for attr in (
        "prefill_session",
        "decode_session",
        "draft_prefill_session",
        "draft_context_session",
        "draft_decode_session",
    ):
        if hasattr(runtime, attr):
            setattr(runtime, attr, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_dense_spec_runtime(runtime: Qwen3_5SpecDecodeONNXModel) -> None:
    runtime._mtp_cache_state = None
    runtime._dflash_cache_state = None


class CompatibleMTPHead:
    def __init__(self, model: MTPModelXH2a, token_embedding: torch.nn.Module):
        self.model = model
        self.embed_tokens = token_embedding
        self._mtp_device = next(model.parameters()).device

    def _embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        embed_device = self.embed_tokens.weight.device
        embeds = self.embed_tokens(token_ids.to(embed_device))
        return embeds.to(self._mtp_device)

    @staticmethod
    def _wrap_cache(cache_value, *, device: torch.device, dtype: torch.dtype):
        if cache_value is None:
            return CacheTensor(torch.empty(0, device=device, dtype=dtype))
        if isinstance(cache_value, CacheTensor):
            return cache_value
        return CacheTensor(cache_value)

    @staticmethod
    def _unwrap_cache(cache_value):
        if cache_value is None:
            return None
        if isinstance(cache_value, CacheTensor):
            return cache_value.data
        return cache_value

    def forward_step(
        self,
        main_hidden: torch.Tensor,
        next_tok: torch.Tensor,
        *,
        position: int,
        kv_cache: dict,
        return_hidden: bool = False,
    ):
        next_token_embedding = self._embed(next_tok)
        logits, hidden, present_k, present_v = self.model(
            next_token_embedding=next_token_embedding,
            pre_norm_hidden=main_hidden.to(self._mtp_device),
            past_seq_length=torch.tensor([position], dtype=torch.int32, device=self._mtp_device),
            current_input_length=torch.tensor([1], dtype=torch.int32, device=self._mtp_device),
            past_key_cache=self._wrap_cache(
                kv_cache.get("key"), device=self._mtp_device, dtype=main_hidden.dtype
            ),
            past_value_cache=self._wrap_cache(
                kv_cache.get("value"), device=self._mtp_device, dtype=main_hidden.dtype
            ),
        )
        if present_k is not None:
            kv_cache["key"] = self._unwrap_cache(present_k)
        if present_v is not None:
            kv_cache["value"] = self._unwrap_cache(present_v)
        if return_hidden:
            return logits, hidden
        return logits

    def forward_batch(
        self,
        main_hidden: torch.Tensor,
        next_tok: torch.Tensor,
        *,
        positions: torch.Tensor,
        kv_cache: dict,
    ):
        seq_len = int(main_hidden.shape[1])
        next_token_embedding = self._embed(next_tok)
        logits, hidden, present_k, present_v = self.model(
            next_token_embedding=next_token_embedding,
            pre_norm_hidden=main_hidden.to(self._mtp_device),
            past_seq_length=torch.tensor([int(positions[0].item())], dtype=torch.int32, device=self._mtp_device),
            current_input_length=torch.tensor([seq_len], dtype=torch.int32, device=self._mtp_device),
            past_key_cache=self._wrap_cache(
                kv_cache.get("key"), device=self._mtp_device, dtype=main_hidden.dtype
            ),
            past_value_cache=self._wrap_cache(
                kv_cache.get("value"), device=self._mtp_device, dtype=main_hidden.dtype
            ),
        )
        if present_k is not None:
            kv_cache["key"] = self._unwrap_cache(present_k)
        if present_v is not None:
            kv_cache["value"] = self._unwrap_cache(present_v)
        return logits, hidden


def build_compatible_mtp_head(model, model_path: str, dtype: str):
    try:
        return build_mtp_head(model, model_path, dtype)
    except RuntimeError:
        target_dtype = FLOAT_DTYPE_MAP[dtype] if dtype != "auto" else model.lm_head.weight.dtype
        head = MTPModelXH2a.from_pretrained(
            model_path,
            dtype=target_dtype,
            input_sequence_length=1,
            max_pe_length=262144,
            use_cache=True,
        )
        target_dev = model.lm_head.weight.device
        head = head.to(device=target_dev, dtype=target_dtype)
        text_model = getattr(model, "model", model)
        return CompatibleMTPHead(head, text_model.embed_tokens)


@torch.no_grad()
def run_float_mtp_metrics(
    *,
    model_path: str,
    prompt: str,
    max_new_tokens: int,
    num_draft_tokens: int,
    dtype: str,
    system_prompt: str,
    enable_thinking: bool | None = None,
) -> Dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=FLOAT_DTYPE_MAP[dtype] if dtype != "auto" else "auto",
        device_map="auto",
    ).eval()
    mtp_head = build_compatible_mtp_head(model, model_path, dtype)
    install_forced_decode_patch()

    device = model.device
    eos = tokenizer.eos_token_id
    prompt_kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if enable_thinking is not None:
        prompt_kwargs["enable_thinking"] = enable_thinking
    prompt_text = tokenizer.apply_chat_template(
        make_messages(prompt, system_prompt=system_prompt),
        **prompt_kwargs,
    )
    inputs = tokenizer([prompt_text], return_tensors="pt").to(device)
    prompt_ids = inputs.input_ids
    prompt_len = prompt_ids.shape[1]

    outputs = model(**inputs, use_cache=True)
    baseline_past_kv = outputs.past_key_values
    baseline_next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
    baseline_tokens = [baseline_next_tok.item()]
    for _ in range(max_new_tokens - 1):
        if baseline_tokens[-1] == eos:
            break
        kv_len = baseline_past_kv.get_seq_length()
        mask = torch.ones(1, kv_len + 1, device=device, dtype=torch.long)
        outputs = model(
            input_ids=baseline_next_tok,
            attention_mask=mask,
            past_key_values=baseline_past_kv,
            use_cache=True,
        )
        baseline_past_kv = outputs.past_key_values
        baseline_next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
        baseline_tokens.append(baseline_next_tok.item())
    baseline = {
        "text": tokenizer.decode(baseline_tokens, skip_special_tokens=True),
        "num_tokens": len(baseline_tokens),
    }
    baseline_decoder_calls = max(int(baseline["num_tokens"]) - 1, 0)

    cap = PreNormCapture(model)
    cap.reset()
    outputs = model(**inputs, use_cache=True)
    past_kv = outputs.past_key_values
    next_tok = outputs.logits[:, -1:, :].argmax(dim=-1)
    prefill_hidden = cap.hidden_states[0]
    committed_len = prompt_len

    mtp_prefill_calls = 0
    mtp_decode_calls = 0
    mtp_kv: dict = {}
    if prompt_len >= 2:
        mtp_head.forward_batch(
            prefill_hidden[:, :-1, :],
            prompt_ids[:, 1:],
            positions=torch.arange(prompt_len - 1, device=mtp_head._mtp_device),
            kv_cache=mtp_kv,
        )
        mtp_prefill_calls += 1

    drafts, _ = generate_drafts(
        mtp_head,
        prefill_hidden[:, -1:, :],
        next_tok,
        num_draft_tokens,
        position=committed_len - 1,
        mtp_kv=mtp_kv,
    )
    mtp_decode_calls += num_draft_tokens

    tokens: List[int] = []
    accepted_drafts_per_round: List[int] = []
    num_main_fwd = 0

    while len(tokens) < max_new_tokens:
        num_main_fwd += 1

        all_toks = torch.cat([next_tok] + [d.to(device) for d in drafts], dim=1)
        kv_len = past_kv.get_seq_length()
        mask = torch.ones(1, kv_len + num_draft_tokens + 1, device=device, dtype=torch.long)

        with ForcedDecodeContext():
            cap.reset()
            out = model(
                input_ids=all_toks,
                attention_mask=mask,
                past_key_values=past_kv,
                use_cache=True,
            )
        past_kv = out.past_key_values
        hidden_all = cap.hidden_states[0]

        accepted_count = 0
        for idx in range(num_draft_tokens):
            verified = out.logits[:, idx, :].argmax(dim=-1, keepdim=True)
            if verified.item() != drafts[idx].item():
                break
            accepted_count += 1
        accepted_drafts_per_round.append(accepted_count)

        if accepted_count == num_draft_tokens:
            tokens.append(next_tok.item())
            for draft in drafts:
                tokens.append(draft.item())
            old_committed = committed_len
            committed_len += num_draft_tokens + 1

            if any(d.item() == eos for d in drafts) or len(tokens) >= max_new_tokens:
                break

            for idx in range(num_draft_tokens):
                mtp_head.forward_step(
                    hidden_all[:, idx : idx + 1, :],
                    drafts[idx],
                    position=old_committed + idx,
                    kv_cache=mtp_kv,
                )
            mtp_decode_calls += num_draft_tokens

            bonus_tok = out.logits[:, num_draft_tokens, :].argmax(dim=-1, keepdim=True)
            next_tok = bonus_tok
            drafts, _ = generate_drafts(
                mtp_head,
                hidden_all[:, num_draft_tokens : num_draft_tokens + 1, :],
                bonus_tok,
                num_draft_tokens,
                position=committed_len - 1,
                mtp_kv=mtp_kv,
            )
            mtp_decode_calls += num_draft_tokens
        else:
            tokens.append(next_tok.item())
            for idx in range(accepted_count):
                tokens.append(drafts[idx].item())
            committed_len += accepted_count + 1

            if next_tok.item() == eos or len(tokens) >= max_new_tokens:
                break
            if any(drafts[idx].item() == eos for idx in range(accepted_count)):
                break

            rollback_deltanet_states(past_kv, rollback_idx=accepted_count)
            trim_full_attention_kv(past_kv, n_trim=num_draft_tokens - accepted_count)

            for idx in range(accepted_count):
                mtp_head.forward_step(
                    hidden_all[:, idx : idx + 1, :],
                    drafts[idx],
                    position=committed_len - accepted_count - 1 + idx,
                    kv_cache=mtp_kv,
                )
            mtp_decode_calls += accepted_count

            replacement = out.logits[:, accepted_count, :].argmax(dim=-1, keepdim=True)
            next_tok = replacement
            drafts, _ = generate_drafts(
                mtp_head,
                hidden_all[:, accepted_count : accepted_count + 1, :],
                replacement,
                num_draft_tokens,
                position=committed_len - 1,
                mtp_kv=mtp_kv,
            )
            mtp_decode_calls += num_draft_tokens

    cap.remove()
    tokens = tokens[:max_new_tokens]
    spec_text = tokenizer.decode(tokens, skip_special_tokens=True)
    return finalise_spec_metrics(
        prompt=prompt,
        draft_mode="mtp",
        model_name=model_path,
        baseline_text=baseline["text"],
        baseline_decoder_calls=baseline_decoder_calls,
        baseline_prefill_calls=1,
        spec_text=spec_text,
        spec_output_tokens=len(tokens),
        target_prefill_calls=1,
        target_decoder_calls=num_main_fwd,
        accepted_drafts_per_round=accepted_drafts_per_round,
        draft_capacity=num_draft_tokens,
        extra_counts={
            "mtp_prefill_calls": mtp_prefill_calls,
            "mtp_decode_calls": mtp_decode_calls,
        },
    )


@torch.no_grad()
def run_float_dflash_metrics(
    *,
    model_path: str,
    draft_model_path: str,
    prompt: str,
    max_new_tokens: int,
    dtype: str,
    block_size: int,
    enable_thinking: bool = False,
) -> Dict[str, Any]:
    if (
        DFlashDraftModel is None
        or dflash_baseline_decode is None
        or generate_dflash_qwen3_5_forced is None
        or dflash_benchmark is None
    ):
        raise RuntimeError("dflash package is not importable. Set PYTHONPATH to /data01/home/yujy/work/dflash.")

    torch_dtype = FLOAT_DTYPE_MAP[dtype] if dtype != "auto" else torch.bfloat16
    target = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    ).eval()
    draft = DFlashDraftModel.from_pretrained(
        draft_model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    baseline = dflash_baseline_decode(
        target,
        tokenizer,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        enable_thinking=enable_thinking,
    )
    baseline_decoder_calls = max(int(baseline.num_output_tokens) - 1, 0)

    draft_decode_calls = 0
    original_forward = draft.forward
    use_multigpu_compatible_path = _is_multi_gpu_dispatched(target) or _is_multi_gpu_dispatched(draft)

    def counted_forward(*args, **kwargs):
        nonlocal draft_decode_calls
        draft_decode_calls += 1
        return original_forward(*args, **kwargs)

    draft.forward = counted_forward  # type: ignore[assignment]
    try:
        if use_multigpu_compatible_path:
            spec = _generate_dflash_qwen3_5_forced_multigpu(
                draft,
                target,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                enable_thinking=enable_thinking,
                block_size=block_size,
            )
        else:
            spec = generate_dflash_qwen3_5_forced(
                draft,
                target,
                tokenizer,
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                enable_thinking=enable_thinking,
                block_size=block_size,
            )
    finally:
        draft.forward = original_forward  # type: ignore[assignment]

    accepted_drafts_per_round = [max(int(v) - 1, 0) for v in spec.acceptance_lengths]
    return finalise_spec_metrics(
        prompt=prompt,
        draft_mode="dflash",
        model_name=model_path,
        baseline_text=baseline.text,
        baseline_decoder_calls=baseline_decoder_calls,
        baseline_prefill_calls=1,
        spec_text=spec.text,
        spec_output_tokens=int(spec.num_output_tokens),
        target_prefill_calls=1,
        target_decoder_calls=len(spec.acceptance_lengths),
        accepted_drafts_per_round=accepted_drafts_per_round,
        draft_capacity=max(block_size - 1, 0),
        extra_counts={
            "dflash_prefill_calls": 0,
            "dflash_decode_calls": draft_decode_calls,
        },
    )


def run_float_mtp_suite_metrics(
    *,
    model_path: str,
    max_new_tokens: int,
    num_draft_tokens: int,
    dtype: str,
    system_prompt: str,
    enable_thinking: bool | None,
) -> Dict[str, Any]:
    case_results = []
    for case in MTP_TEST_PROMPTS:
        result = run_float_mtp_metrics(
            model_path=model_path,
            prompt=case["prompt"],
            max_new_tokens=max_new_tokens,
            num_draft_tokens=num_draft_tokens,
            dtype=dtype,
            system_prompt=system_prompt,
            enable_thinking=enable_thinking,
        )
        result["case_name"] = case["name"]
        case_results.append(result)

    return {
        "suite": "mtp_test_prompts",
        "model": model_path,
        "draft_mode": "mtp",
        "verify_mode": "forced",
        "thinking_mode": thinking_mode_name(enable_thinking),
        "num_draft_tokens": num_draft_tokens,
        "max_new_tokens": max_new_tokens,
        "cases": case_results,
        "summary": build_suite_summary(case_results),
    }


def run_float_dflash_suite_metrics(
    *,
    model_path: str,
    draft_model_path: str,
    dataset: str,
    max_samples: int,
    max_new_tokens: int,
    dtype: str,
    block_size: int,
    enable_thinking: bool,
) -> Dict[str, Any]:
    if load_and_process_dataset is None:
        raise RuntimeError("dflash package is not importable. Set PYTHONPATH to /data01/home/yujy/work/dflash.")

    rows = load_and_process_dataset(dataset)
    prompts = [rows[i]["turns"][0] for i in range(min(max_samples, len(rows)))]
    case_results = []
    for idx, prompt in enumerate(prompts):
        result = run_float_dflash_metrics(
            model_path=model_path,
            draft_model_path=draft_model_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            dtype=dtype,
            block_size=block_size,
            enable_thinking=enable_thinking,
        )
        result["case_name"] = f"{dataset}:{idx}"
        case_results.append(result)

    return {
        "suite": dataset,
        "model": model_path,
        "draft_model": draft_model_path,
        "draft_mode": "dflash",
        "verify_mode": "forced",
        "thinking_mode": thinking_mode_name(enable_thinking),
        "block_size": block_size,
        "max_new_tokens": max_new_tokens,
        "cases": case_results,
        "summary": build_suite_summary(case_results),
    }


@torch.no_grad()
def run_dense_baseline_from_meta(
    *,
    meta_path: str,
    prompt: str,
    max_new_tokens: int,
    dtype: str,
    device: str,
    exec_device: str,
    auto_offload_max_memory=None,
    prefill_auto_offload_max_memory=None,
    decode_auto_offload_max_memory=None,
) -> Dict[str, Any]:
    runtime, tokenizer, _ = load_dense_runtime_from_meta(
        meta_path=meta_path,
        dtype=dtype,
        device=device,
        exec_device=exec_device,
        auto_offload_max_memory=auto_offload_max_memory,
        prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
        decode_auto_offload_max_memory=decode_auto_offload_max_memory,
    )
    prefill_calls = 0
    decode_calls = 0
    original_run_hmonnx = runtime._run_hmonnx

    def counted_run_hmonnx(session, input_feed):
        nonlocal prefill_calls, decode_calls
        if runtime.prefill_session is not None and session is runtime.prefill_session:
            prefill_calls += 1
        elif runtime.decode_session is not None and session is runtime.decode_session:
            decode_calls += 1
        return original_run_hmonnx(session, input_feed)

    runtime._run_hmonnx = counted_run_hmonnx  # type: ignore[assignment]
    try:
        text = runtime.chat(
            prompt=prompt,
            tokenizer=tokenizer,
            history=None,
            system_prompt="",
            max_new_tokens=max_new_tokens,
            enable_thinking=False,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            top_k=1,
            repetition_penalty=1.0,
            presence_penalty=0.0,
            stream_output=False,
        )
    finally:
        runtime._run_hmonnx = original_run_hmonnx  # type: ignore[assignment]
        release_dense_runtime(runtime)

    output_tokens = len(tokenizer.encode(text, add_special_tokens=False))
    return {
        "text": text,
        "target_prefill_calls": prefill_calls,
        "target_decoder_calls": decode_calls,
        "output_tokens": output_tokens,
    }


@torch.no_grad()
def run_dense_target_baseline_from_spec(
    runtime: Qwen3_5SpecDecodeONNXModel,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool = False,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
) -> Dict[str, Any]:
    input_ids = encode_chat_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
    if runtime.max_context_tokens is not None and input_ids.shape[1] > runtime.max_context_tokens:
        input_ids = input_ids[:, -runtime.max_context_tokens :]

    total_prompt_len = int(input_ids.shape[1])
    runtime._ensure_prefill_session()
    prefill_cache_state = _alloc_cache_inputs(runtime.prefill_session, runtime.device)
    prefill_chunk_len = int(runtime._prefill_inputs_info.shape[1])

    prefill_calls = 0
    last_prefill_logits = None
    past_seq_len = 0
    for start in range(0, total_prompt_len, prefill_chunk_len):
        end = min(start + prefill_chunk_len, total_prompt_len)
        chunk_ids = input_ids[:, start:end]
        valid_len = int(chunk_ids.shape[1])
        prefill_calls += 1
        feed = runtime._build_prefill_feed(chunk_ids, valid_len, past_seq_len, prefill_cache_state)
        _, output_map = runtime._run_hmonnx(runtime.prefill_session, feed)
        prefill_logits = runtime._extract_logits(output_map)
        last_prefill_logits = _select_last_valid_logits(prefill_logits, valid_len)
        runtime._update_linear_cache(prefill_cache_state, output_map)
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
    token_val = int(next_token_id[0, 0].item())
    if tokenizer.eos_token_id is not None and token_val == tokenizer.eos_token_id:
        return {
            "text": "",
            "token_ids": [],
            "target_prefill_calls": prefill_calls,
            "target_decoder_calls": 0,
            "output_tokens": 0,
        }

    runtime._ensure_decode_session()
    decode_cache_state = _alloc_cache_inputs(runtime.decode_session, runtime.device)
    for name in decode_cache_state:
        if name in prefill_cache_state and (
            _is_kv_cache_name(name) or name.startswith(("past_conv_cache_", "past_recurrent_state_"))
        ):
            decode_cache_state[name] = prefill_cache_state[name]

    generated_ids = [token_val]
    history_token_ids.append(token_val)
    current_token = next_token_id.to(runtime.device)
    decoder_calls = 0

    for _ in range(max_new_tokens - 1):
        decoder_calls += 1
        decode_feed = runtime._build_decode_feed(current_token, past_seq_len, decode_cache_state, current_input_length=1)
        _, decode_output_map = runtime._run_hmonnx(runtime.decode_session, decode_feed)
        decode_logits = runtime._extract_logits(decode_output_map)
        decode_logits = _select_last_valid_logits(decode_logits, 1)
        next_token_id, _ = _select_greedy_token_with_penalties(
            decode_logits,
            history_token_ids,
            repetition_penalty,
            presence_penalty,
        )
        runtime._update_linear_cache(decode_cache_state, decode_output_map)

        token_val = int(next_token_id[0, 0].item())
        if tokenizer.eos_token_id is not None and token_val == tokenizer.eos_token_id:
            break
        generated_ids.append(token_val)
        history_token_ids.append(token_val)
        current_token = next_token_id.to(runtime.device)
        past_seq_len += 1

    return {
        "text": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "token_ids": generated_ids,
        "target_prefill_calls": prefill_calls,
        "target_decoder_calls": decoder_calls,
        "output_tokens": len(generated_ids),
    }


@torch.no_grad()
def run_dense_spec_metrics(
    *,
    meta_path: str,
    baseline_meta_path: str | None,
    prompt: str,
    max_new_tokens: int,
    dtype: str,
    device: str,
    exec_device: str,
    enable_thinking: bool = False,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
    auto_offload_max_memory=None,
    prefill_auto_offload_max_memory=None,
    decode_auto_offload_max_memory=None,
) -> Dict[str, Any]:
    if baseline_meta_path:
        baseline = run_dense_baseline_from_meta(
            meta_path=baseline_meta_path,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            dtype=dtype,
            device=device,
            exec_device=exec_device,
            auto_offload_max_memory=auto_offload_max_memory,
            prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
            decode_auto_offload_max_memory=decode_auto_offload_max_memory,
        )
    else:
        runtime_base, tokenizer, meta_info = load_dense_runtime_from_meta(
            meta_path=meta_path,
            dtype=dtype,
            device=device,
            exec_device=exec_device,
            auto_offload_max_memory=auto_offload_max_memory,
            prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
            decode_auto_offload_max_memory=decode_auto_offload_max_memory,
        )
        if not isinstance(runtime_base, Qwen3_5SpecDecodeONNXModel):
            raise TypeError(f"{meta_path} is not a dense spec-decode runtime.")
        baseline = run_dense_target_baseline_from_spec(
            runtime_base,
            tokenizer,
            prompt,
            max_new_tokens,
            enable_thinking=enable_thinking,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )
        release_dense_runtime(runtime_base)
        del runtime_base

    runtime, tokenizer, meta_info = load_dense_runtime_from_meta(
        meta_path=meta_path,
        dtype=dtype,
        device=device,
        exec_device=exec_device,
        auto_offload_max_memory=auto_offload_max_memory,
        prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
        decode_auto_offload_max_memory=decode_auto_offload_max_memory,
    )
    if not isinstance(runtime, Qwen3_5SpecDecodeONNXModel):
        raise TypeError(f"{meta_path} is not a dense spec-decode runtime.")

    draft_counts = {"mtp_prefill_calls": 0, "mtp_decode_calls": 0, "dflash_context_calls": 0, "dflash_decode_calls": 0}
    original_run_draft_session = runtime._run_draft_session

    def counted_run_draft_session(session, input_feed):
        if runtime.spec_decode_mode == "mtp":
            if runtime.draft_prefill_session is not None and session is runtime.draft_prefill_session:
                draft_counts["mtp_prefill_calls"] += 1
            elif runtime.draft_decode_session is not None and session is runtime.draft_decode_session:
                draft_counts["mtp_decode_calls"] += 1
        else:
            if runtime.draft_context_session is not None and session is runtime.draft_context_session:
                draft_counts["dflash_context_calls"] += 1
            elif runtime.draft_decode_session is not None and session is runtime.draft_decode_session:
                draft_counts["dflash_decode_calls"] += 1
        return original_run_draft_session(session, input_feed)

    runtime._run_draft_session = counted_run_draft_session  # type: ignore[assignment]
    try:
        input_ids = encode_chat_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
        if runtime.max_context_tokens is not None and input_ids.shape[1] > runtime.max_context_tokens:
            input_ids = input_ids[:, -runtime.max_context_tokens :]

        total_prompt_len = int(input_ids.shape[1])
        runtime._ensure_prefill_session()
        prefill_cache_state = _alloc_cache_inputs(runtime.prefill_session, runtime.device)
        prefill_chunk_len = int(runtime._prefill_inputs_info.shape[1])

        last_prefill_logits = None
        last_hidden = None
        past_seq_len = 0
        target_prefill_calls = 0
        mtp_prefill_seq_len = 0
        mtp_pending_hidden = None

        for start in range(0, total_prompt_len, prefill_chunk_len):
            end = min(start + prefill_chunk_len, total_prompt_len)
            chunk_ids = input_ids[:, start:end]
            valid_len = int(chunk_ids.shape[1])
            target_prefill_calls += 1
            prefill_feed = runtime._build_prefill_feed(chunk_ids, valid_len, past_seq_len, prefill_cache_state)
            _, prefill_output_map = runtime._run_hmonnx(runtime.prefill_session, prefill_feed)
            prefill_logits = runtime._extract_logits(prefill_output_map)
            last_prefill_logits = _select_last_valid_logits(prefill_logits, valid_len)
            prefill_hidden_all = runtime._extract_hidden(prefill_output_map)
            if prefill_hidden_all is not None:
                prefill_hidden_all = prefill_hidden_all[:, :valid_len, :]
                last_hidden = runtime._select_hidden_step(prefill_hidden_all, valid_len - 1)
                if runtime.spec_decode_mode == "dflash":
                    runtime._append_dflash_context(prefill_hidden_all, past_seq_len)
                else:
                    hidden_parts = []
                    token_parts = []
                    if mtp_pending_hidden is not None:
                        hidden_parts.append(mtp_pending_hidden)
                        token_parts.append(chunk_ids[:, :1])
                    if valid_len > 1:
                        hidden_parts.append(prefill_hidden_all[:, : valid_len - 1, :])
                        token_parts.append(chunk_ids[:, 1:valid_len])
                    if hidden_parts:
                        mtp_hidden = torch.cat(hidden_parts, dim=1)
                        mtp_tokens = torch.cat(token_parts, dim=1)
                        runtime._prefill_mtp_chunk(mtp_hidden, mtp_tokens, mtp_prefill_seq_len)
                        mtp_prefill_seq_len += int(mtp_hidden.shape[1])
                    mtp_pending_hidden = prefill_hidden_all[:, valid_len - 1 : valid_len, :]

            runtime._update_linear_cache(prefill_cache_state, prefill_output_map)
            past_seq_len += valid_len

        if last_prefill_logits is None:
            raise RuntimeError("Prefill did not produce logits.")

        history_token_ids = input_ids[0].tolist()
        next_token_id, _ = _select_greedy_token_with_penalties(
            last_prefill_logits,
            history_token_ids,
            repetition_penalty,
            presence_penalty,
        )

        runtime._ensure_decode_session()
        decode_cache_state = _alloc_cache_inputs(runtime.decode_session, runtime.device)
        for name in decode_cache_state:
            if name in prefill_cache_state and (
                _is_kv_cache_name(name) or name.startswith(("past_conv_cache_", "past_recurrent_state_"))
            ):
                decode_cache_state[name] = prefill_cache_state[name]

        eos_token_id = tokenizer.eos_token_id
        generated_ids: List[int] = []
        token_val = int(next_token_id[0][0].item())
        if eos_token_id is None or token_val != eos_token_id:
            generated_ids.append(token_val)
            history_token_ids.append(token_val)

        current_token = next_token_id.to(runtime.device)
        mtp_past_seq_len = mtp_prefill_seq_len
        num_drafts = runtime.block_size - 1 if runtime.spec_decode_mode == "dflash" else runtime.block_size
        total_rounds = 0
        total_accepted_tokens = 0
        accepted_drafts_per_round: List[int] = []
        draft_capacity = num_drafts

        while len(generated_ids) < max_new_tokens:
            total_rounds += 1
            if runtime.spec_decode_mode == "dflash":
                draft_tokens = runtime._run_draft_dflash(current_token, past_seq_len)
            else:
                mtp_cache_snapshot = {
                    name: _clone_cache_value(tensor)
                    for name, tensor in runtime._ensure_mtp_cache_state().items()
                }
                draft_tokens = runtime._run_draft_mtp(current_token, last_hidden, mtp_past_seq_len, num_drafts)

            verify_tokens = [current_token] + draft_tokens
            verify_input_ids = torch.cat(verify_tokens, dim=1)
            initial_seq_len = past_seq_len
            decode_feed = runtime._build_decode_feed(
                verify_input_ids,
                past_seq_len,
                decode_cache_state,
                current_input_length=verify_input_ids.shape[1],
            )
            _, decode_output_map = runtime._run_hmonnx(runtime.decode_session, decode_feed)
            verify_logits = _ensure_logits_shape(runtime._extract_logits(decode_output_map))
            verify_hidden_all = runtime._extract_hidden(decode_output_map)

            accepted_count = 0
            running_history = list(history_token_ids)
            current_token = None
            for idx, draft_tok in enumerate(draft_tokens):
                predicted, _ = _select_greedy_token_with_penalties(
                    verify_logits[:, idx : idx + 1, :],
                    running_history,
                    repetition_penalty,
                    presence_penalty,
                )
                if predicted[0, 0].item() != draft_tok[0, 0].item():
                    current_token = predicted.to(runtime.device)
                    break
                accepted_count += 1
                running_history.append(int(draft_tok[0, 0].item()))

            accepted_drafts_per_round.append(accepted_count)
            total_accepted_tokens += accepted_count
            accepted_steps = accepted_count + 1
            runtime._apply_verify_linear_cache_outputs(
                decode_cache_state,
                decode_output_map,
                accepted_steps=accepted_steps,
            )
            past_seq_len = initial_seq_len + accepted_steps

            eos_hit = False
            for idx in range(accepted_count):
                token_val = int(draft_tokens[idx][0, 0].item())
                generated_ids.append(token_val)
                history_token_ids.append(token_val)
                if eos_token_id is not None and token_val == eos_token_id:
                    eos_hit = True
                    break
                if len(generated_ids) >= max_new_tokens:
                    break
            if eos_hit or len(generated_ids) >= max_new_tokens:
                break

            if accepted_count == len(draft_tokens):
                current_token, _ = _select_greedy_token_with_penalties(
                    verify_logits[:, -1:, :],
                    running_history,
                    repetition_penalty,
                    presence_penalty,
                )
                current_token = current_token.to(runtime.device)

            last_hidden = runtime._select_hidden_step(verify_hidden_all, accepted_count)
            accepted_hidden = (
                verify_hidden_all[:, :accepted_steps, :] if verify_hidden_all is not None else None
            )
            if runtime.spec_decode_mode == "dflash":
                if accepted_hidden is not None:
                    runtime._append_dflash_context(accepted_hidden, initial_seq_len)
            else:
                runtime._mtp_cache_state = {
                    name: _as_cache_value(tensor, tensor.clone())
                    for name, tensor in mtp_cache_snapshot.items()
                }
                if accepted_hidden is not None:
                    mtp_cache_state = runtime._ensure_mtp_cache_state()
                    mtp_initial_seq_len = mtp_past_seq_len
                    for step_idx in range(accepted_steps):
                        next_token_for_cache = (
                            verify_tokens[step_idx + 1]
                            if step_idx < accepted_steps - 1
                            else current_token
                        )
                        draft_output_map = runtime._run_draft_session(
                            runtime.draft_decode_session,
                            runtime._build_mtp_decode_feed(
                                next_token_for_cache,
                                accepted_hidden[:, step_idx : step_idx + 1, :],
                                mtp_initial_seq_len + step_idx,
                                mtp_cache_state,
                            ),
                        )
                        for name in list(mtp_cache_state.keys()):
                            present_name = name.replace("past_", "present_", 1)
                            if present_name in draft_output_map:
                                mtp_cache_state[name] = _as_cache_value(
                                    mtp_cache_state[name],
                                    draft_output_map[present_name],
                                )
                    mtp_past_seq_len = mtp_initial_seq_len + accepted_steps

            token_val = int(current_token[0, 0].item())
            if eos_token_id is not None and token_val == eos_token_id:
                break
            generated_ids.append(token_val)
            history_token_ids.append(token_val)
            if len(generated_ids) >= max_new_tokens:
                break

        extra_counts = (
            {
                "mtp_prefill_calls": draft_counts["mtp_prefill_calls"],
                "mtp_decode_calls": draft_counts["mtp_decode_calls"],
            }
            if runtime.spec_decode_mode == "mtp"
            else {
                "dflash_prefill_calls": draft_counts["dflash_context_calls"],
                "dflash_decode_calls": draft_counts["dflash_decode_calls"],
            }
        )
        return finalise_spec_metrics(
            prompt=prompt,
            draft_mode=runtime.spec_decode_mode,
            model_name=meta_info.get("model_name", meta_path),
            baseline_text=baseline["text"],
            baseline_decoder_calls=int(baseline["target_decoder_calls"]),
            baseline_prefill_calls=int(baseline["target_prefill_calls"]),
            spec_text=tokenizer.decode(generated_ids, skip_special_tokens=True),
            spec_output_tokens=len(generated_ids),
            target_prefill_calls=target_prefill_calls,
            target_decoder_calls=total_rounds,
            accepted_drafts_per_round=accepted_drafts_per_round,
            draft_capacity=draft_capacity,
            extra_counts=extra_counts,
        )
    finally:
        runtime._run_draft_session = original_run_draft_session  # type: ignore[assignment]


@torch.no_grad()
def run_dense_spec_metrics_with_runtime(
    *,
    runtime: Qwen3_5SpecDecodeONNXModel,
    tokenizer,
    model_name: str,
    prompt: str,
    max_new_tokens: int,
    enable_thinking: bool = False,
    repetition_penalty: float = 1.0,
    presence_penalty: float = 0.0,
) -> Dict[str, Any]:
    reset_dense_spec_runtime(runtime)
    draft_counts = {"mtp_prefill_calls": 0, "mtp_decode_calls": 0, "dflash_context_calls": 0, "dflash_decode_calls": 0}
    original_run_draft_session = runtime._run_draft_session

    def counted_run_draft_session(session, input_feed):
        if runtime.spec_decode_mode == "mtp":
            if runtime.draft_prefill_session is not None and session is runtime.draft_prefill_session:
                draft_counts["mtp_prefill_calls"] += 1
            elif runtime.draft_decode_session is not None and session is runtime.draft_decode_session:
                draft_counts["mtp_decode_calls"] += 1
        else:
            if runtime.draft_context_session is not None and session is runtime.draft_context_session:
                draft_counts["dflash_context_calls"] += 1
            elif runtime.draft_decode_session is not None and session is runtime.draft_decode_session:
                draft_counts["dflash_decode_calls"] += 1
        return original_run_draft_session(session, input_feed)

    runtime._run_draft_session = counted_run_draft_session  # type: ignore[assignment]
    try:
        baseline = run_dense_target_baseline_from_spec(
            runtime,
            tokenizer,
            prompt,
            max_new_tokens,
            enable_thinking=enable_thinking,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
        )
        reference_token_ids = list(baseline.get("token_ids", []))[:max_new_tokens]
        if not reference_token_ids:
            extra_counts = (
                {
                    "mtp_prefill_calls": draft_counts["mtp_prefill_calls"],
                    "mtp_decode_calls": draft_counts["mtp_decode_calls"],
                }
                if runtime.spec_decode_mode == "mtp"
                else {
                    "dflash_prefill_calls": draft_counts["dflash_context_calls"],
                    "dflash_decode_calls": draft_counts["dflash_decode_calls"],
                }
            )
            return finalise_spec_metrics(
                prompt=prompt,
                draft_mode=runtime.spec_decode_mode,
                model_name=model_name,
                baseline_text=baseline["text"],
                baseline_decoder_calls=int(baseline["target_decoder_calls"]),
                baseline_prefill_calls=int(baseline["target_prefill_calls"]),
                baseline_output_tokens=int(baseline["output_tokens"]),
                spec_text="",
                spec_output_tokens=0,
                target_prefill_calls=int(baseline["target_prefill_calls"]),
                target_decoder_calls=0,
                accepted_drafts_per_round=[],
                draft_capacity=runtime.block_size - 1 if runtime.spec_decode_mode == "dflash" else runtime.block_size,
                extra_counts=extra_counts,
            )

        reset_dense_spec_runtime(runtime)
        input_ids = encode_chat_prompt(tokenizer, prompt, enable_thinking=enable_thinking)
        if runtime.max_context_tokens is not None and input_ids.shape[1] > runtime.max_context_tokens:
            input_ids = input_ids[:, -runtime.max_context_tokens :]

        total_prompt_len = int(input_ids.shape[1])
        runtime._ensure_prefill_session()
        prefill_cache_state = _alloc_cache_inputs(runtime.prefill_session, runtime.device)
        prefill_chunk_len = int(runtime._prefill_inputs_info.shape[1])

        last_prefill_logits = None
        last_hidden = None
        past_seq_len = 0
        target_prefill_calls = 0
        mtp_prefill_seq_len = 0
        mtp_pending_hidden = None

        for start in range(0, total_prompt_len, prefill_chunk_len):
            end = min(start + prefill_chunk_len, total_prompt_len)
            chunk_ids = input_ids[:, start:end]
            valid_len = int(chunk_ids.shape[1])
            target_prefill_calls += 1
            prefill_feed = runtime._build_prefill_feed(chunk_ids, valid_len, past_seq_len, prefill_cache_state)
            _, prefill_output_map = runtime._run_hmonnx(runtime.prefill_session, prefill_feed)
            prefill_logits = runtime._extract_logits(prefill_output_map)
            last_prefill_logits = _select_last_valid_logits(prefill_logits, valid_len)
            prefill_hidden_all = runtime._extract_hidden(prefill_output_map)
            if prefill_hidden_all is not None:
                prefill_hidden_all = prefill_hidden_all[:, :valid_len, :]
                last_hidden = runtime._select_hidden_step(prefill_hidden_all, valid_len - 1)
                if runtime.spec_decode_mode == "dflash":
                    runtime._append_dflash_context(prefill_hidden_all, past_seq_len)
                else:
                    hidden_parts = []
                    token_parts = []
                    if mtp_pending_hidden is not None:
                        hidden_parts.append(mtp_pending_hidden)
                        token_parts.append(chunk_ids[:, :1])
                    if valid_len > 1:
                        hidden_parts.append(prefill_hidden_all[:, : valid_len - 1, :])
                        token_parts.append(chunk_ids[:, 1:valid_len])
                    if hidden_parts:
                        mtp_hidden = torch.cat(hidden_parts, dim=1)
                        mtp_tokens = torch.cat(token_parts, dim=1)
                        runtime._prefill_mtp_chunk(mtp_hidden, mtp_tokens, mtp_prefill_seq_len)
                        mtp_prefill_seq_len += int(mtp_hidden.shape[1])
                    mtp_pending_hidden = prefill_hidden_all[:, valid_len - 1 : valid_len, :]

            runtime._update_linear_cache(prefill_cache_state, prefill_output_map)
            past_seq_len += valid_len

        if last_prefill_logits is None:
            raise RuntimeError("Prefill did not produce logits.")

        history_token_ids = input_ids[0].tolist()
        next_token_id = torch.tensor(
            [[reference_token_ids[0]]], dtype=torch.long, device=runtime.device
        )

        runtime._ensure_decode_session()
        decode_cache_state = _alloc_cache_inputs(runtime.decode_session, runtime.device)
        for name in decode_cache_state:
            if name in prefill_cache_state and (
                _is_kv_cache_name(name) or name.startswith(("past_conv_cache_", "past_recurrent_state_"))
            ):
                decode_cache_state[name] = prefill_cache_state[name]

        eos_token_id = tokenizer.eos_token_id
        generated_ids: List[int] = [reference_token_ids[0]]
        history_token_ids.append(reference_token_ids[0])

        current_token = next_token_id.to(runtime.device)
        mtp_past_seq_len = mtp_prefill_seq_len
        num_drafts = runtime.block_size - 1 if runtime.spec_decode_mode == "dflash" else runtime.block_size
        total_rounds = 0
        total_accepted_tokens = 0
        accepted_drafts_per_round: List[int] = []
        draft_capacity = num_drafts
        reference_limit = len(reference_token_ids)
        reference_index = 1

        while reference_index < reference_limit:
            total_rounds += 1
            if runtime.spec_decode_mode == "dflash":
                draft_tokens = runtime._run_draft_dflash(current_token, past_seq_len)
            else:
                mtp_cache_snapshot = {
                    name: _clone_cache_value(tensor)
                    for name, tensor in runtime._ensure_mtp_cache_state().items()
                }
                draft_tokens = runtime._run_draft_mtp(current_token, last_hidden, mtp_past_seq_len, num_drafts)

            verify_tokens = [current_token] + draft_tokens
            verify_input_ids = torch.cat(verify_tokens, dim=1)
            initial_seq_len = past_seq_len
            decode_feed = runtime._build_decode_feed(
                verify_input_ids,
                past_seq_len,
                decode_cache_state,
                current_input_length=verify_input_ids.shape[1],
            )
            _, decode_output_map = runtime._run_hmonnx(runtime.decode_session, decode_feed)
            verify_logits = _ensure_logits_shape(runtime._extract_logits(decode_output_map))
            verify_hidden_all = runtime._extract_hidden(decode_output_map)

            accepted_count = 0
            for idx, draft_tok in enumerate(draft_tokens):
                if reference_index + idx >= reference_limit:
                    break
                if draft_tok[0, 0].item() != reference_token_ids[reference_index + idx]:
                    break
                accepted_count += 1

            accepted_drafts_per_round.append(accepted_count)
            total_accepted_tokens += accepted_count
            accepted_steps = accepted_count + 1
            runtime._apply_verify_linear_cache_outputs(
                decode_cache_state,
                decode_output_map,
                accepted_steps=accepted_steps,
            )
            past_seq_len = initial_seq_len + accepted_steps

            eos_hit = False
            for idx in range(accepted_count):
                token_val = int(reference_token_ids[reference_index])
                generated_ids.append(token_val)
                history_token_ids.append(token_val)
                reference_index += 1
                if eos_token_id is not None and token_val == eos_token_id:
                    eos_hit = True
                    break
            if eos_hit or reference_index >= reference_limit:
                break

            current_token = torch.tensor(
                [[reference_token_ids[reference_index]]],
                dtype=torch.long,
                device=runtime.device,
            )

            last_hidden = runtime._select_hidden_step(verify_hidden_all, accepted_count)
            accepted_hidden = (
                verify_hidden_all[:, :accepted_steps, :] if verify_hidden_all is not None else None
            )
            if runtime.spec_decode_mode == "dflash":
                if accepted_hidden is not None:
                    runtime._append_dflash_context(accepted_hidden, initial_seq_len)
            else:
                runtime._mtp_cache_state = {
                    name: _as_cache_value(tensor, tensor.clone())
                    for name, tensor in mtp_cache_snapshot.items()
                }
                if accepted_hidden is not None:
                    mtp_cache_state = runtime._ensure_mtp_cache_state()
                    mtp_initial_seq_len = mtp_past_seq_len
                    for step_idx in range(accepted_steps):
                        next_token_for_cache = (
                            verify_tokens[step_idx + 1]
                            if step_idx < accepted_steps - 1
                            else current_token
                        )
                        draft_output_map = runtime._run_draft_session(
                            runtime.draft_decode_session,
                            runtime._build_mtp_decode_feed(
                                next_token_for_cache,
                                accepted_hidden[:, step_idx : step_idx + 1, :],
                                mtp_initial_seq_len + step_idx,
                                mtp_cache_state,
                            ),
                        )
                        for name in list(mtp_cache_state.keys()):
                            present_name = name.replace("past_", "present_", 1)
                            if present_name in draft_output_map:
                                mtp_cache_state[name] = _as_cache_value(
                                    mtp_cache_state[name],
                                    draft_output_map[present_name],
                                )
                    mtp_past_seq_len = mtp_initial_seq_len + accepted_steps

            token_val = int(reference_token_ids[reference_index])
            if eos_token_id is not None and token_val == eos_token_id:
                break
            generated_ids.append(token_val)
            history_token_ids.append(token_val)
            reference_index += 1

        extra_counts = (
            {
                "mtp_prefill_calls": draft_counts["mtp_prefill_calls"],
                "mtp_decode_calls": draft_counts["mtp_decode_calls"],
            }
            if runtime.spec_decode_mode == "mtp"
            else {
                "dflash_prefill_calls": draft_counts["dflash_context_calls"],
                "dflash_decode_calls": draft_counts["dflash_decode_calls"],
            }
        )
        return finalise_spec_metrics(
            prompt=prompt,
            draft_mode=runtime.spec_decode_mode,
            model_name=model_name,
            baseline_text=baseline["text"],
            baseline_decoder_calls=int(baseline["target_decoder_calls"]),
            baseline_prefill_calls=int(baseline["target_prefill_calls"]),
            baseline_output_tokens=int(baseline["output_tokens"]),
            spec_text=tokenizer.decode(generated_ids[:max_new_tokens], skip_special_tokens=True),
            spec_output_tokens=len(generated_ids[:max_new_tokens]),
            target_prefill_calls=target_prefill_calls,
            target_decoder_calls=total_rounds,
            accepted_drafts_per_round=accepted_drafts_per_round,
            draft_capacity=draft_capacity,
            extra_counts=extra_counts,
        )
    finally:
        runtime._run_draft_session = original_run_draft_session  # type: ignore[assignment]
        reset_dense_spec_runtime(runtime)


@torch.no_grad()
def run_moe_target_baseline_from_spec(
    runtime: Qwen3_5MoeSpecDecodeInference,
    prompt: str,
    max_new_tokens: int,
) -> Dict[str, Any]:
    messages = make_messages(prompt)
    texts = runtime.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
        add_generation_prompt=True,
    )
    if isinstance(texts, str):
        texts = [texts]
    model_inputs = runtime.tokenizer(texts, padding=False, return_tensors="pt")
    batch_input_ids = model_inputs.input_ids.cpu()
    prefill_len = batch_input_ids.shape[1]
    dev = runtime._device

    data_prefill = {"input_ids": batch_input_ids, "past_seq_length": 0}
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
    ) = runtime.prepare_inputs(data_prefill, runtime.prefill_input_sequence_length)
    lin_mask = build_moe_linear_attn_mask(
        int(seq_len_t.item()),
        runtime.prefill_input_sequence_length,
        inputs_embeds.dtype,
        dev,
    )

    runtime.set_phase_prefill(True)
    prefill_logits, _, _ = runtime._forward_with_hidden(
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
    last_valid = prefill_len - 1
    current_token = int(runtime._select_logits_step(prefill_logits, last_valid)[0].argmax(dim=-1).item())

    generated_ids = [current_token]
    past_len = prefill_len
    verify_len = runtime.spec_decode_verify_length
    decoder_calls = 0

    runtime.set_phase_prefill(False)
    try:
        while len(generated_ids) < max_new_tokens:
            if generated_ids[-1] == runtime.tokenizer.eos_token_id:
                break

            tok_emb = runtime._embed_token_ids(torch.tensor([[current_token]], dtype=torch.long, device=dev))
            if verify_len > 1:
                pad = torch.zeros(
                    1,
                    verify_len - 1,
                    tok_emb.shape[2],
                    dtype=tok_emb.dtype,
                    device=dev,
                )
                tok_emb = torch.cat([tok_emb, pad], dim=1)

            vpos = torch.arange(past_len, past_len + verify_len, dtype=torch.int32, device=dev).unsqueeze(0)
            v_past = torch.tensor([past_len], dtype=torch.int32, device=dev)
            v_cur = torch.tensor([1], dtype=torch.int32, device=dev)
            lin_mask_v = build_moe_linear_attn_mask(1, verify_len, tok_emb.dtype, dev)

            decoder_calls += 1
            verify_logits, _, verify_outputs = runtime._forward_with_hidden(
                tok_emb,
                vpos,
                vpos,
                vpos,
                v_past,
                v_cur,
                lin_mask_v,
                past_key_caches,
                past_value_caches,
                past_conv_caches,
                past_recurrent_states,
                update_linear_cache=False,
            )
            runtime._apply_verify_linear_cache_outputs(
                past_conv_caches,
                past_recurrent_states,
                verify_outputs,
                accepted_steps=1,
            )
            current_token = int(verify_logits[0, 0, :].argmax(dim=-1).item())
            if current_token == runtime.tokenizer.eos_token_id:
                break
            generated_ids.append(current_token)
            past_len += 1
    finally:
        runtime.set_phase_prefill(True)

    return {
        "text": runtime.tokenizer.decode(generated_ids, skip_special_tokens=True),
        "target_prefill_calls": 1,
        "target_decoder_calls": decoder_calls,
        "output_tokens": len(generated_ids),
    }


@torch.no_grad()
def run_moe_spec_metrics(
    *,
    meta_path: str,
    prompt: str,
    max_new_tokens: int,
    fast_mode: bool,
) -> Dict[str, Any]:
    runtime_base = load_moe_inference(meta_path, fast_mode=fast_mode)
    if not isinstance(runtime_base, Qwen3_5MoeSpecDecodeInference):
        raise TypeError(f"{meta_path} is not a MoE spec-decode runtime.")
    baseline = run_moe_target_baseline_from_spec(runtime_base, prompt, max_new_tokens)
    release_moe_runtime(runtime_base)
    del runtime_base

    runtime = load_moe_inference(meta_path, fast_mode=fast_mode)
    if not isinstance(runtime, Qwen3_5MoeSpecDecodeInference):
        raise TypeError(f"{meta_path} is not a MoE spec-decode runtime.")

    mtp_prefill_calls = 0
    mtp_decode_calls = 0
    dflash_context_calls = 0
    dflash_decode_calls = 0

    original_prefill_mtp_chunk = runtime._prefill_mtp_chunk
    original_mtp_decode_step = runtime._mtp_decode_step
    original_append_dflash_context = runtime._append_dflash_context
    original_run_draft_dflash = runtime._run_draft_dflash

    def counted_prefill_mtp_chunk(*args, **kwargs):
        nonlocal mtp_prefill_calls
        mtp_prefill_calls += 1
        return original_prefill_mtp_chunk(*args, **kwargs)

    def counted_mtp_decode_step(*args, **kwargs):
        nonlocal mtp_decode_calls
        mtp_decode_calls += 1
        return original_mtp_decode_step(*args, **kwargs)

    def counted_append_dflash_context(*args, **kwargs):
        nonlocal dflash_context_calls
        dflash_context_calls += 1
        return original_append_dflash_context(*args, **kwargs)

    def counted_run_draft_dflash(*args, **kwargs):
        nonlocal dflash_decode_calls
        dflash_decode_calls += 1
        return original_run_draft_dflash(*args, **kwargs)

    runtime._prefill_mtp_chunk = counted_prefill_mtp_chunk  # type: ignore[assignment]
    runtime._mtp_decode_step = counted_mtp_decode_step  # type: ignore[assignment]
    runtime._append_dflash_context = counted_append_dflash_context  # type: ignore[assignment]
    runtime._run_draft_dflash = counted_run_draft_dflash  # type: ignore[assignment]

    try:
        verify_len = runtime.spec_decode_verify_length
        num_drafts = runtime.spec_decode_block_size - 1 if runtime.spec_decode_mode == "dflash" else runtime.spec_decode_block_size

        texts = runtime.tokenizer.apply_chat_template(
            make_messages(prompt),
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=True,
        )
        if isinstance(texts, str):
            texts = [texts]
        model_inputs = runtime.tokenizer(texts, padding=False, return_tensors="pt")
        batch_input_ids = model_inputs.input_ids.cpu()
        prefill_len = batch_input_ids.shape[1]
        dev = runtime._device

        data_prefill = {"input_ids": batch_input_ids, "past_seq_length": 0}
        isl = runtime.prefill_input_sequence_length
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
        ) = runtime.prepare_inputs(data_prefill, isl)
        lin_mask = build_moe_linear_attn_mask(int(seq_len_t.item()), isl, inputs_embeds.dtype, dev)

        runtime.set_phase_prefill(True)
        prefill_logits, prefill_hidden_all, _ = runtime._forward_with_hidden(
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

        last_valid = prefill_len - 1
        current_token = int(runtime._select_logits_step(prefill_logits, last_valid)[0].argmax(dim=-1).item())
        generated_ids = [current_token]
        past_len = prefill_len

        mtp_past_seq_len = 0
        current_mtp_hidden = None
        if runtime.spec_decode_mode == "mtp":
            runtime._reset_mtp_cache()
            if prefill_hidden_all is not None:
                prefill_hidden_all = prefill_hidden_all[:, :prefill_len, :]
                if prefill_len > 1:
                    next_token_embedding = runtime._embed_token_ids(batch_input_ids[:, 1:prefill_len].to(dev))
                    runtime._prefill_mtp_chunk(
                        next_token_embedding,
                        prefill_hidden_all[:, : prefill_len - 1, :],
                        0,
                    )
                    mtp_past_seq_len = prefill_len - 1
                current_mtp_hidden = prefill_hidden_all[:, last_valid : last_valid + 1, :]
        else:
            runtime._append_dflash_context(
                prefill_hidden_all[:, :prefill_len, :] if prefill_hidden_all is not None else None,
                0,
            )

        target_decoder_calls = 0
        accepted_drafts_per_round: List[int] = []

        runtime.set_phase_prefill(False)
        try:
            step = 0
            while step < max_new_tokens:
                if generated_ids[-1] == runtime.tokenizer.eos_token_id:
                    break

                if runtime.spec_decode_mode == "dflash":
                    draft_tokens = runtime._run_draft_dflash(current_token, past_len)
                else:
                    if current_mtp_hidden is not None:
                        mtp_snapshot = (
                            _clone_cache_value(runtime._mtp_k_cache),
                            _clone_cache_value(runtime._mtp_v_cache),
                        )
                    else:
                        mtp_snapshot = None
                    draft_tokens = []
                    mtp_hidden = current_mtp_hidden
                    for idx in range(num_drafts):
                        if mtp_hidden is None:
                            break
                        tok_id = current_token if idx == 0 else draft_tokens[-1]
                        tok_emb = runtime._embed_token_ids(torch.tensor([[tok_id]], dtype=torch.long, device=dev))
                        d_logits, mtp_hidden = runtime._mtp_decode_step(
                            tok_emb,
                            mtp_hidden.to(dev),
                            mtp_past_seq_len + idx,
                        )
                        draft_tokens.append(int(d_logits[0, 0, :].argmax(dim=-1).item()))

                verify_ids = [current_token] + draft_tokens
                actual_vlen = len(verify_ids)
                verify_embeds_t = runtime._embed_token_ids(torch.tensor([verify_ids], dtype=torch.long, device=dev))
                if actual_vlen < verify_len:
                    pad = torch.zeros(
                        1,
                        verify_len - actual_vlen,
                        verify_embeds_t.shape[2],
                        dtype=verify_embeds_t.dtype,
                        device=dev,
                    )
                    verify_embeds_t = torch.cat([verify_embeds_t, pad], dim=1)

                vpos = torch.arange(past_len, past_len + verify_len, dtype=torch.int32, device=dev).unsqueeze(0)
                v_past = torch.tensor([past_len], dtype=torch.int32, device=dev)
                v_cur = torch.tensor([actual_vlen], dtype=torch.int32, device=dev)
                lin_mask_v = build_moe_linear_attn_mask(actual_vlen, verify_len, verify_embeds_t.dtype, dev)

                initial_seq_len = past_len
                target_decoder_calls += 1
                verify_logits, verify_hidden, verify_outputs = runtime._forward_with_hidden(
                    verify_embeds_t,
                    vpos,
                    vpos,
                    vpos,
                    v_past,
                    v_cur,
                    lin_mask_v,
                    past_key_caches,
                    past_value_caches,
                    past_conv_caches,
                    past_recurrent_states,
                    update_linear_cache=False,
                )

                accepted = 0
                new_tok = -1
                for idx, draft_tok in enumerate(draft_tokens):
                    t_tok = int(verify_logits[0, idx, :].argmax(dim=-1).item())
                    if t_tok == draft_tok:
                        accepted += 1
                    else:
                        new_tok = t_tok
                        break
                else:
                    bonus_logit = verify_logits[0, len(draft_tokens), :]
                    new_tok = int(bonus_logit.argmax(dim=-1).item())

                accepted_drafts_per_round.append(accepted)
                accepted_steps = accepted + 1
                runtime._apply_verify_linear_cache_outputs(
                    past_conv_caches,
                    past_recurrent_states,
                    verify_outputs,
                    accepted_steps=accepted_steps,
                )

                generated_ids.extend(draft_tokens[:accepted])
                generated_ids.append(new_tok)
                n_added = accepted_steps
                past_len += n_added
                step += n_added
                current_token = new_tok

                current_mtp_hidden = runtime._select_hidden_step(verify_hidden, accepted)
                accepted_hidden = verify_hidden[:, :accepted_steps, :] if verify_hidden is not None else None
                if runtime.spec_decode_mode == "dflash":
                    runtime._append_dflash_context(accepted_hidden, initial_seq_len)
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
                                torch.tensor([[next_token_for_cache]], dtype=torch.long, device=dev)
                            )
                            runtime._mtp_decode_step(
                                tok_emb,
                                accepted_hidden[:, step_idx : step_idx + 1, :],
                                mtp_initial_seq_len + step_idx,
                            )
                        mtp_past_seq_len = mtp_initial_seq_len + accepted_steps

                if generated_ids[-1] == runtime.tokenizer.eos_token_id:
                    break
        finally:
            runtime.set_phase_prefill(True)

        extra_counts = (
            {
                "mtp_prefill_calls": mtp_prefill_calls,
                "mtp_decode_calls": mtp_decode_calls,
            }
            if runtime.spec_decode_mode == "mtp"
            else {
                "dflash_prefill_calls": dflash_context_calls,
                "dflash_decode_calls": dflash_decode_calls,
            }
        )

        return finalise_spec_metrics(
            prompt=prompt,
            draft_mode=runtime.spec_decode_mode,
            model_name=runtime.meta_info.get("model_name", meta_path),
            baseline_text=baseline["text"],
            baseline_decoder_calls=int(baseline["target_decoder_calls"]),
            baseline_prefill_calls=int(baseline["target_prefill_calls"]),
            spec_text=runtime.tokenizer.decode(generated_ids[:max_new_tokens], skip_special_tokens=True),
            spec_output_tokens=min(len(generated_ids), max_new_tokens),
            target_prefill_calls=1,
            target_decoder_calls=target_decoder_calls,
            accepted_drafts_per_round=accepted_drafts_per_round,
            draft_capacity=num_drafts,
            extra_counts=extra_counts,
        )
    finally:
        runtime._prefill_mtp_chunk = original_prefill_mtp_chunk  # type: ignore[assignment]
        runtime._mtp_decode_step = original_mtp_decode_step  # type: ignore[assignment]
        runtime._append_dflash_context = original_append_dflash_context  # type: ignore[assignment]
        runtime._run_draft_dflash = original_run_draft_dflash  # type: ignore[assignment]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect float/HMONNX speculative-decode metrics.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p = subparsers.add_parser("float-mtp")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--num-draft-tokens", type=int, default=4)
    p.add_argument("--dtype", default="bf16", choices=sorted(FLOAT_DTYPE_MAP.keys()))
    p.add_argument("--system-prompt", default="")
    add_thinking_args(p)

    p = subparsers.add_parser("float-mtp-suite")
    p.add_argument("--model", required=True)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--num-draft-tokens", type=int, default=4)
    p.add_argument("--dtype", default="bf16", choices=sorted(FLOAT_DTYPE_MAP.keys()))
    p.add_argument("--system-prompt", default="You are a helpful assistant.")
    add_thinking_args(p)

    p = subparsers.add_parser("float-dflash")
    p.add_argument("--model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--dtype", default="bf16", choices=sorted(FLOAT_DTYPE_MAP.keys()))
    p.add_argument("--block-size", type=int, default=None)
    add_thinking_args(p)

    p = subparsers.add_parser("float-dflash-suite")
    p.add_argument("--model", required=True)
    p.add_argument("--draft-model", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--max-samples", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=4096)
    p.add_argument("--dtype", default="bf16", choices=sorted(FLOAT_DTYPE_MAP.keys()))
    p.add_argument("--block-size", type=int, default=None)
    add_thinking_args(p)

    p = subparsers.add_parser("hmonnx-dense")
    p.add_argument("--meta", required=True)
    p.add_argument("--baseline-meta", default=None)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--dtype", default="fp16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--exec-device", default="cuda")

    p = subparsers.add_parser("hmonnx-moe")
    p.add_argument("--meta", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--fast", action="store_true")

    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    enable_thinking = resolve_enable_thinking(args)
    if args.command == "float-mtp":
        result = run_float_mtp_metrics(
            model_path=args.model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            num_draft_tokens=args.num_draft_tokens,
            dtype=args.dtype,
            system_prompt=args.system_prompt,
            enable_thinking=enable_thinking,
        )
    elif args.command == "float-mtp-suite":
        result = run_float_mtp_suite_metrics(
            model_path=args.model,
            max_new_tokens=args.max_new_tokens,
            num_draft_tokens=args.num_draft_tokens,
            dtype=args.dtype,
            system_prompt=args.system_prompt,
            enable_thinking=enable_thinking,
        )
    elif args.command == "float-dflash":
        block_size = resolve_dflash_block_size(args.draft_model, args.block_size)
        result = run_float_dflash_metrics(
            model_path=args.model,
            draft_model_path=args.draft_model,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            block_size=block_size,
            enable_thinking=bool(enable_thinking),
        )
    elif args.command == "float-dflash-suite":
        block_size = resolve_dflash_block_size(args.draft_model, args.block_size)
        result = run_float_dflash_suite_metrics(
            model_path=args.model,
            draft_model_path=args.draft_model,
            dataset=args.dataset,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            block_size=block_size,
            enable_thinking=bool(enable_thinking),
        )
    elif args.command == "hmonnx-dense":
        result = run_dense_spec_metrics(
            meta_path=args.meta,
            baseline_meta_path=args.baseline_meta,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            dtype=args.dtype,
            device=args.device,
            exec_device=args.exec_device,
        )
    else:
        result = run_moe_spec_metrics(
            meta_path=args.meta,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            fast_mode=args.fast,
        )

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        Path(args.output_json).write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
