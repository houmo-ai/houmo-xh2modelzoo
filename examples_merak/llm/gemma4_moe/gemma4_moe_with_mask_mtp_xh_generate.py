from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import sys
from pathlib import Path

import torch

project_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gemma4_moe_mtp_common import resolve_layer_type_to_last_index
from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXModel
from xhmodel_merak.xh_llm.models.gemma4.data_preprocess import Gemma4DataPreprocess
from xhquant.core import CacheTensor
from xhquant.api import get_xhquant_logger, xhquant_init
from xhquant.utils import MemoryTracker, TimeProfiler
from xhquant.xhonnxruntime import HMONNXGrapInference


def load_json_file(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute() and not path.exists():
        path = (base_dir / path).resolve()
    return path.resolve() if not path.is_absolute() else path


def load_prompt_inputs(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    return tokenizer([text], return_tensors="pt").input_ids


def resolve_eos_token_ids(tokenizer, *configs: dict) -> list[int]:
    token_ids: list[int] = []
    for config in configs:
        value = config.get("eos_token_id")
        if value is None:
            continue
        if isinstance(value, int):
            value = [value]
        for token_id in value:
            token_id = int(token_id)
            if token_id not in token_ids:
                token_ids.append(token_id)
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is not None:
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        for token_id in eos_token_id:
            token_id = int(token_id)
            if token_id not in token_ids:
                token_ids.append(token_id)
    return token_ids


def resolve_runtime_layer_type_to_last_index(meta_path: Path, meta: dict, target_model) -> dict[str, int]:
    layer_types = meta.get("model_config", {}).get("layer_types")
    if isinstance(layer_types, dict):
        layer_types = None
    if not layer_types:
        runtime_layer_types = getattr(target_model.meta_info.model_config, "layer_types", None)
        if isinstance(runtime_layer_types, list) and runtime_layer_types:
            layer_types = runtime_layer_types
    if not layer_types:
        hf_config_dir = resolve_path(meta_path.parent, getattr(target_model.meta_info, "hf_config", meta["hf_config"]))
        hf_config = load_json_file(hf_config_dir / "config.json")
        text_config = hf_config.get("text_config", hf_config)
        layer_types = text_config.get("layer_types")
    if not isinstance(layer_types, list) or not layer_types:
        raise RuntimeError("Cannot resolve Gemma4 layer_types from runtime meta or exported hf_config.")
    return resolve_layer_type_to_last_index(layer_types)


def load_hf_text_config(meta_path: Path, meta: dict, target_model) -> dict:
    hf_config_dir = resolve_path(meta_path.parent, getattr(target_model.meta_info, "hf_config", meta["hf_config"]))
    hf_config = load_json_file(hf_config_dir / "config.json")
    return hf_config.get("text_config", hf_config)


def resolve_weight_shard(model_dir: Path, weight_name: str) -> Path | None:
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = load_json_file(index_path).get("weight_map", {})
        shard_name = weight_map.get(weight_name)
        if shard_name is not None:
            return model_dir / shard_name
    return None


def load_tied_lm_head_weight(target_model, meta: dict) -> torch.Tensor:
    cached_weight = getattr(target_model, "_mtp_lm_head_weight", None)
    if cached_weight is not None:
        return cached_weight

    from safetensors import safe_open

    candidate_model_dirs = []
    model_config = meta.get("model_config", {})
    for key in ("hf_model", "fallback_hf_model"):
        model_dir = model_config.get(key)
        if model_dir:
            candidate_model_dirs.append(Path(model_dir))

    candidate_weight_names = (
        "lm_head.weight",
        "model.language_model.embed_tokens.weight",
        "language_model.embed_tokens.weight",
        "model.embed_tokens.weight",
    )
    for model_dir in candidate_model_dirs:
        for weight_name in candidate_weight_names:
            shard_path = resolve_weight_shard(model_dir, weight_name)
            if shard_path is None or not shard_path.exists():
                continue
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                if weight_name not in handle.keys():
                    continue
                weight = handle.get_tensor(weight_name).contiguous()
            weight = weight.to(device=target_model.device, dtype=target_model.dtype)
            target_model._mtp_lm_head_weight = weight
            return weight

    raise RuntimeError(
        "Cannot load tied lm_head weight for Gemma4 MTP logits projection from hf_model or fallback_hf_model."
    )


def project_logits_if_needed(
    target_model,
    logits: torch.Tensor,
    meta_path: Path,
    meta: dict,
    text_config: dict | None = None,
) -> torch.Tensor:
    text_config = text_config or load_hf_text_config(meta_path, meta, target_model)
    vocab_size = int(text_config.get("vocab_size", 0))
    if vocab_size > 0 and logits.shape[-1] == vocab_size:
        return logits

    lm_head_weight = load_tied_lm_head_weight(target_model, meta)
    logits = torch.matmul(logits.to(dtype=lm_head_weight.dtype), lm_head_weight.t())
    final_logit_softcapping = text_config.get("final_logit_softcapping")
    if final_logit_softcapping is not None:
        final_logit_softcapping = float(final_logit_softcapping)
        logits = torch.tanh(logits / final_logit_softcapping) * final_logit_softcapping
    return logits


def cache_to_tensor(cache) -> torch.Tensor:
    if isinstance(cache, torch.Tensor):
        return cache
    if hasattr(cache, "data"):
        return cache.data
    if hasattr(cache, "tensor"):
        return cache.tensor
    return torch.as_tensor(cache)


def slice_cache_to_window(cache: torch.Tensor, window_size: int, valid_length: int) -> torch.Tensor:
    if window_size <= 0:
        raise ValueError(f"window_size must be positive, got {window_size}")
    valid_length = min(max(int(valid_length), 0), int(cache.shape[2]))
    start = max(0, valid_length - window_size)
    window = cache[:, :, start:valid_length, :]
    if window.shape[2] < window_size:
        pad = torch.zeros(
            (*window.shape[:2], window_size - window.shape[2], window.shape[3]),
            dtype=window.dtype,
            device=window.device,
        )
        window = torch.cat([window, pad], dim=2)
    return window


class AssistantDraftSession:
    def __init__(self, onnx_path: str, device: torch.device):
        self.session = HMONNXGrapInference(onnx_path)
        if device.type != "cpu":
            self.session.to(device)
        self.session.exec_device = device
        self.input_names = self.session.get_input_names()
        self.output_names = self.session.get_output_names()
        self.input_infos = {name: self.session.get_input(name) for name in self.input_names}
        # The exported HMONNX renames the first positional input to ``input_1``
        # and the second positional integer input to ``valid_length``. Map both
        # back to the semantic names produced by ``build_assistant_round_inputs``.
        self.input_aliases = {
            "input_1": "inputs_embeds",
            "valid_length": "past_seq_length",
        }
        self.device = device

    def __call__(self, **inputs) -> tuple[torch.Tensor, torch.Tensor]:
        feed = {}
        for name in self.input_names:
            source_name = name if name in inputs else self.input_aliases.get(name)
            if source_name is None or source_name not in inputs:
                raise KeyError(name)
            value = inputs[source_name]
            if isinstance(value, torch.Tensor):
                value = value.detach().to(self.device)
                expected_dtype = self.input_infos[name].dtype
                if value.dtype != expected_dtype:
                    value = value.to(expected_dtype)
            feed[name] = value
        outputs = self.session.run(feed)
        if isinstance(outputs, dict):
            output_map = outputs
        else:
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
            output_map = {name: output for name, output in zip(self.output_names, outputs)}
        return output_map[self.output_names[0]], output_map[self.output_names[1]]


def parse_target_outputs(
    outputs,
    target_model,
    meta_path: Path,
    meta: dict,
    text_config: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(outputs, (tuple, list)):
        raise RuntimeError("Expected target HMONNX to return logits and hidden state, but only got a single output.")
    if len(outputs) < 2:
        raise RuntimeError("Target HMONNX export is missing hidden state output; re-export with the MTP export script.")
    logits = project_logits_if_needed(target_model, outputs[0], meta_path, meta, text_config)
    return logits, outputs[1]


def run_prefill(
    target_model,
    input_ids: torch.Tensor,
    meta_path: Path,
    meta: dict,
    text_config: dict,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    prefill_chunk_length = int(target_model.meta_info.model_config.prefill_chunk_length)
    past_seq_length = 0
    logits = None
    hidden = None
    for start in range(0, input_ids.shape[1], prefill_chunk_length):
        chunk = input_ids[:, start : start + prefill_chunk_length]
        data = {
            "input_ids": chunk,
            "past_seq_length": past_seq_length,
        }
        target_model.set_prefill()
        target_model.set_input_sequence_length(prefill_chunk_length)
        model_inputs = target_model.prepare_inputs_with_masks(data)
        outputs = target_model.forward(*model_inputs)
        logits, hidden = parse_target_outputs(outputs, target_model, meta_path, meta, text_config)

        past_seq_length += int(chunk.shape[1])
    assert logits is not None and hidden is not None
    return logits, hidden, past_seq_length


def clone_cache_tensor(cache) -> CacheTensor:
    if isinstance(cache, CacheTensor):
        cloned = type(cache)(cache.data.clone())
        if hasattr(cache, "cache_valid_len"):
            cloned.cache_valid_len = cache.cache_valid_len
        return cloned
    if isinstance(cache, torch.Tensor):
        return CacheTensor(cache.detach().clone())
    raise TypeError(f"Unsupported cache type: {type(cache)!r}")


def clone_cache_list(caches) -> list[CacheTensor]:
    return [clone_cache_tensor(cache) for cache in caches]


def build_verify_model_inputs(
    target_model,
    input_ids: torch.Tensor,
    past_seq_length: int,
    input_sequence_length: int,
    past_key_caches,
    past_value_caches,
):
    model_config = target_model.meta_info.model_config
    sliding_window_cfg = getattr(target_model, "sliding_window_cfg", {})
    data_processor = Gemma4DataPreprocess(
        token_embedding=target_model.get_input_embeddings(),
        input_sequence_length=input_sequence_length,
        context_length=model_config.context_max_length,
        past_key_caches=past_key_caches,
        past_value_caches=past_value_caches,
        pad_token_id=target_model.pad_token_id,
        image_token_id=getattr(model_config, "image_token_id", -1) or -1,
        sliding_window=sliding_window_cfg.get("sliding_window", 1024),
    )
    data_processor.to(target_model.device, target_model.dtype)
    (
        inputs_embeds,
        past_seq_length_tensor,
        seq_length_tensor,
        full_attention_mask,
        sliding_attention_mask,
        past_key_caches,
        past_value_caches,
    ) = data_processor(
        {
            "input_ids": input_ids,
            "past_seq_length": past_seq_length,
        }
    )
    model_inputs = [inputs_embeds, past_seq_length_tensor, seq_length_tensor]
    if sliding_attention_mask is not None:
        model_inputs.append(sliding_attention_mask)
    if full_attention_mask is not None:
        model_inputs.append(full_attention_mask)
    model_inputs.extend(past_key_caches)
    model_inputs.extend(past_value_caches)
    return tuple(model_inputs)


def run_verify(
    target_model,
    input_ids: torch.Tensor,
    past_seq_length: int,
    input_sequence_length: int,
    past_key_caches,
    past_value_caches,
    meta_path: Path,
    meta: dict,
    text_config: dict,
    decode_model=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    target_model.set_decode()
    target_model.set_input_sequence_length(input_sequence_length)
    model_inputs = build_verify_model_inputs(
        target_model,
        input_ids=input_ids,
        past_seq_length=past_seq_length,
        input_sequence_length=input_sequence_length,
        past_key_caches=past_key_caches,
        past_value_caches=past_value_caches,
    )
    outputs = (decode_model or target_model.decode_model)(*model_inputs)
    return parse_target_outputs(outputs, target_model, meta_path, meta, text_config)


def run_decode_one(
    target_model,
    token_id: int,
    past_seq_length: int,
    meta_path: Path,
    meta: dict,
    text_config: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([[token_id]], dtype=torch.long, device=target_model.device)
    return run_verify(
        target_model=target_model,
        input_ids=input_ids,
        past_seq_length=past_seq_length,
        input_sequence_length=1,
        past_key_caches=target_model.past_key_caches,
        past_value_caches=target_model.past_value_caches,
        meta_path=meta_path,
        meta=meta,
        text_config=text_config,
    )


def replace_target_caches(target_model, past_key_caches, past_value_caches) -> None:
    target_model._kvcache_mixin.past_key_caches = past_key_caches
    target_model._kvcache_mixin.past_value_caches = past_value_caches


def commit_verified_caches(
    target_model,
    verified_key_caches,
    verified_value_caches,
    past_seq_length: int,
    commit_length: int,
    verify_length: int,
) -> None:
    if commit_length >= verify_length:
        valid_length = int(past_seq_length) + int(commit_length)
        for cache in [*verified_key_caches, *verified_value_caches]:
            if hasattr(cache, "cache_valid_len"):
                cache.cache_valid_len = valid_length
        replace_target_caches(target_model, verified_key_caches, verified_value_caches)
        return

    committed_key_caches = clone_cache_list(target_model.past_key_caches)
    committed_value_caches = clone_cache_list(target_model.past_value_caches)
    start = max(int(past_seq_length), 0)
    for committed_cache, verified_cache in zip(committed_key_caches, verified_key_caches):
        committed_tensor = cache_to_tensor(committed_cache)
        verified_tensor = cache_to_tensor(verified_cache)
        end = min(int(committed_tensor.shape[2]), start + int(commit_length))
        if start < end:
            committed_tensor[:, :, start:end, :].copy_(verified_tensor[:, :, start:end, :])
        if hasattr(committed_cache, "cache_valid_len"):
            committed_cache.cache_valid_len = start + int(commit_length)

    for committed_cache, verified_cache in zip(committed_value_caches, verified_value_caches):
        committed_tensor = cache_to_tensor(committed_cache)
        verified_tensor = cache_to_tensor(verified_cache)
        end = min(int(committed_tensor.shape[2]), start + int(commit_length))
        if start < end:
            committed_tensor[:, :, start:end, :].copy_(verified_tensor[:, :, start:end, :])
        if hasattr(committed_cache, "cache_valid_len"):
            committed_cache.cache_valid_len = start + int(commit_length)

    replace_target_caches(target_model, committed_key_caches, committed_value_caches)


def run_batch_verify(
    target_model,
    verify_decode_model,
    input_tokens: list[int],
    past_seq_length: int,
    meta_path: Path,
    meta: dict,
    text_config: dict,
) -> tuple[torch.Tensor, torch.Tensor, list[CacheTensor], list[CacheTensor]]:
    temp_key_caches = clone_cache_list(target_model.past_key_caches)
    temp_value_caches = clone_cache_list(target_model.past_value_caches)
    input_ids = torch.tensor([input_tokens], dtype=torch.long, device=target_model.device)
    logits, hidden = run_verify(
        target_model=target_model,
        input_ids=input_ids,
        past_seq_length=past_seq_length,
        input_sequence_length=len(input_tokens),
        past_key_caches=temp_key_caches,
        past_value_caches=temp_value_caches,
        meta_path=meta_path,
        meta=meta,
        text_config=text_config,
        decode_model=verify_decode_model,
    )
    return logits, hidden, temp_key_caches, temp_value_caches


def build_assistant_round_inputs(
    target_model,
    layer_type_to_last_index: dict[str, int],
    last_token_id: int,
    current_hidden: torch.Tensor,
    past_seq_length: int,
    position_index: int,
):
    device = target_model.device
    embed_tokens = target_model.get_input_embeddings().to(device)
    token_tensor = torch.tensor([[last_token_id]], dtype=torch.long, device=device)
    last_token_embed = embed_tokens(token_tensor).to(dtype=current_hidden.dtype)
    inputs_embeds = torch.cat([last_token_embed, current_hidden], dim=-1)

    target_cache_valid_length = max(int(past_seq_length), 0)
    # ``prepare_attention_masks`` expects the *number of valid K positions*
    # already in the cache (i.e. ``past_seq_length``), not the index of the
    # last valid one. Passing ``N - 1`` here would mask the most recently
    # committed token away from the assistant's view, which collapses the
    # MTP accept rate.
    local_attention_mask, global_attention_mask = target_model.prepare_attention_masks(
        last_token_embed,
        target_cache_valid_length,
    )
    sliding_index = layer_type_to_last_index.get("sliding_attention")
    full_index = layer_type_to_last_index.get("full_attention")
    if sliding_index is None or full_index is None:
        raise RuntimeError(
            "Gemma4 MTP requires both sliding_attention and full_attention layers in the exported hf_config."
        )
    required_cache_count = max(sliding_index, full_index) + 1
    if len(target_model.past_key_caches) < required_cache_count or len(target_model.past_value_caches) < required_cache_count:
        raise RuntimeError(
            "Target runtime does not expose enough KV cache groups for Gemma4 MTP. "
            "This usually means the target model was exported in only_first_block/--valid mode. "
            "Please re-export without --valid so both shared cache types are available."
        )

    shared_key_cache_sliding = cache_to_tensor(target_model.past_key_caches[sliding_index])
    shared_value_cache_sliding = cache_to_tensor(target_model.past_value_caches[sliding_index])
    shared_key_cache_full = cache_to_tensor(target_model.past_key_caches[full_index])
    shared_value_cache_full = cache_to_tensor(target_model.past_value_caches[full_index])
    if local_attention_mask is not None:
        local_window = int(local_attention_mask.shape[-1])
        shared_key_cache_sliding = slice_cache_to_window(
            shared_key_cache_sliding,
            local_window,
            target_cache_valid_length,
        )
        shared_value_cache_sliding = slice_cache_to_window(
            shared_value_cache_sliding,
            local_window,
            target_cache_valid_length,
        )

    return dict(
        inputs_embeds=inputs_embeds,
        # The exported assistant graph slices cos/sin caches with DynamicSlice
        # driven by ``past_seq_length`` (int32 shape ``(1,)``) instead of
        # gathering by ``position_ids``. Compute the constant draft starting
        # offset here so each draft round indexes the same row of the cache.
        past_seq_length=torch.tensor([position_index], dtype=torch.int32, device=device),
        local_attention_mask=local_attention_mask,
        global_attention_mask=global_attention_mask,
        shared_key_cache_sliding=shared_key_cache_sliding,
        shared_value_cache_sliding=shared_value_cache_sliding,
        shared_key_cache_full=shared_key_cache_full,
        shared_value_cache_full=shared_value_cache_full,
    )


def generate_with_mtp(
    target_model,
    assistant_session,
    verify_decode_model,
    tokenizer,
    meta: dict,
    layer_type_to_last_index: dict[str, int],
    prompt: str,
    max_new_tokens: int,
):
    input_ids = load_prompt_inputs(tokenizer, prompt).to(target_model.device)
    output_ids = input_ids[0].tolist()
    prompt_length = int(input_ids.shape[1])
    meta_path = Path(meta["_meta_path"])
    text_config = meta["_text_config"]

    def decode_generated_text() -> str:
        return tokenizer.decode(output_ids[prompt_length:], skip_special_tokens=True).strip()

    logits, hidden, past_seq_length = run_prefill(target_model, input_ids, meta_path, meta, text_config)
    current_hidden = hidden
    next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
    eos_token_ids = set(resolve_eos_token_ids(tokenizer, meta, meta.get("model_config", {})))
    spec_decode = meta["spec_decode"]
    num_draft_tokens = int(spec_decode.get("block_size", 4))
    verify_length = int(spec_decode.get("verify_length", num_draft_tokens + 1))

    output_ids.append(next_token)
    generated = 1
    mtp_stats = {
        "verify_rounds": 0,
        "draft_proposed": 0,
        "draft_accepted": 0,
        "committed_tokens": 0,
        "accepted_per_round": [],
    }
    if next_token in eos_token_ids or generated >= max_new_tokens:
        return decode_generated_text(), output_ids, mtp_stats

    while generated < max_new_tokens:
        draft_tokens: list[int] = []
        assistant_hidden = current_hidden
        assistant_last_token_id = next_token
        # vLLM's ``Gemma4Proposer`` sets ``constant_draft_positions=True``:
        # all draft steps in one round use the same rotary position (the
        # last target-model position). The draft model's KV cache is not
        # extended between steps, so advancing the position would only
        # decorrelate Q from the target K already in cache.
        assistant_position = max(past_seq_length, 0)

        for _ in range(num_draft_tokens):
            round_inputs = build_assistant_round_inputs(
                target_model,
                layer_type_to_last_index,
                assistant_last_token_id,
                assistant_hidden,
                past_seq_length,
                assistant_position,
            )
            draft_logits, assistant_hidden = assistant_session(**round_inputs)
            assistant_token = int(torch.argmax(draft_logits[:, -1, :], dim=-1).item())
            draft_tokens.append(assistant_token)
            assistant_last_token_id = assistant_token

        verify_tokens = [next_token] + draft_tokens
        if len(verify_tokens) != verify_length:
            raise RuntimeError(f"Expected verify input length {verify_length}, got {len(verify_tokens)}")
        verify_logits, verify_hidden, verified_key_caches, verified_value_caches = run_batch_verify(
            target_model=target_model,
            verify_decode_model=verify_decode_model,
            input_tokens=verify_tokens,
            past_seq_length=past_seq_length,
            meta_path=meta_path,
            meta=meta,
            text_config=text_config,
        )
        if verify_logits.shape[1] < verify_length or verify_hidden.shape[1] < verify_length:
            raise RuntimeError(
                "Verify HMONNX must return full K+1 logits and hidden states. "
                f"got logits={tuple(verify_logits.shape)}, hidden={tuple(verify_hidden.shape)}, "
                f"verify_length={verify_length}"
            )

        accepted_count = 0
        for index, draft_token in enumerate(draft_tokens):
            verified_token = int(torch.argmax(verify_logits[:, index, :], dim=-1).item())
            if verified_token != draft_token:
                break
            accepted_count += 1

        mtp_stats["verify_rounds"] += 1
        mtp_stats["draft_proposed"] += num_draft_tokens
        mtp_stats["draft_accepted"] += accepted_count
        mtp_stats["accepted_per_round"].append(accepted_count)

        commit_length = accepted_count + 1
        commit_verified_caches(
            target_model,
            verified_key_caches,
            verified_value_caches,
            past_seq_length=past_seq_length,
            commit_length=commit_length,
            verify_length=verify_length,
        )
        past_seq_length += commit_length
        mtp_stats["committed_tokens"] += commit_length
        current_hidden = verify_hidden[:, accepted_count : accepted_count + 1, :]

        for draft_token in draft_tokens[:accepted_count]:
            output_ids.append(draft_token)
            generated += 1
            if draft_token in eos_token_ids or generated >= max_new_tokens:
                return decode_generated_text(), output_ids, mtp_stats

        next_token = int(torch.argmax(verify_logits[:, accepted_count, :], dim=-1).item())
        output_ids.append(next_token)
        generated += 1
        if next_token in eos_token_ids or generated >= max_new_tokens:
            return decode_generated_text(), output_ids, mtp_stats

    return decode_generated_text(), output_ids, mtp_stats


def main(args):
    log_file = Path(args.meta).resolve().parent / "mtp_generate.log"
    xhquant_init(str(log_file), args.debug)
    logger = get_xhquant_logger()

    meta_path = Path(args.meta).resolve()
    meta = load_json_file(meta_path)
    meta["_meta_path"] = str(meta_path)
    spec_decode = meta.get("spec_decode")
    if spec_decode is None:
        raise ValueError("Meta file does not contain spec_decode section. Use golden_meta_info_mtp.json from the export script.")

    device = torch.device(args.device)
    target_model = AutoLLMHONNXModel.from_pretrained(str(meta_path))
    target_model.to(device)
    tokenizer = target_model.get_tokenizer(trust_remote_code=True)
    meta["_text_config"] = load_hf_text_config(meta_path, meta, target_model)
    layer_type_to_last_index = resolve_runtime_layer_type_to_last_index(meta_path, meta, target_model)

    draft_onnx = resolve_path(meta_path.parent, spec_decode["draft_decode_onnx"])
    assistant_session = AssistantDraftSession(str(draft_onnx), device)
    verify_hmonnx = spec_decode.get("verify_decode_onnx") or spec_decode.get("verify_hmonnx") or meta.get("verify_hmonnx")
    if not verify_hmonnx:
        raise ValueError("Meta file does not contain verify_hmonnx; re-export with the MTP export script.")
    verify_decode_model = HMONNXModel(str(resolve_path(meta_path.parent, verify_hmonnx))).to(device)

    if args.smoke_test:
        print("Task09 Gemma4 MTP smoke OK", meta_path)
        return

    with TimeProfiler("gemma4_moe_mtp_hmonnx_generate", logger), MemoryTracker(device=str(device), name="generate", logger=logger):
        with ExitStack() as stack:
            stack.enter_context(LLMInferenceContextManager(target_model, [device]))
            output_text, output_ids, mtp_stats = generate_with_mtp(
                target_model=target_model,
                assistant_session=assistant_session,
                verify_decode_model=verify_decode_model,
                tokenizer=tokenizer,
                meta=meta,
                layer_type_to_last_index=layer_type_to_last_index,
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
            )

    print(output_text)
    logger.info(f"Generated {len(output_ids)} total tokens")

    rounds = mtp_stats["verify_rounds"]
    proposed = mtp_stats["draft_proposed"]
    accepted = mtp_stats["draft_accepted"]
    committed = mtp_stats["committed_tokens"]
    if rounds > 0:
        avg_accepted = accepted / rounds
        accept_rate = accepted / proposed if proposed > 0 else 0.0
        avg_commit = committed / rounds
        logger.info(
            f"[MTP stats] verify_rounds={rounds} draft_proposed={proposed} draft_accepted={accepted} "
            f"committed_tokens={committed} avg_accepted_per_verify={avg_accepted:.3f} "
            f"draft_accept_rate={accept_rate * 100:.2f}% avg_committed_per_verify={avg_commit:.3f}"
        )
        print(
            f"[MTP stats] rounds={rounds} num_draft_per_round={proposed // rounds} "
            f"avg_accepted_per_verify={avg_accepted:.3f} "
            f"draft_accept_rate={accept_rate * 100:.2f}% (accepted {accepted}/{proposed}) "
            f"avg_committed_per_verify={avg_commit:.3f}"
        )
    else:
        logger.info("[MTP stats] no verify rounds executed (sequence ended at prefill).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gemma4 MoE with assistant MTP HMONNX generate example")
    parser.add_argument(
        "--meta",
        type=str,
        required=True,
        help="Path to golden_meta_info_mtp.json produced by the MTP export script.",
    )
    parser.add_argument("--prompt", type=str, default="介绍一下 speculative decoding 的核心思想。")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--debug", action="store_true")
    main(parser.parse_args())