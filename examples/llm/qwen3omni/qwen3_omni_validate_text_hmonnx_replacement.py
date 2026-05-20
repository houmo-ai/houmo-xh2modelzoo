# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Validate text HMONNX as a thinker replacement inside HF Qwen3-Omni.

This script keeps the full HF Qwen3-Omni model shell, replaces only the
`thinker.generate` path with exported text HMONNX prefill/decode sessions, and
compares dialogue outputs against the native HF path on the same prompts.

Scope:
- text dialogue only
- dialogue diff uses `return_audio=False`
- text-only audio generation sanity check is attempted when the processor is available
- greedy decode (`thinker_do_sample=False`)
"""

import argparse
import json
import os.path as osp
import sys
import time
import types
from pathlib import Path
from typing import Iterable, Optional

import soundfile as sf
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hmonnx_pipeline import (  # noqa: E402
    _build_safe_validation_max_memory,
    _create_hmonnx_session,
    _build_dense_deepstack_tensors,
    _ensure_tensor,
    _ensure_mistral_common_reasoning_effort,
    _extract_outputs,
    _extract_primary_output,
    _resolve_validation_device_map,
    build_conversation,
    discover_artifacts,
    save_json,
)
from xhquant.api import CacheTensor, get_root_logger, set_random_seed, xhquant_init  # isort:skip  # noqa: E402

try:
    from qwen_omni_utils import process_mm_info
except ImportError:

    def process_mm_info(conversation, use_audio_in_video=False):
        audios, images, videos = [], [], []
        for turn in conversation:
            for item in turn.get("content", []):
                tp = item.get("type")
                if tp == "audio":
                    audios.append(item.get("audio"))
                elif tp == "image":
                    images.append(item.get("image"))
                elif tp == "video":
                    videos.append(item.get("video"))
        return audios, images, videos


def _default_prompts() -> list[str]:
    return [
        "请用一句话介绍你自己。",
        "把 Hello world 翻译成中文，只输出译文。",
        "用一句话说明 Python 和 C++ 的区别。",
    ]


def _mask_and_scatter_modal_features(
    inputs_embeds: torch.Tensor,
    input_ids_cpu: torch.Tensor,
    token_id: int,
    modal_features: torch.Tensor,
    modal_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    modal_mask = input_ids_cpu == token_id
    expanded_mask = modal_mask.unsqueeze(-1).expand_as(inputs_embeds)
    modal_features = modal_features.to(device=torch.device("cpu"), dtype=inputs_embeds.dtype)
    if inputs_embeds[expanded_mask].numel() != modal_features.numel():
        raise ValueError(
            f"{modal_name} features and placeholder tokens do not match: "
            f"tokens={int(modal_mask.sum())}, features={tuple(modal_features.shape)}"
        )
    inputs_embeds = inputs_embeds.masked_scatter(expanded_mask, modal_features)
    return inputs_embeds, modal_mask


def _extract_sequence_tensor(output) -> torch.Tensor:
    primary = output[0] if isinstance(output, tuple) else output
    if hasattr(primary, "sequences"):
        primary = primary.sequences
    if not isinstance(primary, torch.Tensor):
        raise TypeError(f"Unsupported generation output type: {type(primary)}")
    return primary


def _normalize_text(text: str) -> str:
    return " ".join(text.strip().split()).lower()


def _first_output_ids(output_ids: list[list[int]]) -> list[int]:
    if not output_ids:
        return []
    return [int(token_id) for token_id in output_ids[0]]


def _common_prefix_length(left: Iterable, right: Iterable) -> int:
    prefix_length = 0
    for left_item, right_item in zip(left, right):
        if left_item != right_item:
            break
        prefix_length += 1
    return prefix_length


def _build_report_item(item: dict, baseline: dict, replacement: dict) -> dict:
    baseline_text = baseline["output_text"][0] if baseline["output_text"] else ""
    replacement_text = replacement["output_text"][0] if replacement["output_text"] else ""
    baseline_ids = _first_output_ids(baseline["output_ids"])
    replacement_ids = _first_output_ids(replacement["output_ids"])

    return {
        "case": item["case"] or "text",
        "prompt": item["prompt"] if item["prompt"] is not None else item["case"],
        "rendered_text": baseline["rendered_text"],
        "native_hf_output": baseline_text,
        "text_hmonnx_replacement_output": replacement_text,
        "native_hf_output_ids": baseline["output_ids"],
        "text_hmonnx_replacement_output_ids": replacement["output_ids"],
        "native_token_count": len(baseline_ids),
        "replacement_token_count": len(replacement_ids),
        "common_prefix_token_count": _common_prefix_length(baseline_ids, replacement_ids),
        "common_prefix_char_count": _common_prefix_length(baseline_text, replacement_text),
        "token_exact_match": baseline_ids == replacement_ids,
        "normalized_exact_match": _normalize_text(baseline_text) == _normalize_text(replacement_text),
        "replacement_non_empty": bool(replacement_text.strip()),
    }


def _write_markdown_report(report_path: Path, report: dict) -> None:
    lines = [
        "# Text HMONNX LLM Replacement Report",
        "",
        f"- create_time: {report['create_time']}",
        f"- hf_model_path: {report['hf_model_path']}",
        f"- work_dir: {report['work_dir']}",
        f"- text_meta: {report['text_meta']}",
        f"- max_new_tokens: {report['max_new_tokens']}",
        f"- device_map: {report['device_map']}",
        "",
        "## Result Summary",
        "",
    ]

    for item in report["results"]:
        lines.append(
            "- "
            f"case={item['case']} | normalized_exact_match={item['normalized_exact_match']} | "
            f"token_exact_match={item['token_exact_match']} | "
            f"native_tokens={item['native_token_count']} | replacement_tokens={item['replacement_token_count']} | "
            f"common_prefix_tokens={item['common_prefix_token_count']} | "
            f"common_prefix_chars={item['common_prefix_char_count']}"
        )

    audio_check = report.get("audio_output_replacement_validation")
    if audio_check is not None:
        case_name = audio_check.get("case", "text")
        audio_lines = [
            "",
            f"## Audio Output Replacement Validation: {case_name}",
            "",
            f"- supported: {audio_check['supported']}",
            f"- error_type: {audio_check['error_type']}",
            f"- error_message: {audio_check['error_message']}",
        ]
        if audio_check.get("output_text") is not None:
            audio_lines.append(f"- output_text: {audio_check['output_text']}")
        if audio_check.get("audio_file") is not None:
            audio_lines.append(f"- audio_file: {audio_check['audio_file']}")
        if audio_check.get("output_ids_file") is not None:
            audio_lines.append(f"- output_ids_file: {audio_check['output_ids_file']}")
        lines.extend(
            audio_lines
            + [
                "",
                "### Rendered Text",
                "",
                "```text",
                audio_check["rendered_text"].rstrip("\n"),
                "```",
            ]
        )

    for item in report["results"]:
        lines.extend(
            [
                "",
                f"## Case: {item['case']}",
                "",
                f"- prompt: {item['prompt']}",
                f"- normalized_exact_match: {item['normalized_exact_match']}",
                f"- token_exact_match: {item['token_exact_match']}",
                f"- native_token_count: {item['native_token_count']}",
                f"- replacement_token_count: {item['replacement_token_count']}",
                f"- common_prefix_token_count: {item['common_prefix_token_count']}",
                f"- common_prefix_char_count: {item['common_prefix_char_count']}",
                "",
                "### Rendered Text",
                "",
                "```text",
                item["rendered_text"].rstrip("\n"),
                "```",
                "",
                "### Native HF Output",
                "",
                "```text",
                item["native_hf_output"],
                "```",
                "",
                "### Text HMONNX Replacement Output",
                "",
                "```text",
                item["text_hmonnx_replacement_output"],
                "```",
                "",
                "### Output Token IDs",
                "",
                "```json",
                json.dumps(
                    {
                        "native_hf_output_ids": item["native_hf_output_ids"],
                        "text_hmonnx_replacement_output_ids": item["text_hmonnx_replacement_output_ids"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                "```",
            ]
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _collect_text_mismatches(results: list[dict]) -> list[dict]:
    mismatches = []
    for item in results:
        if item["normalized_exact_match"] and item["replacement_non_empty"]:
            continue
        mismatches.append(item)
    return mismatches


def _get_known_case_limitations(case: str, text_meta: dict) -> list[str]:
    if case == "text":
        return []

    limitations = []
    if not text_meta.get("supports_multimodal_position_ids", False):
        limitations.append(
            "current text HMONNX artifact does not expose multimodal position_ids/get_rope_index support; "
            "non-text cases cannot be validated with strict token-level equality"
        )

    if text_meta.get("prefill_hidden_states_contract") != "accept_hidden_layer_full_prompt_pre_norm":
        limitations.append(
            "current text HMONNX artifact only exports last-token hidden states; multimodal talker audio validation "
            "needs full prompt accept_hidden_layer states"
        )

    return limitations


def _build_unsupported_audio_validation(case: str, rendered_text: str, reasons: list[str]) -> dict:
    return {
        "case": case,
        "rendered_text": rendered_text,
        "supported": False,
        "error_type": "UnsupportedConfiguration",
        "error_message": "; ".join(reasons),
    }


def _resolve_eos_token_ids(eos_token_id, thinker_config) -> set[int]:
    eos_token_ids: set[int] = set()
    if isinstance(eos_token_id, int):
        eos_token_ids.add(int(eos_token_id))
    elif isinstance(eos_token_id, (list, tuple, set)):
        eos_token_ids.update(int(token_id) for token_id in eos_token_id)

    cfg_eos = getattr(thinker_config, "eos_token_id", None)
    if isinstance(cfg_eos, int):
        eos_token_ids.add(int(cfg_eos))
    elif isinstance(cfg_eos, (list, tuple, set)):
        eos_token_ids.update(int(token_id) for token_id in cfg_eos)

    cfg_im_end = getattr(thinker_config, "im_end_token_id", None)
    if isinstance(cfg_im_end, int):
        eos_token_ids.add(int(cfg_im_end))
    return eos_token_ids


def _build_text_hmonnx_generate_patch(thinker, text_meta: dict, logger, accept_hidden_layer: Optional[int] = None):
    token_embedding_state_dict = torch.load(
        Path(text_meta["_root_dir"]) / text_meta["token_embedding_file"],
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(token_embedding_state_dict, dict) or "weight" not in token_embedding_state_dict:
        raise RuntimeError("token embedding file does not contain a standard state_dict")

    token_embedding = nn.Embedding(
        token_embedding_state_dict["weight"].shape[0],
        token_embedding_state_dict["weight"].shape[1],
    )
    token_embedding.load_state_dict(token_embedding_state_dict)
    token_embedding.eval()

    kv_cache_info = text_meta["kv_cache"]
    kv_shape = kv_cache_info["shape"]
    num_layers = kv_cache_info["num_decoder_layers"]
    input_sequence_length = int(text_meta["wrap_cfg"]["input_sequence_length"])

    prefill_session = _create_hmonnx_session(_resolve_meta_path(text_meta, "prefill_onnx"))
    decode_session = _create_hmonnx_session(_resolve_meta_path(text_meta, "decode_onnx"))
    text_supports_position_ids = bool(text_meta.get("supports_multimodal_position_ids", False))
    prefill_tensor_input_count = len(prefill_session.inputs) - (2 * num_layers)
    expected_base_inputs = 6 if text_supports_position_ids else 3
    text_supports_deepstack = prefill_tensor_input_count == expected_base_inputs + 3
    output_names = text_meta.get("output_names")
    text_supports_hidden_states = output_names == ["logits", "hidden_states"]
    text_supports_full_prefill_hidden = (
        text_meta.get("prefill_hidden_states_contract") == "accept_hidden_layer_full_prompt_pre_norm"
    )
    if not text_supports_hidden_states:
        prefill_output_names = getattr(prefill_session, "get_output_names", lambda: [])()
        decode_output_names = getattr(decode_session, "get_output_names", lambda: [])()
        text_supports_hidden_states = len(prefill_output_names) >= 2 and len(decode_output_names) >= 2
    if accept_hidden_layer is None:
        accept_hidden_layer = getattr(getattr(thinker, "config", None), "accept_hidden_layer", None)
    if accept_hidden_layer is None:
        accept_hidden_layer = text_meta.get("accept_hidden_layer")
    if accept_hidden_layer is None:
        raise ValueError("Qwen3-Omni text HMONNX replacement requires accept_hidden_layer")
    accept_hidden_layer = max(int(accept_hidden_layer), 0)

    def _extract_logits_and_hidden_states(output, actual_seq_len: int):
        outputs = _extract_outputs(output)
        logits = _ensure_tensor(outputs[0], torch.device("cpu"), torch.float32)
        if logits.ndim == 2:
            logits = logits.unsqueeze(1)
        logits = logits[:, :actual_seq_len, :]

        hidden_states = None
        if len(outputs) >= 2:
            hidden_states = _ensure_tensor(outputs[1], torch.device("cpu"), torch.float16)
            if hidden_states.ndim == 2:
                hidden_states = hidden_states.unsqueeze(1)
            hidden_states = hidden_states[:, :actual_seq_len, :]
        return logits, hidden_states

    def _build_step_hidden_states(step_embeds: torch.Tensor, step_hidden: Optional[torch.Tensor]):
        step_embeds = step_embeds.to(torch.float16)
        step_hidden = step_embeds if step_hidden is None else step_hidden.to(torch.float16)
        if accept_hidden_layer == 0:
            return (step_embeds,)
        return tuple(step_embeds if index == 0 else step_hidden for index in range(accept_hidden_layer + 1))

    def _empty_caches() -> tuple[list[CacheTensor], list[CacheTensor]]:
        return (
            [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)],
            [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16)) for _ in range(num_layers)],
        )

    def _pad_position_ids(position_ids: torch.Tensor, actual_seq_len: int, target_seq_len: int) -> torch.Tensor:
        position_ids = position_ids.detach().cpu().to(torch.int32)
        position_ids = position_ids[:actual_seq_len]
        if actual_seq_len >= target_seq_len:
            return position_ids
        return torch.cat(
            [
                position_ids,
                torch.zeros(target_seq_len - actual_seq_len, dtype=torch.int32),
            ],
            dim=0,
        )

    def _build_prefill_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        image_grid_thw,
        video_grid_thw,
        feature_attention_mask,
        use_audio_in_video,
        video_second_per_grid,
        target_seq_len: int,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        audio_feature_lengths = None
        if feature_attention_mask is not None:
            audio_feature_lengths = torch.sum(feature_attention_mask, dim=1)
        position_ids, rope_deltas = self.get_rope_index(
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask,
            use_audio_in_video,
            audio_feature_lengths,
            video_second_per_grid,
        )
        delta0 = (1 - attention_mask).sum(dim=-1).unsqueeze(1)
        rope_deltas = rope_deltas - delta0
        actual_seq_len = int(input_ids.shape[1])
        return (
            _pad_position_ids(position_ids[0, 0], actual_seq_len, target_seq_len),
            _pad_position_ids(position_ids[1, 0], actual_seq_len, target_seq_len),
            _pad_position_ids(position_ids[2, 0], actual_seq_len, target_seq_len),
            rope_deltas.detach().cpu().to(torch.long),
        )

    def _build_decode_position_ids(decode_past_seq_length: torch.Tensor, rope_deltas: Optional[torch.Tensor]):
        if rope_deltas is None:
            position_ids = torch.full(
                (3, 1),
                int(decode_past_seq_length.item()),
                dtype=torch.int32,
            )
        else:
            delta = int(decode_past_seq_length.item()) + int(rope_deltas.reshape(-1)[0].item())
            position_ids = torch.full((3, 1), delta, dtype=torch.int32)
        return position_ids[0], position_ids[1], position_ids[2]

    def hmonnx_generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        eos_token_id=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("text HMONNX thinker replacement requires input_ids")

        output_hidden_states = bool(kwargs.pop("output_hidden_states", False))
        return_dict_in_generate = bool(kwargs.pop("return_dict_in_generate", False))
        need_hidden_states = output_hidden_states or return_dict_in_generate
        if need_hidden_states and not text_supports_hidden_states:
            raise NotImplementedError(
                "Current text HMONNX artifact uses the legacy logits-only contract. Re-export text HMONNX with "
                "hidden_states output before validating Qwen3-Omni talker audio generation."
            )

        input_features = kwargs.pop("input_features", None)
        feature_attention_mask = kwargs.pop("feature_attention_mask", None)
        pixel_values = kwargs.pop("pixel_values", None)
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        image_grid_thw = kwargs.pop("image_grid_thw", None)
        video_grid_thw = kwargs.pop("video_grid_thw", None)
        kwargs.pop("audio_feature_lengths", None)
        use_audio_in_video = kwargs.pop("use_audio_in_video", None)
        video_second_per_grid = kwargs.pop("video_second_per_grid", None)
        kwargs.pop("use_cache", None)
        kwargs.pop("cache_position", None)
        kwargs.pop("position_ids", None)

        if pixel_values_videos is not None:
            raise NotImplementedError("Video inputs are not supported by this replacement validator yet")

        input_ids_cpu = input_ids.detach().cpu()
        if input_ids_cpu.ndim != 2 or input_ids_cpu.shape[0] != 1:
            raise NotImplementedError("This validation script only supports batch_size=1")

        inputs_embeds = token_embedding(input_ids_cpu)
        deepstack_tensors = [torch.zeros_like(inputs_embeds, dtype=torch.float16) for _ in range(3)]

        if input_features is not None:
            audio_outputs = self.get_audio_features(
                input_features=input_features,
                feature_attention_mask=feature_attention_mask,
            )
            if hasattr(audio_outputs, "last_hidden_state"):
                audio_features = audio_outputs.last_hidden_state
            else:
                audio_features = audio_outputs
            inputs_embeds, _ = _mask_and_scatter_modal_features(
                inputs_embeds,
                input_ids_cpu,
                int(self.config.audio_token_id),
                audio_features,
                "audio",
            )

        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw)
            if hasattr(image_outputs, "pooler_output") and image_outputs.pooler_output is not None:
                image_embeds = image_outputs.pooler_output
                image_embeds_multiscale = getattr(image_outputs, "deepstack_features", None)
            elif hasattr(image_outputs, "last_hidden_state"):
                image_embeds = image_outputs.last_hidden_state
                image_embeds_multiscale = getattr(image_outputs, "deepstack_features", None)
            elif isinstance(image_outputs, tuple):
                image_embeds, image_embeds_multiscale = image_outputs
            else:
                image_embeds, image_embeds_multiscale = image_outputs, None
            inputs_embeds, image_mask = _mask_and_scatter_modal_features(
                inputs_embeds,
                input_ids_cpu,
                int(self.config.image_token_id),
                image_embeds,
                "image",
            )
            if image_embeds_multiscale is not None:
                deepstack_tensors = _build_dense_deepstack_tensors(
                    inputs_embeds,
                    image_mask,
                    list(image_embeds_multiscale),
                )

        prefill_token_length = int(inputs_embeds.shape[1])
        if prefill_token_length > input_sequence_length:
            raise RuntimeError(
                "text HMONNX prompt length "
                f"{prefill_token_length} exceeds exported input_sequence_length {input_sequence_length}"
            )

        if prefill_token_length < input_sequence_length:
            pad_embeds = torch.zeros(
                (inputs_embeds.shape[0], input_sequence_length - prefill_token_length, inputs_embeds.shape[2]),
                dtype=inputs_embeds.dtype,
            )
            inputs_embeds = torch.cat([inputs_embeds, pad_embeds], dim=1)
            deepstack_tensors = [
                torch.cat([tensor, torch.zeros_like(pad_embeds, dtype=torch.float16)], dim=1)
                for tensor in deepstack_tensors
            ]
        zero_decode_deepstack = [
            torch.zeros((1, 1, inputs_embeds.shape[2]), dtype=torch.float16) for _ in range(3)
        ]
        prefill_step_embeds = inputs_embeds[:, :prefill_token_length, :].detach().cpu()
        rope_deltas = None
        if text_supports_position_ids:
            time_position_ids, height_position_ids, width_position_ids, rope_deltas = _build_prefill_position_ids(
                self,
                input_ids,
                attention_mask,
                image_grid_thw,
                video_grid_thw,
                feature_attention_mask,
                bool(use_audio_in_video),
                video_second_per_grid,
                input_sequence_length,
            )

        past_key_caches, past_value_caches = _empty_caches()
        current_input_length = torch.tensor([prefill_token_length], dtype=torch.int32)
        past_seq_length = torch.tensor([0], dtype=torch.int32)

        prefill_inputs = [
            inputs_embeds.to(torch.float16),
        ]
        if text_supports_position_ids:
            prefill_inputs.extend([time_position_ids, height_position_ids, width_position_ids])
        prefill_inputs.extend([
            past_seq_length,
            current_input_length,
        ])
        if text_supports_deepstack:
            prefill_inputs.extend(deepstack_tensors)

        prefill_logits = prefill_session.forward(
            *prefill_inputs,
            *past_key_caches,
            *past_value_caches,
        )
        prefill_logits, prefill_hidden_states = _extract_logits_and_hidden_states(prefill_logits, prefill_token_length)
        next_token = torch.argmax(prefill_logits[:, -1, :], dim=-1, keepdim=True)
        generated_hidden_states = []
        if need_hidden_states:
            if not text_supports_full_prefill_hidden and prefill_hidden_states is not None and prefill_token_length > 1:
                raise NotImplementedError(
                    "Current text HMONNX artifact does not provide full prefill hidden states. Re-export text HMONNX "
                    "with prefill_full_accept_hidden=True before validating Qwen3-Omni talker audio generation."
                )
            generated_hidden_states.append(_build_step_hidden_states(prefill_step_embeds, prefill_hidden_states))

        eos_token_ids = _resolve_eos_token_ids(eos_token_id, self.config)
        decode_past_seq_length = current_input_length.clone()
        one_length = torch.ones_like(current_input_length)
        kv_max_seq = kv_shape[2] if len(kv_shape) > 2 else kv_shape[-1]
        generated_tokens = [next_token]

        for step in range(max_new_tokens - 1):
            token_id = int(next_token.item())
            if token_id in eos_token_ids:
                break
            if int(decode_past_seq_length.item()) + 1 > kv_max_seq:
                if logger is not None:
                    logger.warning(f"KV cache full at step {step + 1}, stopping thinker decode")
                break

            decode_inputs = [
                token_embedding(next_token.cpu()).to(torch.float16),
            ]
            if text_supports_position_ids:
                decode_inputs.extend(_build_decode_position_ids(decode_past_seq_length, rope_deltas))
            decode_inputs.extend([
                decode_past_seq_length,
                one_length,
            ])
            if text_supports_deepstack:
                decode_inputs.extend(zero_decode_deepstack)

            decode_logits = decode_session.forward(
                *decode_inputs,
                *past_key_caches,
                *past_value_caches,
            )
            decode_step_embeds = token_embedding(next_token.cpu()).detach().cpu()
            decode_logits, decode_hidden_states = _extract_logits_and_hidden_states(decode_logits, 1)
            next_token = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            if need_hidden_states:
                generated_hidden_states.append(_build_step_hidden_states(decode_step_embeds, decode_hidden_states))
            decode_past_seq_length = decode_past_seq_length + 1

        output_ids = torch.cat(generated_tokens, dim=-1).to(input_ids_cpu.dtype)
        sequences = torch.cat([input_ids_cpu, output_ids], dim=-1)
        sequences = sequences.to(input_ids.device)
        if not need_hidden_states:
            return sequences
        return types.SimpleNamespace(
            sequences=sequences,
            hidden_states=tuple(generated_hidden_states),
        )

    return types.MethodType(hmonnx_generate, thinker)


def _resolve_meta_path(meta: dict, field_name: str) -> Path:
    root_dir = Path(meta["_root_dir"])
    return root_dir / meta[field_name]


def _ensure_chat_template(tokenizer, model_dir: Path) -> None:
    if getattr(tokenizer, "chat_template", None):
        return

    chat_template_file = model_dir / "chat_template.json"
    if not chat_template_file.exists():
        raise RuntimeError(
            "Tokenizer chat_template is missing and no chat_template.json was found under "
            f"{model_dir}"
        )

    with chat_template_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    chat_template = payload.get("chat_template") if isinstance(payload, dict) else None
    if not isinstance(chat_template, str) or not chat_template.strip():
        raise RuntimeError(f"chat_template.json in {model_dir} does not contain a valid chat_template string")

    tokenizer.chat_template = chat_template


def _prepare_text_inputs(tokenizer, device: torch.device, text_prompt: Optional[str] = None):
    conversation, _ = build_conversation("text", text_prompt=text_prompt)
    text = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(text=text, return_tensors="pt", padding=True)
    inputs = inputs.to(device)
    return text, inputs, False


def _prepare_case_inputs(processor, case: str, device: torch.device, dtype: torch.dtype, text_prompt: Optional[str] = None):
    conversation, use_audio_in_video = build_conversation(case, text_prompt=text_prompt)
    rendered_text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=use_audio_in_video)
    inputs = processor(
        text=rendered_text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        seconds_per_chunk=2.0,
        position_id_per_seconds=13,
        use_audio_in_video=use_audio_in_video,
    )
    inputs.pop("hm_pixel_values", None)
    inputs.pop("hm_pixel_values_videos", None)
    inputs = inputs.to(device).to(dtype)
    return rendered_text, inputs, use_audio_in_video


def _prepare_inputs_for_case(
    tokenizer,
    device: torch.device,
    dtype: torch.dtype,
    case: str,
    processor=None,
):
    if case == "text":
        return _prepare_text_inputs(tokenizer, device)
    if processor is None:
        raise RuntimeError(f"processor is required for multimodal case={case}")
    return _prepare_case_inputs(processor, case, device, dtype)


def _run_dialogue_case(
    native_model,
    tokenizer,
    case: str,
    max_new_tokens: int,
    processor=None,
):
    device = next(native_model.parameters()).device
    dtype = next(native_model.parameters()).dtype
    rendered_text, inputs, use_audio_in_video = _prepare_inputs_for_case(
        tokenizer,
        device,
        dtype,
        case,
        processor=processor,
    )

    with torch.no_grad():
        output = native_model.generate(
            **inputs,
            return_audio=False,
            use_audio_in_video=use_audio_in_video,
            thinker_do_sample=False,
            thinker_max_new_tokens=max_new_tokens,
        )

    sequences = _extract_sequence_tensor(output)
    new_tokens = sequences[:, inputs["input_ids"].shape[1] :]
    decoded = tokenizer.batch_decode(
        new_tokens,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "case": case,
        "rendered_text": rendered_text,
        "output_ids": new_tokens.detach().cpu().tolist(),
        "output_text": decoded,
    }


def _attempt_audio_output(
    native_model,
    tokenizer,
    processor,
    max_new_tokens: int,
    talker_max_new_tokens: int,
    output_dir: Optional[Path] = None,
    case: str = "text",
):
    device = next(native_model.parameters()).device
    dtype = next(native_model.parameters()).dtype
    rendered_text, inputs, use_audio_in_video = _prepare_inputs_for_case(
        tokenizer,
        device,
        dtype,
        case,
        processor=processor,
    )
    audio_file = None
    output_ids_file = None
    output_text = None

    try:
        with torch.no_grad():
            generated = native_model.generate(
                **inputs,
                return_audio=True,
                speaker="Ethan",
                use_audio_in_video=use_audio_in_video,
                thinker_do_sample=False,
                thinker_max_new_tokens=max_new_tokens,
                talker_max_new_tokens=talker_max_new_tokens,
            )

        sequences = _extract_sequence_tensor(generated)
        new_tokens = sequences[:, inputs["input_ids"].shape[1] :]
        output_text = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        audio = generated[1] if isinstance(generated, tuple) and len(generated) > 1 else None
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            case_slug = case.replace("/", "_").replace(" ", "_")
            output_ids_name = (
                "text_hmonnx_talker_output_ids.pt"
                if case_slug == "text"
                else f"text_hmonnx_talker_{case_slug}_output_ids.pt"
            )
            output_ids_path = output_dir / output_ids_name
            torch.save(
                {
                    "case": case,
                    "input_ids": inputs["input_ids"].detach().cpu(),
                    "output_ids": sequences.detach().cpu(),
                    "new_tokens": new_tokens.detach().cpu(),
                    "output_text": output_text,
                },
                output_ids_path,
            )
            output_ids_file = str(output_ids_path)

        if audio is not None and output_dir is not None:
            audio_name = (
                "text_hmonnx_talker_output.wav"
                if case_slug == "text"
                else f"text_hmonnx_talker_{case_slug}_output.wav"
            )
            audio_path = output_dir / audio_name
            sf.write(
                str(audio_path),
                audio.reshape(-1).to(dtype=torch.float32).detach().cpu().numpy(),
                samplerate=24000,
            )
            audio_file = str(audio_path)
    except Exception as exc:
        return {
            "case": case,
            "rendered_text": rendered_text,
            "supported": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }

    return {
        "case": case,
        "rendered_text": rendered_text,
        "supported": True,
        "output_text": output_text,
        "audio_file": audio_file,
        "output_ids_file": output_ids_file,
        "error_type": None,
        "error_message": None,
    }


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(exist_ok=True, parents=True)
    xhquant_init(work_dir / "text_hmonnx_replacement.log", debug=args.debug)
    logger = get_root_logger()
    set_random_seed(args.seed)

    artifacts = discover_artifacts(work_dir)
    text_meta = artifacts.get("text")
    if text_meta is None:
        raise RuntimeError(f"No qwen3omni text meta.json found under {work_dir}")

    from transformers import AutoTokenizer, Qwen3OmniMoeForConditionalGeneration

    logger.info(f"Loading HF Qwen3-Omni model from {hf_model_path}")
    resolved_device_map = _resolve_validation_device_map(args.device_map, logger)
    max_memory = None
    if resolved_device_map == "auto" and torch.cuda.is_available():
        max_memory = _build_safe_validation_max_memory(logger)
    native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        hf_model_path,
        torch_dtype=torch.float16,
        device_map=resolved_device_map,
        max_memory=max_memory,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    native_model.eval()
    _ensure_mistral_common_reasoning_effort()
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    _ensure_chat_template(tokenizer, Path(hf_model_path))

    validation_case = args.case

    processor = None
    if validation_case != "text":
        from xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor

        processor = Qwen3OmniMoeProcessor.from_pretrained(hf_model_path)
    validation_item = {"case": validation_case, "prompt": None}

    report = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "hf_model_path": hf_model_path,
        "work_dir": str(work_dir),
        "text_meta": str(Path(text_meta["_meta_path"]).resolve()) if "_meta_path" in text_meta else None,
        "case": validation_case,
        "known_case_limitations": [],
        "strict_text_match_enabled": True,
        "max_new_tokens": args.max_new_tokens,
        "talker_max_new_tokens": args.talker_max_new_tokens,
        "device_map": resolved_device_map,
        "audio_output_replacement_validation": None,
        "results": [],
    }

    logger.info("Running native HF baseline dialogue")
    baseline_result = _run_dialogue_case(
        native_model,
        tokenizer,
        validation_case,
        args.max_new_tokens,
        processor=processor,
    )

    known_case_limitations = _get_known_case_limitations(validation_case, text_meta)
    strict_text_match_enabled = not known_case_limitations
    report["known_case_limitations"] = known_case_limitations
    report["strict_text_match_enabled"] = strict_text_match_enabled
    for limitation in known_case_limitations:
        logger.warning(f"known validation limitation for case={validation_case}: {limitation}")

    logger.info("Replacing HF thinker.generate with text HMONNX")
    original_generate = native_model.thinker.generate
    native_accept_hidden_layer = getattr(getattr(native_model.config, "talker_config", None), "accept_hidden_layer", None)
    native_model.thinker.generate = _build_text_hmonnx_generate_patch(
        native_model.thinker,
        text_meta,
        logger,
        accept_hidden_layer=native_accept_hidden_layer,
    )

    try:
        replacement_result = _run_dialogue_case(
            native_model,
            tokenizer,
            validation_case,
            args.max_new_tokens,
            processor=processor,
        )
        if known_case_limitations:
            report["audio_output_replacement_validation"] = _build_unsupported_audio_validation(
                validation_case,
                baseline_result["rendered_text"],
                known_case_limitations,
            )
        else:
            report["audio_output_replacement_validation"] = _attempt_audio_output(
                native_model,
                tokenizer,
                processor,
                args.max_new_tokens,
                args.talker_max_new_tokens,
                output_dir=work_dir / "text_hmonnx_talker_output",
                case=validation_case,
            )
    finally:
        native_model.thinker.generate = original_generate

    report["results"].append(_build_report_item(validation_item, baseline_result, replacement_result))

    report_path = work_dir / "text_hmonnx_llm_replacement_report.json"
    save_json(report_path, report)
    markdown_report_path = work_dir / "text_hmonnx_llm_replacement_report.md"
    _write_markdown_report(markdown_report_path, report)
    logger.info(f"text HMONNX llm replacement report saved to {report_path}")
    logger.info(f"text HMONNX llm replacement markdown saved to {markdown_report_path}")

    for item in report["results"]:
        logger.info(
            "case=%s | prompt=%s | native=%s | replacement=%s | normalized_exact_match=%s",
            item["case"],
            item["prompt"],
            item["native_hf_output"],
            item["text_hmonnx_replacement_output"],
            item["normalized_exact_match"],
        )

    audio_check = report["audio_output_replacement_validation"]
    if audio_check is not None:
        logger.info(
            "audio_output_replacement case=%s | supported=%s | error_type=%s | error_message=%s",
            audio_check.get("case", validation_case),
            audio_check["supported"],
            audio_check["error_type"],
            audio_check["error_message"],
        )

    mismatch_items = _collect_text_mismatches(report["results"])
    if mismatch_items and not getattr(args, "allow_mismatch_report_only", False) and strict_text_match_enabled:
        mismatch_summaries = [
            f"case={item['case']} prompt={item['prompt']} prefix_tokens={item['common_prefix_token_count']}"
            for item in mismatch_items
        ]
        raise RuntimeError(
            "Text HMONNX replacement mismatch detected: " + "; ".join(mismatch_summaries)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate text HMONNX as a thinker replacement inside HF Qwen3-Omni"
    )
    parser.add_argument("--model", type=str, default="/data01/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni/Qwen3-Omni-30B-A3B-Instruct-XH2a-text-2k-w4a8_ssfp-ahl24-mmpos-fullhidden", help="work_dir that contains qwen3omni text HMONNX meta")
    parser.add_argument(
        "--case",
        type=str,
        default="multimodal",
        choices=["text", "vision", "audio", "multimodal"],
        help="single built-in validation case; supports pure text or multimodal input and will validate both text and audio output",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--talker-max-new-tokens", type=int, default=64)
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument(
        "--allow-mismatch-report-only",
        action="store_true",
        help="Write the report without raising even when replacement text mismatches the HF baseline.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
