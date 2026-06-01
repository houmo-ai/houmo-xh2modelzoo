# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Export Qwen3-Omni talker LM to HMONNX — prefill & decode.

This script captures real talker forward inputs by running a full generate,
then uses those inputs to wrap / trace / quant / export the talker module.
"""

import argparse
import os
import os.path as osp
import sys
import time
import types
from copy import deepcopy
from enum import Enum
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _hmonnx_pipeline import (
    _build_safe_validation_max_memory,
    _patch_inputs_embeds_generation_device,
    _patch_runtime_device_property,
    _resolve_validation_device_map,
    discover_artifacts,
    run_dialogue_validation,
    save_json,
)


try:
    from _hmonnx_pipeline import release_export_cuda_memory
except ImportError:

    def release_export_cuda_memory(logger=None, label=None):
        return None


from xh_model_zoo.xh_llm.models.base_converter import BaseConverter
from xh_model_zoo.xh_llm.models.builder import wrap_llm_model


from xhquant.api import (  # isort:skip
    CacheTensor,
    Config,
    ConfigDict,
    DeviceType,
    QuantScheme,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)


from xh_model_zoo.utils.memory_tracker import MemoryTracker  # isort:skip
from xh_model_zoo.utils.time_profiler import TimeProfiler  # isort:skip

_TALKER_DIALOGUE_PREFILL_HEADROOM = 32
_DEFAULT_LONG_DIALOGUE_PROMPT = (
    "What can you see and hear? Please answer in four complete sentences, "
    "with enough detail to make the synthesized speech noticeably longer."
)

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


def _clone_capture_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return deepcopy(value)


def _compute_talker_prefill_static_length(captured_input_sequence_length: int, context_length: int) -> int:
    captured_input_sequence_length = int(captured_input_sequence_length)
    context_length = int(context_length)
    return min(context_length, captured_input_sequence_length + _TALKER_DIALOGUE_PREFILL_HEADROOM)


def _pad_prefill_tensor(tensor: torch.Tensor, target_seq_len: int, fill_value: float = 0.0) -> torch.Tensor:
    current_seq_len = int(tensor.shape[1])
    if current_seq_len >= target_seq_len:
        return tensor[:, :target_seq_len, ...]

    pad_shape = (tensor.shape[0], target_seq_len - current_seq_len, *tensor.shape[2:])
    pad = torch.full(pad_shape, fill_value, dtype=tensor.dtype)
    return torch.cat([tensor, pad], dim=1)


def _load_first_talker_capture(work_dir: Path):
    capture_path = work_dir / "talker_model_inputs.pth"
    if not capture_path.exists():
        return None
    captured = torch.load(capture_path, map_location="cpu", weights_only=False)
    if not captured:
        return None
    return captured[0]


def _build_prefill_fused_inputs(captured_entry, inputs_embeds: torch.Tensor, meta_info):
    seq_len = int(inputs_embeds.shape[1])
    batch = int(inputs_embeds.shape[0])
    hidden_state_size = int(
        meta_info.get(
            "talker_hidden_state_size",
            meta_info.get("talker_thinker_hidden_size", inputs_embeds.shape[-1]),
        )
    )

    if captured_entry is not None:
        hidden_state = captured_entry.get("hidden_state")
        role_mask = captured_entry.get("role_mask")
        bypass_embeds = captured_entry.get("bypass_embeds")
        bypass_mask = captured_entry.get("bypass_mask")
        if all(isinstance(item, torch.Tensor) for item in (hidden_state, role_mask, bypass_embeds, bypass_mask)):
            hidden_state = hidden_state.to(torch.float16).cpu()
            role_mask = role_mask.to(torch.float16).cpu()
            bypass_embeds = bypass_embeds.to(torch.float16).cpu()
            bypass_mask = bypass_mask.to(torch.float16).cpu()
            return (
                _pad_prefill_tensor(hidden_state, seq_len, fill_value=0.0),
                _pad_prefill_tensor(role_mask, seq_len, fill_value=0.0),
                _pad_prefill_tensor(bypass_embeds, seq_len, fill_value=0.0),
                _pad_prefill_tensor(bypass_mask, seq_len, fill_value=0.0),
            )

    return (
        torch.zeros(batch, seq_len, hidden_state_size, dtype=torch.float16),
        torch.zeros(batch, seq_len, 1, dtype=torch.float16),
        inputs_embeds,
        torch.ones(batch, seq_len, 1, dtype=torch.float16),
    )


def _load_native_model_for_capture(hf_model_path: str, logger):
    from transformers import Qwen3OmniMoeForConditionalGeneration

    device_map = _resolve_validation_device_map("auto", logger)
    max_memory = None
    if device_map == "auto":
        max_memory = _build_safe_validation_max_memory(logger)

    load_kwargs = dict(
        torch_dtype=torch.float16,
        device_map=device_map,
        attn_implementation="eager",
        trust_remote_code=True,
    )
    if max_memory is not None:
        load_kwargs["max_memory"] = max_memory

    logger.info(f"Loading HF model from {hf_model_path} for talker export with device_map={device_map}")
    native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        hf_model_path,
        **load_kwargs,
    )
    _force_eager_moe_implementation(native_model, logger)
    native_model.eval()
    _patch_inputs_embeds_generation_device(native_model.talker, "talker", logger)
    _patch_inputs_embeds_generation_device(native_model.talker.code_predictor, "talker.code_predictor", logger)
    if hasattr(native_model, "code2wav"):
        _patch_runtime_device_property(native_model.code2wav, "code2wav", logger)
    return native_model


def _ensure_mistral_common_reasoning_effort():
    try:
        import mistral_common.protocol.instruct.request as request_module
    except ImportError:
        return

    if hasattr(request_module, "ReasoningEffort"):
        return

    class ReasoningEffort(str, Enum):
        none = "none"
        high = "high"

    request_module.ReasoningEffort = ReasoningEffort


def _force_eager_moe_implementation(module, logger=None):
    visited_configs = set()
    updated = 0

    def _visit_config(config):
        nonlocal updated
        if config is None:
            return
        config_id = id(config)
        if config_id in visited_configs:
            return
        visited_configs.add(config_id)

        if hasattr(config, "_experts_implementation") and config._experts_implementation != "eager":
            config._experts_implementation = "eager"
            updated += 1

        config_dict = getattr(config, "__dict__", None)
        if not isinstance(config_dict, dict):
            return
        for value in config_dict.values():
            if hasattr(value, "__dict__"):
                _visit_config(value)

    _visit_config(getattr(module, "config", None))
    for submodule in module.modules():
        _visit_config(getattr(submodule, "config", None))

    if logger is not None and updated:
        logger.info(f"forced {updated} config nodes to use eager MoE experts")


def _capture_talker_inputs(native_model, processor, device, dtype, work_dir, logger):
    """Run a full generate to capture talker.forward inputs, or load from cache."""

    capture_contract_version = 3

    class _TalkerInputsCaptured(RuntimeError):
        pass

    capture_path = work_dir / "talker_model_inputs.pth"
    if capture_path.exists():
        logger.info(f"Loading cached talker inputs from {capture_path}")
        cached = torch.load(capture_path, map_location="cpu", weights_only=False)
        cached_entry = cached[0] if cached else None
        if (
            isinstance(cached_entry, dict)
            and int(cached_entry.get("capture_contract_version", 0)) >= capture_contract_version
        ):
            return cached
        logger.info("Cached talker inputs use a stale capture contract, recapturing with current processor kwargs")

    logger.info("Running full generate to capture talker inputs ...")
    image_path = str(SCRIPT_DIR / "data" / "cars.jpg")
    audio_path = str(SCRIPT_DIR / "data" / "cough.wav")
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "audio", "audio": audio_path},
                {"type": "text", "text": "What can you see and hear? Answer in one short sentence."},
            ],
        },
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        seconds_per_chunk=2.0,
        position_id_per_seconds=13,
        use_audio_in_video=True,
    )
    inputs = inputs.to(device).to(dtype)

    # Monkey-patch talker._validate_model_kwargs to skip custom kwarg validation
    def _skip_validate(self, model_kwargs):
        return

    native_model.talker._validate_model_kwargs = types.MethodType(_skip_validate, native_model.talker)

    capture_ctx = {"generate_kwargs": {}, "segments": []}

    original_generate = native_model.talker.generate
    original_get_user_parts = native_model._get_talker_user_parts
    original_get_assistant_parts = native_model._get_talker_assistant_parts

    def generate_hook(self, *args, **kwargs):
        if not capture_ctx["generate_kwargs"]:
            for key in ("inputs_embeds", "trailing_text_hidden", "tts_pad_embed", "talker_input_ids"):
                if key in kwargs:
                    capture_ctx["generate_kwargs"][key] = _clone_capture_value(kwargs[key])
        return original_generate(*args, **kwargs)

    def user_parts_hook(self, im_start_index, segment_end_index, multimodal_mask, thinker_hidden, thinker_embed):
        user_talker_part = original_get_user_parts(
            im_start_index,
            segment_end_index,
            multimodal_mask,
            thinker_hidden,
            thinker_embed,
        )
        user_mm_mask = multimodal_mask[:, im_start_index:segment_end_index]
        user_source = thinker_embed[:, im_start_index:segment_end_index].clone()
        if user_mm_mask.any():
            user_source[user_mm_mask] = thinker_hidden[:, im_start_index:segment_end_index][user_mm_mask]
        capture_ctx["segments"].append(
            {
                "hidden_state": _clone_capture_value(user_source),
                "role_mask": _clone_capture_value((~user_mm_mask).unsqueeze(-1).to(user_talker_part.dtype)),
                "bypass_embeds": _clone_capture_value(torch.zeros_like(user_talker_part)),
                "bypass_mask": _clone_capture_value(
                    torch.zeros(
                        *user_talker_part.shape[:2], 1, dtype=user_talker_part.dtype, device=user_talker_part.device
                    )
                ),
            }
        )
        return user_talker_part

    def assistant_parts_hook(
        self,
        im_start_index,
        segment_end_index,
        speaker_id,
        thinker_embed,
        tts_pad_embed,
        tts_bos_embed,
        tts_eos_embed,
    ):
        input_embeds, input_ids, trailing_text_hidden = original_get_assistant_parts(
            im_start_index,
            segment_end_index,
            speaker_id,
            thinker_embed,
            tts_pad_embed,
            tts_bos_embed,
            tts_eos_embed,
        )
        assistant_source = torch.zeros(
            input_embeds.shape[0],
            input_embeds.shape[1],
            thinker_embed.shape[-1],
            dtype=thinker_embed.dtype,
            device=thinker_embed.device,
        )
        assistant_role_mask = torch.ones(
            input_embeds.shape[0], input_embeds.shape[1], 1, dtype=input_embeds.dtype, device=input_embeds.device
        )
        assistant_bypass_embeds = input_embeds.clone()
        assistant_bypass_mask = torch.ones(
            input_embeds.shape[0], input_embeds.shape[1], 1, dtype=input_embeds.dtype, device=input_embeds.device
        )

        projected_prefix = min(3, max(segment_end_index - im_start_index, 0))
        if projected_prefix > 0:
            assistant_source[:, :projected_prefix, :] = thinker_embed[
                :, im_start_index : im_start_index + projected_prefix, :
            ]
            assistant_bypass_embeds[:, :projected_prefix, :] = 0
            assistant_bypass_mask[:, :projected_prefix, :] = 0

        capture_ctx["segments"].append(
            {
                "hidden_state": _clone_capture_value(assistant_source),
                "role_mask": _clone_capture_value(assistant_role_mask),
                "bypass_embeds": _clone_capture_value(assistant_bypass_embeds),
                "bypass_mask": _clone_capture_value(assistant_bypass_mask),
                "trailing_text_hidden": _clone_capture_value(trailing_text_hidden),
                "assistant_input_ids": _clone_capture_value(input_ids),
            }
        )
        return input_embeds, input_ids, trailing_text_hidden

    native_model.talker.generate = types.MethodType(generate_hook, native_model.talker)
    native_model._get_talker_user_parts = types.MethodType(user_parts_hook, native_model)
    native_model._get_talker_assistant_parts = types.MethodType(assistant_parts_hook, native_model)

    # Hook talker.forward to capture inputs
    original_forward = native_model.talker.forward
    captured = []

    def forward_hook(*args, **kwargs):
        # Construct attention_mask if not provided (talker.generate doesn't pass it)
        if kwargs.get("attention_mask") is None and "inputs_embeds" in kwargs:
            ie = kwargs["inputs_embeds"]
            kwargs["attention_mask"] = torch.ones(ie.shape[:2], dtype=torch.long, device=ie.device)
        # Only capture the first (prefill) call; skip large past_key_values
        if not captured:
            save_kwargs = {}
            for k, v in kwargs.items():
                if k in ("past_key_values", "cache_position"):
                    continue
                save_kwargs[k] = _clone_capture_value(v)

            for key, value in capture_ctx["generate_kwargs"].items():
                save_kwargs.setdefault(key, value)

            if capture_ctx["segments"]:
                hidden_state = torch.cat([segment["hidden_state"] for segment in capture_ctx["segments"]], dim=1)
                role_mask = torch.cat([segment["role_mask"] for segment in capture_ctx["segments"]], dim=1)
                bypass_embeds = torch.cat([segment["bypass_embeds"] for segment in capture_ctx["segments"]], dim=1)
                bypass_mask = torch.cat([segment["bypass_mask"] for segment in capture_ctx["segments"]], dim=1)
                if int(hidden_state.shape[1]) == int(save_kwargs["inputs_embeds"].shape[1]):
                    save_kwargs["hidden_state"] = hidden_state
                    save_kwargs["role_mask"] = role_mask
                    save_kwargs["bypass_embeds"] = bypass_embeds
                    save_kwargs["bypass_mask"] = bypass_mask

                trailing_text_hidden = next(
                    (
                        segment.get("trailing_text_hidden")
                        for segment in capture_ctx["segments"]
                        if isinstance(segment.get("trailing_text_hidden"), torch.Tensor)
                    ),
                    None,
                )
                if trailing_text_hidden is not None:
                    save_kwargs.setdefault("trailing_text_hidden", trailing_text_hidden)

            save_kwargs["capture_contract_version"] = capture_contract_version
            captured.append(save_kwargs)
            raise _TalkerInputsCaptured()
        return original_forward(*args, **kwargs)

    native_model.talker.forward = forward_hook

    try:
        with torch.no_grad():
            native_model.generate(
                **inputs,
                speaker="Ethan",
                thinker_return_dict_in_generate=True,
                use_audio_in_video=True,
            )
    except _TalkerInputsCaptured:
        logger.info("Captured first talker forward inputs, stopping generate early")
    finally:
        native_model.talker.forward = original_forward
        native_model.talker.generate = original_generate
        native_model._get_talker_user_parts = original_get_user_parts
        native_model._get_talker_assistant_parts = original_get_assistant_parts

    if not captured:
        raise RuntimeError("Failed to capture talker inputs — generate produced no talker calls")

    torch.save(captured, capture_path)
    logger.info(f"Captured {len(captured)} talker forward calls, saved to {capture_path}")
    return captured


def _run_talker_dialogue_validation(
    hf_model_path: str,
    work_dir: Path,
    golden_dir: Path,
    logger,
    meta_info,
    meta_file: Path,
    max_new_tokens: int,
    talker_max_new_tokens: int,
    save_golden: bool,
    validation_prompt: str | None = None,
):
    dialogue_artifacts = {"talker": {**meta_info, "_root_dir": str(work_dir), "_meta_path": str(meta_file)}}
    sibling_artifacts = discover_artifacts(work_dir.parent)
    if "talker_prediction" in sibling_artifacts:
        dialogue_artifacts["talker_prediction"] = sibling_artifacts["talker_prediction"]
    return run_dialogue_validation(
        hf_model_path,
        work_dir,
        logger,
        case="multimodal",
        max_new_tokens=max_new_tokens,
        talker_max_new_tokens=talker_max_new_tokens,
        artifacts=dialogue_artifacts,
        report_name="talker_dialogue_validation.json",
        output_prefix="talker_dialogue",
        save_golden=save_golden,
        golden_dir=golden_dir,
        validation_prompt=validation_prompt,
    )


def _build_talker_validation_inputs(work_dir: Path, meta_info):
    kv_cache_shape = meta_info["talker_kv_cache"]["shape"]
    num_hidden_layers = meta_info["talker_kv_cache"]["num_decoder_layers"]
    static_input_sequence_length = int(meta_info["talker_input_sequence_length"])
    captured_entry = _load_first_talker_capture(work_dir)
    if captured_entry is not None:
        actual_input_sequence_length = int(captured_entry["inputs_embeds"].shape[1])
        inputs_embeds = captured_entry["inputs_embeds"].to(torch.float16).cpu()
        if inputs_embeds.shape[1] > actual_input_sequence_length:
            inputs_embeds = inputs_embeds[:, :actual_input_sequence_length, :]
        inputs_embeds = _pad_prefill_tensor(inputs_embeds, static_input_sequence_length, fill_value=0.0)
    else:
        hidden_size = meta_info["talker_hidden_size"]
        actual_input_sequence_length = static_input_sequence_length
        inputs_embeds = torch.zeros(1, static_input_sequence_length, hidden_size, dtype=torch.float16)

    past_key_caches = [CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)]
    past_value_caches = [
        CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)
    ]
    past_seq_length_t = torch.tensor([0], dtype=torch.int32)
    current_input_length_t = torch.tensor([actual_input_sequence_length], dtype=torch.int32)
    return inputs_embeds, captured_entry, past_key_caches, past_value_caches, past_seq_length_t, current_input_length_t


def _reexec_with_phase(phase: str):
    script_path = str(Path(__file__).resolve())
    passthrough_args = []
    skip_next = False

    for arg in sys.argv[1:]:
        if skip_next:
            skip_next = False
            continue
        if arg == "--phase":
            skip_next = True
            continue
        if arg.startswith("--phase="):
            continue
        passthrough_args.append(arg)

    os.execv(sys.executable, [sys.executable, script_path, *passthrough_args, "--phase", phase])


def main(args):
    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    target_device = DeviceType.XH2a
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)
    quant_config = ConfigDict(create_quant_config(quant_scheme))

    prefix = f"{model_name}-{target_device}-talker-{quant_type}"
    work_dir = Path(args.work_dir) / prefix
    golden_root = Path(args.golden_root)
    if not golden_root.is_absolute():
        golden_root = (SCRIPT_DIR.parents[2] / golden_root).resolve()
    golden_dir = golden_root / prefix / "golden"
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / "convert.log"
    xhquant_init(log_file, debug=args.debug, file_mode="a" if args.phase != "full" else "w")
    logger = get_root_logger()
    native_model = None
    processor = None
    captured = None
    talker = None
    wrapped_talker = None
    quanted_talker = None

    hmonnx_dir = work_dir / "hmonnx"
    hmonnx_dir.mkdir(exist_ok=True, parents=True)
    prefill_file = hmonnx_dir / f"{model_name}-talker_prefill.onnx"
    decode_file = hmonnx_dir / f"{model_name}-talker_decode.onnx"
    meta_file = work_dir / "meta_talker.json"

    # Check if artifacts already exist — skip export if so
    import json

    artifacts_exist = prefill_file.exists() and decode_file.exists() and meta_file.exists()

    if artifacts_exist:
        logger.info("Talker HMONNX artifacts already exist, skipping export")
        with open(meta_file) as f:
            meta_info = json.load(f)
        takeover_ready = meta_info.get("artifact_contract_version", 1) >= 4 and meta_info.get("output_names") == [
            "logits",
            "hidden_states",
        ]
        if not takeover_ready:
            logger.info("Talker artifacts use a stale prefill contract, rebuilding for HMONNX takeover")
            artifacts_exist = False

    if artifacts_exist:
        inputs_embeds, captured_entry, past_key_caches, past_value_caches, past_seq_length_t, current_input_length_t = (
            _build_talker_validation_inputs(work_dir, meta_info)
        )
    else:
        # ---- 1. Load full HF model ----
        from transformers import Qwen3OmniMoeProcessor

        native_model = _load_native_model_for_capture(hf_model_path, logger)
        _ensure_mistral_common_reasoning_effort()
        processor = Qwen3OmniMoeProcessor.from_pretrained(hf_model_path)

        device = next(native_model.parameters()).device
        dtype = next(native_model.parameters()).dtype

        # ---- 2. Capture talker inputs ----
        captured = _capture_talker_inputs(native_model, processor, device, dtype, work_dir, logger)
        captured_entry = captured[0]

        # ---- 3. Register wrap modules and wrap talker ----
        from xh_model_zoo.xh_llm.models.qwen3_omni._talker_model import (
            register_wrap_modules as talker_register_wrap_modules,
        )

        talker_register_wrap_modules()

        talker = native_model.talker.to(torch.float16).cpu()

        batch_size = 1
        context_length = args.context_length
        actual_input_sequence_length = int(captured[0]["inputs_embeds"].shape[1])
        input_sequence_length = _compute_talker_prefill_static_length(actual_input_sequence_length, context_length)

        talker_embedding = talker.model.get_input_embeddings()

        # Snapshot the projection head input-dim BEFORE wrapping (FX
        # conversion may rewrite the module, and we need this number for
        # meta info).
        projection_in_features = int(talker.hidden_projection.linear_fc1.in_features)

        wrap_cfg = Config(
            dict(
                batch_size=batch_size,
                max_sequence_length=context_length,
                input_sequence_length=int(input_sequence_length),
                use_cache=True,
                num_logits_to_keep=1,
                kv_cache=dict(cache_axis=2),
            )
        )

        wrapped_talker = wrap_llm_model(talker, wrap_cfg)

        # ---- 4. Setup KV cache and inputs ----
        num_hidden_layers = wrapped_talker.model.config.num_hidden_layers
        head_dim = wrapped_talker.model.layers[0].self_attn.head_dim
        num_key_value_heads = wrapped_talker.model.config.num_key_value_heads

        kv_cache_shape = [1, num_key_value_heads, context_length, head_dim]
        past_key_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)
        ]
        past_value_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)
        ]

        inputs_embeds = captured[0]["inputs_embeds"].to(torch.float16).cpu()
        if inputs_embeds.shape[1] > actual_input_sequence_length:
            inputs_embeds = inputs_embeds[:, :actual_input_sequence_length, :]
        inputs_embeds = _pad_prefill_tensor(inputs_embeds, input_sequence_length, fill_value=0.0)

        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor([actual_input_sequence_length], dtype=torch.int32)

        # Build calibration inputs for the fused graph (both projection heads
        # baked into prefill/decode). Use the captured HF talker input mix so
        # projection, bypass, and role masks follow the standard Qwen3-Omni
        # prefill path instead of calibrating mostly on an all-bypass shadow.
        source_batch = int(inputs_embeds.shape[0])
        thinker_hs = int(talker.hidden_projection.linear_fc1.in_features)

        source_prefill, role_mask_prefill, bypass_embeds_prefill, bypass_mask_prefill = _build_prefill_fused_inputs(
            captured_entry,
            inputs_embeds,
            {
                "talker_hidden_state_size": thinker_hs,
                "talker_thinker_hidden_size": thinker_hs,
            },
        )

        prefill_inputs = (
            source_prefill,
            role_mask_prefill,
            bypass_embeds_prefill,
            bypass_mask_prefill,
            past_seq_length_t,
            current_input_length_t,
            past_key_caches,
            past_value_caches,
        )

        input_names = [
            "hidden_state",
            "role_mask",
            "bypass_embeds",
            "bypass_mask",
            "past_seq_length",
            "current_input_length",
        ]
        for i in range(num_hidden_layers):
            input_names.append(f"past_key_cache_{i}")
        for i in range(num_hidden_layers):
            input_names.append(f"past_value_cache_{i}")
        output_names = ["logits", "hidden_states"]

        # ---- 5. Export prefill HMONNX ----
        logger.info(f"Exporting talker prefill to {prefill_file}")
        with TimeProfiler("export_talker_prefill", logger), MemoryTracker("cuda:0", "export_talker_prefill", logger):
            quanted_talker = convert_fx_model_to_quanted_model(
                wrapped_talker,
                prefill_inputs,
                target_device,
                quant_config=quant_config,
            )
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(
                quanted_talker,
                prefill_inputs,
                str(prefill_file),
                compatible_names,
                output_names,
            )
        logger.info(f"Talker prefill export successful: {prefill_file}")

        # ---- 6. Export decode HMONNX ----
        # Decode uses seq_len=1. Projection heads still live in the graph
        # (baked in) but the bypass path is active: bypass_mask=1 feeds the
        # pre-projected codec embedding straight into the trunk.
        source_decode = torch.zeros(source_batch, 1, thinker_hs, dtype=torch.float16)
        role_mask_decode = torch.zeros(source_batch, 1, 1, dtype=torch.float16)
        bypass_embeds_decode = inputs_embeds[:, :1, :]
        bypass_mask_decode = torch.ones(source_batch, 1, 1, dtype=torch.float16)

        decode_inputs = (
            source_decode,
            role_mask_decode,
            bypass_embeds_decode,
            bypass_mask_decode,
            past_seq_length_t,
            torch.ones_like(current_input_length_t),
            past_key_caches,
            past_value_caches,
        )
        wrap_cfg.input_sequence_length = 1
        quanted_talker.update_cfg(wrap_cfg)

        logger.info(f"Exporting talker decode to {decode_file}")
        compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(
            quanted_talker,
            decode_inputs,
            str(decode_file),
            compatible_names,
            output_names,
        )
        logger.info(f"Talker decode export successful: {decode_file}")

        # ---- 7. Save talker embedding ----
        embed_file = work_dir / "talker_embedding.pt"
        torch.save(talker_embedding.weight, embed_file)
        logger.info(f"Talker embedding saved to {embed_file}")

        # ---- 8. Save meta ----
        meta_info = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "artifact_contract_version": 4,
            "module": "talker_model",
            "model_name": model_name,
            "talker_prefill_onnx": str(prefill_file.relative_to(work_dir)),
            "talker_decode_onnx": str(decode_file.relative_to(work_dir)),
            "output_names": output_names,
            "talker_projection_in_features": int(projection_in_features),
            "talker_embedding_file": str(embed_file.relative_to(work_dir)),
            "talker_kv_cache": {"shape": kv_cache_shape, "num_decoder_layers": num_hidden_layers},
            "talker_hidden_size": int(inputs_embeds.shape[-1]),
            "talker_hidden_state_size": int(thinker_hs),
            "talker_thinker_hidden_size": int(thinker_hs),
            "talker_input_sequence_length": int(input_sequence_length),
            "talker_prefill_guidance_inputs": ["hidden_state", "role_mask", "bypass_embeds", "bypass_mask"],
            "fused_projection": True,
        }
        trailing_text_hidden = captured_entry.get("trailing_text_hidden") if captured_entry is not None else None
        if isinstance(trailing_text_hidden, torch.Tensor):
            meta_info["talker_style_guidance_hidden_size"] = int(trailing_text_hidden.shape[-1])
        save_json(meta_file, meta_info)
        logger.info(f"Talker export complete. Meta saved to {meta_file}")

    quanted_talker = None
    wrapped_talker = None
    talker = None
    captured = None
    processor = None
    native_model = None
    release_export_cuda_memory(logger, "talker export")

    # ---- 9. Optional validation ----
    if args.valid:
        if args.phase == "full":
            logger.info("Restarting process for talker HMONNX validation after export cleanup")
            _reexec_with_phase("hmonnx-validate")

        if args.phase == "hmonnx-validate":
            logger.info("Validating talker HMONNX ...")
            from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

            # Build fused-graph inputs for validation. Prefer the real captured
            # hidden_state mix so the style-guidance path is exercised; fall
            # back to bypass-only when legacy captures do not contain it.
            prefill_batch = int(inputs_embeds.shape[0])
            prefill_thinker_hs = int(
                meta_info.get(
                    "talker_hidden_state_size",
                    meta_info.get("talker_thinker_hidden_size", inputs_embeds.shape[-1]),
                )
            )
            prefill_hidden_state, prefill_role_mask, prefill_bypass_embeds, prefill_bypass_mask = (
                _build_prefill_fused_inputs(captured_entry, inputs_embeds, meta_info)
            )

            session = HMONNXInference(str(prefill_file))
            output = session(
                prefill_hidden_state,
                prefill_role_mask,
                prefill_bypass_embeds,
                prefill_bypass_mask,
                past_seq_length_t,
                current_input_length_t,
                *past_key_caches,
                *past_value_caches,
            )
            outputs = list(output) if isinstance(output, (list, tuple)) else [output]
            logger.info(
                "Talker prefill HMONNX validation passed, output shapes: "
                + ", ".join(str(tuple(out.shape)) for out in outputs)
            )

            decode_source = torch.zeros(prefill_batch, 1, prefill_thinker_hs, dtype=torch.float16)
            decode_role_mask = torch.zeros(prefill_batch, 1, 1, dtype=torch.float16)
            decode_bypass_mask = torch.ones(prefill_batch, 1, 1, dtype=torch.float16)

            session_d = HMONNXInference(str(decode_file))
            output_d = session_d(
                decode_source,
                decode_role_mask,
                inputs_embeds[:, :1, :],
                decode_bypass_mask,
                past_seq_length_t,
                torch.ones_like(current_input_length_t),
                *past_key_caches,
                *past_value_caches,
            )
            outputs_d = list(output_d) if isinstance(output_d, (list, tuple)) else [output_d]
            logger.info(
                "Talker decode HMONNX validation passed, output shapes: "
                + ", ".join(str(tuple(out.shape)) for out in outputs_d)
            )

            session = None
            session_d = None
            output = None
            output_d = None
            inputs_embeds = None
            past_key_caches = None
            past_value_caches = None
            past_seq_length_t = None
            current_input_length_t = None
            release_export_cuda_memory(logger, "talker hmonnx validation")
            logger.info("Restarting process for talker dialogue validation after direct HMONNX validation")
            _reexec_with_phase("dialogue-validate")

        if args.phase == "dialogue-validate":
            _run_talker_dialogue_validation(
                hf_model_path,
                work_dir,
                golden_dir,
                logger,
                meta_info,
                meta_file,
                max_new_tokens=args.max_new_tokens,
                talker_max_new_tokens=args.talker_max_new_tokens,
                save_golden=args.save_golden,
                validation_prompt=args.validation_prompt,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-Omni talker LM to HMONNX")
    parser.add_argument("--model", type=str, default="/data01/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni")
    parser.add_argument("--quant-type", default="w8a8h0_sefp")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--valid", action="store_true", default=True, help="validate exported HMONNX")
    parser.add_argument("--no-valid", action="store_false", dest="valid", help="skip validation")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--talker-max-new-tokens",
        type=int,
        default=96,
        help="cap talker audio tokens during dialogue validation; increase this for longer generated speech",
    )
    parser.add_argument(
        "--validation-prompt",
        type=str,
        default=_DEFAULT_LONG_DIALOGUE_PROMPT,
        help="prompt used by dialogue validation; use a longer prompt/request to produce longer speech",
    )
    parser.add_argument("--golden-root", type=str, default="work_dirs/qwen3omni_no_projection")
    parser.add_argument("--save-golden", action="store_true", default=True, help="save golden outputs after validation")
    parser.add_argument("--no-save-golden", action="store_false", dest="save_golden", help="skip golden output save")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--phase", choices=["full", "hmonnx-validate", "dialogue-validate"], default="full")
    args = parser.parse_args()
    main(args)
