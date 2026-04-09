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
import os.path as osp
import os
import time
import sys
import types
from copy import deepcopy
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

    logger.info(
        f"Loading HF model from {hf_model_path} for talker export with device_map={device_map}"
    )
    native_model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        hf_model_path,
        **load_kwargs,
    )
    native_model.eval()
    _patch_inputs_embeds_generation_device(native_model.talker, "talker", logger)
    _patch_inputs_embeds_generation_device(native_model.talker.code_predictor, "talker.code_predictor", logger)
    if hasattr(native_model, "code2wav"):
        _patch_runtime_device_property(native_model.code2wav, "code2wav", logger)
    return native_model

def _capture_talker_inputs(native_model, processor, device, dtype, work_dir, logger):
    """Run a full generate to capture talker.forward inputs, or load from cache."""
    class _TalkerInputsCaptured(RuntimeError):
        pass

    capture_path = work_dir / "talker_model_inputs.pth"
    if capture_path.exists():
        logger.info(f"Loading cached talker inputs from {capture_path}")
        return torch.load(capture_path, map_location="cpu", weights_only=False)

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
        text=text, audio=audios, images=images, videos=videos,
        return_tensors="pt", padding=True, use_audio_in_video=True,
    )
    inputs = inputs.to(device).to(dtype)

    # Monkey-patch talker._validate_model_kwargs to skip custom kwarg validation
    def _skip_validate(self, model_kwargs):
        return

    native_model.talker._validate_model_kwargs = types.MethodType(_skip_validate, native_model.talker)

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
                save_kwargs[k] = deepcopy(v) if isinstance(v, torch.Tensor) else v
            captured.append(save_kwargs)
            raise _TalkerInputsCaptured()
        return original_forward(*args, **kwargs)

    native_model.talker.forward = forward_hook

    try:
        with torch.no_grad():
            native_model.generate(
                **inputs, speaker="Ethan",
                thinker_return_dict_in_generate=True, use_audio_in_video=True,
            )
    except _TalkerInputsCaptured:
        logger.info("Captured first talker forward inputs, stopping generate early")
    finally:
        native_model.talker.forward = original_forward

    if not captured:
        raise RuntimeError("Failed to capture talker inputs — generate produced no talker calls")

    torch.save(captured, capture_path)
    logger.info(f"Captured {len(captured)} talker forward calls, saved to {capture_path}")
    return captured


def _run_talker_dialogue_validation(
    hf_model_path: str,
    work_dir: Path,
    logger,
    meta_info,
    meta_file: Path,
    max_new_tokens: int,
    talker_max_new_tokens: int,
):
    dialogue_artifacts = {
        "talker": {**meta_info, "_root_dir": str(work_dir), "_meta_path": str(meta_file)}
    }
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
    )


def _build_talker_validation_inputs(work_dir: Path, meta_info):
    kv_cache_shape = meta_info["talker_kv_cache"]["shape"]
    num_hidden_layers = meta_info["talker_kv_cache"]["num_decoder_layers"]
    capture_path = work_dir / "talker_model_inputs.pth"
    if capture_path.exists():
        captured = torch.load(capture_path, map_location="cpu", weights_only=False)
        input_sequence_length = captured[0]["inputs_embeds"].shape[1]
        inputs_embeds = captured[0]["inputs_embeds"].to(torch.float16).cpu()
        if inputs_embeds.shape[1] > input_sequence_length:
            inputs_embeds = inputs_embeds[:, :input_sequence_length, :]
    else:
        hidden_size = meta_info["talker_hidden_size"]
        input_sequence_length = meta_info["talker_input_sequence_length"]
        inputs_embeds = torch.zeros(1, input_sequence_length, hidden_size, dtype=torch.float16)

    past_key_caches = [CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)]
    past_value_caches = [CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)]
    past_seq_length_t = torch.tensor([0], dtype=torch.int32)
    current_input_length_t = torch.tensor([int(inputs_embeds.shape[1])], dtype=torch.int32)
    return inputs_embeds, past_key_caches, past_value_caches, past_seq_length_t, current_input_length_t


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
        inputs_embeds, past_key_caches, past_value_caches, past_seq_length_t, current_input_length_t = (
            _build_talker_validation_inputs(work_dir, meta_info)
        )
    else:
        # ---- 1. Load full HF model ----
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        native_model = _load_native_model_for_capture(hf_model_path, logger)
        processor = Qwen3OmniMoeProcessor.from_pretrained(hf_model_path)

        device = next(native_model.parameters()).device
        dtype = next(native_model.parameters()).dtype

        # ---- 2. Capture talker inputs ----
        captured = _capture_talker_inputs(native_model, processor, device, dtype, work_dir, logger)

        # ---- 3. Register wrap modules and wrap talker ----
        from xh_model_zoo.xh_llm.models.qwen3_omni._talker_model import (
            register_wrap_modules as talker_register_wrap_modules,
        )

        talker_register_wrap_modules()

        talker = native_model.talker.to(torch.float16).cpu()

        batch_size = 1
        context_length = args.context_length
        input_sequence_length = captured[0]["inputs_embeds"].shape[1]

        talker_embedding = talker.model.get_input_embeddings()

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
        past_key_caches = [CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)]
        past_value_caches = [CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)]

        inputs_embeds = captured[0]["inputs_embeds"].to(torch.float16).cpu()
        if inputs_embeds.shape[1] > input_sequence_length:
            inputs_embeds = inputs_embeds[:, :input_sequence_length, :]

        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor([int(inputs_embeds.shape[1])], dtype=torch.int32)

        prefill_inputs = (
            inputs_embeds,
            past_seq_length_t,
            current_input_length_t,
            past_key_caches,
            past_value_caches,
        )

        input_names = ["inputs_embeds", "past_seq_length", "current_input_length"]
        for i in range(num_hidden_layers):
            input_names.append(f"past_key_cache_{i}")
        for i in range(num_hidden_layers):
            input_names.append(f"past_value_cache_{i}")
        output_names = ["logits"]

        # ---- 5. Export prefill HMONNX ----
        logger.info(f"Exporting talker prefill to {prefill_file}")
        with TimeProfiler("export_talker_prefill", logger), MemoryTracker("cuda:0", "export_talker_prefill", logger):
            quanted_talker = convert_fx_model_to_quanted_model(
                wrapped_talker, prefill_inputs, target_device, quant_config=quant_config,
            )
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(
                quanted_talker, prefill_inputs, str(prefill_file), compatible_names, output_names,
            )
        logger.info(f"Talker prefill export successful: {prefill_file}")

        # ---- 6. Export decode HMONNX ----
        decode_inputs = (
            inputs_embeds[:, :1, :],
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
            quanted_talker, decode_inputs, str(decode_file), compatible_names, output_names,
        )
        logger.info(f"Talker decode export successful: {decode_file}")

        # ---- 7. Save talker embedding ----
        embed_file = work_dir / "talker_embedding.pt"
        torch.save(talker_embedding.weight, embed_file)
        logger.info(f"Talker embedding saved to {embed_file}")

        # ---- 8. Save meta ----
        meta_info = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "module": "talker_model",
            "model_name": model_name,
            "talker_prefill_onnx": str(prefill_file.relative_to(work_dir)),
            "talker_decode_onnx": str(decode_file.relative_to(work_dir)),
            "talker_embedding_file": str(embed_file.relative_to(work_dir)),
            "talker_kv_cache": {"shape": kv_cache_shape, "num_decoder_layers": num_hidden_layers},
            "talker_hidden_size": int(inputs_embeds.shape[-1]),
            "talker_input_sequence_length": int(input_sequence_length),
        }
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

            session = HMONNXInference(str(prefill_file))
            output = session(
                inputs_embeds,
                past_seq_length_t,
                current_input_length_t,
                *past_key_caches,
                *past_value_caches,
            )
            if isinstance(output, (list, tuple)):
                output = output[0]
            logger.info(f"Talker prefill HMONNX validation passed, output shape: {tuple(output.shape)}")

            session_d = HMONNXInference(str(decode_file))
            output_d = session_d(
                inputs_embeds[:, :1, :],
                past_seq_length_t,
                torch.ones_like(current_input_length_t),
                *past_key_caches,
                *past_value_caches,
            )
            if isinstance(output_d, (list, tuple)):
                output_d = output_d[0]
            logger.info(f"Talker decode HMONNX validation passed, output shape: {tuple(output_d.shape)}")

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
                logger,
                meta_info,
                meta_file,
                max_new_tokens=args.max_new_tokens,
                talker_max_new_tokens=args.talker_max_new_tokens,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-Omni talker LM to HMONNX")
    parser.add_argument("--model", type=str, default="/data02/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni")
    parser.add_argument("--quant-type", default="w8a8h0_sefp")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--valid", action="store_true", default=True, help="validate exported HMONNX")
    parser.add_argument("--no-valid", action="store_false", dest="valid", help="skip validation")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument(
        "--talker-max-new-tokens",
        type=int,
        default=16,
        help="cap talker audio tokens during dialogue validation",
    )
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--phase", choices=["full", "hmonnx-validate", "dialogue-validate"], default="full")
    args = parser.parse_args()
    main(args)
