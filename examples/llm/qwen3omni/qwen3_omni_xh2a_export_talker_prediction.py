# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Export Qwen3-Omni talker code predictor to HMONNX — prefill & decode.

The predictor stays a two-graph module for prefill/decode, while auxiliary
codec embeddings and lm_heads are packaged into one asset bundle instead of
being emitted as many loose files.
"""

import argparse
import json
import os
import os.path as osp
import sys
import time
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
    from _hmonnx_pipeline import _ensure_mistral_common_reasoning_effort
except ImportError:

    def _ensure_mistral_common_reasoning_effort():
        try:
            import mistral_common.protocol.instruct.request as request_module
        except ImportError:
            return

        if hasattr(request_module, "ReasoningEffort"):
            return

        from enum import Enum

        class ReasoningEffort(str, Enum):
            none = "none"
            high = "high"

        request_module.ReasoningEffort = ReasoningEffort


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


def _clone_capture_value(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    return deepcopy(value)


def _capture_predictor_inputs(native_model, processor, device, dtype, work_dir, logger):
    """Run a full generate to capture code_predictor.model.forward inputs, or load from cache."""

    capture_contract_version = 2

    class _PredictorInputsCaptured(RuntimeError):
        pass

    capture_path = work_dir / "talker_prediction_inputs.pth"
    if capture_path.exists():
        logger.info(f"Loading cached predictor inputs from {capture_path}")
        cached = torch.load(capture_path, map_location="cpu", weights_only=False)
        cached_entry = cached[0] if cached else None
        if (
            isinstance(cached_entry, dict)
            and int(cached_entry.get("capture_contract_version", 0)) >= capture_contract_version
        ):
            return cached
        logger.info("Cached predictor inputs use a stale capture contract, recapturing with current processor kwargs")

    logger.info("Running full generate to capture predictor inputs ...")
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

    # Skip talker kwarg validation
    def _skip_validate(self, model_kwargs):
        return

    native_model.talker.code_predictor.model._validate_model_kwargs = types.MethodType(
        _skip_validate, native_model.talker.code_predictor.model
    )

    original_forward = native_model.talker.code_predictor.model.forward
    captured = []

    def forward_hook(*args, **kwargs):
        if not captured:
            save_kwargs = {k: _clone_capture_value(v) for k, v in kwargs.items()}
            save_kwargs["capture_contract_version"] = capture_contract_version
            captured.append(save_kwargs)
            raise _PredictorInputsCaptured()
        return original_forward(*args, **kwargs)

    native_model.talker.code_predictor.model.forward = forward_hook

    try:
        with torch.no_grad():
            native_model.generate(
                **inputs,
                speaker="Ethan",
                thinker_return_dict_in_generate=True,
                use_audio_in_video=True,
            )
    except _PredictorInputsCaptured:
        logger.info("Captured first predictor forward inputs, stopping generate early")
    finally:
        native_model.talker.code_predictor.model.forward = original_forward

    if not captured:
        raise RuntimeError("Failed to capture predictor inputs")

    torch.save(captured, capture_path)
    logger.info(f"Captured {len(captured)} predictor forward calls, saved to {capture_path}")
    return captured


def _save_predictor_assets(native_model, asset_file: Path, logger):
    codec_embeddings = [emb.weight.detach().cpu() for emb in native_model.talker.code_predictor.get_input_embeddings()]
    lm_head_weights = [head.weight.detach().cpu() for head in native_model.talker.code_predictor.lm_head]
    asset_payload = {
        "codec_embeddings": codec_embeddings,
        "lm_head_weights": lm_head_weights,
        "num_codec_embeddings": len(codec_embeddings),
        "num_lm_heads": len(lm_head_weights),
    }
    torch.save(asset_payload, asset_file)
    logger.info(f"Saved talker prediction asset bundle to {asset_file}")
    return asset_payload


def _build_predictor_validation_inputs(work_dir: Path, meta_info):
    kv_cache_shape = meta_info["talker_prediction_kv_cache"]["shape"]
    num_hidden_layers = meta_info["talker_prediction_kv_cache"]["num_decoder_layers"]
    capture_path = work_dir / "talker_prediction_inputs.pth"

    if capture_path.exists():
        captured = torch.load(capture_path, map_location="cpu", weights_only=False)
        input_sequence_length = captured[0]["inputs_embeds"].shape[1]
        inputs_embeds = captured[0]["inputs_embeds"].to(torch.float16).cpu()
        if inputs_embeds.shape[1] > input_sequence_length:
            inputs_embeds = inputs_embeds[:, :input_sequence_length, :]
    else:
        hidden_size = meta_info["talker_prediction_hidden_size"]
        input_sequence_length = meta_info["talker_prediction_input_sequence_length"]
        inputs_embeds = torch.zeros(1, input_sequence_length, hidden_size, dtype=torch.float16)

    past_key_caches = [CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)]
    past_value_caches = [
        CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)
    ]
    past_seq_length_t = torch.tensor([0], dtype=torch.int32)
    current_input_length_t = torch.tensor([int(inputs_embeds.shape[1])], dtype=torch.int32)
    return inputs_embeds, past_key_caches, past_value_caches, past_seq_length_t, current_input_length_t


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

    logger.info(f"Loading HF model from {hf_model_path} for talker prediction export with device_map={device_map}")
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

    prefix = f"{model_name}-{target_device}-talker_prediction-{quant_type}"
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
    wrapped_model = None
    quanted_model = None

    hmonnx_dir = work_dir / "hmonnx"
    hmonnx_dir.mkdir(exist_ok=True, parents=True)
    prefill_file = hmonnx_dir / f"{model_name}-talker_prediction_prefill.onnx"
    decode_file = hmonnx_dir / f"{model_name}-talker_prediction_decode.onnx"
    meta_file = work_dir / "meta_talker_prediction.json"
    asset_file = work_dir / "talker_prediction_assets.pt"

    # Check if artifacts already exist — skip export if so
    artifacts_exist = prefill_file.exists() and decode_file.exists() and meta_file.exists()
    meta_info = None

    if artifacts_exist:
        logger.info("Talker prediction HMONNX artifacts already exist, skipping export")
        with open(meta_file) as f:
            meta_info = json.load(f)
        legacy_layout = "codec_embedding_dir" in meta_info or "lm_head_dir" in meta_info
        new_layout_ready = "talker_prediction_assets_file" in meta_info and asset_file.exists()
        takeover_ready = meta_info.get("artifact_contract_version", 1) >= 4 and meta_info.get("output_names") == [
            "logits",
            "hidden_states",
        ]
        if legacy_layout and not new_layout_ready:
            logger.info("Legacy scattered talker prediction artifacts detected, rebuilding consolidated export")
            artifacts_exist = False
        elif "talker_prediction_assets_file" in meta_info and not asset_file.exists():
            logger.info("Talker prediction meta points to missing asset bundle, rebuilding consolidated export")
            artifacts_exist = False
        elif not takeover_ready:
            logger.info("Talker prediction artifacts use legacy single-output contract, rebuilding for HMONNX takeover")
            artifacts_exist = False

    if artifacts_exist:
        inputs_embeds, past_key_caches, past_value_caches, past_seq_length_t, current_input_length_t = (
            _build_predictor_validation_inputs(work_dir, meta_info)
        )
    else:
        _ensure_mistral_common_reasoning_effort()
        from transformers import Qwen3OmniMoeProcessor

        native_model = _load_native_model_for_capture(hf_model_path, logger)
        processor = Qwen3OmniMoeProcessor.from_pretrained(hf_model_path)

        device = next(native_model.parameters()).device
        dtype = next(native_model.parameters()).dtype

        # ---- 2. Package predictor-side assets into one bundle ----
        asset_payload = _save_predictor_assets(native_model, asset_file, logger)

        # ---- 3. Capture predictor inputs ----
        captured = _capture_predictor_inputs(native_model, processor, device, dtype, work_dir, logger)

        # ---- 4. Register wrap modules and wrap predictor ----
        from xh_model_zoo.xh_llm.models.qwen3_omni._talker_prediction import (
            register_wrap_modules as pred_register_wrap_modules,
        )

        pred_register_wrap_modules()

        code_predictor = native_model.talker.code_predictor.to(torch.float16).cpu()
        processor = None
        native_model = None
        release_export_cuda_memory(logger, "talker prediction capture")

        batch_size = 1
        context_length = args.context_length
        input_sequence_length = captured[0]["inputs_embeds"].shape[1]

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

        wrapped_model = wrap_llm_model(code_predictor, wrap_cfg)

        # ---- 5. Setup KV cache and inputs ----
        num_hidden_layers = wrapped_model.model.config.num_hidden_layers
        head_dim = wrapped_model.model.layers[0].self_attn.head_dim
        num_key_value_heads = wrapped_model.model.config.num_key_value_heads

        kv_cache_shape = [1, num_key_value_heads, context_length, head_dim]
        past_key_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)
        ]
        past_value_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)) for _ in range(num_hidden_layers)
        ]

        inputs_embeds = captured[0]["inputs_embeds"].to(torch.float16).cpu()
        if inputs_embeds.shape[1] > input_sequence_length:
            inputs_embeds = inputs_embeds[:, :input_sequence_length, :]

        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor([int(inputs_embeds.shape[1])], dtype=torch.int32)

        num_lm_heads = int(asset_payload["num_lm_heads"])
        batch = int(inputs_embeds.shape[0])
        prefill_seq = int(inputs_embeds.shape[1])
        # Match HF predictor prefill: generation_steps = seq_len - 2.
        prefill_step = max(0, min(prefill_seq - 2, num_lm_heads - 1))
        head_mask_prefill = torch.zeros(batch, prefill_seq, num_lm_heads, 1, dtype=torch.float16)
        head_mask_prefill[:, :, prefill_step, 0] = 1.0
        # Decode: pick head 0 as the calibration path (one-hot).
        head_mask_decode = torch.zeros(batch, 1, num_lm_heads, 1, dtype=torch.float16)
        head_mask_decode[0, 0, 0, 0] = 1.0

        prefill_inputs = (
            inputs_embeds,
            head_mask_prefill,
            past_seq_length_t,
            current_input_length_t,
            past_key_caches,
            past_value_caches,
        )

        input_names = ["inputs_embeds", "head_mask", "past_seq_length", "current_input_length"]
        for i in range(num_hidden_layers):
            input_names.append(f"past_key_cache_{i}")
        for i in range(num_hidden_layers):
            input_names.append(f"past_value_cache_{i}")
        output_names = ["logits", "hidden_states"]

        # ---- 6. Export prefill HMONNX ----
        logger.info(f"Exporting predictor prefill to {prefill_file}")
        with TimeProfiler("export_pred_prefill", logger), MemoryTracker("cuda:0", "export_pred_prefill", logger):
            quanted_model = convert_fx_model_to_quanted_model(
                wrapped_model,
                prefill_inputs,
                target_device,
                quant_config=quant_config,
            )
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(
                quanted_model,
                prefill_inputs,
                str(prefill_file),
                compatible_names,
                output_names,
            )
        logger.info(f"Predictor prefill export successful: {prefill_file}")

        # ---- 7. Export decode HMONNX ----
        decode_inputs = (
            inputs_embeds[:, :1, :],
            head_mask_decode,
            past_seq_length_t,
            torch.ones_like(current_input_length_t),
            past_key_caches,
            past_value_caches,
        )
        wrap_cfg.input_sequence_length = 1
        quanted_model.update_cfg(wrap_cfg)

        logger.info(f"Exporting predictor decode to {decode_file}")
        compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(
            quanted_model,
            decode_inputs,
            str(decode_file),
            compatible_names,
            output_names,
        )
        logger.info(f"Predictor decode export successful: {decode_file}")

        # ---- 8. Save meta ----
        meta_info = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "artifact_contract_version": 4,
            "module": "talker_prediction",
            "model_name": model_name,
            "talker_prediction_prefill_onnx": str(prefill_file.relative_to(work_dir)),
            "talker_prediction_decode_onnx": str(decode_file.relative_to(work_dir)),
            "talker_prediction_assets_file": str(asset_file.relative_to(work_dir)),
            "output_names": output_names,
            "hidden_states_output_contract": "predictor_input_embeds_for_talker_residual_sum",
            "codec_embedding_count": int(asset_payload["num_codec_embeddings"]),
            "lm_head_count": num_lm_heads,
            "talker_prediction_kv_cache": {"shape": kv_cache_shape, "num_decoder_layers": num_hidden_layers},
            "talker_prediction_hidden_size": int(inputs_embeds.shape[-1]),
            "talker_prediction_input_sequence_length": int(input_sequence_length),
        }
        meta_file = work_dir / "meta_talker_prediction.json"
        save_json(meta_file, meta_info)
        logger.info(f"Predictor export complete. Meta saved to {meta_file}")

    quanted_model = None
    wrapped_model = None
    code_predictor = None
    captured = None
    processor = None
    native_model = None
    release_export_cuda_memory(logger, "talker prediction export")

    # ---- 9. Optional validation ----
    if args.valid:
        if args.phase == "full":
            logger.info("Restarting process for predictor HMONNX validation after export cleanup")
            _reexec_with_phase("hmonnx-validate")

        if args.phase == "hmonnx-validate":
            inputs_embeds, past_key_caches, past_value_caches, past_seq_length_t, current_input_length_t = (
                _build_predictor_validation_inputs(work_dir, meta_info)
            )
            num_lm_heads = int(meta_info.get("lm_head_count", 15))
            batch = int(inputs_embeds.shape[0])
            prefill_seq = int(inputs_embeds.shape[1])
            prefill_step = max(0, min(prefill_seq - 2, num_lm_heads - 1))
            head_mask_prefill = torch.zeros(batch, prefill_seq, num_lm_heads, 1, dtype=torch.float16)
            head_mask_prefill[:, :, prefill_step, 0] = 1.0
            head_mask_decode = torch.zeros(batch, 1, num_lm_heads, 1, dtype=torch.float16)
            head_mask_decode[0, 0, 0, 0] = 1.0

            logger.info("Validating predictor HMONNX ...")
            from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

            session = HMONNXInference(str(prefill_file))
            output = session(
                inputs_embeds,
                head_mask_prefill,
                past_seq_length_t,
                current_input_length_t,
                *past_key_caches,
                *past_value_caches,
            )
            outputs = list(output) if isinstance(output, (list, tuple)) else [output]
            logger.info(
                "Predictor prefill HMONNX validation passed, output shapes: "
                + ", ".join(str(tuple(out.shape)) for out in outputs)
            )

            session_d = HMONNXInference(str(decode_file))
            output_d = session_d(
                inputs_embeds[:, :1, :],
                head_mask_decode,
                past_seq_length_t,
                torch.ones_like(current_input_length_t),
                *past_key_caches,
                *past_value_caches,
            )
            outputs_d = list(output_d) if isinstance(output_d, (list, tuple)) else [output_d]
            logger.info(
                "Predictor decode HMONNX validation passed, output shapes: "
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
            release_export_cuda_memory(logger, "talker prediction hmonnx validation")
            logger.info("Restarting process for predictor dialogue validation after direct HMONNX validation")
            _reexec_with_phase("dialogue-validate")

        if args.phase == "dialogue-validate":
            dialogue_artifacts = {
                "talker_prediction": {**meta_info, "_root_dir": str(work_dir), "_meta_path": str(meta_file)}
            }
            run_dialogue_validation(
                hf_model_path,
                work_dir,
                logger,
                case="multimodal",
                max_new_tokens=args.max_new_tokens,
                artifacts=dialogue_artifacts,
                report_name="talker_prediction_dialogue_validation.json",
                output_prefix="talker_prediction_dialogue",
                save_golden=args.save_golden,
                golden_dir=golden_dir,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Qwen3-Omni talker code predictor to HMONNX")
    parser.add_argument("--model", type=str, default="/data01/datasets/Qwen3-Omni-30B-A3B-Instruct/")
    parser.add_argument("--work-dir", type=str, default="work_dirs/qwen3omni")
    parser.add_argument("--quant-type", default="w8a8h0_sefp")
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--valid", action="store_true", default=True, help="validate exported HMONNX")
    parser.add_argument("--no-valid", action="store_false", dest="valid", help="skip validation")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--golden-root", type=str, default="work_dirs/qwen3omni_no_projection")
    parser.add_argument("--save-golden", action="store_true", default=True, help="save golden outputs after validation")
    parser.add_argument("--no-save-golden", action="store_false", dest="save_golden", help="skip golden output save")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--phase", choices=["full", "hmonnx-validate", "dialogue-validate"], default="full")
    args = parser.parse_args()
    main(args)
