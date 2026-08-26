# Copyright 2025 HOUMO AI
#
# File: funaudiochat_xh2a_export_hmonnx.py
# Description:
#   Split export pipeline for FunAudioChat on xhquant/xh2a.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from ._static_export import (
    FunAudioChatStaticAudioTowerWarp,
    FunAudioChatStaticDecoderDecodeWarp,
    FunAudioChatStaticDecoderPrefillWarp,
    FunAudioChatStaticEncoderWarp,
)
from .constant import AUDIO_TEMPLATE
from .qwen3_convert_config import Qwen3LegacyConvertConfig
from .qwen3_converter import Qwen3LegacyConverterXH2a
from xhquant.api import (  # isort:skip
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
from xhquant.core.cache_tensor import CacheTensor


def build_kv_caches(qwen_model, max_sequence_length: int) -> tuple[list[CacheTensor], list[CacheTensor]]:
    if hasattr(qwen_model, "layers"):
        num_decoder_layers = len(qwen_model.layers)
        num_key_value_heads = qwen_model.layers[0].self_attn.config.num_key_value_heads
        head_dim = qwen_model.layers[0].self_attn.head_dim
    else:
        num_decoder_layers = len(qwen_model.model.layers)
        num_key_value_heads = qwen_model.model.layers[0].self_attn.config.num_key_value_heads
        head_dim = qwen_model.model.layers[0].self_attn.head_dim

    kv_cache_shape = [1, num_key_value_heads, max_sequence_length, head_dim]

    past_key_caches = []
    past_value_caches = []
    for _ in range(num_decoder_layers):
        past_key_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
        past_value_caches.append(CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16)))
    return past_key_caches, past_value_caches


DEFAULT_PROMPT = "You are asked to generate both text and speech tokens at the same time. 你的名字是小云。你是一位来自杭州的温柔友善的女孩，声音甜美，举止亲切。你的回复语气自然友好，力求沟通简洁明了。你的回复简短，通常只有一到三句话，避免使用正式的称谓和重复的短语。你能用恰当的声音回复，遵循用户的指示，并能共情他们的情绪。你能用恰当的方言回复，会说四川话和粤语。"


def pad_last_dim(x: torch.Tensor, target_length: int, value: int | float = 0):
    current_length = x.shape[-1]
    if current_length == target_length:
        return x
    if current_length > target_length:
        return x[..., :target_length]
    pad_shape = list(x.shape)
    pad_shape[-1] = target_length - current_length
    pad_tensor = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad_tensor], dim=-1)


def pad_seq_dim(x: torch.Tensor, target_length: int, value: int | float = 0):
    current_length = x.shape[1]
    if current_length == target_length:
        return x
    if current_length > target_length:
        return x[:, :target_length, ...]
    pad_shape = list(x.shape)
    pad_shape[1] = target_length - current_length
    pad_tensor = torch.full(pad_shape, value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad_tensor], dim=1)


def build_padded_chunks(
    flat_input_features: torch.Tensor, chunk_lengths: list[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    max_chunk_len = max(chunk_lengths)
    start = 0
    padded_chunks = []
    padded_masks = []
    for length in chunk_lengths:
        end = start + length
        chunk = flat_input_features[:, start:end]
        pad_len = max_chunk_len - length
        padded_chunks.append(F.pad(chunk, (0, pad_len), value=0.0).to(torch.float16))
        mask = torch.zeros((1, max_chunk_len), dtype=torch.float16)
        mask[:, :length] = 1.0
        padded_masks.append(mask)
        start = end
    return torch.stack(padded_chunks, dim=0), torch.stack(padded_masks, dim=0)


def build_aftercnn_valid_mask(aftercnn_lens: torch.Tensor) -> torch.Tensor:
    max_aftercnn_len = int(aftercnn_lens.max().item())
    mask = torch.zeros((aftercnn_lens.shape[0], max_aftercnn_len, 1), dtype=torch.float16)
    for i, length in enumerate(aftercnn_lens.tolist()):
        mask[i, : int(length), 0] = 1.0
    return mask


def build_audio_attention_mask(aftercnn_lens: torch.Tensor) -> torch.Tensor:
    max_aftercnn_len = int(aftercnn_lens.max().item())
    total_aftercnn_length = int(aftercnn_lens.shape[0]) * max_aftercnn_len
    attention_mask = torch.full(
        (1, 1, total_aftercnn_length, total_aftercnn_length),
        torch.finfo(torch.float16).min,
        dtype=torch.float16,
    )
    for chunk_idx, valid_len in enumerate(aftercnn_lens.tolist()):
        start = chunk_idx * max_aftercnn_len
        end = start + int(valid_len)
        attention_mask[:, :, start:end, start:end] = 0.0
    return attention_mask


def build_continuous_audio_valid_mask(
    pooled_lengths: list[int],
    fixed_pooled_len: int,
    speech_maxlen: int,
) -> torch.Tensor:
    mask = torch.zeros((1, speech_maxlen, 1), dtype=torch.float16)
    offset = 0
    for pooled_len in pooled_lengths:
        mask[:, offset : offset + int(pooled_len), :] = 1.0
        offset += fixed_pooled_len
    return mask


def build_decoder_attention_mask(
    valid_key_length: int,
    total_query_length: int,
    total_key_length: int,
    *,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    attention_mask = torch.zeros((total_query_length, total_key_length), dtype=dtype)
    if valid_key_length < total_key_length:
        attention_mask[:, valid_key_length:total_key_length] = torch.finfo(dtype).min
        attention_mask[valid_key_length:total_key_length, :] = torch.finfo(dtype).min
    return attention_mask


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max().item())


def build_scattered_encoder_outputs(
    inputs_embeds: torch.Tensor,
    audio_features: torch.Tensor,
    # text_features: torch.Tensor,
    # text_mix_mask: torch.Tensor,
    audio_token_positions: torch.Tensor,
    max_audio_tokens: int,
):
    # mix_mask = text_mix_mask
    # mixed_audio_features = (text_features + audio_features) / 2
    # final_audio_features = mixed_audio_features * mix_mask + audio_features * (1.0 - mix_mask)

    # merged_inputs_embeds = inputs_embeds.clone()
    # merged_inputs_embeds[:, audio_token_positions, :] = final_audio_features[:, :max_audio_tokens, :]

    decoder_text_embeds = inputs_embeds.clone()
    decoder_text_embeds[:, audio_token_positions, :] = audio_features  # [:, :max_audio_tokens, :]
    return decoder_text_embeds
    # return merged_inputs_embeds, decoder_text_embeds, final_audio_features


def run_decoder_prefill(
    language_model,
    decoder_text_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    chunk_length: int,
    past_key_caches: list[torch.Tensor],
    past_value_caches: list[torch.Tensor],
) -> torch.Tensor:
    total_length = decoder_text_embeds.shape[1]
    valid_text_length = int(attention_mask.sum().item())

    if total_length <= chunk_length:
        past_seq_length = torch.zeros(
            (decoder_text_embeds.shape[0],), dtype=torch.int32, device=decoder_text_embeds.device
        )
        current_input_length = torch.full(
            (decoder_text_embeds.shape[0],),
            decoder_text_embeds.shape[1],
            dtype=torch.int32,
            device=decoder_text_embeds.device,
        )
        padded_attention_mask = build_decoder_attention_mask(
            valid_text_length,
            total_length,
            total_length,
            dtype=decoder_text_embeds.dtype,
        )
        outputs = language_model(
            attention_mask=padded_attention_mask,
            inputs_embeds=decoder_text_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_caches,
            past_value_cache=past_value_caches,
            output_hidden_states=True,
            return_dict=True,
            use_cache=True,
        )
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]
        if hasattr(outputs, "last_hidden_state"):
            return outputs.last_hidden_state
        return outputs[1]

    hidden_states = []
    processed_length = 0

    for start in range(0, total_length, chunk_length):
        end = min(start + chunk_length, total_length)
        current_length = end - start
        current_embeds = decoder_text_embeds[:, start:end, :]
        padded_embeds = pad_seq_dim(current_embeds, chunk_length, 0.0)

        chunk_attention_mask = build_decoder_attention_mask(
            processed_length + current_length,
            chunk_length,
            processed_length + chunk_length,
            dtype=decoder_text_embeds.dtype,
        ).to(decoder_text_embeds.device)
        past_seq_length = torch.full(
            (decoder_text_embeds.shape[0],),
            processed_length,
            dtype=torch.int32,
            device=decoder_text_embeds.device,
        )
        current_input_length = torch.full(
            (decoder_text_embeds.shape[0],),
            current_length,
            dtype=torch.int32,
            device=decoder_text_embeds.device,
        )

        outputs = language_model(
            attention_mask=chunk_attention_mask,
            inputs_embeds=padded_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_cache=past_key_caches,
            past_value_cache=past_value_caches,
        )
        chunk_hidden_state = outputs.hidden_states[-1]
        hidden_states.append(chunk_hidden_state)
        processed_length += current_length

    return torch.cat(hidden_states, dim=1)


def preprocess_static_inputs(model, model_inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    speech_valid_length = int(model_inputs["speech_attention_mask"].sum(-1)[0].item())
    feature_valid_length = int(model_inputs["feature_attention_mask"].sum(-1)[0].item())
    group_size = int(model.config.audio_config.group_size)
    speech_input_length = int(model_inputs["speech_ids"].shape[-1])
    fixed_speech_length = max(
        speech_input_length,
        ((speech_valid_length + group_size - 1) // group_size) * group_size,
    )
    fixed_speech_length = ((fixed_speech_length + group_size - 1) // group_size) * group_size
    fixed_text_length = fixed_speech_length // group_size
    audio_pad_token_id = int(model.config.audio_config.pad_token_id)
    text_pad_value = int(model.config.text_config.sil_index)

    speech_ids = pad_last_dim(model_inputs["speech_ids"].to(torch.int32), fixed_speech_length, audio_pad_token_id)
    disable_text_branch = bool(model_inputs.get("disable_text_branch", False))
    if (not disable_text_branch) and "text_ids" in model_inputs:
        text_ids = pad_last_dim(model_inputs["text_ids"].to(torch.int32), fixed_text_length, text_pad_value)
        text_attention_mask = pad_last_dim(model_inputs["text_attention_mask"].to(torch.int32), fixed_text_length, 0)
    elif disable_text_branch:
        text_ids = None
        text_attention_mask = None
    else:
        text_ids = torch.full((1, fixed_text_length), text_pad_value, dtype=torch.int32).to(
            model_inputs["input_features"].device
        )
        text_attention_mask = torch.zeros((1, fixed_text_length), dtype=torch.int32).to(
            model_inputs["input_features"].device
        )

    flat_input_features = (
        model_inputs["input_features"]
        .permute(0, 2, 1)[model_inputs["feature_attention_mask"].bool()]
        .permute(1, 0)
        .to(torch.float16)
    )
    feature_lens = torch.tensor([feature_valid_length], dtype=torch.int32).to(model_inputs["input_features"].device)
    n_window = int(model.continuous_audio_tower.n_window)
    full_chunk_len = n_window * 2
    feature_valid_length = int(feature_lens[0].item())
    chunk_num = (feature_valid_length + full_chunk_len - 1) // full_chunk_len
    chunk_lengths = [full_chunk_len] * max(chunk_num - 1, 0)
    last_chunk_len = feature_valid_length - full_chunk_len * max(chunk_num - 1, 0)
    if chunk_num > 0:
        chunk_lengths.append(last_chunk_len if last_chunk_len > 0 else full_chunk_len)
    chunk_feature_lens = torch.tensor(chunk_lengths, dtype=torch.int32).to(model_inputs["input_features"].device)
    chunk_aftercnn_lens, _ = model.continuous_audio_tower._get_feat_extract_output_lengths(chunk_feature_lens)
    pooled_lengths = [int(x) if int(x) < 2 else int(x) // 2 for x in chunk_aftercnn_lens.tolist()]
    fixed_pooled_len = (
        int(chunk_aftercnn_lens.max().item())
        if int(chunk_aftercnn_lens.max().item()) < 2
        else int(chunk_aftercnn_lens.max().item()) // 2
    )
    audio_token_positions = (
        (model_inputs["input_ids"][0] == model.config.audio_token_index)
        .nonzero(as_tuple=False)
        .squeeze(-1)
        .to(torch.int64)
    )

    input_ids = model_inputs["input_ids"].to(torch.int32)
    inputs_embeds = model.get_input_embeddings()(input_ids).to(torch.float16)
    audio_inputs_embeds = model.audio_tower.embed_tokens(speech_ids.to(torch.long)).to(torch.float16)
    if text_ids is not None and text_attention_mask is not None:
        text_ids = text_ids.clone()
        text_ids[text_ids == model.config.text_config.eos_token_id] = model.config.text_config.sil_index
        if model.config.text_config.pad_token_id is not None:
            text_ids[text_ids == model.config.text_config.pad_token_id] = model.config.text_config.sil_index
        text_features = model.get_input_embeddings()(text_ids).to(torch.float16)
        text_mix_mask = text_attention_mask[:, :, None].to(torch.float16)
        text_features = text_features * text_mix_mask
    else:
        text_features = None
        text_mix_mask = None
    padded_input_features, chunk_padded_mask = build_padded_chunks(flat_input_features, chunk_lengths)
    aftercnn_valid_mask = build_aftercnn_valid_mask(chunk_aftercnn_lens)
    audio_attention_mask = build_audio_attention_mask(chunk_aftercnn_lens)
    continuous_audio_valid_mask = build_continuous_audio_valid_mask(
        pooled_lengths,
        fixed_pooled_len,
        fixed_speech_length,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": model_inputs["attention_mask"].to(torch.int32),
        "inputs_embeds": inputs_embeds,
        "speech_ids": speech_ids,
        "audio_inputs_embeds": audio_inputs_embeds,
        "padded_input_features": padded_input_features,
        "chunk_padded_mask": chunk_padded_mask.to(model_inputs["input_features"].device),
        "aftercnn_valid_mask": aftercnn_valid_mask.to(model_inputs["input_features"].device),
        "audio_attention_mask": audio_attention_mask.to(model_inputs["input_features"].device),
        "continuous_audio_valid_mask": continuous_audio_valid_mask.to(model_inputs["input_features"].device),
        "text_features": text_features,
        "text_mix_mask": text_mix_mask,
        "feature_lens": chunk_feature_lens,
        "aftercnn_lens": chunk_aftercnn_lens,
        "speech_maxlen": fixed_speech_length,
        "feature_exist_mask": model_inputs["feature_exist_mask"].to(torch.bool),
        "audio_token_positions": audio_token_positions,
        "chunk_lengths": chunk_lengths,
        "pooled_lengths": pooled_lengths,
    }


def build_dummy_inputs(
    model, processor, audio_path: str, prompt: str
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    import librosa

    audio = [librosa.load(audio_path, sr=16000)[0]]
    conversation = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": AUDIO_TEMPLATE},
    ]
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    model_inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False)

    required = preprocess_static_inputs(model, model_inputs)
    return required, {"chat_text": text}


def validate_static_warp_consistency(model, split_inputs, logger):
    static_warp = FunAudioChatStaticEncoderWarp(
        model,
        feature_lens=split_inputs["feature_lens"],
        aftercnn_lens=split_inputs["aftercnn_lens"],
        speech_maxlen=int(split_inputs["speech_maxlen"]),
        feature_exist_mask=split_inputs["feature_exist_mask"],
        audio_token_positions=split_inputs["audio_token_positions"],
        chunk_lengths=split_inputs["chunk_lengths"],
        pooled_lengths=split_inputs["pooled_lengths"],
    ).eval()

    with torch.no_grad():
        warp_outputs = static_warp(
            split_inputs["speech_ids"],
            split_inputs["audio_inputs_embeds"],
            split_inputs["padded_input_features"],
            split_inputs["chunk_padded_mask"],
            split_inputs["aftercnn_valid_mask"],
            split_inputs["audio_attention_mask"],
            split_inputs["continuous_audio_valid_mask"],
        )
        warp_audio_features = warp_outputs
        warp_decoder_text_embeds = build_scattered_encoder_outputs(
            split_inputs["inputs_embeds"],
            warp_audio_features,
            # warp_text_features,
            # warp_text_mix_mask,
            split_inputs["audio_token_positions"],
            len(split_inputs["pooled_lengths"]),
        )

        continuous_audio_features, continuous_audio_output_lengths = model.get_audio_features(
            split_inputs["padded_input_features"][0, :, : int(split_inputs["chunk_lengths"][0])]
            .unsqueeze(0)
            .transpose(1, 2),
            feature_attention_mask=torch.ones((1, int(split_inputs["feature_lens"][0].item())), dtype=torch.int32),
            speech_maxlen=int(split_inputs["speech_maxlen"]),
        )
        continuous_audio_features = continuous_audio_features.to(torch.float16)
        audio_features, ref_audio_embeds, _ = model.audio_tower(
            split_inputs["speech_ids"],
            inputs_embeds=split_inputs["audio_inputs_embeds"],
            continuous_audio_features=continuous_audio_features,
            continuous_audio_output_lengths=split_inputs["aftercnn_lens"].new_tensor(
                split_inputs["pooled_lengths"], dtype=torch.int32
            ),
            feature_exist_mask=split_inputs["feature_exist_mask"],
        )
        ref_mixed_audio_features = (split_inputs["text_features"] + audio_features) / 2
        ref_audio_features = ref_mixed_audio_features * split_inputs["text_mix_mask"] + audio_features * (
            1.0 - split_inputs["text_mix_mask"]
        )
        ref_merged_inputs_embeds = split_inputs["inputs_embeds"].clone()
        ref_merged_inputs_embeds[:, split_inputs["audio_token_positions"], :] = ref_audio_features[
            :, : len(split_inputs["pooled_lengths"]), :
        ]
        ref_decoder_text_embeds = split_inputs["inputs_embeds"].clone()
        ref_decoder_text_embeds[:, split_inputs["audio_token_positions"], :] = split_inputs["text_features"][
            :, : len(split_inputs["pooled_lengths"]), :
        ]

    logger.info(
        "Static warp consistency: decoder_text=%.6f audio_features=%.6f",
        max_diff(warp_decoder_text_embeds, ref_decoder_text_embeds),
        max_diff(warp_audio_features, ref_audio_features),
    )


def export_audio_encoder(model, processor, args, work_dir: Path, logger) -> Path:
    split_inputs, extra = build_dummy_inputs(model, processor, args.audio, args.system_prompt)
    export_model = FunAudioChatStaticEncoderWarp(
        model,
        feature_lens=split_inputs["feature_lens"],
        aftercnn_lens=split_inputs["aftercnn_lens"],
        speech_maxlen=int(split_inputs["speech_maxlen"]),
        feature_exist_mask=split_inputs["feature_exist_mask"],
        audio_token_positions=split_inputs["audio_token_positions"],
        chunk_lengths=split_inputs["chunk_lengths"],
        pooled_lengths=split_inputs["pooled_lengths"],
    ).eval()

    input_args = (
        split_inputs["speech_ids"],
        split_inputs["audio_inputs_embeds"],
        split_inputs["padded_input_features"],
        split_inputs["chunk_padded_mask"],
        split_inputs["aftercnn_valid_mask"],
        split_inputs["audio_attention_mask"],
        split_inputs["continuous_audio_valid_mask"],
    )
    input_names = [
        "speech_ids",
        "audio_inputs_embeds",
        "padded_input_features",
        "chunk_padded_mask",
        "aftercnn_valid_mask",
        "audio_attention_mask",
        "continuous_audio_valid_mask",
    ]
    output_names = ["audio_features"]

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.audio_quant_type)
    quant_config = ConfigDict(create_quant_config(quant_scheme))
    quanted_model = convert_fx_model_to_quanted_model(
        export_model,
        input_args,
        DeviceType.XH2a,
        quant_config=quant_config,
    )

    out_file = work_dir / "audio_encoder" / "hmonnx" / f"funaudiochat_audio_encoder_{args.audio_quant_type}.onnx"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    convert_quanted_model_to_hmonnx(quanted_model, input_args, str(out_file), input_names, output_names)

    meta = {
        "audio_path": args.audio,
        "chat_text": extra["chat_text"],
        "audio_quant_type": args.audio_quant_type,
        "input_names": input_names,
        "output_names": output_names,
        "feature_lens": split_inputs["feature_lens"].tolist(),
        "aftercnn_lens": split_inputs["aftercnn_lens"].tolist(),
        "speech_maxlen": int(split_inputs["speech_maxlen"]),
        "audio_token_positions": split_inputs["audio_token_positions"].tolist(),
        "chunk_lengths": split_inputs["chunk_lengths"],
        "pooled_lengths": split_inputs["pooled_lengths"],
    }
    (work_dir / "audio_encoder" / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Exported audio encoder to {out_file}")
    return out_file


def export_audio_tower(model, args, work_dir: Path, logger) -> Path:
    export_model = FunAudioChatStaticAudioTowerWarp(model).eval()
    group_size = int(model.audio_tower.group_size)
    speech_ids = torch.full(
        (1, group_size),
        int(model.config.audio_config.pad_token_id),
        dtype=torch.int32,
    )
    input_args = (speech_ids,)
    input_names = ["speech_ids"]
    output_names = ["audio_features"]

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.audio_quant_type)
    quant_config = ConfigDict(create_quant_config(quant_scheme))
    quanted_model = convert_fx_model_to_quanted_model(
        export_model,
        input_args,
        DeviceType.XH2a,
        quant_config=quant_config,
    )

    out_file = work_dir / "audio_tower" / "hmonnx" / f"funaudiochat_audio_tower_{args.audio_quant_type}.onnx"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    convert_quanted_model_to_hmonnx(quanted_model, input_args, str(out_file), input_names, output_names)

    meta = {
        "audio_quant_type": args.audio_quant_type,
        "group_size": group_size,
        "pad_token_id": int(model.config.audio_config.pad_token_id),
        "input_names": input_names,
        "output_names": output_names,
    }
    (work_dir / "audio_tower" / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Exported audio tower to {out_file}")
    return out_file


def export_qwen3(model, args, work_dir: Path):
    qwen_cfg = Qwen3LegacyConvertConfig(
        batch_size=1,
        context_length=args.context_length,
        input_sequence_length=args.input_sequence_length,
        quant_scheme=QuantScheme(target_device=DeviceType.XH2a, quant_type=args.llm_quant_type),
        quant_weight=None,
        mix_search=None,
        num_logits_to_keep=1,
    )
    qwen_dir = work_dir / "qwen3"
    qwen_dir.mkdir(parents=True, exist_ok=True)
    Qwen3LegacyConverterXH2a.convert(model.language_model, qwen_cfg, str(qwen_dir))
    return qwen_dir / "meta.json"


def export_decoder(model, processor, args, work_dir: Path, logger) -> Path:
    del processor

    decoder = model.audio_invert_tower
    decoder_group_size = int(decoder.group_size)
    decoder_input_sequence_length = int(args.input_sequence_length)
    decoder_total_length = decoder_input_sequence_length * decoder_group_size
    decoder_prefill_valid_length = decoder_total_length - (decoder_group_size - 1)

    with torch.no_grad():
        speech_inputs_embeds = torch.zeros(
            (1, decoder_input_sequence_length, int(decoder.hidden_size)),
            dtype=torch.float16,
        )
        decoder_prefill_inputs_embeds = decoder.pre_matching(speech_inputs_embeds)
        decoder_prefill_hidden_states = decoder_prefill_inputs_embeds.reshape(
            decoder_prefill_inputs_embeds.shape[0],
            decoder_total_length,
            -1,
        )
        decoder_prefill_past_seq_length = torch.zeros(
            (decoder_prefill_hidden_states.shape[0],),
            dtype=torch.int32,
            device=decoder_prefill_hidden_states.device,
        )
        decoder_prefill_current_input_length = torch.full(
            (decoder_prefill_hidden_states.shape[0],),
            decoder_prefill_valid_length,
            dtype=torch.int32,
            device=decoder_prefill_hidden_states.device,
        )

    decoder_prefill_wrapper = FunAudioChatStaticDecoderPrefillWarp(decoder).eval()
    padded_attention_mask = build_decoder_attention_mask(
        decoder_prefill_valid_length,
        decoder_total_length,
        decoder_total_length,
        dtype=decoder_prefill_hidden_states.dtype,
    )
    crq_max_sequence_length = decoder_total_length
    prefill_past_key_caches, prefill_past_value_caches = build_kv_caches(
        decoder.crq_transformer,
        crq_max_sequence_length,
    )

    prefill_input_args = (
        decoder_prefill_hidden_states,
        decoder_prefill_past_seq_length,
        decoder_prefill_current_input_length,
        padded_attention_mask,
        prefill_past_key_caches,
        prefill_past_value_caches,
    )
    prefill_input_names = ["crq_inputs_embeds", "past_seq_length", "current_input_length", "attention_mask"]
    for layer_idx in range(len(prefill_past_key_caches)):
        prefill_input_names.append(f"past_key_cache_{layer_idx}")
    for layer_idx in range(len(prefill_past_value_caches)):
        prefill_input_names.append(f"past_value_cache_{layer_idx}")
    prefill_output_names = ["speech_logits"]

    with torch.no_grad():
        warp_prefill_logits = decoder_prefill_wrapper(*prefill_input_args)
        warp_prefill_logits = warp_prefill_logits[:, :decoder_prefill_valid_length, :]
        prefill_next_tokens = torch.argmax(warp_prefill_logits[:, -1, :].float(), dim=-1)
        prefill_next_audio_embeds = decoder.get_embeddings(prefill_next_tokens).to(
            dtype=decoder_prefill_hidden_states.dtype,
            device=decoder_prefill_hidden_states.device,
        )
        decoder.crq_audio_embeds = prefill_next_audio_embeds

        decode_step_inputs_embeds = decoder_prefill_hidden_states[
            :,
            decoder_prefill_valid_length : decoder_prefill_valid_length + 1,
            :,
        ]
        decode_past_seq_length = torch.full(
            (decode_step_inputs_embeds.shape[0],),
            decoder_prefill_valid_length,
            dtype=torch.int32,
            device=decode_step_inputs_embeds.device,
        )
        decode_current_input_length = torch.full(
            (decode_step_inputs_embeds.shape[0],),
            1,
            dtype=torch.int32,
            device=decode_step_inputs_embeds.device,
        )
        decode_attention_mask = build_decoder_attention_mask(
            decoder_prefill_valid_length + 1,
            1,
            decoder_total_length,
            dtype=decode_step_inputs_embeds.dtype,
        ).to(decode_step_inputs_embeds.device)
        decode_input_args = (
            decode_step_inputs_embeds,
            decode_past_seq_length,
            decode_current_input_length,
            decode_attention_mask,
            prefill_past_key_caches,
            prefill_past_value_caches,
        )

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.decoder_quant_type)
    quant_config = ConfigDict(create_quant_config(quant_scheme))

    quanted_prefill_model = convert_fx_model_to_quanted_model(
        decoder_prefill_wrapper,
        prefill_input_args,
        DeviceType.XH2a,
        quant_config=quant_config,
    )

    prefill_out_file = (
        work_dir / "audio_decoder" / "hmonnx" / f"funaudiochat_audio_decoder_prefill_{args.decoder_quant_type}.onnx"
    )
    decode_out_file = (
        work_dir / "audio_decoder" / "hmonnx" / f"funaudiochat_audio_decoder_decode_{args.decoder_quant_type}.onnx"
    )
    prefill_out_file.parent.mkdir(parents=True, exist_ok=True)
    convert_quanted_model_to_hmonnx(
        quanted_prefill_model, prefill_input_args, str(prefill_out_file), prefill_input_names, prefill_output_names
    )

    logger.info(f"********************* start export audio decoder decode model *********************")
    decode_input_names = ["crq_inputs_embeds", "past_seq_length", "current_input_length", "attention_mask"]
    for layer_idx in range(len(prefill_past_key_caches)):
        decode_input_names.append(f"past_key_cache_{layer_idx}")
    for layer_idx in range(len(prefill_past_value_caches)):
        decode_input_names.append(f"past_value_cache_{layer_idx}")
    decode_output_names = ["speech_logits"]

    decoder_dict = Config(
        dict(
            max_sequence_length=decoder_total_length,
            input_sequence_length=1,
            use_cache=True,
            num_logits_to_keep=0,
            kv_cache=dict(
                cache_axis=2,
            ),
        )
    )
    decoder.crq_transformer._setup(decoder_dict)
    quanted_prefill_model.update_cfg(decoder_dict)
    # decoder_decode_wrapper = FunAudioChatStaticDecoderDecodeWarp(decoder).eval()

    # with torch.no_grad():
    #     warp_decode_logits = decoder_decode_wrapper(*decode_input_args)
    #     logger.info(
    #         "Audio decoder warp warmup: prefill_logits=%s decode_logits=%s",
    #         tuple(warp_prefill_logits.shape),
    #         tuple(warp_decode_logits.shape),
    #     )

    convert_quanted_model_to_hmonnx(
        quanted_prefill_model, decode_input_args, str(decode_out_file), decode_input_names, decode_output_names
    )

    meta = {
        "decoder_quant_type": args.decoder_quant_type,
        "input_sequence_length": decoder_input_sequence_length,
        "decoder_total_length": decoder_total_length,
        "kv_cache": {
            "shape": list(prefill_past_key_caches[0].shape),
            "num_decoder_layers": len(prefill_past_key_caches),
        },
        "prefill_input_names": prefill_input_names,
        "prefill_output_names": prefill_output_names,
        "decode_input_names": decode_input_names,
        "decode_output_names": decode_output_names,
    }
    (work_dir / "audio_decoder" / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Exported audio decoder prefill to {prefill_out_file}")
    logger.info(f"Exported audio decoder decode to {decode_out_file}")
    return prefill_out_file, decode_out_file
