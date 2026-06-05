#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: demo_hmonnx_full.py
# Description:
#   Qwen3-Omni ALL-HMONNX GPU inference pipeline.
#   All models (text_llm, audio, visual, talker, predictor, code2wav) run on
#   HMONNX GPU. text_projection/hidden_projection run on-chip as their own
#   standalone HMONNX graphs (output/xh2/hmquant/{text,hidden}_projection) and
#   are injected into the talker through bypass_embeds (bypass_mask=1).
#
#   Rationale: the talker's baked-in projection historically used w16a16_sefp
#   which had insufficient precision (cos~0.994). With w8a8h1_sefp (now the
#   default in the export script), fusion precision matches standalone (cos=1.0).
#   --use-fusion re-enables the in-graph projection (now precise enough).

#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 python demo_hmonnx_full.py --enable-audio-generation
#
# SPDX-License-Identifier: Apache-2.0

import os
import time
import argparse
from pathlib import Path
from typing import List

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from loguru import logger

from xhquant.api import HMONNXInference, CacheTensor

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = SCRIPT_DIR / "Qwen3-Omni-30B-A3B-Instruct"
DEFAULT_HMQUANT_DIR = SCRIPT_DIR / "output" / "xh2" / "hmquant"
DEFAULT_SAMPLE_DIR = SCRIPT_DIR / "sample_data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

# ═══════════════════════════════════════════════════════════════════════════════
# Constants (from config.json, hardcoded to avoid HF model dependency)
# ═══════════════════════════════════════════════════════════════════════════════
IMAGE_TOKEN_ID = 151655
AUDIO_TOKEN_ID = 151675
VIDEO_TOKEN_ID = 151656
IM_START_TOKEN_ID = 151644
SYSTEM_TOKEN_ID = 8948
USER_TOKEN_ID = 872
ASSISTANT_TOKEN_ID = 77091
TTS_BOS_TOKEN_ID = 151672
TTS_EOS_TOKEN_ID = 151673
TTS_PAD_TOKEN_ID = 151671
CODEC_BOS_ID = 2149
CODEC_PAD_ID = 2148
CODEC_EOS_TOKEN_ID = 2150
CODEC_NOTHINK_ID = 2155
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157
NUM_CODE_GROUPS = 16
SPEAKER_ID_MAP = {"chelsie": 2301, "ethan": 2302, "aiden": 2303}
THINKER_HIDDEN_SIZE = 2048
TALKER_HIDDEN_SIZE = 1024
THINKER_NUM_LAYERS = 48
TALKER_NUM_LAYERS = 20
PREDICTOR_NUM_LAYERS = 5
AUDIO_N_WINDOW = 50
AUDIO_N_WINDOW_INFER = 800
AUDIO_MEL_BINS = 128
CODE2WAV_UPSAMPLE = 8 * 5 * 4 * 3 * 2 * 2  # upsample_rates * upsampling_ratios = 1920
TEXT_LLM_STATIC_SEQ = 256
TALKER_STATIC_SEQ = 150
EOS_TOKEN_IDS = {151645, 151643}  # <|im_end|>, <|endoftext|>


# ═══════════════════════════════════════════════════════════════════════════════
# Args
# ═══════════════════════════════════════════════════════════════════════════════
def get_args():
    p = argparse.ArgumentParser(description="Qwen3-Omni ALL-HMONNX GPU pipeline")
    p.add_argument("--model-dir", type=str, default=str(DEFAULT_MODEL_DIR))
    p.add_argument("--hmquant-dir", type=str, default=str(DEFAULT_HMQUANT_DIR))
    p.add_argument("--image", type=str, default=str(DEFAULT_SAMPLE_DIR / "cars.jpg"))
    p.add_argument("--audio", type=str, default=str(DEFAULT_SAMPLE_DIR / "cough.wav"))
    p.add_argument("--prompt", type=str, default="What can you see and hear? Please answer in four complete sentences, with enough detail to make the synthesized speech noticeably longer.")
    p.add_argument("--speaker", type=str, default="ethan")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--talker-max-new-tokens", type=int, default=512)
    p.add_argument("--enable-audio-generation", action="store_true")
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    # The talker baked-in text/hidden projection (bypass_mask=0) was imprecise with w16a16_sefp
    # With w8a8h1_sefp (now default), fusion precision matches standalone (cos=1.0). By default we
    # project with the standalone projection HMONNX graphs (load_hmonnx_projections, cos≈1.0 vs
    # fp16) and feed everything through bypass. --use-fusion re-enables the talker's
    # in-graph projection for experimentation.
    p.add_argument("--use-fusion", action="store_true",
                   help="use the talker's in-graph projection (now precise with w8a8h1_sefp; for debugging)")
    p.add_argument("--save-thinker", type=str, default=None,
                   help="save thinker (text) outputs to this .pt for fast talker-only iteration")
    p.add_argument("--load-thinker", type=str, default=None,
                   help="load thinker outputs from this .pt and skip text-LLM stages")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Embedding Helpers
# ═══════════════════════════════════════════════════════════════════════════════
def load_text_embedding(hmquant_dir: Path) -> torch.Tensor:
    """Load thinker token embedding [vocab_size, 2048]."""
    data = torch.load(hmquant_dir / "quant_embedding.pt", map_location="cpu", weights_only=False)
    if isinstance(data, dict) and "weight" in data:
        return data["weight"].to(torch.float16)
    return data.to(torch.float16)


def load_talker_embedding(hmquant_dir: Path) -> torch.Tensor:
    """Load talker token embedding [3072, 1024]."""
    data = torch.load(hmquant_dir / "quant_embedding_talker.pt", map_location="cpu", weights_only=False)
    if isinstance(data, torch.Tensor):
        return data.to(torch.float16)
    return data.weight.data.to(torch.float16)


def load_codec_embeddings(hmquant_dir: Path) -> List[torch.Tensor]:
    """Load codec embeddings list[15], each [2048, 1024]."""
    data = torch.load(hmquant_dir / "quant_embedding_codec.pt", map_location="cpu", weights_only=False)
    codec_list = data["codec_embeddings"]
    return [e.to(torch.float16) if isinstance(e, torch.Tensor) else e.weight.data.to(torch.float16) for e in codec_list]


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone Projection HMONNX (text_projection / hidden_projection, on-chip)
#   Replaces the onnxruntime FP16 helpers of demo_hmonnx_full_ort.py: the two
#   projection branches are exported as their own HMONNX graphs and run on GPU via
#   HMONNXInference instead of onnxruntime-on-CPU. Each graph is static
#   ("source" [1, S, 2048] -> "output" [1, S, 1024], S=150), so sequences are
#   chunked by S, the tail chunk is zero-padded, and the output is trimmed back.
# ═══════════════════════════════════════════════════════════════════════════════
def load_hmonnx_projections(hmquant_dir: Path, device) -> dict:
    """Load the standalone text/hidden projection HMONNX graphs onto `device`."""
    text_session = HMONNXInference(str(hmquant_dir / "text_projection" / "hmquant_qwen3-omni_with_act.onnx"))
    text_session.to(device)
    hidden_session = HMONNXInference(str(hmquant_dir / "hidden_projection" / "hmquant_qwen3-omni_with_act.onnx"))
    hidden_session.to(device)
    static = int(text_session.inputs[0].shape[1])
    logger.info(f"Loaded standalone text/hidden projection HMONNX (static={static}, on-chip, no onnxruntime).")
    return {"text": text_session, "hidden": hidden_session, "static": static, "device": device}


def _run_hmonnx_projection(session: HMONNXInference, x: torch.Tensor, static: int, device) -> torch.Tensor:
    """x [1, T, 2048] -> [1, T, 1024]. Chunk by `static`, zero-pad the tail, trim output."""
    x = x.to(device, torch.float16)
    batch, total, dim = x.shape
    outs = []
    for start in range(0, total, static):
        end = min(start + static, total)
        seg_len = end - start
        chunk = x[:, start:end, :]
        if seg_len < static:
            pad = torch.zeros(batch, static - seg_len, dim, dtype=torch.float16, device=device)
            chunk = torch.cat([chunk, pad], dim=1)
        out = session.forward(chunk)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, np.ndarray):
            out = torch.from_numpy(out)
        outs.append(out.to(torch.float16)[:, :seg_len, :])
    return torch.cat(outs, dim=1).to("cpu", torch.float16)


def hmonnx_text_projection(sessions: dict, x: torch.Tensor) -> torch.Tensor:
    """text_projection: [1, T, 2048] -> [1, T, 1024]."""
    return _run_hmonnx_projection(sessions["text"], x, sessions["static"], sessions["device"])


def hmonnx_hidden_projection(sessions: dict, x: torch.Tensor) -> torch.Tensor:
    """hidden_projection: [1, T, 2048] -> [1, T, 1024]."""
    return _run_hmonnx_projection(sessions["hidden"], x, sessions["static"], sessions["device"])


# ═══════════════════════════════════════════════════════════════════════════════
# Audio Encoder
# ═══════════════════════════════════════════════════════════════════════════════
def _get_feat_extract_output_lengths(input_lengths: torch.LongTensor) -> torch.LongTensor:
    """Whisper-style CNN output length: (L-1)//2 + 1."""
    return (input_lengths - 1) // 2 + 1


def run_audio_encoder(audio_session: HMONNXInference, input_features: torch.Tensor,
                      feature_attention_mask: torch.Tensor) -> torch.Tensor:
    """Run audio HMONNX encoder, returns audio_embeds [num_audio_tokens, 2048]."""
    audio_feature_lengths = feature_attention_mask.sum(dim=1).long()
    input_feat = input_features.permute(0, 2, 1)[feature_attention_mask.bool()].permute(1, 0)

    aftercnn_lens = _get_feat_extract_output_lengths(audio_feature_lengths)
    n_window = AUDIO_N_WINDOW
    n_window_infer = AUDIO_N_WINDOW_INFER

    chunk_num = torch.ceil(audio_feature_lengths.float() / (n_window * 2)).long()
    chunk_lengths = torch.tensor(
        [n_window * 2] * int(chunk_num.sum().item()), dtype=torch.long)
    tail_chunk_index = torch.nn.functional.pad(chunk_num, (1, 0), value=-1).cumsum(0)[1:]
    chunk_lengths[tail_chunk_index] = audio_feature_lengths % (n_window * 2)
    chunk_lengths[chunk_lengths == 0] = n_window * 2

    chunk_list = input_feat.T.split(chunk_lengths.tolist(), dim=0)
    padded_feature = nn.utils.rnn.pad_sequence(chunk_list, batch_first=True).transpose(1, 2)
    feature_lens_after_cnn = _get_feat_extract_output_lengths(chunk_lengths)

    all_outputs = []
    dev = audio_session.exec_device
    for i in range(padded_feature.shape[0]):
        single_feature = padded_feature[i:i+1].to(dtype=torch.float16, device=dev)
        single_cu = torch.tensor([0, int(feature_lens_after_cnn[i])], dtype=torch.int32, device=dev)
        out_i = audio_session.forward(single_feature, single_cu)
        if isinstance(out_i, (list, tuple)):
            out_i = out_i[0]
        if isinstance(out_i, np.ndarray):
            out_i = torch.from_numpy(out_i)
        out_i = out_i.to(torch.float16)[:int(feature_lens_after_cnn[i])]
        all_outputs.append(out_i)

    return torch.cat(all_outputs, dim=0) if len(all_outputs) > 1 else all_outputs[0]


# ═══════════════════════════════════════════════════════════════════════════════
# Visual Encoder
# ═══════════════════════════════════════════════════════════════════════════════
def run_visual_encoder(vision_session: HMONNXInference, pixel_values: torch.Tensor):
    """Run visual HMONNX encoder.
    Returns: (vision_embeds [N, 2048], deepstack_list [3 x [N, 2048]])
    """
    expected_shape = [int(d) for d in vision_session.inputs[0].shape]
    # Reshape/interpolate pixel_values to match expected input shape
    pv = pixel_values.to(dtype=torch.float16, device=vision_session.exec_device)
    if list(pv.shape) != expected_shape:
        if pv.ndim == 4:  # [N, C, H, W] -> [1, C, temporal, H, W]
            pv = pv.unsqueeze(2)
        if list(pv.shape) != expected_shape:
            pv = torch.nn.functional.interpolate(
                pv.reshape(-1, *pv.shape[-2:]).unsqueeze(1).float(),
                size=expected_shape[-2:], mode="bilinear", align_corners=False,
            ).to(torch.float16).reshape(expected_shape)

    outputs = vision_session.forward(pv)
    if isinstance(outputs, (list, tuple)):
        out_list = list(outputs)
    elif isinstance(outputs, dict):
        out_list = list(outputs.values())
    else:
        out_list = [outputs]

    # Convert to tensors
    result = []
    for o in out_list:
        if isinstance(o, np.ndarray):
            o = torch.from_numpy(o)
        result.append(o.to(torch.float16))

    vision_embeds = result[0]
    deepstack_list = result[1:4] if len(result) >= 4 else [torch.zeros_like(vision_embeds) for _ in range(3)]
    return vision_embeds, deepstack_list


def align_visual_tokens(tensor: torch.Tensor, target_tokens: int) -> torch.Tensor:
    """Align vision HMONNX token count to the processor's image token span.

    The checked-in tmp demo may pair golden vision HMONNX artifacts exported with
    a fixed 7x7 visual grid (49 tokens) with the local processor, which can emit
    a different number of image placeholder tokens (for example 64).  Pad by
    repeating the final visual token or trim excess tokens so assignment into the
    text prompt remains well-defined and the full fused demo can continue.
    """
    current_tokens = int(tensor.shape[0])
    if current_tokens == target_tokens:
        return tensor
    if current_tokens > target_tokens:
        return tensor[:target_tokens]
    if current_tokens == 0:
        return torch.zeros(target_tokens, THINKER_HIDDEN_SIZE, dtype=torch.float16, device=tensor.device)
    pad = tensor[-1:].expand(target_tokens - current_tokens, -1)
    return torch.cat([tensor, pad], dim=0)


# ═══════════════════════════════════════════════════════════════════════════════
# Text LLM (Thinker) - Prefill + Decode
# ═══════════════════════════════════════════════════════════════════════════════
def build_text_prefill_inputs(
    inputs_embeds: torch.Tensor,
    position_ids_3d: torch.Tensor,
    deepstack_tensors: List[torch.Tensor],
    actual_seq_len: int,
):
    """Build text_llm prefill inputs, padded to TEXT_LLM_STATIC_SEQ.
    Returns list of tensors ready for prefill_session.forward().
    """
    static_seq = TEXT_LLM_STATIC_SEQ
    dev = inputs_embeds.device

    # Pad inputs_embeds to static length
    if inputs_embeds.shape[1] < static_seq:
        pad = torch.zeros(1, static_seq - inputs_embeds.shape[1], THINKER_HIDDEN_SIZE, dtype=torch.float16, device=dev)
        inputs_embeds = torch.cat([inputs_embeds, pad], dim=1)

    # Pad position_ids (3D: [3, seq] -> each padded to static_seq)
    def pad_pos(pos, target):
        pos = pos.to(torch.int32)
        if pos.shape[0] >= target:
            return pos[:target].to(dev)
        return torch.cat([pos.to(dev), torch.zeros(target - pos.shape[0], dtype=torch.int32, device=dev)])

    time_pos = pad_pos(position_ids_3d[0], static_seq)
    height_pos = pad_pos(position_ids_3d[1], static_seq)
    width_pos = pad_pos(position_ids_3d[2], static_seq)

    # Pad deepstack tensors
    padded_ds = []
    for ds in deepstack_tensors:
        if ds.shape[1] < static_seq:
            pad = torch.zeros(1, static_seq - ds.shape[1], THINKER_HIDDEN_SIZE, dtype=torch.float16, device=dev)
            ds = torch.cat([ds, pad], dim=1)
        padded_ds.append(ds)

    past_seq_length = torch.tensor([0], dtype=torch.int32, device=dev)
    current_length = torch.tensor([actual_seq_len], dtype=torch.int32, device=dev)

    prefill_inputs = [inputs_embeds, time_pos, height_pos, width_pos,
                      past_seq_length, current_length] + padded_ds
    return prefill_inputs


def run_text_prefill(
    prefill_session: HMONNXInference,
    text_embedding: torch.Tensor,
    inputs_embeds: torch.Tensor,
    position_ids_3d: torch.Tensor,
    deepstack_tensors: List[torch.Tensor],
    actual_seq_len: int,
):
    """Run text LLM prefill, chunked if seq > TEXT_LLM_STATIC_SEQ."""
    num_layers = THINKER_NUM_LAYERS
    kv_shape = [int(d) for d in prefill_session.inputs[9].shape]
    kv_device = prefill_session.exec_device
    static_seq = TEXT_LLM_STATIC_SEQ

    past_key_caches = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16, device=kv_device)) for _ in range(num_layers)]
    past_value_caches = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16, device=kv_device)) for _ in range(num_layers)]

    all_hidden = []
    num_chunks = (actual_seq_len + static_seq - 1) // static_seq
    logger.info(f"Text prefill: seq_len={actual_seq_len}, static={static_seq}, chunks={num_chunks}")

    prefill_logits = None
    past_seq = 0
    for chunk_idx in range(num_chunks):
        start = chunk_idx * static_seq
        end = min(start + static_seq, actual_seq_len)
        chunk_len = end - start

        chunk_embeds = inputs_embeds[:, start:end, :]
        chunk_ds = [ds[:, start:end, :] for ds in deepstack_tensors]
        chunk_pos = torch.stack([position_ids_3d[i][start:end] for i in range(3)])

        chunk_inputs = build_text_prefill_inputs(
            chunk_embeds, chunk_pos, chunk_ds, chunk_len)
        # Override past_seq_length for chunked prefill
        chunk_inputs[4] = torch.tensor([past_seq], dtype=torch.int32, device=kv_device)

        prefill_out = prefill_session.forward(*chunk_inputs, *past_key_caches, *past_value_caches)
        if isinstance(prefill_out, (list, tuple)):
            prefill_logits = prefill_out[0]
            prefill_hidden = prefill_out[1] if len(prefill_out) > 1 else None
        else:
            prefill_logits = prefill_out
            prefill_hidden = None

        if prefill_hidden is not None:
            if isinstance(prefill_hidden, np.ndarray):
                prefill_hidden = torch.from_numpy(prefill_hidden)
            all_hidden.append(prefill_hidden[:, :chunk_len, :].to(torch.float16).cpu())

        past_seq += chunk_len

    if isinstance(prefill_logits, np.ndarray):
        prefill_logits = torch.from_numpy(prefill_logits)
    prefill_logits = prefill_logits.float()
    if prefill_logits.ndim == 2:
        prefill_logits = prefill_logits.unsqueeze(1)

    next_token = torch.argmax(prefill_logits[:, -1, :], dim=-1, keepdim=True)
    return next_token, past_key_caches, past_value_caches, kv_shape, all_hidden


def run_text_decode(
    decode_session: HMONNXInference,
    text_embedding: torch.Tensor,
    rope_deltas: torch.Tensor,
    actual_seq_len: int,
    max_new_tokens: int,
    next_token: torch.Tensor,
    past_key_caches, past_value_caches,
    kv_shape, all_hidden,
):
    """Run text LLM decode loop. Returns generated_ids and hidden_states."""
    kv_device = decode_session.exec_device
    generated_tokens = [next_token]

    decode_past_seq = torch.tensor([actual_seq_len], dtype=torch.int32, device=kv_device)
    one_length = torch.tensor([1], dtype=torch.int32, device=kv_device)
    kv_max_seq = kv_shape[2]
    zero_ds = [torch.zeros(1, 1, THINKER_HIDDEN_SIZE, dtype=torch.float16, device=kv_device) for _ in range(3)]

    for step in range(max_new_tokens - 1):
        token_id = int(next_token.item())
        if token_id in EOS_TOKEN_IDS:
            break
        if int(decode_past_seq.item()) + 1 > kv_max_seq:
            logger.warning(f"KV cache full at step {step+1}")
            break

        # Decode position_ids
        delta = int(decode_past_seq.item()) + int(rope_deltas.reshape(-1)[0].item())
        decode_pos = torch.tensor([delta], dtype=torch.int32, device=kv_device)

        token_embed = text_embedding[token_id].unsqueeze(0).unsqueeze(0).to(torch.float16)  # [1,1,2048]
        decode_inputs = [token_embed, decode_pos, decode_pos, decode_pos,
                         decode_past_seq, one_length] + zero_ds

        decode_out = decode_session.forward(*decode_inputs, *past_key_caches, *past_value_caches)
        if isinstance(decode_out, (list, tuple)):
            decode_logits = decode_out[0]
            decode_hidden = decode_out[1] if len(decode_out) > 1 else None
        else:
            decode_logits = decode_out
            decode_hidden = None

        if isinstance(decode_logits, np.ndarray):
            decode_logits = torch.from_numpy(decode_logits)
        decode_logits = decode_logits.float()
        if decode_logits.ndim == 2:
            decode_logits = decode_logits.unsqueeze(1)

        next_token = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        decode_past_seq = decode_past_seq + 1

        if decode_hidden is not None:
            if isinstance(decode_hidden, np.ndarray):
                decode_hidden = torch.from_numpy(decode_hidden)
            all_hidden.append(decode_hidden.to(torch.float16).cpu())

    generated_ids = torch.cat(generated_tokens, dim=-1)  # [1, N]
    hidden_states = torch.cat(all_hidden, dim=1) if all_hidden else None
    return generated_ids, hidden_states


# ═══════════════════════════════════════════════════════════════════════════════
# Talker Prefill Context Builder (fused source/role_mask/bypass semantics)
# ═══════════════════════════════════════════════════════════════════════════════
def build_talker_prefill_context(
    proj_sessions, text_embedding: torch.Tensor, talker_embedding: torch.Tensor,
    input_ids: torch.Tensor, full_ids: torch.Tensor,
    thinker_embed: torch.Tensor, thinker_hidden: torch.Tensor,
    speaker_name: str, use_fusion: bool = True,
):
    """Build fused talker prefill inputs, exactly reproducing `ptq.py`'s capture
    of the native HF `_get_talker_user_parts` / `_get_talker_assistant_parts`.

    The talker graph fuses as:
        projected = (1-role_mask)*hidden_proj(source) + role_mask*text_proj(source)
        inputs_embeds = (1-bypass_mask)*projected + bypass_mask*bypass_embeds
    i.e. a *select*, never a sum.  So the assistant codec region (which natively
    equals ``text_projection(text) + codec_embeds``) can ONLY be supplied through
    ``bypass_embeds`` as a precomputed sum — it cannot be reconstructed from
    ``source`` because ``bypass_mask=1`` there discards the projection.

    ``proj_sessions`` is the dict from ``load_hmonnx_projections`` (the standalone
    text/hidden projection HMONNX graphs, run on-chip — no onnxruntime, no torch matmul).

    use_fusion=True  : user segment + assistant 3-token prefix go through the
                       talker's baked projection (source, bypass_mask=0); only the
                       6 codec tokens + first text token bypass.  (matches ptq.py)
    use_fusion=False : everything is projected with the standalone HMONNX graphs and
                       supplied via bypass_embeds with bypass_mask=1 (the talker's
                       baked projection is unused). Default — avoids the imprecise
                       in-graph projection.

    Returns:
      (source, role_mask, bypass_embeds, bypass_mask, actual_len,
       trailing_text_hidden, tts_pad_proj)
    where trailing_text_hidden and tts_pad_proj are PROJECTED (1024-dim).
    """
    text_embedding = text_embedding.cpu().to(torch.float16)
    talker_embedding = talker_embedding.cpu().to(torch.float16)
    thinker_embed = thinker_embed.cpu().to(torch.float16)
    thinker_hidden = thinker_hidden.cpu().to(torch.float16)
    prompt_ids = input_ids[0].cpu().to(torch.long)
    full_seq = full_ids[0].cpu().to(torch.long)
    speaker_id = SPEAKER_ID_MAP.get(speaker_name.lower(), SPEAKER_ID_MAP["ethan"])

    def text_proj(x):  # [1, T, 2048] -> [1, T, 1024], standalone HMONNX, returns cpu fp16
        return hmonnx_text_projection(proj_sessions, x)

    def hidden_proj(x):
        return hmonnx_hidden_projection(proj_sessions, x)

    im_start_pos = torch.nonzero(prompt_ids == IM_START_TOKEN_ID).flatten()
    im_start_pos = torch.cat([im_start_pos, torch.tensor([full_seq.shape[0]])])

    mm_mask = ((full_seq == AUDIO_TOKEN_ID) | (full_seq == IMAGE_TOKEN_ID) | (full_seq == VIDEO_TOKEN_ID))

    # tts_*_embed in HF = text_projection(thinker_embed[special_token]) — PROJECTED 1024.
    tts_bos_p = text_proj(text_embedding[torch.tensor([TTS_BOS_TOKEN_ID])].unsqueeze(0))
    tts_eos_p = text_proj(text_embedding[torch.tensor([TTS_EOS_TOKEN_ID])].unsqueeze(0))
    tts_pad_p = text_proj(text_embedding[torch.tensor([TTS_PAD_TOKEN_ID])].unsqueeze(0))

    source_segs = []
    role_segs = []
    bypass_segs = []
    bypass_mask_segs = []
    trailing_text_hidden = None

    def emit(src, role, byp, bypm):
        """Append a segment; in no-fusion mode collapse projection into bypass."""
        if not use_fusion:
            projected = role * text_proj(src) + (1.0 - role) * hidden_proj(src)
            byp = projected * (1.0 - bypm) + byp * bypm
            src = torch.zeros_like(src)
            role = torch.zeros_like(role)
            bypm = torch.ones_like(bypm)
        source_segs.append(src)
        role_segs.append(role)
        bypass_segs.append(byp)
        bypass_mask_segs.append(bypm)

    for idx in range(len(im_start_pos) - 1):
        st, en = int(im_start_pos[idx]), int(im_start_pos[idx + 1])
        role_token = int(prompt_ids[st + 1])

        if role_token == SYSTEM_TOKEN_ID:
            continue

        if role_token == USER_TOKEN_ID:
            # source = thinker_embed, mm positions replaced by thinker_hidden; project internally
            seg_embed = thinker_embed[:, st:en, :].clone()
            seg_hidden = thinker_hidden[:, st:en, :]
            seg_mm = mm_mask[st:en]
            seg_len = en - st
            if seg_mm.any():
                seg_embed[:, seg_mm, :] = seg_hidden[:, seg_mm, :]
            emit(seg_embed,
                 (~seg_mm).unsqueeze(0).unsqueeze(-1).to(torch.float16),
                 torch.zeros(1, seg_len, TALKER_HIDDEN_SIZE, dtype=torch.float16),
                 torch.zeros(1, seg_len, 1, dtype=torch.float16))
            continue

        if role_token == ASSISTANT_TOKEN_ID and idx == len(im_start_pos) - 2:
            asst_embed = thinker_embed[:, st:en, :]
            asst_proj = text_proj(asst_embed)  # [1, L, 1024]
            codec_ids = [CODEC_NOTHINK_ID, CODEC_THINK_BOS_ID, CODEC_THINK_EOS_ID,
                         speaker_id, CODEC_PAD_ID, CODEC_BOS_ID]
            codec_embeds = talker_embedding[torch.tensor(codec_ids)].unsqueeze(0)  # [1,6,1024]

            # Native assistant input_embeds = assistant_text_hidden + assistant_codec_hidden
            assistant_text_hidden = torch.cat([
                asst_proj[:, :3, :],
                tts_pad_p.expand(1, 4, -1),
                tts_bos_p,
                asst_proj[:, 3:4, :],
            ], dim=1)  # [1, 9, 1024]
            assistant_codec_hidden = torch.cat([
                torch.zeros(1, 3, TALKER_HIDDEN_SIZE, dtype=torch.float16),
                codec_embeds,
            ], dim=1)  # [1, 9, 1024]
            input_embeds = assistant_text_hidden + assistant_codec_hidden
            asst_len = int(input_embeds.shape[1])

            # ptq capture: first `projected_prefix` tokens use the talker projection
            # (source=thinker_embed, bypass_mask=0); the rest bypass with the
            # precomputed input_embeds.
            projected_prefix = min(3, max(en - st, 0))
            assistant_source = torch.zeros(1, asst_len, THINKER_HIDDEN_SIZE, dtype=torch.float16)
            assistant_source[:, :projected_prefix, :] = asst_embed[:, :projected_prefix, :]
            assistant_role_mask = torch.ones(1, asst_len, 1, dtype=torch.float16)
            assistant_bypass = input_embeds.clone()
            assistant_bypass[:, :projected_prefix, :] = 0
            assistant_bypass_mask = torch.ones(1, asst_len, 1, dtype=torch.float16)
            assistant_bypass_mask[:, :projected_prefix, :] = 0

            emit(assistant_source, assistant_role_mask, assistant_bypass, assistant_bypass_mask)

            # trailing text for the decode loop is PROJECTED (1024-dim)
            trailing_text_hidden = torch.cat([asst_proj[:, 4:, :], tts_eos_p], dim=1)
            continue

        if role_token == ASSISTANT_TOKEN_ID:
            continue

    source = torch.cat(source_segs, dim=1)
    role_mask = torch.cat(role_segs, dim=1)
    bypass_embeds = torch.cat(bypass_segs, dim=1)
    bypass_mask = torch.cat(bypass_mask_segs, dim=1)
    actual_len = int(source.shape[1])
    return source, role_mask, bypass_embeds, bypass_mask, actual_len, trailing_text_hidden, tts_pad_p


# ═══════════════════════════════════════════════════════════════════════════════
# Talker + Predictor + Code2Wav
# ═══════════════════════════════════════════════════════════════════════════════
def run_talker_generate(
    talker_prefill_session: HMONNXInference,
    talker_decode_session: HMONNXInference,
    predictor_prefill_session: HMONNXInference,
    predictor_decode_session: HMONNXInference,
    talker_embedding: torch.Tensor,
    codec_embeddings: List[torch.Tensor],
    source: torch.Tensor,
    role_mask: torch.Tensor,
    bypass_embeds: torch.Tensor,
    bypass_mask: torch.Tensor,
    actual_len: int,
    trailing_text_hidden: torch.Tensor,
    tts_pad_proj: torch.Tensor,
    max_new_tokens: int,
) -> torch.Tensor:
    """Run talker + predictor decode loop. Returns codec codes [1, 16, N].

    Decode matches HF exactly: each step's talker input embedding is
        codec_residual_sum + projected_trailing_text[step]   (both 1024-dim)
    fed through bypass (bypass_mask=1, source=0).  ``trailing_text_hidden`` and
    ``tts_pad_proj`` are already text_projection-ed (1024-dim).
    """
    num_layers = TALKER_NUM_LAYERS
    kv_shape = [int(d) for d in talker_prefill_session.inputs[6].shape]
    pred_num_layers = PREDICTOR_NUM_LAYERS
    pred_kv_shape = [int(d) for d in predictor_prefill_session.inputs[4].shape]
    dev = talker_prefill_session.exec_device
    static_seq = int(talker_prefill_session.inputs[0].shape[1])

    talker_kc = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16, device=dev)) for _ in range(num_layers)]
    talker_vc = [CacheTensor(torch.zeros(kv_shape, dtype=torch.float16, device=dev)) for _ in range(num_layers)]

    # ── Talker Prefill (chunked if actual_len > TALKER_STATIC_SEQ) ──
    num_chunks = (actual_len + static_seq - 1) // static_seq
    talker_logits = None
    talker_hidden = None
    past_seq_val = 0

    for chunk_idx in range(num_chunks):
        start = chunk_idx * static_seq
        end = min(start + static_seq, actual_len)
        chunk_len = end - start

        padded_source = torch.zeros(1, static_seq, THINKER_HIDDEN_SIZE, dtype=torch.float16, device=dev)
        padded_role = torch.zeros(1, static_seq, 1, dtype=torch.float16, device=dev)
        padded_bypass = torch.zeros(1, static_seq, TALKER_HIDDEN_SIZE, dtype=torch.float16, device=dev)
        padded_bypass_mask = torch.zeros(1, static_seq, 1, dtype=torch.float16, device=dev)

        padded_source[:, :chunk_len, :] = source[:, start:end, :].to(dev)
        padded_role[:, :chunk_len, :] = role_mask[:, start:end, :].to(dev)
        padded_bypass[:, :chunk_len, :] = bypass_embeds[:, start:end, :].to(dev)
        padded_bypass_mask[:, :chunk_len, :] = bypass_mask[:, start:end, :].to(dev)
        past_seq = torch.tensor([past_seq_val], dtype=torch.int32, device=dev)
        cur_len = torch.tensor([chunk_len], dtype=torch.int32, device=dev)

        prefill_out = talker_prefill_session.forward(
            padded_source, padded_role, padded_bypass, padded_bypass_mask, past_seq, cur_len,
            *talker_kc, *talker_vc)
        if isinstance(prefill_out, (list, tuple)):
            talker_logits = prefill_out[0]
            talker_hidden = prefill_out[1]
        else:
            talker_logits = prefill_out
            talker_hidden = None

        past_seq_val += chunk_len

    if isinstance(talker_logits, np.ndarray):
        talker_logits = torch.from_numpy(talker_logits)
    if isinstance(talker_hidden, np.ndarray):
        talker_hidden = torch.from_numpy(talker_hidden)

    # Sample first codec token
    first_codec = int(torch.argmax(talker_logits.float().reshape(-1)).item())
    logger.info(f"Talker first codec: {first_codec}")
    all_codes = [[first_codec]]

    # ── Predictor for first token (prefill with 2 tokens: hidden + codec_embed) ──
    pred_kc = [CacheTensor(torch.zeros(pred_kv_shape, dtype=torch.float16, device=dev)) for _ in range(pred_num_layers)]
    pred_vc = [CacheTensor(torch.zeros(pred_kv_shape, dtype=torch.float16, device=dev)) for _ in range(pred_num_layers)]

    if talker_hidden is not None:
        # Use talker_embedding for primary token (NOT codec_embeddings[0])
        primary_embed = talker_embedding[first_codec].unsqueeze(0).unsqueeze(0).to(torch.float16).to(dev)
        pred_input = torch.cat([talker_hidden.to(torch.float16).reshape(1, 1, -1), primary_embed], dim=1)
        # head_mask: only group 0 active (shape: [1, seq_len, num_heads=15, 1])
        num_pred_heads = NUM_CODE_GROUPS - 1  # 15
        head_mask = torch.zeros(1, 2, num_pred_heads, 1, dtype=torch.float16, device=dev)
        head_mask[:, :, 0, 0] = 1.0
        pred_past = torch.tensor([0], dtype=torch.int32, device=dev)
        pred_cur = torch.tensor([2], dtype=torch.int32, device=dev)

        pred_out = predictor_prefill_session.forward(
            pred_input, head_mask, pred_past, pred_cur, *pred_kc, *pred_vc)
        if isinstance(pred_out, (list, tuple)):
            pred_logits = pred_out[0]
            pred_hidden_0 = pred_out[1] if len(pred_out) > 1 else None
        else:
            pred_logits = pred_out
            pred_hidden_0 = None
        if isinstance(pred_logits, np.ndarray):
            pred_logits = torch.from_numpy(pred_logits)
        if pred_hidden_0 is not None and isinstance(pred_hidden_0, np.ndarray):
            pred_hidden_0 = torch.from_numpy(pred_hidden_0)

        # Decode remaining codec groups (groups 1..14)
        step_codes = [first_codec]
        pred_logits_2d = pred_logits.float().reshape(-1, pred_logits.shape[-1])
        last_logits = pred_logits_2d[-1]
        first_res = int(torch.argmax(last_logits[:2048]).item())
        step_codes.append(first_res)
        # Store predictor hidden states for talker decode bypass
        mid_hiddens = []  # groups 1..14 hidden states
        ct = first_res
        for g in range(1, num_pred_heads):
            # Predictor decode: only activate head g
            codec_embed_g = codec_embeddings[g - 1][ct].unsqueeze(0).unsqueeze(0).to(torch.float16).to(dev)
            head_mask_d = torch.zeros(1, 1, num_pred_heads, 1, dtype=torch.float16, device=dev)
            head_mask_d[:, :, g, 0] = 1.0
            pred_past_d = torch.tensor([2 + g - 1], dtype=torch.int32, device=dev)
            pred_cur_d = torch.tensor([1], dtype=torch.int32, device=dev)
            pred_out_d = predictor_decode_session.forward(
                codec_embed_g, head_mask_d, pred_past_d, pred_cur_d, *pred_kc, *pred_vc)
            if isinstance(pred_out_d, (list, tuple)):
                pred_logits_d = pred_out_d[0]
                pred_hidden_d = pred_out_d[1] if len(pred_out_d) > 1 else None
            else:
                pred_logits_d = pred_out_d
                pred_hidden_d = None
            if isinstance(pred_logits_d, np.ndarray):
                pred_logits_d = torch.from_numpy(pred_logits_d)
            if pred_hidden_d is not None and isinstance(pred_hidden_d, np.ndarray):
                pred_hidden_d = torch.from_numpy(pred_hidden_d)
            last_logits = pred_logits_d.float().reshape(-1)
            ct = int(torch.argmax(last_logits[:2048]).item())
            step_codes.append(ct)
            mid_hiddens.append(pred_hidden_d)
        all_codes = [step_codes]
        last_mid_hiddens = mid_hiddens
        last_primary = first_codec

    # ── Talker Decode Loop ──
    talker_past_seq = torch.tensor([actual_len], dtype=torch.int32, device=dev)
    one_len = torch.tensor([1], dtype=torch.int32, device=dev)
    trailing_idx = 0
    trailing_len = int(trailing_text_hidden.shape[1]) if trailing_text_hidden is not None else 0

    for step in range(1, max_new_tokens):
        prev_codec = all_codes[-1][0]
        if prev_codec == CODEC_EOS_TOKEN_ID:
            break

        # Build bypass_embeds per original: primary_emb + mid_hiddens[1..14] + last_residual_emb
        primary_emb = talker_embedding[last_primary].unsqueeze(0).unsqueeze(0).to(torch.float16).to(dev)
        all_hidden_parts = [primary_emb]
        for h in last_mid_hiddens:
            if h is not None:
                all_hidden_parts.append(h.to(torch.float16).to(dev).reshape(1, 1, -1))
            else:
                all_hidden_parts.append(primary_emb)
        # Last residual embedding (predictor codec_embeddings for last group)
        last_res_token = all_codes[-1][-1]
        last_res_emb = codec_embeddings[num_pred_heads - 1][last_res_token].unsqueeze(0).unsqueeze(0).to(torch.float16).to(dev)
        all_hidden_parts.append(last_res_emb)
        codec_sum = torch.cat(all_hidden_parts, dim=1).sum(dim=1, keepdim=True)

        # Projected text injection (1024-dim): trailing text, then tts_pad padding.
        if trailing_idx < trailing_len:
            text_proj = trailing_text_hidden[:, trailing_idx:trailing_idx+1, :].to(device=dev, dtype=torch.float16)
        else:
            text_proj = tts_pad_proj.to(device=dev, dtype=torch.float16)
        trailing_idx += 1

        # HF decode: inputs_embeds = codec_residual_sum + projected_text. The talker
        # fusion is a select, not a sum, so the sum MUST be precomputed and fed via
        # bypass (bypass_mask=1, source=0).
        decode_bypass = (codec_sum + text_proj).to(dtype=torch.float16, device=dev)
        d_source = torch.zeros(1, 1, THINKER_HIDDEN_SIZE, dtype=torch.float16, device=dev)
        d_role = torch.zeros(1, 1, 1, dtype=torch.float16, device=dev)
        d_bpm = torch.ones(1, 1, 1, dtype=torch.float16, device=dev)

        decode_out = talker_decode_session.forward(
            d_source, d_role, decode_bypass, d_bpm, talker_past_seq, one_len,
            *talker_kc, *talker_vc)
        if isinstance(decode_out, (list, tuple)):
            d_logits = decode_out[0]
            d_hidden = decode_out[1]
        else:
            d_logits = decode_out
            d_hidden = None

        if isinstance(d_logits, np.ndarray):
            d_logits = torch.from_numpy(d_logits)
        if isinstance(d_hidden, np.ndarray):
            d_hidden = torch.from_numpy(d_hidden)

        next_codec = int(torch.argmax(d_logits.float().reshape(-1)).item())
        talker_past_seq = talker_past_seq + 1

        # Don't include EOS step in codes - code2wav only accepts valid codec IDs (0-2047)
        if next_codec == CODEC_EOS_TOKEN_ID:
            break

        step_codes = [next_codec]
        if d_hidden is not None:
            # Use talker_embedding for primary token (NOT codec_embeddings[0])
            primary_embed = talker_embedding[next_codec].unsqueeze(0).unsqueeze(0).to(torch.float16).to(dev)
            pred_input = torch.cat([d_hidden.to(torch.float16).reshape(1, 1, -1), primary_embed], dim=1)
            # head_mask: only group 0 active
            head_mask = torch.zeros(1, 2, num_pred_heads, 1, dtype=torch.float16, device=dev)
            head_mask[:, :, 0, 0] = 1.0
            # Reset predictor KV for each step
            for c in pred_kc: c.data.zero_()
            for c in pred_vc: c.data.zero_()
            pred_past = torch.tensor([0], dtype=torch.int32, device=dev)
            pred_cur = torch.tensor([2], dtype=torch.int32, device=dev)

            pred_out = predictor_prefill_session.forward(
                pred_input, head_mask, pred_past, pred_cur, *pred_kc, *pred_vc)
            if isinstance(pred_out, (list, tuple)):
                pred_logits = pred_out[0]
            else:
                pred_logits = pred_out
            if isinstance(pred_logits, np.ndarray):
                pred_logits = torch.from_numpy(pred_logits)

            pred_logits_2d = pred_logits.float().reshape(-1, pred_logits.shape[-1])
            last_logits = pred_logits_2d[-1]
            first_res = int(torch.argmax(last_logits[:2048]).item())
            step_codes.append(first_res)
            mid_hiddens = []
            ct = first_res
            for g in range(1, num_pred_heads):
                codec_embed_g = codec_embeddings[g - 1][ct].unsqueeze(0).unsqueeze(0).to(torch.float16).to(dev)
                head_mask_d = torch.zeros(1, 1, num_pred_heads, 1, dtype=torch.float16, device=dev)
                head_mask_d[:, :, g, 0] = 1.0
                pred_past_d = torch.tensor([2 + g - 1], dtype=torch.int32, device=dev)
                pred_cur_d = torch.tensor([1], dtype=torch.int32, device=dev)
                pred_out_d = predictor_decode_session.forward(
                    codec_embed_g, head_mask_d, pred_past_d, pred_cur_d, *pred_kc, *pred_vc)
                if isinstance(pred_out_d, (list, tuple)):
                    pred_logits_d = pred_out_d[0]
                    pred_hidden_d = pred_out_d[1] if len(pred_out_d) > 1 else None
                else:
                    pred_logits_d = pred_out_d
                    pred_hidden_d = None
                if isinstance(pred_logits_d, np.ndarray):
                    pred_logits_d = torch.from_numpy(pred_logits_d)
                if pred_hidden_d is not None and isinstance(pred_hidden_d, np.ndarray):
                    pred_hidden_d = torch.from_numpy(pred_hidden_d)
                last_logits = pred_logits_d.float().reshape(-1)
                ct = int(torch.argmax(last_logits[:2048]).item())
                step_codes.append(ct)
                mid_hiddens.append(pred_hidden_d)
            last_mid_hiddens = mid_hiddens
            last_primary = next_codec

        all_codes.append(step_codes)

    # Build codes tensor [1, 16, N]
    num_steps = len(all_codes)
    codes = torch.zeros(1, NUM_CODE_GROUPS, num_steps, dtype=torch.int32, device=dev)
    for t, sc in enumerate(all_codes):
        for g in range(min(len(sc), NUM_CODE_GROUPS)):
            codes[0, g, t] = sc[g]

    logger.info(f"Talker generated {num_steps} steps")
    return codes


# ═══════════════════════════════════════════════════════════════════════════════
# Code2Wav
# ═══════════════════════════════════════════════════════════════════════════════
def run_code2wav(code2wav_session: HMONNXInference, codes: torch.Tensor) -> torch.Tensor:
    """Run code2wav HMONNX with left-context chunked decode (matches original).
    codes: [1, 16, N] -> audio waveform. Codes must be in range [0, 2047].
    """
    expected_shape = [int(d) for d in code2wav_session.inputs[0].shape]
    static_seq = expected_shape[2]
    codes_input = codes.to(torch.int32)
    actual_len = codes_input.shape[2]
    left_context = 25

    def _run_single(chunk):
        cl = chunk.shape[2]
        if cl < static_seq:
            chunk = torch.nn.functional.pad(chunk, (0, static_seq - cl))
        out = code2wav_session.forward(chunk)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if isinstance(out, np.ndarray):
            out = torch.from_numpy(out)
        return out[..., :cl * CODE2WAV_UPSAMPLE]

    if actual_len <= static_seq:
        return _run_single(codes_input)

    # Chunked decode with left context overlap (same as original)
    safe_chunk = max(1, static_seq - left_context)
    all_audio = []
    si = 0
    while si < actual_len:
        ei = min(si + safe_chunk, actual_len)
        ctx = left_context if si - left_context > 0 else si
        chunk_len = ei - si + ctx
        if chunk_len > static_seq:
            ctx = max(0, static_seq - (ei - si))
        chunk = codes_input[:, :, si - ctx:ei]
        wav = _run_single(chunk)
        all_audio.append(wav[..., ctx * CODE2WAV_UPSAMPLE:])
        si = ei

    return torch.cat(all_audio, dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# Position IDs (3D M-RoPE) - computed from attention_mask without HF model
# ═══════════════════════════════════════════════════════════════════════════════
def compute_position_ids_simple(input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """Compute simple 3D position_ids from attention_mask (no multimodal RoPE).
    For multimodal inputs, this is an approximation. Full accuracy requires
    native model's get_rope_index().
    Returns: (position_ids_3d [3, seq], rope_deltas [1,1])
    """
    seq_len = int(attention_mask.shape[1])
    position_ids = attention_mask.float().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    position_ids = position_ids.to(torch.int32)[0]
    # 3D: same for all dimensions (approximation for non-multimodal)
    position_ids_3d = position_ids.unsqueeze(0).expand(3, -1)
    max_pos = position_ids.max()
    rope_deltas = (max_pos + 1 - attention_mask.sum(dim=-1)).to(torch.long).reshape(1, 1)
    return position_ids_3d, rope_deltas


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    args = get_args()
    hmquant_dir = Path(args.hmquant_dir)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── GPU device setup ──
    # When CUDA_VISIBLE_DEVICES=1 is set, GPU1 appears as cuda:0 in the process
    # When not set, use cuda:1 directly
    gpu_id = 0 if os.environ.get("CUDA_VISIBLE_DEVICES") else 1
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    logger.info(f"Using device: {device}")

    t0 = time.time()

    # ── Load tokenizer & processor ──
    logger.info("Loading tokenizer and processor...")
    from xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe import Qwen3OmniMoeProcessor
    processor = Qwen3OmniMoeProcessor.from_pretrained(str(model_dir))
    tokenizer = processor.tokenizer

    # ── Load embeddings (on GPU) ──
    logger.info("Loading embeddings from hmquant...")
    text_embedding = load_text_embedding(hmquant_dir).to(device)
    talker_embedding = load_talker_embedding(hmquant_dir).to(device)
    codec_embeddings = [e.to(device) for e in load_codec_embeddings(hmquant_dir)]
    logger.info(f"  text_embedding: {text_embedding.shape}")
    logger.info(f"  talker_embedding: {talker_embedding.shape}")
    logger.info(f"  codec_embeddings: {len(codec_embeddings)} x {codec_embeddings[0].shape}")

    if args.load_thinker:
        # ── Fast path: reuse cached text-LLM outputs, skip audio/visual/thinker ──
        logger.info(f"Loading thinker cache from {args.load_thinker} (skipping text-LLM stages)")
        ck = torch.load(args.load_thinker, map_location="cpu", weights_only=False)
        input_ids = ck["input_ids"]
        generated_ids = ck["generated_ids"]
        inputs_embeds = ck["inputs_embeds"].to(device)
        hidden_states = ck["hidden_states"]
        output_text = ck["output_text"]
        logger.info(f"Generated text (cached): {output_text}")
    else:
        # ═══════════════════════════════════════════════════════════════════════
        # Stage 1: Load audio + visual + text_prefill → encode + prefill
        # ═══════════════════════════════════════════════════════════════════════
        logger.info("Stage 1: Loading audio/visual/prefill to GPU...")
        audio_session = HMONNXInference(str(hmquant_dir / "audio" / "hmquant_qwen3-omni_with_act.onnx"))
        audio_session.to(device)
        visual_session = HMONNXInference(str(hmquant_dir / "visual" / "hmquant_qwen3-omni_with_act.onnx"))
        visual_session.to(device)
        prefill_session = HMONNXInference(str(hmquant_dir / "prefill" / "hmquant_qwen3-omni_with_act.onnx"))
        prefill_session.to(device)
        logger.info(f"Stage 1 models loaded. ({time.time()-t0:.1f}s)")

        # ── Prepare input ──
        logger.info("Preparing input...")
        try:
            from qwen_omni_utils import process_mm_info
        except ImportError:
            def _load_audio_path(audio):
                if not isinstance(audio, str):
                    return audio
                audio_array, _ = sf.read(audio, dtype="float32")
                if audio_array.ndim > 1:
                    audio_array = audio_array.mean(axis=1)
                return audio_array

            def process_mm_info(conversation, use_audio_in_video=False):
                audios, images, videos = [], [], []
                for turn in conversation:
                    for item in turn.get("content", []):
                        item_type = item.get("type")
                        if item_type == "audio":
                            audios.append(_load_audio_path(item.get("audio")))
                        elif item_type == "image":
                            images.append(item.get("image"))
                        elif item_type == "video":
                            videos.append(item.get("video"))
                return audios, images, videos

        content = []
        if args.image:
            content.append({"type": "image", "image": args.image})
        if args.audio:
            content.append({"type": "audio", "audio": args.audio})
        content.append({"type": "text", "text": args.prompt})
        conversation = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = processor(text=text, audio=audios or None, images=images or None, videos=videos or None,
                           return_tensors="pt", padding=True,
                           seconds_per_chunk=2.0, position_id_per_seconds=13,
                           use_audio_in_video=True)

        input_ids = inputs["input_ids"].cpu()
        attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids)).cpu()
        actual_seq_len = int(input_ids.shape[1])
        logger.info(f"Input sequence length: {actual_seq_len}")

        # ── Phase 1: Encode multimodal ──
        logger.info("Phase 1: Encoding multimodal inputs...")

        # Build inputs_embeds from token embedding
        inputs_embeds = text_embedding[input_ids[0]].unsqueeze(0).to(torch.float16)  # [1, seq, 2048]

        # Audio encoding
        deepstack_tensors = [torch.zeros(1, actual_seq_len, THINKER_HIDDEN_SIZE, dtype=torch.float16, device=device) for _ in range(3)]
        if "input_features" in inputs and "feature_attention_mask" in inputs:
            logger.info("  Running audio encoder...")
            audio_embeds = run_audio_encoder(
                audio_session, inputs["input_features"].cpu(), inputs["feature_attention_mask"].cpu())
            audio_mask = (input_ids[0] == AUDIO_TOKEN_ID)
            num_audio_tokens = int(audio_mask.sum().item())
            audio_embeds = audio_embeds[:num_audio_tokens]
            inputs_embeds[0, audio_mask] = audio_embeds.to(torch.float16)
            logger.info(f"  Audio: {audio_embeds.shape[0]} tokens injected")

        # Visual encoding
        pv_key = "hm_pixel_values" if "hm_pixel_values" in inputs else "pixel_values"
        if pv_key in inputs:
            logger.info("  Running visual encoder...")
            vision_embeds, ds_list = run_visual_encoder(visual_session, inputs[pv_key].cpu())
            image_mask = (input_ids[0] == IMAGE_TOKEN_ID)
            num_image_tokens = int(image_mask.sum().item())
            if int(vision_embeds.shape[0]) != num_image_tokens:
                logger.warning(
                    f"  Visual token count mismatch: encoder={int(vision_embeds.shape[0])}, "
                    f"prompt={num_image_tokens}; aligning for demo inference"
                )
                vision_embeds = align_visual_tokens(vision_embeds, num_image_tokens)
                ds_list = [align_visual_tokens(ds, num_image_tokens) for ds in ds_list]
            inputs_embeds[0, image_mask] = vision_embeds.to(torch.float16)
            # Build dense deepstack tensors
            for i, ds in enumerate(ds_list):
                dense = torch.zeros(1, actual_seq_len, THINKER_HIDDEN_SIZE, dtype=torch.float16, device=device)
                dense[0, image_mask] = ds.to(torch.float16)
                deepstack_tensors[i] = dense
            logger.info(f"  Visual: {vision_embeds.shape[0]} tokens injected")

        # ── Phase 2: Text generation ──
        logger.info("Phase 2: Text LLM generation...")
        position_ids_3d, rope_deltas = compute_position_ids_simple(input_ids, attention_mask)

        # Run prefill (prefill_session already loaded in Stage 1)
        next_token, past_key_caches, past_value_caches, kv_shape, all_hidden = run_text_prefill(
            prefill_session, text_embedding, inputs_embeds,
            position_ids_3d, deepstack_tensors, actual_seq_len,
        )

        # Stage 2: Unload audio/visual/prefill, load decode
        logger.info("Stage 2: Unloading audio/visual/prefill, loading decode...")
        del audio_session, visual_session, prefill_session
        torch.cuda.empty_cache()
        decode_session = HMONNXInference(str(hmquant_dir / "decode" / "hmquant_qwen3-omni_with_act.onnx"))
        decode_session.to(device)
        logger.info(f"Stage 2 ready. ({time.time()-t0:.1f}s)")

        generated_ids, hidden_states = run_text_decode(
            decode_session, text_embedding, rope_deltas,
            actual_seq_len, args.max_new_tokens,
            next_token, past_key_caches, past_value_caches, kv_shape, all_hidden,
        )

        output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        logger.info(f"Generated text: {output_text}")

        del decode_session
        torch.cuda.empty_cache()

        if args.save_thinker:
            torch.save({
                "input_ids": input_ids,
                "generated_ids": generated_ids.cpu(),
                "inputs_embeds": inputs_embeds.cpu(),
                "hidden_states": hidden_states.cpu() if hidden_states is not None else None,
                "output_text": output_text,
            }, args.save_thinker)
            logger.info(f"Saved thinker cache to {args.save_thinker}")

    # ── Phase 3: Audio generation ──
    if args.enable_audio_generation:
        logger.info("Phase 3: Audio generation...")

        # Stage 3: load talker/predictor/code2wav (+ standalone projection HMONNX)
        logger.info("Stage 3: loading talker models...")
        torch.cuda.empty_cache()

        # text_projection / hidden_projection run on-chip via their own HMONNX
        # graphs (output/xh2/hmquant/{text,hidden}_projection). These standalone
        # graphs match the fp16 reference (cos≈1.0); the talker's *baked* projection
        # is imprecise, so we project with these and feed the talker through bypass.
        proj_sessions = load_hmonnx_projections(hmquant_dir, device)

        talker_prefill_session = HMONNXInference(str(hmquant_dir / "talker_prefill" / "hmquant_qwen3-omni_with_act.onnx"))
        talker_prefill_session.to(device)
        talker_decode_session = HMONNXInference(str(hmquant_dir / "talker_decode" / "hmquant_qwen3-omni_with_act.onnx"))
        talker_decode_session.to(device)
        predictor_prefill_session = HMONNXInference(str(hmquant_dir / "talker_prediction_prefill" / "hmquant_qwen3-omni_with_act.onnx"))
        predictor_prefill_session.to(device)
        predictor_decode_session = HMONNXInference(str(hmquant_dir / "talker_prediction_decode" / "hmquant_qwen3-omni_with_act.onnx"))
        predictor_decode_session.to(device)
        code2wav_session = HMONNXInference(str(hmquant_dir / "code2wav" / "hmquant_qwen3-omni_with_act.onnx"))
        code2wav_session.to(device)
        logger.info(f"Stage 3 models loaded. ({time.time()-t0:.1f}s)")

        # Build thinker_embed and thinker_hidden for talker
        # thinker_embed = inputs_embeds (prefill portion)
        # For generated tokens, use their embeddings
        gen_embeds = text_embedding[generated_ids[0]].unsqueeze(0).to(torch.float16)
        full_embed = torch.cat([inputs_embeds, gen_embeds], dim=1)  # [1, total, 2048]
        full_ids = torch.cat([input_ids, generated_ids.cpu()], dim=1)

        # thinker_hidden: pad hidden_states to match full_embed length
        # hidden_states covers [input_seq + decode_steps], full_embed covers [input_seq + all_generated]
        if hidden_states is not None:
            gap = full_embed.shape[1] - hidden_states.shape[1]
            if gap > 0:
                pad_h = hidden_states[:, -1:, :].expand(1, gap, -1)
                thinker_hidden = torch.cat([hidden_states, pad_h], dim=1)
            else:
                thinker_hidden = hidden_states[:, :full_embed.shape[1], :]
        else:
            thinker_hidden = full_embed

        source, role_mask, bypass_embeds, bypass_mask, talker_actual_len, trailing_text_hidden, tts_pad_proj = build_talker_prefill_context(
            proj_sessions, text_embedding, talker_embedding,
            input_ids, full_ids, full_embed, thinker_hidden, args.speaker,
            use_fusion=args.use_fusion,
        )

        codes = run_talker_generate(
            talker_prefill_session, talker_decode_session,
            predictor_prefill_session, predictor_decode_session,
            talker_embedding, codec_embeddings,
            source, role_mask, bypass_embeds, bypass_mask,
            talker_actual_len, trailing_text_hidden, tts_pad_proj,
            args.talker_max_new_tokens,
        )

        # Code2Wav
        logger.info("Running code2wav...")
        audio_wav = run_code2wav(code2wav_session, codes)
        logger.info(f"Audio waveform: {audio_wav.shape}")

        # Save audio
        wav_path = output_dir / "demo_hmonnx_full_output.wav"
        wav_data = audio_wav.float().cpu().numpy().flatten()
        sf.write(str(wav_path), wav_data, 24000)
        logger.info(f"Audio saved: {wav_path}")

    total_time = time.time() - t0
    logger.info(f"Total time: {total_time:.1f}s")
    print(f"\n[Result] text: {output_text}")
    if args.enable_audio_generation:
        print(f"[Result] audio: {wav_path}")


if __name__ == "__main__":
    main()
