#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: export_meta_seacoparaformer.py
# Description:
#   Export utilities for funasr in HOUMO AI xh2modelzoo.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import inspect
import os
import pickle
import torch

from funasr.register import tables
import numpy as np

STATIC_BATCH = 1
STATIC_FEATS_LEN = 334
STATIC_TGT_LEN = 100
USE_PREDICTOR_CUSTOM_EXPORT = os.environ.get("FUNASR_PREDICTOR_CUSTOM_EXPORT", "1") != "0"
KEEP_PREDICTOR_LSTM_SUBGRAPH = os.environ.get("FUNASR_PREDICTOR_KEEP_LSTM", "1") != "0"


def _mask_to_lengths(mask: torch.Tensor, speech: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.full(
            (speech.size(0),), speech.size(1), dtype=torch.int32, device=speech.device
        )
    if isinstance(mask, (tuple, list)):
        mask = mask[0]
    if mask.dim() == 3:
        if mask.shape[1] == 1:
            lengths = mask.squeeze(1).sum(dim=-1)
        else:
            lengths = mask.sum(dim=1).squeeze(-1)
    else:
        lengths = mask.sum(dim=-1)
    return lengths.to(dtype=torch.int32)


def _lengths_from_mask(mask: torch.Tensor) -> torch.Tensor:
    if isinstance(mask, (tuple, list)):
        mask = mask[0]
    if mask.dim() == 3:
        if mask.shape[1] == 1:
            lengths = mask.squeeze(1).sum(dim=-1)
        elif mask.shape[2] == 1:
            lengths = mask.sum(dim=1).squeeze(-1)
        else:
            lengths = mask.sum(dim=-1)
    else:
        lengths = mask.sum(dim=-1)
    return lengths.to(dtype=torch.int32)


def _to_predictor_mask(mask: torch.Tensor) -> torch.Tensor:
    if isinstance(mask, (tuple, list)):
        mask = mask[0]
    if mask.dim() == 2:
        return mask[:, None, :]
    if mask.dim() == 3:
        if mask.shape[1] == 1:
            return mask
        if mask.shape[2] == 1:
            return mask.transpose(1, 2)
    return mask


def _pad_or_trim_time(x: torch.Tensor, target_len: int) -> torch.Tensor:
    cur_len = x.size(1)
    if cur_len == target_len:
        return x
    if cur_len > target_len:
        return x[:, :target_len, :]
    pad = x.new_zeros((x.size(0), target_len - cur_len, x.size(2)))
    return torch.cat([x, pad], dim=1)


def _lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    row = torch.arange(0, max_len, device=lengths.device)
    mask = row[None, :] < lengths[:, None]
    return mask.to(dtype=torch.float32)


def _can_use_predictor_custom_export(predictor: torch.nn.Module) -> bool:
    required = (
        "pad",
        "cif_conv1d",
        "cif_output",
        "smooth_factor",
        "noise_threshold",
        "tail_process_fn",
    )
    return all(hasattr(predictor, name) for name in required)


def _can_use_predictor_lstm_subgraph(predictor: torch.nn.Module) -> bool:
    required = (
        "upsample_times",
        "upsample_cnn",
        "blstm",
        "cif_output2",
        "smooth_factor2",
        "noise_threshold2",
    )
    return all(hasattr(predictor, name) for name in required)


def _cif_v1_export_no_loop(hidden: torch.Tensor, alphas: torch.Tensor):
    # Vectorized CIF implementation: equivalent to cif_export but emits no Loop node.
    device = hidden.device
    dtype = hidden.dtype
    batch_size, _, hidden_size = hidden.size()

    prefix_sum = torch.cumsum(alphas, dim=1, dtype=torch.float64).to(torch.float32)
    prefix_sum_floor = torch.floor(prefix_sum)
    dislocation_prefix_sum = torch.roll(prefix_sum, 1, dims=1)
    dislocation_prefix_sum_floor = torch.floor(dislocation_prefix_sum)
    dislocation_prefix_sum_floor[:, 0] = 0
    dislocation_diff = prefix_sum_floor - dislocation_prefix_sum_floor

    fire_idxs = dislocation_diff > 0
    fires = torch.zeros_like(alphas, dtype=dtype)
    fires[fire_idxs] = 1
    fires = fires + prefix_sum - prefix_sum_floor

    prefix_sum_hidden = torch.cumsum(
        alphas.unsqueeze(-1).repeat(1, 1, hidden_size) * hidden, dim=1
    )
    frames = prefix_sum_hidden[fire_idxs]
    shift_frames = torch.roll(frames, 1, dims=0)

    batch_len = fire_idxs.sum(1)
    batch_idxs = torch.cumsum(batch_len, dim=0)
    shift_batch_idxs = torch.roll(batch_idxs, 1, dims=0)
    shift_batch_idxs[0] = 0
    shift_frames[shift_batch_idxs] = 0

    remains = fires - torch.floor(fires)
    remain_frames = remains[fire_idxs].unsqueeze(-1).repeat(1, hidden_size) * hidden[fire_idxs]
    shift_remain_frames = torch.roll(remain_frames, 1, dims=0)
    shift_remain_frames[shift_batch_idxs] = 0
    frames = frames - shift_frames + shift_remain_frames - remain_frames

    max_label_len = torch.floor(alphas.sum(dim=-1)).max().to(dtype=torch.int64)
    frame_fires = torch.zeros(batch_size, max_label_len, hidden_size, dtype=dtype, device=device)
    indices = torch.arange(max_label_len, device=device).expand(batch_size, -1)
    frame_fires[indices < batch_len.unsqueeze(1)] = frames
    return frame_fires, fires


def _predictor_lstm_token_num(
    predictor: torch.nn.Module,
    enc: torch.Tensor,
    mask: torch.Tensor,
    token_num: torch.Tensor,
) -> torch.Tensor:
    # Keep the original BiCif upsample BLSTM path in graph for export compatibility checks.
    context = enc.transpose(1, 2)
    output2 = predictor.upsample_cnn(context).transpose(1, 2)
    output2, _ = predictor.blstm(output2)
    alphas2 = torch.sigmoid(predictor.cif_output2(output2))
    alphas2 = torch.nn.functional.relu(
        alphas2 * predictor.smooth_factor2 - predictor.noise_threshold2
    )

    mask2 = mask.repeat(1, predictor.upsample_times, 1).transpose(-1, -2).reshape(
        alphas2.shape[0], -1
    )
    alphas2 = alphas2 * mask2.unsqueeze(-1)
    alphas2 = alphas2.squeeze(-1)
    token_num2 = alphas2.sum(-1)
    alphas2 = alphas2 * (token_num / token_num2)[:, None].repeat(1, alphas2.size(1))
    return alphas2.sum(-1)


def _predictor_forward_equivalent_no_loop(
    predictor: torch.nn.Module,
    enc: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    context = enc.transpose(1, 2)
    queries = predictor.pad(context)
    output = torch.relu(predictor.cif_conv1d(queries)).transpose(1, 2)

    output = predictor.cif_output(output)
    alphas = torch.sigmoid(output)
    alphas = torch.nn.functional.relu(alphas * predictor.smooth_factor - predictor.noise_threshold)
    mask_3d = mask.transpose(-1, -2).float()
    alphas = alphas * mask_3d
    alphas = alphas.squeeze(-1)

    mask_2d = mask_3d.squeeze(-1)
    hidden, alphas, token_num = predictor.tail_process_fn(enc, alphas, mask=mask_2d)
    pre_acoustic_embeds, _ = _cif_v1_export_no_loop(hidden, alphas)
    token_num_floor = torch.floor(token_num)

    if KEEP_PREDICTOR_LSTM_SUBGRAPH and _can_use_predictor_lstm_subgraph(predictor):
        # Add a zero-impact dependency so ONNX keeps the original BLSTM subgraph.
        token_num_lstm = _predictor_lstm_token_num(predictor, enc, mask, token_num_floor)
        token_num_floor = token_num_floor + token_num_lstm.to(token_num_floor.dtype) * 0.0

    return pre_acoustic_embeds, token_num_floor.to(dtype=torch.int32)


def _encoder_forward_with_mask(
    self,
    speech: torch.Tensor,
    speech_mask: torch.Tensor,
    online: bool = False,
):
    if not online:
        speech = speech * self._output_size**0.5

    if self.embed is None:
        xs_pad = speech
    else:
        xs_pad = self.embed(speech)

    if isinstance(speech_mask, (tuple, list)):
        mask = speech_mask
    else:
        mask = self.prepare_mask(speech_mask)

    encoder_outs = self.model.encoders0(xs_pad, mask)
    xs_pad = encoder_outs[0]

    encoder_outs = self.model.encoders(xs_pad, mask)
    xs_pad = encoder_outs[0]

    xs_pad = self.model.after_norm(xs_pad)

    if self.ctc_linear is not None:
        xs_pad = self.ctc_linear(xs_pad)
        xs_pad = torch.softmax(xs_pad, dim=2)

    enc_lens = _mask_to_lengths(speech_mask, speech)
    return xs_pad, enc_lens


def _decoder_forward_with_mask_sanm(
    self,
    hs_pad: torch.Tensor,
    enc_mask: torch.Tensor,
    ys_in_pad: torch.Tensor,
    tgt_mask: torch.Tensor,
    return_hidden: bool = False,
    return_both: bool = False,
):
    tgt = ys_in_pad
    ys_in_lens = _lengths_from_mask(tgt_mask)
    tgt_mask, _ = self.prepare_mask(tgt_mask)

    memory = hs_pad
    if isinstance(enc_mask, (tuple, list)):
        _, memory_mask = enc_mask
    else:
        _, memory_mask = self.prepare_mask(enc_mask)

    x = tgt
    x, tgt_mask, memory, memory_mask, _ = self.model.decoders(x, tgt_mask, memory, memory_mask)
    if getattr(self.model, "decoders2", None) is not None:
        x, tgt_mask, memory, memory_mask, _ = self.model.decoders2(
            x, tgt_mask, memory, memory_mask
        )
    x, tgt_mask, memory, memory_mask, _ = self.model.decoders3(x, tgt_mask, memory, memory_mask)
    hidden = self.after_norm(x)

    if self.output_layer is not None and return_hidden is False:
        x = self.output_layer(hidden)
        return x, ys_in_lens
    if return_both:
        x = self.output_layer(hidden)
        return x, hidden, ys_in_lens
    return hidden, ys_in_lens


def _decoder_forward_with_mask_basic(
    self,
    hs_pad: torch.Tensor,
    enc_mask: torch.Tensor,
    ys_in_pad: torch.Tensor,
    tgt_mask: torch.Tensor,
):
    tgt = ys_in_pad
    ys_in_lens = _lengths_from_mask(tgt_mask)
    tgt_mask, _ = self.prepare_mask(tgt_mask)

    memory = hs_pad
    if isinstance(enc_mask, (tuple, list)):
        _, memory_mask = enc_mask
    else:
        _, memory_mask = self.prepare_mask(enc_mask)

    x = tgt
    x, tgt_mask, memory, memory_mask = self.model.decoders(x, tgt_mask, memory, memory_mask)
    x = self.after_norm(x)
    if self.output_layer is not None:
        x = self.output_layer(x)

    return x, ys_in_lens

class ContextualEmbedderExport(torch.nn.Module):
    def __init__(
        self,
        model,
        max_seq_len=512,
        feats_dim=560,
        **kwargs,
    ):
        super().__init__()
        self.embedding = model.decoder.embed  # model.bias_embed
        model.bias_encoder.batch_first = False
        self.bias_encoder = model.bias_encoder

    def forward(self, hotword):
        hotword = self.embedding(hotword).transpose(0, 1)  # batch second
        hw_embed, (_, _) = self.bias_encoder(hotword)
        return hw_embed

    def export_dummy_inputs(self):
        hotword = torch.tensor(
            [
                [10, 11, 12, 13, 14, 10, 11, 12, 13, 14],
                [100, 101, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [10, 11, 12, 13, 14, 10, 11, 12, 13, 14],
                [100, 101, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            dtype=torch.int32,
        )
        # hotword_length = torch.tensor([10, 2, 1], dtype=torch.int32)
        return hotword

    def export_input_names(self):
        return ["hotword"]

    def export_output_names(self):
        return ["hw_embed"]

    def export_dynamic_axes(self):
        return {
            "hotword": {
                0: "num_hotwords",
            },
            "hw_embed": {
                1: "num_hotwords",
            },
        }

    def export_name(self):
        return "model_eb.onnx"


def export_rebuild_model(model, **kwargs):
    model.device = kwargs.get("device")
    is_onnx = kwargs.get("type", "onnx") == "onnx"
    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    model.encoder = encoder_class(model.encoder, onnx=is_onnx)

    predictor_class = tables.predictor_classes.get(kwargs["predictor"] + "Export")
    model.predictor = predictor_class(model.predictor, onnx=is_onnx)

    # before decoder convert into export class
    embedder_class = ContextualEmbedderExport
    embedder_model = embedder_class(model, onnx=is_onnx)

    decoder_class = tables.decoder_classes.get(kwargs["decoder"] + "Export")
    model.decoder = decoder_class(model.decoder, onnx=is_onnx)

    seaco_decoder_class = tables.decoder_classes.get(kwargs["seaco_decoder"] + "Export")
    model.seaco_decoder = seaco_decoder_class(model.seaco_decoder, onnx=is_onnx)

    from funasr.utils.torch_function import sequence_mask

    model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)

    from funasr.utils.torch_function import sequence_mask

    model.make_pad_mask = sequence_mask(kwargs["max_seq_len"], flip=False)
    model.feats_dim = 560
    model.NOBIAS = 8377

    import copy
    import types

    if hasattr(model.encoder, "prepare_mask"):
        model.encoder.forward = types.MethodType(_encoder_forward_with_mask, model.encoder)
    if hasattr(model.decoder, "prepare_mask"):
        decoder_sig = inspect.signature(model.decoder.forward)
        if "return_hidden" in decoder_sig.parameters:
            model.decoder.forward = types.MethodType(
                _decoder_forward_with_mask_sanm, model.decoder
            )
        else:
            model.decoder.forward = types.MethodType(
                _decoder_forward_with_mask_basic, model.decoder
            )

    encoder_model = copy.copy(model)
    predictor_model = copy.copy(model)
    decoder_model = copy.copy(model)

    # backbone
    encoder_model.forward = types.MethodType(export_encoder_forward, encoder_model)
    encoder_model.export_dummy_inputs = types.MethodType(
        export_encoder_dummy_inputs, encoder_model
    )
    encoder_model.export_input_names = types.MethodType(
        export_encoder_input_names, encoder_model
    )
    encoder_model.export_output_names = types.MethodType(
        export_encoder_output_names, encoder_model
    )
    encoder_model.export_dynamic_axes = types.MethodType(
        export_encoder_dynamic_axes, encoder_model
    )
    
    predictor_model.forward = types.MethodType(export_predictor_forward, predictor_model)
    predictor_model.export_dummy_inputs = types.MethodType(
        export_predictor_dummy_inputs, predictor_model
    )
    predictor_model.export_input_names = types.MethodType(
        export_predictor_input_names, predictor_model
    )
    predictor_model.export_output_names = types.MethodType(
        export_predictor_output_names, predictor_model
    )
    predictor_model.export_dynamic_axes = types.MethodType(
        export_predictor_dynamic_axes, predictor_model
    )
    
    
    decoder_model.forward = types.MethodType(export_decoder_forward, decoder_model)
    decoder_model.export_dummy_inputs = types.MethodType(
        export_decoder_dummy_inputs, decoder_model
    )
    decoder_model.export_input_names = types.MethodType(
        export_decoder_input_names, decoder_model
    )
    decoder_model.export_output_names = types.MethodType(
        export_decoder_output_names, decoder_model
    )
    decoder_model.export_dynamic_axes = types.MethodType(
        export_decoder_dynamic_axes, decoder_model
    )
    
    embedder_model.export_name = "model_eb"
    encoder_model.export_name = "encoder"
    predictor_model.export_name = "predictor"
    decoder_model.export_name = "decoder"

    return encoder_model, predictor_model, decoder_model, embedder_model


def export_encoder_forward(
    self,
    speech: torch.Tensor,
    speech_mask: torch.Tensor,
):
    # a. To device
    enc, _ = self.encoder(speech, speech_mask)
    mask = _to_predictor_mask(speech_mask).to(dtype=torch.float32)

    torch.save(enc, "enc.pt")
    torch.save(mask, "mask.pt")
    return enc

def export_predictor_forward(
    self,
    enc: torch.Tensor,
    mask: torch.Tensor
):
    if USE_PREDICTOR_CUSTOM_EXPORT and _can_use_predictor_custom_export(self.predictor):
        pre_acoustic_embeds, pre_token_length = _predictor_forward_equivalent_no_loop(
            self.predictor, enc, mask
        )
    else:
        pre_acoustic_embeds, pre_token_length, alphas, pre_peak_index = self.predictor(enc, mask)
        pre_token_length = pre_token_length.floor().type(torch.int32)
    
    torch.save(pre_acoustic_embeds, "pre_acoustic_embeds.pt")
    torch.save(pre_token_length, "pre_token_length.pt")
    return pre_acoustic_embeds, pre_token_length

def export_decoder_forward(
    self,
    enc: torch.Tensor,
    enc_mask: torch.Tensor,
    pre_acoustic_embeds: torch.Tensor,
    pre_token_mask: torch.Tensor,
):
    decoder_out, _ = self.decoder(enc, enc_mask, pre_acoustic_embeds, pre_token_mask)
    decoder_out = torch.log_softmax(decoder_out, dim=-1)
    return decoder_out


def export_encoder_dummy_inputs(self):
    speech = torch.randn(STATIC_BATCH, STATIC_FEATS_LEN, self.feats_dim)
    speech_mask = torch.ones(STATIC_BATCH, STATIC_FEATS_LEN, dtype=torch.float32)
    return (speech, speech_mask)

def export_predictor_dummy_inputs(self):
    enc = torch.load("enc.pt")
    mask = torch.load("mask.pt")
    return (enc, mask)
    # enc = torch.randn(2, 30, 512)
    # mask = torch.randn(2, 1, 30)
    # return (enc, mask)

def export_decoder_dummy_inputs(self):
    enc = torch.load("enc.pt")
    pre_acoustic_embeds = torch.load("pre_acoustic_embeds.pt")
    pre_token_length = torch.load("pre_token_length.pt")
    enc_mask = torch.load("mask.pt")
    pre_acoustic_embeds = _pad_or_trim_time(pre_acoustic_embeds, STATIC_TGT_LEN)
    pre_token_length = torch.clamp(pre_token_length, max=STATIC_TGT_LEN)
    pre_token_mask = _lengths_to_mask(pre_token_length, STATIC_TGT_LEN)
    return (enc, enc_mask, pre_acoustic_embeds, pre_token_mask)


def export_encoder_input_names(self):
    return ["speech", "speech_mask"]


def export_encoder_output_names(self):
    return ["enc"]

def export_predictor_input_names(self):
    return ["enc", "mask"]

def export_predictor_output_names(self):
    return ["pre_acoustic_embeds", "pre_token_length"]

def export_decoder_input_names(self):
    return ["enc", "enc_mask", "pre_acoustic_embeds", "pre_token_mask"]

def export_decoder_output_names(self):
    return ["decoder_out"]

def export_encoder_dynamic_axes(self):
    return {}

def export_predictor_dynamic_axes(self):
    return {
        "enc": {0: "batch_size", 1: "feats_length"},
        "mask": {0: "batch_size", 2: "feats_length"},
        "pre_acoustic_embeds": {0: "batch_size", 1: "pre_acoustic_embeds_length"},
        "pre_token_length": {0: "batch_size"},
    }


def export_decoder_dynamic_axes(self):
    return {}
