# Copyright 2025 HOUMO AI
#
# File: paddleocr_vl_llm_model.py
# Description:
#   PaddleOCR-VL LLM model wrapper for xh2modelzoo.
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

from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

# import transformers_modules
from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
from xhquant.api import FrontendGraph, FrontendType, to_frontend_graph, to_quant_graph
from xhquant.core import CacheTensor

from ..base_llm_model import LLMBaseModel
from ..builder import MODELS
from .modeling_paddleocr_vl import PaddleOCRVLCausalLMOutputWithPast, PaddleOCRVLForConditionalGeneration


def gptqmodel_torch_qlinear_converter(self: nn.Module):
    import torch as t  # conflict with torch.py

    if self.bits in [2, 4, 8]:
        zeros = t.bitwise_right_shift(
            t.unsqueeze(self.qzeros, 2).expand(-1, -1, self.pack_factor),
            self.wf_unsqueeze_zero,  # self.wf.unsqueeze(0),
        ).to(self.dequant_dtype)
        zeros = t.bitwise_and(zeros, self.maxq).reshape(self.scales.shape)

        weight = t.bitwise_and(
            t.bitwise_right_shift(
                t.unsqueeze(self.qweight, 1).expand(-1, self.pack_factor, -1),
                self.wf_unsqueeze_neg_one,  # self.wf.unsqueeze(-1)
            ).to(self.dequant_dtype),
            self.maxq,
        )
    elif self.bits == 3:
        zeros = self.qzeros.reshape(
            self.qzeros.shape[0], self.qzeros.shape[1] // 3, 3, 1
        ).expand(-1, -1, -1, 12)
        zeros = zeros >> self.wf_unsqueeze_zero  # self.wf.unsqueeze(0)
        zeros[:, :, 0, 10] = (zeros[:, :, 0, 10] & 0x3) | (
            (zeros[:, :, 1, 0] << 2) & 0x4
        )
        zeros[:, :, 1, 11] = (zeros[:, :, 1, 11] & 0x1) | (
            (zeros[:, :, 2, 0] << 1) & 0x6
        )
        zeros = zeros & 0x7
        zeros = t.cat(
            [zeros[:, :, 0, :11], zeros[:, :, 1, 1:12], zeros[:, :, 2, 1:11]],
            dim=2,
        ).reshape(self.scales.shape)

        weight = self.qweight.reshape(
            self.qweight.shape[0] // 3, 3, 1, self.qweight.shape[1]
        ).expand(-1, -1, 12, -1)
        weight = (weight >> self.wf_unsqueeze_neg_one) & 0x7  # self.wf.unsqueeze(-1)
        weight[:, 0, 10] = (weight[:, 0, 10] & 0x3) | ((weight[:, 1, 0] << 2) & 0x4)
        weight[:, 1, 11] = (weight[:, 1, 11] & 0x1) | ((weight[:, 2, 0] << 1) & 0x6)
        weight = weight & 0x7
        weight = t.cat(
            [weight[:, 0, :11], weight[:, 1, 1:12], weight[:, 2, 1:11]], dim=1
        )
    weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])

    quant_weight = weight - zeros[self.g_idx.long()]
    weight = self.scales[self.g_idx.long()] * quant_weight
    maxq = (2**self.bits) / 2

    assert (
        quant_weight.max() < maxq and quant_weight.min() >= -maxq
    ), f"{quant_weight.max()} {quant_weight}.min()"
    if hasattr(self, "qweight"):
        delattr(self, "qweight")
    if hasattr(self, "qzeros"):
        delattr(self, "qzeros")
    if hasattr(self, "scales"):
        delattr(self, "scales")
    if hasattr(self, "g_idx"):
        delattr(self, "g_idx")
    weight = weight.t()
    quant_weight = quant_weight.t()
    self.register_parameter("weight", nn.Parameter(weight))
    self.register_buffer("quant_weight", quant_weight)
    self.__class__ = nn.Linear


@MODELS.register_module()
class XHPaddleOCRVLLLMModel(LLMBaseModel):
    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type="TorchFX",
        allow_quant=True,
        export_cfg=None,
        is_gptqmodel=False,
    ):
        super().__init__(
            hf_model,
            wrap_cfg,
            quant_config,
            frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )
        self.rope_deltas = None
        self.is_gptqmodel = is_gptqmodel

    def get_hf_model(
        self, device_map="cpu", **kwargs
    ) -> PaddleOCRVLForConditionalGeneration:
        assert self.hf_model_dir is not None
        print(f"hf_model_dir: {self.hf_model_dir}")
        attn_implementation = kwargs.get("attn_implementation", "eager")
        if attn_implementation is not None:
            attn_implementation = attn_implementation.lower()
        hf_model = PaddleOCRVLForConditionalGeneration.from_pretrained(
            self.hf_model_dir,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map,
            attn_implementation=attn_implementation,
        ).eval()

        if hf_model.config.tie_word_embeddings:
            hf_model.config.torchscript = True
            hf_model.tie_weights()
            hf_model.config.tie_word_embeddings = False
            hf_model.config.torchscript = False

        if self.is_gptqmodel:
            hf_model = self.untied_weights(hf_model)

            assert hasattr(hf_model, "hf_quantizer"), "hf_model must have hf_quantizer"
            hf_quantizer: GptqHfQuantizer = hf_model.hf_quantizer

            from transformers.utils import (
                is_auto_gptq_available,
                is_gptqmodel_available,
            )

            converter: Optional[Callable] = None

            QuantLinear = hf_quantizer.optimum_quantizer.quant_linear
            if is_gptqmodel_available():
                from gptqmodel.nn_modules.qlinear.marlin import MarlinQuantLinear
                from gptqmodel.nn_modules.qlinear.torch import TorchQuantLinear

                if QuantLinear is TorchQuantLinear:
                    converter = gptqmodel_torch_qlinear_converter
                elif QuantLinear is MarlinQuantLinear:
                    converter = None

            assert converter is not None, f"Not implemented for {QuantLinear} yet"

            for name, module in hf_model.named_modules():  # type: ignore
                if isinstance(module, QuantLinear):
                    if converter is not None:
                        converter(module)
                    else:
                        raise NotImplementedError(
                            f"Not implemented for {type(QuantLinear)} yet"
                        )

        hf_model.quantization_method = None  # type: ignore
        hf_model._is_hf_initialized = False  # type: ignore
        return hf_model

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()

        from ._llm_model_impl import register_wrap_cls as llm_register_wrap_cls

        llm_register_wrap_cls(hf_model)
        llm_model = getattr(hf_model, "model", None)
        if llm_model is None:
            llm_model = getattr(hf_model, "language_model", None)
        if llm_model is None:
            raise AttributeError(
                "Expected hf_model.model or hf_model.language_model, but neither exists."
            )
        self.config = llm_model.config
        self.model_config = hf_model.config
        self.token_embedding = llm_model.embed_tokens
        wraped_model = super().init_wrap_model(hf_model)
        hf_model = wraped_model

        self.generation_config = hf_model.generation_config
        self.num_hidden_layers = self.config.num_hidden_layers

        if hasattr(self.config, "head_dim"):
            head_dim = self.config.head_dim
        else:
            head_dim = self.config.hidden_size // self.config.num_attention_heads

        batch_size = 1
        if self.use_cache:
            max_seq_len = 0
            input_seq_len = 0
            if hasattr(self.wrap_cfg, "get"):
                max_seq_len = int(self.wrap_cfg.get("max_sequence_length", 0) or 0)
                input_seq_len = int(self.wrap_cfg.get("input_sequence_length", 0) or 0)
            else:
                max_seq_len = int(getattr(self.wrap_cfg, "max_sequence_length", 0) or 0)
                input_seq_len = int(
                    getattr(self.wrap_cfg, "input_sequence_length", 0) or 0
                )
            cfg_max_pos = int(getattr(self.config, "max_position_embeddings", 0) or 0)
            cache_candidates = [int(getattr(self, "cache_length", 0) or 0)]
            if max_seq_len > 0:
                cache_candidates.append(max_seq_len)
            if input_seq_len > 0:
                cache_candidates.append(input_seq_len)
            if cfg_max_pos > 0:
                if max_seq_len > 0:
                    cache_candidates.append(min(cfg_max_pos, max_seq_len))
                else:
                    cache_candidates.append(cfg_max_pos)
            self.cache_length = max(cache_candidates)
            if max_seq_len > 0:
                self.cache_length = min(self.cache_length, max_seq_len)
            if self.cache_length <= 0:
                self.cache_length = max(1, input_seq_len)
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            if only_first_block:
                num_decoder_layers = 1
            self.prepare_kv_cache(
                num_decoder_layers,
                [
                    batch_size,
                    self.config.num_key_value_heads,
                    self.cache_length,
                    head_dim,
                ],
            )
            inputs_cfg = None
            if hasattr(self.quant_cfg, "get"):
                inputs_cfg = self.quant_cfg.get("inputs", None)
                if inputs_cfg is None:
                    inputs_cfg = {}
                    self.quant_cfg["inputs"] = inputs_cfg
            elif isinstance(self.quant_cfg, dict):
                inputs_cfg = self.quant_cfg.get("inputs", {})
                self.quant_cfg["inputs"] = inputs_cfg
            if inputs_cfg is not None:
                for layer_idx in range(num_decoder_layers):
                    inputs_cfg[f"past_key_cache_{layer_idx}"] = {
                        "quantizer": {"qspec": {"fake_dtype": "float16"}}
                    }
                    inputs_cfg[f"past_value_cache_{layer_idx}"] = {
                        "quantizer": {"qspec": {"fake_dtype": "float16"}}
                    }

        hf_model = None
        return wraped_model

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with mordern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embeddin for text part.
            Examples:
                Assume we have a video input with 3 temporal patches, 2 height patches and 2 width patches.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [3, 4, 5, 6, 7]
                text height position_ids: [3, 4, 5, 6, 7]
                text width position_ids: [3, 4, 5, 6, 7]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        spatial_merge_size = self.model_config.vision_config.spatial_merge_size
        image_token_id = self.model_config.image_token_id
        video_token_id = self.model_config.video_token_id
        vision_start_token_id = self.model_config.vision_start_token_id
        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t, llm_grid_h, llm_grid_w = (
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    t_index = (
                        torch.arange(llm_grid_t)
                        .view(-1, 1)
                        .expand(-1, llm_grid_h * llm_grid_w)
                        .flatten()
                    )
                    h_index = (
                        torch.arange(llm_grid_h)
                        .view(1, -1, 1)
                        .expand(llm_grid_t, -1, llm_grid_w)
                        .flatten()
                    )
                    w_index = (
                        torch.arange(llm_grid_w)
                        .view(1, 1, -1)
                        .expand(llm_grid_t, llm_grid_h, -1)
                        .flatten()
                    )
                    llm_pos_ids_list.append(
                        torch.stack([t_index, h_index, w_index]) + text_len + st_idx
                    )
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        input_ids = data["input_ids"]
        input_seq_len = input_ids.shape[-1]
        steps = (
            input_seq_len + self.input_sequence_length - 1
        ) // self.input_sequence_length
        old_input_seq_len = self.input_sequence_length
        self.input_sequence_length = self.input_sequence_length * steps
        (
            inputs_embeds,
            time_position_ids,
            hight_position_ids,
            width_position_ids,
            past_seq_length,
            seg_length,
            past_key_caches,
            past_value_caches,
        ) = self.prepare_inputs(data)
        self.input_sequence_length = old_input_seq_len

        return (
            inputs_embeds[:, : self.input_sequence_length, :],
            time_position_ids[: self.input_sequence_length],
            hight_position_ids[: self.input_sequence_length],
            width_position_ids[: self.input_sequence_length],
            past_seq_length,
            seg_length,
            past_key_caches,
            past_value_caches,
        )

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        device = self.execution_device

        input_ids = data["input_ids"].to(device)

        attention_mask = None

        seq_length = input_ids.shape[1]

        assert self.token_embedding is not None, "Token embedding is not available."
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."

        assert (
            seq_length <= self.input_sequence_length
        ), f"Input sequence length is too long. max input sequence length is {self.input_sequence_length} but got {seq_length}"
        if self.input_sequence_length > seq_length:
            padding_input_ids = torch.zeros(
                (1, self.input_sequence_length - seq_length), dtype=torch.long
            ).to(device)
            padding_input_ids.fill_(self.pad_token_id)
            input_ids = torch.cat([input_ids, padding_input_ids], dim=-1)

        inputs_embeds = self.token_embedding.to(device)(input_ids.to(device))

        n_image_tokens = torch.sum(input_ids == self.model_config.image_token_id).item()
        if n_image_tokens > 0:
            image_embeds = data["image_embeds"]
            n_image_features = image_embeds.shape[0]
            if n_image_tokens != n_image_features:
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_mask = (
                (input_ids == self.model_config.image_token_id)
                .unsqueeze(-1)
                .expand_as(inputs_embeds)
                .to(inputs_embeds.device)
            )
            image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        else:
            pass

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."

        if past_seq_length == 0:
            # prefill
            image_grid_thw = data["image_grid_thw"]
            video_grid_thw = None
            position_ids, rope_deltas = self.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, attention_mask
            )
            self.rope_deltas = rope_deltas
        else:
            assert (
                self.rope_deltas is not None
            ), f"rope_deltas is None, but past_seq_length is {past_seq_length}"
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = past_seq_length + self.rope_deltas
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches
        time_position_ids = position_ids[0, 0].to(torch.float16)
        hight_position_ids = position_ids[1, 0].to(torch.float16)
        width_position_ids = position_ids[2, 0].to(torch.float16)
        return (
            inputs_embeds.to(self.execution_device),
            time_position_ids.to(self.execution_device),
            hight_position_ids.to(self.execution_device),
            width_position_ids.to(self.execution_device),
            torch.tensor([past_seq_length], dtype=torch.int32).to(
                self.execution_device
            ),
            torch.tensor([seq_length], dtype=torch.int32).to(self.execution_device),
            past_key_caches,
            past_value_caches,
        )

    def _forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        time_position_ids: Tensor = None,
        hight_position_ids: Tensor = None,
        width_position_ids: Tensor = None,
        past_seq_length: Tensor = None,
        current_input_length: Tensor = None,
        past_key_caches: Optional[List[Tensor]] = None,
        past_value_caches: Optional[List[Tensor]] = None,
    ):

        logits = self(
            inputs_embeds,
            time_position_ids,
            hight_position_ids,
            width_position_ids,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        )
        return CausalLMOutputWithPast(
            logits=logits,
        )

    @torch.no_grad()
    def test_step(self, data: Union[dict, tuple, list]):
        input_ids = data["input_ids"]
        input_seq_len = input_ids.shape[-1]
        steps = (
            input_seq_len + self.input_sequence_length - 1
        ) // self.input_sequence_length
        old_input_seq_len = self.input_sequence_length
        self.input_sequence_length = self.input_sequence_length * steps
        inputs = self.prepare_inputs(data)
        self.input_sequence_length = old_input_seq_len

        (
            inputs_embeds,
            time_position_ids,
            hight_position_ids,
            width_position_ids,
            past_seq_length,
            _,
            past_key_caches,
            past_value_caches,
        ) = inputs
        for i in range(steps):
            start = i * self.input_sequence_length
            end = (i + 1) * self.input_sequence_length
            current_input_length = min(end, input_seq_len) - start
            if current_input_length <= 0:
                continue
            output = self._forward(
                inputs_embeds[:, start:end, :],
                time_position_ids[start:end],
                hight_position_ids[start:end],
                width_position_ids[start:end],
                past_seq_length,
                torch.tensor([current_input_length], dtype=torch.int32).to(
                    inputs_embeds.device
                ),
                past_key_caches,
                past_value_caches,
            )
            past_seq_length += current_input_length

        return output
