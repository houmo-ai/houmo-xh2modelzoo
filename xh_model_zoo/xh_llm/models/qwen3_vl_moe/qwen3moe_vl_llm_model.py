# Copyright 2025 HOUMO AI
#
# File: qwen3moe_vl_llm_model.py
# Description:
#   Qwen3moe Vl Llm Model model implementation.
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

from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
from torch import Tensor
from xhquant import nn as xhnn
# import transformers_modules
from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
from .modeling_qwen3moe_vl import Qwen3VLMoeForConditionalGeneration
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeCausalLMOutputWithPast
from xhquant.core import CacheTensor
from xhquant.api import (
    FrontendGraph,
    FrontendType,
    to_frontend_graph,
    to_quant_graph,
    QuantGraph
)
from ..base_llm_model import LLMBaseModel
from ..builder import MODELS


@MODELS.register_module()
class XHQwen3VLMoeLLMModel(LLMBaseModel):
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

    def get_hf_model(self, device_map="cpu", **kwargs) -> Qwen3VLMoeForConditionalGeneration:
        assert self.hf_model_dir is not None
        print(f"hf_model_dir: {self.hf_model_dir}")
        attn_implementation = kwargs.get("attn_implementation", "eager")
        if attn_implementation is not None:
            attn_implementation = attn_implementation.lower()
        hf_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
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

        hf_model.quantization_method = None  # type: ignore
        hf_model._is_hf_initialized = False  # type: ignore
        return hf_model

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()

        from ._llm_model_impl import register_wrap_cls as llm_register_wrap_cls

        llm_register_wrap_cls(hf_model)
        # llm_model = hf_model.model
        self.config = hf_model.language_model.config
        self.model_config = hf_model.config
        self.token_embedding = hf_model.language_model.embed_tokens
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
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            if only_first_block:
                num_decoder_layers = 1
            self.prepare_kv_cache(
                num_decoder_layers,
                [batch_size, self.config.num_key_value_heads, self.cache_length, head_dim],
            )

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
        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3, input_ids.shape[0], input_ids.shape[1], dtype=input_ids.dtype, device=input_ids.device
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(input_ids == vision_start_token_id).squeeze(1)
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

                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                    t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
                    h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
                    w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
                    llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
                    st = ed + llm_grid_t * llm_grid_h * llm_grid_w

                if st < len(input_tokens):
                    st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(llm_positions.max() + 1 - len(total_input_ids[i]))
            mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(-1, keepdim=True)[0]
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
        steps = (input_seq_len + self.input_sequence_length - 1) // self.input_sequence_length
        old_input_seq_len = self.input_sequence_length
        self.input_sequence_length = self.input_sequence_length * steps
        (
            inputs_embeds,
            time_position_ids,
            hight_position_ids,
            width_position_ids,
            past_seq_length,
            seg_length,
            deepstack_image_embed_0,
            deepstack_image_embed_1,
            deepstack_image_embed_2,
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
            deepstack_image_embed_0[:, : self.input_sequence_length, :],
            deepstack_image_embed_1[:, : self.input_sequence_length, :],
            deepstack_image_embed_2[:, : self.input_sequence_length, :],
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
            padding_input_ids = torch.zeros((1, self.input_sequence_length - seq_length), dtype=torch.long).to(device)
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

            deepstack_image_embeds = data["deepstack_image_embeds"]
            new_deepstack_image_embeds = list()
            for deepstack_image_embed in deepstack_image_embeds:
                new_deepstack_image_embed = torch.zeros_like(inputs_embeds).to(deepstack_image_embed)
                new_deepstack_image_embed = new_deepstack_image_embed.masked_scatter(image_mask.to(deepstack_image_embed.device), deepstack_image_embed)
                new_deepstack_image_embeds.append(new_deepstack_image_embed)
            deepstack_image_embed_0 = new_deepstack_image_embeds[0].to(self.execution_device)
            deepstack_image_embed_1 = new_deepstack_image_embeds[1].to(self.execution_device)
            deepstack_image_embed_2 = new_deepstack_image_embeds[2].to(self.execution_device)
        else:
            deepstack_image_embed_0 = torch.zeros_like(inputs_embeds).to(self.execution_device)
            deepstack_image_embed_1 = torch.zeros_like(inputs_embeds).to(self.execution_device)
            deepstack_image_embed_2 = torch.zeros_like(inputs_embeds).to(self.execution_device)

        past_seq_length = data["past_seq_length"]
        assert past_seq_length >= 0, "past_seq_length should be non-negative."

        if past_seq_length == 0:
            # prefill
            image_grid_thw = data["image_grid_thw"]
            video_grid_thw = None
            position_ids, rope_deltas = self.get_rope_index(input_ids, image_grid_thw, video_grid_thw, attention_mask)
            self.rope_deltas = rope_deltas
        else:
            assert self.rope_deltas is not None, f"rope_deltas is None, but past_seq_length is {past_seq_length}"
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = past_seq_length + self.rope_deltas
            position_ids = torch.arange(seq_length, device=inputs_embeds.device)
            position_ids = position_ids.view(1, -1).expand(batch_size, -1)
            delta = delta.repeat_interleave(batch_size // delta.shape[0], dim=0)
            position_ids = position_ids.add(delta)
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches
        time_position_ids = position_ids[0, 0]
        hight_position_ids = position_ids[1, 0]
        width_position_ids = position_ids[2, 0]
        return (
            inputs_embeds.to(self.execution_device),
            time_position_ids.to(self.execution_device),
            hight_position_ids.to(self.execution_device),
            width_position_ids.to(self.execution_device),
            torch.tensor([past_seq_length], dtype=torch.int32).to(self.execution_device),
            torch.tensor([seq_length], dtype=torch.int32).to(self.execution_device),
            deepstack_image_embed_0,
            deepstack_image_embed_1,
            deepstack_image_embed_2,
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
        deepstack_image_embed_0: Tensor = None,
        deepstack_image_embed_1: Tensor = None,
        deepstack_image_embed_2: Tensor = None,
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
            deepstack_image_embed_0,
            deepstack_image_embed_1,
            deepstack_image_embed_2,
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
        steps = (input_seq_len + self.input_sequence_length - 1) // self.input_sequence_length
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
            deepstack_image_embed_0,
            deepstack_image_embed_1,
            deepstack_image_embed_2,
            past_key_caches,
            past_value_caches,
        ) = inputs
        for i in range(steps):
            start = i * self.input_sequence_length
            end = (i + 1) * self.input_sequence_length
            current_input_length = min(end, input_seq_len) - start
            output = self._forward(
                inputs_embeds[:, start:end, :],
                time_position_ids[start:end],
                hight_position_ids[start:end],
                width_position_ids[start:end],
                past_seq_length,
                torch.tensor([current_input_length], dtype=torch.int32).to(inputs_embeds.device),
                deepstack_image_embed_0[:, start:end, :],
                deepstack_image_embed_1[:, start:end, :],
                deepstack_image_embed_2[:, start:end, :],
                past_key_caches,
                past_value_caches,
            )

            past_seq_length += current_input_length

        return output
    
    def convert_to_quant_graph(self, target_device: str,quant_cfg) -> Optional[QuantGraph]:
        super().convert_to_quant_graph(target_device)
        assert self._quanted_model is not None
        # TODO: 临时解决方案，后续需要修改
        if quant_cfg.extra_cfg is not None:
            if "attn_weights" in quant_cfg.extra_cfg:
                attn_weights_cfg = quant_cfg.extra_cfg["attn_weights"]
                if "act_schema" in attn_weights_cfg:
                    act_scheme = attn_weights_cfg["act_schema"]
                else:
                    act_scheme = attn_weights_cfg["act_scheme"]

                if "act_schema_2" in attn_weights_cfg:
                    weight_scheme = attn_weights_cfg["act_schema_2"]
                else:
                    weight_scheme = attn_weights_cfg["act_scheme_2"]
                for node in self._quanted_model.graph.nodes:
                    if (node.op == "call_module") and ('mlp' not in node.name):
                        m = self._quanted_model.get_submodule(node.target)
                        if isinstance(m, (xhnn.MaskedSoftmax, xhnn.SoftmaxPlus, nn.Softmax)):
                            i_node = node.args[0]
                            matmul_module = self._quanted_model.get_submodule(i_node.target)
                            assert isinstance(matmul_module, xhnn.MatMul), f"{type(matmul_module)}"
                            act_bit = act_scheme.get("bits")
                            if act_bit is not None:
                                matmul_module.i_cfg.qspec.man_bit = act_bit
                            w_bit = weight_scheme.get("bits")
                            if w_bit is not None:
                                matmul_module.i_cfg_2.qspec.man_bit = w_bit

        return self._quanted_model    