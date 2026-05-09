# -*- coding: utf-8 -*-
# Copyright 2025 The Qwen Team, Alibaba Group and The HuggingFace Inc. team. All rights reserved.
# Copyright 2025 HOUMO AI. All rights reserved.
#
# Modifications:
# - Portions of this file have been modified by HOUMO AI.
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
# File: qwen3_5_llm_model.py
# Description:
#   Qwen3.5 LLM model adapted for the xh2 model zoo (xh2modelzoo).

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, cast

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoModelForImageTextToText
from transformers.cache_utils import Cache
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5CausalLMOutputWithPast

from xhmodel_merak.xh_llm.llm_data_processor import BaseLLMInputProcessor
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_processor import XHQwen3_5Processor
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_vision_model import XHQwen3_5VisionModel
from xhquant.utils import get_xhquant_logger, log_function_call
from xhquant.utils.registry import _DMRegistryCls

from ...builder import register_llm_model
from ...kv_cache_mixin import KVCacheWithLinearMixin
from ...text_llm_hf_compatible import TextLLMHFCompatible
from ...types import ExportData, KVCacheWithLinearConfig, LLMModelState, ModelSwitcher, VLLMModelMeta
from ...vision_llm_model import VisionLLMModel
from .data_preprocess import Qwen3_5_DataPreprocess
from .modeling_qwen3_5 import Qwen3_5ForConditionalGeneration
from .modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as XHQwen3_5ForConditionalGeneration
from .modeling_qwen3_5_patch import qwen3_5_patch
from .qwen3_5_hmonnx_inference import XHQwen3_5_HMONNXModel
from .xh_qwen3_5_config import XHQwen3_5ModelConfig


try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    no_init_weights = init_empty_weights


def _copy_model_shared_params(model: nn.Module) -> nn.Module:
    """深拷贝模型结构，所有 parameter 和 buffer 与原模型共享数据（零额外显存）。"""
    memo: dict[int, Any] = {}
    for param in model.parameters():
        if id(param) not in memo:
            memo[id(param)] = nn.Parameter(param.data, requires_grad=param.requires_grad)
    for buf in model.buffers():
        if id(buf) not in memo:
            memo[id(buf)] = buf
    return copy.deepcopy(model, memo)


class _Qwen3_5HFCompatible(TextLLMHFCompatible):  # noqa: N801
    def _setup(self: XHQwen3_5ForConditionalGeneration, text_llm_model: "XHQwen3_5Model"):
        model = super()._setup(text_llm_model)
        if model is not None:
            # if hasattr(model, "model"):
            #     del model.model
            del model.model.visual
            del model.model.language_model
            if hasattr(model, "lm_head"):
                del model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, Qwen3_5CausalLMOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds = None

        if pixel_values is not None:
            image_embeds = list()
            for i in range(len(pixel_values)):
                image_embeds_i = self._llm_model.visual.forward(
                    pixel_values[i].type(self._llm_model.visual.dtype).to(self._llm_model.visual.device),
                )
                image_embeds.append(image_embeds_i)

            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_embeds = image_embeds.squeeze(0)

        # seq_length = inputs_embeds.shape[1]
        data_processor = self._llm_model.get_data_preprocessor()
        # net_input_seq_len = self._llm_model.get_input_sequence_length()
        # steps = (seq_length + net_input_seq_len - 1) // net_input_seq_len

        data_batch = {
            "input_ids": input_ids,
            "image_embeds": image_embeds,
            "past_seq_length": self._past_seq_length,
            "image_grid_thw": image_grid_thw,
            "video_grid_thw": video_grid_thw,
        }
        data_input = data_processor(data_batch)
        (
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_seq_length,
            linear_mask,
            past_key_values,
            past_value_caches,
            past_conv_caches,
            past_recurrent_states,
        ) = data_input

        logits, conv_cache_out_list, recurrent_state_out_list = self._llm_model.forward(
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_seq_length,
            linear_mask,
            past_key_values,
            past_value_caches,
            past_conv_caches,
            past_recurrent_states,
        )

        return Qwen3_5CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            rope_deltas=data_processor.rope_deltas,
        )


def build_qwen3_5_hf_compatible_model(
    hf_model: Qwen3_5ForConditionalGeneration,
    xh_model: "XHQwen3_5Model",
):
    llm_compatible_modules = _DMRegistryCls("XHCompatible")
    hf_model_cls = type(hf_model)
    if hf_model_cls not in llm_compatible_modules:
        llm_compatible_modules.register_module({hf_model_cls: hf_model_cls.__name__}, _Qwen3_5HFCompatible)
    return llm_compatible_modules.convert(hf_model, text_llm_model=xh_model)


class Qwen3_5_ModelMeta(VLLMModelMeta):  # noqa: N801
    KVCACHE_CONFOG_CLS = KVCacheWithLinearConfig


@register_llm_model("Qwen3_5ForConditionalGeneration")
class XHQwen3_5Model(VisionLLMModel):  # noqa: N801
    HF_MODEL_CLS = XHQwen3_5ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = Qwen3_5_ModelMeta
    HMONNXINFERENCE_CLS = XHQwen3_5_HMONNXModel
    CONFIG_CLS = XHQwen3_5ModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_qwen3_5_hf_compatible_model)
    transformers_min_version = "5.2.0"

    def __init__(self, config: XHQwen3_5ModelConfig):
        super().__init__(config)

        if hasattr(config, "visual_config") and config.visual_config is not None and config.visual_config.enable:
            self.visual = XHQwen3_5VisionModel(config.visual_config)
            self.visual.config.model_name = (
                f"{self.config.model_name}_{self.visual.config.max_size_w}x{self.visual.config.max_size_h}"
            )
        self.full_attention_layer_indices: list[int] = []
        self.linear_attention_layer_indices: list[int] = []
        self._kvcache_config = KVCacheWithLinearConfig()
        self._kvcache_config.use_cache = self.config.use_cache
        self._kvcache_mixin = KVCacheWithLinearMixin(self.kvcache_config)
        self.config = cast(XHQwen3_5ModelConfig, self.config)
        self.wrap_cfg["linear_attention_mode"] = "auto"

        # self.wrap_cfg["linear_attention_mode"] = "chunk"  # for prefill
        # self.wrap_cfg["linear_attention_mode"] = "recurrent"  # for decode

    @VisionLLMModel.work_dir.setter
    def work_dir(self, work_dir: str):
        self.config.work_dir = work_dir
        if hasattr(self, "visual") and self.visual is not None:
            self.visual.work_dir = str(Path(work_dir) / "visual")

    def is_support_dynamic_input(self) -> bool:
        if self._state in [
            LLMModelState.EAGER_ALIGNED,
            LLMModelState.EAGER_FAST,
        ]:
            return True
        if self._state in [
            LLMModelState.WRAP,
            LLMModelState.FRONTED,
            LLMModelState.QUANTED_ALIGNED,
            LLMModelState.QUANTED_FAST,
        ]:
            return False
        return False

    def get_dummy_inputs(self):
        data_batch = super().get_dummy_inputs()
        if hasattr(self, "visual") and self.visual is not None:
            visual_dummy_inputs = self.visual._get_dummy_inputs()
            data_batch["image_grid_thw"] = visual_dummy_inputs.get("image_grid_thw", None)
            data_batch["video_grid_thw"] = visual_dummy_inputs.get("video_grid_thw", None)
        return data_batch

    @property
    def past_conv_caches(self):
        return self._kvcache_mixin.past_conv_caches

    @property
    def past_recurrent_states(self):
        return self._kvcache_mixin.past_recurrent_states

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        native_model = super().get_hf_model(hf_model_dir, quant_weight, **kwargs)
        return qwen3_5_patch(native_model)

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs) -> Any:
        native_hf_model = super().get_empty_hf_model(hf_model_dir)
        native_hf_model = qwen3_5_patch(native_hf_model)
        return native_hf_model

    def _get_language_model(self, hf_model: Any) -> Any:
        return hf_model.model.language_model

    def get_tf_processor(self):
        if hasattr(self, "visual") and self.visual is not None:
            return self.visual.get_tf_processor()
        else:
            return XHQwen3_5Processor.from_pretrained(self.hf_model_dir)

    def get_inference_model(self):
        inference_model = super().get_inference_model()
        if isinstance(inference_model, (list, tuple)):
            if self.is_prefill():
                return inference_model[0]
            else:
                return inference_model[1]
        else:
            return inference_model

    def set_prefill(self):
        self.wrap_cfg["linear_attention_mode"] = "chunk"
        if self._state == LLMModelState.FRONTED:
            self._frontend_model.set_activate_model("prefill")
        elif self._state in [LLMModelState.QUANTED_ALIGNED, LLMModelState.QUANTED_FAST, LLMModelState.QUANTED_DISABLE]:
            self._quanted_model.set_activate_model("prefill")
        super().set_prefill()

    def set_decode(self):
        self.wrap_cfg["linear_attention_mode"] = "recurrent"
        if self._state == LLMModelState.FRONTED:
            self._frontend_model.set_activate_model("decode")
        elif self._state in [LLMModelState.QUANTED_ALIGNED, LLMModelState.QUANTED_FAST, LLMModelState.QUANTED_DISABLE]:
            self._quanted_model.set_activate_model("decode")
        super().set_decode()

    def _to_quanted(self, frontend_model, state):
        prefill_fronted_model = frontend_model.prefill
        self.set_prefill()
        prefill_quanted_model = super()._to_quanted(prefill_fronted_model, state)

        decode_fronted_model = frontend_model.decode
        self.set_decode()
        decode_quanted_model = super()._to_quanted(decode_fronted_model, state)
        self.set_prefill()
        model = ModelSwitcher({"prefill": prefill_quanted_model, "decode": decode_quanted_model})
        model.set_activate_model("prefill")
        return model

    def _to_fronted(self, wrap_model):
        # 将模型转换成前端图，准备进行量化
        self.set_prefill()
        prefill_wrap_model = wrap_model
        decode_wrap_model = wrap_model
        decode_wrap_model = _copy_model_shared_params(wrap_model)
        self._wrap_model = prefill_wrap_model
        prefill_frontend_model = super()._to_fronted(prefill_wrap_model)

        self._wrap_model = decode_wrap_model
        self.set_decode()
        decode_frontend_model = super()._to_fronted(decode_wrap_model)
        self._wrap_model = prefill_wrap_model
        self.set_prefill()
        fronted_model = ModelSwitcher({"prefill": prefill_frontend_model, "decode": decode_frontend_model})
        fronted_model.set_activate_model("prefill")

        return fronted_model

    def _wraped_pre(self, hf_model: XHQwen3_5ForConditionalGeneration):
        super()._wraped_pre(hf_model)
        llm_model = self._get_language_model(hf_model)
        text_config = llm_model.config
        self.layer_types = list(text_config.layer_types)

        if self.config.only_first_block:
            self.linear_attention_layer_indices = []
            for idx, layer_type in enumerate(self.layer_types):
                if layer_type == "full_attention":
                    self.full_attention_layer_indices = [idx]
                    break
                else:
                    self.linear_attention_layer_indices.append(idx)
            self.config.max_layers = len(self.full_attention_layer_indices) + len(self.linear_attention_layer_indices)
            self.config.only_first_block = False
        else:
            self.full_attention_layer_indices = [
                idx for idx, layer_type in enumerate(self.layer_types) if layer_type == "full_attention"
            ]
            self.linear_attention_layer_indices = [
                idx for idx, layer_type in enumerate(self.layer_types) if layer_type == "linear_attention"
            ]
        del hf_model.model.visual
        return hf_model

    def _wraped_post(self, hf_model: XHQwen3_5ForConditionalGeneration):
        self.config.image_token_id = hf_model.config.image_token_id
        self.config.video_token_id = hf_model.config.video_token_id
        self.config.vision_start_token_id = hf_model.config.vision_start_token_id

        self.config.vision_end_token_id = hf_model.config.vision_end_token_id
        self.config.spatial_merge_size = hf_model.config.vision_config.spatial_merge_size

        hf_model = self._wrap_model
        llm_model = self._get_language_model(hf_model)
        # self.embed_tokens.weight 和 lm_head.weight 可能是相同对象
        self.embed_tokens = copy.deepcopy(llm_model.get_input_embeddings())
        text_config = llm_model.config
        self.pad_token_id = llm_model.config.eos_token_id
        self.layer_types = list(text_config.layer_types)

        self_attn = llm_model.layers[self.full_attention_layer_indices[0]].self_attn
        linear_attn = llm_model.layers[self.linear_attention_layer_indices[0]].linear_attn

        linear_kv_cache_config = self.kvcache_config.linear_kv_cache_config
        linear_kv_cache_config.conv_dim = linear_attn.conv_dim
        linear_kv_cache_config.conv_kernel_size = linear_attn.conv_kernel_size
        linear_kv_cache_config.num_v_heads = linear_attn.num_v_heads
        linear_kv_cache_config.head_k_dim = linear_attn.head_k_dim
        linear_kv_cache_config.head_v_dim = linear_attn.head_v_dim
        linear_kv_cache_config.num_layers = len(self.linear_attention_layer_indices)

        if self.use_cache:
            num_decoder_layers = len(self.full_attention_layer_indices)
            head_dim = self_attn.head_dim

            self.kvcache_config.num_layers = num_decoder_layers
            self.kvcache_config.kv_cache_shape = [
                1,
                text_config.num_key_value_heads,
                self.config.context_max_length,
                head_dim,
            ]

        hf_model = None

    def init_wrap_model(self, hf_model: XHQwen3_5ForConditionalGeneration) -> Any:
        from ._llm_model_impl import register_wrap_modules

        register_wrap_modules()
        wrap_model = super().init_wrap_model(hf_model)
        return wrap_model

    def forward(self, *args, **kwargs):
        logits, conv_cache_out_list, recurrent_state_out_list = super().forward(*args, **kwargs)
        past_conv_caches = self._kvcache_mixin.past_conv_caches
        past_recurrent_states = self._kvcache_mixin.past_recurrent_states
        # 更新cache
        for past_conv_cache, conv_cache_out in zip(past_conv_caches, conv_cache_out_list, strict=True):
            past_conv_cache[:] = conv_cache_out[:]

        for past_recurrent_state, recurrent_state_out in zip(
            past_recurrent_states, recurrent_state_out_list, strict=True
        ):
            past_recurrent_state[:] = recurrent_state_out[:]
        return logits, conv_cache_out_list, recurrent_state_out_list

    def _set_device(self, device):
        super()._set_device(device)
        if hasattr(self, "visual") and self.visual is not None:
            self.visual._set_device(device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        if hasattr(self, "visual") and self.visual is not None:
            self.visual._set_dtype(dtype)
        return self

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        data_preprocess = Qwen3_5_DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            image_size_w=self.config.visual_config.max_size_w,
            image_size_h=self.config.visual_config.max_size_h,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            patch_size=self.config.visual_config.patch_size,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            spatial_merge_size=self.config.spatial_merge_size,
        )
        return data_preprocess

    def get_export_cfg(self) -> dict[str, list[str]]:
        export_cfg = {
            "input_names": [
                "inputs_embeds",
                "time_position_ids",
                "hight_position_ids",
                "width_position_ids",
                "past_seq_length",
                "current_input_length",
                "linear_attn_mask",
            ],
            "output_names": ["logits"],
        }
        for layer_idx in range(self.kvcache_config.num_layers):
            export_cfg["input_names"].append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(self.kvcache_config.num_layers):
            export_cfg["input_names"].append(f"past_value_cache_{layer_idx}")

        linear_num_layers = self.kvcache_config.linear_kv_cache_config.num_layers
        for cache_idx in range(linear_num_layers):
            export_cfg["input_names"].append(f"past_conv_cache_{cache_idx}")
        for cache_idx in range(linear_num_layers):
            export_cfg["input_names"].append(f"past_recurrent_state_{cache_idx}")
        for cache_idx in range(linear_num_layers):
            export_cfg["output_names"].append(f"conv_cache_out_{cache_idx}")
        for cache_idx in range(linear_num_layers):
            export_cfg["output_names"].append(f"recurrent_state_out_{cache_idx}")
        return export_cfg

    def export_llm_hmonnx(self, output_dir):
        super().export_hmonnx(output_dir)

    def export_visual_hmonnx(self, output_dir):
        assert hasattr(self, "visual") and self.visual is not None, (
            "Visual model is not initialized, cannot export visual hmonnx."
        )
        self.visual.export_hmonnx(output_dir)

    def get_export_info(self, output_dir) -> ExportData:
        str_datetime = datetime.now().strftime("%Y%m%d")
        model_name = self.config.model_name.lower()
        if model_name is None or len(model_name) == 0:
            raise ValueError("Model name is not specified in config, please set model_name in config before exporting.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        assert hasattr(self, "visual") and self.visual is not None, (
            "Visual model is not initialized, cannot get visual config for export."
        )
        image_size_h = self.visual.config.max_size_h
        image_size_w = self.visual.config.max_size_w
        model_name = f"hmquant_{model_name}_{image_size_w}x{image_size_h}_{str_datetime}"
        output_dir = Path(output_dir) / model_name
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_info = self.create_export_metadata(output_dir)
        export_data = ExportData()
        export_data.exported_dir = str(output_dir)
        export_data.meta = meta_info
        export_data.model_name = model_name
        export_data.str_datetime = str_datetime
        return export_data

    @log_function_call()
    def export_hmonnx(self, output_dir: str) -> VLLMModelMeta:
        logger = get_xhquant_logger()
        self.work_dir = str(output_dir)
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.prefill.fixed()
        self._quanted_model.decode.fixed()
        assert hasattr(self, "visual") and self.visual is not None, (
            "Visual model is not initialized, cannot export hmonnx."
        )
        self.visual.quanted_model.fixed()
        exported_info = self.get_export_info(output_dir)
        self.config.model_name = exported_info.model_name
        visual_output_dir = str(Path(exported_info.exported_dir) / "visual")
        # 导出visual
        self.visual.config.model_name = (
            f"{exported_info.model_name}_{self.visual.config.max_size_w}x{self.visual.config.max_size_h}"
        )
        visual_meta = self.visual.export_hmonnx(visual_output_dir)
        visual_meta.hmonnx = str(Path(visual_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix())
        meta_info = exported_info.meta
        meta_info = cast(VLLMModelMeta, meta_info)
        assert isinstance(meta_info, VLLMModelMeta), f"meta_info expected VLLMModelMeta, but get {type(meta_info)}"
        meta_info.visual_config = visual_meta
        self._export_hmonnx(exported_info)
        json.dump(
            meta_info.to_dict(), open(str(Path(exported_info.exported_dir) / "golden_meta_info.json"), "w"), indent=4
        )
        logger.info(f"Exporting completed! Exported model is saved at: {output_dir}")
        return meta_info
