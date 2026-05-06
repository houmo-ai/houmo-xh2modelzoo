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
# File: qwen3_vl_llm_model.py
# Description:
#   Qwen3-VL LLM model adapted for the xh2 model zoo (xh2modelzoo).

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, cast

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, Cache
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
)

from xhmodel_merak.xh_llm.llm_data_processor import BaseLLMInputProcessor
from xhquant.utils import get_xhquant_logger, log_function_call
from xhquant.utils.registry import _DMRegistryCls

from ...builder import register_llm_model
from ...text_llm_hf_compatible import TextLLMHFCompatible
from ...types import ExportData, LLMModelState, VLLMModelMeta
from ...vision_llm_model import VisionLLMModel
from .data_preprocess import Qwen3VLDataPreprocess
from .modeling_qwen3_vl import Qwen3VLForConditionalGeneration as XHQwen3VLForConditionalGeneration
from .modeling_qwen3_vl_patch import qwen3_vl_patch
from .qwen3_vl_hmonnx_inference import XHQwen3VLHMONNXModel
from .qwen3_vl_visual_model import XHQwen3VLVisualModel
from .xh_qwen3_vl_config import XHQwen3VLModelConfig


class _Qwen3VLHFCompatible(TextLLMHFCompatible):
    def _setup(self: Qwen3VLForConditionalGeneration, xh_model: "XHQwen3VLModel"):
        m = super()._setup(xh_model)
        if m is not None:
            if hasattr(m, "model"):
                del m.model
            if hasattr(m, "lm_head"):
                del m.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return m

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
        hm_pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds = None
        deepstack_image_embeds = None

        if hm_pixel_values is not None:
            image_embeds = list()
            deepstack_image_embeds = list()
            for i in range(len(hm_pixel_values)):
                image_embeds_i, deepstack_image_embeds_i = self._llm_model.visual.forward(
                    hm_pixel_values[i].type(self._llm_model.visual.dtype).to(self._llm_model.visual.device),
                )
                image_embeds.append(image_embeds_i)
                deepstack_image_embeds.append(deepstack_image_embeds_i)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_embeds = image_embeds.squeeze(0)
            deepstack_image_embeds = [
                torch.cat([deepstack_image_embeds[i][i_d] for i in range(len(deepstack_image_embeds))], dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                for i_d in range(len(deepstack_image_embeds[0]))
            ]
        data_processor = self._llm_model.get_data_preprocessor()

        data_batch = {
            "input_ids": input_ids,
            "image_embeds": image_embeds,
            "deepstack_image_embeds": deepstack_image_embeds,
            "past_seq_length": self._past_seq_length,
            "image_grid_thw": image_grid_thw,
        }
        data_input = data_processor(data_batch)
        (
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_seq_length,
            deepstack_image_embed_0,
            deepstack_image_embed_1,
            deepstack_image_embed_2,
            past_key_values,
            past_value_caches,
        ) = data_input

        logits = self._llm_model.forward(
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_seq_length,
            deepstack_image_embed_0,
            deepstack_image_embed_1,
            deepstack_image_embed_2,
            past_key_values,
            past_value_caches,
        )

        return Qwen3VLCausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            rope_deltas=data_processor.rope_deltas,
        )


def build_qwen3_vl_hf_compatible_model(hf_model: Qwen3VLForConditionalGeneration, xh_model: "XHQwen3VLModel"):
    # 构建一个兼容HF的视觉模型，主要用于导出ONNX
    LLM_COMPATIBLE_MODULES = _DMRegistryCls("XHCompatible")  # noqa: N806
    hf_model_cls = type(hf_model)
    if hf_model_cls not in LLM_COMPATIBLE_MODULES:
        LLM_COMPATIBLE_MODULES.register_module(
            {
                hf_model_cls: hf_model_cls.__name__,
            },
            _Qwen3VLHFCompatible,
        )
    LLM_COMPATIBLE_MODULES.convert(hf_model, xh_model=xh_model)
    return hf_model


@register_llm_model("Qwen3VLForConditionalGeneration")
class XHQwen3VLModel(VisionLLMModel):
    transformers_min_version = "4.57.6"
    HF_MODEL_CLS = XHQwen3VLForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    HMONNXINFERENCE_CLS = XHQwen3VLHMONNXModel
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_qwen3_vl_hf_compatible_model)
    CONFIG_CLS = XHQwen3VLModelConfig

    def __init__(self, config: XHQwen3VLModelConfig):
        super().__init__(config)

        self.visual = XHQwen3VLVisualModel(config.visual_config)
        self.visual.config.model_name = (
            f"{self.config.model_name}_{self.visual.config.max_size_w}x{self.visual.config.max_size_h}"
        )

    @VisionLLMModel.work_dir.setter
    def work_dir(self, work_dir: str):
        self.config.work_dir = work_dir
        self.visual.config.work_dir = str(Path(work_dir) / "visual")

    def _wraped_post(self, hf_model):
        super()._wraped_post(hf_model)

        self.config.image_token_id = hf_model.config.image_token_id
        self.config.video_token_id = hf_model.config.video_token_id
        self.config.vision_start_token_id = hf_model.config.vision_start_token_id

        self.config.vision_end_token_id = hf_model.config.vision_end_token_id
        self.config.spatial_merge_size = hf_model.config.vision_config.spatial_merge_size

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._llm_model_impl import register_wrap_cls as qwen3_vl_register_wrap_modules

        qwen3_vl_register_wrap_modules(hf_model)

        super().init_wrap_model(hf_model)

    def _get_language_model(self, hf_model: Any) -> Any:
        return hf_model.model.language_model

    def get_tf_processor(self):
        # processor = XHQwen3VLProcessor.from_pretrained(self.hf_model_dir)
        # processor.config.patch_size = self.config.visual_config.patch_size
        # processor.config.max_size_h = self.config.visual_config.max_size_h
        # processor.config.max_size_w = self.config.visual_config.max_size_w
        # return processor
        return self.visual.get_tf_processor()

    def to_wrap(self):
        if self._state == LLMModelState.WRAP:
            return

        if self._state != LLMModelState.NONE:
            raise RuntimeError(f"Invalid state transition: {self._state} -> {LLMModelState.WRAP}")

        hf_model = self.get_native_model()
        self.visual.to_wrap(hf_model)
        self._to_wrap(hf_model)

        self._state = LLMModelState.WRAP

    # def prepare_for_inference(self, *args, **kwargs):
    #     super().prepare_for_inference(*args, **kwargs)
    #     self.visual.prepare_for_inference(*args, **kwargs)

    def release_inference_model(self):
        super().release_inference_model()
        self.visual.release_inference_model()

    def _set_device(self, device):
        super()._set_device(device)
        self.visual._set_device(device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        self.visual._set_dtype(dtype)
        return self

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        data_preprocess = Qwen3VLDataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            image_size_w=self.config.visual_config.max_size_w,
            image_size_h=self.config.visual_config.max_size_h,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            patch_size=self.config.visual_config.patch_size,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            spatial_merge_size=self.config.spatial_merge_size,
        )
        return data_preprocess

    def get_dummy_inputs(self):
        data_batch = super().get_dummy_inputs()
        visual_dummy_inputs = self.visual._get_dummy_inputs()
        data_batch["image_grid_thw"] = visual_dummy_inputs.get("image_grid_thw", None)
        # data_batch["image_embeds"] = (image_embeds,)
        # data_batch["deepstack_image_embeds"] = (deepstack_image_embeds,)

        return data_batch

    # @classmethod
    # def _get_hf_model_for_compatible(cls, hf_model_dir=None):
    #     return cls.get_empty_hf_model(hf_model_dir)

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        native_model = super().get_hf_model(hf_model_dir, quant_weight, **kwargs)
        return qwen3_vl_patch(native_model)

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs) -> Any:
        native_hf_model = super().get_empty_hf_model(hf_model_dir)
        native_hf_model = qwen3_vl_patch(native_hf_model)
        return native_hf_model

    @classmethod
    def _load_quant_weight(cls, quant_weight_path: str, native_hf_model: nn.Module, strict: bool = True) -> bool:
        # 加载量化权重时，先加载到原始模型结构上，再进行patch
        # 这样可以避免量化权重和模型结构不匹配导致的加载失败问题
        # 只有当strict=True时，才会严格检查权重和模型结构的匹配，否则会忽略不匹配的权重
        # 量化后的权重文件中，不包含visual部分，strict必须为False
        return super()._load_quant_weight(quant_weight_path, native_hf_model, strict=False)

    # @classmethod
    # def _get_hf_model_for_compatible(cls, hf_model_dir=None):
    #     return cls.get_hf_model(hf_model_dir)

    def export_llm_hmonnx(self, output_dir):
        super().export_hmonnx(output_dir)

    def export_visual_hmonnx(self, output_dir):
        self.visual.export_hmonnx(output_dir)

    def get_export_info(self, output_dir) -> ExportData:
        str_datetime = datetime.now().strftime("%Y%m%d")
        model_name = self.config.model_name.lower()
        if model_name is None or len(model_name) == 0:
            raise ValueError("Model name is not specified in config, please set model_name in config before exporting.")
        Path(output_dir).mkdir(parents=True, exist_ok=True)
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

    def get_dummy_inputs(self):
        data_batch = super().get_dummy_inputs()
        visual_dummy_inputs = self.visual._get_dummy_inputs()
        data_batch["image_grid_thw"] = visual_dummy_inputs["image_grid_thw"]
        return data_batch

    def get_export_cfg(self) -> dict[str, list[str]]:
        export_cfg = dict(
            input_names=[
                "inputs_embeds",
                "time_position_ids",
                "height_position_ids",
                "width_position_ids",
                "past_seq_length",
                "current_input_length",
                "deepstack_image_embed_0",
                "deepstack_image_embed_1",
                "deepstack_image_embed_2",
            ],
            output_names=["logits"],
        )

        num_decoder_layers = self.kvcache_config.num_layers
        for layer_idx in range(num_decoder_layers):
            export_cfg["input_names"].append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(num_decoder_layers):
            export_cfg["input_names"].append(f"past_value_cache_{layer_idx}")

        return export_cfg

    @log_function_call()
    def export_hmonnx(self, output_dir: str) -> VLLMModelMeta:
        logger = get_xhquant_logger()
        self.work_dir = str(output_dir)
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.fixed()
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

    def forward(self, *args, **kwargs):
        assert self.visual._state == self._state, (
            f"Sub model state {self.visual._state} is different from main model state {self._state}"
        )
        return super().forward(*args, **kwargs)
