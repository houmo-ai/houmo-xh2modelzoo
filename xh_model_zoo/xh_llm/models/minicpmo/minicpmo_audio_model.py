# Copyright 2025 HOUMO AI
#
# File: minicpmo_audio_model.py
# Description:
#   Minicpmo Audio Model model implementation.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..base_model import BaseModel
from ..builder import MODELS, wrap_llm_model
from .minicpmo_base_model import XHMiniCPMOBaseModel


@MODELS.register_module()
class XHMiniCPMOAudioModel(XHMiniCPMOBaseModel):
    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type,
        allow_quant=True,
        export_cfg=None,
    ):
        super().__init__(
            hf_model,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            frontend_type=frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )

    def get_hf_model(self, device_map="cpu", **kwargs):
        hf_model = super().get_hf_model(device_map=device_map, **kwargs)
        self.audio_projection_layer = hf_model.audio_projection_layer
        self.audio_avg_pooler = hf_model.audio_avg_pooler
        return hf_model

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()
        from ._audio_model_impl import register_wrap_cls as audio_register_wrap_cls  # noqa F401

        audio_register_wrap_cls(hf_model)
        apm = hf_model.apm

        wraped_apm = super().init_wrap_model(apm)
        wraped_apm.audio_projection_layer = hf_model.audio_projection_layer
        wraped_apm.audio_encoder_layer = hf_model.audio_encoder_layer
        wraped_apm.audio_avg_pooler = hf_model.audio_avg_pooler
        return wraped_apm

    def _set_device(self, device: torch.device) -> None:
        super()._set_device(device)

    def prepare_inputs_for_graph(self, data: Dict[str, Union[torch.Tensor, Any]]) -> Any:
        raise NotImplementedError(
            "prepare_inputs_for_graph is not implemented in XHMiniCPMOAudioModel, please implement it in your own model."
        )

    def prepare_inputs(self, data: Dict[str, Union[torch.Tensor, Any]]) -> Any:
        raise NotImplementedError(
            "prepare_inputs is not implemented in XHMiniCPMOAudioModel, please implement it in your own model."
        )

    def _forward(
        self,
        input_features: Tensor,
        audio_attention_mask: Tensor,
        **kwargs,
    ) -> Tensor:
        audio_embeding = self(
            input_features,
            audio_attention_mask,
        )
        return audio_embeding
