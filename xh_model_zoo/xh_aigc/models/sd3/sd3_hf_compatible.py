# Copyright 2025 HOUMO AI
#
# File: sd3_hf_compatible.py
# Description:
#   Sd3 Hf Compatible implementation.
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
from diffusers import StableDiffusion3Pipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.clip.modeling_clip import CLIPTextModelOutput

from xhquant.api import HMONNXInference

from ....utils.device_dtype_mixin import DeviceDtypeMixin
from .sd3_inference import SD3Inference


class HMONNXModule(DeviceDtypeMixin):
    def __init__(self, runtime: HMONNXInference):
        super().__init__()
        self.runtime = runtime
        self._set_device(getattr(runtime, "device", torch.device("cpu")))
        self._set_exec_device(self.device)
        self._set_dtype(getattr(runtime, "dtype", torch.float16))

    def to(self, *args, **kwargs):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            device = torch.device(device)
            if hasattr(self.runtime, "to"):
                self.runtime.to(device)
            self._set_device(device)
            self._set_exec_device(device)
        if dtype is not None:
            self._set_dtype(dtype)
        return self

    @property
    def device(self) -> torch.device:
        return self._device


class Clip(HMONNXModule):
    def __init__(self, clip: HMONNXInference):
        super().__init__(clip)
        self.clip = self.runtime

    @property
    def dtype(self):
        return torch.float16

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPTextModelOutput]:
        input_ids = input_ids.to(torch.int32)
        outputs = self.clip(input_ids)
        text_embeds, hidden_states = outputs
        text_embeds = text_embeds.to(self.dtype)
        hidden_states = hidden_states.to(self.dtype)
        outputs = CLIPTextModelOutput(
            text_embeds=text_embeds,
            last_hidden_state=None,
            hidden_states=[hidden_states, 0],
            attentions=None,
        )
        return outputs


class T5(HMONNXModule):
    def __init__(self, t5: HMONNXInference):
        super().__init__(t5)
        self.t5 = self.runtime

    @property
    def dtype(self):
        return torch.float16

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        input_ids = input_ids.to(torch.int32)  # type: ignore # noqa: F401
        text_embeds = self.t5(input_ids)
        return (text_embeds,)


class MMDIT(HMONNXModule):
    def __init__(self, mmdit: HMONNXInference):
        super().__init__(mmdit)
        self.mmdit = self.runtime

    @property
    def dtype(self):
        return torch.float16

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        pooled_projections: torch.FloatTensor = None,
        timestep: torch.LongTensor = None,
        block_controlnet_hidden_states: List = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        output = self.mmdit(
            hidden_states.half(),
            encoder_hidden_states.half(),
            pooled_projections.half(),
            timestep.half(),
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


class VAE(HMONNXModule):
    def __init__(self, vae: HMONNXInference):
        super().__init__(vae)
        self.vae = self.runtime

    def forward(
        self,
        sample: torch.Tensor,
        latent_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.vae(sample.half())
        return outputs


class SD3HFCompatible(StableDiffusion3Pipeline):
    def setup(self, sd3_inference: SD3Inference):
        pass

    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model_or_path: Union[StableDiffusion3Pipeline, str],
        mmdit: Optional[HMONNXInference] = None,
        clip_l: Optional[HMONNXInference] = None,
        clip: Optional[HMONNXInference] = None,
        vae: Optional[HMONNXInference] = None,
        t5: Optional[HMONNXInference] = None,
    ) -> StableDiffusion3Pipeline:
        if isinstance(hf_model_or_path, StableDiffusion3Pipeline):
            pipe = hf_model_or_path
        else:
            pipe = StableDiffusion3Pipeline.from_pretrained(
                hf_model_or_path,
                torch_dtype=torch.float16,
            )

        if mmdit is not None:
            wrap_mmdit = MMDIT(mmdit)
            wrap_mmdit.config = pipe.transformer.config

            pipe.transformer.cpu()
            del pipe.transformer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pipe.transformer = wrap_mmdit

        if clip_l is not None:
            wrap_clip_l = Clip(clip_l)
            wrap_clip_l.config = pipe.text_encoder_2.config
            pipe.text_encoder_2.cpu()
            del pipe.text_encoder_2
            pipe.text_encoder_2 = wrap_clip_l

        if clip is not None:
            wrap_clip = Clip(clip)
            wrap_clip.config = pipe.text_encoder.config
            pipe.text_encoder.cpu()
            del pipe.text_encoder
            pipe.text_encoder = wrap_clip

        if vae is not None:
            wrap_vae = VAE(vae)
            pipe.vae.decoder.cpu()
            del pipe.vae.decoder
            pipe.vae.decoder = wrap_vae

        if t5 is not None:
            wrap_t5 = T5(t5)
            pipe.text_encoder_3.cpu()
            del pipe.text_encoder_3
            pipe.text_encoder_3 = wrap_t5

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return pipe
