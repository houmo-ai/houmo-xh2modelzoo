# Copyright 2025 HOUMO AI
#
# File: sd3_custom_a_hf_compatible.py
# Description:
#   Sd3 Custom A Hf Compatible implementation.
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

from turtle import isvisible
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
import torch.nn as nn
import transformers
from accelerate import init_empty_weights
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from tqdm import tqdm
from transformers import AutoConfig, GenerationMixin
from transformers.cache_utils import Cache
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_utils import no_init_weights
from transformers.models.clip.modeling_clip import CLIPTextModelOutput
from xhquant.api import HMONNXInference

from ..sd3 import SD3HFCompatible
from .sd3_custom_a_diffusion_pipe import SD3CustomADiffusion3Pipe


class SD3CustomAHFCompatible(SD3HFCompatible):
    @classmethod
    def to_hf_compatible(
        cls,
        hf_model_or_path: Union[StableDiffusion3Pipeline, str],
        mmdit: Optional[HMONNXInference] = None,
        clip_l: Optional[HMONNXInference] = None,
        clip: Optional[HMONNXInference] = None,
        vae: Optional[HMONNXInference] = None,
        t5: Optional[HMONNXInference] = None,
        **kwargs,
    ) -> StableDiffusion3Pipeline:
        assert (
            transformers.__version__ == "4.46.0"
        ), "transformers version must be 4.46.0, please pip install transformers==4.46.0"
        custom_a_model_or_path: str = kwargs.get("custom_a_model", "")
        assert isinstance(hf_model_or_path, str)
        pipe = SD3CustomADiffusion3Pipe.from_pretrained(hf_model_or_path, custom_a_model_or_path)
        pipe = SD3HFCompatible.to_hf_compatible(pipe, mmdit, clip_l, clip, vae, t5)
        return pipe
