from turtle import isvisible
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union

import torch
import torch.nn as nn
import transformers
from accelerate import init_empty_weights
from diffusers import SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from tqdm import tqdm
from transformers import AutoConfig
from transformers import GenerationMixin
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
