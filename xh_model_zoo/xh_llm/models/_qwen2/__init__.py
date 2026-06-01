# Copyright 2025 HOUMO AI
#
# File: __init__.py
# Description:
#    Qwen2 module initialization.
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

# from typing import Union

# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from transformers.models.qwen2.modeling_qwen2 import Qwen2ForCausalLM
# from xhquant.api import ConfigDict

from .inference import Qwen2Inference
from .qwen2_convert_config import Qwen2ConvertConfig

# from ..builder import wrap_llm_model
from .qwen2_converter import Qwen2ConverterXH2a
from .qwen2_hf_gptq_convert import Qwen2GPTQConverterXH2a

__all__ = [
    "Qwen2ConvertConfig",
    "Qwen2ConverterXH2a",
    "Qwen2GPTQConverterXH2a",
    "Qwen2Inference",
]
# def load_qwen_model(model_or_path: Union[str, Qwen2ForCausalLM], wrap_cfg: ConfigDict):

#     if isinstance(model_or_path, str):
#         model = AutoModelForCausalLM.from_pretrained(
#             model_or_path, trust_remote_code=True, torch_dtype=torch.float16, device_map="cpu"
#         )
#     else:
#         model = model_or_path

#     assert isinstance(model, Qwen2ForCausalLM), f"The model is not Qwen2ForCausalLM, but {type(model)}"

#     from ._model import _Qwen2ForCausalLM  # noqa: F403, F401

#     wraped_model = wrap_llm_model(model, wrap_cfg)
#     return wraped_model
