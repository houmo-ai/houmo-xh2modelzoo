# Copyright 2025 HOUMO AI
#
# File: builder.py
# Description:
#   Model registry and wrapper for xh_llm models.
#   This module provides utilities for registering and wrapping HuggingFace
#   models to support torch.fx.Tracer.
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
from typing import Optional, Union

import accelerate
import torch.nn as nn
from xhquant.api import ConfigDict
from xhquant.utils.logger import get_root_logger
from xhquant.utils.registry import Registry
from xhquant.utils.registry.dynamic_module import DynamicModule, _DMRegistryCls

XHLLM_TRACEABLE_MODULES = _DMRegistryCls("XHTrace")
LLM_COMPATIBLE_MODULES = _DMRegistryCls("XHCompatible")
XHLLM_TRACEABLE_MODULES_TORCH_COMPILE = _DMRegistryCls("XHTrace_Torch_Compile")
MODELS = Registry("xh_llm_models")


class DynamicRegister(DynamicModule):
    @classmethod
    def register(cls: type, hf_cls: type):
        if hf_cls not in XHLLM_TRACEABLE_MODULES:
            XHLLM_TRACEABLE_MODULES.register_module(
                {
                    hf_cls: hf_cls.__name__,
                },
                cls,
            )


def wrap_llm_model(llm_model: nn.Module, config: Optional[Union[dict, ConfigDict]] = None) -> nn.Module:
    """
    对huggingface的模型进行转换, 使其支持torch.fx.Tracer.
    转换的原理是, 对于每一个支持的模块都替换成DynamicModule, 并将原模块的权重、属性等保存下来.
    这样, 就可以使用torch.fx.Tracer来进行模型的trace.
    """
    llm_model = accelerate.hooks.remove_hook_from_module(llm_model, recurse=True)
    logger = get_root_logger()
    if config is None:
        config = ConfigDict()
    if isinstance(config, dict):
        config = ConfigDict(config)
    for name, module in list(llm_model.named_modules()):
        if type(module) in XHLLM_TRACEABLE_MODULES:
            logger.debug(f"Model {type(module)} will be wrapped")
            convert_module(module, config)
    wrap_llm_model = llm_model
    if type(llm_model) in XHLLM_TRACEABLE_MODULES:
        logger.debug(f"Model {type(llm_model)} will be wrapped")
        wrap_llm_model = convert_module(llm_model, config)  # id(wrap_llm_model) == id(llm_model)
    return wrap_llm_model


def convert_module(module: nn.Module, config: ConfigDict) -> DynamicModule:
    """
    这里不会创建新的module, 而是直接修改原module的__class__属性,
    module的__class__将被替换成新的类型, 因此, 原module的权重、属性等不会丢失.
    """
    # already converted
    if isinstance(module, DynamicModule):
        return module

    nn_cls = type(module)

    dm_cls = XHLLM_TRACEABLE_MODULES.get(nn_cls)
    if dm_cls is None:
        raise ValueError(f"Unsupported module: {nn_cls}")
    qmodule = dm_cls.convert(module, config)
    return qmodule
