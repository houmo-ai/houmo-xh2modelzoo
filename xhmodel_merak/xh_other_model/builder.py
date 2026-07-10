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
import importlib
from collections.abc import Mapping
from typing import Any, Optional, Type, Union

import accelerate
import torch.nn as nn
from xhquant.api import ConfigDict
from xhquant.utils.logger import get_root_logger
from xhquant.utils.registry import Registry
from xhquant.utils.registry.dynamic_module import DynamicModule, _DMRegistryCls

from .scan_model_types import get_support_all_model_types

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


def register_other_model(model_type: str, master: bool = True, force: bool = False, module: Optional[Type] = None):
    return MODELS.register_module(model_type, force=force, module=module)


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


def auto_load_library_for_model(model_type: str) -> None:
    module_name = get_support_all_model_types().get(model_type)
    if module_name is None:
        raise ValueError(f"Unsupported model type: {model_type}")
    importlib.import_module(f"{__package__}.models.{module_name}")


def get_model_class(cfg: Mapping[str, Any] | ConfigDict) -> type:
    model_type = _get_model_type(cfg)
    if model_type not in MODELS:
        auto_load_library_for_model(model_type)
    if model_type not in MODELS:
        raise ValueError(f"Unsupported model type: {model_type}")
    return MODELS.get(model_type)


def is_model_type_supported(model_type: str) -> bool:
    if not isinstance(model_type, str) or not model_type:
        return False
    if model_type in MODELS:
        return True
    try:
        auto_load_library_for_model(model_type)
    except (ImportError, ValueError):
        return False
    return model_type in MODELS


def _get_model_type(cfg: Mapping[str, Any] | ConfigDict) -> str:
    if isinstance(cfg, Mapping):
        model_type = cfg.get("type")
    else:
        model_type = getattr(cfg, "type", None)
    if not isinstance(model_type, str) or not model_type:
        raise ValueError("xh_other_model config must specify non-empty field 'type'")
    return model_type
