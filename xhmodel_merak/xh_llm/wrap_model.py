import copy
import gc
from contextlib import contextmanager
from functools import partial
from typing import Callable

import torch.nn as nn
from tqdm import tqdm

from xhquant.api import ConfigDict, get_xhquant_logger

from .register import XHLLM_TRACEABLE_MODULES, XHLLM_TRACEABLE_MODULES_TORCH_COMPILE, DynamicModule


@contextmanager
def traceable_module_placeholder_context(
    type_names: list[str], registry=XHLLM_TRACEABLE_MODULES, callback: Callable = None
):
    registry_state = copy.deepcopy(registry.__dict__)
    type_name_set = set(type_names)
    try:
        module_types = [module_type for module_type in registry._registry if module_type.__name__ in type_name_set]
        for module_type in module_types:
            for dynamic_type in list(registry._dynamic_classes.keys()):
                if issubclass(dynamic_type, module_type):
                    registry._dynamic_classes.pop(dynamic_type)
            registry._registry.pop(module_type)
            registry._key_registry.pop(module_type, None)

        if callback is not None:
            callback(registry=registry)

        yield
    finally:
        registry.__dict__.clear()
        registry.__dict__.update(registry_state)


def _is_fx_leaf_module(module: nn.Module, module_qualified_name: str, tracer) -> bool:
    return tracer.is_leaf_module(module, module_qualified_name)


def _is_container_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.ModuleList, nn.ModuleDict, nn.Sequential))


def _collect_wrapping_modules(module: nn.Module, registry, logger) -> list[nn.Module]:
    from xhquant.frontend.torchfx.xh_fx import XHTracer

    wrapping_modules = []
    tracer = XHTracer()
    visited_modules = set()

    def visit(current_module: nn.Module, module_qualified_name: str) -> None:
        module_id = id(current_module)
        if module_id in visited_modules:
            return
        visited_modules.add(module_id)

        if type(current_module) in registry:
            logger.debug(f"Model {type(current_module)} will be wrapped")
            wrapping_modules.append(current_module)

        if _is_fx_leaf_module(current_module, module_qualified_name, tracer) and not _is_container_module(
            current_module
        ):
            return

        for child_name, child_module in current_module.named_children():
            child_qualified_name = child_name if not module_qualified_name else f"{module_qualified_name}.{child_name}"
            visit(child_module, child_qualified_name)

    visit(module, "")
    return wrapping_modules


def convert_module(module: nn.Module, config: ConfigDict, registry) -> DynamicModule:
    """
    这里不会创建新的module, 而是直接修改原module的__class__属性,
    module的__class__将被替换成新的类型, 因此, 原module的权重、属性等不会丢失.
    """
    if isinstance(module, DynamicModule):
        return module
    nn_cls = type(module)

    dm_cls = registry.get(nn_cls)
    if dm_cls is None:
        raise ValueError(f"Unsupported module: {nn_cls}")
    qmodule = dm_cls.convert(module, config)
    return qmodule


def _wrap_llm_model(llm_model: nn.Module, config: ConfigDict, registry) -> nn.Module:
    """
    对huggingface的模型进行转换, 使其支持torch.fx.Tracer.
    转换的原理是, 对于每一个支持的模块都替换成DynamicModule, 并将原模块的权重、属性等保存下来.
    这样, 就可以使用torch.fx.Tracer来进行模型的trace.
    """
    try:
        import accelerate

        llm_model = accelerate.hooks.remove_hook_from_module(llm_model, recurse=True)
    except ImportError:
        pass
    logger = get_xhquant_logger()
    wrapping_modules = _collect_wrapping_modules(llm_model, registry, logger)
    wrap_llm_model = llm_model
    for module in tqdm(wrapping_modules, desc="wrap modules"):
        convert_module(module, config, registry)
    gc.collect()
    return wrap_llm_model


wrap_llm_model = partial(_wrap_llm_model, registry=XHLLM_TRACEABLE_MODULES)
wrap_llm_model_for_torch_compile = partial(_wrap_llm_model, registry=XHLLM_TRACEABLE_MODULES_TORCH_COMPILE)
