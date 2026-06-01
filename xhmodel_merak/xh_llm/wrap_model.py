from functools import partial

import torch.nn as nn

from xhquant.api import ConfigDict, get_xhquant_logger

from .register import XHLLM_TRACEABLE_MODULES, XHLLM_TRACEABLE_MODULES_TORCH_COMPILE, DynamicModule


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
    for _, module in list(llm_model.named_modules()):
        if type(module) in registry:
            logger.debug(f"Model {type(module)} will be wrapped")
            convert_module(module, config, registry)
    wrap_llm_model = llm_model
    if type(llm_model) in registry:
        logger.debug(f"Model {type(llm_model)} will be wrapped")
        wrap_llm_model = convert_module(llm_model, config, registry)  # id(wrap_llm_model) == id(llm_model)
    return wrap_llm_model


wrap_llm_model = partial(_wrap_llm_model, registry=XHLLM_TRACEABLE_MODULES)
wrap_llm_model_for_torch_compile = partial(_wrap_llm_model, registry=XHLLM_TRACEABLE_MODULES_TORCH_COMPILE)
