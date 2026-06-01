import importlib
from typing import Optional, Type

from xhquant.api import get_xhquant_logger

from .base_llm_model import BaseLLMModel
from .configuration_auto import MODEL_TYPE_MAPPING_MODULES
from .register import XH_LLM_MODELS
from .types import BaseLLMModelConfig


def register_llm_model(
    model_type: str, chip_arch: str = "XH2a", master: bool = True, force: bool = False, module: Optional[Type] = None
):
    """
    master=True表示注册主模型，master=False表示注册辅助模型（如视觉编码器），同一model_type只能有一个主模型，但可以有多个辅助模型,
    辅助模型的model_type不会出现在xhmodel_merak.xh_llm.support_llm_model_types.support_llm_model_types列表中。
    force=True表示强制注册，如果model_type已存在则覆盖，否则会抛出异常。module参数用于指定模型类所在的模块，默认为None表示在当前模块中查找。
    """
    if chip_arch not in ["XH2a", "YueHui"]:
        raise ValueError(f"Unsupported chip architecture: {chip_arch}")
    return XH_LLM_MODELS.register_module(model_type, force=force, module=module)


def auto_load_library_for_model(model_type: str):
    module_name = MODEL_TYPE_MAPPING_MODULES.get(model_type, None)
    if module_name is None:
        raise ValueError(f"Unsupported model_type: {model_type}")
    # 动态加载 models 目录下的模块
    try:
        logger = get_xhquant_logger()
        # __name__ 是完整模块名：xhmodel_merak.xh_llm.hmonnx.auto_llm_hmonnx_model
        # __package__ 是当前包名：xhmodel_merak.xh_llm.hmonnx
        importlib.import_module(f".models.{module_name}", package=__package__)
        logger.info(f"Dynamically loaded module: {module_name}")
    except ImportError as e:
        raise ImportError(f"Failed to load module '{module_name}': {e}") from e


def get_model_class(cfg: BaseLLMModelConfig | dict) -> type[BaseLLMModel]:
    if isinstance(cfg, dict):
        cfg = BaseLLMModelConfig.from_dict(cfg)
    chip_arch = cfg.chip_arch
    model_type = cfg.model_type
    if chip_arch in ["XH2a", "YueHui"]:
        register = XH_LLM_MODELS
    else:
        raise ValueError(f"Unsupported chip architecture: {chip_arch}")
    if model_type not in register:
        auto_load_library_for_model(model_type)
    if model_type not in register:
        raise ValueError(f"Unsupported model type: {model_type} for chip architecture: {chip_arch}")

    return register.get(model_type)


def get_config_class(cfg: BaseLLMModelConfig | dict) -> type[BaseLLMModelConfig]:
    if isinstance(cfg, dict):
        cfg = BaseLLMModelConfig.from_dict(cfg)
    model_cls = get_model_class(cfg)
    return model_cls.config_class()
    # if model_type not in register:
    #     raise ValueError(f"Unsupported model type: {model_type} for chip architecture: {chip_arch}")
    # return register.get(model_type)
