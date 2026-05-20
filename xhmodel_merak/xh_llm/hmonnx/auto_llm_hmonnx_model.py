import importlib
import json
from typing import TYPE_CHECKING

from xhquant.api import get_xhquant_logger

from ..builder import get_model_class
from ..configuration_auto import MODEL_TYPE_MAPPING_MODULES


if TYPE_CHECKING:
    from .base_llm_hmonnx_model import BaseLLMHMONNXModel


class AutoLLMHONNXModel:
    @classmethod
    def from_pretrained(cls, meta_or_path: dict | str, **kwargs) -> "BaseLLMHMONNXModel":
        logger = get_xhquant_logger()
        if isinstance(meta_or_path, str):
            meta_dict = json.load(open(meta_or_path, "r"))
            meta_dict["_meta_path_"] = meta_or_path
        elif isinstance(meta_or_path, dict):
            meta_dict = meta_or_path
        else:
            meta_dict = meta_or_path
        if isinstance(meta_dict, dict):
            model_type = meta_dict.get("model_config", {}).get("model_type", None)
            if model_type is None:
                raise ValueError("model_type must be specified in meta_dict['model_config']")
            model_cls = get_model_class(meta_dict["model_config"])
            if model_cls is None:
                module_name = MODEL_TYPE_MAPPING_MODULES.get(model_type, None)
                if module_name is None:
                    raise ValueError(f"Unsupported model_type: {model_type}")
                # 动态加载 models 目录下的模块
                try:
                    # __name__ 是完整模块名：xhmodel_merak.xh_llm.hmonnx.auto_llm_hmonnx_model
                    # __package__ 是当前包名：xhmodel_merak.xh_llm.hmonnx
                    importlib.import_module(f"..models.{module_name}", package=__package__)
                    logger.info(f"Dynamically loaded module: {module_name}")
                except ImportError as e:
                    raise ImportError(f"Failed to load module '{module_name}': {e}") from e
                model_cls = get_model_class(meta_dict["model_config"])
                if model_cls is None:
                    raise ValueError(f"Could not find model class for model_type: {model_type}")

            meta_info = model_cls.META_CLS.from_dict(meta_dict)
        else:
            meta_info = meta_dict
        model_type = meta_info.model_config.model_type
        hmonnx_model = model_cls.from_hmonnx_meta(meta_info, **kwargs)
        return hmonnx_model
