import copy
from pathlib import Path

from transformers import AutoConfig

from xhquant.api import get_xhquant_logger

from .builder import get_config_class
from .types import BaseLLMModelConfig


class AutoLLMConfig:
    @classmethod
    def from_pretrained(cls, config: BaseLLMModelConfig | dict) -> BaseLLMModelConfig:
        logger = get_xhquant_logger()
        if isinstance(config, dict):
            config_dict = copy.deepcopy(config)
            base_config = BaseLLMModelConfig.from_dict(config)
        else:
            base_config = config
            config_dict = base_config.to_dict()
        model_type = base_config.model_type
        pretrained_model_path = base_config.hf_model
        if pretrained_model_path is None:
            raise ValueError("pretrained_model_path must be specified in convert_config")
        if not Path(pretrained_model_path).exists():
            raise ValueError(f"pretrained_model_path does not exist: {pretrained_model_path}")

        if model_type is None or (isinstance(model_type, str) and len(model_type) == 0):
            hf_config = AutoConfig.from_pretrained(pretrained_model_path, trust_remote_code=True)
            architectures = hf_config.architectures
            if architectures is None:
                raise ValueError("architectures is None")
            model_type = architectures[0]
            config_dict["model_type"] = model_type
            logger.info(f"model_type is not specified, inferred from config: {model_type}")
        if config_dict.get("model_type") is None:
            raise ValueError("model_type must be specified in convert_config")
        cls = get_config_class(base_config)
        model_config = cls.from_dict(config_dict)
        return model_config
