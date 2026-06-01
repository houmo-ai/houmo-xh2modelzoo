from pathlib import Path

from transformers import AutoConfig

from xhquant.api import get_xhquant_logger

from .base_llm_model import BaseLLMModel
from .builder import get_model_class
from .types import BaseLLMModelConfig


class AutoLLMModel:
    @classmethod
    def from_pretrained(cls, config: BaseLLMModelConfig) -> BaseLLMModel:
        logger = get_xhquant_logger()
        model_type = config.model_type
        pretrained_model_path = config.hf_model
        if pretrained_model_path is None:
            raise ValueError("pretrained_model_path must be specified in convert_config")
        if not Path(pretrained_model_path).exists():
            raise ValueError(f"pretrained_model_path does not exist: {pretrained_model_path}")

        if model_type is None or (isinstance(model_type, str) and len(model_type) == 0):
            config = AutoConfig.from_pretrained(pretrained_model_path, trust_remote_code=True)
            architectures = config.architectures
            if architectures is None:
                raise ValueError("architectures is None")
            model_type = architectures[0]
            config.model_type = model_type
            logger.info(f"model_type is not specified, inferred from config: {model_type}")
        if config.model_type is None:
            raise ValueError("model_type must be specified in convert_config")
        cls = get_model_class(config)
        model = cls.from_pretrained(config)
        model.check_transformer_version()
        return model
