import torch.nn as nn
from typing import List
# from xh_model_zoo_new.converters.llm import BaseHFLLMConverter, BaseHFLLMConverterConfig
from xh_model_zoo_new.xh_llm.base_llm_converter import BaseLLMConverter
from xh_model_zoo_new.xh_llm.base_llm_converter_config import BaseLLMConverterConfig
from dataclasses import dataclass


class Qwen3ConverterConfig(BaseLLMConverterConfig):
    pass



class Qwen3Converter(BaseLLMConverter):
    MODEL_KEYS: List[str] = ["Qwen3ForCausalLM", "Qwen3", "qwen3"]
    config: Qwen3ConverterConfig

    def _register_wrap_module(self, native_model: nn.Module = None):
        from ._model import register_wrap_modules as qwen3_register_wrap_modules  # noqa: F403, F401

        qwen3_register_wrap_modules(native_model)

    @classmethod
    def convert_and_export(
        cls, hf_model_path: str, config: Qwen3ConverterConfig, output_dir: str, generate_golden=False
    ):
        config = Qwen3ConverterConfig.from_dict_or_other(config)
        return cls(hf_model_path, config).export(output_dir, generate_golden)
