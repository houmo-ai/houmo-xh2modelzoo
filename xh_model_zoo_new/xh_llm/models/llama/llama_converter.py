import torch.nn as nn
from typing import List

# from xh_model_zoo_new.converters.llm import BaseHFLLMConverter, BaseHFLLMConverterConfig
from xh_model_zoo_new.xh_llm.base_llm_converter import BaseLLMConverter
from xh_model_zoo_new.xh_llm.base_llm_converter_config import BaseLLMConverterConfig
from dataclasses import dataclass


class LlamaConverterConfig(BaseLLMConverterConfig):
    pass


class LlamaConverter(BaseLLMConverter):
    MODEL_KEYS: List[str] = ["LlamaForCausalLM", "Llama", "llama"]
    config: LlamaConverterConfig

    def _register_wrap_module(self, native_model: nn.Module = None):
        from ._model import register_wrap_modules as llama_register_wrap_modules  # noqa: F403, F401

        llama_register_wrap_modules(native_model)

    @classmethod
    def convert_and_export(
        cls, hf_model_path: str, config: LlamaConverterConfig, output_dir: str, generate_golden=False
    ):
        config = LlamaConverterConfig.from_dict_or_other(config)
        return cls(hf_model_path, config).export(output_dir, generate_golden)
