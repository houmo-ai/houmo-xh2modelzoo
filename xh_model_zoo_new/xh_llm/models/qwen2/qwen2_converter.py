from typing import List
from xh_model_zoo_new.xh_llm.base_llm_converter import BaseLLMConverter
from xh_model_zoo_new.xh_llm.base_llm_converter_config import BaseLLMConverterConfig
from dataclasses import dataclass


@dataclass
class Qwen2ConverterConfig(BaseLLMConverterConfig):
    pass


class Qwen2Converter(BaseLLMConverter):
    MODEL_KEYS: List[str] = ["Qwen2ForCausalLM", "Qwen2", "qwen2"]
    config: Qwen2ConverterConfig

    @classmethod
    def convert_and_export(cls, hf_model_path: str, config: Qwen2ConverterConfig, output_dir: str):
        from ._model import register_wrap_modules as qwen2_register_wrap_modules  # noqa: F403, F401

        qwen2_register_wrap_modules()

        config = BaseLLMConverterConfig.from_dict_or_other(config)
        return cls(hf_model_path, config).export(output_dir)
