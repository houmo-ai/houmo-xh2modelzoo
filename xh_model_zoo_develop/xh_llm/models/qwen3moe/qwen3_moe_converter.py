import torch.nn as nn
from typing import List
from xh_model_zoo_develop.xh_llm.base_llm_converter import BaseLLMConverter
from xh_model_zoo_develop.xh_llm.base_llm_converter_config import BaseLLMConverterConfig
from dataclasses import dataclass




@dataclass
class Qwen3MoeConverterConfig(BaseLLMConverterConfig):
    pass


class Qwen3MoeConverter(BaseLLMConverter):
    MODEL_KEYS: List[str] = ["Qwen3MoeForCausalLM", "Qwen3Moe", "qwen3moe"]
    config: Qwen3MoeConverterConfig

    def _register_wrap_module(self, native_model: nn.Module = None):
        from ._moe_model import register_wrap_modules as qwen3_moe_register_wrap_modules  # noqa: F403, F401
        qwen3_moe_register_wrap_modules()

    @classmethod
    def convert_and_export(cls, hf_model_path: str, config: Qwen3MoeConverterConfig, output_dir: str,generate_golden: bool = False):
        config = Qwen3MoeConverterConfig.from_dict_or_other(config)
        return cls(hf_model_path, config).export(output_dir,generate_golden)
