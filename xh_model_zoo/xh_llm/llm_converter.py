from typing import Any, Optional

from transformers import AutoConfig
from xhquant.api import get_root_logger

from .llm_convert_config import LLMConvertConfig


class LLMConverter:
    """
    将transformers的模型转换为HMONNX格式
    """

    @staticmethod
    def from_pretrained(
        pretrained_model_path: str, architecture: Optional[str], convert_config: LLMConvertConfig, out_dir: str
    ):
        config = AutoConfig.from_pretrained(pretrained_model_path, trust_remote_code=True)
        if architecture is None:
            architectures = config.architectures
            if architectures is None:
                raise ValueError("architectures is None")
            architecture = architectures[0]
        assert architecture is not None
        converter_cls: Optional[Any] = None
        if architecture == "Qwen2VLForConditionalGeneration":
            if hasattr(config, "quantization_config"):
                if config.quantization_config["quant_method"].lower() == "awq":
                    from .models.qwen2_vl import Qwen2VLAWQConverterXH2a

                    converter_cls = Qwen2VLAWQConverterXH2a
                else:
                    raise ValueError(f"Unsupported quantization method: {config.quantization_config.quant_method}")
            else:
                from .models.qwen2_vl import Qwen2VLAWQConverterXH2a

                converter_cls = Qwen2VLAWQConverterXH2a
        elif architecture == "Qwen2ForCausalLM":
            if hasattr(config, "quantization_config"):
                if config.quantization_config["quant_method"].lower() == "gptq":
                    from .models.qwen2.qwen2_hf_gptq_convert import Qwen2GPTQConverterXH2a

                    converter_cls = Qwen2GPTQConverterXH2a
                else:
                    raise ValueError(f"Unsupported quantization method: {config.quantization_config.quant_method}")
            else:
                from .models.qwen2 import Qwen2ConverterXH2a

                converter_cls = Qwen2ConverterXH2a
        elif architecture == "LlamaForCausalLM":
            if hasattr(config, "quantization_config"):
                raise ValueError("LlamaForCausalLM does not support quantization")
            else:
                from .models.llama import LlamaConverterXH2a

                converter_cls = LlamaConverterXH2a
        elif architecture == "CogVLMForCausalLM":
            if hasattr(config, "quantization_config"):
                raise ValueError("CogVLMForCausalLM does not support quantization")
            else:
                from .models.cogvlm2 import CogVLM2ConverterXH2a

                converter_cls = CogVLM2ConverterXH2a
        elif architecture == "Qwen3ForCausalLM_legacy":
            from .models.qwen3_legacy import Qwen3LegacyConverterXH2a

            converter_cls = Qwen3LegacyConverterXH2a

        elif architecture == "Qwen2ForCausalLM_legacy":
            from .models.qwen2_legacy import Qwen2LegacyConverterXH2a

            converter_cls = Qwen2LegacyConverterXH2a
        elif architecture == "BertModel_Reranker":
            from .models.bge_reranker import BGERerankerConverterXH2a

            converter_cls = BGERerankerConverterXH2a

        if converter_cls is None:
            raise ValueError(f"Unsupported architecture: {architecture}")

        logger = get_root_logger()
        logger.info(f"Start converting use converter: {converter_cls.__name__}")
        converter_cls.convert(pretrained_model_path, convert_config, out_dir)
