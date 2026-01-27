# Copyright 2025 HOUMO AI
#
# File: llm_converter.py
# Description:
#   Base converter class for transforming transformers models to HMONNX format.
#   This module provides the LLMConverter class that handles model conversion
#   for various architectures including Qwen2VL, Qwen3, and others.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
from typing import Any, Optional

from transformers import AutoConfig
from xhquant.api import get_root_logger

from .llm_convert_config import BaseConvertConfig


class LLMConverter:
    """
    将transformers的模型转换为HMONNX格式
    """

    @staticmethod
    def from_pretrained(
        pretrained_model_path: str, architecture: Optional[str], convert_config: BaseConvertConfig, out_dir: str
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
        elif architecture == "Qwen2_5_VLForConditionalGeneration":
            from .models.qwen2_5_vl import Qwen2_5_VLConverterXH2a

            converter_cls = Qwen2_5_VLConverterXH2a
        elif architecture == "Qwen3VLForConditionalGeneration":
            from .models.qwen3_vl import Qwen3_VLConverterXH2a

            converter_cls = Qwen3_VLConverterXH2a
        elif architecture == "Qwen3VLMoeForConditionalGeneration":
            from .models.qwen3_vl_moe import Qwen3_VL_MOEConverterXH2a
            converter_cls = Qwen3_VL_MOEConverterXH2a
        elif architecture == "Qwen2ForCausalLM":
            if hasattr(config, "quantization_config"):
                if config.quantization_config["quant_method"].lower() == "gptq":
                    from .models._qwen2.qwen2_hf_gptq_convert import Qwen2GPTQConverterXH2a

                    converter_cls = Qwen2GPTQConverterXH2a
                else:
                    raise ValueError(f"Unsupported quantization method: {config.quantization_config.quant_method}")
            else:
                from .models._qwen2 import Qwen2ConverterXH2a

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
        elif architecture == "Qwen3ForCausalLM":
            if hasattr(config, "quantization_config"):
                if config.quantization_config["quant_method"].lower() == "awq":
                    from .models._qwen3 import Qwen3AWQConverterXH2a

                    converter_cls = Qwen3AWQConverterXH2a
                elif config.quantization_config["quant_method"].lower() == "gptq":
                    from .models._qwen3 import Qwen3GPTQConverterXH2a

                    converter_cls = Qwen3GPTQConverterXH2a
                elif (
                    hasattr(config, "quantization_config")
                    and config.quantization_config["quant_method"].lower() == "auto-round"
                ):
                    from .models._qwen3 import Qwen3GPTQConverterXH2a

                    converter_cls = Qwen3GPTQConverterXH2a
                else:
                    raise ValueError(f"Unsupported quantization method: {config.quantization_config.quant_method}")
            else:
                from .models._qwen3 import Qwen3ConverterXH2a

                converter_cls = Qwen3ConverterXH2a
        elif architecture == "Qwen3MoeForCausalLM":
            from .models.qwen3moe import Qwen3MoeConverterXH2a

            converter_cls = Qwen3MoeConverterXH2a
        elif architecture == "Qwen2ForCausalLM_legacy":
            from .models.qwen2_legacy import Qwen2LegacyConverterXH2a

            converter_cls = Qwen2LegacyConverterXH2a
        elif architecture == "GptOssForCausalLM":
            from .models.gpt_oss_with_mask import GptOssWithMaskConverterXH2a

            converter_cls = GptOssWithMaskConverterXH2a
        elif architecture == "BertModel_Reranker":
            from .models.bge_reranker import BGERerankerConverterXH2a

            converter_cls = BGERerankerConverterXH2a
        elif architecture == "gte_qwen2":
            from .models.qwen2_ste import SteQwen2ConverterXH2a

            converter_cls = SteQwen2ConverterXH2a
        elif architecture == "MiniCPMWhisperEncoder":
            from .models.minicpmo.minicpmo_audio_convert import MinicpmoAudioConverterXH2a

            converter_cls = MinicpmoAudioConverterXH2a
        elif architecture == "MiniCPMOVisionEncoder":
            from .models.minicpmo.minicpmo_vision_convert import MinicpmoVisionConverterXH2a

            converter_cls = MinicpmoVisionConverterXH2a
        elif architecture == "MiniCPMOLLMEncoder":
            from .models.minicpmo.minicpmo_llm_convert import MinicpmoLLMConverterXH2a

            converter_cls = MinicpmoLLMConverterXH2a
        elif architecture == "MiniCPMOTTS":
            from .models.minicpmo.minicpmo_tts_convert import MinicpmoTTSConverterXH2a

            converter_cls = MinicpmoTTSConverterXH2a
        elif architecture == "MiniCPMOTTSDVAEEncoder":
            from .models.minicpmo.minicpmo_tts_dvae_convert import MinicpmoTTSDVAEConverterXH2a

            converter_cls = MinicpmoTTSDVAEConverterXH2a
        elif architecture == "MiniCPMOTTSVOCOS":
            from .models.minicpmo.minicpmo_tts_vocos_convert import MinicpmoTTSVocosConverterXH2a

            converter_cls = MinicpmoTTSVocosConverterXH2a
        elif architecture == "Qwen3ForCausalLM_LoRA":
            from .models.qwen3_legacy_lora.qwen3_converter import Qwen3LegacyLoRAConverterXH2a

            converter_cls = Qwen3LegacyLoRAConverterXH2a
        elif architecture == "DeepSeekV2":
            from .models.deepseek_ocr.deepseekv2_converter import DeepSeekV2ConverterXH2a
            converter_cls = DeepSeekV2ConverterXH2a
        if converter_cls is None:
            raise ValueError(f"Unsupported architecture: {architecture}")

        logger = get_root_logger()
        logger.info(f"Start converting use converter: {converter_cls.__name__}")
        converter_cls.convert(pretrained_model_path, convert_config, out_dir)
