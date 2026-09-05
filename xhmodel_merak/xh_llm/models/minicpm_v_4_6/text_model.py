"""MiniCPM-V-4.6 text-only adapter for the Merak Qwen3.5 stack."""

import json
from pathlib import Path

import torch.nn as nn
from accelerate import init_empty_weights

from xhmodel_merak.xh_llm.base_llm_model import BaseLLMModel
from xhmodel_merak.xh_llm.llm_data_processor import BaseLLMInputProcessor
from xhmodel_merak.xh_llm.models.qwen3_5.configuration_qwen3_5 import (
    Qwen3_5Config,
)
from xhmodel_merak.xh_llm.models.qwen3_5.data_preprocess import (
    Qwen3_5_DataPreprocess,
)
from xhmodel_merak.xh_llm.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import (
    XHQwen3_5Model,
)
from xhmodel_merak.xh_llm.types import LLMModelState
from xhquant.api import get_xhquant_logger
from xhquant.utils import log_function_call


class MiniCPMV46TextModel(XHQwen3_5Model):
    """Export the embedded Qwen3.5 backend without Qwen's Vision graph.

    MiniCPM-V owns a different Vision tower, but its ``language_model`` and
    ``lm_head`` follow the Qwen3.5 ABI.  The regular Qwen3.5 loader can
    therefore consume the composite MiniCPM checkpoint directly.  Only the
    export orchestration needs to be text-only.
    """

    transformers_min_version = "5.7.0"

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs):
        """Load MiniCPM, then expose its text branch through a Qwen3.5 shell."""

        full_model = super().get_hf_model(
            hf_model_dir,
            quant_weight=quant_weight,
            **kwargs,
        )
        qwen_config = Qwen3_5Config(
            text_config=full_model.config.text_config.to_dict(),
            image_token_id=full_model.config.image_token_id,
            video_token_id=full_model.config.video_token_id,
            vision_start_token_id=248053,
            vision_end_token_id=248054,
            tie_word_embeddings=False,
        )
        with init_empty_weights():
            text_model = Qwen3_5ForConditionalGeneration(qwen_config)

        text_model.model.language_model = full_model.model.language_model
        text_model.lm_head = full_model.lm_head
        text_model.generation_config = full_model.generation_config
        text_model.name_or_path = str(hf_model_dir)

        input_weight = text_model.model.language_model.get_input_embeddings().weight
        if text_model.lm_head.weight is input_weight:
            text_model.lm_head.weight = nn.Parameter(
                input_weight.detach().clone(),
                requires_grad=input_weight.requires_grad,
            )
        text_model.config.text_config.tie_word_embeddings = False
        del full_model
        return text_model.eval()

    def _wraped_post(self, hf_model):
        # These fields only describe the text-graph preprocessing ABI. The
        # MiniCPM runtime intentionally uses sequential LLM positions and
        # passes image_grid_thw=None, matching the proven legacy adapter.
        hf_model.config.vision_start_token_id = (
            getattr(
                hf_model.config,
                "vision_start_token_id",
                None,
            )
            or 248053
        )
        hf_model.config.vision_end_token_id = (
            getattr(
                hf_model.config,
                "vision_end_token_id",
                None,
            )
            or 248054
        )
        hf_model.config.vision_config.spatial_merge_size = 2
        return super()._wraped_post(hf_model)

    def _get_data_preprocessor(self) -> BaseLLMInputProcessor:
        return Qwen3_5_DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            patch_size=14,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
            vision_end_token_id=self.config.vision_end_token_id,
            spatial_merge_size=2,
        )

    def get_export_info(self, output_dir):
        return BaseLLMModel.get_export_info(self, output_dir)

    @log_function_call()
    def export_hmonnx(self, output_dir: str):
        """Export Qwen3.5 Prefill/Decode without its unrelated Vision graph."""

        logger = get_xhquant_logger()
        self.work_dir = str(output_dir)
        self.to_wrap()

        exported_info = self.get_export_info(output_dir)
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()

        # Qwen3.5 builds distinct Prefill and Decode graphs.  BaseLLMModel's
        # single-graph ``fixed`` call is therefore not valid for this model.
        self._quanted_model.prefill.fixed()
        self._quanted_model.decode.fixed()
        self.config.model_name = exported_info.model_name
        self._export_hmonnx(exported_info)

        meta_info = exported_info.meta
        with open(Path(exported_info.exported_dir) / "golden_meta_info.json", "w", encoding="utf-8") as file:
            json.dump(meta_info.to_dict(), file, indent=4)
        logger.info("Exporting completed! Exported model is saved at: %s", exported_info.exported_dir)
        return meta_info


__all__ = ["MiniCPMV46TextModel"]
