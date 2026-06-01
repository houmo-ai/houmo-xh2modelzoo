from typing import Any

import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ._audio_model import _get_feat_extract_output_lengths
from .qwen3_omni_common import build_empty_omni_root_model, load_omni_root_model
from .xh_qwen3_omni_config import XHQwen3OmniAudioConfig


@register_llm_model("Qwen3OmniMoeForConditionalGeneration_audio", master=False)
class XHQwen3OmniMoeAudioEncoderModel(BaseVisionModel):
    transformers_min_version = "4.57.0"
    HF_MODEL_CLS = Qwen3OmniMoeAudioEncoder
    HF_AUTO_MODEL_CLS = Qwen3OmniMoeAudioEncoder
    CONFIG_CLS = XHQwen3OmniAudioConfig

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        root_model = load_omni_root_model(hf_model_dir, cls.HF_MODEL_DTYPE, **kwargs)
        return root_model.thinker.audio_tower

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs) -> Any:
        return build_empty_omni_root_model(hf_model_dir).thinker.audio_tower

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._audio_model import register_wrap_modules

        register_wrap_modules()
        return super().init_wrap_model(hf_model)

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        num_mel_bins = int(getattr(self.wrap_model, "num_mel_bins", 128))
        seq_length = int(getattr(self.wrap_model, "n_window", 300) * 2)
        aftercnn_length = int(_get_feat_extract_output_lengths(torch.tensor([seq_length])).item())
        return {
            "padded_feature": torch.zeros((1, num_mel_bins, seq_length), dtype=self.dtype),
            "padded_mask_after_cnn": torch.ones((1, aftercnn_length), dtype=torch.bool),
            "cu_seqlens": torch.tensor([0, aftercnn_length], dtype=torch.int32),
        }

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {
            "input_names": ["padded_feature", "padded_mask_after_cnn", "cu_seqlens"],
            "output_names": ["last_hidden_state"],
        }
