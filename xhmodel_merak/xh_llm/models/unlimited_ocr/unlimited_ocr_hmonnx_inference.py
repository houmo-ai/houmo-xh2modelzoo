"""HMONNX inference runtime for Unlimited-OCR (base/no-crop single image)."""

from __future__ import annotations

import torch
import torch.nn as nn

from xhmodel_merak.xh_llm.utils import unfold_args
from xhquant.api import get_xhquant_logger

from ...hmonnx.hmonnx_model import HMONNXModel
from ...hmonnx.vision_llm_hmonnx_model import VisonLLMHMONNXModel
from ...types import LLMModelMeta
from .data_preprocess import UnlimitedOCRDataPreprocess
from .unlimited_ocr_processor import XHUnlimitedOCRProcessor


class XHUnlimitedOCRHMONNXModel(VisonLLMHMONNXModel):
    """Runtime that presents the exported visual + LLM prefill/decode as one VLM.

    The visual and LLM HMONNX graphs are exported separately, but generation
    calls ``visual.forward`` first to turn the global-view image into image
    embeddings, then scatters them into the LLM prompt at ``<image>`` positions
    via :class:`UnlimitedOCRDataPreprocess`.
    """

    def __init__(self, meta_info: LLMModelMeta, **kwargs):
        super().__init__(meta_info, **kwargs)
        self.visual_meta = meta_info.visual_config
        if self.visual_meta is None or getattr(self.visual_meta, "hmonnx", None) is None:
            raise ValueError("Unlimited-OCR runtime requires visual_config.hmonnx in the exported metadata.")
        self.visual = HMONNXModel(self.visual_meta.hmonnx)

    @staticmethod
    def _build_embed_tokens_from_meta(meta: LLMModelMeta) -> nn.Embedding:
        """Build the embedding from the exported quant_embedding state dict.

        Overrides the base implementation to avoid ``AutoConfig`` with
        ``trust_remote_code``: Unlimited-OCR's exported ``hf_config`` declares an
        ``auto_map`` pointing at ``modeling_unlimitedocr.py`` which is not copied
        into the export directory. The embedding shape is fully determined by the
        saved ``weight`` tensor, so no HF config lookup is needed.
        """
        from pathlib import Path

        quant_embedding_path = Path(meta.quant_embedding)
        try:
            loaded = torch.load(str(quant_embedding_path), map_location="cpu", weights_only=True)
        except Exception:
            loaded = torch.load(str(quant_embedding_path), map_location="cpu", weights_only=False)

        if isinstance(loaded, nn.Embedding):
            return loaded

        state_dict = loaded
        weight = state_dict["weight"]
        vocab_size, hidden_size = weight.shape
        embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=weight.dtype)
        embed_tokens.load_state_dict(state_dict)
        get_xhquant_logger().info(f"Loaded quantized embedding from {quant_embedding_path}")
        return embed_tokens

    def _set_device(self, device):
        super()._set_device(device)
        self.visual.to(device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        self.visual._set_dtype(dtype)
        return self

    def to_fast(self):
        self.visual.to_fast()
        super().to_fast()
        return self

    def get_tf_processor(self) -> XHUnlimitedOCRProcessor:
        visual_meta = self.visual_meta
        return XHUnlimitedOCRProcessor.from_pretrained(
            self.hf_model_dir,
            image_token_id=self.meta_info.image_token_id,
            image_size=int(getattr(visual_meta, "image_size_w", 1024)),
            base_size=int(getattr(visual_meta, "base_size", 1024)),
            patch_size=int(getattr(visual_meta, "patch_size", 16)),
            downsample_ratio=int(getattr(visual_meta, "downsample_ratio", 4)),
            crop_mode=bool(getattr(visual_meta, "crop_mode", False)),
        )

    def forward(self, *args):
        args = unfold_args(args)
        # HMONNX kernels expect int32 shape/index tensors; HF processors often
        # produce int64 ids by default.
        args = [arg.to(torch.int32) if arg.dtype == torch.int64 else arg for arg in args]
        return super().forward(*args)

    def _get_data_preprocessor(self) -> UnlimitedOCRDataPreprocess:
        visual_meta = self.visual_meta
        data_preprocess = UnlimitedOCRDataPreprocess(
            token_embedding=self.get_input_embeddings(),
            input_sequence_length=self.get_input_sequence_length(),
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            image_token_id=self.meta_info.image_token_id,
            image_size=int(getattr(visual_meta, "image_size_w", 1024)),
            patch_size=int(getattr(visual_meta, "patch_size", 16)),
            downsample_ratio=int(getattr(visual_meta, "downsample_ratio", 4)),
            crop_mode=bool(getattr(visual_meta, "crop_mode", False)),
            pad_token_id=self.pad_token_id if self.pad_token_id is not None else 0,
        )
        data_preprocess.to(self._device, self._dtype)
        return data_preprocess
