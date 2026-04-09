from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, cast

import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration as XHGemma4ForConditionalGeneration

from xhquant.api import FrontendType, get_xhquant_logger, to_frontend_graph

from ...base_vision_model import BaseVisionModel
from ...llm_data_processor import BaseVisualProcessor
from ...builder import register_llm_model
from ...onnx_lazy_load import lazy_load_onnx
from ...types import VisualModelMeta
from .gemma4_processor import XHGemma4Processor
from .xh_gemma4_config import XHGemma4VisualConfig


class Gemma4VisualAdapter(nn.Module):
    """Trace-friendly adapter wrapping the Gemma4 vision tower + embed_vision.

    Avoids two trace-unsafe operations:

    1. ``create_bidirectional_mask`` in transformers 5.5 (trace failure) — replaced by
       manually iterating encoder layers with a pre-computed eager mask.
    2. ``Gemma4VisionPooler._avg_pool_by_positions`` which uses ``F.one_hot`` and
       integer division (``NonZero`` / ``one_hot`` not supported in frontend graph) —
       replaced by a pre-computed constant pooling weight matrix stored as a buffer.

    ``pooler_weights`` is ``(B, num_patches, output_length)`` and ``num_image_tokens``
    is the count of valid (non-padding) pooled tokens.  Both are deterministic for a
    fixed image size and are computed once during ``init_wrap_model``.
    """

    def __init__(
        self,
        vision_tower: nn.Module,
        embed_vision: nn.Module,
        pooler_weights: torch.Tensor,
        num_image_tokens: int,
    ):
        super().__init__()
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision
        self.num_image_tokens = num_image_tokens
        self.register_buffer("pooler_weights", pooler_weights)

    def forward(self, pixel_values: torch.Tensor, image_position_ids: torch.Tensor):
        vt = self.vision_tower
        pe = vt.patch_embedder

        padding_positions = (image_position_ids == -1).all(dim=-1)  # (B, num_patches)

        # --- patch embedding (bypass _position_embeddings to avoid clamp/one_hot on ints) ---
        pixel_values_norm = 2 * (pixel_values - 0.5)
        hidden_states = pe.input_proj(pixel_values_norm.to(pe.input_proj.weight.dtype))

        # Position embeddings via F.embedding lookup (replaces clamp + one_hot + matmul)
        pos_safe = torch.where(
            image_position_ids >= 0,
            image_position_ids,
            torch.zeros_like(image_position_ids),
        )
        x_emb = torch.nn.functional.embedding(pos_safe[:, :, 0], pe.position_embedding_table[0])
        y_emb = torch.nn.functional.embedding(pos_safe[:, :, 1], pe.position_embedding_table[1])
        position_embeddings = x_emb + y_emb
        valid_mask = (~padding_positions).unsqueeze(-1).to(position_embeddings.dtype)
        position_embeddings = position_embeddings * valid_mask
        hidden_states = hidden_states + position_embeddings

        # --- encoder forward (bypass create_bidirectional_mask) ---
        rope_cos_sin = vt.encoder.rotary_emb(hidden_states, image_position_ids)

        attn_mask_2d = ~padding_positions  # (B, seq_len)  True=valid
        bsz, seq_len, _ = hidden_states.shape
        attn_mask_4d = attn_mask_2d[:, None, None, :].expand(bsz, 1, seq_len, seq_len).to(hidden_states.dtype)
        attn_mask_4d = (1.0 - attn_mask_4d) * torch.finfo(hidden_states.dtype).min

        for layer in vt.encoder.layers[: vt.encoder.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=attn_mask_4d,
                position_embeddings=rope_cos_sin,
                position_ids=image_position_ids,
            )
        # --- end encoder forward ---

        # --- pooler (pre-computed weights, no one_hot / integer ops) ---
        hidden_states = hidden_states.masked_fill(padding_positions.unsqueeze(-1), 0.0)
        pw = self.pooler_weights.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = pw.transpose(1, 2) @ hidden_states
        hidden_states = hidden_states * vt.pooler.root_hidden_size

        # Valid tokens are contiguous [0 .. num_image_tokens-1]
        hidden_states = hidden_states[:, : self.num_image_tokens, :]
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])

        if vt.config.standardize:
            hidden_states = (hidden_states - vt.std_bias) * vt.std_scale

        return self.embed_vision(inputs_embeds=hidden_states)


def _set_vision_attn_impl(adapter: Gemma4VisualAdapter, impl: str):  # noqa: ARG001
    """No-op. Kept for reference. The adapter already bypasses create_bidirectional_mask."""


class _Gemma4VisualProcessor(BaseVisualProcessor):
    """Preprocessor that passes both pixel_values AND image_position_ids to the vision graph."""

    def forward(self, data: dict) -> tuple[torch.Tensor, ...]:
        return (data["image"], data["image_position_ids"])


@register_llm_model("Gemma4ForConditionalGeneration_visual", master=False)
class XHGemma4VisionModel(BaseVisionModel):  # noqa: N801
    transformers_min_version = "5.5.0"
    HF_MODEL_CLS = XHGemma4ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = VisualModelMeta
    CONFIG_CLS = XHGemma4VisualConfig

    def __init__(self, config: XHGemma4VisualConfig):
        super().__init__(config)
        gemma4_config = AutoConfig.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        self.config = cast(XHGemma4VisualConfig, self.config)
        if self.config.model_type is None:
            self.config.model_type = "Gemma4ForConditionalGeneration_visual"
        if hasattr(gemma4_config, "vision_config") and gemma4_config.vision_config is not None:
            vision_config = gemma4_config.vision_config
            if isinstance(vision_config, dict):
                patch_size = vision_config.get("patch_size")
                pooling_kernel_size = vision_config.get("pooling_kernel_size")
            else:
                patch_size = vision_config.patch_size
                pooling_kernel_size = vision_config.pooling_kernel_size
            assert patch_size == config.patch_size
            assert pooling_kernel_size == config.pooling_kernel_size

    def _to_eager(self, aligned: bool = True):
        raise NotImplementedError("Eager mode is not implemented for vision model yet.")

    def _get_data_preprocessor(self):
        return _Gemma4VisualProcessor()

    def _to_fronted(self, wrap_model):
        logger = get_xhquant_logger()
        dummy_inputs = self._get_dummy_inputs()
        pixel_values = dummy_inputs["pixel_values"].float().cpu()
        image_position_ids = dummy_inputs["image_position_ids"].long().cpu()
        work_dir = self.config.work_dir
        _tmp_dir_ctx = None
        if not work_dir:
            _tmp_dir_ctx = tempfile.TemporaryDirectory()
            work_dir = _tmp_dir_ctx.name
        try:
            onnx_file = str(Path(work_dir) / "onnx" / "gemma4_visual.onnx")
            Path(onnx_file).parent.mkdir(parents=True, exist_ok=True)
            if not Path(onnx_file).exists():
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                    torch.onnx.export(
                        wrap_model.float().cpu(),
                        (pixel_values, image_position_ids),
                        tmp_onnx_file,
                        export_params=True,
                        opset_version=18,
                        do_constant_folding=True,
                        input_names=["pixel_values", "image_position_ids"],
                        output_names=["image_embeds"],
                        verbose=False,
                    )
                    onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)
                    onnx.save(
                        onnx_model,
                        onnx_file,
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=f"{Path(onnx_file).stem}_external_data",
                    )
                self._wrap_model.to(self.device, self.dtype)
            else:
                logger.info(f"from cached onnx: {onnx_file}")

            lazy_model = lazy_load_onnx(onnx_file)
            lazy_model.load_all_tensors()
            onnx_model = lazy_model.model
            lazy_model.close()
            return to_frontend_graph(onnx_model, FrontendType.ONNX, [pixel_values, image_position_ids])
        finally:
            if _tmp_dir_ctx is not None:
                _tmp_dir_ctx.cleanup()

    def _get_dummy_inputs(self) -> Any:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (224, 224), color="white")},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        processor = self.get_tf_processor()
        inputs = processor.apply_chat_template(messages)
        return {
            "pixel_values": inputs["pixel_values"],
            "image_position_ids": inputs["image_position_ids"],
        }

    def get_dummy_inputs(self) -> Any:
        dummy = self._get_dummy_inputs()
        return {"image": dummy["pixel_values"], "image_position_ids": dummy["image_position_ids"]}

    def init_wrap_model(self, hf_model: XHGemma4ForConditionalGeneration = None):
        dummy = self._get_dummy_inputs()
        pid = dummy["image_position_ids"]  # (1, num_patches, 2)
        k = self.config.pooling_kernel_size
        k2 = k * k
        output_length = pid.shape[1] // k2

        # Pre-compute pooler weight matrix (avoids one_hot / integer ops during ONNX trace)
        clamped = pid.clamp(min=0)
        max_x = clamped[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(clamped, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        pooler_weights = F.one_hot(kernel_idxs.long(), output_length).float() / k2

        # Count valid (non-padding) pooled positions
        padding = (pid == -1).all(dim=-1)
        pooled_padding = padding.reshape(1, output_length, k2).all(dim=-1)
        num_image_tokens = int((~pooled_padding).sum().item())

        visual = Gemma4VisualAdapter(
            hf_model.model.vision_tower,
            hf_model.model.embed_vision,
            pooler_weights=pooler_weights,
            num_image_tokens=num_image_tokens,
        )
        return super().init_wrap_model(visual)

    def get_tf_processor(self):
        return XHGemma4Processor.from_pretrained(self.hf_model_dir, trust_remote_code=True)

    def forward(self, *args, **kwargs):
        return self._inference_model(*args, **kwargs)

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        kwargs.setdefault("dtype", torch.bfloat16)
        kwargs.setdefault("device_map", "auto")
        kwargs.setdefault("trust_remote_code", True)
        return super().get_hf_model(hf_model_dir, quant_weight, **kwargs)

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {"input_names": ["pixel_values", "image_position_ids"], "output_names": ["image_embeds"]}

    def export_hmonnx(self, output_dir: str) -> VisualModelMeta:
        meta_info = self.create_export_metadata(output_dir)
        exported_hmonnx_file = super()._export_hmonnx(output_dir)
        meta_info.hmonnx = str(exported_hmonnx_file)
        return meta_info

    def create_export_metadata(self, output_dir: str) -> VisualModelMeta:
        meta_info = cast(VisualModelMeta, self.get_export_metadata_cls()())
        meta_info.image_size_w = 224
        meta_info.image_size_h = 224
        meta_info.patch_size = self.config.patch_size
        return meta_info
