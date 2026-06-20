from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration as XHGemma4ForConditionalGeneration,
    Gemma4RMSNorm,
)

import xhquant.nn.modules as xhnn
from xhquant.api import to_frontend_graph

from ...base_vision_model import BaseVisionModel
from ...llm_data_processor import BaseVisualProcessor
from ...types import VisualModelMeta
from .gemma4_series_processor import XHGemma4Processor
from .xh_gemma4_series_config import XHGemma4SeriesVisualConfig


def _replace_rmsnorm(module: nn.Module) -> None:
    """Replace all Gemma4RMSNorm instances with xhnn.RMSNorm in-place."""
    for name, child in list(module.named_children()):
        if isinstance(child, Gemma4RMSNorm):
            if child.with_scale:
                hidden_size = child.weight.shape[0]
                device = child.weight.device
                dtype = child.weight.dtype
                fused = xhnn.RMSNorm(hidden_size, eps=child.eps)
                fused.weight.data.copy_(child.weight.data)
                fused = fused.to(device=device, dtype=dtype)
            else:
                hidden_size = _infer_rmsnorm_dim(module, name, child)
                fused = xhnn.RMSNorm(hidden_size, eps=child.eps)
                fused.weight.data.fill_(1.0)
                fused.weight.requires_grad = False
                # Move to same device/dtype as sibling parameters
                for p in module.parameters():
                    fused = fused.to(device=p.device, dtype=p.dtype)
                    break
            setattr(module, name, fused)
        else:
            _replace_rmsnorm(child)


def _infer_rmsnorm_dim(parent: nn.Module, name: str, norm: Gemma4RMSNorm) -> int:
    """Infer the hidden_size of a with_scale=False RMSNorm from context."""
    if hasattr(parent, "multimodal_hidden_size"):
        return parent.multimodal_hidden_size
    if name == "embedding_pre_projection_norm" and hasattr(parent, "embedding_projection"):
        return parent.embedding_projection.in_features
    # v_norm in attention has head_dim size
    if hasattr(parent, "head_dim"):
        return parent.head_dim
    # Fallback: check if the norm has been used with a known tensor size
    # For Gemma4 vision, with_scale=False only appears in v_norm (head_dim=72)
    return 72  # safe default for gemma4 vision


def _make_vision_attn_traceable(attn: nn.Module) -> None:
    """Override Gemma4VisionAttention.forward to be torch.fx-traceable.

    HF's forward uses ``(*input_shape, -1, head_dim)`` and
    ``apply_multidimensional_rope`` derives ``ndim`` from
    ``position_ids.shape[-1]`` then iterates ``range(ndim)``. Both unpack /
    iterate Proxy values that ``torch.fx.Tracer`` cannot handle. Additionally
    HF's ``rotate_half`` does ``x.shape[-1] // 2`` which produces a ``FloorDiv``
    node the xh2a quant pipeline cannot lower.

    This rewrite hardcodes ``[B, S, ...]`` rank and ``ndim=2`` (vision RoPE
    operates on x/y spatial dims), and replaces HF ``apply_rotary_pos_emb``
    with ``xhnn.Rope`` — an FX leaf module whose rotate_half internals stay
    opaque to fx (so no shape-dependent FloorDiv leaks into the graph).
    The Split + Rope op pattern matches the gemma4_moe export onnx.
    """
    import types

    fused_rope = _FusedMultidimRope(head_dim=attn.head_dim, ndim=2)
    attn._fused_multidim_rope = fused_rope
    attn.masked_add = xhnn.MaskedAdd()

    def _forward(
        self,
        hidden_states,
        position_embeddings=None,
        attention_mask=None,
        position_ids=None,
        **kwargs,
    ):
        bsz = hidden_states.shape[0]
        seq = hidden_states.shape[1]
        n_q = self.config.num_attention_heads
        n_kv = self.config.num_key_value_heads
        hd = self.head_dim

        q = self.q_proj(hidden_states).view(bsz, seq, n_q, hd)
        q = self.q_norm(q)
        q = self._fused_multidim_rope(q, position_embeddings).transpose(1, 2)

        k = self.k_proj(hidden_states).view(bsz, seq, n_kv, hd)
        k = self.k_norm(k)
        k = self._fused_multidim_rope(k, position_embeddings).transpose(1, 2)

        v = self.v_proj(hidden_states).view(bsz, seq, n_kv, hd)
        v = self.v_norm(v)
        v = v.transpose(1, 2)

        if n_kv != n_q:
            k = k.repeat_interleave(n_q // n_kv, dim=1)
            v = v.repeat_interleave(n_q // n_kv, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        if attention_mask is not None:
            attn_weights = self.masked_add(attn_weights, attention_mask)
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v).transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, seq, -1).contiguous()
        return self.o_proj(attn_output), None

    attn.forward = types.MethodType(_forward, attn)


class _FusedMultidimRope(nn.Module):
    """Multi-dim RoPE = split head_dim into ndim halves, apply xhnn.Rope per half, concat.

    For Gemma4 vision (ndim=2), x is split along the head_dim into x/y halves,
    each half gets its own xhnn.Rope call (with the matching cos/sin half), and
    the results are concatenated back. The split sizes are baked in at module
    construction so they never depend on traced Proxy shapes.
    """

    def __init__(self, head_dim: int, ndim: int = 2):
        super().__init__()
        self.ndim = ndim
        self.split_size = head_dim // ndim
        self.rope = xhnn.Rope()

    def forward(
        self,
        x: torch.Tensor,
        cos_sin_parts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        # x: [B, S, n_heads, head_dim]. cos/sin parts are prepared once as
        # [B, S, 1, half_hd], avoiding per-block split/unsqueeze on position ids.
        cos_x, cos_y, sin_x, sin_y = cos_sin_parts
        x_x, x_y = torch.split(x, self.split_size, dim=-1)
        y_x = self.rope(x_x, cos_x, sin_x)
        y_y = self.rope(x_y, cos_y, sin_y)
        return torch.cat((y_x, y_y), dim=-1)


class Gemma4VisualAdapter(nn.Module):
    """Trace-friendly adapter wrapping the Gemma4 vision tower + embed_vision.

    Exported visual graph follows HF's padded Gemma4 input contract:

    * ``pixel_values``: [1, 2520, 768], padded at the tail.
    * ``pixel_position_ids``: [1, 2520, 2], NPU-safe positions where padding
      rows are [0, 0] instead of HF's [-1, -1].
    * ``pooling_matrix``: [1, 280, 2520], host-generated average-pooling
      matrix for the position-aware 3x3 pool, pre-transposed for MatMul.
    * ``attention_mask``: [1, 1, 1, 2520], additive key mask.

    Position embeddings are gathered from HF's x/y embedding table online.
    RoPE cos/sin are gathered from constant precomputed tables by
    ``pixel_position_ids``. This keeps ``Cos`` / ``Sin`` out of the ONNX graph
    without adding rope tensors as runtime inputs. Pooling uses a host-built
    matrix multiplication instead of runtime gather/reduce.
    """

    def __init__(
        self,
        vision_tower: nn.Module,
        embed_vision: nn.Module,
        num_image_tokens: int,
        position_embedding_table: torch.Tensor,
        rope_cos_table: torch.Tensor,
        rope_sin_table: torch.Tensor,
    ):
        super().__init__()
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision
        self.num_image_tokens = num_image_tokens
        self.num_layers = vision_tower.encoder.config.num_hidden_layers
        self.pooling_kernel_area = int(vision_tower.config.pooling_kernel_size) ** 2
        self.hidden_size = int(vision_tower.config.hidden_size)
        self.register_buffer("position_embedding_x", position_embedding_table[0].contiguous())
        self.register_buffer("position_embedding_y", position_embedding_table[1].contiguous())
        self.register_buffer("rope_cos_table", rope_cos_table)
        self.register_buffer("rope_sin_table", rope_sin_table)

    def _position_embeddings(self, pixel_position_ids: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        pos = pixel_position_ids.long()
        pos_x = F.embedding(pos[..., 0], self.position_embedding_x.to(dtype))
        pos_y = F.embedding(pos[..., 1], self.position_embedding_y.to(dtype))
        return pos_x + pos_y

    def _pool_by_matrix(self, hidden_states: torch.Tensor, pooling_matrix: torch.Tensor) -> torch.Tensor:
        pooling_matrix = pooling_matrix.to(hidden_states.dtype)
        pooled = torch.matmul(pooling_matrix, hidden_states.float())
        return pooled.to(hidden_states.dtype)

    def _rope_embeddings(
        self,
        pixel_position_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pos = pixel_position_ids.long()
        cos_table = self.rope_cos_table.to(dtype)
        sin_table = self.rope_sin_table.to(dtype)
        cos_x = F.embedding(pos[..., 0], cos_table)
        cos_y = F.embedding(pos[..., 1], cos_table)
        sin_x = F.embedding(pos[..., 0], sin_table)
        sin_y = F.embedding(pos[..., 1], sin_table)
        return (
            cos_x.unsqueeze(2),
            cos_y.unsqueeze(2),
            sin_x.unsqueeze(2),
            sin_y.unsqueeze(2),
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        pooling_matrix: torch.Tensor,
        attention_mask: torch.Tensor,
    ):
        vt = self.vision_tower
        pe = vt.patch_embedder

        pixel_values_norm = 2 * (pixel_values - 0.5)
        hidden_states = pe.input_proj(pixel_values_norm.to(pe.input_proj.weight.dtype))
        hidden_states = hidden_states + self._position_embeddings(pixel_position_ids, hidden_states.dtype)

        rope_cos_sin = self._rope_embeddings(pixel_position_ids, hidden_states.dtype)
        attention_mask = attention_mask.to(hidden_states.dtype)
        for layer in vt.encoder.layers[: self.num_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=rope_cos_sin,
                position_ids=pixel_position_ids,
            )

        pooled = self._pool_by_matrix(hidden_states, pooling_matrix)
        pooled = pooled * vt.pooler.root_hidden_size

        if vt.config.standardize:
            pooled = (pooled - vt.std_bias) * vt.std_scale

        return self.embed_vision(inputs_embeds=pooled)


def _set_vision_attn_impl(adapter: Gemma4VisualAdapter, impl: str):  # noqa: ARG001
    """No-op. Kept for reference. The adapter already bypasses create_bidirectional_mask."""


class _Gemma4VisualProcessor(BaseVisualProcessor):
    """Preprocessor that passes padded Gemma4 vision tensors to the graph."""

    def forward(self, data: dict) -> tuple[torch.Tensor, ...]:
        return (
            data["image"],
            data["pixel_position_ids"],
            data["pooling_matrix"],
            data["attention_mask"],
        )


class XHGemma4SeriesVisionModel(BaseVisionModel):
    transformers_min_version = "5.5.0"
    HF_MODEL_CLS = XHGemma4ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = VisualModelMeta
    CONFIG_CLS = XHGemma4SeriesVisualConfig

    def __init__(self, config: XHGemma4SeriesVisualConfig):
        super().__init__(config)
        gemma4_config = AutoConfig.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        self.config = cast(XHGemma4SeriesVisualConfig, self.config)
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
        dummy_inputs = self._get_dummy_inputs()
        pixel_values = dummy_inputs["pixel_values"].float().cpu()
        pixel_position_ids = dummy_inputs["pixel_position_ids"].long().cpu()
        pooling_matrix = dummy_inputs["pooling_matrix"].float().cpu()
        attention_mask = dummy_inputs["attention_mask"].float().cpu()
        return to_frontend_graph(
            wrap_model.float().cpu(),
            "TorchFX",
            [pixel_values, pixel_position_ids, pooling_matrix, attention_mask],
        )

    def _get_dummy_inputs(self) -> Any:
        processor = self.get_tf_processor()
        if getattr(self.config, "input_modality", "image") == "video":
            frames = [Image.new("RGB", (224, 224), color="white") for _ in range(32)]
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": frames},
                        {"type": "text", "text": "Describe this video."},
                    ],
                }
            ]
            inputs = processor.apply_chat_template(messages)
            return {
                "pixel_values": inputs["pixel_values_videos"][:, 0],
                "pixel_position_ids": inputs["video_pixel_position_ids"][:, 0],
                "pooling_matrix": inputs["video_pooling_matrix"][:, 0],
                "attention_mask": inputs["video_visual_attention_mask"][:, 0],
                "image_soft_token_count": inputs["video_soft_token_count"][:, 0],
            }

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (224, 224), color="white")},
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        inputs = processor.apply_chat_template(messages)
        return {
            "pixel_values": inputs["pixel_values"],
            "pixel_position_ids": inputs["pixel_position_ids"],
            "pooling_matrix": inputs["pooling_matrix"],
            "attention_mask": inputs["visual_attention_mask"],
            "image_soft_token_count": inputs["image_soft_token_count"],
        }

    def get_dummy_inputs(self) -> Any:
        dummy = self._get_dummy_inputs()
        return {
            "image": dummy["pixel_values"],
            "pixel_position_ids": dummy["pixel_position_ids"],
            "pooling_matrix": dummy["pooling_matrix"],
            "attention_mask": dummy["attention_mask"],
        }


    @staticmethod
    def _build_rope_table(vision_tower: nn.Module, *, kind: str) -> torch.Tensor:
        inv_freq = vision_tower.encoder.rotary_emb.inv_freq.detach().float().cpu()
        max_positions = int(vision_tower.patch_embedder.position_embedding_table.shape[1])
        positions = torch.arange(max_positions, dtype=torch.float32).unsqueeze(1)
        freqs = positions * inv_freq.unsqueeze(0)
        emb = torch.cat((freqs, freqs), dim=-1)
        table = emb.cos() if kind == "cos" else emb.sin()
        attention_scaling = float(getattr(vision_tower.encoder.rotary_emb, "attention_scaling", 1.0))
        return table * attention_scaling

    def init_wrap_model(self, hf_model: XHGemma4ForConditionalGeneration = None):
        vt = hf_model.model.vision_tower
        embed_vision = hf_model.model.embed_vision
        num_image_tokens = int(self.config.image_seq_length)

        visual = Gemma4VisualAdapter(
            vt,
            embed_vision,
            num_image_tokens=num_image_tokens,
            position_embedding_table=vt.patch_embedder.position_embedding_table.detach().cpu(),
            rope_cos_table=self._build_rope_table(vt, kind="cos"),
            rope_sin_table=self._build_rope_table(vt, kind="sin"),
        )
        _replace_rmsnorm(visual.vision_tower)
        _replace_rmsnorm(visual.embed_vision)
        for layer in visual.vision_tower.encoder.layers:
            _make_vision_attn_traceable(layer.self_attn)
        return super().init_wrap_model(visual)

    def get_tf_processor(self):
        if getattr(self.config, "input_modality", "image") == "video":
            return XHGemma4Processor.from_pretrained(
                self.hf_model_dir,
                trust_remote_code=True,
                video_max_patches=self.config.max_patches,
                video_image_seq_length=self.config.image_seq_length,
                video_pooling_kernel_size=self.config.pooling_kernel_size,
            )
        return XHGemma4Processor.from_pretrained(self.hf_model_dir, trust_remote_code=True)

    def forward(self, *args, **kwargs):
        return self._inference_model(*args, **kwargs)

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        kwargs.setdefault("dtype", torch.bfloat16)
        kwargs.setdefault("device_map", "cpu")
        kwargs.setdefault("trust_remote_code", True)
        return super().get_hf_model(hf_model_dir, quant_weight, **kwargs)

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {
            "input_names": ["pixel_values", "pixel_position_ids", "pooling_matrix", "attention_mask"],
            "output_names": ["image_embeds"],
        }

    def export_hmonnx(self, output_dir: str) -> VisualModelMeta:
        meta_info = self.create_export_metadata(output_dir)
        exported_hmonnx_file = super()._export_hmonnx(output_dir)
        meta_info.hmonnx = str(exported_hmonnx_file)
        return meta_info

    def create_export_metadata(self, output_dir: str) -> VisualModelMeta:
        meta_info = cast(VisualModelMeta, self.get_export_metadata_cls()())
        meta_info.image_size_w = 0
        meta_info.image_size_h = 0
        meta_info.patch_size = self.config.patch_size
        meta_info.max_patches = self.config.max_patches
        meta_info.num_image_tokens = self.config.image_seq_length
        meta_info.pooling_kernel_size = self.config.pooling_kernel_size
        meta_info.input_modality = self.config.input_modality
        return meta_info


XHGemma4VisionModel = XHGemma4SeriesVisionModel


__all__ = [
    "Gemma4VisualAdapter",
    "XHGemma4SeriesVisionModel",
    "XHGemma4VisionModel",
]
