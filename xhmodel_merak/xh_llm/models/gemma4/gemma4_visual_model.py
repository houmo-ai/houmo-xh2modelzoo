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
from ...builder import register_llm_model
from ...types import VisualModelMeta
from .gemma4_processor import XHGemma4Processor
from .xh_gemma4_config import XHGemma4VisualConfig


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
        cos, sin = position_embeddings

        q = self.q_proj(hidden_states).view(bsz, seq, n_q, hd)
        q = self.q_norm(q)
        q = self._fused_multidim_rope(q, cos, sin).transpose(1, 2)

        k = self.k_proj(hidden_states).view(bsz, seq, n_kv, hd)
        k = self.k_norm(k)
        k = self._fused_multidim_rope(k, cos, sin).transpose(1, 2)

        v = self.v_proj(hidden_states).view(bsz, seq, n_kv, hd)
        v = self.v_norm(v)
        v = v.transpose(1, 2)

        if n_kv != n_q:
            k = k.repeat_interleave(n_q // n_kv, dim=1)
            v = v.repeat_interleave(n_q // n_kv, dim=1)

        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # x: [B, S, n_heads, head_dim]; cos/sin: [B, S, head_dim] (concat over ndim spatial dims)
        x_parts = torch.split(x, self.split_size, dim=-1)
        cos_parts = torch.split(cos, self.split_size, dim=-1)
        sin_parts = torch.split(sin, self.split_size, dim=-1)
        y_parts = []
        for k in range(self.ndim):
            c = cos_parts[k].unsqueeze(2)  # [B, S, 1, half_hd] — xhnn.Rope requires rank 4
            s = sin_parts[k].unsqueeze(2)
            y_parts.append(self.rope(x_parts[k], c, s))
        return torch.cat(y_parts, dim=-1)


class Gemma4VisualAdapter(nn.Module):
    """Trace-friendly adapter wrapping the Gemma4 vision tower + embed_vision.

    Avoids two trace-unsafe operations:

    1. ``create_bidirectional_mask`` in transformers 5.5 (trace failure) — replaced by
       manually iterating encoder layers with ``attention_mask=None``.
    2. ``Gemma4VisionPooler._avg_pool_by_positions`` which uses ``F.one_hot`` and
       integer division (``NonZero`` / ``one_hot`` not supported in frontend graph) —
       replaced by a pre-computed constant pooling weight matrix stored as a buffer.

    ``pooler_weights`` is ``(B, output_length, num_patches)`` (pre-transposed) and
    ``num_image_tokens`` is the count of pooled tokens.  Both are deterministic for a
    fixed image size and are computed once during ``init_wrap_model``.

    Only *real* (non-padding) patches are fed to the adapter.  For a 224×224 image
    this gives 2304 patches and 256 pooled tokens — no padding handling required.
    """

    def __init__(
        self,
        vision_tower: nn.Module,
        embed_vision: nn.Module,
        pooler_weights: torch.Tensor,
        num_image_tokens: int,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        pos_embed: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ):
        super().__init__()
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision
        self.num_image_tokens = num_image_tokens
        self.num_layers = vision_tower.encoder.config.num_hidden_layers
        self.register_buffer("pooler_weights", pooler_weights)
        self.register_buffer("rope_cos", rope_cos)
        self.register_buffer("rope_sin", rope_sin)
        self.register_buffer("pos_embed", pos_embed)
        if position_ids is not None:
            self.register_buffer("position_ids", position_ids)
        else:
            self.position_ids = None

    def forward(self, pixel_values: torch.Tensor):
        vt = self.vision_tower
        pe = vt.patch_embedder

        pixel_values_norm = 2 * (pixel_values - 0.5)
        hidden_states = pe.input_proj(pixel_values_norm.to(pe.input_proj.weight.dtype))
        hidden_states = hidden_states + self.pos_embed.to(hidden_states.dtype)

        rope_cos_sin = (self.rope_cos.to(hidden_states.dtype), self.rope_sin.to(hidden_states.dtype))
        for layer in vt.encoder.layers[: self.num_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_embeddings=rope_cos_sin,
                position_ids=self.position_ids,
            )

        pooled = (self.pooler_weights @ hidden_states.float()).to(hidden_states.dtype)
        pooled = pooled * vt.pooler.root_hidden_size

        if vt.config.standardize:
            pooled = (pooled - vt.std_bias) * vt.std_scale

        return self.embed_vision(inputs_embeds=pooled)


def _set_vision_attn_impl(adapter: Gemma4VisualAdapter, impl: str):  # noqa: ARG001
    """No-op. Kept for reference. The adapter already bypasses create_bidirectional_mask."""


class _Gemma4VisualProcessor(BaseVisualProcessor):
    """Preprocessor that passes pixel_values to the vision graph."""

    def forward(self, data: dict) -> tuple[torch.Tensor, ...]:
        return (data["image"],)


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
        dummy_inputs = self._get_dummy_inputs()
        pixel_values = dummy_inputs["pixel_values"].float().cpu()
        return to_frontend_graph(wrap_model.float().cpu(), "TorchFX", [pixel_values])

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
        pv = inputs["pixel_values"]        # (1, total_patches, patch_dim)
        pid = inputs["image_position_ids"]  # (1, total_patches, 2) or (total_patches, 2)
        if pid.dim() == 2:
            pid = pid.unsqueeze(0)
        # Strip padding patches (position_ids == -1) — keep only real patches
        is_real = ~(pid == -1).all(dim=-1)  # (1, total_patches)
        n_real = int(is_real[0].sum().item())
        return {
            "pixel_values": pv[:, :n_real, :],
            "image_position_ids": pid[:, :n_real, :],
        }

    def get_dummy_inputs(self) -> Any:
        dummy = self._get_dummy_inputs()
        return {"image": dummy["pixel_values"]}

    def init_wrap_model(self, hf_model: XHGemma4ForConditionalGeneration = None):
        dummy = self._get_dummy_inputs()
        pid = dummy["image_position_ids"]  # (1, num_real_patches, 2), no padding
        k = self.config.pooling_kernel_size
        k2 = k * k
        num_patches = pid.shape[1]
        output_length = num_patches // k2
        num_image_tokens = output_length  # all pooled tokens are valid

        # Pre-compute pooler weight matrix (no padding — every row is real)
        # Pre-transpose to (B, output_length, num_patches) so forward is a single matmul
        max_x = pid[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(pid, k, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
        pooler_weights = F.one_hot(kernel_idxs.long(), output_length).float() / k2
        pooler_weights = pooler_weights.transpose(1, 2)

        vt = hf_model.model.vision_tower
        embed_vision = hf_model.model.embed_vision

        with torch.no_grad():
            pid_cpu = pid.cpu()
            rope_cfg = vt.config
            head_dim = getattr(rope_cfg, "head_dim", None) or rope_cfg.hidden_size // rope_cfg.num_attention_heads
            spatial_dim = head_dim // 2
            rope_theta = rope_cfg.rope_parameters["rope_theta"]
            inv_freq = 1.0 / (
                rope_theta
                ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim)
            )
            inv_freq_expanded = inv_freq[None, :, None]
            all_cos, all_sin = [], []
            for dim_i in range(2):
                dim_pos = pid_cpu[:, :, dim_i].float()
                freqs = (inv_freq_expanded @ dim_pos[:, None, :]).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                all_cos.append(emb.cos())
                all_sin.append(emb.sin())
            rope_cos = torch.cat(all_cos, dim=-1).to(dtype=torch.bfloat16)
            rope_sin = torch.cat(all_sin, dim=-1).to(dtype=torch.bfloat16)

        pe = vt.patch_embedder
        with torch.no_grad():
            no_padding = torch.zeros(1, num_patches, dtype=torch.bool)
            pos_embed = pe._position_embeddings(pid_cpu, no_padding).to(dtype=torch.bfloat16)

        visual = Gemma4VisualAdapter(
            vt,
            embed_vision,
            pooler_weights=pooler_weights,
            num_image_tokens=num_image_tokens,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            pos_embed=pos_embed,
            position_ids=pid,
        )
        _replace_rmsnorm(visual.vision_tower)
        for layer in visual.vision_tower.encoder.layers:
            _make_vision_attn_traceable(layer.self_attn)
        return super().init_wrap_model(visual)

    def get_tf_processor(self):
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
        return {"input_names": ["pixel_values"], "output_names": ["image_embeds"]}

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
        # 224×224 → 768×768 → 48×48 = 2304 real patches → 256 pooled tokens
        meta_info.num_image_tokens = 256
        return meta_info
