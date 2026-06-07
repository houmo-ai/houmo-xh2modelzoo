from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...types import VisualModelMeta
from ...utils import unfold_args
from ..gemma4.gemma4_visual_model import _make_vision_attn_traceable
from .xh_gemma4_moe_config import XHGemma4MoeVisualConfig


DEFAULT_IMAGE_SIZE = (448, 448)
DEFAULT_PATCH_SIZE = 16
DEFAULT_MAX_SOFT_TOKENS = 280


def _resolve_pooling_kernel_size(upsample_token: bool) -> int:
    return 3 if upsample_token else 1


def _build_position_ids(height: int, width: int, patch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
    patch_height = height // patch_size
    patch_width = width // patch_size
    patch_grid = torch.meshgrid(
        torch.arange(patch_width, device=device),
        torch.arange(patch_height, device=device),
        indexing="xy",
    )
    return torch.stack(patch_grid, dim=-1).reshape(-1, 2)


def _get_aspect_ratio_preserving_size(
    height: int,
    width: int,
    patch_size: int,
    max_patches: int,
    pooling_kernel_size: int,
) -> tuple[int, int]:
    target_pixels = max_patches * patch_size**2
    scale = math.sqrt(target_pixels / (height * width))
    side_multiple = pooling_kernel_size * patch_size
    target_height = int(math.floor((height * scale) / side_multiple)) * side_multiple
    target_width = int(math.floor((width * scale) / side_multiple)) * side_multiple

    if target_height == 0 and target_width == 0:
        raise ValueError(
            "Attempting to resize to a 0 x 0 image. "
            f"Resized dimensions must be divisible by {side_multiple}."
        )

    max_side_length = (max_patches // pooling_kernel_size**2) * side_multiple
    if target_height == 0:
        target_height = side_multiple
        target_width = min(int(math.floor(width / height)) * side_multiple, max_side_length)
    elif target_width == 0:
        target_width = side_multiple
        target_height = min(int(math.floor(height / width)) * side_multiple, max_side_length)

    if target_height * target_width > target_pixels:
        raise ValueError(
            f"Resizing [{height}x{width}] to [{target_height}x{target_width}] exceeds "
            f"{max_patches} patches with patch_size {patch_size}."
        )
    return target_height, target_width


def _image_to_patches(
    image: Any,
    patch_size: int,
    max_soft_tokens: int,
    pooling_kernel_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    from PIL import Image

    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    width, height = image.size
    max_patches = max_soft_tokens * pooling_kernel_size**2
    height, width = _get_aspect_ratio_preserving_size(
        height=height,
        width=width,
        patch_size=patch_size,
        max_patches=max_patches,
        pooling_kernel_size=pooling_kernel_size,
    )
    if image.size != (width, height):
        resample = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
        image = image.resize((width, height), resample)
    data = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    data = data.reshape(height, width, 3).permute(2, 0, 1).float() / 255.0
    num_channels, image_height, image_width = data.shape
    num_patches_height = image_height // patch_size
    num_patches_width = image_width // patch_size
    patches = data.reshape(num_channels, num_patches_height, patch_size, num_patches_width, patch_size)
    patches = patches.permute(1, 3, 2, 4, 0).reshape(num_patches_height * num_patches_width, -1)
    position_ids = _build_position_ids(image_height, image_width, patch_size)
    return patches, position_ids


class XHGemma4MoeVisualProcessor:
    """Small Gemma4 visual-only processor used by Merak visual export.

    The xhquant environment used by this workspace does not expose HF's
    Gemma4 processor through ``AutoProcessor``. This processor keeps the same
    visual tensor contract needed by the source scripts: ``pixel_values``,
    ``image_position_ids`` and ``num_soft_tokens_per_image``.
    """

    class _ImageProcessor:
        def __init__(self, patch_size: int, max_soft_tokens: int, pooling_kernel_size: int):
            self.patch_size = patch_size
            self.max_soft_tokens = max_soft_tokens
            self.pooling_kernel_size = pooling_kernel_size

    def __init__(
        self,
        *,
        patch_size: int = DEFAULT_PATCH_SIZE,
        max_soft_tokens: int = DEFAULT_MAX_SOFT_TOKENS,
        pooling_kernel_size: int = 1,
        image_seq_length: int | None = None,
    ):
        self.image_processor = self._ImageProcessor(patch_size, max_soft_tokens, pooling_kernel_size)
        self.image_seq_length = image_seq_length if image_seq_length is not None else max_soft_tokens

    @classmethod
    def from_pretrained(cls, hf_model_dir: str | Path, *, upsample_token: bool = False):
        import json

        processor_config_path = Path(hf_model_dir) / "processor_config.json"
        patch_size = DEFAULT_PATCH_SIZE
        max_soft_tokens = DEFAULT_MAX_SOFT_TOKENS
        pooling_kernel_size = _resolve_pooling_kernel_size(upsample_token)
        if processor_config_path.exists():
            with open(processor_config_path) as f:
                processor_config = json.load(f)
            image_config = processor_config.get("image_processor", {})
            patch_size = int(image_config.get("patch_size", patch_size))
            max_soft_tokens = int(image_config.get("max_soft_tokens", max_soft_tokens))
            # Prefer the value from processor_config.json; only fall back to
            # the upsample-token-derived value when the config omits it.
            pooling_kernel_size = int(image_config.get("pooling_kernel_size", pooling_kernel_size))
        image_seq_length = max_soft_tokens if upsample_token else DEFAULT_IMAGE_SIZE[0] // 28 * DEFAULT_IMAGE_SIZE[1] // 28
        return cls(
            patch_size=patch_size,
            max_soft_tokens=max_soft_tokens,
            pooling_kernel_size=pooling_kernel_size,
            image_seq_length=image_seq_length,
        )

    def __call__(self, images: Any, *, return_tensors: str = "pt", **_: Any) -> dict[str, Any]:
        if not isinstance(images, (list, tuple)):
            images = [images]
        pixel_values = []
        position_ids = []
        num_soft_tokens_per_image = []
        patch_size = self.image_processor.patch_size
        pooling_kernel_size = self.image_processor.pooling_kernel_size
        max_patches = self.image_processor.max_soft_tokens * pooling_kernel_size**2
        for image in images:
            patches, positions = _image_to_patches(
                image,
                patch_size,
                self.image_processor.max_soft_tokens,
                pooling_kernel_size,
            )
            num_soft_tokens_per_image.append(patches.shape[0] // pooling_kernel_size**2)
            if patches.shape[0] < max_patches:
                padding = max_patches - patches.shape[0]
                patches = torch.nn.functional.pad(patches, (0, 0, 0, padding), value=0)
                positions = torch.nn.functional.pad(positions, (0, 0, 0, padding), value=-1)
            pixel_values.append(patches)
            position_ids.append(positions)
        data = {
            "pixel_values": torch.stack(pixel_values, dim=0),
            "image_position_ids": torch.stack(position_ids, dim=0),
            "num_soft_tokens_per_image": num_soft_tokens_per_image,
        }
        if return_tensors != "pt":
            raise ValueError("Gemma4 visual processor currently supports return_tensors='pt' only.")
        return data


class Gemma4MoeVisionWrapper(nn.Module):
    """Gemma4 vision_tower + embed_vision wrapper aligned with xhquant_llm."""

    def __init__(self, vision_tower: nn.Module, embed_vision: nn.Module, vision_config: Any):
        super().__init__()
        self.vision_tower = vision_tower
        self.embed_vision = embed_vision
        self.vision_config = vision_config
        self._bypass_pooler = False
        self._output_length = 0

    @torch.no_grad()
    def precompute_constants(self, pixel_values: torch.Tensor, pixel_position_ids: torch.Tensor):
        import torch.nn.functional as F

        vt = self.vision_tower
        padding_positions = (pixel_position_ids == -1).all(dim=-1)
        valid_mask = ~padding_positions
        self._valid_mask = valid_mask

        n_valid = valid_mask[0].sum().item()
        valid_pv = pixel_values[:, valid_mask[0]]
        valid_pid = pixel_position_ids[:, valid_mask[0]]
        valid_pad = padding_positions[:, valid_mask[0]]

        pooling_kernel_size = vt.config.pooling_kernel_size
        output_length = n_valid // (pooling_kernel_size * pooling_kernel_size)
        self._output_length = output_length
        self._bypass_pooler = pooling_kernel_size == 1

        pos_embed = vt.patch_embedder._position_embeddings(valid_pid, valid_pad)
        self.register_buffer("precomputed_pos_embed", pos_embed)

        inputs_embeds = vt.patch_embedder(valid_pv, valid_pid, valid_pad)
        cos, sin = vt.encoder.rotary_emb(inputs_embeds, valid_pid)
        self.register_buffer("precomputed_cos", cos)
        self.register_buffer("precomputed_sin", sin)
        self.register_buffer("dummy_position_ids", valid_pid.clone())

        if not self._bypass_pooler:
            clamped_positions = valid_pid.clamp(min=0)
            k = int((n_valid // output_length) ** 0.5)
            k_squared = k**2
            max_x = clamped_positions[..., 0].max(dim=-1, keepdim=True)[0] + 1
            kernel_idxs = torch.div(clamped_positions, k, rounding_mode="floor")
            kernel_idxs = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
            weights = F.one_hot(kernel_idxs.long(), output_length).float() / k_squared
            self.register_buffer("precomputed_pool_weights", weights.transpose(1, 2))

        self._root_hidden_size = vt.pooler.root_hidden_size

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vt = self.vision_tower
        pixel_values_scaled = 2 * (pixel_values - 0.5)
        hidden_states = vt.patch_embedder.input_proj(
            pixel_values_scaled.to(vt.patch_embedder.input_proj.weight.dtype)
        )
        hidden_states = hidden_states + self.precomputed_pos_embed

        position_embeddings = (self.precomputed_cos, self.precomputed_sin)
        for layer in vt.encoder.layers[: vt.encoder.config.num_hidden_layers]:
            hidden_states = layer(
                hidden_states,
                attention_mask=None,
                position_embeddings=position_embeddings,
                position_ids=self.dummy_position_ids,
            )

        if self._bypass_pooler:
            pooled = hidden_states * self._root_hidden_size
        else:
            pooled = self.precomputed_pool_weights @ hidden_states.float()
            pooled = pooled.to(hidden_states.dtype)
            pooled = pooled * self._root_hidden_size

        if vt.config.standardize:
            pooled = (pooled - vt.std_bias) * vt.std_scale
        return self.embed_vision(pooled)


@register_llm_model("Gemma4ForConditionalGeneration_visual", master=False)
class XHGemma4MoeVisualModel(BaseVisionModel):
    transformers_min_version = "4.57.0"
    HF_MODEL_CLS = AutoModelForImageTextToText
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = VisualModelMeta
    CONFIG_CLS = XHGemma4MoeVisualConfig

    def __init__(self, config: XHGemma4MoeVisualConfig):
        super().__init__(config)
        self.config = cast(XHGemma4MoeVisualConfig, self.config)
        if self.config.model_type is None:
            self.config.model_type = "Gemma4ForConditionalGeneration_visual"

    def _to_eager(self, aligned: bool = True):
        raise NotImplementedError("Eager mode is not implemented for Gemma4 visual model yet.")

    def _to_fronted(self, wrap_model):
        from xhquant.api import FrontendType, to_frontend_graph

        dummy_inputs = self.get_dummy_inputs()
        dummy_input = unfold_args(self.get_data_preprocessor()(dummy_inputs))[0]
        return to_frontend_graph(wrap_model, FrontendType.TorchFX, [dummy_input])

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        patch_size = self._get_patch_size()
        width = self.config.max_size_w
        height = self.config.max_size_h
        patch_dim = patch_size * patch_size * 3
        patch_count = (width // patch_size) * (height // patch_size)
        return {
            "image": torch.zeros((1, patch_count, patch_dim), dtype=torch.float32),
        }

    def init_wrap_model(self, hf_model: Any = None):
        if hf_model is None:
            hf_model = self.get_native_model()
        vision_tower = hf_model.model.vision_tower
        embed_vision = hf_model.model.embed_vision
        vision_config = hf_model.config.vision_config
        vision_config._attn_implementation = "eager"
        vision_config.pooling_kernel_size = _resolve_pooling_kernel_size(self.config.upsample_token)
        vision_wrapper = Gemma4MoeVisionWrapper(vision_tower, embed_vision, vision_config)
        processor = self.get_tf_processor()
        from PIL import Image

        processed = processor(images=Image.new("RGB", (self.config.max_size_w, self.config.max_size_h)))
        vision_wrapper.precompute_constants(processed["pixel_values"], processed["image_position_ids"])
        for layer in vision_wrapper.vision_tower.encoder.layers:
            _make_vision_attn_traceable(layer.self_attn)
        return super().init_wrap_model(vision_wrapper)

    def get_tf_processor(self) -> XHGemma4MoeVisualProcessor:
        return XHGemma4MoeVisualProcessor.from_pretrained(
            self.hf_model_dir,
            upsample_token=self.config.upsample_token,
        )

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {
            "input_names": ["pixel_values"],
            "output_names": ["image_embeds"],
        }

    def export_hmonnx(self, output_dir: str) -> VisualModelMeta:
        meta_info = self.create_export_metadata(output_dir)
        exported_hmonnx_file = super()._export_hmonnx(output_dir)
        meta_info.hmonnx = str(exported_hmonnx_file)
        return meta_info

    def create_export_metadata(self, output_dir: str) -> VisualModelMeta:
        meta_info = self.get_export_metadata_cls()()
        meta_info = cast(VisualModelMeta, meta_info)
        meta_info.image_size_w = self.config.max_size_w
        meta_info.image_size_h = self.config.max_size_h
        meta_info.patch_size = self._get_patch_size()
        meta_info.max_soft_tokens = DEFAULT_MAX_SOFT_TOKENS
        meta_info.upsample_token = self.config.upsample_token
        meta_info.processor_pooling_kernel_size = _resolve_pooling_kernel_size(self.config.upsample_token)
        meta_info.vision_variant = self._build_vision_variant_name()
        meta_info.hf_config = str(Path(output_dir) / "hf_config")
        return meta_info

    def _get_patch_size(self) -> int:
        import json

        processor_config_path = Path(self.hf_model_dir) / "processor_config.json"
        if processor_config_path.exists():
            with open(processor_config_path) as f:
                processor_config = json.load(f)
            return int(processor_config.get("image_processor", {}).get("patch_size", DEFAULT_PATCH_SIZE))
        return DEFAULT_PATCH_SIZE

    def _build_vision_variant_name(self) -> str:
        token_tag = "upsample_token" if self.config.upsample_token else "no_upsample_token"
        return f"{token_tag}_{self.config.max_size_w}x{self.config.max_size_h}"