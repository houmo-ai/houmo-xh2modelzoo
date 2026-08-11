from typing import Any, cast

import torch
import torch.nn as nn
from xhquant.api import to_frontend_graph

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...types import LLMModelState, VisualModelMeta
from .modeling_unlimitedocr import UnlimitedOCRForCausalLM
from .modeling_unlimitedocr_patch import unlimited_ocr_patch
from .xh_unlimited_ocr_config import XHUnlimitedOCRVisualConfig


class UnlimitedOCRBaseVisualModel(nn.Module):
    """Standalone Unlimited-OCR visual branch for base/no-crop export."""

    def __init__(self, hf_model: UnlimitedOCRForCausalLM):
        super().__init__()
        if not hasattr(hf_model, "model"):
            raise TypeError(f"Unlimited-OCR HF model should expose .model, got {type(hf_model)!r}")
        model = hf_model.model
        required_attrs = ("sam_model", "vision_model", "projector", "image_newline", "view_seperator")
        missing_attrs = [name for name in required_attrs if not hasattr(model, name)]
        if missing_attrs:
            raise AttributeError(f"Unlimited-OCR HF model is missing visual attrs: {missing_attrs}")

        self.sam_model = model.sam_model
        self.vision_model = model.vision_model
        self.projector = model.projector
        self.image_newline = model.image_newline
        self.view_seperator = model.view_seperator

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Return visual tokens for one or more 1024 base/no-crop images."""
        sam_features = self.sam_model(image)
        clip_features = self.vision_model(image, sam_features)
        visual_features = torch.cat((clip_features[:, 1:], sam_features.flatten(2).permute(0, 2, 1)), dim=-1)
        visual_features = self.projector(visual_features)

        batch_size, hw, hidden_size = visual_features.shape
        grid_size = int(hw**0.5)
        if grid_size * grid_size != hw:
            raise ValueError(f"Unlimited-OCR visual features must form a square grid, got {hw} patches.")

        visual_features = visual_features.view(batch_size, grid_size, grid_size, hidden_size)
        newline = self.image_newline.view(1, 1, 1, hidden_size).expand(batch_size, grid_size, 1, hidden_size)
        visual_features = torch.cat([visual_features, newline], dim=2).reshape(batch_size, -1, hidden_size)
        separator = self.view_seperator.view(1, 1, hidden_size).expand(batch_size, 1, hidden_size)
        return torch.cat([visual_features, separator], dim=1)

    def _encode(self, image: torch.Tensor) -> torch.Tensor:
        """SAM + CLIP + projector for a batch of images -> (n, hw, hidden)."""
        sam_features = self.sam_model(image)
        clip_features = self.vision_model(image, sam_features)
        features = torch.cat((clip_features[:, 1:], sam_features.flatten(2).permute(0, 2, 1)), dim=-1)
        return self.projector(features)

    def forward_crop(
        self,
        image_ori: torch.Tensor,
        images_crop: torch.Tensor,
        width_crop_num: int,
        height_crop_num: int,
    ) -> torch.Tensor:
        """Return crop/gundam visual tokens for a single image.

        Ordering matches the token layout built by ``XHUnlimitedOCRProcessor``:
        ``[global grid + per-row newline]`` + ``[view separator]`` +
        (when cropped) ``[local grid + per-row newline]``. This differs from the
        original model's ``[local, global, sep]`` feature order, but token order
        and feature order must agree for the masked-scatter, so both sides use
        this global-first layout in the Merak adaptation.
        """
        global_features = self._encode(image_ori)
        _, hw, hidden_size = global_features.shape
        h = w = int(hw**0.5)
        global_features = global_features.view(h, w, hidden_size)
        global_features = torch.cat(
            [global_features, self.image_newline[None, None, :].expand(h, 1, hidden_size)], dim=1
        )
        global_features = global_features.reshape(-1, hidden_size)

        separator = self.view_seperator[None, :]

        if (width_crop_num > 1 or height_crop_num > 1) and images_crop.shape[0] > 0:
            local_features = self._encode(images_crop)
            _, hw2, hidden2 = local_features.shape
            h2 = w2 = int(hw2**0.5)
            local_features = (
                local_features.view(height_crop_num, width_crop_num, h2, w2, hidden2)
                .permute(0, 2, 1, 3, 4)
                .reshape(height_crop_num * h2, width_crop_num * w2, hidden2)
            )
            local_features = torch.cat(
                [local_features, self.image_newline[None, None, :].expand(height_crop_num * h2, 1, hidden2)],
                dim=1,
            )
            local_features = local_features.reshape(-1, hidden2)
            combined = torch.cat([global_features, separator, local_features], dim=0)
        else:
            combined = torch.cat([global_features, separator], dim=0)

        return combined.unsqueeze(0)


@register_llm_model("UnlimitedOCRForCausalLM_visual", master=False)
class XHUnlimitedOCRVisualModel(BaseVisionModel):
    HF_MODEL_CLS = UnlimitedOCRBaseVisualModel
    HF_AUTO_MODEL_CLS = UnlimitedOCRForCausalLM
    META_CLS = VisualModelMeta
    CONFIG_CLS = XHUnlimitedOCRVisualConfig

    def __init__(self, config: XHUnlimitedOCRVisualConfig):
        super().__init__(config)
        self.config = config
        # crop/gundam runs through the eager visual (forward_crop) for runtime;
        # the traced/quant export path stays base/no-crop because dynamic crop
        # counts break static-shape HMONNX export (see issue 014 notes).
        if not self.config.crop_mode:
            if self.config.image_size != 1024 or self.config.base_size != 1024:
                raise ValueError("Unlimited-OCR base visual expects image_size=1024 and base_size=1024.")
        else:
            if self.config.base_size != 1024:
                raise ValueError("Unlimited-OCR gundam visual expects base_size=1024.")
        if self.config.model_type is None:
            self.config.model_type = "UnlimitedOCRForCausalLM_visual"

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs: Any):
        if "dtype" not in kwargs and "torch_dtype" not in kwargs:
            kwargs["dtype"] = cls.get_hf_model_dtype()
        kwargs.setdefault("trust_remote_code", False)
        native_model = UnlimitedOCRForCausalLM.from_pretrained(hf_model_dir, **kwargs)
        native_model = cls.untied_weights(native_model)
        if quant_weight is not None and len(quant_weight) > 0:
            cls._load_quant_weight(quant_weight, native_model, strict=False)
        return UnlimitedOCRBaseVisualModel(unlimited_ocr_patch(native_model))

    def init_wrap_model(self, hf_model: Any = None):
        from ._visual_model_impl import register_wrap_cls

        register_wrap_cls(hf_model)
        return super().init_wrap_model(hf_model)

    def _to_fronted(self, wrap_model):
        try:
            dummy_inputs = self._get_dummy_inputs()
            image = dummy_inputs["image"].float().cpu()
            return to_frontend_graph(wrap_model.float().cpu(), "TorchFX", [image])
        except Exception as exc:
            raise RuntimeError(f"Unlimited-OCR visual to_fronted failed: {exc}") from exc

    def _to_quanted(self, frontend_model, state):
        try:
            return super()._to_quanted(frontend_model, state)
        except Exception as exc:
            raise RuntimeError(f"Unlimited-OCR visual quant alignment failed: {exc}") from exc

    def forward(self, *args: Any, **kwargs: Any):
        return self._inference_model(*args, **kwargs)

    def forward_crop(self, image_ori: torch.Tensor, images_crop: torch.Tensor, *args: Any, **kwargs: Any):
        """Delegate crop/gundam visual forward to an eager base visual model.

        Crop mode runs eagerly (dynamic crop counts are not exportable), so it
        does not use the wrapped/quantized inference model. A native
        ``UnlimitedOCRBaseVisualModel`` is built once from the HF weights and
        reused for both the global view and the local crops.
        """
        eager = self._get_eager_visual(image_ori.device, image_ori.dtype)
        return eager.forward_crop(image_ori, images_crop, *args, **kwargs)

    def _get_eager_visual(self, device, dtype) -> "UnlimitedOCRBaseVisualModel":
        cached = getattr(self, "_eager_visual", None)
        if cached is None:
            native = self.get_native_model()
            if not isinstance(native, UnlimitedOCRBaseVisualModel):
                native = UnlimitedOCRBaseVisualModel(native)
            cached = native.eval()
            self._eager_visual = cached
        return cached.to(device=device, dtype=dtype)

    def _get_dummy_inputs(self):
        image = torch.zeros(
            getattr(self.config, "batch_size", 1),
            3,
            self.config.image_size,
            self.config.image_size,
            dtype=torch.float32,
        )
        return {"image": image}

    def get_dummy_inputs(self):
        return self._get_dummy_inputs()

    def get_export_cfg(self):
        return {
            "input_names": ["image"],
            "output_names": ["image_embeds"],
        }

    def export_hmonnx(self, output_dir: str) -> VisualModelMeta:
        if not self.config.hmonnx_export:
            raise NotImplementedError(
                "Unlimited-OCR gundam/crop visual export is not HMONNX-supported; "
                f"export_mode={self.config.export_mode!r}, crop_mode={self.config.crop_mode}, "
                f"hmonnx_export={self.config.hmonnx_export}. "
                "Use base/no-crop configs for HMONNX export or eager forward_crop for crop mode."
            )
        try:
            meta_info = self.create_export_metadata(output_dir)
            exported_hmonnx_file = super()._export_hmonnx(output_dir)
            meta_info.hmonnx = str(exported_hmonnx_file)
            return meta_info
        except Exception as exc:
            raise RuntimeError(f"Unlimited-OCR visual export_hmonnx failed: {exc}") from exc

    def create_export_metadata(self, output_dir: str) -> VisualModelMeta:
        meta_info = self.get_export_metadata_cls()()
        meta_info = cast(VisualModelMeta, meta_info)
        meta_info.image_size_w = self.config.image_size
        meta_info.image_size_h = self.config.image_size
        meta_info.base_size = self.config.base_size
        meta_info.patch_size = self.config.patch_size
        meta_info.downsample_ratio = self.config.downsample_ratio
        meta_info.export_mode = self.config.export_mode
        meta_info.crop_mode = self.config.crop_mode
        meta_info.hmonnx_export = self.config.hmonnx_export
        meta_info.image_token_id = self.config.image_token_id
        image_grid_size = self.config.image_size // self.config.patch_size // self.config.downsample_ratio
        meta_info.image_token_count = (image_grid_size + 1) * image_grid_size + 1
        return meta_info

    @property
    def state(self) -> LLMModelState:
        return self._state
