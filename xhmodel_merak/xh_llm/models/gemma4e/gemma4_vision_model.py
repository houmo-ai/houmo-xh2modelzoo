import tempfile
from pathlib import Path
from typing import Any, cast

import onnx
from PIL import Image
import torch
from torch import nn
from transformers import AutoModelForImageTextToText
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
from xhquant.api import FrontendType, get_xhquant_logger, to_frontend_graph

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...llm_data_processor import BaseVisualProcessor
from ...types import VisualModelMeta
from .gemma4_processor import XHGemma4Processor
from .xh_gemma4_config import XHGemma4VisualConfig


class _Gemma4VisionExportBridge(nn.Module):
    def __init__(self, hf_model: Gemma4ForConditionalGeneration):
        super().__init__()
        self.vision_tower = hf_model.model.vision_tower
        self.embed_vision = hf_model.model.embed_vision

    def forward(self, pixel_values, image_position_ids):
        vision_hidden_states = self.vision_tower(
            pixel_values=pixel_values,
            pixel_position_ids=image_position_ids,
        )
        if hasattr(vision_hidden_states, "last_hidden_state"):
            vision_hidden_states = vision_hidden_states.last_hidden_state
        vision_hidden_mask = None
        if isinstance(vision_hidden_states, (tuple, list)):
            vision_hidden_states, vision_hidden_mask = vision_hidden_states

        image_embeds = self.embed_vision(inputs_embeds=vision_hidden_states)
        if vision_hidden_mask is None:
            return image_embeds
        return image_embeds, vision_hidden_mask


class _Gemma4VisualProcessor(BaseVisualProcessor):
    def forward(self, data: dict) -> list[torch.Tensor]:
        assert isinstance(data, dict), "Input data should be a dictionary."
        pixel_values = data.get("pixel_values", data.get("image"))
        image_position_ids = data.get("image_position_ids")
        assert pixel_values is not None and image_position_ids is not None, (
            "Gemma4 visual export requires both `pixel_values` and `image_position_ids`."
        )
        return (pixel_values, image_position_ids)


@register_llm_model("Gemma4ForConditionalGeneration_visual", master=False)
class XHGemma4VisionModel(BaseVisionModel):
    HF_MODEL_CLS = Gemma4ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = VisualModelMeta
    CONFIG_CLS = XHGemma4VisualConfig

    def __init__(self, config: XHGemma4VisualConfig):
        super().__init__(config)
        self.config = cast(XHGemma4VisualConfig, self.config)

    def init_wrap_model(self, hf_model: Any = None) -> Any:
        from ._vision_model_impl import register_wrap_modules

        register_wrap_modules(hf_model)
        if hf_model is not None and hasattr(hf_model, "model") and hasattr(hf_model.model, "vision_tower"):
            hf_model = _Gemma4VisionExportBridge(hf_model)
        return super().init_wrap_model(hf_model)

    def get_tf_processor(self):
        processor = XHGemma4Processor.from_pretrained(self.hf_model_dir)
        processor.config.max_size_h = self.config.max_size_h
        processor.config.max_size_w = self.config.max_size_w
        processor.config.patch_size = self.config.patch_size
        return processor

    def _get_data_preprocessor(self) -> BaseVisualProcessor:
        return _Gemma4VisualProcessor()

    def get_dummy_inputs(self) -> Any:
        processor = self.get_tf_processor()
        model_inputs = processor(
            text="<|image|>Describe the image briefly.",
            images=Image.new("RGB", (self.config.max_size_w, self.config.max_size_h), color="white"),
            return_tensors="pt",
        )
        return {
            "pixel_values": model_inputs["pixel_values"],
            "image_position_ids": model_inputs["image_position_ids"],
        }

    def _to_fronted(self, wrap_model):
        logger = get_xhquant_logger()
        dummy_inputs = self.get_dummy_inputs()
        input_names = list(dummy_inputs.keys())
        dummy_values = tuple(value.float().cpu() if value.is_floating_point() else value.cpu() for value in dummy_inputs.values())

        work_dir = self.config.work_dir
        tmp_dir_ctx = None
        if not work_dir:
            tmp_dir_ctx = tempfile.TemporaryDirectory()
            work_dir = tmp_dir_ctx.name

        try:
            onnx_file = str(Path(work_dir) / "onnx" / "gemma4_visual.onnx")
            Path(onnx_file).parent.mkdir(parents=True, exist_ok=True)
            if not Path(onnx_file).exists():
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                    torch.onnx.export(
                        wrap_model.float().cpu(),
                        dummy_values,
                        tmp_onnx_file,
                        export_params=True,
                        opset_version=18,
                        do_constant_folding=True,
                        input_names=input_names,
                        output_names=["image_embeds", "image_embeds_mask"],
                        verbose=False,
                    )
                    onnx_model = onnx.load(tmp_onnx_file, load_external_data=True)
                    from xhquant.utils.onnx_simplify import onnx_simplify

                    onnx_model_simp, check = onnx_simplify(onnx_model)
                    if check:
                        onnx_model = onnx_model_simp
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

            onnx_model = onnx.load(onnx_file)
            return to_frontend_graph(onnx_model, FrontendType.ONNX, list(dummy_values))
        finally:
            if tmp_dir_ctx is not None:
                tmp_dir_ctx.cleanup()

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {
            "input_names": ["pixel_values", "image_position_ids"],
            "output_names": ["image_embeds", "image_embeds_mask"],
        }

    def export_hmonnx(self, output_dir: str) -> VisualModelMeta:
        meta_info = self.create_export_metadata(output_dir)
        exported_hmonnx_file = super()._export_hmonnx(output_dir)
        meta_info.hmonnx = str(exported_hmonnx_file)
        return meta_info

    def create_export_metadata(self, output_dir: str) -> VisualModelMeta:
        meta_info = cast(VisualModelMeta, self.get_export_metadata_cls()())
        meta_info.image_size_w = self.config.max_size_w
        meta_info.image_size_h = self.config.max_size_h
        meta_info.patch_size = self.config.patch_size
        meta_info.image_seq_length = self.config.image_seq_length
        return meta_info
