# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, NoReturn

import onnx
import torch
from transformers import AutoModelForImageTextToText
from transformers.models.hunyuan_vl.modeling_hunyuan_vl import (
    HunYuanVLForConditionalGeneration,
)

from xhquant.api import FrontendType, get_xhquant_logger, to_frontend_graph
from xhquant.frontend.utils import module_graph_forward_guard

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...onnx_lazy_load import lazy_load_onnx
from ...types import LLMModelState, VisualModelMeta
from . import _visual_model_impl as _hunyuan_ocr_visual_impl  # noqa: F401
from .xh_hunyuan_ocr_config import (
    HUNYUAN_OCR_VISUAL_MODEL_TYPE,
    XHHunYuanOCRVisualConfig,
)


@register_llm_model("HunYuanVLForConditionalGeneration_visual", master=False)
class XHHunYuanOCRVisualModel(BaseVisionModel):
    """Registration boundary for the vision tower owned by the main HF model."""

    transformers_min_version = "5.13.0"
    transformers_max_version = "5.13.0"
    HF_MODEL_CLS = HunYuanVLForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    HF_MODEL_DTYPE = torch.bfloat16
    BUILD_HF_COMPATIBLE_FUNC = None
    HMONNXINFERENCE_CLS = None
    META_CLS = VisualModelMeta
    CONFIG_CLS = XHHunYuanOCRVisualConfig

    def __init__(self, config: XHHunYuanOCRVisualConfig) -> None:
        super().__init__(config)
        if self.config.model_type is None:
            self.config.model_type = HUNYUAN_OCR_VISUAL_MODEL_TYPE

    @staticmethod
    def _raise_checkpoint_owner_error() -> NoReturn:
        raise RuntimeError(
            "XHHunYuanOCRVisualModel does not load a native model; "
            "load it through XHHunYuanOCRModel and call select_native_vision_tower()."
        )

    def get_native_model(self) -> Any:
        self._raise_checkpoint_owner_error()

    def get_empty_native_model(self) -> Any:
        self._raise_checkpoint_owner_error()

    def get_compatible_native_model(self) -> Any:
        self._raise_checkpoint_owner_error()

    def init_wrap_model(self, hf_model: HunYuanVLForConditionalGeneration | None = None) -> Any:
        if hf_model is None:
            self._raise_checkpoint_owner_error()
        return super().init_wrap_model(self.select_native_vision_tower(hf_model))

    def _to_fronted(self, wrap_model):
        dummy_input = self.get_dummy_inputs()["image"].float().cpu()
        work_dir = self.config.work_dir
        temporary_work_dir = None
        if not work_dir:
            temporary_work_dir = tempfile.TemporaryDirectory()
            work_dir = temporary_work_dir.name
        try:
            onnx_file = Path(work_dir) / "onnx" / "hunyuan_ocr_visual.onnx"
            onnx_file.parent.mkdir(parents=True, exist_ok=True)
            if not onnx_file.exists():
                with tempfile.TemporaryDirectory() as temporary_export_dir:
                    temporary_onnx = Path(temporary_export_dir) / onnx_file.name
                    with module_graph_forward_guard(wrap_model):
                        torch.onnx.export(
                            wrap_model.float().cpu(),
                            (dummy_input,),
                            temporary_onnx,
                            export_params=True,
                            opset_version=18,
                            do_constant_folding=True,
                            input_names=["pixel_values"],
                            output_names=["image_embeds"],
                            verbose=False,
                            dynamo=False,
                        )
                    onnx_model = onnx.load(str(temporary_onnx), load_external_data=True)
                    if not onnx_model.ir_version or onnx_model.ir_version > 10:
                        onnx_model.ir_version = 10
                    onnx.save(
                        onnx_model,
                        str(onnx_file),
                        save_as_external_data=True,
                        all_tensors_to_one_file=True,
                        location=f"{onnx_file.stem}_external_data",
                    )
            else:
                get_xhquant_logger().info(f"from cached onnx: {onnx_file}")

            lazy_model = lazy_load_onnx(str(onnx_file))
            lazy_model.load_all_tensors()
            onnx_model = lazy_model.model
            lazy_model.close()
            return to_frontend_graph(onnx_model, FrontendType.ONNX, [dummy_input])
        finally:
            if temporary_work_dir is not None:
                temporary_work_dir.cleanup()

    def get_dummy_inputs(self) -> dict[str, torch.Tensor]:
        patch_dim = 3 * self.config.patch_size * self.config.patch_size
        return {
            "image": torch.zeros(
                self.config.num_patches,
                patch_dim,
                dtype=torch.float32,
            )
        }

    def validate_single_bucket_inputs(
        self,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> None:
        """Reject inputs outside the fixed visual bucket before vision inference."""

        expected_grid = (1, self.config.grid_h, self.config.grid_w)
        grid = torch.as_tensor(image_grid_thw)
        if grid.shape != (1, 3) or tuple(int(value) for value in grid[0].tolist()) != expected_grid:
            raise ValueError(
                "HunyuanOCR fixed visual export supports exactly one image grid "
                f"{expected_grid}, got shape={tuple(grid.shape)} values={grid.tolist()}"
            )

        patch_dim = 3 * self.config.patch_size * self.config.patch_size
        expected_pixels = (self.config.num_patches, patch_dim)
        if tuple(pixel_values.shape) != expected_pixels:
            raise ValueError(
                "HunyuanOCR fixed visual bucket requires pixel_values shape "
                f"{expected_pixels}, got {tuple(pixel_values.shape)}"
            )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if self._state == LLMModelState.WRAP:
            with module_graph_forward_guard(self._inference_model):
                return self._inference_model(*args, **kwargs)
        return self._inference_model(*args, **kwargs)

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {
            "input_names": ["pixel_values"],
            "output_names": ["image_embeds"],
        }

    def create_export_metadata(self, output_dir: str):
        del output_dir
        from .hunyuan_ocr_llm_model import HunyuanOCRVisualMeta

        return HunyuanOCRVisualMeta(
            bucket_id=f"bucket_{self.config.image_size_w}x{self.config.image_size_h}",
            image_size_w=int(self.config.image_size_w),
            image_size_h=int(self.config.image_size_h),
            input_shape=[int(self.config.num_patches), 3 * int(self.config.patch_size) ** 2],
            output_shape=[1, int(self.config.image_token_count), int(self.config.text_hidden_size)],
            image_grid_thw=[1, int(self.config.grid_h), int(self.config.grid_w)],
            patch_size=int(self.config.patch_size),
            spatial_merge_size=int(self.config.spatial_merge_size),
            num_patches=int(self.config.num_patches),
            image_token_count=int(self.config.image_token_count),
            hidden_size=int(self.config.text_hidden_size),
        )

    def export_hmonnx(self, output_dir: str):
        metadata = self.create_export_metadata(output_dir)
        metadata.hmonnx = str(super()._export_hmonnx(output_dir))
        graph_path = Path(metadata.hmonnx)
        metadata.external_data = str(
            graph_path.with_name(f"{graph_path.stem.removesuffix('_with_act')}_external_data")
        )
        return metadata

    @staticmethod
    def select_native_vision_tower(hf_model: HunYuanVLForConditionalGeneration) -> Any:
        """Return the shared vision tower without loading another checkpoint."""

        return hf_model.model.vision_tower
