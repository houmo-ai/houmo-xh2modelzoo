"""
XHQwen3_5VisionModel — Vision model wrapper for Qwen3.5 xhquant export.

Follows the same pattern as xhquant_llm/models/qwen3_vl/qwen3_vl_vision_model.py,
but adapted for Qwen3.5 (no deepstack features).
"""

import copy
import gc
import json
import types
from pathlib import Path
from typing import Any, cast

import torch
from accelerate import init_empty_weights
from safetensors import safe_open
from transformers import AutoConfig, AutoModelForImageTextToText

from xhquant.api import FrontendType, get_xhquant_logger, to_frontend_graph
from xhquant.utils.registry import DynamicModule, _DMRegistryCls

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...llm_data_processor import BaseVisualProcessor
from ...types import VisualModelMeta
from ...utils import unfold_args
from .modeling_qwen3_5 import Qwen3_5ForConditionalGeneration as XHQwen3_5ForConditionalGeneration
from .modeling_qwen3_5 import Qwen3_5VisionModel as HFQwen3_5VisionModel
from .modeling_qwen3_5_patch import qwen3_5_patch
from .qwen3_5_processor import XHQwen3_5Processor
from .visual_token_gears import (
    VISUAL_ATTENTION_MASK_FORMAT,
    VISUAL_ATTENTION_MASK_OPERATOR,
    VISUAL_ATTENTION_MASK_SHAPE,
    VISUAL_INPUT_PATCHES,
    VISUAL_ROTARY_POSITION_FORMAT,
    build_visual_gear_manifest,
    build_visual_token_gear_inputs,
    factor_image_token_grid,
    patch_token_capacity,
)
from .xh_qwen3_5_config import XHQwen3_5_VisualConfig


class _Qwen3_5_VisualHFCompatible(DynamicModule):  # noqa: N801
    def _setup(self: HFQwen3_5VisionModel, xh_visual_model: "XHQwen3_5VisionModel"):
        dtype = self.dtype
        device = self.device

        del self.blocks
        del self.pos_embed
        del self.patch_embed
        del self.merger
        del self.rotary_pos_emb

        # 没有这行，self.dtype时会报错，因为所有参数都被移除了。
        self._dummy_param = torch.nn.Parameter(torch.empty(0, dtype=dtype, device=device))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.visual = xh_visual_model

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.visual.forward(hidden_states, *args, **kwargs)


class _Qwen3_5VisualTokenInputProcessor(BaseVisualProcessor):  # noqa: N801
    INPUT_NAMES = (
        "pixel_values",
        "position_ids",
        "position_weights",
        "rotary_position_ids",
        "attention_mask",
    )

    def forward(self, data: dict) -> tuple[torch.Tensor, ...]:
        missing = [name for name in self.INPUT_NAMES if name not in data]
        if missing:
            raise ValueError(f"missing Qwen3.5 visual token inputs: {missing}")
        return tuple(data[name] for name in self.INPUT_NAMES)


def build_qwen3_5_visual_hf_compatible_model(
    hf_model: XHQwen3_5ForConditionalGeneration, xh_visual_model: "XHQwen3_5VisionModel"
):
    # 构建一个兼容HF的视觉模型，主要用于导出ONNX
    LLM_COMPATIBLE_MODULES = _DMRegistryCls("XHCompatible")  # noqa: N806
    hf_model_cls = type(hf_model.model.visual)
    if hf_model_cls not in LLM_COMPATIBLE_MODULES:
        LLM_COMPATIBLE_MODULES.register_module(
            {
                hf_model_cls: hf_model_cls.__name__,
            },
            _Qwen3_5_VisualHFCompatible,
        )
    LLM_COMPATIBLE_MODULES.convert(hf_model.model.visual, xh_visual_model=xh_visual_model)
    from .modeling_qwen3_5_patch import get_image_features

    hf_model.model.get_image_features = types.MethodType(get_image_features, hf_model.model)

    return hf_model


@register_llm_model("Qwen3_5ForConditionalGeneration_visual", master=False)
class XHQwen3_5VisionModel(BaseVisionModel):  # noqa: N801
    transformers_min_version = "5.2.0"
    HF_MODEL_CLS = XHQwen3_5ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    VISUAL_HF_MODEL_CLS = HFQwen3_5VisionModel
    VISUAL_WEIGHT_PREFIX = "model.visual."
    # HMONNXINFERENCE_CLS = Qwen3HMONNXModel
    BUILD_HF_COMPATIBLE_FUNC = build_qwen3_5_visual_hf_compatible_model

    META_CLS = VisualModelMeta
    CONFIG_CLS = XHQwen3_5_VisualConfig
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.qwen3_5.workflow:Qwen35Workflow"

    def __init__(self, config: XHQwen3_5_VisualConfig):
        super().__init__(config)
        qwen3_5_config = AutoConfig.from_pretrained(self.hf_model_dir)
        assert qwen3_5_config.vision_config.patch_size == config.patch_size
        assert qwen3_5_config.vision_config.temporal_patch_size == config.temporal_patch_size
        assert qwen3_5_config.vision_config.spatial_merge_size == config.spatial_merge_size
        self._vision_config = qwen3_5_config.vision_config
        self.config = cast(XHQwen3_5_VisualConfig, self.config)
        if self.config.model_type is None:
            self.config.model_type = "Qwen3_5ForConditionalGeneration_visual"

    def _to_eager(self, aligned: bool = True):
        raise NotImplementedError("Eager mode is not implemented for vision model yet.")

    def _to_fronted(self, wrap_model):
        logger = get_xhquant_logger()
        dummy_inputs = unfold_args(self.get_data_preprocessor()(self.get_dummy_inputs()))
        if not isinstance(dummy_inputs, (tuple, list)):
            dummy_inputs = [dummy_inputs]
        dummy_inputs = [
            value.float().cpu() if torch.is_floating_point(value) else value.cpu() for value in dummy_inputs
        ]
        # Keep xhquant leaf modules (especially MaskedAdd) intact until
        # quantization. An intermediate eager ONNX export expands MaskedAdd
        # into Add + Add + Clip and loses the QMaskedAdd lowering contract.
        logger.info("Converting patch-token visual model through TorchFX frontend")
        return to_frontend_graph(
            wrap_model.float().cpu(),
            FrontendType.TorchFX,
            dummy_inputs,
        )

    def get_dummy_inputs(self) -> Any:
        dummy = self._get_dummy_inputs()
        return {name: dummy[name] for name in _Qwen3_5VisualTokenInputProcessor.INPUT_NAMES}

    def _get_dummy_inputs(self) -> Any:
        image_capacity = int(self.config.image_token_capacity)
        capacity = patch_token_capacity(image_capacity, self.config.spatial_merge_size)
        grid_thw = factor_image_token_grid(image_capacity, self.config.spatial_merge_size)
        patch_dim = (
            self._vision_config.in_channels
            * self.config.temporal_patch_size
            * self.config.patch_size
            * self.config.patch_size
        )
        dummy = {
            "pixel_values": torch.zeros((1, capacity, patch_dim), dtype=torch.float32),
            "image_grid_thw": grid_thw,
        }
        dummy.update(
            build_visual_token_gear_inputs(
                grid_thw,
                patch_capacity=capacity,
                num_position_embeddings=self._vision_config.num_position_embeddings,
                spatial_merge_size=self.config.spatial_merge_size,
                dtype=torch.float32,
                rotary_cache_length=self.config.visual_rope_cache_length,
            )
        )
        return dummy

    def init_wrap_model(self, hf_model: XHQwen3_5ForConditionalGeneration = None):
        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls

        vision_register_wrap_cls(hf_model)
        if isinstance(hf_model, self.VISUAL_HF_MODEL_CLS):
            visual = hf_model
        else:
            visual = hf_model.model.visual
        # self.config = visual.config
        wraped_model = super().init_wrap_model(visual)
        return wraped_model

    @classmethod
    def _load_visual_state_dict(cls, hf_model_dir: str) -> dict[str, torch.Tensor]:
        """Load only ``model.visual.*`` tensors from a sharded HF checkpoint."""
        model_dir = Path(hf_model_dir)
        index_file = model_dir / "model.safetensors.index.json"
        if index_file.is_file():
            weight_map = json.loads(index_file.read_text(encoding="utf-8"))["weight_map"]
            weight_files = sorted(
                {
                    model_dir / filename
                    for name, filename in weight_map.items()
                    if name.startswith(cls.VISUAL_WEIGHT_PREFIX)
                }
            )
        else:
            weight_files = sorted(model_dir.glob("*.safetensors"))

        state_dict: dict[str, torch.Tensor] = {}
        for weight_file in weight_files:
            with safe_open(weight_file, framework="pt", device="cpu") as checkpoint:
                for full_name in checkpoint.keys():
                    if not full_name.startswith(cls.VISUAL_WEIGHT_PREFIX):
                        continue
                    name = full_name.removeprefix(cls.VISUAL_WEIGHT_PREFIX)
                    state_dict[name] = checkpoint.get_tensor(full_name)
        if not state_dict:
            raise RuntimeError(f"No visual weights with prefix {cls.VISUAL_WEIGHT_PREFIX!r} found in {hf_model_dir}")
        return state_dict

    def get_native_model(self):
        """Materialize only the visual tower instead of the complete VLM.

        Qwen3.5 AutoRound checkpoints keep the visual weights in BF16 while the
        language model is packed INT4.  Loading the root conditional-generation
        model would immediately dequantize the whole language model even though
        visual export only needs ``model.visual.*``.
        """
        hf_config = AutoConfig.from_pretrained(self.hf_model_dir)
        with init_empty_weights():
            visual = self.VISUAL_HF_MODEL_CLS._from_config(hf_config.vision_config)

        state_dict = self._load_visual_state_dict(self.hf_model_dir)
        visual.load_state_dict(state_dict, strict=True, assign=True)
        visual.eval()
        get_xhquant_logger().info(
            "Loaded visual-only checkpoint: %d tensors, %.3f GiB",
            len(state_dict),
            sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values()) / 1024**3,
        )
        return visual

    def get_tf_processor(self):
        processor = XHQwen3_5Processor.from_pretrained(self.hf_model_dir)
        processor.config.patch_size = self.config.patch_size
        processor.config.visual_input_mode = VISUAL_INPUT_PATCHES
        largest_gear = max(self.config.image_token_gears)
        processor.config.max_pixels = (
            patch_token_capacity(largest_gear, self.config.spatial_merge_size)
            * self.config.patch_size
            * self.config.patch_size
        )
        return processor

    def _get_data_preprocessor(self):
        return _Qwen3_5VisualTokenInputProcessor()

    def forward(self, *args, **kwargs):
        out = self._inference_model(*args, **kwargs)
        return out

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        native_model = super().get_hf_model(hf_model_dir, quant_weight, **kwargs)
        return qwen3_5_patch(native_model)

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs) -> Any:
        native_hf_model = super().get_empty_hf_model(hf_model_dir)
        native_hf_model = qwen3_5_patch(native_hf_model)
        return native_hf_model

    @classmethod
    def get_compatible_model(cls, hf_model_dir, **kwargs):
        # 单独调试visual模型时，不能加载空模型
        return cls.get_hf_model(hf_model_dir, **kwargs)

    def get_export_cfg(self) -> dict[str, list[str]]:
        return dict(
            input_names=list(_Qwen3_5VisualTokenInputProcessor.INPUT_NAMES),
            output_names=["image_embeds"],
        )

    def export_hmonnx(self, output_dir: str, *, write_standalone_metadata: bool = True) -> VisualModelMeta:
        """Export visual_m* stages directly under the full or visual-only package root."""
        return self._export_token_gear_set(
            output_dir, self.config.image_token_gears, write_standalone_metadata=write_standalone_metadata
        )

    def _export_single_hmonnx(self, output_dir: str) -> VisualModelMeta:
        meta_info = self.create_export_metadata(output_dir)
        exported_hmonnx_file = super()._export_hmonnx(output_dir)
        meta_info.hmonnx = str(exported_hmonnx_file)
        return meta_info

    def _export_token_gear_set(
        self, output_dir: str, gears: list[int], *, write_standalone_metadata: bool = True
    ) -> VisualModelMeta:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        base_model_name = self.config.model_name
        active_capacity = int(self.config.image_token_capacity)
        original_work_dir = self.config.work_dir
        artifacts_by_gear: dict[int, dict] = {}

        def export_one(model: "XHQwen3_5VisionModel", gear: int) -> None:
            gear_dir = output_path / f"visual_m{gear}"
            original_name = model.config.model_name
            model.config.model_name = f"{base_model_name}_m{gear}"
            try:
                gear_meta = model._export_single_hmonnx(str(gear_dir))
            finally:
                model.config.model_name = original_name
            artifacts_by_gear[gear] = {
                "image_token_capacity": gear,
                "patch_token_capacity": patch_token_capacity(gear, self.config.spatial_merge_size),
                "hmonnx": str(Path(gear_meta.hmonnx).relative_to(output_path).as_posix()),
            }

        try:
            export_one(self, active_capacity)
            self.config.work_dir = original_work_dir
            for gear in gears:
                if gear == active_capacity:
                    continue
                gear_config = copy.deepcopy(self.config)
                gear_config.image_token_capacity = int(gear)
                gear_config.image_token_gears = [int(gear)]
                gear_config.model_name = f"{base_model_name}_m{gear}"
                gear_config.work_dir = None
                gear_model = type(self)(gear_config)
                try:
                    export_one(gear_model, int(gear))
                finally:
                    del gear_model
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        finally:
            self.config.work_dir = original_work_dir

        artifacts = [artifacts_by_gear[int(gear)] for gear in gears]
        manifest = build_visual_gear_manifest(
            artifacts,
            spatial_merge_size=self.config.spatial_merge_size,
            visual_rope_cache_length=self.config.visual_rope_cache_length,
        )
        manifest_path = output_path / "visual_gears.json"
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        meta_info = self.create_export_metadata(output_dir)
        largest_artifact = artifacts[-1]
        meta_info.hmonnx = str(output_path / largest_artifact["hmonnx"])
        meta_info.image_token_capacity = largest_artifact["image_token_capacity"]
        meta_info.patch_token_capacity = largest_artifact["patch_token_capacity"]
        meta_info.image_token_gears = [artifact["image_token_capacity"] for artifact in artifacts]
        meta_info.gears = artifacts
        meta_info.gear_manifest = str(manifest_path)
        meta_info.routing_policy = manifest["routing_policy"]
        meta_info.shared_weight_loader = manifest["shared_weight_loader"]

        # A full VLM package uses golden_meta_info.json; a visual-only marker at
        # that root would make workflow/quick-test routing mistake it for visual-only.
        if write_standalone_metadata:
            standalone_meta = meta_info.to_dict()
            standalone_meta["hmonnx"] = largest_artifact["hmonnx"]
            standalone_meta["gear_manifest"] = manifest_path.name
            (output_path / "visual_meta_info.json").write_text(
                json.dumps(standalone_meta, indent=2) + "\n",
                encoding="utf-8",
            )
        return meta_info

    def create_export_metadata(self, output_dir: str) -> VisualModelMeta:
        meta_info = self.get_export_metadata_cls()()
        meta_info = cast(VisualModelMeta, meta_info)
        meta_info.patch_size = self.config.patch_size
        meta_info.max_size_t = self.config.max_size_t
        meta_info.temporal_patch_size = self.config.temporal_patch_size
        meta_info.spatial_merge_size = self.config.spatial_merge_size
        meta_info.visual_input_mode = VISUAL_INPUT_PATCHES
        meta_info.image_token_capacity = self.config.image_token_capacity
        meta_info.patch_token_capacity = patch_token_capacity(
            self.config.image_token_capacity,
            self.config.spatial_merge_size,
        )
        meta_info.input_names = list(_Qwen3_5VisualTokenInputProcessor.INPUT_NAMES)
        meta_info.attention_mask_format = VISUAL_ATTENTION_MASK_FORMAT
        meta_info.attention_mask_shape = VISUAL_ATTENTION_MASK_SHAPE
        meta_info.attention_mask_operator = VISUAL_ATTENTION_MASK_OPERATOR
        meta_info.hidden_size = self._vision_config.hidden_size
        meta_info.num_heads = self._vision_config.num_heads
        meta_info.in_channels = self._vision_config.in_channels
        meta_info.num_position_embeddings = self._vision_config.num_position_embeddings
        meta_info.rotary_position_format = VISUAL_ROTARY_POSITION_FORMAT
        meta_info.visual_rope_cache_length = self.config.visual_rope_cache_length
        meta_info.image_token_gears = list(self.config.image_token_gears)
        return meta_info
