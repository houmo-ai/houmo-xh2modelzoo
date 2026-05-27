import tempfile
from pathlib import Path
import shutil
from typing import Any, cast

import onnx
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
from xhquant.api import FrontendType, get_xhquant_logger, to_frontend_graph

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...llm_data_processor import BaseVisualProcessor
from ...types import VisualModelMeta
from ..gemma4.gemma4_visual_model import Gemma4VisualAdapter as _CompactGemma4VisualAdapter
from ..gemma4.gemma4_visual_model import _replace_rmsnorm
from ..gemma4.gemma4_visual_model import _wrap_vision_modules as _wrap_compact_vision_modules
from .gemma4_processor import XHGemma4Processor, configure_gemma4_visual_processor
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


class _Gemma4CompactVisualProcessor(BaseVisualProcessor):
    def forward(self, data: dict) -> tuple[torch.Tensor, ...]:
        return (data["image"],)


class _Gemma4CompactVisualAdapter(_CompactGemma4VisualAdapter):
    def forward(self, pixel_values: torch.Tensor):
        return super().forward(pixel_values)


@register_llm_model("Gemma4ForConditionalGeneration_visual", master=False)
class XHGemma4VisionModel(BaseVisionModel):
    HF_MODEL_CLS = Gemma4ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = VisualModelMeta
    CONFIG_CLS = XHGemma4VisualConfig

    def __init__(self, config: XHGemma4VisualConfig):
        super().__init__(config)
        self.config = cast(XHGemma4VisualConfig, self.config)
        self.export_mode = self.config.export_mode

    def _use_compact_export(self) -> bool:
        return self.export_mode == "compact"

    def _get_export_inputs(self) -> dict[str, torch.Tensor]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": Image.new("RGB", (self.config.max_size_w, self.config.max_size_h), color="white")},
                    {"type": "text", "text": "Describe the image briefly."},
                ],
            }
        ]
        processor = self.get_tf_processor()
        inputs = processor.apply_chat_template(messages)
        pixel_values = inputs["pixel_values"]
        image_position_ids = inputs["image_position_ids"]
        if image_position_ids.dim() == 2:
            image_position_ids = image_position_ids.unsqueeze(0)
        is_real = ~(image_position_ids == -1).all(dim=-1)
        real_patch_count = int(is_real[0].sum().item())
        return {
            "pixel_values": pixel_values[:, :real_patch_count, :],
            "image_position_ids": image_position_ids[:, :real_patch_count, :],
        }

    def _get_compact_export_inputs(self) -> dict[str, torch.Tensor]:
        return self._get_export_inputs()

    def _init_compact_wrap_model(self, hf_model: Gemma4ForConditionalGeneration | None) -> Any:
        if hf_model is None:
            hf_model = self.get_native_model()

        gemma4_config = AutoConfig.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        if self.config.model_type is None:
            self.config.model_type = "Gemma4ForConditionalGeneration_visual"
        if hasattr(gemma4_config, "vision_config") and gemma4_config.vision_config is not None:
            vision_config = gemma4_config.vision_config
            if isinstance(vision_config, dict):
                patch_size = vision_config.get("patch_size")
            else:
                patch_size = vision_config.patch_size
            assert patch_size == self.config.patch_size

        dummy_inputs = self._get_compact_export_inputs()
        image_position_ids = dummy_inputs["image_position_ids"]
        pooling_kernel_size = self.config.pooling_kernel_size
        pooling_kernel_area = pooling_kernel_size * pooling_kernel_size
        num_patches = image_position_ids.shape[1]
        output_length = num_patches // pooling_kernel_area

        max_x = image_position_ids[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(image_position_ids, pooling_kernel_size, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // pooling_kernel_size) * kernel_idxs[..., 1]
        pooler_weights = F.one_hot(kernel_idxs.long(), output_length).float() / pooling_kernel_area

        vision_tower = hf_model.model.vision_tower
        embed_vision = hf_model.model.embed_vision
        with torch.no_grad():
            image_position_ids_cpu = image_position_ids.cpu()
            rope_cfg = vision_tower.config
            head_dim = getattr(rope_cfg, "head_dim", None) or rope_cfg.hidden_size // rope_cfg.num_attention_heads
            spatial_dim = head_dim // 2
            rope_theta = rope_cfg.rope_parameters["rope_theta"]
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
            inv_freq_expanded = inv_freq[None, :, None]
            all_cos = []
            all_sin = []
            for dim_idx in range(2):
                dim_pos = image_position_ids_cpu[:, :, dim_idx].float()
                freqs = (inv_freq_expanded @ dim_pos[:, None, :]).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                all_cos.append(emb.cos())
                all_sin.append(emb.sin())
            rope_cos = torch.cat(all_cos, dim=-1).to(dtype=torch.bfloat16)
            rope_sin = torch.cat(all_sin, dim=-1).to(dtype=torch.bfloat16)

        patch_embedder = vision_tower.patch_embedder
        with torch.no_grad():
            no_padding = torch.zeros(1, num_patches, dtype=torch.bool)
            # Use _position_embeddings directly to get the *pure* positional contribution.
            # Feeding zero pixel_values through patch_embedder() leaks an
            # ``input_proj(-ones)`` constant via the ``2*(x-0.5)`` normalization, which the
            # adapter then double-counts at runtime — see parent gemma4 model for details.
            pos_embed = patch_embedder._position_embeddings(image_position_ids_cpu, no_padding).to(dtype=torch.bfloat16)

        compact_visual = _Gemma4CompactVisualAdapter(
            vision_tower,
            embed_vision,
            pooler_weights=pooler_weights,
            num_image_tokens=output_length,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            attn_mask_4d=torch.zeros(1, 1, num_patches, num_patches),
            pos_embed=pos_embed,
        )
        _wrap_compact_vision_modules(compact_visual.vision_tower)
        return super().init_wrap_model(compact_visual)

    def init_wrap_model(self, hf_model: Any = None) -> Any:
        if self._use_compact_export():
            return self._init_compact_wrap_model(hf_model)

        from ._vision_model_impl import register_wrap_modules

        register_wrap_modules(hf_model)
        if hf_model is not None and hasattr(hf_model, "model") and hasattr(hf_model.model, "vision_tower"):
            _wrap_compact_vision_modules(hf_model.model.vision_tower)
            hf_model = _Gemma4VisionExportBridge(hf_model)
        return super().init_wrap_model(hf_model)

    def get_tf_processor(self):
        processor = XHGemma4Processor.from_pretrained(self.hf_model_dir)
        return configure_gemma4_visual_processor(
            processor,
            export_mode=self.config.export_mode,
            max_size_w=self.config.max_size_w,
            max_size_h=self.config.max_size_h,
            patch_size=self.config.patch_size,
            image_seq_length=self.config.image_seq_length,
        )

    def _get_data_preprocessor(self) -> BaseVisualProcessor:
        if self._use_compact_export():
            return _Gemma4CompactVisualProcessor()
        return _Gemma4VisualProcessor()

    def get_dummy_inputs(self) -> Any:
        if self._use_compact_export():
            compact_inputs = self._get_compact_export_inputs()
            return {"image": compact_inputs["pixel_values"]}
        return self._get_export_inputs()

    def _to_fronted(self, wrap_model):
        if self._use_compact_export():
            dummy_inputs = self.get_dummy_inputs()
            pixel_values = dummy_inputs["image"].float().cpu()
            if self.work_dir:
                self._export_plain_onnx_sidecar(self.work_dir)
            return to_frontend_graph(wrap_model.float().cpu(), "TorchFX", [pixel_values])

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
        if self._use_compact_export():
            return {
                "input_names": ["pixel_values"],
                "output_names": ["image_embeds"],
            }
        return {
            "input_names": ["pixel_values", "image_position_ids"],
            "output_names": ["image_embeds", "image_embeds_mask"],
        }

    def _export_plain_onnx_sidecar(self, output_dir: str, cached_onnx_path: Path | None = None) -> Path:
        output_dir_path = Path(output_dir)
        onnx_dir = output_dir_path / "onnx"
        onnx_dir.mkdir(parents=True, exist_ok=True)
        onnx_path = onnx_dir / "gemma4_visual.onnx"
        if onnx_path.exists():
            return onnx_path

        if cached_onnx_path is None:
            cached_onnx_path = Path(self.work_dir) / "onnx" / "gemma4_visual.onnx" if self.work_dir else None
        if cached_onnx_path is not None and cached_onnx_path.exists():
            shutil.copy2(cached_onnx_path, onnx_path)
            for sidecar in cached_onnx_path.parent.glob(f"{cached_onnx_path.stem}*"):
                if sidecar == cached_onnx_path:
                    continue
                shutil.copy2(sidecar, onnx_dir / sidecar.name)
            return onnx_path

        if self._wrap_model is None:
            self.to_wrap()

        wrap_model = self._wrap_model.float().cpu().eval()
        dummy_inputs = self.get_dummy_inputs()
        export_cfg = self.get_export_cfg()
        if self._use_compact_export():
            dummy_values = (dummy_inputs["image"].float().cpu(),)
        else:
            dummy_values = tuple(
                value.float().cpu() if value.is_floating_point() else value.cpu() for value in dummy_inputs.values()
            )

        torch.onnx.export(
            wrap_model,
            dummy_values,
            str(onnx_path),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=export_cfg["input_names"],
            output_names=export_cfg["output_names"],
            verbose=False,
        )
        self._wrap_model.to(self.device, self.dtype)
        return onnx_path

    def export_hmonnx(self, output_dir: str) -> VisualModelMeta:
        meta_info = self.create_export_metadata(output_dir)
        cached_onnx_path = Path(self.work_dir) / "onnx" / "gemma4_visual.onnx" if self.work_dir else None
        exported_hmonnx_file = super()._export_hmonnx(output_dir)
        meta_info.hmonnx = str(exported_hmonnx_file)
        meta_info.onnx = str(self._export_plain_onnx_sidecar(output_dir, cached_onnx_path))
        return meta_info

    def create_export_metadata(self, output_dir: str) -> VisualModelMeta:
        del output_dir
        meta_info = cast(VisualModelMeta, self.get_export_metadata_cls()())
        meta_info.image_size_w = self.config.max_size_w
        meta_info.image_size_h = self.config.max_size_h
        meta_info.patch_size = self.config.patch_size
        meta_info.image_seq_length = self.config.image_seq_length
        meta_info.export_mode = self.config.export_mode
        if not self._use_compact_export():
            vision_config = AutoConfig.from_pretrained(self.hf_model_dir, trust_remote_code=True).vision_config
            meta_info.output_scale = float(getattr(vision_config, "hidden_size", 768) ** -0.25)
        if self._use_compact_export():
            meta_info.num_image_tokens = self.config.image_seq_length
        return meta_info
