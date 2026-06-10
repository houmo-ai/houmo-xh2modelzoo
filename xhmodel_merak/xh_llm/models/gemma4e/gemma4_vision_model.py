import shutil
from pathlib import Path
from typing import Any, cast

from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
from xhquant.api import to_frontend_graph

from ...base_vision_model import BaseVisionModel
from ...builder import register_llm_model
from ...llm_data_processor import BaseVisualProcessor
from ...types import VisualModelMeta
from ..gemma4.gemma4_visual_model import Gemma4VisualAdapter as _CompactGemma4VisualAdapter
from ..gemma4.gemma4_visual_model import _make_vision_attn_traceable, _replace_rmsnorm
from .gemma4_processor import XHGemma4Processor, configure_gemma4_visual_processor
from .xh_gemma4_config import XHGemma4VisualConfig


class _Gemma4CompactVisualProcessor(BaseVisualProcessor):
    def forward(self, data: dict) -> tuple[torch.Tensor, ...]:
        return (data["image"],)


class _Gemma4CompactVisualAdapter(_CompactGemma4VisualAdapter):
    def forward(self, pixel_values: torch.Tensor):
        return super().forward(pixel_values)


class _Gemma4OfflineFullVisualAdapter(_CompactGemma4VisualAdapter):
    """Full-mode wrapper that reuses the compact offline path with pooling.

    All position-dependent computations (RoPE cos/sin, positional embeddings,
    pooler weights) are pre-baked as buffers. Only ``pixel_values`` is needed
    as input — matching the gemma4-26b-a4b visual export graph.
    """

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

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        kwargs.setdefault("dtype", torch.bfloat16)
        kwargs.setdefault("device_map", "cpu")
        kwargs.setdefault("trust_remote_code", True)
        config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
        quantization_config = getattr(config, "quantization_config", None)
        quant_method = getattr(quantization_config, "quant_method", None)
        if isinstance(quantization_config, dict):
            quant_method = quantization_config.get("quant_method", quant_method)
        if str(quant_method).lower() == "gptq":
            assert quant_weight is None or len(quant_weight) == 0, (
                "Model is already quantized, quant_weight should be None or empty when loading quantized model."
            )
            native_hf_model = cls._load_hf_model(hf_model_dir, **kwargs)
            # Visual export only consumes vision_tower/embed_vision. Dense Gemma4
            # AutoRound/GPTQ checkpoints quantize the text stack, while visual
            # weights remain regular tensors and should be quantized by this
            # visual submodel's own quant_scheme.
            if hasattr(native_hf_model.config, "quantization_config"):
                native_hf_model.config.quantization_config = None
            return native_hf_model
        return super().get_hf_model(hf_model_dir, quant_weight, **kwargs)

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
        adapter_kwargs, hf_model = self._build_offline_adapter_kwargs(hf_model)
        compact_visual = _Gemma4CompactVisualAdapter(
            hf_model.model.vision_tower,
            hf_model.model.embed_vision,
            **adapter_kwargs,
        )
        _replace_rmsnorm(compact_visual)
        for layer in compact_visual.vision_tower.encoder.layers:
            _make_vision_attn_traceable(layer.self_attn)
        return super().init_wrap_model(compact_visual)

    def _init_offline_full_wrap_model(self, hf_model: Gemma4ForConditionalGeneration | None) -> Any:
        if hf_model is None:
            hf_model = self.get_native_model()
        adapter_kwargs, hf_model = self._build_offline_adapter_kwargs(hf_model)
        full_visual = _Gemma4OfflineFullVisualAdapter(
            hf_model.model.vision_tower,
            hf_model.model.embed_vision,
            **adapter_kwargs,
        )
        _replace_rmsnorm(full_visual)
        for layer in full_visual.vision_tower.encoder.layers:
            _make_vision_attn_traceable(layer.self_attn)
        return super().init_wrap_model(full_visual)

    def _build_offline_adapter_kwargs(
        self, hf_model: Gemma4ForConditionalGeneration
    ) -> tuple[dict[str, Any], Gemma4ForConditionalGeneration]:
        gemma4_config = AutoConfig.from_pretrained(self.hf_model_dir, trust_remote_code=True)
        if self.config.model_type is None:
            self.config.model_type = "Gemma4ForConditionalGeneration_visual"
        if hasattr(gemma4_config, "vision_config") and gemma4_config.vision_config is not None:
            vision_config = gemma4_config.vision_config
            patch_size = vision_config.get("patch_size") if isinstance(vision_config, dict) else vision_config.patch_size
            assert patch_size == self.config.patch_size

        dummy_inputs = self._get_export_inputs()
        image_position_ids = dummy_inputs["image_position_ids"]
        pooling_kernel_size = self.config.pooling_kernel_size
        pooling_kernel_area = pooling_kernel_size * pooling_kernel_size
        num_patches = image_position_ids.shape[1]
        output_length = num_patches // pooling_kernel_area

        max_x = image_position_ids[..., 0].max(dim=-1, keepdim=True)[0] + 1
        kernel_idxs = torch.div(image_position_ids, pooling_kernel_size, rounding_mode="floor")
        kernel_idxs = kernel_idxs[..., 0] + (max_x // pooling_kernel_size) * kernel_idxs[..., 1]
        pooler_weights = F.one_hot(kernel_idxs.long(), output_length).float() / pooling_kernel_area
        pooler_weights = pooler_weights.transpose(1, 2)

        vision_tower = hf_model.model.vision_tower
        with torch.no_grad():
            image_position_ids_cpu = image_position_ids.cpu()
            rope_cfg = vision_tower.config
            head_dim = getattr(rope_cfg, "head_dim", None) or rope_cfg.hidden_size // rope_cfg.num_attention_heads
            spatial_dim = head_dim // 2
            rope_theta = rope_cfg.rope_parameters["rope_theta"]
            inv_freq = 1.0 / (rope_theta ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim))
            inv_freq_expanded = inv_freq[None, :, None]
            all_cos, all_sin = [], []
            for dim_idx in range(2):
                dim_pos = image_position_ids_cpu[:, :, dim_idx].float()
                freqs = (inv_freq_expanded @ dim_pos[:, None, :]).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                all_cos.append(emb.cos())
                all_sin.append(emb.sin())
            rope_cos = torch.cat(all_cos, dim=-1).to(dtype=torch.bfloat16)
            rope_sin = torch.cat(all_sin, dim=-1).to(dtype=torch.bfloat16)

            no_padding = torch.zeros(1, num_patches, dtype=torch.bool)
            pos_embed = vision_tower.patch_embedder._position_embeddings(
                image_position_ids_cpu, no_padding
            ).to(dtype=torch.bfloat16)

        kwargs = dict(
            pooler_weights=pooler_weights,
            num_image_tokens=output_length,
            rope_cos=rope_cos,
            rope_sin=rope_sin,
            pos_embed=pos_embed,
            position_ids=image_position_ids,
        )
        return kwargs, hf_model

    def init_wrap_model(self, hf_model: Any = None) -> Any:
        if self._use_compact_export():
            return self._init_compact_wrap_model(hf_model)
        return self._init_offline_full_wrap_model(hf_model)

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
        return _Gemma4CompactVisualProcessor()

    def get_dummy_inputs(self) -> Any:
        if self._use_compact_export():
            compact_inputs = self._get_compact_export_inputs()
            return {"image": compact_inputs["pixel_values"]}
        export_inputs = self._get_export_inputs()
        return {"image": export_inputs["pixel_values"]}

    def _to_fronted(self, wrap_model):
        dummy_inputs = self.get_dummy_inputs()
        pixel_values = dummy_inputs["image"].float().cpu()
        if self.work_dir:
            self._export_plain_onnx_sidecar(self.work_dir)
        return to_frontend_graph(wrap_model.float().cpu(), "TorchFX", [pixel_values])

    def get_export_cfg(self) -> dict[str, list[str]]:
        return {
            "input_names": ["pixel_values"],
            "output_names": ["image_embeds"],
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
        dummy_values = (dummy_inputs["image"].float().cpu(),)

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
