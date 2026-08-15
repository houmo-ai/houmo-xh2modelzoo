from __future__ import annotations

import copy
import shutil
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn as nn
from accelerate import init_empty_weights
from transformers import AutoModelForCausalLM, GenerationConfig, PreTrainedModel

from xhquant.api import get_xhquant_logger

from ...base_model import get_model_param_buffer_size_gb
from ...builder import register_llm_model
from ...types import VLLMModelMeta
from ...utils import get_cpu_memory_mb
from ..qwen3_5.qwen3_5_llm_model import XHQwen3_5Model
from ..qwen3_next.qwen3_next_model import (
    Qwen3NextModelMeta,
    XHQwen3NextModel,
    build_qwen3_next_hf_compatible_model,
)
from ._compat import load_ling_config, patch_ling_remote_code_compatibility
from .data_preprocess import Ling3FlashDataPreprocess
from .ling_3_flash_hmonnx_inference import XHLing3FlashHMONNXModel
from .xh_ling_3_flash_config import XHLing3FlashModelConfig


try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    no_init_weights = init_empty_weights


class Ling3FlashModelMeta(Qwen3NextModelMeta):
    pass


@register_llm_model("BailingMoeV3ForCausalLM")
class XHLing3FlashModel(XHQwen3NextModel):
    """Merak-native Ling-3-Flash target model.

    The checkpoint ships its HF implementation as trusted remote code, so the
    concrete HF class is only known after loading.  ``PreTrainedModel`` keeps
    the base lifecycle type checks useful without importing a 255-GB model at
    module-import time.
    """

    HF_MODEL_CLS = PreTrainedModel
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    # The source checkpoint advertises BF16, but XH export/runtime defaults to
    # FP16.  An explicit export.model.dtype still overrides this fallback.
    HF_MODEL_DTYPE = torch.float16
    META_CLS = Ling3FlashModelMeta
    HMONNXINFERENCE_CLS = XHLing3FlashHMONNXModel
    CONFIG_CLS = XHLing3FlashModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_qwen3_next_hf_compatible_model)
    transformers_min_version = "5.5.0"
    WORKFLOW_CLS = (
        "xhmodel_merak.xh_llm.models.ling_3_flash.workflow:Ling3FlashWorkflow"
    )

    def __init__(self, config: XHLing3FlashModelConfig):
        super().__init__(config)
        self.config = cast(XHLing3FlashModelConfig, self.config)
        self.visual = None
        self.wrap_cfg["fuse_gdr_ops"] = self.config.fuse_gdr_ops
        self.wrap_cfg["split_conv_cache"] = True

    @classmethod
    def _load_hf_model(cls, hf_model_dir: str, **kwargs):
        patch_ling_remote_code_compatibility()
        config = load_ling_config(hf_model_dir)
        kwargs.setdefault("trust_remote_code", True)
        kwargs["config"] = config
        return super()._load_hf_model(hf_model_dir, **kwargs)

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs):
        # Prime the dynamic config class before BaseModel's AutoConfig probe.
        _ = load_ling_config(hf_model_dir)
        return super().get_hf_model(hf_model_dir, quant_weight=quant_weight, **kwargs)

    @classmethod
    def _load_gptqmodel(cls, hf_model_dir: str, device_map="cpu", **kwargs):
        patch_ling_remote_code_compatibility()
        hf_model = super()._load_gptqmodel(
            hf_model_dir,
            device_map=device_map,
            **kwargs,
        )
        return cls._trim_mtp_layers(hf_model)

    @classmethod
    def _trim_mtp_layers(cls, hf_model: nn.Module) -> nn.Module:
        """Keep only causal decoder layers for export and normal generation.

        Ling stores its extra MTP predictor in the decoder ``ModuleList``
        after ``config.num_hidden_layers``.  Transformers allocates the
        generation cache for the causal layer count, so letting that MTP layer
        run attempts to update an out-of-range cache slot.  The target HMONNX
        graph never exports MTP; applying the same structural contract at the
        GPTQModel load boundary keeps direct checkpoint inference and export
        aligned.
        """

        language_model = getattr(hf_model, "model", None)
        layers = getattr(language_model, "layers", None)
        config = getattr(language_model, "config", None)
        target_layers = getattr(config, "num_hidden_layers", None)
        if layers is None or target_layers is None:
            return hf_model

        target_layers = int(target_layers)
        if len(layers) <= target_layers:
            return hf_model
        language_model.layers = nn.ModuleList(list(layers[:target_layers]))
        if hasattr(language_model, "num_nextn_predict_layers"):
            language_model.num_nextn_predict_layers = 0
        if hasattr(hf_model, "num_nextn_predict_layers"):
            hf_model.num_nextn_predict_layers = 0
        return hf_model

    @classmethod
    def _postprocess_gptqmodel_structure(
        cls,
        native_hf_model: nn.Module,
        **kwargs,
    ) -> nn.Module:
        del cls, kwargs
        # GPTQModel's Ling definition temporarily presents FLA
        # ShortConvolution as a plain Module so its cache-aware forward is not
        # replaced by the generic Conv1d capture wrapper. Restore the original
        # class before Merak/XHQuant converts the dequantized HF model.
        for module in native_hf_model.modules():
            original_cls = getattr(
                type(module),
                "_gptqmodel_ling_original_class",
                None,
            )
            if isinstance(original_cls, type):
                module.__class__ = original_cls
        return native_hf_model

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir, **kwargs):
        patch_ling_remote_code_compatibility()
        include_buffers = bool(kwargs.pop("include_buffers", False))
        config = load_ling_config(hf_model_dir)
        kwargs.setdefault("trust_remote_code", True)
        kwargs.setdefault("dtype", cls.HF_MODEL_DTYPE)
        with no_init_weights(), init_empty_weights(include_buffers=include_buffers):
            hf_model = cls.HF_AUTO_MODEL_CLS.from_config(config, **kwargs)
            if hf_model.can_generate():
                try:
                    hf_model.generation_config = GenerationConfig.from_pretrained(hf_model_dir)
                except OSError:
                    get_xhquant_logger().info(
                        "Ling generation_config.json not found; using the config-derived defaults."
                    )
        return hf_model

    def _get_language_model(self, hf_model: Any) -> Any:
        return hf_model.model

    @staticmethod
    def _layer_type(layer: nn.Module) -> str:
        return "full_attention" if layer.attention_layer_type == "attention" else "linear_attention"

    def _wraped_pre(self, hf_model: PreTrainedModel):
        self._trim_mtp_layers(hf_model)
        language_model = self._get_language_model(hf_model)
        source_pad_token_id = getattr(language_model.config, "pad_token_id", None)
        if source_pad_token_id is None:
            source_pad_token_id = getattr(language_model.config, "eos_token_id", None)
        self.pad_token_id = source_pad_token_id
        self.layer_types = [self._layer_type(layer) for layer in language_model.layers]
        if self.config.only_first_block:
            selected = []
            for layer_type in self.layer_types:
                selected.append(layer_type)
                if layer_type == "full_attention":
                    break
            self.config.max_layers = len(selected)
            self.config.only_first_block = False
        else:
            max_layers = self.config.get_max_decode_layers()
            selected = self.layer_types if max_layers <= 0 else self.layer_types[:max_layers]

        self.full_attention_layer_indices = [
            idx for idx, layer_type in enumerate(selected) if layer_type == "full_attention"
        ]
        self.linear_attention_layer_indices = [
            idx for idx, layer_type in enumerate(selected) if layer_type == "linear_attention"
        ]
        if not self.full_attention_layer_indices or not self.linear_attention_layer_indices:
            raise ValueError("Ling export prefix must contain both KDA and MLA layers")

        count, size_gb = get_model_param_buffer_size_gb(hf_model)
        logger = get_xhquant_logger()
        logger.info(f"Ling pre-wrap parameters: {count} B ({size_gb:.2f} GB)")
        logger.info(f"CPU memory before Ling wrap: {get_cpu_memory_mb()}")
        return hf_model

    def _wraped_post(self, hf_model: PreTrainedModel):
        del hf_model
        language_model = self._get_language_model(self._wrap_model)
        # Base wrapping canonicalizes the executable graph to fp16.  Ling's
        # source checkpoint is BF16, so keep the detached input embedding in
        # the same runtime dtype instead of leaving a BF16/FP16 boundary.
        self.embed_tokens = copy.deepcopy(language_model.get_input_embeddings()).to(self._dtype)
        text_config = language_model.config
        wrapped_pad_token_id = getattr(text_config, "pad_token_id", None)
        if wrapped_pad_token_id is None:
            wrapped_pad_token_id = getattr(text_config, "eos_token_id", None)
        if wrapped_pad_token_id is not None:
            self.pad_token_id = wrapped_pad_token_id
        self.layer_types = [self._layer_type(layer) for layer in language_model.layers]

        full_attn = language_model.layers[self.full_attention_layer_indices[0]].attention
        linear_attn = language_model.layers[self.linear_attention_layer_indices[0]].attention
        key_projection_dim = linear_attn.num_heads * linear_attn.head_k_dim
        value_projection_dim = linear_attn.num_heads * linear_attn.head_dim

        linear_cfg = self.kvcache_config.linear_kv_cache_config
        linear_cfg.conv_dim = 2 * key_projection_dim + value_projection_dim
        linear_cfg.conv_kernel_size = linear_attn.conv_size
        linear_cfg.num_v_heads = linear_attn.num_heads
        linear_cfg.head_k_dim = linear_attn.head_k_dim
        linear_cfg.head_v_dim = linear_attn.head_dim
        linear_cfg.num_layers = len(self.linear_attention_layer_indices)
        linear_cfg.batch_size = self.config.batch_size

        self._kvcache_mixin.split_conv_cache = True
        self._kvcache_mixin._linear_key_dim = key_projection_dim
        self._kvcache_mixin._linear_value_dim = value_projection_dim

        if self.use_cache:
            self.kvcache_config.num_layers = len(self.full_attention_layer_indices)
            cache_heads = 1
            key_dim = full_attn.kv_lora_rank + full_attn.qk_rope_head_dim
            value_dim = full_attn.kv_lora_rank
            # Deliberately separate shapes and cache objects. K SEFP shares
            # exponents over head-dim while V SEFP shares them over tokens.
            # Both explicit and FlashAttention paths cache the same absorbed
            # latent representation.
            self.kvcache_config.kv_cache_shape = [
                [
                    self.config.batch_size,
                    cache_heads,
                    self.config.context_max_length,
                    key_dim,
                ],
                [
                    self.config.batch_size,
                    cache_heads,
                    self.config.context_max_length,
                    value_dim,
                ],
            ]

        count, size_gb = get_model_param_buffer_size_gb(self._wrap_model)
        logger = get_xhquant_logger()
        logger.info(f"Ling wrapped parameters: {count} B ({size_gb:.2f} GB)")
        logger.info(f"CPU memory after Ling wrap: {get_cpu_memory_mb()}")

    def _extra_export_metadata(
        self,
        output_dir: str,
        meta_info: Ling3FlashModelMeta,
    ) -> Ling3FlashModelMeta:
        """Make the exported HF metadata self-contained for trusted code.

        Ling checkpoints resolve ``AutoConfig`` through ``auto_map``.  The
        common exporter copies JSON/tokenizer assets, but the runtime also
        needs the checkpoint-side Python modules named by that map.  Keep the
        small remote-code files beside ``config.json`` so an exported model
        can be loaded without referring back to the source checkpoint.
        """

        meta_info = super()._extra_export_metadata(output_dir, meta_info)
        hf_config_dir = Path(output_dir) / meta_info.hf_config
        for source in sorted(Path(self.config.hf_model).glob("*.py")):
            shutil.copyfile(source, hf_config_dir / source.name)
        source_config = load_ling_config(self.config.hf_model)
        pad_token_id = getattr(source_config, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(source_config, "eos_token_id", None)
        if pad_token_id is None:
            raise ValueError("Ling checkpoint must define pad_token_id or eos_token_id")
        meta_info.pad_token_id = int(pad_token_id)
        return meta_info

    def _sync_split_conv_cache_state(self) -> None:
        self._kvcache_mixin.split_conv_cache = True
        language_model = (
            self._get_language_model(self._wrap_model) if self._wrap_model is not None else None
        )
        if language_model is not None and self.linear_attention_layer_indices:
            linear_attn = language_model.layers[self.linear_attention_layer_indices[0]].attention
            key_projection_dim = linear_attn.num_heads * linear_attn.head_k_dim
            value_projection_dim = linear_attn.num_heads * linear_attn.head_dim
            self._kvcache_mixin._linear_key_dim = key_projection_dim
            self._kvcache_mixin._linear_value_dim = value_projection_dim
        caches = self._kvcache_mixin.past_conv_caches
        if caches and not isinstance(caches[0], (list, tuple)):
            self._kvcache_mixin.clear_other_cache()
            self._kvcache_mixin.prepare_other_cache()

    def init_wrap_model(self, hf_model: PreTrainedModel) -> Any:
        from ._model import register_wrap_modules

        register_wrap_modules(hf_model)
        # Skip Qwen3.5/Qwen3-Next's static-class registration hooks; Ling's
        # checkpoint-side classes have just been registered dynamically.
        return super(XHQwen3_5Model, self).init_wrap_model(hf_model)

    def _get_data_preprocessor(self) -> Ling3FlashDataPreprocess:
        return Ling3FlashDataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            enable_page_attention=self._kvcache_mixin.enable_page_attention,
            pad_token_id=self.pad_token_id,
        )

    def _get_big_language_placeholder_export_components(self):
        from ._ling_3_flash_big_export import Ling3FlashBigHFModel

        return Ling3FlashBigHFModel, Ling3FlashBigHFModel.PLACEHOLDER_TYPES

    def _check_big_language_placeholder_export_supported(self, empty_hf_model: Any) -> None:
        language_model = self._get_language_model(empty_hf_model)
        found = {type(module).__name__ for module in language_model.modules()}
        _, placeholder_types = self._get_big_language_placeholder_export_components()
        if not any(name in found for name in placeholder_types):
            raise NotImplementedError(
                "Ling low-memory export could not find a sparse-MoE placeholder boundary; "
                f"expected one of {placeholder_types}, found {sorted(found)}"
            )

    def _export_big_language_hmonnx(self, exported_info):
        patch_ling_remote_code_compatibility()
        return super()._export_big_language_hmonnx(exported_info)

    def export_hmonnx(self, output_dir: str) -> VLLMModelMeta:
        # Qwen3-Next's override predates the low-memory language exporter.
        # Ling has no separate visual/MTP graph, so the Qwen3.5 implementation
        # provides both the regular and HUGE_MODEL_EXPORT_ENABLED placeholder paths.
        return XHQwen3_5Model.export_hmonnx(self, output_dir)


__all__ = ["Ling3FlashModelMeta", "XHLing3FlashModel"]
