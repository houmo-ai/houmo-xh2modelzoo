"""Merak model lifecycle integration for DeepSeek-V4 Flash."""

from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
from typing import Any, Optional, Union, cast

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4ForCausalLM

from xhmodel_merak.utils import calculate_file_md5
from xhquant.api import get_xhquant_logger
from xhquant.utils.registry import _DMRegistryCls

from ...builder import register_llm_model
from ...hmonnx.deterministic_export import canonicalize_hmonnx_artifact
from ...text_llm_hf_compatible import TextLLMHFCompatible
from ...text_llm_model import TextLLMModel
from ...types import LLMModelMeta, LLMModelState, ModelSwitcher
from ...utils import is_huge_model_export_enabled
from .cache_abi import CSA, HCA, SLIDING, DeepSeekV4CacheABI, default_layer_types
from .deepseek_v4_hmonnx_inference import XHDeepSeekV4HMONNXModel
from .full_model import StaticDeepSeekV4ForCausalLM
from .runtime import DeepSeekV4CacheMixin, DeepSeekV4DataPreprocess
from .static_cache import DeepSeekV4StaticCacheSpec
from .xh_deepseek_v4_config import (
    XHDeepSeekV4ModelConfig,
    resolve_deepseek_v4_pad_token_id,
)


class _DeepSeekV4HFCompatible(TextLLMHFCompatible):  # noqa: N801
    """Generation adapter that preserves token ids for hash-MoE routing."""

    def _setup(self, text_llm_model: "XHDeepSeekV4Model"):
        model = super()._setup(text_llm_model)
        if model is not None:
            if hasattr(model, "model"):
                del model.model
            if hasattr(model, "lm_head"):
                del model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return model

    def set_experts_implementation(self, *args, **kwargs):
        del args, kwargs
        return self

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del attention_mask, position_ids, labels, output_attentions
        del output_hidden_states, cache_position, logits_to_keep, kwargs
        if input_ids is None:
            raise ValueError("DeepSeek-V4 inference requires input_ids for hash-MoE routing")
        if inputs_embeds is not None:
            raise ValueError("pass input_ids only; the static runtime owns token embedding")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("DeepSeek-V4 inference currently supports input_ids with batch size 1")
        if use_cache is not False and past_key_values is None:
            past_key_values = DynamicCache()

        seq_length = int(input_ids.shape[1])
        graph_length = int(self._llm_model.get_input_sequence_length())
        steps = (seq_length + graph_length - 1) // graph_length
        outputs_logits = []
        for step in range(steps):
            start = step * graph_length
            end = min(start + graph_length, seq_length)
            data_input = self._llm_model.get_data_preprocessor()(
                {
                    "input_ids": input_ids[:, start:end],
                    "past_seq_length": self._past_seq_length + start,
                }
            )
            result = self._llm_model.forward(*data_input)
            outputs_logits.append(result if isinstance(result, torch.Tensor) else result[0])
        return CausalLMOutputWithPast(
            logits=outputs_logits[-1],
            past_key_values=past_key_values,
        )


def build_deepseek_v4_hf_compatible_model(
    hf_model: DeepseekV4ForCausalLM,
    xh_model: "XHDeepSeekV4Model",
):
    compatible_modules = _DMRegistryCls("XHCompatible")
    hf_model_cls = type(hf_model)
    if hf_model_cls not in compatible_modules:
        compatible_modules.register_module(
            {hf_model_cls: hf_model_cls.__name__},
            _DeepSeekV4HFCompatible,
        )
    return compatible_modules.convert(hf_model, text_llm_model=xh_model)


@register_llm_model("DeepseekV4ForCausalLM")
class XHDeepSeekV4Model(TextLLMModel):
    """Fixed prefill-256/decode-1 DeepSeek-V4 Flash exporter."""

    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.deepseek_v4.workflow:DeepSeekV4Workflow"
    transformers_min_version = "5.13.0"
    HF_MODEL_CLS = DeepseekV4ForCausalLM
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    HF_MODEL_DTYPE = torch.bfloat16
    HMONNXINFERENCE_CLS = XHDeepSeekV4HMONNXModel
    CONFIG_CLS = XHDeepSeekV4ModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_deepseek_v4_hf_compatible_model)
    _GPTQMODEL_METHODS = {"gptq", "auto-round", "auto_round", "autoround"}

    @classmethod
    def _validate_packed_quant_inventory(
        cls,
        model: nn.Module,
        *,
        packed_type: type[nn.Module] | tuple[type[nn.Module], ...] | None = None,
    ) -> dict[str, int]:
        """Require the complete mixed W4/W8 checkpoint before dequantization.

        Transformers 5.13 does not recognize GPTQModel's ``auto_round``
        spelling and otherwise silently loads only the unquantized tensors.
        Counting the actual packed modules makes that failure impossible to
        mistake for a valid export.
        """

        if packed_type is None:
            from gptqmodel.nn_modules.qlinear import PackableQuantLinear

            packed_type = PackableQuantLinear
        layer_count = int(model.config.num_hidden_layers)
        expected_experts = {
            f"model.layers.{layer}.mlp.experts.{expert}.{projection}"
            for layer in range(layer_count)
            for expert in range(256)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        expected_base = {
            f"model.layers.{layer}.{suffix}"
            for layer in range(layer_count)
            for suffix in (
                "self_attn.q_a_proj",
                "self_attn.q_b_proj",
                "self_attn.kv_proj",
                "self_attn.o_b_proj",
                "mlp.shared_experts.gate_proj",
                "mlp.shared_experts.up_proj",
                "mlp.shared_experts.down_proj",
            )
        }
        packed = {name: module for name, module in model.named_modules() if isinstance(module, packed_type)}
        actual = set(packed)
        expected = expected_experts | expected_base
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        wrong_bits = sorted(
            (name, int(getattr(packed[name], "bits", -1)), 4 if name in expected_experts else 8)
            for name in actual & expected
            if int(getattr(packed[name], "bits", -1)) != (4 if name in expected_experts else 8)
        )
        if missing or unexpected or wrong_bits:
            raise RuntimeError(
                "DeepSeek-V4 packed checkpoint inventory mismatch: "
                f"expected={len(expected)}, actual={len(actual)}, "
                f"missing={missing[:8]}, unexpected={unexpected[:8]}, "
                f"wrong_bits={wrong_bits[:8]}"
            )
        return {
            "layers": layer_count,
            "packed_total": len(actual),
            "routed_expert_w4": len(expected_experts),
            "base_w8": len(expected_base),
        }

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        """Load GPTQ and AutoRound artifacts through GPTQModel itself.

        Both recipe outputs use GPTQModel's packed checkpoint format.  Loading
        AutoRound through vanilla Transformers currently logs an unknown
        ``auto_round`` method and skips all packed weights, so route both
        algorithms through the same known-good loader and converter.
        """

        packed_weight_only = bool(kwargs.pop("packed_weight_only", False))
        config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
        quantization_config = getattr(config, "quantization_config", None)
        quant_method = cls._get_quantization_method(quantization_config)
        if quant_method not in cls._GPTQMODEL_METHODS:
            return super().get_hf_model(hf_model_dir, quant_weight=quant_weight, **kwargs)
        if quant_weight is not None and len(quant_weight) > 0:
            raise RuntimeError(
                "Model is already quantized; quant_weight must be empty when loading a GPTQModel artifact."
            )

        # Packed export does not need a transient dense FP16 model. Ask
        # GPTQModel to build its quantized module tree on meta and materialize
        # the checkpoint directly into packed buffers. The option is explicit
        # so every other model keeps GPTQModel's established loading path.
        if packed_weight_only:
            kwargs.setdefault("use_meta_shell", True)
        hf_model = cls._load_gptqmodel(hf_model_dir, **kwargs)
        inventory = cls._validate_packed_quant_inventory(hf_model)
        if packed_weight_only:
            hf_model = cls._retain_gptqmodel_packed_hf_model(hf_model)
        else:
            hf_model = cls._dequantize_gptqmodel_hf_model(hf_model)
        hf_model = cls._postprocess_gptqmodel_structure(hf_model, **kwargs)
        from xhquant.nn import GPTQPackedLinear

        converted = sum(
            1
            for _, module in hf_model.named_modules()
            if (
                isinstance(module, GPTQPackedLinear)
                if packed_weight_only
                else torch.is_tensor(getattr(module, "quant_weight", None))
            )
        )
        if converted != inventory["packed_total"]:
            raise RuntimeError(
                "DeepSeek-V4 packed modules were not fully converted: "
                f"expected={inventory['packed_total']}, converted={converted}, "
                f"packed_weight_only={packed_weight_only}"
            )
        get_xhquant_logger().info(
            f"DeepSeek-V4 packed quant inventory: {inventory}, packed_weight_only={packed_weight_only}"
        )
        return hf_model

    def get_native_model(self):
        resume_from = self.config.quant_weight
        kwargs = {
            "device_map": "auto" if self.config.enable_auto_offload else "cpu",
            "packed_weight_only": self.config.packed_weight_only,
        }
        native_hf_model = self.get_hf_model(
            self.hf_model_dir,
            quant_weight=resume_from,
            dtype=self.dtype,
            **kwargs,
        )
        hf_model_cls = self.get_hf_model_cls()
        if not isinstance(native_hf_model, hf_model_cls):
            raise TypeError(f"The model is not {hf_model_cls.__name__}, but {type(native_hf_model)}")
        return native_hf_model

    def __init__(self, config: XHDeepSeekV4ModelConfig):
        super().__init__(config)
        self.config = cast(XHDeepSeekV4ModelConfig, self.config)
        layer_count = self.config.get_max_decode_layers()
        layer_count = 43 if layer_count <= 0 else layer_count
        spec = DeepSeekV4StaticCacheSpec(
            max_context_length=self.config.context_max_length,
            prefill_chunk_length=self.config.prefill_chunk_length,
        )
        self.cache_abi = DeepSeekV4CacheABI(
            spec=spec,
            layer_types=default_layer_types(layer_count),
            batch_size=self.config.batch_size,
        )
        self.kvcache_config.num_layers = layer_count
        self.kvcache_config.batch_size = self.config.batch_size
        self.kvcache_config.cache_dtype = "float16"
        self.kvcache_config.use_cache = True
        self.wrap_cfg.kv_cache.num_layers = layer_count
        self.wrap_cfg.kv_cache.batch_size = self.config.batch_size
        self.wrap_cfg.kv_cache.cache_dtype = "float16"
        self.wrap_cfg.kv_cache.use_cache = True
        self._kvcache_mixin = DeepSeekV4CacheMixin(self.cache_abi, self.kvcache_config)
        self._decode_wrap_model: StaticDeepSeekV4ForCausalLM | None = None
        self.layer_types = self.cache_abi.layer_types

    @staticmethod
    def _validate_checkpoint(model: DeepseekV4ForCausalLM, expected_layers: tuple[str, ...]) -> None:
        config = model.config
        checks = {
            "hidden_size": (int(config.hidden_size), 4096),
            "num_attention_heads": (int(config.num_attention_heads), 64),
            "head_dim": (int(config.head_dim), 512),
            "q_lora_rank": (int(config.q_lora_rank), 1024),
            "sliding_window": (int(config.sliding_window), 128),
            "hc_mult": (int(config.hc_mult), 4),
            "hc_sinkhorn_iters": (int(config.hc_sinkhorn_iters), 20),
            "index_topk": (int(config.index_topk), 512),
        }
        mismatches = {
            name: {"checkpoint": actual, "expected": expected}
            for name, (actual, expected) in checks.items()
            if actual != expected
        }
        if mismatches:
            raise ValueError(f"checkpoint does not match DeepSeek-V4 Flash-0731: {mismatches}")
        rates = config.compress_rates
        if int(rates[CSA]) != 4 or int(rates[HCA]) != 128:
            raise ValueError(f"unexpected compressor rates: {rates}")
        actual_layers = tuple(config.layer_types[: len(expected_layers)])
        if actual_layers != expected_layers:
            raise ValueError("checkpoint attention schedule does not match the static cache ABI")
        if len(model.model.layers) < len(expected_layers):
            raise ValueError("checkpoint has fewer decoder layers than the requested export prefix")

    @staticmethod
    def _retain_requested_layer_prefix(
        model: DeepseekV4ForCausalLM,
        requested_layers: int,
    ) -> None:
        """Drop unused meta layer shells before streamed first-N loading.

        Checkpoint validation runs against the complete model first.  Once the
        requested prefix is known, keeping layers beyond it in the empty HF
        tree only makes the generic big-model loader materialize attention and
        mHC tensors which can never be reached by the exported static graph.
        The config intentionally remains unchanged so checkpoint metadata and
        the released 43-layer architecture are still preserved.
        """

        requested_layers = int(requested_layers)
        layers = model.model.layers
        if requested_layers <= 0 or requested_layers > len(layers):
            raise ValueError(
                "requested layer prefix is outside the empty model: "
                f"requested={requested_layers}, available={len(layers)}"
            )
        if requested_layers == len(layers):
            return
        model.model.layers = nn.ModuleList(list(layers[:requested_layers]))

    def init_wrap_model(self, hf_model: Any) -> Any:
        if not isinstance(hf_model, DeepseekV4ForCausalLM):
            raise TypeError(f"expected DeepseekV4ForCausalLM, got {type(hf_model)}")
        self._validate_checkpoint(hf_model, self.layer_types)
        max_layers = len(self.layer_types)
        common = {
            "max_context_length": self.config.context_max_length,
            "max_layers": max_layers,
            "moe_fast_mode": True,
            "swa_backing_length": self.cache_abi.persistent_swa_length,
        }
        prefill = StaticDeepSeekV4ForCausalLM.from_hf(
            hf_model,
            input_sequence_length=self.config.prefill_chunk_length,
            **common,
        )
        decode = StaticDeepSeekV4ForCausalLM.from_hf(
            hf_model,
            input_sequence_length=1,
            **common,
        )
        # The host performs token embedding. Keep an independent copy because
        # the static graph retains only the LM head side of a tied weight.
        self.embed_tokens = copy.deepcopy(hf_model.model.embed_tokens)
        self.pad_token_id = resolve_deepseek_v4_pad_token_id(hf_model.config)
        prefill.to(self._dtype)
        decode.to(self._dtype)
        self.embed_tokens.to(self._dtype)
        self._wrap_model = prefill
        self._decode_wrap_model = decode
        return prefill

    def _to_fronted(self, wrap_model):
        if self._decode_wrap_model is None:
            raise RuntimeError("decode graph was not initialized")
        self.set_prefill()
        self._wrap_model = wrap_model
        prefill = super()._to_fronted(wrap_model)

        self.set_decode()
        self._wrap_model = self._decode_wrap_model
        decode = super()._to_fronted(self._decode_wrap_model)
        self._decode_wrap_model = None

        self._wrap_model = wrap_model
        self.set_prefill()
        result = ModelSwitcher({"prefill": prefill, "decode": decode})
        result.set_activate_model("prefill")
        return result

    def _to_quanted(self, frontend_model, state, **kwargs):
        # A full DeepSeek-V4 graph owns tens of thousands of packed weight
        # modules per stage. Shape metadata is already present on the traced
        # frontend graph, so running MetaInfoPro is unnecessary.  More
        # importantly, infer_shape=False lets the existing PTQ lifecycle drop
        # each packed source/cache and offload its converted HM weight as soon
        # as that module is fixed, instead of retaining the whole stage on one
        # GPU until the loop finishes.
        kwargs.setdefault("infer_shape", False)
        self.set_prefill()
        prefill = super()._to_quanted(frontend_model.prefill, state, **kwargs)
        prefill.cpu()
        frontend_model.prefill.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.set_decode()
        decode = super()._to_quanted(frontend_model.decode, state, **kwargs)
        decode.cpu()
        frontend_model.decode.cpu()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.set_prefill()
        result = ModelSwitcher({"prefill": prefill, "decode": decode})
        result.set_activate_model("prefill")
        return result

    def set_prefill(self):
        if self._state == LLMModelState.FRONTED:
            self._frontend_model.set_activate_model("prefill")
        elif self._state in {
            LLMModelState.QUANTED_ALIGNED,
            LLMModelState.QUANTED_FAST,
            LLMModelState.QUANTED_DISABLE,
        }:
            self._quanted_model.set_activate_model("prefill")
        super().set_prefill()

    def set_decode(self):
        if self._state == LLMModelState.FRONTED:
            self._frontend_model.set_activate_model("decode")
        elif self._state in {
            LLMModelState.QUANTED_ALIGNED,
            LLMModelState.QUANTED_FAST,
            LLMModelState.QUANTED_DISABLE,
        }:
            self._quanted_model.set_activate_model("decode")
        super().set_decode()

    def _get_data_preprocessor(self):
        if self.embed_tokens is None:
            raise RuntimeError("token embedding is not initialized")
        return DeepSeekV4DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            cache_mixin=self._kvcache_mixin,
            pad_token_id=self.pad_token_id,
        )

    def _release_prefill_quanted_model_after_export(self) -> bool:
        """Bound full-network export memory to one serialized stage at a time."""

        return True

    @staticmethod
    def _cache_input_suffixes(layer_type: str) -> tuple[str, ...]:
        if layer_type == SLIDING:
            return ("swa_kv",)
        if layer_type == CSA:
            return (
                "swa_kv",
                "main",
                "index_k",
                "main_kv_state",
                "main_score_state",
                "index_kv_state",
                "index_score_state",
            )
        return (
            "swa_kv",
            "main",
            "main_kv_state",
            "main_score_state",
        )

    def get_export_cfg(self) -> dict[str, list[str]]:
        inputs = [
            "inputs_embeds",
            "input_ids",
            "past_seq_length",
            "current_input_length",
            "last_token_index",
            "csa_write_start",
            "hca_write_start",
            "swa_attention_mask",
            "csa_index_validity",
            "csa_attention_mask",
            "hca_attention_mask",
            "csa_compressor_validity",
            "csa_compressor_new_count",
            "csa_compressor_offset",
            "csa_compressor_phase_indices",
            "hca_compressor_validity",
            "hca_compressor_new_count",
            "hca_compressor_offset",
            "hca_compressor_phase_indices",
        ]
        for layer, layer_type in enumerate(self.layer_types):
            inputs.extend(f"layer_{layer}_{suffix}_input" for suffix in self._cache_input_suffixes(layer_type))

        outputs = ["logits"]
        for layer in self.cache_abi.csa_layers:
            outputs.extend(
                f"layer_{layer}_{suffix}_output"
                for suffix in (
                    "main_kv_state",
                    "main_score_state",
                    "index_kv_state",
                    "index_score_state",
                )
            )
        for layer in self.cache_abi.hca_layers:
            outputs.extend(
                f"layer_{layer}_{suffix}_output"
                for suffix in (
                    "main_kv_state",
                    "main_score_state",
                )
            )
        return {"input_names": inputs, "output_names": outputs}

    def _get_big_language_placeholder_export_components(self):
        from ._deepseek_v4_big_export import DeepSeekV4BigHFModel

        return DeepSeekV4BigHFModel, DeepSeekV4BigHFModel.PLACEHOLDER_TYPES

    def _check_big_language_placeholder_export_supported(
        self,
        empty_hf_model: Any,
    ) -> None:
        language_model = self._get_language_model(empty_hf_model)
        found = {type(module).__name__ for module in language_model.modules()}
        _, placeholder_types = self._get_big_language_placeholder_export_components()
        if not any(name in found for name in placeholder_types):
            raise NotImplementedError(
                "DeepSeek-V4 low-memory export could not find its SparseMoeBlock "
                f"boundary; expected one of {placeholder_types}, found {sorted(found)}"
            )

    def _export_big_language_hmonnx(self, exported_info) -> None:
        """Export the static main graphs while streaming one complete MoE at a time."""

        if self._state != LLMModelState.NONE:
            raise RuntimeError("DeepSeek-V4 low-memory export must start before the full model is loaded")
        if not self.config.packed_weight_only:
            raise ValueError("DeepSeek-V4 low-memory export requires packed_weight_only=True")

        logger = get_xhquant_logger()
        big_hf_model_cls, placeholder_types = self._get_big_language_placeholder_export_components()
        empty_hf_model = self.get_empty_hf_model(
            self.hf_model_dir,
            dtype=self.dtype,
        )
        self._validate_checkpoint(empty_hf_model, self.layer_types)
        self._check_big_language_placeholder_export_supported(empty_hf_model)
        available_placeholder_prefixes = big_hf_model_cls.resolve_placeholder_prefixes(
            empty_hf_model,
            placeholder_types,
        )
        available_placeholder_set = set(available_placeholder_prefixes)
        main_placeholder_prefixes = [f"model.layers.{layer}.mlp" for layer in range(len(self.layer_types))]
        active_layer_prefixes = [f"model.layers.{layer}" for layer in range(len(self.layer_types))]
        missing_placeholder_prefixes = sorted(set(main_placeholder_prefixes) - available_placeholder_set)
        if missing_placeholder_prefixes:
            raise RuntimeError(
                "DeepSeek-V4 checkpoint is missing requested MoE placeholders: "
                f"requested_layers={len(self.layer_types)}, "
                f"available={len(available_placeholder_prefixes)}, "
                f"missing={missing_placeholder_prefixes[:8]}"
            )

        # Validation above sees the complete checkpoint. From this point on,
        # physically retain only the requested layer prefix so the generic
        # materializer cannot load unused layer>=N attention/mHC tensors.
        self._retain_requested_layer_prefix(
            empty_hf_model,
            len(self.layer_types),
        )

        # Keep expert shells only in the placeholder template and all other
        # requested-layer tensors only in the main graph template.
        empty_hf_model_for_placeholder = copy.deepcopy(empty_hf_model)
        big_hf_model_cls._preprocess_quantized_hf_model(
            empty_hf_model_for_placeholder,
            self.hf_model_dir,
            include_module_prefixes=main_placeholder_prefixes,
        )
        big_hf_model_cls._preprocess_quantized_hf_model(
            empty_hf_model,
            self.hf_model_dir,
            include_module_prefixes=active_layer_prefixes,
            skip_module_prefixes=main_placeholder_prefixes,
        )
        big_hf_model = big_hf_model_cls(
            self.hf_model_dir,
            empty_hf_model,
            placeholder_types,
        )
        marked = big_hf_model_cls.mark_streaming_placeholders(
            empty_hf_model,
            main_placeholder_prefixes,
        )
        if marked != len(self.layer_types):
            raise RuntimeError(
                "DeepSeek-V4 marked MoE count does not match requested layers: "
                f"marked={marked}, layers={len(self.layer_types)}"
            )

        self.to_wrap(empty_hf_model)
        if self._decode_wrap_model is None:
            raise RuntimeError("DeepSeek-V4 decode graph was not initialized")
        static_placeholders = big_hf_model.register_static_placeholder_modules(
            self._wrap_model,
            self._decode_wrap_model,
        )
        expected_static = 2 * len(self.layer_types)
        if static_placeholders != expected_static:
            raise RuntimeError(
                "DeepSeek-V4 static placeholder count mismatch: "
                f"expected={expected_static}, actual={static_placeholders}"
            )
        self.to_fronted()
        if not isinstance(self._frontend_model, ModelSwitcher):
            raise TypeError("DeepSeek-V4 low-memory export requires prefill/decode graphs")

        exported_dir = Path(exported_info.exported_dir)
        self.set_prefill()
        prefill_wrap_cfg = copy.deepcopy(self.get_wrap_cfg())
        big_hf_model.register_layer_as_place_holder(self._frontend_model.prefill)
        self.set_decode()
        decode_wrap_cfg = copy.deepcopy(self.get_wrap_cfg())
        big_hf_model.register_layer_as_place_holder(self._frontend_model.decode)

        big_hf_model.export_prefill_decode_placeholder_layers(
            self._frontend_model.prefill,
            self._frontend_model.decode,
            empty_hf_model_for_placeholder,
            self.config.chip_arch,
            prefill_wrap_cfg,
            decode_wrap_cfg,
            self.get_quant_cfg(),
            exported_dir / "prefill" / "placeholders",
            exported_dir / "decode" / "placeholders",
            # A single visible GPU is sufficient and is the default V4 path.
            placeholder_export_workers=1,
        )

        self.set_prefill()
        self.to_quanted_aligned()
        if not isinstance(self._quanted_model, ModelSwitcher):
            raise TypeError("DeepSeek-V4 quantization lost the prefill/decode switcher")
        self._quanted_model.prefill.fixed()
        self._quanted_model.decode.fixed()
        self._export_hmonnx(exported_info)

        meta_info = exported_info.meta
        prefill_hmonnx = str(exported_dir / meta_info.prefill_hmonnx)
        decode_hmonnx = str(exported_dir / meta_info.decode_hmonnx)
        big_hf_model.replace_hmonnx_placeholders_with_subgraphs(prefill_hmonnx)
        big_hf_model.replace_hmonnx_placeholders_with_subgraphs(decode_hmonnx)
        logger.info(
            "DeepSeek-V4 low-memory export inlined %d MoE targets into both graphs.",
            len(self.layer_types),
        )

    def export_hmonnx(self, output_dir: str) -> LLMModelMeta:
        """Export the fixed prefill and decode graphs through the standard optimizer path."""

        logger = get_xhquant_logger()
        exported_info = self.get_export_info(output_dir)
        if is_huge_model_export_enabled():
            logger.info("HUGE_MODEL_EXPORT_ENABLED is active; using streamed DeepSeek-V4 MoE export.")
            self._export_big_language_hmonnx(exported_info)
        else:
            if self._state != LLMModelState.QUANTED_ALIGNED:
                self.to_quanted_aligned()
            if not isinstance(self._quanted_model, ModelSwitcher):
                raise TypeError("DeepSeek-V4 export requires separate prefill and decode graphs")
            self._quanted_model.prefill.fixed()
            self._quanted_model.decode.fixed()
            self._export_hmonnx(exported_info)
        meta_info = exported_info.meta
        self._finalize_deterministic_hmonnx(exported_info)
        meta_info.prefill_onnx = meta_info.prefill_hmonnx
        meta_info.decode_onnx = meta_info.decode_hmonnx
        meta_info.token_embedding_file = meta_info.quant_embedding
        meta_info.max_context_tokens = self.config.context_max_length
        meta_path = Path(exported_info.exported_dir) / "golden_meta_info.json"
        meta_path.write_text(json.dumps(meta_info.to_dict(), indent=4), encoding="utf-8")
        self._write_merak_config(exported_info.exported_dir)
        logger.info(f"DeepSeek-V4 export completed: {exported_info.exported_dir}")
        return meta_info

    @staticmethod
    def _write_merak_config(exported_dir: str | Path) -> Path:
        """Write the vLLM Merak package entrypoint beside the HMONNX metadata."""

        config_path = Path(exported_dir) / "merak_config.json"
        config = {
            "architectures": ["MerakForCausalLM"],
            "config_format": "merak_llm",
            "load_format": "merak_llm",
            "xh_model": {
                "model_type": "hmonnx",
                "meta_info": "golden_meta_info.json",
            },
            "enable_page_attention": False,
            "model_type": "merak_llm",
        }
        config_path.write_text(json.dumps(config, indent=4) + "\n", encoding="utf-8")
        return config_path

    @staticmethod
    def _finalize_deterministic_hmonnx(exported_info) -> None:
        """Give regular and streamed exports one byte-stable serialization."""

        logger = get_xhquant_logger()
        exported_dir = Path(exported_info.exported_dir)
        meta_info = exported_info.meta
        for path_field, md5_field in (
            ("prefill_hmonnx", "prefill_hmonnx_md5"),
            ("decode_hmonnx", "decode_hmonnx_md5"),
        ):
            hmonnx_path = Path(getattr(meta_info, path_field))
            if not hmonnx_path.is_absolute():
                hmonnx_path = exported_dir / hmonnx_path
            canonicalize_hmonnx_artifact(hmonnx_path, logger=logger)
            setattr(meta_info, md5_field, calculate_file_md5(str(hmonnx_path)))

    def _extra_export_metadata(self, output_dir: str, meta_info):
        del output_dir
        # Huge-model export creates metadata before ``init_wrap_model`` has a
        # chance to copy this value from the HF shell.  Resolve it directly
        # from config so fixed-shape short-prefill padding is always usable.
        hf_config = AutoConfig.from_pretrained(
            self.hf_model_dir,
            trust_remote_code=True,
        )
        self.pad_token_id = resolve_deepseek_v4_pad_token_id(hf_config)
        meta_info.pad_token_id = self.pad_token_id
        meta_info.layer_types = list(self.layer_types)
        return meta_info


__all__ = ["XHDeepSeekV4Model"]
