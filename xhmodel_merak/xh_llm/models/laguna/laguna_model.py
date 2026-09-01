import copy
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig, PreTrainedModel

from xhmodel_merak.utils import calculate_file_md5
from xhquant.api import get_xhquant_logger

from ...builder import register_llm_model
from ...llm_data_processor import BaseInputProcessorConfig
from ...text_llm_model import TextLLMModel, TextLLMModelConfig
from ...types import LLMModelState, ModelSwitcher
from ...utils import is_huge_model_export_enabled
from .data_preprocess import LagunaDataPreprocess, LagunaSlidingMaskForwardMixin
from .kv_cache import LagunaKVCacheMixin, build_laguna_layer_kv_shapes
from .laguna_hf_compatible import build_laguna_hf_compatible_model
from .laguna_hmonnx_inference import XHLagunaHMONNXModel


class XHLagunaModelConfig(TextLLMModelConfig):
    pass


@register_llm_model("LagunaForCausalLM")
class XHLagunaModel(LagunaSlidingMaskForwardMixin, TextLLMModel):
    HF_MODEL_CLS = PreTrainedModel
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    HMONNXINFERENCE_CLS = XHLagunaHMONNXModel
    CONFIG_CLS = XHLagunaModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_laguna_hf_compatible_model)
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.laguna.workflow:LagunaWorkflow"

    def __init__(self, config: XHLagunaModelConfig):
        super().__init__(config)
        self._kvcache_mixin = LagunaKVCacheMixin(self.kvcache_config)
        self.uses_explicit_sliding_attention_mask = True
        self.sliding_window = 0
        self.layer_types: list[str] = []

    @staticmethod
    def _ensure_autoround_available(model_dir: str | Path) -> None:
        try:
            import auto_round  # noqa: F401

            return
        except ImportError:
            pass

        resolved_model_dir = Path(model_dir).resolve()
        candidate_roots = [
            resolved_model_dir.parents[1] / "third_party" / "auto-round"
            if len(resolved_model_dir.parents) >= 2
            else None,
            Path(__file__).resolve().parents[5] / "gptqmodel" / "third_party" / "auto-round",
        ]
        for candidate_root in candidate_roots:
            if candidate_root is None or not (candidate_root / "auto_round" / "__init__.py").is_file():
                continue
            if str(candidate_root) not in sys.path:
                sys.path.insert(0, str(candidate_root))
            import auto_round  # noqa: F401

            return

        raise ImportError(
            "Laguna AutoRound checkpoint loading requires the auto_round package. "
            "Install auto-round or keep the checkpoint under a GPTQModel checkout containing "
            "third_party/auto-round."
        )

    @classmethod
    def _uses_autoround(cls, quantization_config: Any) -> bool:
        quant_method = cls._get_quantization_method(quantization_config)
        if isinstance(quantization_config, dict):
            provider = quantization_config.get("provider")
            packing_format = quantization_config.get("packing_format")
        else:
            provider = getattr(quantization_config, "provider", None)
            packing_format = getattr(quantization_config, "packing_format", None)
        provider = str(getattr(provider, "value", provider) or "").lower().replace("_", "-")
        packing_format = str(getattr(packing_format, "value", packing_format) or "").lower()
        return (
            quant_method in {"auto-round", "auto_round", "autoround"}
            or provider == "auto-round"
            or packing_format.startswith("auto_round:")
        )

    @classmethod
    def _load_hf_model(cls, hf_model_dir: str | Path, **kwargs):
        from .float_checkpoint_compat import fuse_split_experts, split_expert_checkpoint_loader

        kwargs.setdefault("trust_remote_code", True)
        hf_config = AutoConfig.from_pretrained(str(hf_model_dir), trust_remote_code=True)
        quantization_config = getattr(hf_config, "quantization_config", None)
        is_autoround = cls._uses_autoround(quantization_config)
        if is_autoround:
            cls._ensure_autoround_available(hf_model_dir)
        with split_expert_checkpoint_loader(str(hf_model_dir)) as native_experts_cls:
            native_model = super()._load_hf_model(str(hf_model_dir), **kwargs)
        if native_model.__class__.__name__ != "LagunaForCausalLM":
            raise TypeError(
                "Laguna support expects the trust_remote_code class LagunaForCausalLM, "
                f"got {native_model.__class__.__module__}.{native_model.__class__.__name__}"
            )
        if is_autoround:
            return native_model
        expected_sparse_layers = int(native_model.config.num_hidden_layers) - len(native_model.config.mlp_only_layers)
        converted_sparse_layers = fuse_split_experts(native_model, native_experts_cls)
        if converted_sparse_layers != expected_sparse_layers:
            raise RuntimeError(
                "Laguna floating-point checkpoint expert conversion was incomplete: "
                f"expected {expected_sparse_layers}, converted {converted_sparse_layers}"
            )
        return native_model

    @classmethod
    def _dequantize_autoround_hf_model(cls, native_hf_model: Any):
        from .gptqmodel_compat import restore_gptqmodel_moe_structure

        model_dir = str(getattr(native_hf_model.config, "_name_or_path", ""))
        if not model_dir:
            raise ValueError("Laguna AutoRound expert restoration requires the checkpoint model directory")
        native_hf_model = super()._dequantize_autoround_hf_model(native_hf_model)
        expected = int(native_hf_model.config.num_hidden_layers) - len(native_hf_model.config.mlp_only_layers)
        converted = restore_gptqmodel_moe_structure(native_hf_model, model_dir)
        if converted != expected:
            raise RuntimeError(
                "Laguna AutoRound expert restoration was incomplete: "
                f"expected {expected}, converted {converted}"
            )
        return native_hf_model

    @classmethod
    def _load_gptqmodel(cls, hf_model_dir: str, device_map="cpu", **kwargs):
        from .gptqmodel_compat import laguna_gptqmodel_loader

        with laguna_gptqmodel_loader(hf_model_dir):
            return super()._load_gptqmodel(hf_model_dir, device_map=device_map, **kwargs)

    @classmethod
    def _postprocess_gptqmodel_structure(cls, native_hf_model: Any, **kwargs) -> Any:
        from .gptqmodel_compat import restore_gptqmodel_moe_structure

        model_dir = str(kwargs.get("hf_model_dir") or kwargs.get("model_dir") or "")
        if not model_dir:
            model_dir = str(getattr(native_hf_model.config, "_name_or_path", ""))
        if not model_dir:
            raise ValueError("Laguna GPTQModel structure restoration requires the checkpoint model directory")
        expected = int(native_hf_model.config.num_hidden_layers) - len(native_hf_model.config.mlp_only_layers)
        converted = restore_gptqmodel_moe_structure(native_hf_model, model_dir)
        if converted != expected:
            raise RuntimeError(
                "Laguna GPTQModel expert restoration was incomplete: "
                f"expected {expected}, converted {converted}"
            )
        return native_hf_model

    @classmethod
    def get_empty_hf_model(cls, hf_model_dir: str | Path, **kwargs) -> Any:
        from accelerate import init_empty_weights

        try:
            from transformers.modeling_utils import no_init_weights
        except ImportError:
            no_init_weights = init_empty_weights

        kwargs.setdefault("trust_remote_code", True)
        model_dtype = cls.get_hf_model_dtype()
        if "dtype" not in kwargs:
            kwargs["dtype"] = model_dtype
        config = AutoConfig.from_pretrained(str(hf_model_dir), trust_remote_code=True)
        with no_init_weights(), init_empty_weights():
            hf_model = cls.HF_AUTO_MODEL_CLS.from_config(config, **kwargs)
            if hf_model.__class__.__name__ != "LagunaForCausalLM":
                raise TypeError(
                    "Laguna empty model expects the trust_remote_code class LagunaForCausalLM, "
                    f"got {hf_model.__class__.__module__}.{hf_model.__class__.__name__}"
                )
            if hf_model.can_generate():
                try:
                    hf_model.generation_config = GenerationConfig.from_pretrained(str(hf_model_dir))
                except OSError:
                    get_xhquant_logger().info(
                        "Generation config file not found, using a generation config created from the model config."
                    )
        return hf_model

    @classmethod
    def _get_hf_model_for_compatible(cls, hf_model_dir=None):
        return cls.get_empty_hf_model(hf_model_dir, trust_remote_code=True)

    @classmethod
    def _get_low_memory_empty_hf_model(cls, hf_model_dir: str | Path, **kwargs) -> Any:
        from ._laguna_big_export import checkpoint_router_bias_layout, move_router_bias_to_checkpoint_layout
        from .float_checkpoint_compat import split_expert_checkpoint_loader

        with split_expert_checkpoint_loader(str(hf_model_dir)):
            hf_model = cls.get_empty_hf_model(hf_model_dir, **kwargs)
        quantization_config = getattr(hf_model.config, "quantization_config", None)
        if cls._uses_autoround(quantization_config):
            cls._ensure_autoround_available(hf_model_dir)
            # ARK needs post_init before forward, but low-memory FX tracing runs first.
            if isinstance(quantization_config, dict):
                if quantization_config.get("backend", "auto") == "auto":
                    quantization_config["backend"] = "torch"
            elif getattr(quantization_config, "backend", "auto") == "auto":
                quantization_config.backend = "torch"
        expected_sparse_layers = int(hf_model.config.num_hidden_layers) - len(hf_model.config.mlp_only_layers)
        router_bias_layout = checkpoint_router_bias_layout(hf_model_dir)
        if router_bias_layout == "experts":
            prepared_router_biases = move_router_bias_to_checkpoint_layout(hf_model)
        else:
            prepared_router_biases = sum(
                "e_score_correction_bias" in gate._parameters
                for sparse_block in hf_model.modules()
                if (gate := getattr(sparse_block, "gate", None)) is not None
            )
        if prepared_router_biases != expected_sparse_layers:
            raise RuntimeError(
                "Laguna low-memory checkpoint layout preparation was incomplete: "
                f"layout={router_bias_layout}, expected={expected_sparse_layers}, "
                f"prepared={prepared_router_biases}"
            )
        return hf_model

    def _get_big_language_placeholder_export_components(self):
        from ._laguna_big_export import LagunaBigHFModel

        return LagunaBigHFModel, LagunaBigHFModel.PLACEHOLDER_TYPES

    def _check_big_language_placeholder_export_supported(self, empty_hf_model: Any) -> None:
        language_model = self._get_language_model(empty_hf_model)
        found = {type(module).__name__ for module in language_model.modules()}
        _, placeholder_types = self._get_big_language_placeholder_export_components()
        missing = sorted(set(placeholder_types) - found)
        if missing:
            raise NotImplementedError(
                "Laguna low-memory export could not find its sparse-MoE boundary: "
                f"expected={placeholder_types}, missing={missing}"
            )

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._model import register_wrap_modules

        # The low-memory trace context has already replaced the sparse-MoE
        # registry entry with its weightless placeholder. Re-registering the
        # normal wrapper here would expand the split expert shell again.
        if not getattr(self, "_low_memory_export_active", False):
            register_wrap_modules(hf_model)
        return super().init_wrap_model(hf_model)

    def _to_fronted(self, wrap_model):
        if not getattr(self, "_low_memory_export_active", False):
            return super()._to_fronted(wrap_model)

        self.set_prefill()
        prefill_wrap_model = wrap_model
        decode_wrap_model = copy.deepcopy(wrap_model)
        self._wrap_model = prefill_wrap_model
        prefill_frontend_model = super()._to_fronted(prefill_wrap_model)

        self._wrap_model = decode_wrap_model
        self.set_decode()
        decode_frontend_model = super()._to_fronted(decode_wrap_model)

        self._wrap_model = prefill_wrap_model
        self.set_prefill()
        frontend_model = ModelSwitcher(
            {"prefill": prefill_frontend_model, "decode": decode_frontend_model}
        )
        frontend_model.set_activate_model("prefill")
        return frontend_model

    def _to_quanted(
        self,
        frontend_model,
        state,
        auto_release_unused_parameters: bool = True,
        infer_shape: bool = True,
    ):
        if not getattr(self, "_low_memory_export_active", False) or not isinstance(frontend_model, ModelSwitcher):
            return super()._to_quanted(
                frontend_model,
                state,
                auto_release_unused_parameters=auto_release_unused_parameters,
                infer_shape=infer_shape,
            )

        self.set_prefill()
        prefill_quanted_model = super()._to_quanted(
            frontend_model.prefill,
            state,
            auto_release_unused_parameters=auto_release_unused_parameters,
            infer_shape=infer_shape,
        )
        self.set_decode()
        decode_quanted_model = super()._to_quanted(
            frontend_model.decode,
            state,
            auto_release_unused_parameters=auto_release_unused_parameters,
            infer_shape=infer_shape,
        )
        self.set_prefill()
        quanted_model = ModelSwitcher(
            {"prefill": prefill_quanted_model, "decode": decode_quanted_model}
        )
        quanted_model.set_activate_model("prefill")
        return quanted_model

    def _wraped_post(self, hf_model: Any):
        super()._wraped_post(hf_model)
        llm_model = self._get_language_model(self._wrap_model)
        language_config = llm_model.config
        num_decoder_layers = language_config.num_hidden_layers
        max_layers = self.config.get_max_decode_layers()
        if max_layers > 0:
            assert max_layers <= num_decoder_layers
            num_decoder_layers = max_layers
        self.sliding_window = int(getattr(language_config, "sliding_window", 0) or 0)
        self.layer_types = list(
            getattr(language_config, "layer_types", ["full_attention"] * language_config.num_hidden_layers)
        )[:num_decoder_layers]
        if self.use_cache:
            batch_size = self.config.batch_size
            layer_kv_shapes = build_laguna_layer_kv_shapes(
                layer_types=self.layer_types,
                batch_size=batch_size,
                num_key_value_heads=language_config.num_key_value_heads,
                context_max_length=self.config.context_max_length,
                sliding_window=self.sliding_window,
                input_sequence_length=self.config.prefill_chunk_length,
                head_dim=language_config.head_dim,
            )
            self._kvcache_mixin.set_layer_kv_shapes(layer_kv_shapes)
            self.kvcache_config.batch_size = batch_size
        pad_token_id = getattr(language_config, "pad_token_id", None)
        if pad_token_id is None:
            eos_token_id = language_config.eos_token_id
            pad_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
        self.pad_token_id = pad_token_id

    def _get_data_preprocessor(self) -> LagunaDataPreprocess:
        return LagunaDataPreprocess(
            BaseInputProcessorConfig(
                embed_tokens=self.embed_tokens,
                input_sequence_length=self.wrap_cfg.input_sequence_length,
                past_key_caches=self.past_key_caches,
                past_value_caches=self.past_value_caches,
                pad_token_id=self.pad_token_id,
            ),
            sliding_window=self.sliding_window,
        )

    def get_export_cfg(self) -> dict[str, list[str]]:
        export_cfg = super().get_export_cfg()
        export_cfg["input_names"].insert(3, "sliding_attention_mask")
        return export_cfg

    def _extra_export_metadata(self, output_dir: str, meta_info):
        meta_info = super()._extra_export_metadata(output_dir, meta_info)
        meta_info.pad_token_id = self.pad_token_id
        meta_info.uses_explicit_sliding_attention_mask = True
        meta_info.sliding_window = self.sliding_window
        meta_info.layer_types = self.layer_types
        meta_info.layer_kv_shapes = [list(shape) for shape in self._kvcache_mixin.layer_kv_shapes]
        meta_info.sliding_kv_cache_input_mode = "slice_window"
        return meta_info

    def create_export_metadata(self, output_dir: str):
        meta_info = super().create_export_metadata(output_dir)
        hf_config_dir = Path(output_dir) / meta_info.hf_config
        for filename in ("configuration_laguna.py", "modeling_laguna.py"):
            source = Path(self.config.hf_model) / filename
            if not source.is_file():
                raise FileNotFoundError(f"Laguna export requires remote-code file: {source}")
            shutil.copy2(source, hf_config_dir / filename)
        tokenizer_config_path = hf_config_dir / "tokenizer_config.json"
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
        tokenizer_config["fix_mistral_regex"] = True
        tokenizer_config_path.write_text(
            json.dumps(tokenizer_config, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return meta_info

    def _export_big_language_hmonnx_impl(self, exported_info):
        """Export Laguna while materializing at most one sparse MoE block."""

        from ...wrap_model import traceable_module_placeholder_context

        if self._state != LLMModelState.NONE:
            raise RuntimeError("Laguna low-memory export must start before the full model is loaded")

        big_hf_model_cls, placeholder_types = self._get_big_language_placeholder_export_components()
        empty_hf_model = self._get_low_memory_empty_hf_model(
            self.hf_model_dir,
            dtype=self.dtype,
        )
        big_hf_model_cls.initialize_process_worker_after_model_load(empty_hf_model)
        self._check_big_language_placeholder_export_supported(empty_hf_model)
        main_placeholder_prefixes = big_hf_model_cls.resolve_placeholder_prefixes(
            empty_hf_model,
            placeholder_types,
        )

        empty_hf_model_for_placeholder = copy.deepcopy(empty_hf_model)
        big_hf_model_cls._preprocess_quantized_hf_model(
            empty_hf_model_for_placeholder,
            self.hf_model_dir,
        )
        big_hf_model_cls.initialize_process_worker_after_quantized_preprocess(
            empty_hf_model_for_placeholder
        )
        big_hf_model_cls._preprocess_quantized_hf_model(
            empty_hf_model,
            self.hf_model_dir,
            skip_module_prefixes=main_placeholder_prefixes,
        )
        big_hf_model_cls.initialize_process_worker_after_quantized_preprocess(empty_hf_model)

        big_hf_model = big_hf_model_cls(
            self.hf_model_dir,
            empty_hf_model,
            placeholder_types,
        )
        big_hf_model.register_layer_as_placeholder(empty_hf_model)

        def placeholder_callback(registry):
            return big_hf_model_cls.register_placeholder(
                registry,
                hf_model=empty_hf_model,
            )

        with traceable_module_placeholder_context(
            placeholder_types,
            callback=placeholder_callback,
        ):
            self.to_wrap(empty_hf_model)

        big_hf_model.strip_unwrapped_placeholder_members(empty_hf_model)
        big_hf_model.register_layer_as_placeholder(empty_hf_model)
        self.to_fronted()
        if not isinstance(self._frontend_model, ModelSwitcher):
            raise TypeError("Laguna low-memory export requires separate prefill and decode graphs")

        exported_dir = Path(exported_info.exported_dir)
        self._extra_export_metadata(str(exported_dir), exported_info.meta)
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
            placeholder_export_workers=1,
        )

        self.set_prefill()
        self.to_quanted_aligned()
        if not isinstance(self._quanted_model, ModelSwitcher):
            raise TypeError("Laguna low-memory quantization lost the prefill/decode switcher")
        self._quanted_model.prefill.fixed()
        self._quanted_model.decode.fixed()
        self._export_hmonnx(exported_info)

        meta_info = exported_info.meta
        prefill_hmonnx = exported_dir / meta_info.prefill_hmonnx
        decode_hmonnx = exported_dir / meta_info.decode_hmonnx
        big_hf_model.replace_hmonnx_placeholders_with_subgraphs(str(prefill_hmonnx))
        big_hf_model.replace_hmonnx_placeholders_with_subgraphs(str(decode_hmonnx))
        meta_info.prefill_hmonnx_md5 = calculate_file_md5(str(prefill_hmonnx))
        meta_info.decode_hmonnx_md5 = calculate_file_md5(str(decode_hmonnx))
        meta_dict = meta_info.to_dict()
        (exported_dir / "golden_meta_info.json").write_text(
            json.dumps(meta_dict, indent=4),
            encoding="utf-8",
        )
        get_xhquant_logger().info(
            "Laguna low-memory export completed: %s",
            exported_info.exported_dir,
        )
        return meta_dict

    def _export_big_language_hmonnx(self, exported_info):
        previous_export_state = getattr(self, "_low_memory_export_active", False)
        self._low_memory_export_active = True
        try:
            return self._export_big_language_hmonnx_impl(exported_info)
        finally:
            self._low_memory_export_active = previous_export_state

    def export_hmonnx(self, output_dir: str):
        if not is_huge_model_export_enabled():
            return super().export_hmonnx(output_dir)
        exported_info = self.get_export_info(output_dir)
        return self._export_big_language_hmonnx(exported_info)
