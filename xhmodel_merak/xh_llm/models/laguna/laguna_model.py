import json
import shutil
import sys
from pathlib import Path
from typing import Any

from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig, PreTrainedModel

from xhquant.api import get_xhquant_logger

from ...builder import register_llm_model
from ...text_llm_model import TextLLMModel, TextLLMModelConfig
from .laguna_hf_compatible import build_laguna_hf_compatible_model
from .laguna_hmonnx_inference import XHLagunaHMONNXModel


class XHLagunaModelConfig(TextLLMModelConfig):
    pass


@register_llm_model("LagunaForCausalLM")
class XHLagunaModel(TextLLMModel):
    HF_MODEL_CLS = PreTrainedModel
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    HMONNXINFERENCE_CLS = XHLagunaHMONNXModel
    CONFIG_CLS = XHLagunaModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_laguna_hf_compatible_model)
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.laguna.workflow:LagunaWorkflow"

    def __init__(self, config: XHLagunaModelConfig):
        super().__init__(config)

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
    def _load_hf_model(cls, hf_model_dir: str | Path, **kwargs):
        from .float_checkpoint_compat import fuse_split_experts, split_expert_checkpoint_loader

        kwargs.setdefault("trust_remote_code", True)
        hf_config = AutoConfig.from_pretrained(str(hf_model_dir), trust_remote_code=True)
        quantization_config = getattr(hf_config, "quantization_config", None)
        quant_method = cls._get_quantization_method(quantization_config)
        if isinstance(quantization_config, dict):
            provider = quantization_config.get("provider")
            packing_format = quantization_config.get("packing_format")
        else:
            provider = getattr(quantization_config, "provider", None)
            packing_format = getattr(quantization_config, "packing_format", None)
        provider = str(getattr(provider, "value", provider) or "").lower().replace("_", "-")
        packing_format = str(getattr(packing_format, "value", packing_format) or "").lower()
        is_autoround = (
            quant_method in {"auto-round", "auto_round", "autoround"}
            or provider == "auto-round"
            or packing_format.startswith("auto_round:")
        )
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

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._model import register_wrap_modules

        register_wrap_modules(hf_model)
        return super().init_wrap_model(hf_model)

    def _wraped_post(self, hf_model: Any):
        super()._wraped_post(hf_model)
        if self.use_cache:
            llm_model = self._get_language_model(self._wrap_model)
            batch_size = self.config.batch_size
            num_decoder_layers = llm_model.config.num_hidden_layers
            max_layers = self.config.get_max_decode_layers()
            if max_layers > 0:
                assert max_layers <= num_decoder_layers
                num_decoder_layers = max_layers

            self.kvcache_config.num_layers = num_decoder_layers
            self.kvcache_config.kv_cache_shape = [
                batch_size,
                llm_model.config.num_key_value_heads,
                self.config.context_max_length,
                llm_model.config.head_dim,
            ]
            self.kvcache_config.batch_size = batch_size

        language_config = self._get_language_model(self._wrap_model).config
        pad_token_id = getattr(language_config, "pad_token_id", None)
        if pad_token_id is None:
            eos_token_id = language_config.eos_token_id
            pad_token_id = eos_token_id[0] if isinstance(eos_token_id, list) else eos_token_id
        self.pad_token_id = pad_token_id

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
