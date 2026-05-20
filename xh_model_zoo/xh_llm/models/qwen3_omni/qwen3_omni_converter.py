import gc
import json
import re
import shutil
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoTokenizer
from transformers import Qwen3MoeForCausalLM
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeThinkerConfig
from transformers.utils.quantization_config import QuantizationMethod

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model
from .modeling_qwen3_omni_moe import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeThinkerForConditionalGeneration
from .qwen3_omni_convert_config import Qwen3OmniMoeConvertConfig

from xhquant.api import (  # isort:skip
    CacheTensor,
    Config,
    ConfigDict,
    DeviceType,
    convert_fx_model_to_hmonnx,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
    is_ssfp_quant_config,
)


def _release_cuda_memory(logger=None, label: Optional[str] = None):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if logger is not None:
        suffix = f" after {label}" if label else ""
        logger.info(f"released converter resources and cleared CUDA cache{suffix}")


def _iter_qwen3omni_nested_configs(config):
    seen = set()
    stack = [config]
    while stack:
        nested_config = stack.pop()
        if nested_config is None or not hasattr(nested_config, "__dict__"):
            continue
        nested_id = id(nested_config)
        if nested_id in seen:
            continue
        seen.add(nested_id)
        yield nested_config
        for attr_name, child_config in vars(nested_config).items():
            if attr_name.endswith("_config") and hasattr(child_config, "__dict__"):
                stack.append(child_config)


def _normalize_text_rope_scaling(config) -> None:
    for nested_config in _iter_qwen3omni_nested_configs(config):
        rope_scaling = getattr(nested_config, "rope_scaling", None)
        if rope_scaling is None:
            rope_scaling = getattr(nested_config, "rope_parameters", None)
        if isinstance(rope_scaling, dict):
            rope_scaling = dict(rope_scaling)
            rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
            if rope_type == "default":
                rope_type = "linear"
            rope_scaling["rope_type"] = rope_type
            rope_scaling["type"] = rope_type
            if rope_type == "linear":
                rope_scaling.setdefault("factor", 1.0)
            nested_config.rope_scaling = rope_scaling


def _set_missing_config_value(config, name: str, value: Optional[int]) -> None:
    if config is None or value is None:
        return
    if getattr(config, name, None) is None:
        setattr(config, name, int(value))


def _ensure_qwen3omni_special_token_ids(config, model_dir: str) -> None:
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    except Exception:
        tokenizer = None

    bos_token_id = getattr(tokenizer, "bos_token_id", None) if tokenizer is not None else None
    eos_token_id = getattr(tokenizer, "eos_token_id", None) if tokenizer is not None else None
    pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None

    if bos_token_id is None:
        bos_token_id = getattr(config, "bos_token_id", None)
    if bos_token_id is None:
        bos_token_id = getattr(config, "im_start_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(config, "eos_token_id", None)
    if eos_token_id is None:
        eos_token_id = getattr(config, "im_end_token_id", None)
    if pad_token_id is None:
        pad_token_id = getattr(config, "pad_token_id", None)
    if pad_token_id is None:
        pad_token_id = eos_token_id

    for nested_config in _iter_qwen3omni_nested_configs(config):
        _set_missing_config_value(nested_config, "bos_token_id", bos_token_id)
        _set_missing_config_value(nested_config, "eos_token_id", eos_token_id)
        _set_missing_config_value(nested_config, "pad_token_id", pad_token_id)


def _resolve_accept_hidden_layer(config: Qwen3OmniMoeConvertConfig, thinker) -> Optional[int]:
    accept_hidden_layer = getattr(config, "accept_hidden_layer", None)
    if accept_hidden_layer is not None:
        return int(accept_hidden_layer)

    thinker_config = getattr(thinker, "config", None)
    accept_hidden_layer = getattr(thinker_config, "accept_hidden_layer", None)
    if accept_hidden_layer is not None:
        return int(accept_hidden_layer)

    talker_config = getattr(thinker_config, "talker_config", None)
    accept_hidden_layer = getattr(talker_config, "accept_hidden_layer", None)
    if accept_hidden_layer is not None:
        return int(accept_hidden_layer)

    return None


def _install_qwen3omni_thinker_auto_class_compat() -> None:
    try:
        AutoConfig.register("qwen3_omni_moe_thinker", Qwen3OmniMoeThinkerConfig, exist_ok=True)
    except TypeError:
        try:
            AutoConfig.register("qwen3_omni_moe_thinker", Qwen3OmniMoeThinkerConfig)
        except ValueError:
            pass
    except ValueError:
        pass

    for auto_model_cls in (AutoModel, AutoModelForCausalLM):
        try:
            auto_model_cls.register(Qwen3OmniMoeThinkerConfig, Qwen3OmniMoeThinkerForConditionalGeneration, exist_ok=True)
        except TypeError:
            try:
                auto_model_cls.register(Qwen3OmniMoeThinkerConfig, Qwen3OmniMoeThinkerForConditionalGeneration)
            except ValueError:
                pass
        except ValueError:
            pass


class Qwen3OmniMoeConverterXH2a(HFTransfromersConverter):
    """Qwen3Omni converter (thinker text path) for XH2a HMONNX export via FX."""

    target_device = DeviceType.XH2a

    def __init__(self, config: Qwen3OmniMoeConvertConfig):
        super().__init__()
        self.config = config

    @staticmethod
    def _is_gptqmodel_checkpoint(hf_model_dir: str) -> bool:
        cfg_path = Path(hf_model_dir) / "config.json"
        if not cfg_path.exists():
            return False
        try:
            with open(cfg_path, encoding="utf-8") as f:
                cfg = json.load(f)
            quant_config = cfg.get("quantization_config", {})
            if not isinstance(quant_config, dict):
                return False
            quant_method = str(quant_config.get("quant_method", "")).lower()
            checkpoint_format = str(quant_config.get("checkpoint_format", "")).lower()
            return quant_method == "gptq" or checkpoint_format.startswith("gptq")
        except Exception:
            return False

    @staticmethod
    def _read_config_payload(hf_model_dir: str) -> Dict[str, Any]:
        cfg_path = Path(hf_model_dir) / "config.json"
        with open(cfg_path, encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_model_from_pretrained(model_cls, hf_model_dir: str, **kwargs):
        model_kwargs = dict(kwargs)
        config = model_kwargs.pop("config", None)
        if config is None:
            config = model_cls.config_class.from_pretrained(
                hf_model_dir,
                trust_remote_code=model_kwargs.get("trust_remote_code", True),
            )
        _ensure_qwen3omni_special_token_ids(config, hf_model_dir)
        _normalize_text_rope_scaling(config)
        return model_cls.from_pretrained(hf_model_dir, config=config, **model_kwargs)

    @staticmethod
    def _unpack_gptq_weight(
        qweight: torch.Tensor,
        qzeros: torch.Tensor,
        scales: torch.Tensor,
        g_idx: torch.Tensor,
        bits: int = 4,
        raw_checkpoint_qzeros: bool = False,
    ) -> torch.Tensor:
        pack_factor = 32 // bits
        maxq = (2**bits) - 1
        wf = torch.arange(0, 32, bits, device=qweight.device, dtype=torch.int32)
        zeros = torch.bitwise_and(
            torch.bitwise_right_shift(qzeros.unsqueeze(2).expand(-1, -1, pack_factor), wf.view(1, 1, -1)),
            maxq,
        ).reshape(scales.shape)
        if raw_checkpoint_qzeros:
            zeros = zeros + 1
        weight = torch.bitwise_and(
            torch.bitwise_right_shift(qweight.unsqueeze(1).expand(-1, pack_factor, -1), wf.view(1, -1, 1)),
            maxq,
        )
        weight = weight.reshape(weight.shape[0] * weight.shape[1], weight.shape[2])
        quant_weight = weight - zeros[g_idx.long()]
        weight = scales[g_idx.long()] * quant_weight
        return weight.t().contiguous()

    @staticmethod
    def _extract_thinker_model(native_model: nn.Module) -> nn.Module:
        return native_model.thinker if hasattr(native_model, "thinker") else native_model

    @classmethod
    def _hydrate_packed_moe_experts_from_gptq_checkpoint(cls, native_model: nn.Module, hf_model_dir: str) -> None:
        logger = get_root_logger()
        index_path = Path(hf_model_dir) / "model.safetensors.index.json"
        if not index_path.exists():
            logger.warning(f"{index_path} not exists, skip packed MoE expert hydration")
            return

        thinker = cls._extract_thinker_model(native_model)
        text_model = getattr(thinker, "model", None)
        if text_model is None or not hasattr(text_model, "layers"):
            return

        first_experts = getattr(text_model.layers[0].mlp, "experts", None)
        if not (hasattr(first_experts, "gate_up_proj") and hasattr(first_experts, "down_proj")):
            return

        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})

        key_re = re.compile(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.qweight$")
        prefixes: List[Tuple[int, int, str, str]] = []
        for key, shard_name in weight_map.items():
            match = key_re.match(key)
            if match is None:
                continue
            layer_idx = int(match.group(1))
            expert_idx = int(match.group(2))
            proj_name = match.group(3)
            prefix = key[: -len(".qweight")]
            prefixes.append((layer_idx, expert_idx, proj_name, prefix))

        if not prefixes:
            return

        logger.info("Hydrating packed Qwen3-Omni thinker experts from defused GPTQ tensors")
        suffixes = ("qweight", "qzeros", "scales", "g_idx")
        layer_seen = set()
        with torch.no_grad():
            with ExitStack() as stack:
                shard_handles = {}

                def get_tensor(tensor_name: str) -> torch.Tensor:
                    shard_name = weight_map.get(tensor_name)
                    if shard_name is None:
                        raise KeyError(f"{tensor_name} not found in {index_path}")
                    if shard_name not in shard_handles:
                        shard_handles[shard_name] = stack.enter_context(
                            safe_open(Path(hf_model_dir) / shard_name, framework="pt", device="cpu")
                        )
                    return shard_handles[shard_name].get_tensor(tensor_name)

                for idx, (layer_idx, expert_idx, proj_name, prefix) in enumerate(sorted(prefixes)):
                    tensors = {suffix: get_tensor(f"{prefix}.{suffix}") for suffix in suffixes}
                    weight = cls._unpack_gptq_weight(
                        tensors["qweight"],
                        tensors["qzeros"],
                        tensors["scales"],
                        tensors["g_idx"],
                        raw_checkpoint_qzeros=True,
                    )
                    experts = text_model.layers[layer_idx].mlp.experts
                    target_dtype = experts.gate_up_proj.dtype
                    target_device = experts.gate_up_proj.device
                    intermediate_dim = getattr(experts, "intermediate_dim", experts.gate_up_proj.shape[1] // 2)
                    weight = weight.to(device=target_device, dtype=target_dtype)
                    if proj_name == "gate_proj":
                        experts.gate_up_proj[expert_idx, :intermediate_dim, :].copy_(weight)
                    elif proj_name == "up_proj":
                        experts.gate_up_proj[expert_idx, intermediate_dim:, :].copy_(weight)
                    else:
                        experts.down_proj[expert_idx].copy_(weight)
                    layer_seen.add(layer_idx)
                    if (idx + 1) % 1024 == 0:
                        logger.info(f"Hydrated {idx + 1}/{len(prefixes)} packed Qwen3-Omni GPTQ tensors")
        logger.info(f"Hydrated packed Qwen3-Omni experts for {len(layer_seen)} layers ({len(prefixes)} tensors)")

    def dequantize_hf_model(self, native_hf_model: nn.Module) -> nn.Module:
        hf_model = native_hf_model
        quantization_config = getattr(hf_model.config, "quantization_config", None)
        if quantization_config is None:
            return hf_model

        quant_method = getattr(quantization_config, "quant_method", None)
        if quant_method is None and isinstance(quantization_config, dict):
            quant_method = quantization_config.get("quant_method")
        if isinstance(quant_method, str):
            quant_method = quant_method.lower()

        if quant_method in (QuantizationMethod.AWQ, "awq"):
            hf_model = self._dequantize_awq_hf_model(hf_model)
        elif quant_method in (QuantizationMethod.GPTQ, "gptq"):
            hf_model = self._dequantize_gptq_hf_model(hf_model)
        elif quant_method in (QuantizationMethod.COMPRESSED_TENSORS, "compressed-tensors"):
            hf_model = self._dequantize_compressed_tensors_hf_model(hf_model)
        else:
            raise Exception(f"Unsupported quantization method: {quant_method}")
        return hf_model

    def load_hf_model(self, hf_model_dir: str, **kwargs):
        logger = get_root_logger()
        config_payload = self._read_config_payload(hf_model_dir)
        is_qwen3moe_compat_view = bool(config_payload.get("xh_qwen3omni_thinker_qwen3moe_compat", False))
        architectures = list(config_payload.get("architectures", []) or [])
        model_cls = Qwen3OmniMoeThinkerForConditionalGeneration
        if is_qwen3moe_compat_view:
            model_cls = Qwen3MoeForCausalLM
        elif "Qwen3OmniMoeThinkerForConditionalGeneration" not in architectures:
            model_cls = Qwen3OmniMoeForConditionalGeneration

        if model_cls is Qwen3OmniMoeThinkerForConditionalGeneration:
            _install_qwen3omni_thinker_auto_class_compat()

        if self._is_gptqmodel_checkpoint(hf_model_dir):
            try:
                from gptqmodel import BACKEND, GPTQModel  # type: ignore
                from gptqmodel.models.base import BaseQModel  # type: ignore

                logger.info(f"Detected GPTQModel checkpoint; using GPTQModel.load(): {hf_model_dir}")
                torch_dtype = kwargs.get("torch_dtype", torch.float16)
                native_qmodel = GPTQModel.load(
                    hf_model_dir,
                    device="cpu",
                    dtype=torch_dtype,
                    backend=BACKEND.TORCH,
                    trust_remote_code=kwargs.get("trust_remote_code", True),
                )
                model = native_qmodel.model if isinstance(native_qmodel, BaseQModel) else native_qmodel
                self._hydrate_packed_moe_experts_from_gptq_checkpoint(model, hf_model_dir)
                model = self._dequantize_gptq_hf_model(model)
                if hasattr(model, "config"):
                    _normalize_text_rope_scaling(model.config)
                    model.config.quantization_config = None
            except ImportError:
                logger.warning("gptqmodel not available; falling back to direct HF load")
                if is_qwen3moe_compat_view:
                    model = AutoModelForCausalLM.from_pretrained(hf_model_dir, **kwargs)
                else:
                    model = self._load_model_from_pretrained(model_cls, hf_model_dir, **kwargs)
                model = self.dequantize_hf_model(model)
            except Exception as exc:
                logger.warning(f"GPTQModel.load failed for {hf_model_dir}, fallback to direct HF load: {exc}")
                if is_qwen3moe_compat_view:
                    model = AutoModelForCausalLM.from_pretrained(hf_model_dir, **kwargs)
                else:
                    model = self._load_model_from_pretrained(model_cls, hf_model_dir, **kwargs)
                model = self.dequantize_hf_model(model)
        else:
            if is_qwen3moe_compat_view:
                model = AutoModelForCausalLM.from_pretrained(hf_model_dir, **kwargs)
            else:
                model = self._load_model_from_pretrained(model_cls, hf_model_dir, **kwargs)
            model = self.dequantize_hf_model(model)

        assert isinstance(
            model,
            (Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3MoeForCausalLM),
        ), f"The model is not a supported Qwen3Omni model, but {type(model)}"
        model.eval()
        return model

    def _convert(self, hf_model_path: str, output_dir: str):
        logger = get_root_logger()
        config = self.config

        # ---- 1. Load model ----
        native_model = self.load_hf_model(
            hf_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
            attn_implementation="eager",
        )

        # Extract components for later use
        thinker = self._extract_thinker_model(native_model)
        audio_tower = thinker.audio_tower if hasattr(thinker, 'audio_tower') else None
        visual = thinker.visual if hasattr(thinker, 'visual') else None
        talker = native_model.talker if hasattr(native_model, 'talker') else None
        token2wav = native_model.token2wav if hasattr(native_model, 'token2wav') else None

        # Keep the component references for export, but release any stale CUDA cache first.
        _release_cuda_memory(logger, "model load")

        # Load quantisation weights if available
        resume_from = config.quant_weight
        if resume_from is not None:
            self.load_quant_weight(resume_from, thinker)

        lm_head = thinker.lm_head
        if not hasattr(lm_head, "quant_weight"):
            config.quant_scheme.nodes["lm_head"] = "w8a8h1_sefp"

        model_name = Path(hf_model_path).name
        target_device = config.quant_scheme.target_device
        batch_size = config.batch_size
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length
        assert target_device == DeviceType.XH2a, f"Only support XH2a, got {target_device}"
        quant_type = config.quant_scheme.quant_type
        quant_config = create_quant_config(config.quant_scheme)
        quant_config = ConfigDict(quant_config)

        # ---- 2. Metadata skeleton ----
        meta_info: Dict[str, Any] = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            device=str(target_device),
            model_name=model_name,
            hf_model_path=hf_model_path,
            quant_scheme=config.quant_scheme.to_dict(),
            quant_weight=resume_from,
        )
        if self._is_gptqmodel_checkpoint(hf_model_path):
            meta_info["gptq_expert_qzeros_normalized"] = True

        work_dir = Path(output_dir)

        # Copy HF config files
        hf_config_dir = work_dir / "hf_config"
        hf_config_dir.mkdir(exist_ok=True, parents=True)
        for cfg_file in [
            "config.json",
            "generation_config.json",
            "tokenizer_config.json",
            "vocab.json",
            "tokenizer.json",
        ]:
            src = Path(hf_model_path) / cfg_file
            if src.exists():
                shutil.copyfile(src, hf_config_dir / cfg_file)
            else:
                logger.warning(f"{src} not exists, skip copy")
        meta_info["hf_config"] = str(hf_config_dir.relative_to(work_dir))

        # Save token embedding
        token_embedding = thinker.model.get_input_embeddings()
        token_embedding_file = work_dir / "token_embedding.pt"
        torch.save(token_embedding.state_dict(), str(token_embedding_file))
        meta_info["token_embedding_file"] = str(token_embedding_file.relative_to(work_dir))

        # ---- 3. Register wrapper modules (side-effect import) ----
        from ._text_model import register_wrap_modules as text_register_wrap_modules
        text_register_wrap_modules()

        # ---- 4. Wrap model for FX tracing ----
        accept_hidden_layer = _resolve_accept_hidden_layer(config, thinker)
        if accept_hidden_layer is None:
            raise ValueError("Qwen3-Omni text export requires accept_hidden_layer")
        wrap_cfg_dict = dict(
            batch_size=batch_size,
            max_sequence_length=context_length,
            input_sequence_length=input_sequence_length,
            use_cache=True,
            num_logits_to_keep=config.num_logits_to_keep,
            kv_cache=dict(cache_axis=2),
            accept_hidden_layer=accept_hidden_layer,
            use_multimodal_position_ids=config.use_multimodal_position_ids,
            prefill_full_accept_hidden=config.prefill_full_accept_hidden,
        )
        meta_info["accept_hidden_layer"] = accept_hidden_layer
        meta_info["hidden_states_output_contract"] = "accept_hidden_layer_pre_norm"
        meta_info["supports_multimodal_position_ids"] = bool(config.use_multimodal_position_ids)
        meta_info["position_ids_contract"] = "qwen3_omni_get_rope_index_t_h_w"
        meta_info["prefill_hidden_states_contract"] = (
            "accept_hidden_layer_full_prompt_pre_norm"
            if config.prefill_full_accept_hidden
            else "accept_hidden_layer_last_token_pre_norm"
        )
        meta_info["decode_hidden_states_contract"] = "accept_hidden_layer_single_token_pre_norm"
        wrap_cfg = Config(wrap_cfg_dict)
        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        wrapped_model = wrap_llm_model(thinker, wrap_cfg)

        # ---- 5. Setup KV cache ----
        num_hidden_layers = wrapped_model.model.config.num_hidden_layers
        head_dim = wrapped_model.model.layers[0].self_attn.head_dim
        num_key_value_heads = wrapped_model.model.config.num_key_value_heads
        num_decoder_layers = num_hidden_layers

        kv_cache_shape = [1, num_key_value_heads, context_length, head_dim]
        meta_info["kv_cache"] = dict(
            shape=kv_cache_shape,
            num_decoder_layers=num_decoder_layers,
        )

        past_key_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            for _ in range(num_decoder_layers)
        ]
        past_value_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            for _ in range(num_decoder_layers)
        ]

        # ---- 6. Prepare prefill inputs ----
        input_ids_t = torch.randint(0, 1000, (1, input_sequence_length), dtype=torch.long)
        inputs_embeds = token_embedding(input_ids_t)
        position_ids_t = torch.arange(input_sequence_length, dtype=torch.long)
        time_position_ids_t = position_ids_t.clone()
        height_position_ids_t = position_ids_t.clone()
        width_position_ids_t = position_ids_t.clone()
        deepstack_visual_embed_0 = torch.zeros_like(inputs_embeds)
        deepstack_visual_embed_1 = torch.zeros_like(inputs_embeds)
        deepstack_visual_embed_2 = torch.zeros_like(inputs_embeds)

        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor([input_sequence_length], dtype=torch.int32)

        inputs = (
            inputs_embeds,
            time_position_ids_t,
            height_position_ids_t,
            width_position_ids_t,
            past_seq_length_t,
            current_input_length_t,
            deepstack_visual_embed_0,
            deepstack_visual_embed_1,
            deepstack_visual_embed_2,
            past_key_caches,
            past_value_caches,
        )

        input_names = [
            "inputs_embeds",
            "time_position_ids",
            "height_position_ids",
            "width_position_ids",
            "past_seq_length",
            "current_input_length",
            "deepstack_visual_embed_0",
            "deepstack_visual_embed_1",
            "deepstack_visual_embed_2",
        ]
        for i in range(num_decoder_layers):
            input_names.append(f"past_key_cache_{i}")
        for i in range(num_decoder_layers):
            input_names.append(f"past_value_cache_{i}")
        output_names = ["logits", "hidden_states"]
        meta_info["artifact_contract_version"] = 3
        meta_info["output_names"] = output_names

        prefix = f"{model_name}-{target_device}-{context_length // 1024}k-{quant_type}"

        # ---- 7. Export prefill HMONNX (FX path) ----
        prefill_onnx_file = work_dir / "hmonnx" / "prefill" / f"{prefix}_prefill.onnx"
        prefill_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["prefill_onnx"] = str(prefill_onnx_file.relative_to(work_dir))

        logger.info("Start exporting prefill model (FX path)")
        quanted_model = convert_fx_model_to_quanted_model(
            wrapped_model,
            inputs,
            target_device,
            quant_config=quant_config,
        )

        compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(
            quanted_model, inputs, str(prefill_onnx_file), compatible_names, output_names
        )
        logger.info(f"Export prefill model to {prefill_onnx_file}")

        # ---- 8. Export decode HMONNX ----
        decode_inputs = (
            inputs_embeds[:, :1, :],
            time_position_ids_t[:1],
            height_position_ids_t[:1],
            width_position_ids_t[:1],
            past_seq_length_t,
            torch.ones_like(current_input_length_t),
            deepstack_visual_embed_0[:, :1, :],
            deepstack_visual_embed_1[:, :1, :],
            deepstack_visual_embed_2[:, :1, :],
            past_key_caches,
            past_value_caches,
        )

        wrap_cfg.input_sequence_length = 1
        quanted_model.update_cfg(wrap_cfg)

        decode_onnx_file = work_dir / "hmonnx" / "decode" / f"{prefix}_decoder.onnx"
        decode_onnx_file.parent.mkdir(exist_ok=True, parents=True)
        meta_info["decode_onnx"] = str(decode_onnx_file.relative_to(work_dir))

        logger.info("Start exporting decode model")
        compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
        convert_quanted_model_to_hmonnx(
            quanted_model, decode_inputs, str(decode_onnx_file), compatible_names, output_names
        )
        logger.info(f"Export decode model to {decode_onnx_file}")

        # ---- 9. Export Audio Encoder (if exists) ----
        if config.export_audio_encoder and audio_tower is not None:
            logger.info("Start exporting audio encoder...")
            audio_meta = self._export_audio_encoder(
                audio_tower, work_dir, meta_info, config, quant_config, model_name, target_device, quant_type
            )
            meta_info.update(audio_meta)
            logger.info("Audio encoder export complete")

        # ---- 10. Export Vision Encoder (if exists) ----
        if config.export_vision_encoder and visual is not None:
            logger.info("Start exporting vision encoder...")
            vision_meta = self._export_vision_encoder(
                visual, work_dir, meta_info, config, quant_config, model_name, target_device, quant_type
            )
            meta_info.update(vision_meta)
            logger.info("Vision encoder export complete")

        # ---- 11. Export Talker LM (if exists) ----
        if config.export_talker_model and talker is not None:
            logger.info("Start exporting talker model...")
            talker_meta = self._export_talker_lm(
                talker, work_dir, meta_info, config, quant_config, model_name, target_device, quant_type
            )
            meta_info.update(talker_meta)
            logger.info("Talker model export complete")

            # ---- 12. Export Talker Prediction (if exists) ----
            code_predictor = getattr(talker, "code_predictor", None)
            if config.export_talker_prediction and code_predictor is not None:
                logger.info("Start exporting talker prediction model...")
                talker_pred_meta = self._export_talker_prediction(
                    code_predictor,
                    work_dir,
                    meta_info,
                    config,
                    quant_config,
                    model_name,
                    target_device,
                    quant_type,
                )
                meta_info.update(talker_pred_meta)
                logger.info("Talker prediction export complete")

        # ---- 13. Save metadata ----
        with open(work_dir / "meta.json", "w") as f:
            json.dump(meta_info, f, indent=4)
        logger.info(f"Conversion complete. Artifacts in {work_dir}")

        lm_head = None
        token_embedding = None
        quanted_model = None
        wrapped_model = None
        past_key_caches = None
        past_value_caches = None
        inputs_embeds = None
        deepstack_visual_embed_0 = None
        deepstack_visual_embed_1 = None
        deepstack_visual_embed_2 = None
        token2wav = None
        talker = None
        visual = None
        audio_tower = None
        thinker = None
        native_model = None
        _release_cuda_memory(logger, "converter export")

    def _export_audio_encoder(self, audio_tower, work_dir, meta_info, config, quant_config, 
                             model_name, target_device, quant_type):
        """Export audio encoder module."""
        logger = get_root_logger()
        from ._audio_model import register_wrap_modules as audio_register_wrap_modules
        
        audio_register_wrap_modules()
        
        # Prepare wrapped audio encoder for export to avoid raw HF forward Python control-flow.
        audio_tower = audio_tower.to(torch.float16).cpu()
        wrapped_audio = wrap_llm_model(audio_tower, Config(dict()))

        # After FX trace, only padded_feature and cu_seqlens have graph users
        # (padded_mask_after_cnn is declared in forward but unused)
        batch_size = 1
        mel_bins = int(getattr(audio_tower.config, "num_mel_bins", 128))
        mel_length = 100
        cnn_steps = 13
        dummy_feature = torch.randn(batch_size, mel_bins, mel_length, dtype=torch.float16)
        dummy_cu = torch.tensor([0, cnn_steps], dtype=torch.int32)

        inputs = (dummy_feature, dummy_cu)
        input_names = ["padded_feature", "cu_seqlens"]
        output_names = ["audio_embeds"]
        
        # Export audio encoder HMONNX
        audio_dir = work_dir / "hmonnx" / "audio"
        audio_dir.mkdir(exist_ok=True, parents=True)
        audio_onnx_file = audio_dir / f"{model_name}-{target_device}-audio_encoder.onnx"
        
        try:
            logger.info(f"Exporting audio encoder to {audio_onnx_file}")
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_fx_model_to_hmonnx(
                wrapped_audio,
                inputs,
                target_device,
                audio_onnx_file,
                input_names=compatible_names,
                output_names=output_names,
            )
            logger.info(f"Audio encoder export successful: {audio_onnx_file}")
            return {
                "audio_encoder_onnx": str(audio_onnx_file.relative_to(work_dir)),
                "audio_mel_dim": mel_bins,
                "audio_max_length": mel_length,
                "audio_batch_size": batch_size,
            }
        except Exception as e:
            logger.warning(f"Audio encoder export failed: {e}")
            raise RuntimeError(f"Audio encoder export failed: {e}") from e

    def _export_vision_encoder(self, visual, work_dir, meta_info, config, quant_config,
                              model_name, target_device, quant_type):
        """Export vision encoder module."""
        logger = get_root_logger()
        from ._vision_model import register_wrap_modules as vision_register_wrap_modules
        
        vision_register_wrap_modules()
        
        # Prepare wrapped vision encoder for export to align with existing XH wrapper path.
        visual = visual.to(torch.float16).cpu()
        vision_wrap_cfg = Config(
            dict(
                max_size_w=224,
                max_size_h=224,
                max_size_t=2,
                temporal_patch_size=2,
                patch_size=16,
                only_first_block=False,
            )
        )
        wrapped_visual = wrap_llm_model(visual, vision_wrap_cfg)

        patch_size = 16
        channels = 3
        height = 224
        width = 224
        frames = 2
        dummy_pixels = torch.randn(1, channels, frames, height, width, dtype=torch.float16)

        inputs = (dummy_pixels,)
        input_names = ["pixel_values"]
        output_names = ["vision_embeds", "deepstack_0", "deepstack_1", "deepstack_2"]
        
        # Export vision encoder HMONNX
        vision_dir = work_dir / "hmonnx" / "vision"
        vision_dir.mkdir(exist_ok=True, parents=True)
        vision_onnx_file = vision_dir / f"{model_name}-{target_device}-vision_encoder.onnx"
        
        try:
            logger.info(f"Exporting vision encoder to {vision_onnx_file}")
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_fx_model_to_hmonnx(
                wrapped_visual,
                inputs,
                target_device,
                vision_onnx_file,
                input_names=compatible_names,
                output_names=output_names,
            )
            logger.info(f"Vision encoder export successful: {vision_onnx_file}")
            return {
                "vision_encoder_onnx": str(vision_onnx_file.relative_to(work_dir)),
                "vision_patch_size": patch_size,
                "vision_input_size": [height, width],
                "vision_channels": channels,
            }
        except Exception as e:
            logger.warning(f"Vision encoder export failed: {e}")
            raise RuntimeError(f"Vision encoder export failed: {e}") from e

    def _export_talker_lm(self, talker, work_dir, meta_info, config, quant_config,
                         model_name, target_device, quant_type):
        """Export talker LM module (similar to thinker but for audio code generation)."""
        logger = get_root_logger()
        from ._talker_model import register_wrap_modules as talker_register_wrap_modules
        
        talker_register_wrap_modules()
        
        # Prepare talker for export
        talker = talker.to(torch.float16).cpu()
        
        batch_size = config.batch_size
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length
        
        # Get embedding from talker
        talker_embedding = talker.model.get_input_embeddings()
        
        wrap_cfg = Config(
            dict(
                batch_size=batch_size,
                max_sequence_length=context_length,
                input_sequence_length=input_sequence_length,
                use_cache=True,
                num_logits_to_keep=config.num_logits_to_keep,
                kv_cache=dict(cache_axis=2),
            )
        )
        
        wrapped_talker = wrap_llm_model(talker, wrap_cfg)
        
        # Setup KV cache
        num_hidden_layers = wrapped_talker.model.config.num_hidden_layers
        head_dim = wrapped_talker.model.layers[0].self_attn.head_dim
        num_key_value_heads = wrapped_talker.model.config.num_key_value_heads
        
        kv_cache_shape = [1, num_key_value_heads, context_length, head_dim]
        past_key_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            for _ in range(num_hidden_layers)
        ]
        past_value_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            for _ in range(num_hidden_layers)
        ]
        
        # Prepare prefill inputs
        input_ids_t = torch.randint(0, 1000, (1, input_sequence_length), dtype=torch.long)
        inputs_embeds = talker_embedding(input_ids_t)
        
        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor([input_sequence_length], dtype=torch.int32)
        
        inputs = (
            inputs_embeds,
            past_seq_length_t,
            current_input_length_t,
            past_key_caches,
            past_value_caches,
        )
        
        input_names = ["inputs_embeds", "past_seq_length", "current_input_length"]
        for i in range(num_hidden_layers):
            input_names.append(f"past_key_cache_{i}")
        for i in range(num_hidden_layers):
            input_names.append(f"past_value_cache_{i}")
        output_names = ["logits"]
        
        talker_meta = {}
        
        try:
            # Export talker prefill
            talker_dir = work_dir / "hmonnx" / "talker"
            talker_dir.mkdir(exist_ok=True, parents=True)
            talker_prefill_file = talker_dir / f"{model_name}-{target_device}-talker_prefill.onnx"
            
            logger.info(f"Exporting talker prefill to {talker_prefill_file}")
            quanted_talker = convert_fx_model_to_quanted_model(
                wrapped_talker,
                inputs,
                target_device,
                quant_config=quant_config,
            )
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(
                quanted_talker, inputs, str(talker_prefill_file), compatible_names, output_names
            )
            logger.info(f"Talker prefill export successful: {talker_prefill_file}")
            talker_meta["talker_prefill_onnx"] = str(talker_prefill_file.relative_to(work_dir))
            
            # Export talker decode
            decode_inputs = (
                inputs_embeds[:, :1, :],
                past_seq_length_t,
                torch.ones_like(current_input_length_t),
                past_key_caches,
                past_value_caches,
            )
            wrap_cfg.input_sequence_length = 1
            quanted_talker.update_cfg(wrap_cfg)
            
            talker_decode_file = talker_dir / f"{model_name}-{target_device}-talker_decode.onnx"
            logger.info(f"Exporting talker decode to {talker_decode_file}")
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(
                quanted_talker, decode_inputs, str(talker_decode_file), compatible_names, output_names
            )
            logger.info(f"Talker decode export successful: {talker_decode_file}")
            talker_meta["talker_decode_onnx"] = str(talker_decode_file.relative_to(work_dir))
            talker_meta["talker_kv_cache"] = {
                "shape": kv_cache_shape,
                "num_decoder_layers": num_hidden_layers,
            }
            talker_meta["talker_hidden_size"] = int(inputs_embeds.shape[-1])
            talker_meta["talker_input_sequence_length"] = int(input_sequence_length)
            
        except Exception as e:
            logger.warning(f"Talker export failed: {e}")
            raise RuntimeError(f"Talker export failed: {e}") from e

        return talker_meta

    def _export_talker_prediction(
        self,
        talker_prediction,
        work_dir,
        meta_info,
        config,
        quant_config,
        model_name,
        target_device,
        quant_type,
    ):
        """Export talker code-predictor module (prefill/decode)."""
        logger = get_root_logger()
        from ._talker_prediction import register_wrap_modules as talker_prediction_register_wrap_modules

        talker_prediction_register_wrap_modules()

        talker_prediction = talker_prediction.to(torch.float16).cpu()

        batch_size = config.batch_size
        context_length = config.context_length
        input_sequence_length = config.input_sequence_length

        codec_embedding = talker_prediction.model.get_input_embeddings()
        if isinstance(codec_embedding, (list, tuple, nn.ModuleList)):
            codec_embedding = codec_embedding[0]

        wrap_cfg = Config(
            dict(
                batch_size=batch_size,
                max_sequence_length=context_length,
                input_sequence_length=input_sequence_length,
                use_cache=True,
                num_logits_to_keep=config.num_logits_to_keep,
                kv_cache=dict(cache_axis=2),
            )
        )

        wrapped_model = wrap_llm_model(talker_prediction, wrap_cfg)

        num_hidden_layers = wrapped_model.model.config.num_hidden_layers
        head_dim = wrapped_model.model.layers[0].self_attn.head_dim
        num_key_value_heads = wrapped_model.model.config.num_key_value_heads

        kv_cache_shape = [1, num_key_value_heads, context_length, head_dim]
        past_key_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            for _ in range(num_hidden_layers)
        ]
        past_value_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            for _ in range(num_hidden_layers)
        ]

        input_ids_t = torch.randint(0, 1000, (1, input_sequence_length), dtype=torch.long)
        inputs_embeds = codec_embedding(input_ids_t)
        past_seq_length_t = torch.tensor([0], dtype=torch.int32)
        current_input_length_t = torch.tensor([input_sequence_length], dtype=torch.int32)

        inputs = (
            inputs_embeds,
            past_seq_length_t,
            current_input_length_t,
            past_key_caches,
            past_value_caches,
        )

        input_names = ["inputs_embeds", "past_seq_length", "current_input_length"]
        for i in range(num_hidden_layers):
            input_names.append(f"past_key_cache_{i}")
        for i in range(num_hidden_layers):
            input_names.append(f"past_value_cache_{i}")
        output_names = ["logits"]

        meta = {}
        try:
            out_dir = work_dir / "hmonnx" / "talker_prediction"
            out_dir.mkdir(exist_ok=True, parents=True)

            prefill_file = out_dir / f"{model_name}-{target_device}-talker_prediction_prefill.onnx"
            logger.info(f"Exporting talker prediction prefill to {prefill_file}")

            quanted_model = convert_fx_model_to_quanted_model(
                wrapped_model,
                inputs,
                target_device,
                quant_config=quant_config,
            )
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(
                quanted_model,
                inputs,
                str(prefill_file),
                compatible_names,
                output_names,
            )

            decode_inputs = (
                inputs_embeds[:, :1, :],
                past_seq_length_t,
                torch.ones_like(current_input_length_t),
                past_key_caches,
                past_value_caches,
            )
            wrap_cfg.input_sequence_length = 1
            quanted_model.update_cfg(wrap_cfg)

            decode_file = out_dir / f"{model_name}-{target_device}-talker_prediction_decode.onnx"
            logger.info(f"Exporting talker prediction decode to {decode_file}")
            compatible_names = BaseConverter.xh1_hmonnx_compatible(input_names)
            convert_quanted_model_to_hmonnx(
                quanted_model,
                decode_inputs,
                str(decode_file),
                compatible_names,
                output_names,
            )

            meta["talker_prediction_prefill_onnx"] = str(prefill_file.relative_to(work_dir))
            meta["talker_prediction_decode_onnx"] = str(decode_file.relative_to(work_dir))
            meta["talker_prediction_kv_cache"] = {
                "shape": kv_cache_shape,
                "num_decoder_layers": num_hidden_layers,
            }
            meta["talker_prediction_hidden_size"] = int(inputs_embeds.shape[-1])
            meta["talker_prediction_input_sequence_length"] = int(input_sequence_length)
        except Exception as e:
            logger.warning(f"Talker prediction export failed: {e}")
            raise RuntimeError(f"Talker prediction export failed: {e}") from e

        return meta

    @classmethod
    def convert(cls, hf_model_path: str, config: Qwen3OmniMoeConvertConfig, output_dir: str):
        cls(config)._convert(hf_model_path, output_dir)
