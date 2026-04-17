from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Union, cast

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration as XHGemma4ForConditionalGeneration

from xhquant.utils import get_xhquant_logger, log_function_call
from xhquant.utils.registry import DynamicModule, _DMRegistryCls

from ...builder import register_llm_model
from ...kv_cache_mixin import KVCacheMixin
from ...text_llm_hf_compatible import TextLLMHFCompatible
from ...types import ExportData, KVCacheConfig, LLMModelState, ModelSwitcher, VLLMModelMeta
from ...vision_llm_model import VisionLLMModel
from .data_preprocess import Gemma4DataPreprocess
from .gemma4_hmonnx_inference import XHGemma4HMONNXModel
from .gemma4_visual_model import XHGemma4VisionModel
from .xh_gemma4_config import XHGemma4ModelConfig


def _copy_model_shared_params(model: nn.Module) -> nn.Module:
    """Deep-copy model structure; all parameters and buffers share data with the original (zero extra VRAM)."""
    memo: dict[int, Any] = {}
    for param in model.parameters():
        if id(param) not in memo:
            memo[id(param)] = nn.Parameter(param.data, requires_grad=param.requires_grad)
    for buf in model.buffers():
        if id(buf) not in memo:
            memo[id(buf)] = buf
    return copy.deepcopy(model, memo)


class Gemma4KVCacheMixin(KVCacheMixin):
    def __init__(self, kv_cache_config: KVCacheConfig):
        super().__init__(kv_cache_config)
        self.layer_kv_shapes: list[list[int]] = []

    def set_layer_kv_shapes(self, layer_kv_shapes: list[list[int]]):
        self.layer_kv_shapes = layer_kv_shapes
        self.kvcache_config.num_layers = len(layer_kv_shapes)
        if layer_kv_shapes:
            self.kvcache_config.kv_cache_shape = layer_kv_shapes[0]

    def prepare_kv_cache(self, dtype=torch.float16):
        if not self.use_cache:
            return
        self.past_key_caches.clear()
        self.past_value_caches.clear()
        for shape in self.layer_kv_shapes:
            self.past_key_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(shape, dtype=dtype)))
            self.past_value_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(shape, dtype=dtype)))


class _Gemma4HFCompatible(TextLLMHFCompatible):
    """HF-compatible wrapper for Gemma4. Redirects forward() to our HMONNX/quanted model
    while keeping the HF generate() loop happy."""

    def _setup(self: XHGemma4ForConditionalGeneration, text_llm_model: "XHGemma4Model"):
        model = super()._setup(text_llm_model)
        if model is not None:
            for attr in ("model",):
                if hasattr(model, attr):
                    delattr(model, attr)
            if hasattr(model, "lm_head"):
                delattr(model, "lm_head")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._gemma4_pixel_values = None
        self._gemma4_image_position_ids = None
        self._gemma4_mm_token_type_ids = None
        return model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Retrieve vision args stored by generate()
        pixel_values = kwargs.pop("pixel_values", None) or self._gemma4_pixel_values
        image_position_ids = kwargs.pop("image_position_ids", None) or self._gemma4_image_position_ids
        mm_token_type_ids = kwargs.pop("mm_token_type_ids", None) or self._gemma4_mm_token_type_ids

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Vision: run visual model on first call (prefill) only
        image_embeds = None
        if pixel_values is not None and image_position_ids is not None:
            # Strip padding patches (position_ids == -1) before feeding to visual
            if image_position_ids.dim() == 2:
                is_real = ~(image_position_ids == -1).all(dim=-1)
            else:
                is_real = ~(image_position_ids == -1).all(dim=-1)[0]  # (total_patches,)
            n_real = int(is_real.sum().item())
            pixel_values = pixel_values[:, :n_real, :]
            image_embeds = self._llm_model.visual.forward(
                pixel_values.to(dtype=self._llm_model.visual.dtype, device=self._llm_model.visual.device),
            )
            if isinstance(image_embeds, (tuple, list)):
                image_embeds = image_embeds[0]
            image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            # Offload visual model to free GPU memory for text prefill
            self._llm_model.visual.to("cpu")
            torch.cuda.empty_cache()
            # Clear so vision is not re-run on decode steps
            self._gemma4_pixel_values = None
            self._gemma4_image_position_ids = None

        seq_length = inputs_embeds.shape[1]
        data_processor: Gemma4DataPreprocess = self._llm_model.get_data_preprocessor()
        device = inputs_embeds.device
        chunk_len = data_processor.input_sequence_length

        if seq_length <= chunk_len:
            # --- Single-chunk path (original) ---
            data_batch = {
                "input_ids": input_ids,
                "image_embeds": image_embeds,
                "past_seq_length": self._past_seq_length,
                "mm_token_type_ids": mm_token_type_ids,
            }
            data_input = data_processor(data_batch)
            (
                inputs_embeds_proc,
                past_seq_length_t,
                current_input_length_t,
                full_attention_mask,
                sliding_attention_mask,
                past_key_caches,
                past_value_caches,
            ) = data_input

            logits = self._llm_model.forward(
                inputs_embeds_proc,
                past_seq_length_t,
                current_input_length_t,
                full_attention_mask,
                sliding_attention_mask,
                *past_key_caches,
                *past_value_caches,
            )
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.dim() == 3 and logits.shape[1] > seq_length:
                logits = logits[:, :seq_length, :]
        else:
            # --- Multi-chunk prefill ---
            # 1. Replace image tokens in the full embedding
            if image_embeds is not None and input_ids is not None:
                image_token_id = data_processor.image_token_id
                n_image_tokens = (input_ids == image_token_id).sum().item()
                if n_image_tokens > 0:
                    image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                    if image_embeds.shape[0] != n_image_tokens:
                        raise ValueError(
                            f"Image features and image tokens do not match: "
                            f"tokens={n_image_tokens}, features={image_embeds.shape[0]}"
                        )
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            # 2. Prepare mm_token_type_ids for the full sequence
            if mm_token_type_ids is None:
                mm_full = torch.zeros(seq_length, dtype=torch.long, device=device)
            else:
                mm_full = mm_token_type_ids.to(device).flatten()[:seq_length]

            # 3. Chunk and process
            steps = (seq_length + chunk_len - 1) // chunk_len
            pad_len = steps * chunk_len - seq_length
            if pad_len > 0:
                pad_embeds = self.get_input_embeddings()(
                    torch.zeros(1, pad_len, dtype=torch.long, device=device)
                )
                inputs_embeds = torch.cat([inputs_embeds, pad_embeds], dim=1)
                mm_full = torch.cat([mm_full, torch.zeros(pad_len, dtype=torch.long, device=device)])

            past_key_caches = data_processor.past_key_caches
            past_value_caches = data_processor.past_value_caches
            running_past_seq = self._past_seq_length

            outputs_logits = []
            for i in range(steps):
                start = i * chunk_len
                end = (i + 1) * chunk_len
                sub_embeds = inputs_embeds[:, start:end, :]
                sub_current_len = min(end, seq_length) - start
                sub_mm = mm_full[start:end]

                full_mask, sliding_mask = data_processor._build_attention_masks(
                    current_input_length=sub_current_len,
                    past_seq_length=running_past_seq,
                    mm_token_type_ids=sub_mm,
                    device=device,
                )

                chunk_logits = self._llm_model.forward(
                    sub_embeds,
                    torch.tensor([running_past_seq], dtype=torch.int32, device=device),
                    torch.tensor([sub_current_len], dtype=torch.int32, device=device),
                    full_mask,
                    sliding_mask,
                    *past_key_caches,
                    *past_value_caches,
                )
                if isinstance(chunk_logits, (tuple, list)):
                    chunk_logits = chunk_logits[0]
                outputs_logits.append(chunk_logits)
                running_past_seq += sub_current_len

            # Use last chunk's logits, trimmed to valid length
            last_valid = min(chunk_len, seq_length - (steps - 1) * chunk_len)
            logits = outputs_logits[-1][:, :last_valid, :]

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )

    def generate(self, *args, **kwargs):
        # Extract Gemma4-specific kwargs before HF's generate validates them
        self._gemma4_pixel_values = kwargs.pop("pixel_values", None)
        self._gemma4_image_position_ids = kwargs.pop("image_position_ids", None)
        self._gemma4_mm_token_type_ids = kwargs.pop("mm_token_type_ids", None)
        return super().generate(*args, **kwargs)

    def set_experts_implementation(self, *args, **kwargs):
        """No-op: Gemma4 has no MoE experts; prevents HF generate crash."""


def build_gemma4_hf_compatible_model(
    hf_model: XHGemma4ForConditionalGeneration,
    xh_model: "XHGemma4Model",
):
    llm_compatible_modules = _DMRegistryCls("XHCompatible")
    hf_model_cls = type(hf_model)
    if hf_model_cls not in llm_compatible_modules:
        llm_compatible_modules.register_module({hf_model_cls: hf_model_cls.__name__}, _Gemma4HFCompatible)
    return llm_compatible_modules.convert(hf_model, text_llm_model=xh_model)


@register_llm_model("Gemma4ForConditionalGeneration")
class XHGemma4Model(VisionLLMModel):  # noqa: N801
    transformers_min_version = "5.5.0"
    HF_MODEL_CLS = XHGemma4ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = VLLMModelMeta
    HMONNXINFERENCE_CLS = XHGemma4HMONNXModel
    CONFIG_CLS = XHGemma4ModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_gemma4_hf_compatible_model)

    def __init__(self, config: XHGemma4ModelConfig):
        super().__init__(config)
        self.visual = XHGemma4VisionModel(config.visual_config)
        self._kvcache_config = KVCacheConfig()
        self._kvcache_mixin = Gemma4KVCacheMixin(self.kvcache_config)
        self.config = cast(XHGemma4ModelConfig, self.config)

    @VisionLLMModel.work_dir.setter
    def work_dir(self, work_dir: str):
        self.config.work_dir = work_dir
        self.visual.work_dir = str(Path(work_dir) / "visual")

    def get_tf_processor(self):
        return self.visual.get_tf_processor()

    def _get_language_model(self, hf_model: Any) -> Any:
        return hf_model.model.language_model

    @classmethod
    def get_hf_model(cls, hf_model_dir: str, quant_weight=None, **kwargs) -> Any:
        kwargs.setdefault("dtype", torch.bfloat16)
        # Load entirely on CPU to avoid accelerate meta tensors and hooks.
        # The model is moved to GPU only when needed (ptq quantization).
        kwargs.setdefault("device_map", "cpu")
        kwargs.setdefault("trust_remote_code", True)
        # Gemma4 is not registered in gptqmodel yet; bypass GPTQModel.load and
        # rely on transformers' native auto-gptq loading + our _dequantize_gptq
        # converter. This produces per-layer `quant_weight` buffers just like
        # the qwen3.5 path.
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
            native_hf_model = cls._dequantize_hf_model(native_hf_model, quant_weight=None, **kwargs)
            return native_hf_model
        return super().get_hf_model(hf_model_dir, quant_weight, **kwargs)

    def _wraped_pre(self, hf_model: XHGemma4ForConditionalGeneration):
        model = hf_model.model
        for name in ["vision_tower", "embed_vision", "audio_tower", "embed_audio"]:
            if hasattr(model, name):
                delattr(model, name)
        return hf_model

    def _wraped_post(self, hf_model: XHGemma4ForConditionalGeneration):
        self.config.image_token_id = getattr(hf_model.config, "image_token_id", None)
        self.config.video_token_id = getattr(hf_model.config, "video_token_id", None)
        self.config.audio_token_id = getattr(hf_model.config, "audio_token_id", None)
        self.config.boi_token_id = getattr(hf_model.config, "boi_token_id", None)
        self.config.eoi_token_id = getattr(hf_model.config, "eoi_token_id", None)

        hf_model = self._wrap_model
        llm_model = self._get_language_model(hf_model)
        # Deep-copy the embedding on CPU to avoid GPU OOM (262k vocab × hidden_dim is large).
        # .cpu() moves the original in-place, so save the device and restore after deepcopy.
        orig_embed = llm_model.get_input_embeddings()
        orig_device = orig_embed.weight.device
        embed_copy = copy.deepcopy(orig_embed.cpu())
        orig_embed.to(orig_device)  # restore original embedding (may be tied with lm_head)
        # Gemma4TextScaledWordEmbedding multiplies by embed_scale during forward().
        # Bake the scale into the weight so quant_embedding.pt works with plain nn.Embedding.
        if hasattr(embed_copy, "embed_scale"):
            embed_copy.weight.data = (embed_copy.weight.float() * embed_copy.embed_scale).to(embed_copy.weight.dtype)
        self.embed_tokens = nn.Embedding(
            embed_copy.num_embeddings, embed_copy.embedding_dim, _weight=embed_copy.weight
        ).to(orig_device)
        self.pad_token_id = int(getattr(llm_model.config, "pad_token_id", 0) or 0)
        self.sliding_window = int(getattr(llm_model.config, "sliding_window", 1024))
        self.layer_types = list(getattr(llm_model.config, "layer_types", []))

        layer_kv_shapes: list[list[int]] = []
        # All layers get the same cache input size (context_max_length).
        # LLMCache with attention_max_length=sliding_window handles output truncation;
        # the compiler reads this marker and optimizes the actual read pattern.
        cache_seq_len = self.config.context_max_length
        for idx, layer in enumerate(llm_model.layers):
            attn = layer.self_attn
            num_key_value_heads = attn.k_proj.out_features // attn.head_dim
            layer_kv_shapes.append([1, num_key_value_heads, cache_seq_len, attn.head_dim])
        self._kvcache_mixin.set_layer_kv_shapes(layer_kv_shapes)

    def init_wrap_model(self, hf_model: XHGemma4ForConditionalGeneration) -> Any:
        from ._llm_model_impl import register_wrap_modules

        register_wrap_modules()
        return super().init_wrap_model(hf_model)

    def _to_fronted(self, wrap_model):
        # Model is on CPU (loaded with device_map="cpu"), no hook removal needed.

        self.set_prefill()
        prefill_wrap_model = wrap_model
        decode_wrap_model = _copy_model_shared_params(wrap_model)
        self._wrap_model = wrap_model
        prefill_frontend_model = super()._to_fronted(prefill_wrap_model)

        self._wrap_model = decode_wrap_model
        self.set_decode()
        decode_frontend_model = super()._to_fronted(decode_wrap_model)
        self._frontend_model = prefill_frontend_model
        self._wrap_model = prefill_frontend_model
        self.set_prefill()
        return ModelSwitcher({"prefill": prefill_frontend_model, "decode": decode_frontend_model})

    def _to_quanted(self, frontend_model, state):
        # With device_map="cpu", both models start on CPU.
        # Move each to GPU one at a time for ptq quantization, then back to CPU.
        decode_fronted_model = frontend_model.decode
        prefill_fronted_model = frontend_model.prefill

        def _detach_quant_weights(m: nn.Module):
            """Temporarily detach int8 quant_weight buffers so .cuda() doesn't move them to GPU."""
            cache = {}
            for name, sub in m.named_modules():
                if isinstance(sub, nn.Linear) and "quant_weight" in sub._buffers:
                    cache[name] = sub._buffers.pop("quant_weight")
            return cache

        def _reattach_quant_weights(m: nn.Module, cache: dict):
            for name, buf in cache.items():
                sub = m.get_submodule(name)
                sub.register_buffer("quant_weight", buf)

        # Quantize prefill on GPU
        self.set_prefill()
        prefill_qw_cache = _detach_quant_weights(prefill_fronted_model)
        prefill_fronted_model.cuda()
        _reattach_quant_weights(prefill_fronted_model, prefill_qw_cache)
        prefill_quanted_model = super()._to_quanted(prefill_fronted_model, state)

        # Move quantized prefill to CPU to free GPU for decode
        prefill_quanted_model.cpu()
        torch.cuda.empty_cache()

        # Quantize decode on GPU
        decode_qw_cache = _detach_quant_weights(decode_fronted_model)
        decode_fronted_model.cuda()
        _reattach_quant_weights(decode_fronted_model, decode_qw_cache)
        torch.cuda.empty_cache()

        self.set_decode()
        decode_quanted_model = super()._to_quanted(decode_fronted_model, state)
        self.set_prefill()
        return ModelSwitcher({"prefill": prefill_quanted_model, "decode": decode_quanted_model})

    def _get_data_preprocessor(self):
        return Gemma4DataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            context_length=self.config.context_max_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            pad_token_id=self.pad_token_id,
            image_token_id=self.config.image_token_id or -1,
            sliding_window=self.sliding_window,
        )

    def get_export_cfg(self) -> dict[str, list[str]]:
        export_cfg = {
            "input_names": [
                "inputs_embeds",
                "past_seq_length",
                "current_input_length",
                "full_attention_mask",
                "sliding_attention_mask",
            ],
            "output_names": ["logits"],
        }
        for layer_idx in range(self.kvcache_config.num_layers):
            export_cfg["input_names"].append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(self.kvcache_config.num_layers):
            export_cfg["input_names"].append(f"past_value_cache_{layer_idx}")
        return export_cfg

    def _extra_export_metadata(self, output_dir: str, meta_info):
        meta_info.layer_types = self.layer_types
        meta_info.layer_kv_shapes = self._kvcache_mixin.layer_kv_shapes
        meta_info.sliding_window = self.sliding_window
        return meta_info

    def get_export_info(self, output_dir) -> ExportData:
        str_datetime = datetime.now().strftime("%Y%m%d")
        model_name = self.config.model_name.lower()
        output_dir = Path(output_dir) / f"hmquant_{model_name}_{str_datetime}"
        output_dir.mkdir(parents=True, exist_ok=True)
        meta_info = self.create_export_metadata(output_dir)
        export_data = ExportData()
        export_data.exported_dir = str(output_dir)
        export_data.meta = meta_info
        export_data.model_name = f"hmquant_{model_name}_{str_datetime}"
        export_data.str_datetime = str_datetime
        return export_data

    @log_function_call()
    def export_hmonnx(self, output_dir: str):
        logger = get_xhquant_logger()
        self.work_dir = str(output_dir)
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.prefill.fixed()
        self._quanted_model.decode.fixed()
        self.visual.quanted_model.fixed()
        exported_info = self.get_export_info(output_dir)
        visual_output_dir = str(Path(exported_info.exported_dir) / "visual")
        visual_meta = self.visual.export_hmonnx(visual_output_dir)
        visual_meta.hmonnx = str(Path(visual_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix())
        meta_info = cast(VLLMModelMeta, exported_info.meta)
        meta_info.visual_config = visual_meta
        self._export_hmonnx(exported_info)
        json.dump(meta_info.to_dict(), open(str(Path(exported_info.exported_dir) / "golden_meta_info.json"), "w"), indent=4)
        logger.info(f"Exporting completed! Exported model is saved at: {exported_info.exported_dir}")
        return meta_info
