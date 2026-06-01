import json
from pathlib import Path
from typing import Any, Optional, cast

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForImageTextToText
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration
from xhquant.utils import get_xhquant_logger, log_function_call
from xhquant.utils.registry import _DMRegistryCls

from ...builder import register_llm_model
from ...kv_cache_mixin import KVCacheMixin
from ...text_llm_hf_compatible import TextLLMHFCompatible
from ...types import ExportData, LLMModelState
from ...vision_llm_model import VisionLLMModel
from .data_preprocess import Gemma4DataPreprocess, Gemma4InputProcessorConfig, Gemma4PerLayerInputBuilder
from .gemma4_audio_model import XHGemma4AudioModel
from .gemma4_hmonnx_inference import XHGemma4_HMONNXModel
from .gemma4_processor import XHGemma4Processor, configure_gemma4_visual_processor
from .gemma4_vision_model import XHGemma4VisionModel
from .xh_gemma4_config import Gemma4ModelMeta, XHGemma4ModelConfig


class Gemma4KVCacheMixin(KVCacheMixin):
    def __init__(self, kv_cache_config):
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


class _Gemma4TextExportBridgeBase(nn.Module):
    def __init__(self, hf_model: Gemma4ForConditionalGeneration):
        super().__init__()
        self.config = hf_model.config
        self.language_model = hf_model.model.language_model
        self.lm_head = hf_model.lm_head

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def _run(
        self,
        inputs_embeds,
        position_ids,
        past_seq_length,
        current_input_length,
        local_attention_mask,
        global_attention_mask,
        past_key_cache,
        past_value_cache,
        per_layer_inputs,
    ):
        hidden_states = self.language_model(
            inputs_embeds=inputs_embeds,
            per_layer_inputs=per_layer_inputs,
            position_ids=position_ids,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            local_attention_mask=local_attention_mask,
            global_attention_mask=global_attention_mask,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )
        logits = self.lm_head(hidden_states)
        final_logit_softcapping = self.config.get_text_config().final_logit_softcapping
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping
        return logits


class _Gemma4TextExportBridgePLE(_Gemma4TextExportBridgeBase):
    """Bridge for PLE models (hidden_size_per_layer_input > 0); per_layer_inputs is first arg."""

    def forward(
        self,
        per_layer_inputs,
        inputs_embeds,
        position_ids,
        past_seq_length,
        current_input_length,
        local_attention_mask,
        global_attention_mask,
        past_key_cache=None,
        past_value_cache=None,
    ):
        return self._run(
            inputs_embeds, position_ids, past_seq_length, current_input_length,
            local_attention_mask, global_attention_mask, past_key_cache, past_value_cache,
            per_layer_inputs,
        )


class _Gemma4TextExportBridgeDense(_Gemma4TextExportBridgeBase):
    """Bridge for dense models (hidden_size_per_layer_input = 0); no per_layer_inputs arg."""

    def forward(
        self,
        inputs_embeds,
        position_ids,
        past_seq_length,
        current_input_length,
        local_attention_mask,
        global_attention_mask,
        past_key_cache=None,
        past_value_cache=None,
    ):
        return self._run(
            inputs_embeds, position_ids, past_seq_length, current_input_length,
            local_attention_mask, global_attention_mask, past_key_cache, past_value_cache,
            None,
        )


def _make_text_export_bridge(hf_model: Gemma4ForConditionalGeneration):
    text_config = hf_model.config.get_text_config()
    if getattr(text_config, "hidden_size_per_layer_input", 0):
        return _Gemma4TextExportBridgePLE(hf_model)
    return _Gemma4TextExportBridgeDense(hf_model)


def _flatten_multimodal_features(features: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if features is None:
        return None
    if features.ndim == 2:
        return features
    return features.reshape(-1, features.shape[-1])


def _split_multimodal_features_by_chunk(
    input_ids: torch.Tensor,
    *,
    token_id: int,
    features: Optional[torch.Tensor],
    chunk_size: int,
) -> list[Optional[torch.Tensor]]:
    steps = (input_ids.shape[1] + chunk_size - 1) // chunk_size
    if token_id < 0 or features is None:
        return [None] * steps

    flat_features = _flatten_multimodal_features(features)
    assert flat_features is not None
    total_token_count = int((input_ids == token_id).sum().item())
    if total_token_count != flat_features.shape[0]:
        raise ValueError(
            f"Feature count does not match token count for token id {token_id}: "
            f"{flat_features.shape[0]} vs {total_token_count}"
        )

    chunked_features: list[Optional[torch.Tensor]] = []
    start_feature = 0
    for start in range(0, input_ids.shape[1], chunk_size):
        end = min(start + chunk_size, input_ids.shape[1])
        chunk_token_count = int((input_ids[:, start:end] == token_id).sum().item())
        if chunk_token_count == 0:
            chunked_features.append(None)
            continue
        stop_feature = start_feature + chunk_token_count
        chunked_features.append(flat_features[start_feature:stop_feature])
        start_feature = stop_feature

    return chunked_features


def _trim_masked_multimodal_features(
    features: Optional[torch.Tensor], feature_mask: Optional[torch.Tensor]
) -> Optional[torch.Tensor]:
    if features is None or feature_mask is None:
        return features

    mask = feature_mask.to(torch.bool)
    if features.ndim == 3 and mask.ndim == 2:
        return features[mask]
    if features.ndim == 2 and mask.ndim == 1:
        return features[mask]
    raise ValueError(
        f"Unsupported multimodal feature mask shapes: features={tuple(features.shape)}, mask={tuple(mask.shape)}"
    )


def _get_visual_export_mode(llm_model: Any) -> str:
    visual = getattr(llm_model, "visual", None)
    if visual is not None and getattr(visual, "export_mode", None) is not None:
        return visual.export_mode

    model_config = getattr(llm_model, "config", None)
    visual_config = getattr(model_config, "visual_config", None)
    if visual_config is not None and getattr(visual_config, "export_mode", None) is not None:
        return visual_config.export_mode

    visual_meta = getattr(llm_model, "visual_meta", None)
    if visual_meta is not None and getattr(visual_meta, "export_mode", None) is not None:
        return visual_meta.export_mode

    return "full"


def _run_visual_model(
    visual_model: Any,
    pixel_values: torch.Tensor,
    image_position_ids: Optional[torch.Tensor],
    *,
    export_mode: str,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    pixel_values = pixel_values.to(visual_model.device, visual_model.dtype)
    image_count = pixel_values.shape[0] if pixel_values.ndim >= 3 else 1

    def _trim_visual_patch_padding(
        visual_pixel_values: torch.Tensor,
        visual_position_ids: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if visual_position_ids is None or visual_pixel_values.ndim < 3:
            return visual_pixel_values
        if visual_position_ids.ndim == 2:
            visual_position_ids = visual_position_ids.unsqueeze(0)
        valid_positions = ~(visual_position_ids == -1).all(dim=-1)
        real_patch_count = int(valid_positions[0].sum().item())
        if real_patch_count <= 0 or visual_pixel_values.shape[1] == real_patch_count:
            return visual_pixel_values
        return visual_pixel_values[:, :real_patch_count, :].contiguous()

    def _forward_one(image_index: int | None = None):
        if image_index is None:
            visual_pixel_values = pixel_values
            visual_position_ids = image_position_ids
        else:
            visual_pixel_values = pixel_values[image_index : image_index + 1].contiguous()
            visual_position_ids = (
                image_position_ids[image_index : image_index + 1].contiguous()
                if image_position_ids is not None and image_position_ids.ndim >= 3
                else image_position_ids
            )

        visual_pixel_values = _trim_visual_patch_padding(
            visual_pixel_values,
            visual_position_ids,
        )

        if visual_position_ids is not None:
            trimmed_position_ids = visual_position_ids
            if trimmed_position_ids.ndim == 2:
                trimmed_position_ids = trimmed_position_ids.unsqueeze(0)
            real_count = visual_pixel_values.shape[1]
            trimmed_position_ids = trimmed_position_ids[:, :real_count, :].contiguous()
            return visual_model.forward(visual_pixel_values, trimmed_position_ids)
        return visual_model.forward(visual_pixel_values)

    if image_count <= 1:
        return _forward_one()

    image_embeds_list: list[torch.Tensor] = []
    image_embed_mask_list: list[torch.Tensor] = []
    has_embed_mask = False
    for image_index in range(image_count):
        image_outputs = _forward_one(image_index)
        image_embed_mask = None
        if isinstance(image_outputs, (tuple, list)) and len(image_outputs) == 2:
            image_embeds, image_embed_mask = image_outputs
            has_embed_mask = True
        else:
            image_embeds = image_outputs
        image_embeds_list.append(image_embeds)
        if image_embed_mask is not None:
            image_embed_mask_list.append(image_embed_mask)

    image_embeds = torch.cat(image_embeds_list, dim=0)
    if has_embed_mask:
        if len(image_embed_mask_list) != image_count:
            raise ValueError("Gemma4 visual runtime returned masks for only part of the image batch.")
        return image_embeds, torch.cat(image_embed_mask_list, dim=0)
    return image_embeds


class _Gemma4HFCompatible(TextLLMHFCompatible):
    def _setup(self: Gemma4ForConditionalGeneration, text_llm_model: "XHGemma4Model"):
        model = super()._setup(text_llm_model)
        if model is not None and hasattr(model, "model"):
            if hasattr(model.model, "language_model"):
                del model.model.language_model
            if hasattr(model.model, "vision_tower"):
                del model.model.vision_tower
            if hasattr(model.model, "audio_tower"):
                del model.model.audio_tower
            if hasattr(model.model, "embed_vision"):
                del model.model.embed_vision
            if hasattr(model.model, "embed_audio"):
                del model.model.embed_audio
            if hasattr(model, "lm_head"):
                del model.lm_head
        return model

    def set_experts_implementation(self, experts_implementation):  # noqa: D401
        """No-op override for Transformers decode optimization hooks."""
        self.config.experts_implementation = experts_implementation

    def get_correct_experts_implementation(self, experts_implementation):
        return experts_implementation

    def _grouped_mm_can_dispatch(self):
        return False

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        position_ids=None,
        pixel_values=None,
        pixel_values_videos=None,
        input_features=None,
        attention_mask=None,
        input_features_mask=None,
        token_type_ids=None,
        image_position_ids=None,
        mm_token_type_ids=None,
        use_cache=True,
        logits_to_keep=None,
        labels=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            input_features=input_features,
            attention_mask=attention_mask,
            input_features_mask=input_features_mask,
            token_type_ids=token_type_ids,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            labels=labels,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )

        if is_first_iteration or not use_cache:
            model_inputs["image_position_ids"] = image_position_ids
            model_inputs["mm_token_type_ids"] = mm_token_type_ids
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        input_features: Optional[torch.Tensor] = None,
        input_features_mask: Optional[torch.Tensor] = None,
        image_position_ids: Optional[torch.LongTensor] = None,
        past_seq_length: Optional[int] = None,
        use_cache: Optional[bool] = None,
        mm_token_type_ids: Optional[torch.LongTensor] = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        del labels, position_ids, past_seq_length, use_cache

        if input_ids is None:
            raise ValueError("Gemma4 HF-compatible path requires input_ids.")

        image_embeds = None
        if pixel_values is not None and getattr(self._llm_model, "visual", None) is not None:
            visual_model = self._llm_model.visual
            image_outputs = _run_visual_model(
                visual_model,
                pixel_values,
                image_position_ids,
                export_mode=_get_visual_export_mode(self._llm_model),
            )
            image_embed_mask = None
            if isinstance(image_outputs, (tuple, list)) and len(image_outputs) == 2:
                image_embeds, image_embed_mask = image_outputs
            else:
                image_embeds = image_outputs
            image_embeds = _trim_masked_multimodal_features(image_embeds, image_embed_mask)

        audio_embeds = None
        if (
            input_features is not None
            and input_features_mask is not None
            and getattr(self._llm_model, "audio", None) is not None
        ):
            audio_outputs = self._llm_model.audio.forward(
                input_features.to(self._llm_model.audio.device, self._llm_model.audio.dtype),
                input_features_mask.to(self._llm_model.audio.device),
            )
            audio_embed_mask = None
            if isinstance(audio_outputs, (tuple, list)) and len(audio_outputs) == 2:
                audio_embeds, audio_embed_mask = audio_outputs
            else:
                audio_embeds = audio_outputs
            audio_embeds = _trim_masked_multimodal_features(audio_embeds, audio_embed_mask)

        data_processor = self._llm_model.get_data_preprocessor()
        seq_length = input_ids.shape[1]
        net_input_seq_len = self._llm_model.get_input_sequence_length()
        steps = (seq_length + net_input_seq_len - 1) // net_input_seq_len

        if steps == 1:
            data_input = data_processor(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "image_embeds": image_embeds,
                    "audio_embeds": audio_embeds,
                    "past_seq_length": self._past_seq_length,
                    "inputs_embeds": inputs_embeds,
                    "mm_token_type_ids": mm_token_type_ids,
                }
            )
            logits = self._llm_model.forward(*data_input)
            if not isinstance(logits, torch.Tensor):
                logits = logits[0]
        else:
            image_embed_chunks = _split_multimodal_features_by_chunk(
                input_ids,
                token_id=self.config.image_token_id,
                features=image_embeds,
                chunk_size=net_input_seq_len,
            )
            audio_embed_chunks = _split_multimodal_features_by_chunk(
                input_ids,
                token_id=self.config.audio_token_id,
                features=audio_embeds,
                chunk_size=net_input_seq_len,
            )

            output_logits = []
            for chunk_index in range(steps):
                start = chunk_index * net_input_seq_len
                end = min(start + net_input_seq_len, seq_length)
                chunk_inputs_embeds = None
                if inputs_embeds is not None:
                    chunk_inputs_embeds = inputs_embeds[:, start:end, :]

                data_input = data_processor(
                    {
                        "input_ids": input_ids[:, start:end],
                        "attention_mask": None if attention_mask is None else attention_mask[:, start:end],
                        "image_embeds": image_embed_chunks[chunk_index],
                        "audio_embeds": audio_embed_chunks[chunk_index],
                        "past_seq_length": self._past_seq_length + start,
                        "inputs_embeds": chunk_inputs_embeds,
                        "mm_token_type_ids": None
                        if mm_token_type_ids is None
                        else mm_token_type_ids[:, start:end],
                    }
                )
                chunk_logits = self._llm_model.forward(*data_input)
                if not isinstance(chunk_logits, torch.Tensor):
                    chunk_logits = chunk_logits[0]
                output_logits.append(chunk_logits)

            # Last chunk's tail is padded up to ``net_input_seq_len`` for the
            # static-shape kernel. Trim it to the real-token count so that
            # HF generate's ``logits[:, -1, :]`` lands on the actual last token
            # rather than a padded slot — without this, the first generated
            # token (prefill output) is garbage. Decode is unaffected.
            last_valid = min(net_input_seq_len, seq_length - (steps - 1) * net_input_seq_len)
            if logits_to_keep != 0:
                logits = output_logits[-1][:, :last_valid, :]
            else:
                logits = torch.cat(output_logits, dim=1)[:, :seq_length, :]
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


def build_gemma4_hf_compatible_model(
    hf_model: Gemma4ForConditionalGeneration,
    xh_model: "XHGemma4Model",
):
    llm_compatible_modules = _DMRegistryCls("XHCompatible")
    hf_model_cls = type(hf_model)
    if hf_model_cls not in llm_compatible_modules:
        llm_compatible_modules.register_module({hf_model_cls: hf_model_cls.__name__}, _Gemma4HFCompatible)
    return llm_compatible_modules.convert(hf_model, text_llm_model=xh_model)


@register_llm_model("Gemma4ForConditionalGeneration")
class XHGemma4Model(VisionLLMModel):
    HF_MODEL_CLS = Gemma4ForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = Gemma4ModelMeta
    HMONNXINFERENCE_CLS = XHGemma4_HMONNXModel
    CONFIG_CLS = XHGemma4ModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_gemma4_hf_compatible_model)

    def __init__(self, config: XHGemma4ModelConfig):
        super().__init__(config)
        self.config = cast(XHGemma4ModelConfig, self.config)
        self._kvcache_mixin = Gemma4KVCacheMixin(self.kvcache_config)
        self.visual = XHGemma4VisionModel(config.visual_config) if config.visual_config is not None else None
        self.audio = XHGemma4AudioModel(config.audio_config) if config.audio_config is not None else None
        self.per_layer_input_builder: Gemma4PerLayerInputBuilder | None = None

    @VisionLLMModel.work_dir.setter
    def work_dir(self, work_dir: str):
        self.config.work_dir = work_dir
        if self.visual is not None:
            self.visual.work_dir = str(Path(work_dir) / "visual")
        if self.audio is not None:
            self.audio.work_dir = str(Path(work_dir) / "audio")

    def _get_language_model(self, hf_model: Any) -> Any:
        if hasattr(hf_model, "language_model"):
            return hf_model.language_model
        if hasattr(hf_model, "model") and hasattr(hf_model.model, "language_model"):
            return hf_model.model.language_model
        return hf_model

    def _wraped_post(self, hf_model: Any):
        super()._wraped_post(hf_model)
        language_model = self._get_language_model(self._wrap_model)
        text_config = language_model.config
        self.pad_token_id = text_config.pad_token_id
        layer_kv_shapes: list[list[int]] = []
        if self.use_cache:
            for layer in language_model.layers:
                attn = layer.self_attn
                if getattr(attn, "is_kv_shared_layer", False):
                    continue
                layer_kv_shapes.append(
                    [
                        1,
                        attn.k_proj.out_features // attn.head_dim,
                        self.config.context_max_length,
                        attn.head_dim,
                    ]
                )
        self._kvcache_mixin.set_layer_kv_shapes(layer_kv_shapes)
        self.config.sliding_window = text_config.sliding_window
        self.config.image_token_id = hf_model.config.image_token_id
        self.config.audio_token_id = hf_model.config.audio_token_id
        self.config.video_token_id = hf_model.config.video_token_id
        self.config.boi_token_id = hf_model.config.boi_token_id
        self.config.eoi_token_id = hf_model.config.eoi_token_id
        self.config.boa_token_id = hf_model.config.boa_token_id
        self.config.eoa_token_id = hf_model.config.eoa_token_id
        if getattr(text_config, "hidden_size_per_layer_input", 0):
            self.per_layer_input_builder = Gemma4PerLayerInputBuilder.from_language_model(language_model)

    def init_wrap_model(self, hf_model: Any) -> Any:
        from ._llm_model_impl import register_wrap_modules

        self.config.image_token_id = hf_model.config.image_token_id
        self.config.audio_token_id = hf_model.config.audio_token_id
        self.config.video_token_id = hf_model.config.video_token_id
        register_wrap_modules(hf_model)
        wrap_model = super().init_wrap_model(_make_text_export_bridge(hf_model))
        wrap_model.language_model.rotary_emb.set_target_dtype(wrap_model.language_model.embed_tokens.weight.dtype)
        return wrap_model

    def get_tf_processor(self):
        processor = XHGemma4Processor.from_pretrained(self.hf_model_dir)
        if self.visual is not None:
            processor = configure_gemma4_visual_processor(
                processor,
                export_mode=self.visual.config.export_mode,
                max_size_w=self.visual.config.max_size_w,
                max_size_h=self.visual.config.max_size_h,
                patch_size=self.visual.config.patch_size,
                image_seq_length=self.visual.config.image_seq_length,
            )
        if self.audio is not None:
            processor.config.sampling_rate = self.audio.config.sampling_rate
        return processor

    def get_prefill_dummy_inputs(self) -> dict[str, torch.Tensor | int]:
        image_token_id = getattr(self.config, "image_token_id", -1)
        if self.visual is None or image_token_id < 0:
            return super().get_prefill_dummy_inputs()

        text_config = AutoConfig.from_pretrained(self.hf_model_dir, trust_remote_code=True).get_text_config()
        image_token_count = min(self.config.prefill_chunk_length, self.visual.config.image_seq_length)
        input_ids = torch.full((1, image_token_count), image_token_id, dtype=torch.long)
        mm_token_type_ids = torch.ones_like(input_ids)
        generator = torch.Generator(device="cpu").manual_seed(0)
        image_embeds = torch.randn(
            image_token_count,
            text_config.hidden_size,
            generator=generator,
            dtype=torch.float32,
        ) * 0.55
        return {
            "input_ids": input_ids,
            "mm_token_type_ids": mm_token_type_ids,
            "image_embeds": image_embeds,
            "past_seq_length": 0,
        }

    def get_quant_cfg(self):
        return super().get_quant_cfg()

    def _get_data_preprocessor(self) -> Gemma4DataPreprocess:
        config = Gemma4InputProcessorConfig(
            embed_tokens=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            per_layer_input_builder=self.per_layer_input_builder,
            pad_token_id=self.pad_token_id,
            context_max_length=self.config.context_max_length,
            sliding_window=self.config.sliding_window,
            use_explicit_attention_mask=getattr(self.config, "use_explicit_attention_mask", True),
            image_token_id=self.config.image_token_id,
            audio_token_id=self.config.audio_token_id,
            video_token_id=self.config.video_token_id,
        )
        return Gemma4DataPreprocess(config)

    def get_export_cfg(self) -> dict[str, list[str]]:
        input_names = []
        if self.per_layer_input_builder is not None:
            input_names.append("per_layer_inputs")
        input_names += [
            "inputs_embeds",
            "position_ids",
            "past_seq_length",
            "current_input_length",
            "local_attention_mask",
            "global_attention_mask",
        ]
        export_cfg = {"input_names": input_names, "output_names": ["logits"]}
        for layer_idx in range(self.kvcache_config.num_layers):
            export_cfg["input_names"].append(f"past_key_cache_{layer_idx}")
        for layer_idx in range(self.kvcache_config.num_layers):
            export_cfg["input_names"].append(f"past_value_cache_{layer_idx}")
        return export_cfg

    def _extra_export_metadata(self, output_dir: str, meta_info: Gemma4ModelMeta) -> Gemma4ModelMeta:
        del output_dir
        meta_info.layer_kv_shapes = self._kvcache_mixin.layer_kv_shapes
        return meta_info

    @log_function_call()
    def export_hmonnx(self, output_dir: str) -> Gemma4ModelMeta:
        logger = get_xhquant_logger()
        self.work_dir = str(output_dir)
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        self._quanted_model.fixed()
        if self.visual is not None:
            self.visual.quanted_model.fixed()
        if self.audio is not None:
            self.audio.quanted_model.fixed()

        exported_info = self.get_export_info(output_dir)
        meta_info = cast(Gemma4ModelMeta, exported_info.meta)

        if self.visual is not None:
            visual_output_dir = str(Path(exported_info.exported_dir) / "visual")
            self.visual.config.model_name = f"{exported_info.model_name}_visual"
            visual_meta = self.visual.export_hmonnx(visual_output_dir)
            visual_meta.hmonnx = str(Path(visual_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix())
            if getattr(visual_meta, "onnx", None):
                visual_meta.onnx = str(Path(visual_meta.onnx).relative_to(exported_info.exported_dir).as_posix())
            meta_info.visual_config = visual_meta

        if self.audio is not None:
            audio_output_dir = str(Path(exported_info.exported_dir) / "audio")
            self.audio.config.model_name = f"{exported_info.model_name}_audio"
            audio_meta = self.audio.export_hmonnx(audio_output_dir)
            audio_meta.hmonnx = str(Path(audio_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix())
            if getattr(audio_meta, "onnx", None):
                audio_meta.onnx = str(Path(audio_meta.onnx).relative_to(exported_info.exported_dir).as_posix())
            meta_info.audio_config = audio_meta

        if self.per_layer_input_builder is not None:
            per_layer_input_builder_path = Path(exported_info.exported_dir) / "per_layer_input_builder.pt"
            self.per_layer_input_builder.save_artifact(per_layer_input_builder_path)
            meta_info.per_layer_input_builder = per_layer_input_builder_path.relative_to(exported_info.exported_dir).as_posix()

        self._export_hmonnx(exported_info)
        json.dump(
            meta_info.to_dict(),
            open(str(Path(exported_info.exported_dir) / "golden_meta_info.json"), "w"),
            indent=4,
        )
        logger.info(f"Exporting completed! Exported model is saved at: {exported_info.exported_dir}")
        return meta_info
