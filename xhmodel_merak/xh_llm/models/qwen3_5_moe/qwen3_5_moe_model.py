from typing import Any, Optional, Union

import torch
from accelerate import init_empty_weights
from transformers import AutoModelForImageTextToText
from transformers.cache_utils import Cache
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeCausalLMOutputWithPast,
    Qwen3_5MoeForConditionalGeneration,
)

from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_vision_model import XHQwen3_5MoeVisionModel
from xhquant.utils.registry import _DMRegistryCls

from ...builder import register_llm_model
from ...text_llm_hf_compatible import TextLLMHFCompatible
from ...vision_llm_model import VisionLLMModel
from ..qwen3_5.qwen3_5_llm_model import (
    Qwen3_5_ModelMeta,
    XHQwen3_5Model,
    _enforce_split_conv_cache_wrap_cfg,
)
from .qwen3_5_moe_hmonnx_inference import XHQwen3_5MoeHMONNXModel
from .xh_qwen3_5_moe_config import XHQwen3_5MoeModelConfig


try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    no_init_weights = init_empty_weights


class _Qwen3_5HFCompatible(TextLLMHFCompatible):  # noqa: N801
    def _setup(self: Qwen3_5MoeForConditionalGeneration, text_llm_model: "XHQwen3_5MoeModel"):
        model = super()._setup(text_llm_model)
        if model is not None:
            # if hasattr(model, "model"):
            #     del model.model
            del model.model.visual
            del model.model.language_model
            if hasattr(model, "lm_head"):
                del model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return model

    def set_experts_implementation(self, experts_implementation):  # noqa: D401
        """No-op override for Transformers decode optimization hooks."""
        self.config.experts_implementation = experts_implementation

    def get_correct_experts_implementation(self, experts_implementation):
        return experts_implementation

    def _grouped_mm_can_dispatch(self):
        return False

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, Qwen3_5MoeCausalLMOutputWithPast]:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_embeds = None

        if pixel_values is not None:
            image_embeds = list()
            for i in range(len(pixel_values)):
                pv = pixel_values[i].type(self._llm_model.visual.dtype).to(self._llm_model.visual.device)
                if pv.dim() == 4:
                    pv = pv.unsqueeze(0)
                for j in range(pv.shape[0]):
                    image_embeds_j = self._llm_model.visual.forward(pv[j : j + 1])
                    image_embeds.append(image_embeds_j)

            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_embeds = image_embeds.squeeze(0)

        seq_length = inputs_embeds.shape[1]
        data_processor = self._llm_model.get_data_preprocessor()
        net_input_seq_len = self._llm_model.get_input_sequence_length()

        if seq_length <= net_input_seq_len:
            data_batch = {
                "input_ids": input_ids,
                "image_embeds": image_embeds,
                "past_seq_length": self._past_seq_length,
                "image_grid_thw": image_grid_thw,
                "video_grid_thw": video_grid_thw,
            }
            data_input = data_processor(data_batch)
            (
                inputs_embeds,
                time_position_ids,
                height_position_ids,
                width_position_ids,
                past_seq_length,
                current_seq_length,
                linear_mask,
                past_key_values,
                past_value_caches,
                past_conv_caches,
                past_recurrent_states,
            ) = data_input

            logits, conv_cache_out_list, recurrent_state_out_list = self._llm_model.forward(
                inputs_embeds,
                time_position_ids,
                height_position_ids,
                width_position_ids,
                past_seq_length,
                current_seq_length,
                linear_mask,
                past_key_values,
                past_value_caches,
                past_conv_caches,
                past_recurrent_states,
            )
        else:
            device = inputs_embeds.device

            if image_embeds is not None and input_ids is not None:
                image_token_id = data_processor.image_token_id
                n_image_tokens = int((input_ids == image_token_id).sum().item())
                if n_image_tokens > 0:
                    n_image_features = int(image_embeds.shape[0])
                    if n_image_features != n_image_tokens:
                        raise ValueError(
                            "Image features and image tokens do not match: "
                            f"tokens={n_image_tokens}, features={n_image_features}"
                        )
                    image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                    image_embeds_merged = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds_merged)

            position_ids, rope_deltas = data_processor.get_rope_index(
                input_ids, inputs_embeds, image_grid_thw, video_grid_thw, None
            )
            data_processor.rope_deltas = rope_deltas

            steps = (seq_length + net_input_seq_len - 1) // net_input_seq_len
            pad_len = steps * net_input_seq_len - seq_length
            if pad_len > 0:
                padding_embeds = self.get_input_embeddings()(torch.zeros((1, pad_len), dtype=torch.long, device=device))
                inputs_embeds = torch.cat([inputs_embeds, padding_embeds], dim=1)
                last_pos = position_ids[:, :, -1:].expand(-1, -1, pad_len)
                position_ids = torch.cat([position_ids, last_pos], dim=2)

            past_key_caches = data_processor.past_key_caches
            past_value_caches = data_processor.past_value_caches
            past_conv_caches = data_processor.past_conv_caches
            past_recurrent_states = data_processor.past_recurrent_states
            running_past_seq = self._past_seq_length

            outputs_logits = []
            for step_index in range(steps):
                start = step_index * net_input_seq_len
                end = (step_index + 1) * net_input_seq_len
                sub_embeds = inputs_embeds[:, start:end, :]
                sub_current_len = min(end, seq_length) - start

                sub_time_pos = position_ids[0, 0, start:end].to(torch.int64)
                sub_height_pos = position_ids[1, 0, start:end].to(torch.int64)
                sub_width_pos = position_ids[2, 0, start:end].to(torch.int64)

                linear_mask = (
                    torch.cat(
                        [
                            torch.ones(sub_current_len, device=device),
                            torch.zeros(net_input_seq_len - sub_current_len, device=device),
                        ]
                    )
                    .unsqueeze(0)
                    .to(dtype=torch.float16)
                )

                chunk_logits, _, _ = self._llm_model.forward(
                    sub_embeds,
                    sub_time_pos,
                    sub_height_pos,
                    sub_width_pos,
                    torch.tensor([running_past_seq], dtype=torch.int32, device=device),
                    torch.tensor([sub_current_len], dtype=torch.int32, device=device),
                    linear_mask,
                    past_key_caches,
                    past_value_caches,
                    past_conv_caches,
                    past_recurrent_states,
                )
                outputs_logits.append(chunk_logits)
                running_past_seq += sub_current_len

            last_valid = min(net_input_seq_len, seq_length - (steps - 1) * net_input_seq_len)
            logits = outputs_logits[-1][:, :last_valid, :]

        return Qwen3_5MoeCausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
            rope_deltas=data_processor.rope_deltas,
        )


def build_qwen3_5_moe_hf_compatible_model(
    hf_model: Qwen3_5MoeForConditionalGeneration,
    xh_model: "XHQwen3_5MoeModel",
):
    llm_compatible_modules = _DMRegistryCls("XHCompatible")
    hf_model_cls = type(hf_model)
    if hf_model_cls not in llm_compatible_modules:
        llm_compatible_modules.register_module({hf_model_cls: hf_model_cls.__name__}, _Qwen3_5HFCompatible)
    return llm_compatible_modules.convert(hf_model, text_llm_model=xh_model)


class Qwen3_5Moe_ModelMeta(Qwen3_5_ModelMeta):  # noqa: N801
    pass


@register_llm_model("Qwen3_5MoeForConditionalGeneration")
class XHQwen3_5MoeModel(XHQwen3_5Model):  # noqa: N801
    HF_MODEL_CLS = Qwen3_5MoeForConditionalGeneration
    HF_AUTO_MODEL_CLS = AutoModelForImageTextToText
    META_CLS = Qwen3_5Moe_ModelMeta
    HMONNXINFERENCE_CLS = XHQwen3_5MoeHMONNXModel
    CONFIG_CLS = XHQwen3_5MoeModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_qwen3_5_moe_hf_compatible_model)
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.qwen3_5.workflow:Qwen35Workflow"

    def __init__(self, config: XHQwen3_5MoeModelConfig):
        super().__init__(config)

        if hasattr(config, "visual_config") and config.visual_config is not None and config.visual_config.enable:
            self.visual = XHQwen3_5MoeVisionModel(config.visual_config)
            self.visual.config.model_name = f"{self.config.model_name}_visual"
        else:
            self.visual = None
        # self.full_attention_layer_indices: list[int] = []
        # self.linear_attention_layer_indices: list[int] = []
        # self._kvcache_config = KVCacheWithLinearConfig()
        # self._kvcache_mixin = KVCacheWithLinearMixin(self.kvcache_config)
        # self.config = cast(XHQwen3_5MoeModelConfig, self.config)
        # self.wrap_cfg["linear_attention_mode"] = "auto"

        # self.wrap_cfg["linear_attention_mode"] = "chunk"  # for prefill
        # self.wrap_cfg["linear_attention_mode"] = "recurrent"  # for decode

    def _wraped_post(self, hf_model: Qwen3_5MoeForConditionalGeneration):
        super()._wraped_post(hf_model)

        # The MoE wrapper follows the dense Qwen3.5 cache contract: external
        # export/runtime signatures are flat q/k/v conv-cache tensors, while
        # trace-time linear attention consumes per-layer (q, k, v) tuples.
        # Re-apply the split flag after MoE wrapping so child DynamicModules do
        # not silently trace the merged qkv path when the cache mixin/export
        # side is already split.
        language_model = self._get_language_model(self._wrap_model)
        _enforce_split_conv_cache_wrap_cfg(language_model, self.wrap_cfg)

        split_conv_cache = bool(self.wrap_cfg.get("split_conv_cache", False))
        self._kvcache_mixin.split_conv_cache = split_conv_cache
        if split_conv_cache and self.linear_attention_layer_indices:
            linear_attn = language_model.layers[self.linear_attention_layer_indices[0]].linear_attn
            self._kvcache_mixin._linear_key_dim = linear_attn.key_dim
            self._kvcache_mixin._linear_value_dim = linear_attn.value_dim

    def _get_big_language_placeholder_export_components(self):
        # Import the real MoE wrappers before entering
        # ``traceable_module_placeholder_context``.  The context temporarily
        # replaces these registrations with lightweight placeholder wrappers;
        # importing ``_moe_model`` for the first time from inside that context
        # would make its decorators register the same HF classes twice.
        from ._moe_model import register_wrap_modules
        from ._qwen3_5_moe_big_export import Qwen3_5_MOE_BigHFModel

        register_wrap_modules()
        return Qwen3_5_MOE_BigHFModel, Qwen3_5_MOE_BigHFModel.PLACEHOLDER_TYPES

    def _check_big_language_placeholder_export_supported(self, empty_hf_model: Any) -> None:
        hf_model_type = str(getattr(empty_hf_model.config, "model_type", "")).lower()
        model_type_name = type(empty_hf_model).__name__.lower()
        if "moe" not in hf_model_type and "moe" not in model_type_name:
            raise NotImplementedError(
                "Qwen3.5 dense big-model placeholder export must use the dense placeholder components."
            )

        language_model = self._get_language_model(empty_hf_model)
        if not hasattr(language_model, "modules"):
            raise NotImplementedError("Qwen3.5 MoE big-model placeholder export requires an nn.Module language model.")

        found_types = {type(module).__name__ for module in language_model.modules()}

        _, placeholder_types = self._get_big_language_placeholder_export_components()
        required_hf_types = {
            "Qwen3_5MoeAttention",
            "Qwen3_5MoeGatedDeltaNet",
            "Qwen3_5MoeSparseMoeBlock",
        }
        missing_types = sorted(required_hf_types - found_types)
        if missing_types:
            raise NotImplementedError(
                "Qwen3.5 MoE big-model placeholder export requires placeholder modules "
                f"{placeholder_types}, but missing {missing_types}."
            )

    def init_wrap_model(self, hf_model: Qwen3_5MoeForConditionalGeneration) -> Any:
        from ._moe_model import register_wrap_modules
        
        register_wrap_modules()
        wrap_model = super(VisionLLMModel, self).init_wrap_model(hf_model)
        return wrap_model
