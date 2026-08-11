from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Optional, Union, cast

import torch
from transformers import AutoModelForCausalLM, DynamicCache
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextForCausalLM

from xhquant.api import get_xhquant_logger
from xhquant.utils.registry import _DMRegistryCls

from ...base_llm_model import BaseLLMModel
from ...base_model import get_model_param_buffer_size_gb
from ...builder import register_llm_model
from ...text_llm_hf_compatible import TextLLMHFCompatible
from ...types import ExportData, LLMModelState, VLLMModelMeta
from ...utils import get_cpu_memory_mb, is_huge_model_export_enabled
from ..qwen3_5.qwen3_5_llm_model import (
    Qwen3_5_ModelMeta,
    XHQwen3_5Model,
    _enforce_split_conv_cache_wrap_cfg,
    build_qwen35_spec_decode_contract,
)
from .data_preprocess import Qwen3NextDataPreprocess
from .qwen3_next_hmonnx_inference import XHQwen3NextHMONNXModel
from .xh_qwen3_next_config import XHQwen3NextModelConfig


class Qwen3NextModelMeta(Qwen3_5_ModelMeta):
    pass


class _Qwen3NextHFCompatible(TextLLMHFCompatible):  # noqa: N801
    """Hugging Face generation adapter for the text-only hybrid runtime.

    Qwen3-Next has the same full-attention/GDN cache topology as Qwen3.5, but
    it has no visual tower or M-RoPE inputs.  Keeping this adapter local avoids
    inheriting Qwen3.5's visual-model teardown and visual preprocessor schema.
    """

    def _setup(self, text_llm_model: "XHQwen3NextModel"):
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
        """Keep Transformers' decode optimization from mutating the HMONNX graph.

        Transformers 5.5 temporarily selects a grouped-GEMM MoE backend around
        generation.  The converted object no longer contains HF expert modules,
        so this lifecycle hook must intentionally be a no-op.
        """
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
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if inputs_embeds.shape[0] != 1:
            raise ValueError("Qwen3-Next inference currently supports batch size 1")
        if use_cache is not False and past_key_values is None:
            # GenerationMixin uses this object for token-position bookkeeping;
            # the real full-attention and GDN states live in the Merak runtime.
            past_key_values = DynamicCache()

        seq_length = inputs_embeds.shape[1]
        graph_length = self._llm_model.get_input_sequence_length()
        steps = (seq_length + graph_length - 1) // graph_length
        outputs_logits = []
        for step in range(steps):
            start = step * graph_length
            end = min(start + graph_length, seq_length)
            data_input = self._llm_model.get_data_preprocessor()(
                {
                    "inputs_embeds": inputs_embeds[:, start:end],
                    "past_seq_length": self._past_seq_length + start,
                }
            )
            prepare_page_attention = getattr(
                self._llm_model, "prepare_page_attention_context", None
            )
            if callable(prepare_page_attention):
                prepare_page_attention(self._past_seq_length + start, end - start)
            result = self._llm_model.forward(*data_input)
            outputs_logits.append(result if isinstance(result, torch.Tensor) else result[0])

        if self._llm_model.get_num_logits_to_keep() != 0:
            logits = outputs_logits[-1]
        else:
            logits = torch.cat(outputs_logits, dim=1)[:, :seq_length]
        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)


def build_qwen3_next_hf_compatible_model(
    hf_model: Qwen3NextForCausalLM,
    xh_model: "XHQwen3NextModel",
):
    compatible_modules = _DMRegistryCls("XHCompatible")
    hf_model_cls = type(hf_model)
    if hf_model_cls not in compatible_modules:
        compatible_modules.register_module(
            {hf_model_cls: hf_model_cls.__name__}, _Qwen3NextHFCompatible
        )
    return compatible_modules.convert(hf_model, text_llm_model=xh_model)


@register_llm_model("Qwen3NextForCausalLM")
class XHQwen3NextModel(XHQwen3_5Model):
    """Merak-native, text-only Qwen3-Next target model."""

    HF_MODEL_CLS = Qwen3NextForCausalLM
    HF_AUTO_MODEL_CLS = AutoModelForCausalLM
    META_CLS = Qwen3NextModelMeta
    HMONNXINFERENCE_CLS = XHQwen3NextHMONNXModel
    CONFIG_CLS = XHQwen3NextModelConfig
    BUILD_HF_COMPATIBLE_FUNC = staticmethod(build_qwen3_next_hf_compatible_model)
    transformers_min_version = "4.57.0"
    WORKFLOW_CLS = "xhmodel_merak.xh_llm.models.qwen3_next.workflow:Qwen3NextWorkflow"

    def __init__(self, config: XHQwen3NextModelConfig):
        super().__init__(config)
        self.config = cast(XHQwen3NextModelConfig, self.config)
        self.visual = None

    def _get_language_model(self, hf_model: Any) -> Any:
        return hf_model.model

    def _wraped_pre(self, hf_model: Qwen3NextForCausalLM):
        # Match the selected prefix exactly. A prefix ending on a full-attention
        # layer is a useful reduced export while still exercising preceding GDNs.
        language_model = self._get_language_model(hf_model)
        self.layer_types = list(language_model.config.layer_types)
        if self.config.only_first_block:
            self.linear_attention_layer_indices = []
            self.full_attention_layer_indices = []
            for idx, layer_type in enumerate(self.layer_types):
                if layer_type == "full_attention":
                    self.full_attention_layer_indices.append(idx)
                    break
                self.linear_attention_layer_indices.append(idx)
            self.config.max_layers = len(self.full_attention_layer_indices) + len(
                self.linear_attention_layer_indices
            )
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

        logger = get_xhquant_logger()
        count, size_gb = get_model_param_buffer_size_gb(hf_model)
        logger.info(f"Qwen3-Next pre-wrap parameters: {count} B ({size_gb:.2f} GB)")
        return hf_model

    def _wraped_post(self, hf_model: Qwen3NextForCausalLM):
        language_model = self._get_language_model(self._wrap_model)
        _enforce_split_conv_cache_wrap_cfg(language_model, self.wrap_cfg)
        self.embed_tokens = copy.deepcopy(language_model.get_input_embeddings())
        text_config = language_model.config
        self.pad_token_id = text_config.eos_token_id
        self.layer_types = list(text_config.layer_types)

        if not self.full_attention_layer_indices:
            raise ValueError("Qwen3-Next export prefix must contain at least one full-attention layer")
        if not self.linear_attention_layer_indices:
            raise ValueError("Qwen3-Next export prefix must contain at least one GatedDeltaNet layer")
        self_attn = language_model.layers[self.full_attention_layer_indices[0]].self_attn
        linear_attn = language_model.layers[self.linear_attention_layer_indices[0]].linear_attn

        linear_cfg = self.kvcache_config.linear_kv_cache_config
        linear_cfg.conv_dim = linear_attn.conv_dim
        linear_cfg.conv_kernel_size = linear_attn.conv_kernel_size
        linear_cfg.num_v_heads = linear_attn.num_v_heads
        linear_cfg.head_k_dim = linear_attn.head_k_dim
        linear_cfg.head_v_dim = linear_attn.head_v_dim
        linear_cfg.num_layers = len(self.linear_attention_layer_indices)
        linear_cfg.batch_size = self.config.batch_size

        self._kvcache_mixin.split_conv_cache = bool(self.config.split_conv_cache)
        if self._kvcache_mixin.split_conv_cache:
            self._kvcache_mixin._linear_key_dim = linear_attn.key_dim
            self._kvcache_mixin._linear_value_dim = linear_attn.value_dim

        if self.use_cache:
            self.kvcache_config.num_layers = len(self.full_attention_layer_indices)
            self.kvcache_config.kv_cache_shape = [
                self.config.batch_size,
                text_config.num_key_value_heads,
                self.config.context_max_length,
                self_attn.head_dim,
            ]

        logger = get_xhquant_logger()
        count, size_gb = get_model_param_buffer_size_gb(self._wrap_model)
        logger.info(f"Qwen3-Next wrapped parameters: {count} B ({size_gb:.2f} GB)")
        logger.info(f"CPU memory after Qwen3-Next wrap: {get_cpu_memory_mb()}")

    def init_wrap_model(self, hf_model: Qwen3NextForCausalLM) -> Any:
        from ._model import register_wrap_modules

        register_wrap_modules()
        return super(XHQwen3_5Model, self).init_wrap_model(hf_model)

    def _get_data_preprocessor(self) -> Qwen3NextDataPreprocess:
        return Qwen3NextDataPreprocess(
            token_embedding=self.embed_tokens,
            input_sequence_length=self.wrap_cfg.input_sequence_length,
            past_key_caches=self.past_key_caches,
            past_value_caches=self.past_value_caches,
            past_conv_caches=self.past_conv_caches,
            past_recurrent_states=self.past_recurrent_states,
            enable_page_attention=self._kvcache_mixin.enable_page_attention,
            pad_token_id=self.pad_token_id,
        )

    def get_export_cfg(self) -> dict[str, list[str]]:
        self._sync_split_conv_cache_state()
        inputs = [
            "inputs_embeds",
            "past_seq_length",
            "current_input_length",
            "linear_attn_mask",
        ]
        outputs = ["logits"]
        if not self._kvcache_mixin.enable_page_attention:
            inputs.extend(
                f"past_key_cache_{idx}" for idx in range(self.kvcache_config.num_layers)
            )
            inputs.extend(
                f"past_value_cache_{idx}" for idx in range(self.kvcache_config.num_layers)
            )

        linear_layers = self.kvcache_config.linear_kv_cache_config.num_layers
        suppress_recurrent_state_outputs = self._set_recurrent_state_output_contract()
        verify_steps = 1
        if (
            bool(self.wrap_cfg.get("verify_output_intermediates", False))
            and int(self.wrap_cfg.get("input_sequence_length", 1)) > 1
        ):
            verify_steps = int(self.wrap_cfg.input_sequence_length)

        if self._kvcache_mixin.split_conv_cache:
            for idx in range(linear_layers):
                inputs.extend(f"past_conv_cache_{branch}_{idx}" for branch in ("q", "k", "v"))
        else:
            inputs.extend(f"past_conv_cache_{idx}" for idx in range(linear_layers))
        inputs.extend(f"past_recurrent_state_{idx}" for idx in range(linear_layers))

        if verify_steps > 1:
            if self._kvcache_mixin.split_conv_cache:
                for idx in range(linear_layers):
                    for branch in ("q", "k", "v"):
                        outputs.extend(
                            f"conv_cache_out_{branch}_{idx}_{step}"
                            for step in range(verify_steps)
                        )
            else:
                for idx in range(linear_layers):
                    outputs.extend(
                        f"conv_cache_out_{idx}_{step}" for step in range(verify_steps)
                    )
            if not suppress_recurrent_state_outputs:
                for idx in range(linear_layers):
                    outputs.extend(
                        f"recurrent_state_out_{idx}_{step}" for step in range(verify_steps)
                    )
        else:
            if self._kvcache_mixin.split_conv_cache:
                for idx in range(linear_layers):
                    outputs.extend(f"conv_cache_out_{branch}_{idx}" for branch in ("q", "k", "v"))
            else:
                outputs.extend(f"conv_cache_out_{idx}" for idx in range(linear_layers))
            if not suppress_recurrent_state_outputs:
                outputs.extend(f"recurrent_state_out_{idx}" for idx in range(linear_layers))
        if self.wrap_cfg.get("output_post_norm_hidden", False):
            outputs.append("post_norm_hidden")
        return {"input_names": inputs, "output_names": outputs}

    def get_export_info(self, output_dir) -> ExportData:
        return BaseLLMModel.get_export_info(self, output_dir)

    def _get_big_language_placeholder_export_components(self):
        from ._model import register_wrap_modules
        from ._qwen3_next_big_export import (
            Qwen3NextBigHFModel,
            register_runtime_wrap_modules,
        )

        register_wrap_modules()
        register_runtime_wrap_modules()
        return Qwen3NextBigHFModel, Qwen3NextBigHFModel.PLACEHOLDER_TYPES

    def _check_big_language_placeholder_export_supported(self, empty_hf_model: Any) -> None:
        hf_model_type = str(getattr(empty_hf_model.config, "model_type", "")).lower()
        if hf_model_type != "qwen3_next":
            raise NotImplementedError(
                "Qwen3-Next big-model placeholder export requires a qwen3_next model."
            )

        language_model = self._get_language_model(empty_hf_model)
        if not hasattr(language_model, "modules"):
            raise NotImplementedError(
                "Qwen3-Next big-model placeholder export requires an nn.Module language model."
            )

        found_types = {type(module).__name__ for module in language_model.modules()}
        required_hf_types = {
            "Qwen3NextAttention",
            "Qwen3NextGatedDeltaNet",
            "Qwen3NextSparseMoeBlock",
        }
        missing_types = sorted(required_hf_types - found_types)
        if missing_types:
            raise NotImplementedError(
                "Qwen3-Next big-model placeholder export is missing required modules "
                f"{missing_types}."
            )

    def export_hmonnx(self, output_dir: str) -> VLLMModelMeta:
        """Export target graphs and, when requested, independent MTP graphs."""
        exported_info = self.get_export_info(output_dir)
        if is_huge_model_export_enabled():
            return self._export_big_language_hmonnx(exported_info)
        return self._export_language_hmonnx_impl(exported_info)

    def _export_language_hmonnx_impl(
        self,
        exported_info: ExportData,
        lora_adapters: Optional[list[Any]] = None,
    ) -> VLLMModelMeta:
        if lora_adapters:
            raise NotImplementedError("Qwen3-Next LoRA export is not supported.")
        logger = get_xhquant_logger()
        if self._state != LLMModelState.QUANTED_ALIGNED:
            self.to_quanted_aligned()
        # Hybrid models keep separate prefill/chunk and decode/recurrent graphs.
        self._quanted_model.prefill.fixed()
        self._quanted_model.decode.fixed()

        # Reuse the target-side speculative contract from Qwen3.5 instead of
        # configuring only the decode dummy shape here.  MTP prefill consumes
        # one post-norm hidden row per prompt token to build its private KV
        # cache, so both target prefill and verify must retain every row
        # (num_logits_to_keep=0).  A decode-only override silently exported a
        # [B, 1, H] prefill hidden output and left the draft cache uninitialized.
        self._configure_spec_decode_target_export()

        self._export_hmonnx(exported_info)
        meta_info = cast(VLLMModelMeta, exported_info.meta)

        if self.config.spec_decode_mode == "mtp":
            if self.config.mtp_config is None:
                raise ValueError("mtp_config is required when spec_decode_mode='mtp'")
            from .qwen3_next_mtp_model import XHQwen3NextMTPDraftModel

            base_cfg = self.config.mtp_config
            for stage, seq_len in (
                ("mtp_draft_prefill", self.config.prefill_chunk_length),
                ("mtp_draft_decode", 1),
            ):
                draft_cfg = copy.deepcopy(base_cfg)
                draft_cfg.input_sequence_length = seq_len
                draft_cfg.model_name = f"{exported_info.model_name}_{stage}"
                draft_cfg.work_dir = str(Path(exported_info.exported_dir) / stage)
                draft_model = XHQwen3NextMTPDraftModel(draft_cfg)
                draft_model.to_quanted_aligned()
                draft_meta = draft_model.export_hmonnx(draft_cfg.work_dir)
                draft_meta.hmonnx = str(
                    Path(draft_meta.hmonnx).relative_to(exported_info.exported_dir).as_posix()
                )
                setattr(meta_info, f"{stage.removeprefix('mtp_draft_')}_mtp_config", draft_meta)
                if stage.endswith("prefill"):
                    meta_info.mtp_prefill_config = draft_meta
                else:
                    meta_info.mtp_decode_config = draft_meta

            # Qwen3-Next shares the Qwen3.5 proposer ABI. Use the common
            # manifest builder so verify length, W4 draft-head precision,
            # private-cache binding, and graph aliases cannot drift between
            # the two model families.
            spec_decode_section = build_qwen35_spec_decode_contract(
                self.config,
                meta_info,
            )
            meta_info.spec_decode_draft_head_weight_bits = (
                spec_decode_section["draft_head_weight_bits"]
            )
            meta_info.spec_decode = spec_decode_section

        meta_info.prefill_onnx = meta_info.prefill_hmonnx
        meta_info.decode_onnx = meta_info.decode_hmonnx
        meta_info.token_embedding_file = meta_info.quant_embedding
        meta_info.max_context_tokens = self.config.context_max_length
        meta_path = Path(exported_info.exported_dir) / "golden_meta_info.json"
        meta_path.write_text(json.dumps(meta_info.to_dict(), indent=4), encoding="utf-8")
        logger.info(f"Qwen3-Next export completed: {exported_info.exported_dir}")
        return meta_info
