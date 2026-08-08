from typing import List, Optional, Union

import torch
from torch import Tensor
from transformers import AutoTokenizer, GemmaModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from ...base_llm_model import LLMBaseModel
from ...builder import XHLLM_TRACEABLE_MODULES, register_other_model


def _clear_gemma_traceable_modules() -> None:
    from transformers.models.gemma.modeling_gemma import (
        GemmaAttention,
        GemmaDecoderLayer,
        GemmaForCausalLM,
        GemmaModel,
        GemmaRMSNorm,
    )

    classes = {
        GemmaRMSNorm,
        GemmaAttention,
        GemmaDecoderLayer,
        GemmaModel,
        GemmaForCausalLM,
    }
    classes.update(
        cls for cls in XHLLM_TRACEABLE_MODULES._registry if cls.__module__ == "lerobot.policies.pi_gemma"
    )
    for cls in classes:
        XHLLM_TRACEABLE_MODULES._registry.pop(cls, None)
        XHLLM_TRACEABLE_MODULES._key_registry.pop(cls, None)
    XHLLM_TRACEABLE_MODULES._dynamic_classes.clear()

@register_other_model("XHGemmaLLMModel")
class XHGemmaLLMModel(LLMBaseModel):
    WORKFLOW_CLS = "xhmodel_merak.xh_other_model.models.pi05.workflow:PI05Workflow"

    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type="TorchFX",
        allow_quant=True,
        export_cfg=None,
    ):
        super().__init__(
            hf_model,
            wrap_cfg,
            quant_config,
            frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )

    def get_hf_model(self, device_map="cpu", **kwargs) -> GemmaModel:
        assert self.hf_model_dir is not None
        if kwargs["model"] == "pi0.5":
            from ._export_utils import load_pi05_policy

            policy = load_pi05_policy(self.hf_model_dir, device=str(device_map))
        else:
            from lerobot.policies.pi0 import PI0Policy

            policy = PI0Policy.from_pretrained(
                pretrained_name_or_path=self.hf_model_dir,
                strict=True
            ).eval()
        # if policy.config.tie_word_embeddings:
        #     policy.config.torchscript = True
        #     policy.tie_weights()
        #     policy.config.tie_word_embeddings = False
        #     policy.config.torchscript = False
        return policy
    
    def get_tokenizer(self, config_dir: str):
        # tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        tokenizer = AutoTokenizer.from_pretrained(config_dir)
        return tokenizer

    def init_wrap_model(self, hf_model: Optional[GemmaModel] = None):

        _clear_gemma_traceable_modules()
        from ._llm_model_impl import register_wrap_cls as llm_register_wrap_cls
        llm_register_wrap_cls(hf_model)
        super().init_wrap_model(hf_model)
        
        _llm = self.wrap_model
        self.config = _llm.config
        self.token_embedding = _llm.embed_tokens
        self.generation_config = getattr(_llm, "generation_config", None)
        self.num_hidden_layers = _llm.config.num_hidden_layers
        head_dim = _llm.layers[0].self_attn.head_dim
        batch_size = 1
        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            if only_first_block:
                num_decoder_layers = 1
            self.prepare_kv_cache(
                num_decoder_layers,
                [batch_size, _llm.config.num_key_value_heads, self.cache_length, head_dim],
            )
        _llm = None

    def prepare_inputs(self, data: Union[dict, tuple, list], out_padding=True):
        raw_input_ids: List[List[int]] = data["input_ids"]
        assert self.token_embedding is not None, "Token embedding is not available."

        device = self.execution_device
        input_ids = []
        requested_lengths = data.get("current_input_length")
        current_input_length = []
        for batch_idx, input_id in enumerate(raw_input_ids):
            input_id = torch.as_tensor(input_id, dtype=torch.long)
            seq_length = input_id.shape[0]
            past_seq_length = data["past_seq_length"][batch_idx]
            current_input_length.append(
                int(requested_lengths[batch_idx]) if requested_lengths is not None else seq_length
            )
            assert (
                seq_length <= self.input_sequence_length
            ), (
                f"Input sequence length is too long. max input sequence length is "
                f"{self.input_sequence_length} but got {seq_length}"
            )
            if self.input_sequence_length > seq_length and out_padding:
                padding_input_ids = torch.zeros(
                    (self.input_sequence_length - seq_length), dtype=torch.long, device=input_id.device
                )
                padding_input_ids.fill_(self.pad_token_id)
                input_id = torch.cat([input_id, padding_input_ids], dim=-1)

            input_id = input_id.unsqueeze(0)
            input_ids.append(input_id)

        input_ids = torch.cat(input_ids, dim=0).to(device)

        current_input_length = torch.tensor(current_input_length, dtype=torch.int32).to(device)

        self.token_embedding.to(device)
        inputs_embeds = self.token_embedding(input_ids)
        past_seq_length = data["past_seq_length"]
        past_seq_length = torch.tensor(past_seq_length, dtype=torch.int32).to(device)
        assert torch.all(past_seq_length >= 0)
        if torch.any(current_input_length <= 0) or torch.any(current_input_length > self.input_sequence_length):
            raise ValueError("PI05 Gemma current_input_length is outside the static input sequence")
        if torch.any(past_seq_length + current_input_length > self.cache_length):
            raise ValueError("PI05 Gemma cache write exceeds cache capacity")
        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches
        if "attention_mask" not in data:
            attention_mask = torch.full(
                (1, 1, 1, self.cache_length),
                torch.finfo(torch.float16).min,
                dtype=torch.float16,
                device=device,
            )
            attention_mask[..., : int((past_seq_length + current_input_length).max().item())] = 0
        else:
            attention_mask = data['attention_mask']

        return (
            inputs_embeds.to(device),
            past_seq_length.to(device),
            current_input_length,
            attention_mask,
            past_key_caches,
            past_value_caches,
        )

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        inputs_embeds, past_seq_length, seg_length, attention_mask, past_key_caches, past_value_caches = (
            self.prepare_inputs(data)
        )
        return (
            inputs_embeds,
            past_seq_length,
            seg_length,
            attention_mask,
            past_key_caches,
            past_value_caches,
        )

    def _forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        attention_mask: Tensor,
        past_key_caches: List[Tensor],
        past_value_caches: List[Tensor],
    ):
        last_hidden_state = self(
            inputs_embeds,
            past_seq_length,
            current_input_length,
            attention_mask,
            past_key_caches,
            past_value_caches,
        )
        return BaseModelOutputWithPast(
            hidden_states=last_hidden_state,
        )
