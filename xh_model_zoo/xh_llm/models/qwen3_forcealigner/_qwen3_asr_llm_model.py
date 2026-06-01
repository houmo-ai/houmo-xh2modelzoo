from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch import Tensor

from ._llm_model_impl import _Qwen3ASRThinkerTextModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from ..base_llm_model import LLMBaseModel
from ..builder import MODELS

from xh_model_zoo.xh_llm.models.qwen3_forcealigner.modeling_qwen3_asr import (
    Qwen3ASRForConditionalGeneration
)

from qwen_asr.core.transformers_backend import (
    Qwen3ASRProcessor
)

@MODELS.register_module()
class XHQwen3ASRLLMModel(LLMBaseModel):
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

    def get_hf_model(self, device_map="cpu", **kwargs):
        if isinstance(self.hf_model_dir, nn.Module):
            return self.hf_model_dir
        hf_model = Qwen3ASRForConditionalGeneration.from_pretrained(
            self.hf_model_dir,
            dtype=torch.float16,
            device_map=device_map,
        ).eval()
        hf_model.config.forced_decoder_ids = None
        hf_model.config._attn_implementation = "eager"
        return hf_model
    
    def get_tokenizer(self):
        processor = Qwen3ASRProcessor.from_pretrained(self.hf_model_dir, fix_mistral_regex=True)
        return processor.tokenizer

    def get_processor(self):
        processor = Qwen3ASRProcessor.from_pretrained(self.hf_model_dir, fix_mistral_regex=True)
        return processor

    def init_wrap_model(self, hf_model: Optional[_Qwen3ASRThinkerTextModel] = None):

        from ._llm_model_impl import register_wrap_cls as llm_register_wrap_cls
        llm_register_wrap_cls(hf_model)
        super().init_wrap_model(hf_model)

        _llm = self.wrap_model
        self.config = _llm.config
        self.token_embedding = _llm.embed_tokens
        self.generation_config = _llm.generation_config
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
        device = self.execution_device
        
        if "input_ids" in data:
            raw_input_ids: List[List[int]] = data["input_ids"]
            assert self.token_embedding is not None, "Token embedding is not available."

            input_ids = []
            current_input_length = []
            for batch_idx, input_id in enumerate(raw_input_ids):
                input_id = torch.tensor(input_id, dtype=torch.long)
                seq_length = input_id.shape[0]
                past_seq_length = data["past_seq_length"][batch_idx]
                current_input_length.append(seq_length)
                assert (
                    seq_length <= self.input_sequence_length
                ), f"Input sequence length is too long. max input sequence length is {self.input_sequence_length} but got {seq_length}"
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
            
        elif "input_embeds" in data:
            inputs_embeds = data["input_embeds"]
            current_input_length = inputs_embeds.shape[1]
            current_input_length = torch.tensor(current_input_length, dtype=torch.int32).to(device)
            device = self.execution_device
            inputs_embeds = inputs_embeds.to(device)
            
        past_seq_length = data["past_seq_length"]
        past_seq_length = torch.tensor(past_seq_length, dtype=torch.int32).to(device)
        assert torch.all(past_seq_length >= 0)
        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches
        # past_key_caches  = []
        # past_value_caches = []
        # for i in range(self.num_hidden_layers):
        #     past_key_caches.append(getattr(self, f"past_k_cache_{i}"))
        #     past_value_caches.append(getattr(self, f"past_v_cache_{i}"))
        # breakpoint()
        return (
            inputs_embeds.to(device),
            past_seq_length.to(device),
            current_input_length.unsqueeze(0).to(device),
            past_key_caches,
            past_value_caches,
        )

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        inputs_embeds, past_seq_length, seg_length, past_key_caches, past_value_caches = (
            self.prepare_inputs(data)
        )
        return (
            inputs_embeds,
            past_seq_length,
            seg_length,
            past_key_caches,
            past_value_caches,
        )

    def _forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: List[Tensor],
        past_value_caches: List[Tensor],
    ):
        last_hidden_state = self(
            inputs_embeds,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        )
        return BaseModelOutputWithPast(
            hidden_states=last_hidden_state,
        )