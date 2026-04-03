from typing import List, Optional, Union

import torch
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama import LlamaForCausalLM

from ..base_llm_model import LLMBaseModel
from ..builder import MODELS


@MODELS.register_module()
class XHLlamaModel(LLMBaseModel):
    def __init__(
        self,
        hf_model: Union[LlamaForCausalLM, str],
        wrap_cfg,
        quant_config,
        frontend_type,
        allow_quant=True,
        export_cfg=None,
    ):
        super().__init__(
            hf_model,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            frontend_type=frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )

    def _set_dtype(self, dtype):
        self.token_embedding = self.token_embedding.to(dtype)
        return super()._set_dtype(dtype)

    def init_wrap_model(self, hf_model=None):
        from ._model import register_wrap_modules

        register_wrap_modules()

        wrap_model = super().init_wrap_model(hf_model)
        hf_model = wrap_model
        self.token_embedding = hf_model.model.get_input_embeddings()
        self.generation_config = hf_model.generation_config
        self.config = hf_model.config
        self.num_hidden_layers = hf_model.model.config.num_hidden_layers
        head_dim = hf_model.model.config.hidden_size // hf_model.model.config.num_attention_heads
        # self.pad_token_id = hf_model.config.eos_token_id[0]
        self.pad_token_id = hf_model.config.pad_token_id
        batch_size = self.wrap_cfg.batch_size
        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            if only_first_block:
                num_decoder_layers = 1
            self.prepare_kv_cache(
                num_decoder_layers,
                [batch_size, hf_model.model.config.num_key_value_heads, self.cache_length, head_dim],
            )

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        raw_input_ids: List[List[int]] = data["input_ids"]
        assert self.token_embedding is not None, "Token embedding is not available."

        device = self.execution_device
        input_ids = []
        current_input_length = []
        position_ids = []
        for batch_idx, input_id in enumerate(raw_input_ids):
            input_id = torch.tensor(input_id, dtype=torch.int32)
            seq_length = input_id.shape[0]
            past_seq_length = data["past_seq_length"][batch_idx]
            position_id = torch.arange(past_seq_length, past_seq_length + seq_length, dtype=torch.int32).to(input_id.device)
            current_input_length.append(seq_length)
            assert (
                seq_length <= self.input_sequence_length
            ), f"Input sequence length is too long. max input sequence length is {self.input_sequence_length} but got {seq_length}"
            if self.input_sequence_length > seq_length:
                padding_input_ids = torch.zeros(
                    (self.input_sequence_length - seq_length), dtype=torch.int32, device=input_id.device
                )
                padding_input_ids.fill_(self.pad_token_id)
                input_id = torch.cat([input_id, padding_input_ids], dim=-1)

                padding_position_id = torch.ones(
                    (self.input_sequence_length - seq_length), dtype=torch.int32, device=input_id.device
                )
                position_id = torch.cat([position_id, padding_position_id], dim=-1)

            input_id = input_id.unsqueeze(0)
            position_id = position_id.unsqueeze(0)
            position_ids.append(position_id)
            input_ids.append(input_id)

        input_ids = torch.cat(input_ids, dim=0).to(device)
        position_ids = torch.cat(position_ids, dim=0).to(device)

        current_input_length = torch.tensor(current_input_length, dtype=torch.int32).to(device)

        self.token_embedding.to(device)
        inputs_embeds = self.token_embedding(input_ids)
        past_seq_length = data["past_seq_length"]
        past_seq_length = torch.tensor(past_seq_length, dtype=torch.int32).to(device)
        assert torch.all(past_seq_length >= 0)
        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches
        attention_mask = torch.ones(
                        (1, 1, self.input_sequence_length, 1024),
                        dtype=torch.float16,
                    )

        return (
            inputs_embeds.to(device),
            past_seq_length.to(device),
            current_input_length,
            position_ids.to(device),
            attention_mask.to(device),
            past_key_caches,
            past_value_caches,
        )

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        inputs_embeds, past_seq_length, seg_length, position_ids, attention_mask, past_key_caches, past_value_caches = (
            self.prepare_inputs(data)
        )
        return (
            inputs_embeds,
            past_seq_length,
            seg_length,
            position_ids,
            attention_mask,
            past_key_caches,
            past_value_caches,
        )

    def _forward(
        self,
        inputs_embeds: Optional[Tensor] = None,
        past_seq_length: Tensor = None,
        current_input_length: Tensor = None,
        position_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        past_key_caches: Optional[List[Tensor]] = None,
        past_value_caches: Optional[List[Tensor]] = None,
    ):

        hidden_states = self(
            inputs_embeds,
            past_seq_length,
            current_input_length,
            position_ids,
            attention_mask,
            past_key_caches,
            past_value_caches,
        )
        return CausalLMOutputWithPast(
            hidden_states=hidden_states,
        )
