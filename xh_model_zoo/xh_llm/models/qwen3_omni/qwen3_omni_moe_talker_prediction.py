from copy import deepcopy
from typing import List, Optional, Union, cast

import torch
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..base_llm_model import LLMBaseModel
from ..builder import MODELS
from .modeling_qwen3_omni_moe import Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration


@MODELS.register_module()
class XHQwen3OmniMoeTalkerPrediction(LLMBaseModel):
    def __init__(
        self,
        hf_model: str,
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

    def get_input_embeddings(self):
        return self.token_embedding

    def init_wrap_model(self, hf_model: Optional[Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration] = None):
        from ._talker_prediction import register_wrap_modules as qwen3omni_register_wrap_modules

        qwen3omni_register_wrap_modules()

        super().init_wrap_model(hf_model)
        hf_model = cast(Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration, self.wrap_model)

        self.codec_embedding = deepcopy(hf_model.model.get_input_embeddings())
        # Keep naming consistent with LLMBaseModel conventions.
        self.token_embedding = self.codec_embedding
        self.generation_config = hf_model.generation_config
        self.config = hf_model.config
        self.num_hidden_layers = hf_model.model.config.num_hidden_layers
        self.head_dim = hf_model.model.layers[0].self_attn.head_dim
        pad_token_id = getattr(hf_model.config, "pad_token_id", None)
        self.pad_token_id = pad_token_id if pad_token_id is not None else hf_model.config.eos_token_id
        batch_size = self.wrap_cfg.batch_size
        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            if only_first_block:
                num_decoder_layers = 1
            self.prepare_kv_cache(
                num_decoder_layers,
                [batch_size, hf_model.model.config.num_key_value_heads, self.cache_length, self.head_dim],
            )

        hf_model = None

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        device = self.execution_device
        
        # 支持直接输入 embeddings 或 input_ids
        if "inputs_embeds" in data:
            # 直接使用输入的 embeddings
            raw_inputs_embeds = data["inputs_embeds"]
            embeddings_list = []
            current_input_length = []
            position_ids = []
            
            for batch_idx, input_embed in enumerate(raw_inputs_embeds):
                if isinstance(input_embed, list):
                    input_embed = torch.tensor(input_embed, dtype=torch.float32)
                elif not isinstance(input_embed, torch.Tensor):
                    input_embed = torch.tensor(input_embed, dtype=torch.float32)
                
                # input_embed shape: [seq_length, hidden_size]
                seq_length = input_embed.shape[0]
                past_seq_length = data["past_seq_length"][batch_idx]
                position_id = torch.arange(
                    past_seq_length, past_seq_length + seq_length, dtype=torch.long, device=input_embed.device
                )
                current_input_length.append(seq_length)
                
                assert (
                    seq_length <= self.input_sequence_length
                ), f"Input sequence length is too long. max input sequence length is {self.input_sequence_length} but got {seq_length}"
                
                if self.input_sequence_length > seq_length:
                    # 对 embeddings 进行 padding
                    hidden_size = input_embed.shape[-1]
                    padding_embeds = torch.zeros(
                        (self.input_sequence_length - seq_length, hidden_size),
                        dtype=input_embed.dtype,
                        device=input_embed.device
                    )
                    input_embed = torch.cat([input_embed, padding_embeds], dim=0)
                    
                    padding_position_id = torch.ones(
                        (self.input_sequence_length - seq_length), dtype=torch.long, device=input_embed.device
                    )
                    position_id = torch.cat([position_id, padding_position_id], dim=-1)
                
                input_embed = input_embed.unsqueeze(0)
                position_id = position_id.unsqueeze(0)
                position_ids.append(position_id)
                embeddings_list.append(input_embed)
            
            inputs_embeds = torch.cat(embeddings_list, dim=0).to(device)
            position_ids = torch.cat(position_ids, dim=0).to(device)
            current_input_length = torch.tensor(current_input_length, dtype=torch.int32).to(device)
            
        else:
            # 原有的 input_ids 逻辑
            raw_input_ids: List[List[int]] = data["input_ids"]
            assert self.token_embedding is not None, "Token embedding is not available."
            
            input_ids = []
            current_input_length = []
            position_ids = []
            for batch_idx, input_id in enumerate(raw_input_ids):
                input_id = torch.tensor(input_id, dtype=torch.long)
                seq_length = input_id.shape[0]
                past_seq_length = data["past_seq_length"][batch_idx]
                position_id = torch.arange(
                    past_seq_length, past_seq_length + seq_length, dtype=torch.long, device=input_id.device
                )
                current_input_length.append(seq_length)
                assert (
                    seq_length <= self.input_sequence_length
                ), f"Input sequence length is too long. max input sequence length is {self.input_sequence_length} but got {seq_length}"
                if self.input_sequence_length > seq_length:
                    padding_input_ids = torch.zeros(
                        (self.input_sequence_length - seq_length), dtype=torch.long, device=input_id.device
                    )
                    padding_input_ids.fill_(self.pad_token_id)
                    input_id = torch.cat([input_id, padding_input_ids], dim=-1)

                    padding_position_id = torch.ones(
                        (self.input_sequence_length - seq_length), dtype=torch.long, device=input_id.device
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

        return (
            inputs_embeds.to(device),
            past_seq_length.to(device),
            current_input_length,
            # position_ids.to(device),
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
            # position_ids,
            past_key_caches,
            past_value_caches,
        )

    def _forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        position_ids: Optional[Tensor],
        past_key_caches: List[Tensor],
        past_value_caches: List[Tensor],
    ):
        logits = self(
            inputs_embeds,
            past_seq_length,
            current_input_length,
            position_ids,
            past_key_caches,
            past_value_caches,
        )
        return CausalLMOutputWithPast(
            logits=logits,
        )
