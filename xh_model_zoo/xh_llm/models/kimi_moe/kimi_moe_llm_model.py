from copy import deepcopy
from typing import List, Optional, Union, cast

import torch
from torch import Tensor
# from transformers.modeling_outputs import CausalLMOutputWithPast
# from transformers.models.qwen3_moe import Qwen3MoeForCausalLM
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from ..base_llm_model import LLMBaseModel
from ..builder import MODELS


@MODELS.register_module()
class XHKimiMoeModel(LLMBaseModel):
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
    
    def get_tokenizer(self, **kwargs):
        assert self.hf_model_dir is not None
        tokenizer = AutoTokenizer.from_pretrained(self.hf_model_dir, trust_remote_code=True, **kwargs)
        return tokenizer

    def get_hf_model(self, device_map="cpu", **kwargs):
        assert self.hf_model_dir is not None
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.hf_model_dir,
            # dtype=torch.float16,
            # torch_dtype=torch.float16,
            trust_remote_code=True,
            device_map=device_map,
            # low_cpu_mem_usage=True,
            **kwargs,
        ).eval()
        if hf_model.config.tie_word_embeddings:
            hf_model.config.torchscript = True
            hf_model.tie_weights()
            hf_model.config.tie_word_embeddings = False
            hf_model.config.torchscript = False

        hf_model = self.untied_weights(hf_model)
        import accelerate
        import accelerate.hooks

        accelerate.hooks.remove_hook_from_module(hf_model)

        hf_model = self.dequantize_hf_model(hf_model)
        return hf_model


    def init_wrap_model(self, hf_model=None):
        from ._moe_model import register_wrap_modules as kimimoe_register_wrap_modules

        kimimoe_register_wrap_modules()

        super().init_wrap_model(hf_model)
        hf_model = self.wrap_model

        # self.token_embedding.weight 和 lm_head.weight 是相同对象
        self.token_embedding = deepcopy(hf_model.model.get_input_embeddings())
        self.generation_config = hf_model.generation_config
        self.config = hf_model.config
        self.num_hidden_layers = hf_model.model.config.num_hidden_layers
        self.head_dim = hf_model.model.layers[0].self_attn.head_dim
        self.pad_token_id = hf_model.config.eos_token_id
        batch_size = self.wrap_cfg.batch_size
        q_head_dim = hf_model.model.layers[0].self_attn.q_head_dim
        v_head_dim = hf_model.model.layers[0].self_attn.v_head_dim
        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            print("only_first_block:", only_first_block)
            if only_first_block:
                num_decoder_layers = 1
            self.prepare_kv_cache(
                num_decoder_layers,
                [batch_size, hf_model.model.config.num_key_value_heads, self.cache_length, q_head_dim],
                [batch_size, hf_model.model.config.num_key_value_heads, self.cache_length, v_head_dim],
            )
        hf_model = None

    def prepare_inputs(self, data: Union[dict, tuple, list]):
        raw_input_ids: List[List[int]] = data["input_ids"]
        assert self.token_embedding is not None, "Token embedding is not available."

        device = self.execution_device
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
            position_ids.to(device),
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

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        inputs_embeds, past_seq_length, seg_length, position_ids, past_key_caches, past_value_caches = (
            self.prepare_inputs(data)
        )
        return (
            inputs_embeds,
            past_seq_length,
            seg_length,
            position_ids,
            past_key_caches,
            past_value_caches,
        )