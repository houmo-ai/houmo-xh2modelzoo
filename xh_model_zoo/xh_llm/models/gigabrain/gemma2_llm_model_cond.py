from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

# import transformers_modules
from transformers import GemmaModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from xhquant.api import get_root_logger
from xhquant.core import HybridCacheTensor

from ..base_llm_model import LLMBaseModel
from ..builder import MODELS

@MODELS.register_module()
class XHGemma2CondLLMModel(LLMBaseModel):
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
        """加载 GigaBrain 模型"""
        assert self.hf_model_dir is not None
        
        # 添加 giga_models 路径
        import sys
        giga_models_path = "/data01/home/she.gao/vla/giga/giga-models"
        if giga_models_path not in sys.path:
            sys.path.insert(0, giga_models_path)
            
        from giga_models.models import GigaBrain0Policy
        
        policy = GigaBrain0Policy.from_pretrained(
            self.hf_model_dir,
            torch_dtype=torch.float32,
        )
        policy.eval()
        return policy
    
    def get_tokenizer(self, config_dir: str):
        # tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224")
        tokenizer = AutoTokenizer.from_pretrained(config_dir)
        return tokenizer

    def init_wrap_model(self, hf_model: Optional[GemmaModel] = None):

        from ._llm_model_impl_cond import register_wrap_cls as llm_register_wrap_cls
        llm_register_wrap_cls(hf_model)
        super().init_wrap_model(hf_model)
        
        _llm = self.wrap_model
        self.config = _llm.config
        self.token_embedding = _llm.model.embed_tokens
        self.generation_config = _llm.generation_config
        self.num_hidden_layers = _llm.config.num_hidden_layers
        head_dim = _llm.model.layers[0].self_attn.head_dim
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

    # def prepare_kv_cache(self, num_decoder_layers, kv_cache_shape):
    #     self.past_key_caches = []
    #     self.past_value_caches = []
    #     if self.use_cache:
    #         for i in range(num_decoder_layers):
    #             layer_kv_cache_shape = kv_cache_shape
    #             self.past_key_caches.append(HybridCacheTensor(torch.zeros(layer_kv_cache_shape, dtype=torch.float16)))
    #             self.past_value_caches.append(HybridCacheTensor(torch.zeros(layer_kv_cache_shape, dtype=torch.float16)))

    #         for layer_idx in range(num_decoder_layers):
    #             self.export_cfg.input_names.append(f"past_key_cache_{layer_idx}")
    #         for layer_idx in range(num_decoder_layers):
    #             self.export_cfg.input_names.append(f"past_value_cache_{layer_idx}")
    def prepare_inputs(self, data: Union[dict, tuple, list], out_padding=True):
        raw_input_ids: List[List[int]] = data["input_ids"]
        assert self.token_embedding is not None, "Token embedding is not available."

        device = self.execution_device
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
        past_seq_length = data["past_seq_length"]
        past_seq_length = torch.tensor(past_seq_length, dtype=torch.int32).to(device)
        assert torch.all(past_seq_length >= 0)
        past_key_caches = self.past_key_caches
        past_value_caches = self.past_value_caches
        torch.manual_seed(42)
        cond = torch.randn(1, 1024).to(torch.bfloat16).to(device)
        if "attention_mask" in data:
            attention_mask = data['attention_mask']
        else:
            # attention_mask = torch.zeros((1, 8, 50, 1024), dtype=torch.bfloat16, device=device)
            attention_mask = torch.full(
                (1, 1, 50, 2048),
                torch.finfo(torch.float16).min,
                dtype=torch.float16,
                device=device,
            )

            attention_mask[..., :50] = 0.0

        # inputs_embeds = torch.load("/data01/home/she.gao/lerobot/suffix_embs.pt").to(torch.float32)
        # cond = torch.load("/data01/home/she.gao/xhquant_llm/examples/cond.pt")
        # cond = cond.to(torch.float16)
        cond = torch.ones(1, 1024, device=device, dtype=torch.float32)
        return (
            inputs_embeds.to(device),
            past_seq_length.to(device),
            current_input_length,
            cond,
            attention_mask,
            past_key_caches,
            past_value_caches,
        )

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        inputs_embeds, past_seq_length, seg_length, cond, attention_mask, past_key_caches, past_value_caches = (
            self.prepare_inputs(data)
        )
        return (
            inputs_embeds,
            past_seq_length,
            seg_length,
            cond,
            attention_mask,
            past_key_caches,
            past_value_caches,
        )

    def _forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        cond: Tensor,
        attention_mask: Tensor,
        past_key_caches: List[Tensor],
        past_value_caches: List[Tensor],
    ):
        last_hidden_state = self(
            inputs_embeds,
            past_seq_length,
            current_input_length,
            cond,
            attention_mask,
            past_key_caches,
            past_value_caches,
            
        )
        return CausalLMOutputWithPast(
            hidden_states=last_hidden_state,
        )