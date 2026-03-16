from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor

# import transformers_modules
from transformers import AutoTokenizer

from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

from ..base_llm_model import LLMBaseModel
from ..builder import MODELS
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy
from lerobot.policies.xvla.modeling_florence2 import Florence2Encoder
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors

@MODELS.register_module()
class XHFlorence2EncoderLLMModel(LLMBaseModel):
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

    def get_hf_model(self, device_map="cpu", **kwargs) -> Florence2Encoder:
        assert self.hf_model_dir is not None
        config = PreTrainedConfig.from_pretrained(self.hf_model_dir)
        policy = XVLAPolicy.from_pretrained(self.hf_model_dir, config=config, strict=False)
        encoder = policy.model.vlm.language_model.model.encoder.eval()
        encoder = encoder.half()
        return encoder
    
    def get_tokenizer(self,tokenizer_path=None):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        return tokenizer

    def init_wrap_model(self, hf_model: Optional[Florence2Encoder] = None):

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
                [batch_size, _llm.config.encoder_attention_heads, self.cache_length, head_dim],
            )
        _llm = None

    def prepare_inputs(self, data: Union[dict, tuple, list], out_padding=True):
        device = self.execution_device
        if "inputs_embeds" in data:
            inputs_embeds = data["inputs_embeds"].to(device)
        elif "input_ids" in data:
            # 简化处理，假设 input_ids 已经 pad 好或者不需要复杂处理
            input_ids = data["input_ids"]
            if isinstance(input_ids, list):
                 input_ids = torch.tensor(input_ids, dtype=torch.long)
            input_ids = input_ids.to(device)
            self.token_embedding.to(device)
            inputs_embeds = self.token_embedding(input_ids)
        else:
             raise ValueError("data must contain inputs_embeds or input_ids")

        if "attention_mask" in data:
            attention_mask = data['attention_mask'].to(device)
        else:
            # Default mask if not provided
            attention_mask = torch.ones((inputs_embeds.shape[0], inputs_embeds.shape[1]), device=device, dtype=inputs_embeds.dtype)

        return (
            None, # input_ids
            attention_mask,
            None, # head_mask
            inputs_embeds
        )

    def prepare_inputs_for_graph(self, data: Union[dict, tuple, list]):
        return self.prepare_inputs(data)

    def _forward(
        self,
        input_ids: Optional[Tensor],
        attention_mask: Optional[Tensor],
        head_mask: Optional[Tensor],
        inputs_embeds: Optional[Tensor],
    ):
        # Call wrap_model directly
        outputs = self.wrap_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        # Check output type
        if isinstance(outputs, tuple):
             hidden_states = outputs[0]
        else:
             hidden_states = outputs.last_hidden_state

        return BaseModelOutputWithPast(
            hidden_states=hidden_states,
        )