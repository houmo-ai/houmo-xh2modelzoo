import math
import sys
import types
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import transformers
import transformers.models
from torch import Tensor
from xhquant.api import ConfigDict
from xhquant.utils import digit_version
from xhquant import nn as xhnn
from xhquant.nn import LLMCache, MaskedSoftmax, RMSNorm, Rope
from xhquant.utils.registry import DynamicModule
import torch.nn.functional as F
from ..builder import XHLLM_TRACEABLE_MODULES

from transformers.models.bert.modeling_bert import BertModel, BertEncoder, BertLayer, BertSdpaSelfAttention, BertForMaskedLM
from transformers.cache_utils import Cache

@XHLLM_TRACEABLE_MODULES.register_module({BertModel: "BertModel"})
class _BertModel(DynamicModule):
    def forward(
        self,
        token_embedding: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.Tensor] = None,
        token_type_embeddings: Optional[torch.Tensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
    ):
        output_attentions = False
        output_hidden_states = False
        use_cache = False


        # batch_size, seq_length = input_ids.size()
        # device = input_ids.device if input_ids is not None else inputs_embeds.device
        # past_key_values_length = 0

        # embedding_output = self.embeddings(
        #     input_ids=input_ids,
        #     position_ids=position_ids, # none
        #     token_type_ids=token_type_ids, # [0,0,0,0]
        #     inputs_embeds=inputs_embeds, # none
        #     past_key_values_length=past_key_values_length, # 0
        # )

        embedding = token_embedding + token_type_embeddings + position_embeddings
        embedding_output = self.embeddings.LayerNorm(embedding)

        # attention_mask = torch.ones((batch_size, seq_length + past_key_values_length), device=device)
        extended_attention_mask = None
        encoder_extended_attention_mask = None
        head_mask = None

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache, # false
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        # sequence_output = encoder_outputs[0]

        return encoder_outputs
    def _setup(self, cfg: Optional[Dict] = None):
        return self


@XHLLM_TRACEABLE_MODULES.register_module({BertEncoder: "BertEncoder"})
class _BertEncoder(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        cache_position: Optional[torch.Tensor] = None,            
    ):
        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                None,
                encoder_hidden_states,  # as a positional argument for gradient checkpointing
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]

        return hidden_states
    
    def _setup(self, cfg: Optional[Dict] = None):
        return self    


@XHLLM_TRACEABLE_MODULES.register_module({BertLayer: "BertLayer"})
class _BertLayer(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ):
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = self.feed_forward_chunk(attention_output)
        
        outputs = (layer_output,) + outputs
        return outputs      

    def _setup(self, cfg: Optional[Dict] = None):
        return self      

@XHLLM_TRACEABLE_MODULES.register_module({BertSdpaSelfAttention: "BertSdpaSelfAttention"})
class _BertSdpaSelfAttention(DynamicModule):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        cache_position: Optional[torch.Tensor] = None,
    ):
        bsz = 1
        # bsz, tgt_len, _ = hidden_states.size()

        query_layer = self.query(hidden_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        
        current_states = hidden_states


        key_layer = self.key(current_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        value_layer = self.value(current_states).view(bsz, -1, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        query = query_layer * self.kv_scale
        key = key_layer.transpose(2, 3)
        attn_weights = torch.matmul(query, key) 
        attn_weights = self.maskedadd(attn_weights, attention_mask)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_layer) # [1, 1, 16384, 512]

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, -1, self.all_head_size)

        return attn_output, None     

    def _setup(self, cfg: Optional[Dict] = None):
        _kv_scale = 1 / math.sqrt(self.attention_head_size)
        self.register_buffer("kv_scale", torch.tensor(_kv_scale, dtype=torch.float16), persistent=False)

        self.maskedadd = xhnn.MaskedAdd()
        return self   

@XHLLM_TRACEABLE_MODULES.register_module({BertForMaskedLM: "BertForMaskedLM"})
class _BertForMaskedLM(DynamicModule):
    def forward(
        self,
        token_embedding: Optional[torch.Tensor] = None,
        token_type_embeddings = None,
        position_embeddings = None,        
        attention_mask: Optional[torch.Tensor] = None,
        # token_type_ids: Optional[torch.Tensor] = None,
        # position_ids: Optional[torch.Tensor] = None,
        # head_mask: Optional[torch.Tensor] = None,
        # inputs_embeds: Optional[torch.Tensor] = None,
        # encoder_hidden_states: Optional[torch.Tensor] = None,
        # encoder_attention_mask: Optional[torch.Tensor] = None,
        # labels: Optional[torch.Tensor] = None,
        # output_attentions: Optional[bool] = None,
        # output_hidden_states: Optional[bool] = None,
        # return_dict: Optional[bool] = None,
    ):
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            token_embedding,
            attention_mask=attention_mask,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=False,
            token_type_embeddings=token_type_embeddings,
            position_embeddings=position_embeddings,
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        return prediction_scores      
            
    
    def _setup(self, cfg: Optional[Dict] = None):
        return self   

def register_wrap_modules(hf_model = None):
    pass