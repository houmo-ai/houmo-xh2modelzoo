from typing import Optional, Union

import torch
from transformers.cache_utils import Cache, DynamicCache, HybridCache
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM, GemmaModel
from typing_extensions import Self

from ..base_llm_model import BaseModel

from ..llm_onnx_model import LLMONNXModel

def create_expert_wraped_cls(cls):
    class _GemmaForCausalLM(cls):
        
        @property
        def prefill(self) -> bool:
            return self._prefill

        @prefill.setter
        def prefill(self, value: bool) -> None:
            self._prefill = value
            
        def generate(self, past_seq_length, past_key_caches, past_value_caches, *args, **kwargs):
            self._past_seq_length = past_seq_length
            self.past_key_caches = past_key_caches
            self.past_value_caches= past_value_caches
            output = self.forward(*args, **kwargs)
            return output

        def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[HybridCache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            cond: Optional[torch.LongTensor] = None,
            **kwargs,
        ) -> BaseModelOutputWithPast:

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )

            use_cache = use_cache if use_cache is not None else self.config.use_cache

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if not isinstance(past_key_values, (type(None), Cache)):
                raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

            if use_cache and past_key_values is None:
                past_key_values = DynamicCache()

            data = {}
            data["inputs_embeds"] = inputs_embeds
            data["attention_mask"] = attention_mask
            seq_length = inputs_embeds.shape[1]

            past_seq_length = [self._past_seq_length]

            data["past_seq_length"] = past_seq_length
            input_sequence_length = self._llm_model.get_input_sequence_length()
            
            if cond is not None:
                data["cond"] = cond

            pad_input_seq_length = (
                (seq_length + input_sequence_length - 1) // input_sequence_length
            ) * input_sequence_length
            self._llm_model.set_input_sequence_length(pad_input_seq_length)
            (
                inputs_embeds,
                past_seq_length,
                current_input_length,
                cond,
                attention_mask,
                past_key_caches,
                past_value_caches,
            ) = self._llm_model.prepare_inputs(data)
            self._llm_model.set_input_sequence_length(input_sequence_length)
            past_key_caches = self.past_key_caches
            past_value_caches = self.past_value_caches
            pad_seq_lenght = inputs_embeds.shape[1]
            assert (
                pad_seq_lenght % input_sequence_length == 0
            ), "pad_seq_lenght must be divisible by input_sequence_length"
            steps = pad_seq_lenght // input_sequence_length
            for i in range(steps):
                start = i * input_sequence_length
                end = (i + 1) * input_sequence_length
                sub_inputs_embeds = inputs_embeds[:, start:end, :]
                sub_past_seq_length = past_seq_length + start
                sub_current_input_length = torch.tensor(
                    [min(end, seq_length) - start], dtype=current_input_length.dtype
                ).to(current_input_length.device)
                outputs = self._llm_model(
                    sub_inputs_embeds,
                    sub_past_seq_length,
                    sub_current_input_length,
                    cond,
                    attention_mask,
                    past_key_caches,
                    past_value_caches,
                )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return outputs

    return _GemmaForCausalLM

class GemmaCHFCompatible(GemmaForCausalLM):
    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")

    def set_input_sequence_length(self, input_seq_length: int) -> None:
        self._llm_model.set_input_sequence_length(input_seq_length)
    
    def __setup__(self, llm_model: LLMONNXModel) -> Self:
        """
        初始化模型
        """
        self._prefill = True
        self._llm_model = llm_model
        self.embed_tokens = llm_model.token_embedding
        self._past_seq_length = 0
        return self

    def get_output_embeddings(self):
        """
        lm_eval需要这个接口
        """
        return None

    @classmethod
    def to_hf_compatible(
        cls,
        policy: Union[GemmaModel, str],
        expert_llm_model: Optional[LLMONNXModel] = None,
    ) -> Self:
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        expert = policy.paligemma_with_expert.gemma_expert
        # hf_model_expert = policy.paligemma_with_expert
        if expert_llm_model is not None:
            assert isinstance(expert_llm_model, LLMONNXModel)
            _llm_cls = create_expert_wraped_cls(type(expert))
            expert.__class__ = _llm_cls
            expert._llm_model = expert_llm_model
            expert._prefill = True
            expert._past_seq_length = 0
            # hf_model.embed_tokens = hf_model.model.embed_tokens
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return expert
