from typing import Any, Optional, Union

import torch
import torch.nn as nn
from typing_extensions import Self
from accelerate import init_empty_weights
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, Qwen3ForCausalLM
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import no_init_weights

# from transformers.models.qwen2.modeling_qwen2 import KwargsForCausalLM
from xh_model_zoo_new.core.converter import Demo

from ..base_llm_model import BaseModel
from ..builder import LLM_DYNAMIC_MODULES
from ..common import LLM_HFCompatible


def get_empty_hf_model(hf_model_dir, device_map="cpu", **kwargs) -> Any:
    """
    仅仅加载模型结构,不初始化权重,不占用显存
    """
    config = AutoConfig.from_pretrained(hf_model_dir)
    with no_init_weights(), init_empty_weights():
        hf_model: nn.Module = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.float16,
            **kwargs,
        )
    return hf_model


class Qwen3Legacy_HFCompatible(Qwen3ForCausalLM):
    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")

    def __setup__(self, llm_model: BaseModel) -> Self:
        """"""
        """
        初始化模型
        """
        self._prefill = True
        self._llm_model = llm_model
        self.embed_tokens = llm_model.token_embedding
        self._past_seq_length = 0
        self._dynamic_input: bool = False
        return self

    @property
    def dynamic_input(self) -> bool:
        return self._dynamic_input

    @dynamic_input.setter
    def dynamic_input(self, value: bool):
        self._dynamic_input = value

    @dynamic_input.setter
    def dynamic_input(self, value: bool) -> None:
        self._dynamic_input = value

    def get_output_embeddings(self):
        """
        lm_eval需要这个接口
        """
        return None

    @property
    def prefill(self) -> bool:
        return self._prefill

    @prefill.setter
    def prefill(self, value: bool) -> None:
        self._prefill = value

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
        if past_key_values is None:
            pass

        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-8B")
        >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # if self.gradient_checkpointing and self.training and use_cache:
        #     # logger.warning_once(
        #     #     "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        #     # )
        #     use_cache = False

        # TODO (joao): remove this exception in v4.56 -- it exists for users that try to pass a legacy cache
        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        assert inputs_embeds.shape[0] == 1
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )

        # # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # outputs: BaseModelOutputWithPast = self.model(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     position_ids=position_ids,
        #     past_key_values=past_key_values,
        #     inputs_embeds=inputs_embeds,
        #     use_cache=use_cache,
        #     output_attentions=output_attentions,
        #     output_hidden_states=output_hidden_states,
        #     cache_position=cache_position,
        #     **kwargs,
        # )

        # hidden_states = outputs.last_hidden_state
        # # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        # slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # logits = self.lm_head(hidden_states[:, slice_indices, :])
        past_seq_length = torch.tensor([self._past_seq_length], dtype=torch.int32).to(inputs_embeds.device)

        # TODO: 需要根据attention mask计算seq_length
        seq_length = input_ids.shape[-1]
        current_input_length = torch.tensor([seq_length], dtype=torch.int32).to(inputs_embeds.device)

        past_key_caches = self._llm_model.past_value_caches
        past_value_caches = self._llm_model.past_key_caches

        if past_key_values is None or (hasattr(self, "use_cache") and not self.use_cache):
            past_key_caches = [torch.tensor([])] * len(past_key_caches)
            past_value_caches = [torch.tensor([])] * len(past_value_caches)

        if self.dynamic_input and isinstance(self._llm_model, BaseModel):
            self._llm_model.set_input_sequence_length(seq_length)

        # outputs = self._llm_model._forward(
        #     inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches
        # )

        net_input_seq_len = self._llm_model.get_input_sequence_length()
        steps = (seq_length + net_input_seq_len - 1) // net_input_seq_len

        padding_len = steps * net_input_seq_len - seq_length
        padding_embeds = self.embed_tokens(
            torch.zeros(inputs_embeds.shape[0], padding_len, dtype=torch.long, device=inputs_embeds.device)
        )
        inputs_embeds = torch.cat([inputs_embeds, padding_embeds], dim=1)
        if steps > 1:
            for i in tqdm(range(steps)):
                start = i * net_input_seq_len
                end = (i + 1) * net_input_seq_len
                sub_inputs_embeds = inputs_embeds[:, start:end, :]
                sub_past_seq_length = past_seq_length + start
                sub_current_input_length = torch.tensor(
                    [min(end, seq_length) - start], dtype=current_input_length.dtype
                ).to(current_input_length.device)
                self._llm_model.set_input_sequence_length(int(sub_current_input_length.item()))
                outputs = self._llm_model.forward(
                    sub_inputs_embeds, sub_past_seq_length, sub_current_input_length, past_key_caches, past_value_caches
                )
        else:
            outputs = self._llm_model.forward(
                inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches
            )

        # self._past_seq_length += seq_length
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.logits
        # loss = None
        # if labels is not None:
        #     loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return CausalLMOutputWithPast(
            # loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
        )

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model_or_path: Union[Qwen3ForCausalLM, str],
        llm_model: Optional[BaseModel] = None,
    ) -> Self:
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        if isinstance(hf_model_or_path, str):
            hf_model = Qwen3ForCausalLM.from_pretrained(hf_model_or_path, torch_dtype=torch.float16, device_map="auto")
        elif isinstance(hf_model_or_path, Qwen3ForCausalLM):
            hf_model = hf_model_or_path

        if llm_model is not None:
            assert isinstance(llm_model, BaseModel)
            hf_model.__class__ = cls
            hf_model.__setup__(llm_model)
            # hf_model.embed_tokens = hf_model.model.embed_tokens
            del hf_model.model
            del hf_model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return hf_model

    def _sample_forward(
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
        # TODO: 需要根据attention mask计算seq_length
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[-2]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")
        if self._prefill:
            if self.dynamic_input:
                self._llm_model.set_input_sequence_length(seq_length)
        else:
            self._llm_model.set_input_sequence_length(1)

        out = self._xh_orig_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )

        self._past_seq_length += seq_length
        if self.prefill:
            self.prefill = False
        return out

    def generate(self, *args, **kwargs):
        self.prefill = True
        self._past_seq_length = 0
        self._xh_orig_forward = self.forward
        self.forward = self._sample_forward
        self.prefill_input_sequence_length = self._llm_model.get_input_sequence_length()
        out = super().generate(*args, **kwargs)
        self._llm_model.set_input_sequence_length(self.prefill_input_sequence_length)
        self.forward = self._xh_orig_forward
        del self._xh_orig_forward
        return out


class _Qwen3Legacy_HFCompatible_(LLM_HFCompatible):
    def _setup(self, llm_model: BaseModel):
        m = super()._setup(llm_model)
        if llm_model is not None:
            assert isinstance(llm_model, BaseModel)
            # hf_model.embed_tokens = hf_model.model.embed_tokens
            del m.model
            del m.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return m


def build_qwen3_hf_compatible(hf_model_or_path: Union[Qwen3ForCausalLM, str], llm_model: BaseModel) -> LLM_HFCompatible:
    if Qwen3ForCausalLM not in LLM_DYNAMIC_MODULES:
        LLM_DYNAMIC_MODULES.register_module(
            {
                Qwen3ForCausalLM: "Qwen3ForCausalLM",
            },
            _Qwen3Legacy_HFCompatible_,
        )
    if isinstance(hf_model_or_path, str):
        hf_model = Qwen3ForCausalLM.from_pretrained(hf_model_or_path, torch_dtype=torch.float16, device_map="auto")
    elif isinstance(hf_model_or_path, Qwen3ForCausalLM):
        hf_model = hf_model_or_path
    return LLM_DYNAMIC_MODULES.convert(hf_model, llm_model=llm_model)


class Qwen3Demo(Demo):
    pass
