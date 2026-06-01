from copy import deepcopy
from typing import Optional, Union, cast

import torch
from qwen_tts import Qwen3TTSModel
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSTalkerOutputWithPast,
)
from tqdm import tqdm
from transformers import Cache, DynamicCache

from xhquant.api import QuantGraph

from ..base_llm_model import LLMBaseModel
from ..base_model import BaseModel
from ..builder import LLM_COMPATIBLE_MODULES, MODELS
from ..common.llm_hfcompatible import LLM_HFCompatible
from .qwen3_tts import XHQwen3TTSModel
from .qwen3_tts import XHQwen3TTSTalkerForConditionalGeneration as Qwen3TTSTalkerForConditionalGeneration


@MODELS.register_module()
class XHQwen3TTSTalker(LLMBaseModel):
    def _set_dtype(self, dtype):
        # self.token_embedding = self.token_embedding.to(dtype)
        return super()._set_dtype(dtype)

    def get_input_embeddings(self):
        return self.token_embedding

    def get_text_embeddings(self):
        return self.text_embedding

    def _set_exec_device(self, device):
        super()._set_exec_device(device)
        self.text_embedding = self.text_embedding.to(device)

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()
        from ._talker_model import register_wrap_modules as register_talker_wrap_modules

        register_talker_wrap_modules(hf_model.model.talker)
        super().init_wrap_model(hf_model.model.talker)
        hf_model = self.wrap_model
        if isinstance(hf_model, Qwen3TTSTalkerForConditionalGeneration):
            llm_model = hf_model.model
        else:
            raise ValueError(f"{type(hf_model)} is not supported")

        self._default_pad_token_id = 0
        # self.token_embedding.weight 和 lm_head.weight 是相同对象
        self.token_embedding = deepcopy(llm_model.get_input_embeddings())
        self.text_embedding = deepcopy(llm_model.get_text_embeddings())
        self.generation_config = hf_model.generation_config
        self.config = hf_model.config
        self.num_hidden_layers = llm_model.config.num_hidden_layers

        head_dim = llm_model.layers[0].self_attn.head_dim
        self.pad_token_id = hf_model.config.eos_token_id
        self.head_dim = head_dim
        if self.use_cache:
            num_decoder_layers = self.num_hidden_layers
            only_first_block = self.wrap_cfg.get("only_first_block", False)
            max_layers = -1
            if only_first_block:
                max_layers = 1
            if "max_layers" in self.wrap_cfg:
                max_layers = self.wrap_cfg["max_layers"]

            if max_layers > 0:
                assert max_layers <= num_decoder_layers
                num_decoder_layers = max_layers

            self.prepare_kv_cache(
                num_decoder_layers,
                [1, llm_model.config.num_key_value_heads, self.cache_length, head_dim],
            )

        hf_model = None

    def get_hf_model(self, device_map="cpu", **kwargs) -> XHQwen3TTSModel:
        dtype = kwargs.get("dtype", torch.float16)
        hf_model = XHQwen3TTSModel.from_pretrained(
            self.hf_model_dir,
            device_map=device_map,
            dtype=dtype,
            # attn_implementation="flash_attention_2",
        )
        hf_model = cast(XHQwen3TTSModel, hf_model)
        return hf_model

    def convert_to_quant_graph(self, target_device: str) -> Optional[QuantGraph]:
        # raise NotImplementedError("Qwen3TTSTalker暂不支持量化导出")
        super().convert_to_quant_graph(target_device)
        # assert self._quanted_model is not None
        # # TODO: 临时解决方案，后续需要修改
        # if self.extra_quant_cfg is not None:
        #     if "attn_weights" in self.extra_quant_cfg:
        #         attn_weights_cfg = self.extra_quant_cfg["attn_weights"]
        #         if "act_schema" in attn_weights_cfg:
        #             act_scheme = attn_weights_cfg["act_schema"]
        #         else:
        #             act_scheme = attn_weights_cfg["act_scheme"]

        #         if "act_schema_2" in attn_weights_cfg:
        #             weight_scheme = attn_weights_cfg["act_schema_2"]
        #         else:
        #             weight_scheme = attn_weights_cfg["act_scheme_2"]
        #         for node in self._quanted_model.graph.nodes:
        #             if node.op == "call_module":
        #                 m = self._quanted_model.get_submodule(node.target)
        #                 if isinstance(m, (xhnn.MaskedSoftmax, xhnn.SoftmaxPlus, nn.Softmax)):
        #                     i_node = node.args[0]
        #                     matmul_module = self._quanted_model.get_submodule(i_node.target)
        #                     assert isinstance(matmul_module, xhnn.MatMul), f"{type(matmul_module)}"
        #                     act_bit = act_scheme.get("bits")
        #                     if act_bit is not None:
        #                         matmul_module.i_cfg.qspec.man_bit = act_bit
        #                     w_bit = weight_scheme.get("bits")
        #                     if w_bit is not None:
        #                         matmul_module.i_cfg_2.qspec.man_bit = w_bit

        return self._quanted_model


class Qwen3TTSTalkerForConditionalGenerationHFCompatible(LLM_HFCompatible):
    def _setup(self, talker_xh_model: BaseModel):
        m = super()._setup(talker_xh_model)
        if talker_xh_model is not None:
            assert isinstance(talker_xh_model, BaseModel)
            del m.model
            del m.codec_head
            self.model = talker_xh_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return m

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
    ) -> Qwen3TTSTalkerOutputWithPast:
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

        past_seq_length = torch.tensor([self._past_seq_length], dtype=torch.int32).to(inputs_embeds.device)

        # TODO: 需要根据attention mask计算seq_length
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[1]

        current_input_length = torch.tensor([seq_length], dtype=torch.int32).to(inputs_embeds.device)

        past_key_caches = self._llm_model.past_value_caches
        past_value_caches = self._llm_model.past_key_caches

        if past_key_values is None or (hasattr(self, "use_cache") and not self.use_cache):
            past_key_caches = [torch.tensor([])] * len(past_key_caches)
            past_value_caches = [torch.tensor([])] * len(past_value_caches)

        if self.dynamic_input and isinstance(self._llm_model, BaseModel):
            self._llm_model.set_input_sequence_length(seq_length)

        net_input_seq_len = self._llm_model.get_input_sequence_length()
        steps = (seq_length + net_input_seq_len - 1) // net_input_seq_len

        padding_len = steps * net_input_seq_len - seq_length
        padding_embeds = self.embed_tokens(
            torch.zeros(inputs_embeds.shape[0], padding_len, dtype=torch.long, device=inputs_embeds.device)
        )
        inputs_embeds = torch.cat([inputs_embeds, padding_embeds], dim=1)
        num_logits_to_keep = self._llm_model.get_num_logits_to_keep()
        assert num_logits_to_keep == 1, "Qwen3TTSTalker目前只支持num_logits_to_keep=1的情况"
        logits = None
        past_hidden = None
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
                output = self._llm_model.forward(
                    sub_inputs_embeds, sub_past_seq_length, sub_current_input_length, past_key_caches, past_value_caches
                )
                if isinstance(output, (list, tuple)):
                    logits, past_hidden = output
                else:
                    logits = output.logits
                    past_hidden = output.past_hidden

        else:
            output = self._llm_model.forward(
                inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches
            )
            if isinstance(output, (list, tuple)):
                logits, past_hidden = output
            else:
                logits = output["logits"]
                past_hidden = output["past_hidden"]

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return Qwen3TTSTalkerOutputWithPast(
            logits=logits,
            past_hidden=past_hidden,
            past_key_values=past_key_values,
        )

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
        trailing_text_hidden=None,
        tts_pad_embed=None,
        generation_step=None,
        subtalker_dosample=None,
        subtalker_top_p=None,
        subtalker_top_k=None,
        subtalker_temperature=None,
        **kwargs,
    ):
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


def build_qwen3_tts_talker_hf_compatible(hf_model: Qwen3TTSModel, xh_talker_model: BaseModel) -> LLM_HFCompatible:
    if Qwen3TTSTalkerForConditionalGeneration not in LLM_COMPATIBLE_MODULES:
        LLM_COMPATIBLE_MODULES.register_module(
            {
                Qwen3TTSTalkerForConditionalGeneration: "Qwen3TTSTalkerForConditionalGeneration",
            },
            Qwen3TTSTalkerForConditionalGenerationHFCompatible,
        )

    hf_model.model.talker = LLM_COMPATIBLE_MODULES.convert(hf_model.model.talker, talker_xh_model=xh_talker_model)
    return hf_model
