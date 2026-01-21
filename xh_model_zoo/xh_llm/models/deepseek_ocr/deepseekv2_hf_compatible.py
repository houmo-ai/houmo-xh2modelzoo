from typing import Any, Optional, Type, TypeVar, Union
import inspect
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, DynamicCache, GenerationMixin,TextStreamer
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import no_init_weights

# from xh_model_zoo_new.xh_llm.models.qwen3.inference import Qwen3LegacyInference
from xh_model_zoo.xh_llm.models.deepseek_ocr.modeling_deepseekocr import DeepseekOCRForCausalLM
from xh_model_zoo.xh_llm.models.deepseek_ocr.modeling_deepseekv2 import DeepseekV2ForCausalLM

from transformers.cache_utils import (
    Cache,
    DynamicCache,
    EncoderDecoderCache,
    OffloadedCache,
    QuantizedCacheConfig,
    StaticCache,
)

def get_empty_hf_model(hf_model_dir, device_map="cpu", **kwargs):
    """
    仅仅加载模型结构,不初始化权重,不占用显存
    """
    config = AutoConfig.from_pretrained(hf_model_dir)
    with no_init_weights(), init_empty_weights():
        hf_model = DeepseekOCRForCausalLM.from_config(
            config,
            torch_dtype=torch.float16,
            **kwargs,
        )
    return hf_model

class DeepseekV2HFCompatible(DeepseekV2ForCausalLM):
    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")
    
    def __setup__(self, llm_model):
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

    def get_output_embeddings(self):
        """
        lm_eval需要这个接口
        """
        return None

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
        # position_ids: Optional[torch.LongTensor] = None,
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
            self.embed_tokens.weight = torch.nn.Parameter(self.embed_tokens.weight.to(input_ids.device))
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)

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
        seq_length = inputs_embeds.shape[1]
        current_input_length = torch.tensor([seq_length], dtype=torch.int32).to(inputs_embeds.device)

        past_key_caches = self._llm_model.past_value_caches
        past_value_caches = self._llm_model.past_key_caches

        if past_key_values is None or (hasattr(self, "use_cache") and not self.use_cache):
            past_key_caches = [torch.tensor([])] * len(past_key_caches)
            past_value_caches = [torch.tensor([])] * len(past_value_caches)

        # if self.dynamic_input and isinstance(self._llm_model, BaseModel):
        #     self._llm_model.set_input_sequence_length(seq_length)

        # outputs = self._llm_model._forward(
        #     inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches
        # )

        net_input_seq_len = self._llm_model.get_input_sequence_length()
        steps = (seq_length + net_input_seq_len - 1) // net_input_seq_len

        padding_len = steps * net_input_seq_len - seq_length
        padding_embeds = self.embed_tokens(torch.zeros(inputs_embeds.shape[0], padding_len, dtype=torch.long, device=inputs_embeds.device))
        inputs_embeds = torch.cat([inputs_embeds, padding_embeds], dim=1)
        if steps > 1:
            outputs = []
            for i in tqdm(range(steps)):
                start = i * net_input_seq_len
                end = (i + 1) * net_input_seq_len
                sub_inputs_embeds = inputs_embeds[:, start:end, :]
                sub_past_seq_length = past_seq_length + start
                sub_current_input_length = torch.tensor(
                    [min(end, seq_length) - start], dtype=current_input_length.dtype
                ).to(current_input_length.device)
                self._llm_model.set_input_sequence_length(int(sub_current_input_length.item()))
                output=self._llm_model.forward(
                    sub_inputs_embeds, sub_past_seq_length, sub_current_input_length, past_key_caches, past_value_caches
                )
                if isinstance(output, torch.Tensor):
                    logits = output
                else:
                    logits = output.logits
                outputs.append(logits)
            outputs = torch.concat(outputs,dim=1)
        else:
            outputs = self._llm_model.forward(
                inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches
            )
        

        # self._past_seq_length += seq_length
        if isinstance(outputs, torch.Tensor):
            logits = outputs
        else:
            logits = outputs.logits
        logits = logits[:,:seq_length,:]
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
        hf_model_or_path,
        llm_model = None,
    ) -> DeepseekV2ForCausalLM:
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        if isinstance(hf_model_or_path, str):
            hf_model = DeepseekV2ForCausalLM.from_pretrained(
                hf_model_or_path, torch_dtype=torch.float16, device_map="auto"
            )
        elif isinstance(hf_model_or_path, DeepseekV2ForCausalLM):
            hf_model = hf_model_or_path

        if llm_model is not None:
            # assert isinstance(llm_model, BaseModel)
            hf_model.__class__ = cls
            hf_model._prefill = True
            hf_model._llm_model = llm_model
            hf_model.embed_tokens = llm_model.model.get_input_embeddings()
            hf_model._prefill = True
            hf_model._past_seq_length = 0
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

        out = self._xh_orig_forward(
            input_ids=None,
            attention_mask=attention_mask,
            # position_ids=position_ids,
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
        # TODO: 需要根据attention mask计算seq_length
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[-2]
        else:
            raise ValueError("You must specify either input_ids or inputs_embeds")
        self._past_seq_length += seq_length
        if self._prefill:
            self._prefill = False
        return out

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        """
        Prepare the model inputs for generation. In includes operations like computing the 4D attention mask or
        slicing inputs given the existing cache.

        See the forward pass in the model documentation for expected arguments (different models might have different
        requirements for e.g. `past_key_values`). This function should work as is for most LLMs.
        """

        # 1. Handle BC:
        model_inputs = {}
        # - some models don't have `Cache` support (which implies they don't expect `cache_position` in `forward`)
        if self._supports_cache_class:
            model_inputs["cache_position"] = cache_position
        # - `cache_position` was not a mandatory input in `prepare_inputs_for_generation` for those models, and this
        #   function may be called outside of `generate`. Handle most use cases by creating `cache_position` on the fly
        #   (this alternative is not as robust as calling `generate` and letting it create `cache_position`)
        elif cache_position is None:
            past_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0
            cache_position = torch.arange(past_length, input_ids.shape[1], dtype=torch.long, device=input_ids.device)

        # 2. Generic cache-dependent input preparation
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        # Exception 3: with synced GPUs cache_position may go out of bounds, but we only want dummy token in that case
        if past_key_values is not None:
            model_inputs["past_key_values"] = past_key_values
            if inputs_embeds is not None or cache_position[-1] >= input_ids.shape[1]:  # Exception 1 or Exception 3
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        # 3. Prepare base model inputs
        input_ids_key = "decoder_input_ids" if self.config.is_encoder_decoder else "input_ids"
        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if not self.config.is_encoder_decoder:
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs[input_ids_key] = None
                model_inputs["inputs_embeds"] = inputs_embeds
            else:
                # `clone` calls in this function ensure a consistent stride. See #32227
                model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)
                model_inputs["inputs_embeds"] = None
        else:
            model_inputs[input_ids_key] = input_ids.clone(memory_format=torch.contiguous_format)

        # 4. Create missing `position_ids` on the fly
        if (
            attention_mask is not None
            and kwargs.get("position_ids") is None
            and "position_ids" in set(inspect.signature(self.forward).parameters.keys())
        ):
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            kwargs["position_ids"] = position_ids  # placed in kwargs for further processing (see below)

        # 5. Slice model inputs if it's an input that should have the same length as `input_ids`
        for model_input_name in ["position_ids", "token_type_ids"]:
            model_input = kwargs.get(model_input_name)
            if model_input is not None:
                if past_key_values:
                    model_input = model_input[:, -input_ids.shape[1] :]
                    model_input = model_input.clone(memory_format=torch.contiguous_format)
                model_inputs[model_input_name] = model_input

        # 6. Create 4D attention mask is we are using a `StaticCache` (important for performant compiled forward pass)
        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs[input_ids_key].shape
                device = model_inputs[input_ids_key].device

            # Create the causal mask with fixed shape in advance, to reduce recompilations. If the function to create
            # the 4D causal mask exists, it should be present in the base model (XXXModel class).
            base_model = getattr(self, self.base_model_prefix, None)
            if base_model is None:
                causal_mask_creation_function = getattr(
                    self, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            else:
                causal_mask_creation_function = getattr(
                    base_model, "_prepare_4d_causal_attention_mask_with_cache_position", None
                )
            if causal_mask_creation_function is None:
                print(
                    f"{self.__class__.__name__} has no `_prepare_4d_causal_attention_mask_with_cache_position` method "
                    "defined in its base modeling class. Compiled forward passes will be sub-optimal. If you're "
                    "writing code, see Llama for an example implementation. If you're a user, please report this "
                    "issue on GitHub."
                )
            else:
                attention_mask = causal_mask_creation_function(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_cache_shape(),
                    dtype=self.dtype,
                    device=device,
                    cache_position=cache_position,
                    batch_size=batch_size,
                    config=self.config,
                    past_key_values=past_key_values,
                )
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask

        # 7. Forward ALL kwargs that are uninitialized (e.g. `use_cache`).
        for key, value in kwargs.items():
            if key not in model_inputs:
                model_inputs[key] = value

        # 8. Remove unexpected `generate` inputs (TODO @joao: fix trainer and examples)
        model_inputs.pop("labels", None)
        return model_inputs
    

    def generate(self, *args, **kwargs):
        self._prefill = True
        self._past_seq_length = 0
        self._xh_orig_forward = self.forward
        self.forward = self._sample_forward
        out = super().generate(*args, **kwargs)
        self.forward = self._xh_orig_forward
        del self._xh_orig_forward
        return out

    def demo(self, 
             prompt: str = "你好",
             inputs_embeds = None,
             tokenizer  = None,
             max_generation_length: int = 128):
        """
        Demo the model with a given prompt.
        Args:
            prompt: The prompt to generate.
            max_generation_length: The maximum length of the generated text.
        Returns:
            The generated text.
        """

        # Generate with streaming
        input_ids = torch.randint(0, self.config.vocab_size, (inputs_embeds.shape[0],  inputs_embeds.shape[1]))
        generated_ids = self.generate(input_ids, inputs_embeds=inputs_embeds, max_new_tokens=max_generation_length)
        output_ids = generated_ids[0]
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
        return content
