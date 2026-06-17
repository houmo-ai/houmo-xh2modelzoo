# ================================================================== #
#  File: deepseek_v4_hf_compatible.py                                #
#  Description:                                                       #
#    HF-compatible wrapper for DeepSeek-V4 ONNX inference.           #
#                                                                     #
#    Converts the HMONNX inference engine into a HuggingFace         #
#    GenerationMixin-compatible model, enabling generate() and        #
#    lm_eval integration.                                            #
# ================================================================== #

from typing import Any, Optional, Union

import torch
from accelerate import init_empty_weights
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    GenerationMixin,
)
from transformers.modeling_outputs import CausalLMOutputWithPast


try:
    from transformers.modeling_utils import no_init_weights
except ImportError:
    import contextlib

    @contextlib.contextmanager
    def no_init_weights(_enable=True):
        yield


try:
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import (
        DeepseekV4ForCausalLM,
    )
except ImportError:
    DeepseekV4ForCausalLM = None

from .inference import DeepseekV4Inference


# ================================================================== #
#  Helpers                                                            #
# ================================================================== #


def _get_empty_hf_model(hf_model_dir, device_map="cpu"):
    """Load model structure without weights."""
    config = AutoConfig.from_pretrained(hf_model_dir, trust_remote_code=True)
    with no_init_weights(), init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    return model


# ================================================================== #
#  HF-compatible wrapper                                              #
# ================================================================== #


class DeepseekV4HFCompatible(DeepseekV4ForCausalLM):
    """Monkey-patch wrapper: HF interface -> ONNX inference."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("Use to_hf_compatible() instead of direct construction")

    def setup(self, llm_model: DeepseekV4Inference, embed_tokens):
        self._prefill = False
        self._past_seq_length = 0
        self._embed_tokens = embed_tokens
        self._llm_model = llm_model

    def get_output_embeddings(self):
        return None

    # -------------------------------------------------------------- #
    #  forward: HF interface -> ONNX chunked calls                    #
    # -------------------------------------------------------------- #

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        if inputs_embeds is None:
            if self._embed_tokens is not None and input_ids is not None:
                inputs_embeds = self._embed_tokens(input_ids)
            else:
                raise ValueError("No input_ids or inputs_embeds provided")

        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[1]
        input_sequence_length = self._llm_model.get_input_sequence_length()

        # -- Pad to multiple of input_sequence_length --
        pad_seq_len = ((seq_length + input_sequence_length - 1) // input_sequence_length) * input_sequence_length

        data = dict(
            input_ids=input_ids,
            past_seq_length=self._past_seq_length,
        )
        (
            inputs_embeds,
            position_ids,
            past_sl,
            cur_il,
            iid,
            kv_caches,
        ) = self._llm_model.prepare_inputs(data, pad_seq_len)

        steps = pad_seq_len // input_sequence_length
        all_logits = []

        for i in range(steps):
            start = i * input_sequence_length
            end = (i + 1) * input_sequence_length
            sub_embeds = inputs_embeds[:, start:end, :]
            sub_past_sl = past_sl + start
            sub_cur_il = torch.tensor(
                [min(end, seq_length) - start],
                dtype=cur_il.dtype,
                device=cur_il.device,
            )

            out = self._llm_model(
                sub_embeds,
                position_ids,
                sub_past_sl,
                sub_cur_il,
                iid,
                kv_caches,
            )
            all_logits.append(out)

        logits = torch.cat(all_logits, dim=1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return CausalLMOutputWithPast(logits=logits)

    # -------------------------------------------------------------- #
    #  generate-time forward with state tracking                       #
    # -------------------------------------------------------------- #

    def _sample_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        logits_to_keep=0,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        if past_key_values is None:
            self.prefill = True
            self._past_seq_length = 0

        out = self._original_forward(
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

        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        elif inputs_embeds is not None:
            seq_length = inputs_embeds.shape[-2]
        else:
            raise ValueError("No input_ids or inputs_embeds")

        self._past_seq_length += seq_length
        if self.prefill:
            self.prefill = False
        return out

    # -------------------------------------------------------------- #
    #  generate: swap forward for token-by-token mode                  #
    # -------------------------------------------------------------- #

    def generate(self, *args: Any, **kwargs: Any) -> torch.LongTensor:
        self.prefill = True
        self.past_seq_length = 0

        class _ForwardContext:
            def __init__(self, model):
                self._model = model
                self._model._original_forward = self._model.forward
                self._saved_input_seq_len = self._model._llm_model.get_input_sequence_length()

            def __enter__(self):
                self._model.forward = self._model._sample_forward
                return self

            def __exit__(self, *exc):
                self._model.forward = self._model._original_forward
                self._model._llm_model.set_input_sequence_length(self._saved_input_seq_len)

        with _ForwardContext(self):
            return GenerationMixin.generate(self, *args, **kwargs)

    # -------------------------------------------------------------- #
    #  Properties                                                      #
    # -------------------------------------------------------------- #

    @property
    def prefill(self):
        return self._prefill

    @prefill.setter
    def prefill(self, val: bool):
        self._prefill = val
        if self._llm_model is not None and hasattr(self._llm_model, "set_phase_prefill"):
            self._llm_model.set_phase_prefill(val)

    @property
    def past_seq_length(self):
        return self._past_seq_length

    @past_seq_length.setter
    def past_seq_length(self, val: int):
        self._past_seq_length = val

    # -------------------------------------------------------------- #
    #  Factory                                                         #
    # -------------------------------------------------------------- #

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model_or_path: Union[str, DeepseekV4ForCausalLM],
        llm_model: Optional[DeepseekV4Inference] = None,
    ) -> DeepseekV4ForCausalLM:
        """Create HF-compatible model backed by ONNX inference."""
        if isinstance(hf_model_or_path, str):
            hf_model = _get_empty_hf_model(hf_model_or_path, device_map="auto")
        else:
            hf_model = hf_model_or_path

        if llm_model is not None:
            assert isinstance(llm_model, DeepseekV4Inference)
            hf_model.__class__ = cls
            assert isinstance(hf_model, DeepseekV4HFCompatible)
            hf_model.setup(
                llm_model=llm_model,
                embed_tokens=llm_model.token_embedding,
            )
            if hasattr(hf_model, "model"):
                del hf_model.model
            if hasattr(hf_model, "lm_head"):
                del hf_model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return hf_model
