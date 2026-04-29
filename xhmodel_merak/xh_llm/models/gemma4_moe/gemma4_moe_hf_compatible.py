from __future__ import annotations

from typing import Optional, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast

from ...text_llm_hf_compatible import TextLLMHFCompatible


class Gemma4MoeWithMaskHFCompatible(TextLLMHFCompatible):
    """HF-compatible facade for Gemma4 with explicit local/global masks.

    Task 05 only wires the conversion surface. Task 09/10 extend runtime
    generation paths after the with-mask graph and HMONNX runtime are ready.
    """

    def _setup(self, text_llm_model):
        model = super()._setup(text_llm_model)
        if model is not None:
            for attr in ("model", "lm_head"):
                if hasattr(model, attr):
                    delattr(model, attr)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._gemma4_image_embeds = None
        self._gemma4_mm_token_type_ids = None
        return model

    def set_experts_implementation(self, experts_implementation):  # noqa: D401
        """No-op override for Transformers decode optimization hooks."""
        self.config.experts_implementation = experts_implementation

    def get_correct_experts_implementation(self, experts_implementation):
        return experts_implementation

    def _grouped_mm_can_dispatch(self):
        return False

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
        image_embeds = kwargs.pop("image_embeds", None)
        if image_embeds is None:
            image_embeds = self._gemma4_image_embeds

        mm_token_type_ids = kwargs.pop("mm_token_type_ids", None)
        if mm_token_type_ids is None:
            mm_token_type_ids = self._gemma4_mm_token_type_ids

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            input_embedding = self.get_input_embeddings()
            embedding_device = input_embedding.weight.device
            if input_ids.device != embedding_device:
                input_ids = input_ids.to(embedding_device)
            inputs_embeds = input_embedding(input_ids)

        if mm_token_type_ids is not None and mm_token_type_ids.device != inputs_embeds.device:
            mm_token_type_ids = mm_token_type_ids.to(inputs_embeds.device)

        if image_embeds is not None and image_embeds.dim() == 3 and image_embeds.shape[0] == 1:
            image_embeds = image_embeds[0]

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        del attention_mask, position_ids, labels, output_attentions, output_hidden_states, use_cache, cache_position

        if past_key_values is None:
            pass

        assert inputs_embeds.shape[0] == 1, "Batch size should be 1 in inference mode."
        seq_length = inputs_embeds.shape[1]
        if self.is_support_dynamic_input:
            target_input_sequence_length = seq_length
            if past_key_values is None and hasattr(self, "prefill_input_sequence_length"):
                target_input_sequence_length = max(seq_length, int(self.prefill_input_sequence_length))
            self._llm_model.set_input_sequence_length(target_input_sequence_length)

        data_processor = self._llm_model.get_data_preprocessor()
        chunk_len = data_processor.input_sequence_length

        if seq_length <= chunk_len:
            data_batch = {
                "input_ids": input_ids,
                "inputs_embeds": inputs_embeds,
                "image_embeds": image_embeds,
                "past_seq_length": self._past_seq_length,
                "mm_token_type_ids": mm_token_type_ids,
            }
            (
                inputs_embeds_proc,
                past_seq_length_t,
                current_input_length_t,
                full_attention_mask,
                sliding_attention_mask,
                past_key_caches,
                past_value_caches,
            ) = data_processor(data_batch)

            logits = self._llm_model.forward(
                inputs_embeds_proc,
                past_seq_length_t,
                current_input_length_t,
                sliding_attention_mask,
                full_attention_mask,
                past_key_caches,
                past_value_caches,
            )
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            if logits.dim() == 3:
                num_logits_to_keep = self._llm_model.get_num_logits_to_keep()
                if num_logits_to_keep == 0:
                    logits = logits[:, :seq_length, :]
                else:
                    logits = logits[:, -num_logits_to_keep:, :]
        else:
            device = inputs_embeds.device

            if image_embeds is not None and input_ids is not None:
                image_token_id = data_processor.image_token_id
                n_image_tokens = int((input_ids == image_token_id).sum().item())
                if n_image_tokens > 0:
                    image_mask = (input_ids == image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                    image_embeds = image_embeds.to(device=inputs_embeds.device, dtype=inputs_embeds.dtype)
                    if image_embeds.shape[0] != n_image_tokens:
                        raise ValueError(
                            "Image features and image tokens do not match: "
                            f"tokens={n_image_tokens}, features={image_embeds.shape[0]}"
                        )
                    inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if mm_token_type_ids is None:
                mm_full = torch.zeros(seq_length, dtype=torch.long, device=device)
            else:
                mm_full = mm_token_type_ids.to(device).flatten()[:seq_length]

            steps = (seq_length + chunk_len - 1) // chunk_len
            pad_len = steps * chunk_len - seq_length
            if pad_len > 0:
                padding_embeds = self.get_input_embeddings()(torch.zeros((1, pad_len), dtype=torch.long, device=device))
                inputs_embeds = torch.cat([inputs_embeds, padding_embeds], dim=1)
                mm_full = torch.cat([mm_full, torch.zeros(pad_len, dtype=torch.long, device=device)])

            past_key_caches = data_processor.past_key_caches
            past_value_caches = data_processor.past_value_caches
            running_past_seq = self._past_seq_length
            outputs_logits = []
            for step_index in range(steps):
                start = step_index * chunk_len
                end = (step_index + 1) * chunk_len
                sub_embeds = inputs_embeds[:, start:end, :]
                sub_current_len = min(end, seq_length) - start
                sub_mm = mm_full[start:end]
                full_attention_mask, sliding_attention_mask = data_processor._build_attention_masks(
                    current_input_length=sub_current_len,
                    past_seq_length=running_past_seq,
                    mm_token_type_ids=sub_mm,
                    device=device,
                )
                chunk_logits = self._llm_model.forward(
                    sub_embeds,
                    torch.tensor([running_past_seq], dtype=torch.int32, device=device),
                    torch.tensor([sub_current_len], dtype=torch.int32, device=device),
                    sliding_attention_mask,
                    full_attention_mask,
                    past_key_caches,
                    past_value_caches,
                )
                if isinstance(chunk_logits, (tuple, list)):
                    chunk_logits = chunk_logits[0]
                outputs_logits.append(chunk_logits)
                running_past_seq += sub_current_len

            last_valid = min(chunk_len, seq_length - (steps - 1) * chunk_len)
            logits = outputs_logits[-1][:, :last_valid, :]

        return CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values)

    def generate(self, *args, **kwargs):
        self._gemma4_image_embeds = kwargs.pop("image_embeds", None)
        self._gemma4_mm_token_type_ids = kwargs.pop("mm_token_type_ids", None)
        return super().generate(*args, **kwargs)


def build_gemma4_moe_with_mask_hf_compatible_model(hf_model, text_llm_model) -> Gemma4MoeWithMaskHFCompatible:
    from xhquant.utils.registry import _DMRegistryCls

    llm_compatible_modules = _DMRegistryCls("XHCompatible")
    hf_model_cls = type(hf_model)
    if hf_model_cls not in llm_compatible_modules:
        llm_compatible_modules.register_module({hf_model_cls: hf_model_cls.__name__}, Gemma4MoeWithMaskHFCompatible)
    return llm_compatible_modules.convert(hf_model, text_llm_model=text_llm_model)
