
# Copyright 2025 HOUMO AI
#
# File: minicpmo_hf_compatible.py
# Description:
#   MiniCPM-O HuggingFace compatibility wrapper for xh2modelzoo inference.
#
# Portions of this file adapt or wrap upstream OpenBMB MiniCPM / MiniCPM-V
# behavior for local HuggingFace compatibility, XH2A integration, and export
# workflows.
# Original project: https://github.com/OpenBMB/MiniCPM-V
# Original license: Apache License, Version 2.0
# Modifications by HOUMO AI: local compatibility, wrapping, and export support.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import math
from typing import Optional, Tuple, Union, List
from tqdm import tqdm
import torch
import torch.nn.functional as F
from transformers import AutoModel, DynamicCache
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPooling, CausalLMOutputWithPast
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
from ..base_llm_model import BaseModel

def create_llm_wraped_cls(cls):
    class _Qwen2ForCausalLM(cls):
        def _setup(self, *args, **kwargs):
            pass

        @property
        def prefill(self) -> bool:
            return self._prefill

        @prefill.setter
        def prefill(self, value: bool) -> None:
            self._prefill = value

        def get_output_embeddings(self):
            """
            lm_eval需要这个接口
            """
            return None

        def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            **kwargs,
        ):
            if past_key_values is not None:
                if isinstance(past_key_values, Cache):
                    cache_length = past_key_values.get_seq_length()
                    past_length = past_key_values.seen_tokens
                else:
                    cache_length = past_length = past_key_values[0][0].shape[2]

                # Keep only the unprocessed tokens:
                # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
                # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as
                # input)
                if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                    input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
                # input_ids based on the past_length.
                elif past_length < input_ids.shape[1]:
                    input_ids = input_ids[:, past_length:]
                
                input_ids = input_ids[:, -1:]
                # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # if ∈putsembeds∈putsembedsinputs_embeds are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}

                if attention_mask is not None and position_ids is None:
                    # create position_ids on the fly for batch generation
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
            else:
                model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}
                if attention_mask is not None and position_ids is None:
                    # create position_ids on the fly for batch generation
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    position_ids = position_ids[:, -1:]

            if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
                if model_inputs["inputs_embeds"] is not None:
                    batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                    device = model_inputs["inputs_embeds"].device
                else:
                    batch_size, sequence_length = model_inputs["input_ids"].shape
                    device = model_inputs["input_ids"].device

                dtype = self.lm_head.weight.dtype
                min_dtype = torch.finfo(dtype).min

                attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_length(),
                    dtype=dtype,
                    device=device,
                    min_dtype=min_dtype,
                    cache_position=cache_position,
                    batch_size=batch_size,
                )

            model_inputs.update(
                {
                    "position_ids": position_ids,
                    # "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "attention_mask": attention_mask,
                }
            )
            return model_inputs

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
            self._llm_model.set_num_logits_to_keep(1)
            self._xh_orig_forward = self.forward
            self.forward = self._sample_forward
            self.prefill_input_sequence_length = self._llm_model.get_input_sequence_length()
            out = super().generate(*args, **kwargs)
            self._llm_model.set_input_sequence_length(self.prefill_input_sequence_length)
            self.forward = self._xh_orig_forward
            del self._xh_orig_forward
            return out

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
            seq_length = inputs_embeds.shape[-2]
            current_input_length = torch.tensor([seq_length], dtype=torch.int32).to(inputs_embeds.device)

            past_key_caches = self._llm_model.past_value_caches
            past_value_caches = self._llm_model.past_key_caches

            if past_key_values is None or (hasattr(self, "use_cache") and not self.use_cache):
                past_key_caches = [torch.tensor([])] * len(past_key_caches)
                past_value_caches = [torch.tensor([])] * len(past_value_caches)

            if past_key_values is None or (hasattr(self, "use_cache") and not self.use_cache):
                past_key_caches = [torch.tensor([])] * len(past_key_caches)
                past_value_caches = [torch.tensor([])] * len(past_value_caches)

            outputs = self._llm_model._forward(
                inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches
            )

            logits = outputs.logits[0]
            hidden_states = [outputs.logits[1]]

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return CausalLMOutputWithPast(
                logits=logits,
                past_key_values=past_key_values,
                hidden_states=hidden_states,
            )

    return _Qwen2ForCausalLM


def create_vision_wraped_cls(cls):
    class _SiglipVisionTransformer(cls):
        def forward(
            self,
            data,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
            return self._vision_model.test_step(data)

    return _SiglipVisionTransformer

def create_audio_wraped_cls(cls):
    class _MiniCPMWhisperEncoder(cls):
        def forward(
            self,
            input_features,
            attention_mask,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
            return self._audio_model(input_features, attention_mask)

    return _MiniCPMWhisperEncoder


def make_streaming_chunk_mask_generation(
    inputs_embeds: torch.Tensor,
    past_seen_tokens: int,
    streaming_tts_text_mask: torch.Tensor,
    streaming_reserved_length: int = 300,
    streaming_audio_chunk_size: int = 50,
    streaming_text_chunk_size: int = 10,
    num_spk_emb: int = 1,
    use_spk_emb: bool = True,
) -> torch.Tensor:
    """
    In streaming audio generation, determine which `text` positions the TTS model can attend to when generating each chunk of `audio` tokens.

    This function creates a mask that allows the model to attend to a specific chunk of text
    tokens when generating each chunk of audio tokens, enabling streaming TTS generation.

    Args:
        inputs_embeds (torch.Tensor): Input embeddings tensor.
        past_seen_tokens (int): Number of tokens already seen by the model.
        streaming_tts_text_mask (torch.Tensor): Mask for the text tokens.
        streaming_reserved_length (int, optional): Number of reserved tokens for streaming. Defaults to 300.
        streaming_chunk_length (int, optional): Length of each streaming chunk. Defaults to 50.
        streaming_text_chunk_size (int, optional): Size of each text chunk. Defaults to 7.

    Returns:
        torch.Tensor: Causal mask for streaming TTS generation, shape is [batch_size=1, 1, seq_len=1, past_seen_tokens+1]

    Raises:
        AssertionError: If the batch size is not 1 (only supports batch size of 1 for inference).
    """
    assert inputs_embeds.shape[0] == 1

    dtype = inputs_embeds.dtype
    device = inputs_embeds.device
    min_dtype = torch.finfo(dtype).min

    # Add `1` to the past seen tokens to account for new `tokens` during `generate`
    causal_mask = torch.full((1, past_seen_tokens + inputs_embeds.shape[1]), fill_value=0, dtype=dtype, device=device)

    # Calculate the start of invisible text tokens
    invisible_text_tokens_start = (
        min(
            math.ceil((past_seen_tokens - streaming_reserved_length) / streaming_audio_chunk_size)
            * streaming_text_chunk_size,
            streaming_reserved_length,
        )
        + 1
        + num_spk_emb * use_spk_emb
    )  # Add 1 for [Stts] and N for [spk_emb] tokens if `use_spk_emb` is True

    invisible_text_tokens_end = (
        streaming_reserved_length + 1 + num_spk_emb * use_spk_emb + 1
    )  # Add 1 for [Ptts] (aka `audio_bos_token_id`)

    # Set invisible text tokens to min_dtype (effectively -inf)
    causal_mask[0, invisible_text_tokens_start:invisible_text_tokens_end] = min_dtype

    # Mask padding positions in the text mask
    causal_mask[0, 0 : 1 + num_spk_emb * use_spk_emb + streaming_reserved_length + 1].masked_fill_(
        streaming_tts_text_mask == 0, min_dtype
    )

    # Add extra dimensions for batch and heads
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)

    return causal_mask

@dataclass
class ConditionalChatTTSGenerationOutput(ModelOutput):
    """
    Output class for ConditionalChatTTS generation.

    Args:
        new_ids (torch.LongTensor): Newly generated audio code sequence, shape (batch_size, sequence_length, num_vq).
        audio_input_ids (torch.LongTensor): Updated input IDs including condition and generated audio codes, shape (batch_size, full_sequence_length, num_vq).
        past_key_values (Tuple[Tuple[torch.FloatTensor]]): Tuple containing pre-computed keys and values used for attention mechanism. Each element has shape (batch_size, num_heads, sequence_length, embed_size_per_head).
        finished (bool): Boolean indicating whether generation is complete.

    """

    new_ids: torch.LongTensor = None
    audio_input_ids: torch.LongTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    finished: bool = None

def create_tts_model_wraped_cls(cls):
    class _ChatTTSModel(cls):
        @torch.inference_mode()
        def prefill_text(
            self,
            input_ids: torch.Tensor,
            position_ids: torch.LongTensor,
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
            lm_spk_emb_last_hidden_states: Optional[torch.Tensor] = None,
        ):
            """Prefill a chunk of new text tokens in streaming setting.
            Specifically speaking, update `past_key_values` using new text tokens, then the model will read the new text tokens.

            Args:
                input_ids (Tensor): Tensor of shape [batch_size, seq_len]
                position_ids (LongTensor): Tensor of shape [batch_size, seq_len]
                past_key_values (List[Tuple[Tensor]]): KV Cache of all layers, each layer is a tuple (Tensor, Tensor) denoting keys and values. Each tensor is of seq_len = `self.streaming_text_reserved_len`. `past_key_values` will be updated.
                lm_spk_emb_last_hidden_states (Tensor, optional): Tensor of shape [batch_size, num_spk_emb, llm_dim]. Defaults to None.
                lm_last_hidden_states (Tensor, optional): _description_. Defaults to None.

            Note that all `batch_size` should be `1`.
            """
            assert input_ids.shape[0] == 1
            assert past_key_values is not None

            # Merge text and LLM embeddings
            inputs_embeds = self.merge_inputs_embeds(
                input_ids=input_ids,
                lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
            )

            past_key_caches = self._tts_llama_model.past_key_caches
            past_value_caches = self._tts_llama_model.past_value_caches

            past_seq_length = position_ids[0][0].unsqueeze(0).to(torch.int32)
            current_input_length = torch.tensor([position_ids.shape[-1]], dtype=torch.int32).to(inputs_embeds.device)

            attention_mask = torch.ones((self._tts_llama_model.max_sequence_length, ), dtype=torch.float16).to(inputs_embeds.device) * torch.finfo(torch.float16).min
            attention_mask[:position_ids[0][-1] + 1] = 0
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
            # Model forward
            self._tts_llama_model.set_input_sequence_length(current_input_length.cpu().item())
            outputs_prefill = self._tts_llama_model._forward(
                inputs_embeds=inputs_embeds,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_key_caches=past_key_caches,
                past_value_caches=past_value_caches,
                attention_mask=attention_mask

            )

            # Get model updated KV Cache
            # past_key_values_for_prefill_updated = outputs_prefill.past_key_values

            # # Update generated KV Cache to input `past_key_values`
            for layer_idx in range(len(past_key_values)):
                # Update keys
                past_key_values[layer_idx][0][:, :, position_ids[:, 0] : position_ids[:, -1] + 1, :] = (
                    past_key_caches[layer_idx][
                        :, :, position_ids[:, 0] : position_ids[:, -1] + 1
                    ].clone()
                )
                # Update values
                past_key_values[layer_idx][1][:, :, position_ids[:, 0] : position_ids[:, -1] + 1, :] = (
                    past_value_caches[layer_idx][
                        :, :, position_ids[:, 0] : position_ids[:, -1] + 1
                    ].clone()
                )

            return past_key_values

        @torch.inference_mode()
        def generate(
            self,
            input_ids: torch.Tensor,
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
            temperature: torch.Tensor,
            eos_token: Union[int, torch.Tensor],
            streaming_tts_text_mask=None,
            force_no_stop=False,
            min_new_token=10,
            max_new_token=50,
            logits_warpers = [],
            logits_processors = [],
            show_tqdm=False,
        ):
            # We only support batch size `1` for now
            assert input_ids.shape[0] == 1
            assert past_key_values is not None

            # fix: this should not be `input_ids.shape[1]`
            # start_idx = input_ids.shape[1]
            start_idx = 1 + self.num_spk_embs * self.use_speaker_embedding + self.streaming_text_reserved_len + 1

            finish = torch.zeros(input_ids.shape[0], device=input_ids.device).bool()

            temperature = temperature.unsqueeze(0).expand(input_ids.shape[0], -1).contiguous().view(-1, 1)

            progress = input_ids.shape[1]

            input_ids_buf = torch.zeros(
                input_ids.shape[0],  # batch_size
                progress + max_new_token,  # max_possible_seq_len = input_ids.shape[1] + max_new_token
                input_ids.shape[2],  # self.num_vqs
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

            # Copy existing `input_ids` to `input_ids_buf`
            input_ids_buf.narrow(1, 0, progress).copy_(input_ids)

            del input_ids
            input_ids = input_ids_buf.narrow(1, 0, progress)

            pbar: Optional[tqdm] = None
            if show_tqdm:
                pbar = tqdm(
                    total=max_new_token,
                    desc="code",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
                )

            condition_length = 1 + self.num_spk_embs * self.use_speaker_embedding + self.streaming_text_reserved_len + 1

            for i in range(max_new_token):
                # Prepare generation inputs
                audio_bos = False

                # If this is the first audio token, the case is SPECIAL
                if progress == condition_length:
                    audio_bos = True

                # assert progress == (
                #     past_key_values[0][0].shape[2] + 1
                # )  # If you are using according to the guidelines, this should be passed.

                if audio_bos:
                    # Generate the first token, activate the model with `self.audio_bos_token_id`, the model will predict a new audio token. This is a special case because without the `audio bos token`, it is impossible to generate the first audio token in our streaming setting.
                    narrowed_input_ids = torch.tensor([[self.audio_bos_token_id]], dtype=torch.long, device=self.device)
                    inputs_embeds = self.emb_text(narrowed_input_ids)
                    del narrowed_input_ids
                else:
                    # Generate the following audio tokens, it is applicable to all other cases, including second and the following calling of `generate`.
                    narrowed_input_ids = input_ids.narrow(dim=1, start=input_ids.shape[1] - 1, length=1)
                    code_emb = [self.emb_code[i](narrowed_input_ids[:, :, i]) for i in range(self.num_vq)]
                    inputs_embeds = torch.stack(code_emb, 3).sum(3)

                position_ids = torch.tensor(
                    [past_key_values[0][0].shape[2]], dtype=torch.long, device=self.device
                ).unsqueeze(0)

                cache_position = position_ids.clone()

                # Make causal mask
                causal_mask = make_streaming_chunk_mask_generation(
                    inputs_embeds=inputs_embeds,
                    past_seen_tokens=past_key_values[0][0].shape[2],
                    streaming_tts_text_mask=streaming_tts_text_mask,
                    streaming_reserved_length=self.streaming_text_reserved_len,
                    streaming_text_chunk_size=self.streaming_text_chunk_size,
                )

                past_seq_length = position_ids[0][0].unsqueeze(0).to(torch.int32)
                current_input_length = torch.tensor([position_ids.shape[-1]], dtype=torch.int32).to(inputs_embeds.device)

                past_key_caches = self._tts_llama_model.past_key_caches
                past_value_caches = self._tts_llama_model.past_value_caches

                self._tts_llama_model.set_input_sequence_length(current_input_length.cpu().item())

                attention_mask = torch.ones((self._tts_llama_model.max_sequence_length, ), dtype=torch.float16).to(inputs_embeds.device) * torch.finfo(torch.float16).min

                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)  
                attention_mask[:, :, :, :causal_mask.shape[-1]] = causal_mask

                # Model forward
                logits = self._tts_llama_model._forward(
                    inputs_embeds=inputs_embeds,
                    past_seq_length=past_seq_length,
                    current_input_length=current_input_length,
                    past_key_caches=past_key_caches,
                    past_value_caches=past_value_caches,
                    attention_mask=attention_mask
                )

                for layer_idx in range(len(past_key_values)):
                    past_key_values[layer_idx] = (torch.cat((past_key_values[layer_idx][0], 
                        past_key_caches[layer_idx][
                            :, :, position_ids[:, 0] : position_ids[:, -1] + 1
                        ].clone()), dim=2),  torch.cat((past_key_values[layer_idx][1], 
                        past_value_caches[layer_idx][
                            :, :, position_ids[:, 0] : position_ids[:, -1] + 1
                        ].clone()), dim=2))

                del position_ids
                del inputs_embeds
                del cache_position
                del causal_mask

                # logits = logits[:, -1].float()
                logits = logits.narrow(1, -1, 1).squeeze_(1).float()

                # logits = rearrange(logits, "b c n -> (b n) c")
                logits = logits.permute(0, 2, 1)
                logits = logits.reshape(-1, logits.size(2))
                # logits_token = rearrange(input_ids[:, start_idx:], "b c n -> (b n) c")
                input_ids_sliced = input_ids.narrow(
                    1,
                    start_idx,
                    input_ids.size(1) - start_idx,
                ).permute(0, 2, 1)
                logits_token = input_ids_sliced.reshape(
                    input_ids_sliced.size(0) * input_ids_sliced.size(1),
                    -1,
                ).to(self.device)
                del input_ids_sliced

                logits /= temperature

                if not audio_bos:
                    for logitsProcessors in logits_processors:
                        logits = logitsProcessors(logits_token, logits)
                if not audio_bos:
                    for logitsWarpers in logits_warpers:
                        logits = logitsWarpers(logits_token, logits)

                del logits_token

                if i < min_new_token:
                    logits[:, eos_token] = -torch.inf

                if force_no_stop:
                    logits[:, eos_token] = -torch.inf

                scores = F.softmax(logits, dim=-1)

                del logits
                idx_next = torch.multinomial(scores, num_samples=1)  # .to(finish.device)

                del scores

                # idx_next = rearrange(idx_next, "(b n) 1 -> b n", n=self.num_vq)
                idx_next = idx_next.view(-1, self.num_vq)
                finish_or = idx_next.eq(eos_token).any(1)
                finish.logical_or_(finish_or)

                del finish_or
                # Store new `token` into `input_ids_buf`
                input_ids_buf.narrow(1, progress, 1).copy_(idx_next.unsqueeze_(1))

                if i == 0 and finish.any():
                    # raise Exception
                    break

                del idx_next
                progress += 1
                input_ids = input_ids_buf.narrow(1, 0, progress)

                if finish.all():
                    break

                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            del input_ids_buf

            if finish.all():
                # the last may contains eos token
                genrated_input_ids = input_ids[:, condition_length:-1, :]
            else:
                # there is no eos token
                genrated_input_ids = input_ids[:, condition_length:, :]

            return ConditionalChatTTSGenerationOutput(
                new_ids=genrated_input_ids,
                audio_input_ids=input_ids,  # for update purpose
                past_key_values=past_key_values,  # for update purpose
                finished=finish.all(),
            )           
            
    return _ChatTTSModel


def create_minicpo_wraped_cls(cls):
    class _MiniCPMO(cls):
        def get_audio_embedding_streaming(self, data):
            raise NotImplementedError(
                "get_audio_embedding_streaming is not implemented in XHMiniCPMOAudioModel, please implement it in your own model."
            )

        def _get_audio_embedding(self, data, chunk_length=-1, dummy=True):
            r"""
            Extract full audio embeddings with optional chunk-based attention.

            This method computes embeddings for all audio frames at once, either using full attention (when
            `chunk_length` is -1) or chunk-based attention (when `chunk_length` is a positive number). It does
            not use key-value caching and is suitable for non-streaming inference.

            Args:
                data (dict):
                    - **"audio_features"** (`torch.FloatTensor`): Input mel-spectrograms of shape `(batch_size, 80, frames)`.
                    - **"audio_feature_lens"** (List[List[int]]): Lengths of each audio segment for each item in the batch.
                chunk_length (int, optional): Determines whether to use full attention (-1) or chunk-based
                    attention (>0) during embedding computation.

            Returns:
                List[List[torch.Tensor]]: audio embeddings
            """

            wavforms = data.get("audio_features", [])  # (bs, 80, frames) or [], multi audios need filled in advance
            audio_feature_lens_raw = data.get("audio_feature_lens", [])  # list, [[x1, x2], [y1], [z1]]

            # exist audio
            if len(wavforms) > 0:
                audio_feature_lens = torch.hstack(audio_feature_lens_raw)
                batch_size, _, max_mel_seq_len = wavforms.shape
                max_seq_len = (max_mel_seq_len - 1) // 2 + 1

                # Create a sequence tensor of shape (batch_size, max_seq_len)
                seq_range = (
                    torch.arange(0, max_seq_len, dtype=audio_feature_lens.dtype, device=audio_feature_lens.device)
                    .unsqueeze(0)
                    .expand(batch_size, max_seq_len)
                )
                lengths_expand = audio_feature_lens.unsqueeze(1).expand(batch_size, max_seq_len)
                # Create mask
                padding_mask = seq_range >= lengths_expand  # 1 for padded values

                audio_attention_mask_ = padding_mask.view(batch_size, 1, 1, max_seq_len).expand(
                    batch_size, 1, max_seq_len, max_seq_len
                )
                audio_attention_mask = audio_attention_mask_.to(
                    dtype=self.apm.conv1.weight.dtype, device=self.apm.conv1.weight.device
                )

                if chunk_length > 0:
                    chunk_num_frame = int(chunk_length * 50)
                    chunk_mask = self.subsequent_chunk_mask(
                        size=max_seq_len,
                        chunk_size=chunk_num_frame,
                        num_left_chunks=-1,
                        device=audio_attention_mask_.device,
                    )
                    audio_attention_mask_ = torch.logical_or(audio_attention_mask_, torch.logical_not(chunk_mask))

                audio_attention_mask[audio_attention_mask_] = float("-inf")
                # audio_states = self.apm(
                #     wavforms, output_hidden_states=True, attention_mask=audio_attention_mask
                # ).hidden_states[self.audio_encoder_layer]
                # audio_embeds = self.audio_projection_layer(audio_states)

                # audio_embeds = audio_embeds.transpose(1, 2)
                # audio_embeds = self.audio_avg_pooler(audio_embeds)
                # audio_embeds = audio_embeds.transpose(1, 2)
                dtype = self.apm.conv1.weight.dtype
                audio_embeds = self.apm(wavforms.to(dtype), audio_attention_mask)
                _, feature_lens_after_pooling = self._get_feat_extract_output_lengths(audio_feature_lens)

                num_audio_tokens = feature_lens_after_pooling

                final_audio_embeds = []
                idx = 0
                for i in range(len(audio_feature_lens_raw)):
                    target_audio_embeds = []
                    for _ in range(len(audio_feature_lens_raw[i])):
                        target_audio_embeds.append(audio_embeds[idx, : num_audio_tokens[idx], :])
                        idx += 1
                    final_audio_embeds.append(target_audio_embeds)
                return final_audio_embeds
            elif self.training and dummy:
                dtype = self.apm.embed_positions.weight.dtype
                device = self.apm.embed_positions.weight.device

                dummy_wavs = torch.zeros((1, 80, 100), device=device, dtype=dtype)
                audio_states = self.apm(dummy_wavs, output_hidden_states=True).hidden_states[self.audio_encoder_layer]

                audio_embeds = self.audio_projection_layer(audio_states)

                audio_embeds = audio_embeds.transpose(1, 2)
                audio_embeds = self.audio_avg_pooler(audio_embeds)
                audio_embeds = audio_embeds.transpose(1, 2)
                return [audio_embeds]

            else:
                return []

        def _get_vllm_embedding(self, data):
            # assert False
            if "vision_hidden_states" not in data:
                dtype = self.llm.model.embed_tokens.weight.dtype
                device = self.llm.model.embed_tokens.weight.device
                tgt_sizes = data["tgt_sizes"]
                pixel_values_list = data["pixel_values"]
                vision_hidden_states = []
                all_pixel_values = []
                img_cnt = []
                for pixel_values in pixel_values_list:
                    img_cnt.append(len(pixel_values))
                    all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

                # exist image
                if all_pixel_values:
                    # (
                    #     all_pixel_values,
                    #     attention_mask,
                    #     position_ids,
                    #     resampler_pos_embed,
                    #     resampler_key_padding_mask,
                    #     _tgt_sizes,
                    #     _imgs_cnt,
                    # ) = self.prepare_inputs(data)
                    # all_pixel_values = all_pixel_values.type(dtype)

                    # tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                    # tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                    # max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                    # all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                    #     all_pixel_values, batch_first=True, padding_value=0.0
                    # )
                    # B, L, _ = all_pixel_values.shape
                    # all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                    # patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                    # for i in range(B):
                    #     patch_attn_mask[i, 0, : tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                    # vision_batch_size = 1
                    # all_pixel_values = all_pixel_values.type(dtype)
                    # if B > vision_batch_size:
                    #     hs = []
                    #     for i in range(0, B, vision_batch_size):
                    #         start_idx = i
                    #         end_idx = i + vision_batch_size
                    #         tmp_hs = self.vpm(
                    #             all_pixel_values[start_idx:end_idx],
                    #             patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                    #             tgt_sizes=tgt_sizes[start_idx:end_idx],
                    #         ).last_hidden_state
                    #         hs.append(tmp_hs)
                    #     vision_embedding = torch.cat(hs, dim=0)
                    # else:
                    #     vision_embedding = self.vpm(
                    #         all_pixel_values, patch_attention_mask=patch_attn_mask, tgt_sizes=tgt_sizes
                    #     ).last_hidden_state
                    # vision_embedding = self.resampler(vision_embedding, tgt_sizes)
                    # B = all_pixel_values.shape[0]
                    # hs = []
                    # for i in range(0, B):
                    #     start_idx = i
                    #     end_idx = i + vision_batch_size
                    #     tmp_hs = self._forward(
                    #         all_pixel_values[start_idx:end_idx],
                    #         attention_mask[start_idx:end_idx],
                    #         position_ids[start_idx:end_idx],
                    #         resampler_pos_embed[:, start_idx:end_idx],
                    #         resampler_key_padding_mask[start_idx:end_idx],
                    #         # tgt_sizes[start_idx:end_idx],
                    #     )
                    #     # tmp_hs = self.resampler(
                    #     #     tmp_hs, resampler_pos_embed[:, start_idx:end_idx], resampler_key_padding_mask[start_idx:end_idx]
                    #     # )
                    #     hs.append(tmp_hs)
                    # vision_embedding = torch.cat(hs, dim=0)
                    # vision_embedding = self.vpm(data)
                    # assert vision_embedding is not None
                    # start = 0
                    # for pixel_values in pixel_values_list:
                    #     img_cnt = len(pixel_values)
                    #     if img_cnt > 0:
                    #         vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                    #         start += img_cnt
                    #     else:
                    #         vision_hidden_states.append([])
                    vision_hidden_states = self.vpm(data)
                else:  # no image
                    if self.training:
                        raise RuntimeError("Training mode must have image.")
                        dummy_image = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
                        tgt_sizes = torch.Tensor(
                            [[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]
                        ).type(torch.int32)
                        dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes)
                    else:
                        dummy_feature = []
                    for _ in range(len(pixel_values_list)):
                        vision_hidden_states.append(dummy_feature)

            else:
                vision_hidden_states = data["vision_hidden_states"]

            if hasattr(self.llm.config, "scale_emb"):
                vllm_embedding = self.llm.model.embed_tokens(data["input_ids"]) * self.llm.config.scale_emb
            else:
                vllm_embedding = self.llm.model.embed_tokens(data["input_ids"])

            vision_hidden_states = [
                i.type(vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i for i in vision_hidden_states
            ]

            bs = len(data["input_ids"])
            for i in range(bs):
                cur_vs_hs = vision_hidden_states[i]
                if len(cur_vs_hs) > 0:
                    cur_vllm_emb = vllm_embedding[i]
                    cur_image_bound = data["image_bound"][i]
                    if len(cur_image_bound) > 0:
                        image_indices = torch.stack(
                            [torch.arange(r[0], r[1], dtype=torch.long) for r in cur_image_bound]
                        ).to(vllm_embedding.device)

                        cur_vllm_emb.scatter_(
                            0,
                            image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                            cur_vs_hs.view(-1, cur_vs_hs.shape[-1]),
                        )
                    elif self.training:
                        cur_vllm_emb += cur_vs_hs[0].mean() * 0

            return vllm_embedding, vision_hidden_states

    return _MiniCPMO


class MiniCPMO_HFCompatible:
    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model,
        vision_model: Optional[BaseModel] = None,
        llm_model: Optional[BaseModel] = None,
        audio_model: Optional[BaseModel] = None,
        tts_llama_model: Optional[BaseModel] = None,
    ):
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        if llm_model is not None:
            minicpm_llm = hf_model.llm
            _llm_cls = create_llm_wraped_cls(type(minicpm_llm))
            minicpm_llm.__class__ = _llm_cls
            minicpm_llm._llm_model = llm_model
            minicpm_llm.embed_tokens = hf_model.llm.model.embed_tokens
            # In upstream modeling, `self.llm.prepare_inputs_for_generation` is monkey-patched on the instance.
            # An instance attribute shadows the class method, so our wrapper's implementation won't be invoked.
            # Remove the instance-bound attribute to fall back to the class method we define here.
            if "prepare_inputs_for_generation" in minicpm_llm.__dict__:
                delattr(minicpm_llm, "prepare_inputs_for_generation")

        if vision_model is not None or audio_model is not None or llm_model is not None or tts_llama_model is not None:
            minicpmo_cls = create_minicpo_wraped_cls(type(hf_model))
            hf_model.__class__ = minicpmo_cls

        if vision_model is not None:
            assert isinstance(vision_model, BaseModel)
            minicpo_vision = hf_model.vpm
            _vision_cls = create_vision_wraped_cls(type(minicpo_vision))
            minicpo_vision.__class__ = _vision_cls
            minicpo_vision._vision_model = vision_model
            hf_model.wrap_cfg = vision_model.wrap_cfg
            vpm = hf_model.vpm
            hf_model.patch_size = vpm.embeddings.patch_size
            hf_model.num_patches_per_side = vpm.embeddings.num_patches_per_side
            hf_model._old_get_vllm_embedding = hf_model.get_vllm_embedding
            hf_model.get_vllm_embedding = hf_model._get_vllm_embedding

        if audio_model is not None:
            assert isinstance(audio_model, BaseModel)
            minicpo_apm = hf_model.apm
            _audio_cls = create_audio_wraped_cls(type(minicpo_apm))
            minicpo_apm.__class__ = _audio_cls
            minicpo_apm._audio_model = audio_model
            hf_model._old_get_audio_embedding = hf_model.get_audio_embedding
            hf_model.get_audio_embedding = hf_model._get_audio_embedding

        if tts_llama_model is not None:
            assert isinstance(tts_llama_model, BaseModel)
            minicpo_tts_model = hf_model.tts
            _tts_cls = create_tts_model_wraped_cls(type(minicpo_tts_model))
            minicpo_tts_model.__class__ = _tts_cls
            minicpo_tts_model._tts_llama_model = tts_llama_model
        return hf_model
