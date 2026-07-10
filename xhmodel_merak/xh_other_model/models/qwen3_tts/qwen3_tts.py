### 对Qwen3-tts的整体流程进行重构，以适配导出流程，改动主要在talker模型的推理流程，重构后推理流程更清晰
from typing import Optional, Union, cast

import torch
import torch.nn as nn
from qwen_tts import Qwen3TTSModel  # noqa: F401, F403
from qwen_tts.core.models.modeling_qwen3_tts import (
    Qwen3TTSForConditionalGeneration,
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration,
    Qwen3TTSTalkerCodePredictorOutputWithPast,
    Qwen3TTSTalkerForConditionalGeneration,
    Qwen3TTSTalkerOutputWithPast,
)
from transformers import Cache, GenerationConfig, LogitsProcessorList, StoppingCriteriaList
from transformers.generation.streamers import BaseStreamer
from transformers.generation.utils import (
    GenerateDecoderOnlyOutput,
    GenerateEncoderDecoderOutput,
    GenerateNonBeamOutput,
)
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, ModelOutput
from transformers.utils import can_return_tuple


class XHQwen3TTSTalkerForConditionalGeneration(Qwen3TTSTalkerForConditionalGeneration):
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )

        inputs_embeds = inputs.get("inputs_embeds", None)
        if inputs_embeds is None:
            input_ids = inputs["input_ids"]
            past_hidden = inputs["past_hidden"]
            subtalker_dosample = inputs["subtalker_dosample"]
            subtalker_top_p = inputs["subtalker_top_p"]
            subtalker_top_k = inputs["subtalker_top_k"]
            subtalker_temperature = inputs["subtalker_temperature"]
            generation_step = inputs["generation_step"]
            trailing_text_hidden = inputs["trailing_text_hidden"]
            tts_pad_embed = inputs["tts_pad_embed"]

            last_id_hidden = self.get_input_embeddings()(input_ids)
            predictor_result = self.code_predictor.generate(
                inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
                max_new_tokens=self.config.num_code_groups - 1,
                do_sample=subtalker_dosample,
                top_p=subtalker_top_p,
                top_k=subtalker_top_k,
                temperature=subtalker_temperature,
                output_hidden_states=True,
                return_dict_in_generate=True,
            )
            codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)
            codec_hiddens = torch.cat(
                [last_id_hidden]
                + [
                    self.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i : i + 1])
                    for i in range(self.config.num_code_groups - 1)
                ],
                dim=1,
            )
            inputs_embeds = codec_hiddens.sum(1, keepdim=True)

            if generation_step < trailing_text_hidden.shape[1]:
                inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step.item()].unsqueeze(1)
            else:
                inputs_embeds = inputs_embeds + tts_pad_embed
            inputs["inputs_embeds"] = inputs_embeds
            inputs["input_ids"] = None
            inputs["codec_ids"] = codec_ids
        return inputs

    @can_return_tuple
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
        past_hidden=None,
        trailing_text_hidden=None,
        tts_pad_embed=None,
        generation_step=None,
        subtalker_dosample=None,
        subtalker_top_p=None,
        subtalker_top_k=None,
        subtalker_temperature=None,
        **kwargs,
    ) -> Qwen3TTSTalkerOutputWithPast:
        """Forward pass for Qwen3TTS Talker model.

        Args:
            labels: Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        # Prefill
        # if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
        #     generation_step = -1
        #     # codec_ids = None
        # elif inputs_embeds is not None and inputs_embeds.shape[1] == 1:
        #     # codec_ids = kwargs["codec_ids"]
        #     pass
        # Generate
        # else:
        #     last_id_hidden = self.get_input_embeddings()(input_ids)
        #     predictor_result = self.code_predictor.generate(
        #         inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1),
        #         max_new_tokens=self.config.num_code_groups - 1,
        #         do_sample=subtalker_dosample,
        #         top_p=subtalker_top_p,
        #         top_k=subtalker_top_k,
        #         temperature=subtalker_temperature,
        #         output_hidden_states=True,
        #         return_dict_in_generate=True,
        #     )
        #     codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)
        #     codec_hiddens = torch.cat(
        #         [last_id_hidden]
        #         + [
        #             self.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i : i + 1])
        #             for i in range(self.config.num_code_groups - 1)
        #         ],
        #         dim=1,
        #     )
        #     inputs_embeds = codec_hiddens.sum(1, keepdim=True)

        #     if generation_step < trailing_text_hidden.shape[1]:
        #         inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step].unsqueeze(1)
        #     else:
        #         inputs_embeds = inputs_embeds + tts_pad_embed
        if attention_mask is not None:
            if (
                cache_position is None
                or (cache_position is not None and cache_position[0] == 0)
                or self.rope_deltas is None
            ):
                if attention_mask.dim() != 2:
                    _attention_mask = attention_mask.any(dim=-1).long().squeeze(1)  # [1, 78]
                else:
                    _attention_mask = attention_mask
                delta0 = (1 - _attention_mask).sum(dim=-1).unsqueeze(1)
                position_ids, rope_deltas = self.get_rope_index(_attention_mask)
                rope_deltas = rope_deltas - delta0
                self.rope_deltas = rope_deltas
            else:
                # batch_size, seq_length = input_ids.shape
                batch_size, seq_length = inputs_embeds.shape[:2]
                delta = cache_position[0] + self.rope_deltas if cache_position is not None else 0
                # position_ids = torch.arange(seq_length, device=input_ids.device)
                position_ids = torch.arange(seq_length, device=inputs_embeds.device)
                position_ids = position_ids.view(1, -1).expand(batch_size, -1)
                position_ids = position_ids.add(delta)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        hidden_states = hidden_states[:, -1:, :]  # 生成任务, 只需要关注最后一个位置的hidden state
        logits = self.codec_head(hidden_states)

        # loss = None
        # if labels is not None:
        #     loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return Qwen3TTSTalkerOutputWithPast(
            # loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            # hidden_states=([outputs.last_hidden_state], codec_ids),
            # attentions=outputs.attentions,
            past_hidden=hidden_states,
            # generation_step=generation_step + 1,
            # trailing_text_hidden=trailing_text_hidden,
            # tts_pad_embed=tts_pad_embed,
        )

    def _sample(  # noqa: C901, N802
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: BaseStreamer | None = None,
        **model_kwargs,
    ) -> GenerateNonBeamOutput | torch.LongTensor:
        """Generate sequences using multinomial sampling.

        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        # compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        # if compile_forward:
        #     os.environ["TOKENIZERS_PARALLELISM"] = "0"
        #     # If we use FA2 and a static cache, we cannot compile with fullgraph
        #     if self.config._attn_implementation == "flash_attention_2":
        #         # only raise warning if the user passed an explicit compile-config
        #         if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
        #             logger.warning_once(
        #                 "When using Flash Attention 2 and a static cache, you cannot use the option "
        #                 "`CompileConfig(fullgraph=True)` as FA2 introduces graph breaks. "
        #                 "We overrode the option with `fullgraph=False`."
        #             )
        #             generation_config.compile_config.fullgraph = False
        #     model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True

        trailing_text_hidden = model_kwargs["trailing_text_hidden"]
        tts_pad_embed = model_kwargs["tts_pad_embed"]
        generation_step = torch.tensor([-1], device=input_ids.device)
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if streamer is not None and hasattr(streamer, "put_codec_ids"):
                codec_ids = model_inputs.get("codec_ids", None)
                if codec_ids is not None:
                    streamer.put_codec_ids(codec_ids)

            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)

            generation_step += 1
            # assert outputs.generation_step == generation_step, (
            #     f"Expected generation_step to be {generation_step}, but got {outputs.generation_step}"
            # )
            outputs.generation_step = generation_step  # type: ignore
            outputs.tts_pad_embed = tts_pad_embed  # type: ignore
            outputs.trailing_text_hidden = trailing_text_hidden  # type: ignore

            outputs.hidden_states = (None, model_inputs.get("codec_ids", None))  # type: ignore
            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


class XHQwen3TTSForConditionalGeneration(Qwen3TTSForConditionalGeneration):
    @torch.no_grad()
    def generate(
        self,
        input_ids: Optional[list[torch.Tensor]] = None,
        instruct_ids: Optional[list[torch.Tensor]] = None,
        ref_ids: Optional[list[torch.Tensor]] = None,
        voice_clone_prompt: list[dict] = None,
        languages: list[str] = None,
        speakers: list[str] = None,
        non_streaming_mode=False,
        max_new_tokens: int = 4096,
        do_sample: bool = True,
        top_k: int = 50,
        top_p: float = 1.0,
        temperature: float = 0.9,
        subtalker_dosample: bool = True,
        subtalker_top_k: int = 50,
        subtalker_top_p: float = 1.0,
        subtalker_temperature: float = 0.9,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.05,
        **kwargs,
    ):
        talker_kwargs = {
            "max_new_tokens": max_new_tokens,
            "min_new_tokens": 2,
            "do_sample": do_sample,
            "top_k": top_k,
            "top_p": top_p,
            "temperature": temperature,
            "subtalker_dosample": subtalker_dosample,
            "subtalker_top_k": subtalker_top_k,
            "subtalker_top_p": subtalker_top_p,
            "subtalker_temperature": subtalker_temperature,
            "eos_token_id": eos_token_id if eos_token_id is not None else self.config.talker_config.codec_eos_token_id,
            "repetition_penalty": repetition_penalty,
            "suppress_tokens": [
                i
                for i in range(self.config.talker_config.vocab_size - 1024, self.config.talker_config.vocab_size)
                if i not in (self.config.talker_config.codec_eos_token_id,)
            ],
            "output_hidden_states": getattr(kwargs, "output_hidden_states", True),
            "return_dict_in_generate": getattr(kwargs, "return_dict_in_generate", True),
            "streamer": kwargs.get("streamer", None),
        }

        talker_input_embeds = [[] for _ in range(len(input_ids))]

        voice_clone_spk_embeds = None
        # voice clone speaker prompt generate
        if voice_clone_prompt is not None:
            voice_clone_spk_embeds = self.generate_speaker_prompt(voice_clone_prompt)

        # instruct text prompt generate
        if instruct_ids is not None:
            for index, instruct_id in enumerate(instruct_ids):
                if instruct_id is not None:
                    talker_input_embeds[index].append(
                        self.talker.text_projection(self.talker.get_text_embeddings()(instruct_id))
                    )

        # tts text prompt generate
        trailing_text_hiddens = []
        if speakers is None:
            speakers = [None] * len(input_ids)
        for index, (input_id, language, speaker) in enumerate(zip(input_ids, languages, speakers, strict=True)):
            if voice_clone_spk_embeds is None:
                if speaker == "" or speaker is None:  # Instruct create speaker
                    speaker_embed = None
                else:
                    if speaker.lower() not in self.config.talker_config.spk_id:
                        raise NotImplementedError(f"Speaker {speaker} not implemented")
                    else:
                        spk_id = self.config.talker_config.spk_id[speaker.lower()]
                        speaker_embed = self.talker.get_input_embeddings()(
                            torch.tensor(
                                spk_id,
                                device=self.talker.device,
                                dtype=input_id.dtype,
                            )
                        )
            else:
                if voice_clone_prompt["x_vector_only_mode"][index] or voice_clone_prompt["icl_mode"][index]:
                    speaker_embed = voice_clone_spk_embeds[index]
                else:
                    speaker_embed = None

            assert language is not None

            if language.lower() == "auto":
                language_id = None
            else:
                if language.lower() not in self.config.talker_config.codec_language_id:
                    raise NotImplementedError(f"Language {language} not implemented")
                else:
                    language_id = self.config.talker_config.codec_language_id[language.lower()]

            if (
                language.lower() in ["chinese", "auto"]
                and speaker != ""
                and speaker is not None
                and self.config.talker_config.spk_is_dialect[speaker.lower()]
            ):
                dialect = self.config.talker_config.spk_is_dialect[speaker.lower()]
                language_id = self.config.talker_config.codec_language_id[dialect]

            tts_bos_embed, tts_eos_embed, tts_pad_embed = self.talker.text_projection(
                self.talker.get_text_embeddings()(
                    torch.tensor(
                        [[self.config.tts_bos_token_id, self.config.tts_eos_token_id, self.config.tts_pad_token_id]],
                        device=self.talker.device,
                        dtype=input_id.dtype,
                    )
                )
            ).chunk(3, dim=1)  # 3 * [1 1 d]

            # codec: tag and speaker
            if language_id is None:
                codec_prefill_list = [
                    [
                        self.config.talker_config.codec_nothink_id,
                        self.config.talker_config.codec_think_bos_id,
                        self.config.talker_config.codec_think_eos_id,
                    ]
                ]
            else:
                codec_prefill_list = [
                    [
                        self.config.talker_config.codec_think_id,
                        self.config.talker_config.codec_think_bos_id,
                        language_id,
                        self.config.talker_config.codec_think_eos_id,
                    ]
                ]

            codec_input_emebdding_0 = self.talker.get_input_embeddings()(
                torch.tensor(
                    codec_prefill_list,
                    device=self.talker.device,
                    dtype=input_id.dtype,
                )
            )
            codec_input_emebdding_1 = self.talker.get_input_embeddings()(
                torch.tensor(
                    [
                        [
                            self.config.talker_config.codec_pad_id,
                            self.config.talker_config.codec_bos_id,
                        ]
                    ],
                    device=self.talker.device,
                    dtype=input_id.dtype,
                )
            )
            if speaker_embed is None:
                codec_input_emebdding = torch.cat([codec_input_emebdding_0, codec_input_emebdding_1], dim=1)
            else:
                codec_input_emebdding = torch.cat(
                    [codec_input_emebdding_0, speaker_embed.view(1, 1, -1), codec_input_emebdding_1], dim=1
                )

            # '<|im_start|>assistant\n我叫通义千问，是阿里云的开源大模型。<|im_end|>\n<|im_start|>assistant\n'

            # <|im_start|>assistant\n
            _talker_input_embed_role = self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, :3]))

            # tts_pad * 4 + tts_bos
            _talker_input_embed = (
                torch.cat(
                    (
                        tts_pad_embed.expand(-1, codec_input_emebdding.shape[1] - 2, -1),
                        tts_bos_embed,
                    ),
                    dim=1,
                )
                + codec_input_emebdding[:, :-1]
            )

            talker_input_embed = torch.cat((_talker_input_embed_role, _talker_input_embed), dim=1)

            if (
                voice_clone_prompt is not None
                and voice_clone_prompt["ref_code"] is not None
                and voice_clone_prompt["icl_mode"][index]
            ):
                icl_input_embed, trailing_text_hidden = self.generate_icl_prompt(
                    text_id=input_id[:, 3:-5],
                    ref_id=ref_ids[index][:, 3:-2],
                    ref_code=voice_clone_prompt["ref_code"][index].to(self.talker.device),
                    tts_pad_embed=tts_pad_embed,
                    tts_eos_embed=tts_eos_embed,
                    non_streaming_mode=non_streaming_mode,
                )
                talker_input_embed = torch.cat([talker_input_embed, icl_input_embed], dim=1)
            else:
                #  tts_text_first_token
                talker_input_embed = torch.cat(
                    [
                        talker_input_embed,
                        self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:4]))
                        + codec_input_emebdding[:, -1:],
                    ],
                    dim=1,
                )
                if non_streaming_mode:
                    talker_input_embed = talker_input_embed[:, :-1]  # 去掉原本放进去的text
                    talker_input_embed = torch.cat(
                        [
                            talker_input_embed,
                            torch.cat(
                                (
                                    self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 3:-5])),
                                    tts_eos_embed,
                                ),
                                dim=1,
                            )
                            + self.talker.get_input_embeddings()(
                                torch.tensor(
                                    [
                                        [
                                            self.config.talker_config.codec_pad_id,
                                        ]
                                        * (input_id[:, 3:-5].shape[1] + 1)
                                    ],
                                    device=self.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                            tts_pad_embed
                            + self.talker.get_input_embeddings()(
                                torch.tensor(
                                    [
                                        [
                                            self.config.talker_config.codec_bos_id,
                                        ]
                                    ],
                                    device=self.talker.device,
                                    dtype=input_id.dtype,
                                )
                            ),
                        ],
                        dim=1,
                    )
                    trailing_text_hidden = tts_pad_embed
                else:
                    # 叫通义千问，是阿里云的开源大模型。
                    trailing_text_hidden = torch.cat(
                        (
                            self.talker.text_projection(self.talker.get_text_embeddings()(input_id[:, 4:-5])),
                            tts_eos_embed,
                        ),
                        dim=1,
                    )
            talker_input_embeds[index].append(talker_input_embed)
            trailing_text_hiddens.append(trailing_text_hidden)

        for index, talker_input_embed in enumerate(talker_input_embeds):
            talker_input_embeds[index] = torch.cat([item for item in talker_input_embed if item is not None], dim=1)

        # for batch inferquence
        original_lengths = torch.tensor([t.shape[1] for t in talker_input_embeds])
        # left padding for talker input embeds
        sequences = [t.squeeze(0) for t in talker_input_embeds]
        sequences_reversed = [t.flip(dims=[0]) for t in sequences]
        padded_reversed = torch.nn.utils.rnn.pad_sequence(sequences_reversed, batch_first=True, padding_value=0.0)
        talker_input_embeds = padded_reversed.flip(dims=[1])
        # generate mask
        batch_size, max_len = talker_input_embeds.shape[0], talker_input_embeds.shape[1]
        indices = torch.arange(max_len).expand(batch_size, -1)
        num_pads = max_len - original_lengths
        talker_attention_mask = (indices >= num_pads.unsqueeze(1)).long().to(talker_input_embeds.device)
        # padding trailing text hiddens
        pad_embedding_vector = tts_pad_embed.squeeze()
        sequences_to_pad = [t.squeeze(0) for t in trailing_text_hiddens]
        trailing_text_original_lengths = [s.shape[0] for s in sequences_to_pad]
        padded_hiddens = torch.nn.utils.rnn.pad_sequence(sequences_to_pad, batch_first=True, padding_value=0.0)
        arange_tensor = torch.arange(max(trailing_text_original_lengths), device=padded_hiddens.device).expand(
            len(trailing_text_original_lengths), -1
        )
        lengths_tensor = torch.tensor(trailing_text_original_lengths, device=padded_hiddens.device).unsqueeze(1)
        padding_mask = arange_tensor >= lengths_tensor
        padded_hiddens[padding_mask] = pad_embedding_vector
        trailing_text_hiddens = padded_hiddens

        # forward
        talker_result = self.talker.generate(
            inputs_embeds=talker_input_embeds,
            attention_mask=talker_attention_mask,
            trailing_text_hidden=trailing_text_hiddens,
            tts_pad_embed=tts_pad_embed,
            **talker_kwargs,
        )

        talker_codes = torch.stack([hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1)
        # talker_hidden_states = torch.cat([hid[0][-1][:, -1:] for hid in talker_result.hidden_states], dim=1)[:, :-1]

        first_codebook = talker_codes[:, :, 0]
        is_stop_token = first_codebook == self.config.talker_config.codec_eos_token_id
        stop_indices = torch.argmax(is_stop_token.int(), dim=1)
        has_stop_token = is_stop_token.any(dim=1)
        effective_lengths = torch.where(has_stop_token, stop_indices, talker_codes.shape[1])

        talker_codes_list = [
            talker_codes[
                i,
                :length,
            ]
            for i, length in enumerate(effective_lengths)
        ]
        # talker_hidden_states_list = [talker_hidden_states[i, :length, :] for i, length in enumerate(effective_lengths)]

        return talker_codes_list, None


class XHQwen3TTSTalkerCodePredictorModelForConditionalGeneration(
    Qwen3TTSTalkerCodePredictorModelForConditionalGeneration
):
    def _setup(self) -> "Qwen3TTSTalkerCodePredictorModelForConditionalGeneration":
        linear_cnt = len(self.lm_head)
        out_features, in_features = self.lm_head[0].weight.shape  # in_features:1024, out_features:2048
        dtype = self.lm_head[0].weight.dtype
        device = self.lm_head[0].weight.device
        self.weight_embedding = nn.Embedding(linear_cnt, in_features * out_features)
        self.weight_embedding.to(device=device, dtype=dtype)

        # 初始化 embedding 权重
        for idx, lm_head in enumerate(self.lm_head):
            # lm_head.weight: [2048, 1024] -> 转置: [1024, 2048] -> 展平: [1024*2048]
            weight_t_flat = lm_head.weight.data.t().reshape(-1)
            self.weight_embedding.weight.data[idx] = weight_t_flat
        self.in_features = in_features
        self.out_features = out_features
        self.weight_embedding.to(self.lm_head[0].weight.device)
        del self.lm_head  # 删除原有的 lm_head
        return self

    @can_return_tuple
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
        generation_steps=None,
        **kwargs,
    ) -> Qwen3TTSTalkerCodePredictorOutputWithPast:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # # Prefill stage
        # if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
        #     generation_steps = inputs_embeds.shape[1] - 2  # hidden & layer 0
        # # Generation stage
        # else:
        #     inputs_embeds = self.model.get_input_embeddings()[generation_steps - 1](input_ids)
        inputs_embeds = self.small_to_mtp_projection(inputs_embeds)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # logits = self.lm_head[generation_steps](hidden_states)
        weight_t_flat = self.weight_embedding(generation_steps)  # [1024*2048]
        weight_t = weight_t_flat.view(self.in_features, self.out_features)  # [1024, 2048]

        # 使用 matmul 计算
        # x: [batch, seq, 1024], weight_t: [1024, 2048]
        # result: [batch, seq, 2048]
        logits = torch.matmul(hidden_states, weight_t)

        # loss = None
        # if labels is not None:
        #     loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return Qwen3TTSTalkerCodePredictorOutputWithPast(
            # loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
            # generation_steps=generation_steps + 1,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            **kwargs,
        )
        generation_steps = kwargs.get("generation_steps", -1)
        if inputs["inputs_embeds"] is None:
            input_ids = inputs["input_ids"]
            if isinstance(generation_steps, torch.Tensor):
                generation_steps = generation_steps.item()
            inputs["inputs_embeds"] = self.model.get_input_embeddings()[generation_steps - 1](input_ids)
            inputs["input_ids"] = None
        return inputs

    def _sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: LogitsProcessorList,
        stopping_criteria: StoppingCriteriaList,
        generation_config: GenerationConfig,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GenerateNonBeamOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            generation_config ([`~generation.GenerationConfig`]):
                The generation configuration to be used as parametrization of the decoding method.
            synced_gpus (`bool`):
                Whether to continue running the while loop until max_length (needed to avoid deadlocking with
                `FullyShardedDataParallel` and DeepSpeed ZeRO Stage 3).
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GenerateDecoderOnlyOutput`], [`~generation.GenerateEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GenerateDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GenerateEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.
        """
        # init values
        pad_token_id = generation_config._pad_token_tensor
        output_attentions = generation_config.output_attentions
        output_hidden_states = generation_config.output_hidden_states
        output_scores = generation_config.output_scores
        output_logits = generation_config.output_logits
        return_dict_in_generate = generation_config.return_dict_in_generate
        has_eos_stopping_criteria = any(hasattr(criteria, "eos_token_id") for criteria in stopping_criteria)
        do_sample = generation_config.do_sample

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        raw_logits = () if (return_dict_in_generate and output_logits) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        batch_size, cur_len = input_ids.shape[:2]
        this_peer_finished = False
        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=input_ids.device)
        model_kwargs = self._get_initial_cache_position(cur_len, input_ids.device, model_kwargs)

        model_forward = self.__call__
        # compile_forward = self._valid_auto_compile_criteria(model_kwargs, generation_config)
        # if compile_forward:
        #     os.environ["TOKENIZERS_PARALLELISM"] = "0"
        #     # If we use FA2 and a static cache, we cannot compile with fullgraph
        #     if self.config._attn_implementation == "flash_attention_2":
        #         # only raise warning if the user passed an explicit compile-config
        #         if generation_config.compile_config is not None and generation_config.compile_config.fullgraph:
        #             logger.warning_once(
        #                 "When using Flash Attention 2 and a static cache, you cannot use the option `CompileConfig(fullgraph=True)` as "
        #                 "FA2 introduces graph breaks. We overrode the option with `fullgraph=False`."
        #             )
        #             generation_config.compile_config.fullgraph = False
        #     model_forward = self.get_compiled_call(generation_config.compile_config)

        if generation_config.prefill_chunk_size is not None:
            model_kwargs = self._prefill_chunking(input_ids, generation_config, **model_kwargs)
            is_prefill = False
        else:
            is_prefill = True
        generation_steps = torch.tensor([0], device=input_ids.device)
        while self._has_unfinished_sequences(this_peer_finished, synced_gpus, device=input_ids.device):
            # prepare model inputs
            model_kwargs["generation_steps"] = generation_steps
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            model_inputs["generation_steps"] = generation_steps
            if is_prefill:
                outputs = self(**model_inputs, return_dict=True)
                is_prefill = False
            else:
                outputs = model_forward(**model_inputs, return_dict=True)
            generation_steps += 1
            outputs.generation_steps = generation_steps  # type: ignore
            # synced_gpus: don't waste resources running the code we don't need; kwargs must be updated before skipping
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs,
                model_kwargs,
                is_encoder_decoder=self.config.is_encoder_decoder,
            )
            if synced_gpus and this_peer_finished:
                continue

            # Copy is needed to avoid keeping a hanging ref to outputs.logits which may be very large for first iteration
            # (the clone itself is always small)
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_logits:
                    raw_logits += (next_token_logits,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,) if self.config.is_encoder_decoder else (outputs.hidden_states,)
                    )

            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if has_eos_stopping_criteria:
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())

            unfinished_sequences = unfinished_sequences & ~stopping_criteria(input_ids, scores)
            this_peer_finished = unfinished_sequences.max() == 0
            cur_len += 1

            # This is needed to properly delete outputs.logits which may be very large for first iteration
            # Otherwise a reference to outputs is kept which keeps the logits alive in the next iteration
            del outputs

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.config.is_encoder_decoder:
                return GenerateEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
            else:
                return GenerateDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    logits=raw_logits,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                    past_key_values=model_kwargs.get("past_key_values"),
                )
        else:
            return input_ids


class XHQwen3TTSModel(Qwen3TTSModel):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "XHQwen3TTSModel":
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.__class__ = XHQwen3TTSModel
        model = cast(XHQwen3TTSModel, model)
        model.model.__class__ = XHQwen3TTSForConditionalGeneration
        model.model.talker.__class__ = XHQwen3TTSTalkerForConditionalGeneration
        model.model.talker.code_predictor.__class__ = XHQwen3TTSTalkerCodePredictorModelForConditionalGeneration
        code_predictor = cast(
            XHQwen3TTSTalkerCodePredictorModelForConditionalGeneration, model.model.talker.code_predictor
        )
        code_predictor._setup()
        return model

    def to(self, *args, **kwargs) -> "XHQwen3TTSModel":
        """Overrides this method to call :meth:`BaseDataPreprocessor.to`
        additionally.

        Returns:
            nn.Module: The model itself.
        """

        # Since Torch has not officially merged
        # the npu-related fields, using the _parse_to function
        # directly will cause the NPU to not be found.
        # Here, the input parameters are processed to avoid errors.

        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self.model.to(device)
            self.device = self.model.device
            if self.model.speech_tokenizer is not None:
                self.model.speech_tokenizer.model.to(device)
                self.model.speech_tokenizer.device = device
            return self
        return self
