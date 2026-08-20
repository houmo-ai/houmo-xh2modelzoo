# ruff: noqa: E501

import math
from dataclasses import dataclass
from functools import wraps
from types import MethodType
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers.cache_utils import Cache, StaticCache
from transformers.modeling_outputs import BaseModelOutputWithPooling, CausalLMOutputWithPast, ModelOutput


def _official_tts_gen_logits(num_code: int, repetition_penalty: float):
    import sys

    gen_logits = None
    for module in sys.modules.values():
        if module is not None and str(getattr(module, "__name__", "")).endswith("modeling_minicpmo"):
            candidate = getattr(module, "gen_logits", None)
            if candidate is not None:
                gen_logits = candidate
                break
    if gen_logits is None:
        raise RuntimeError("official TTS sampling integration unavailable")
    return gen_logits(num_code=num_code, repetition_penalty=repetition_penalty)


def patch_tts_streaming_generator_cache_compat(generator_cls) -> None:
    if getattr(generator_cls, "_xh_cache_compat_patched", False):
        return
    original_generate = generator_cls.generate_with_buffer

    def generate_with_cache_compat(self, *args, **kwargs):
        supported = {"full_attention", "sliding_window", "sliding_recompute", "reindex"}
        if self.attention_type not in supported:
            raise ValueError(f"unsupported attention_type: {self.attention_type}")
        runtime_cache = getattr(self.tts.model, "hf_cache", None)
        for item in original_generate(self, *args, **kwargs):
            if runtime_cache is not None:
                self.past_key_values = runtime_cache
            yield item
        if runtime_cache is not None:
            self.past_key_values = runtime_cache

    generator_cls.generate_with_buffer = generate_with_cache_compat
    generator_cls._xh_cache_compat_patched = True


def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor | None,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    if attention_mask is not None and attention_mask.dim() == 4:
        return attention_mask

    causal_mask = torch.full(
        (sequence_length, target_length),
        fill_value=min_dtype,
        dtype=dtype,
        device=device,
    )
    if sequence_length != 1:
        causal_mask = torch.triu(causal_mask, diagonal=1)
    causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
    causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
    if attention_mask is not None:
        causal_mask = causal_mask.clone()
        mask_length = attention_mask.shape[-1]
        padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
        causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
            padding_mask == 0,
            min_dtype,
        )
    return causal_mask


class _AlignedSpeechOutputs(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error


def patch_tts_cache_position_compat(minicpm_model) -> None:
    tts_model = getattr(getattr(minicpm_model, "tts", None), "model", None)
    if tts_model is None or getattr(tts_model, "_xh_cache_position_patched", False):
        return

    original_forward = tts_model.forward

    @wraps(original_forward)
    def forward_with_cache_position_compat(*args, **kwargs):
        cache_position = kwargs.get("cache_position")
        if isinstance(cache_position, torch.Tensor) and cache_position.ndim == 2:
            if cache_position.shape[0] != 1:
                raise ValueError("MiniCPM TTS cache_position only supports batch size 1")
            kwargs["cache_position"] = cache_position[0]
        return original_forward(*args, **kwargs)

    tts_model.forward = forward_with_cache_position_compat
    tts_model._xh_cache_position_patched = True


def extract_tts_decode_alignment(outputs, tts_bound, tts_proj_layer):
    full_sequences = getattr(outputs, "full_sequences", None)
    if full_sequences is None:
        full_sequences = outputs["full_sequences"]
    full_sequence = full_sequences[0]
    decode_hidden_states = outputs.hidden_states[1:]
    input_ids_length = len(full_sequence) - len(decode_hidden_states)
    start = max(int(tts_bound[0]), input_ids_length)
    end = len(full_sequence) if tts_bound[1] is None else int(tts_bound[1])
    decode_start = start - input_ids_length
    decode_end = min(end - input_ids_length, len(decode_hidden_states))
    token_ids = full_sequence[start : input_ids_length + decode_end]
    selected_hidden_states = []
    for step in decode_hidden_states[decode_start:decode_end]:
        hidden_state = step[tts_proj_layer]
        while hidden_state.ndim > 2:
            hidden_state = hidden_state[0]
        selected_hidden_states.append(hidden_state)
    hidden_states = torch.vstack(selected_hidden_states)
    aligned_length = min(len(token_ids), hidden_states.shape[0])
    return token_ids[:aligned_length], hidden_states[:aligned_length]


# Official TTS sampling defaults (mirror of utils.TTSSamplingParams). Kept in one
# place so export (base model) and demo/golden (ensure_tts_sampling_config) agree.
TTS_SAMPLING_DEFAULTS: dict[str, float] = {
    "top_p": 0.85,
    "top_k": 25,
    "repetition_penalty": 1.05,
    "temperature": 0.8,
}


def ensure_tts_sampling_config(config):
    for name, value in TTS_SAMPLING_DEFAULTS.items():
        if not hasattr(config.tts_config, name):
            setattr(config.tts_config, name, value)
    return config


class _HMONNXVisionResampler(nn.Module):
    def forward(self, hidden_states, _target_sizes):
        return hidden_states


class _HMONNXAudoProjectionIdentity(nn.Module):
    """Identity stand-in for the audio projection layer.

    The streaming audio graphs export ``audio_projection_layer`` (1024 -> 4096)
    AND ``audio_avg_pooler`` (AvgPool1d pool_step=5) in-graph, mirroring the
    non-streaming main graph (_AudioFullEncoder folds apm -> projection ->
    pooling). The official ``get_audio_embedding_streaming`` host code still
    applies Step 2 (projection) and Step 3 (pooling) to the graph output, so
    both layers are replaced with identity to avoid double projection/pooling
    (which would also fail on the already-projected 4096-width / pooled input).
    The Host still slices by pooled lengths via _get_feat_extract_output_lengths.
    """

    def forward(self, audio_states):
        return audio_states


class _HMONNXAudioPoolerIdentity(nn.Module):
    """Identity stand-in for the audio avg-pooler (streaming graphs pool in-graph)."""

    def forward(self, audio_states):
        return audio_states


def patch_speech_generation_capture(minicpm_model):
    if getattr(minicpm_model, "_xh_speech_capture_patched", False):
        return
    original_generate = minicpm_model.tts.generate
    original_speech = minicpm_model._generate_speech_non_streaming
    patch_tts_cache_position_compat(minicpm_model)

    def generate_with_capture(*args, **kwargs):
        output = original_generate(*args, **kwargs)
        tokens = getattr(output, "new_ids", None)
        minicpm_model._xh_last_speech_tokens = None if tokens is None else tokens.detach().cpu()
        return output

    def speech_with_capture(*args, **kwargs):
        minicpm_model._xh_last_speech_tokens = None
        minicpm_model._xh_last_waveform = None
        outputs = kwargs.get("outputs")
        tts_bound = kwargs.get("tts_bound")
        tts_proj_layer = kwargs.get("tts_proj_layer")
        if (
            outputs is not None
            and tts_bound is not None
            and tts_proj_layer is not None
            and len(outputs.hidden_states) > 1
        ):
            token_ids, hidden_states = extract_tts_decode_alignment(outputs, tts_bound, tts_proj_layer)
            aligned_steps = tuple(
                (hidden_states[index : index + 1].unsqueeze(0),) for index in range(hidden_states.shape[0])
            )
            kwargs["outputs"] = _AlignedSpeechOutputs(
                full_sequences=token_ids.unsqueeze(0),
                hidden_states=aligned_steps,
            )
            kwargs["tts_bound"] = (0, None)
            kwargs["tts_proj_layer"] = 0
        waveform = original_speech(*args, **kwargs)
        if isinstance(waveform, torch.Tensor):
            minicpm_model._xh_last_waveform = waveform.detach().cpu()
        return waveform

    minicpm_model.tts.generate = generate_with_capture
    minicpm_model._generate_speech_non_streaming = speech_with_capture
    minicpm_model._xh_speech_capture_patched = True


def reuse_initialized_tts(host_model, tokenizer, create_duplex):
    original_init_tts = host_model.init_tts

    def reuse_tts(*args, **kwargs):
        del args, kwargs
        return tokenizer

    host_model.init_tts = reuse_tts
    try:
        return create_duplex()
    finally:
        host_model.init_tts = original_init_tts


def bind_duplex_hmonnx_reset(duplex, reset_components, is_active=lambda: True):
    if not hasattr(duplex, "model"):
        return duplex
    model = duplex.model
    if not hasattr(model, "init_streaming_processor"):
        return duplex
    if getattr(model, "_xh_duplex_reset_bound", False):
        model._xh_duplex_reset_components = reset_components
        model._xh_duplex_reset_pending = True
        return duplex
    original = model.init_streaming_processor

    def init_streaming_processor_with_reset(*args, **kwargs):
        if model._xh_duplex_reset_pending and is_active():
            model._xh_duplex_reset_components()
            model._xh_duplex_reset_pending = False
        return original(*args, **kwargs)

    init_streaming_processor_with_reset = wraps(original)(init_streaming_processor_with_reset)

    model._xh_duplex_reset_components = reset_components
    model._xh_duplex_reset_pending = True
    model.init_streaming_processor = init_streaming_processor_with_reset
    model._xh_duplex_reset_bound = True
    return duplex


def patch_dynamic_cache_legacy_methods(cache_cls) -> None:
    """Restore the legacy remote-code cache methods removed in Transformers 4.57."""
    if not hasattr(cache_cls, "seen_tokens"):
        cache_cls.seen_tokens = property(lambda self: self.get_seq_length())
    if not hasattr(cache_cls, "get_usable_length"):
        cache_cls.get_usable_length = lambda self, _new_seq_length, layer_idx=0: self.get_seq_length(layer_idx)
    if not hasattr(cache_cls, "key_cache"):
        cache_cls.key_cache = property(lambda self: [layer.keys for layer in self.layers])
    if not hasattr(cache_cls, "value_cache"):
        cache_cls.value_cache = property(lambda self: [layer.values for layer in self.layers])


def patch_dynamic_cache_seen_tokens(cache_cls) -> None:
    """Compatibility alias; the full legacy-method patch is consolidated above."""
    patch_dynamic_cache_legacy_methods(cache_cls)


def patch_remote_cache_helpers(model: object) -> None:
    module = __import__(type(model).__module__, fromlist=["get_kv_cache_length"])
    helper = getattr(module, "get_kv_cache_length", None)
    if helper is None:
        return

    def get_kv_cache_length(cache: object) -> int:
        layers = getattr(cache, "layers", None)
        if layers is not None:
            return 0 if not layers else int(layers[0].get_seq_length())
        get_seq_length = getattr(cache, "get_seq_length", None)
        if get_seq_length is not None:
            return int(get_seq_length())
        return int(helper(cache))

    module.get_kv_cache_length = get_kv_cache_length


def patch_empty_audio_cache(model: object) -> None:
    original = model.get_audio_embedding_streaming
    if getattr(original, "_xh_empty_cache_patched", False):
        return

    @wraps(original)
    def wrapped(*args, **kwargs):
        cache = getattr(model, "audio_past_key_values", None)
        self_attention_cache = None if cache is None else getattr(cache, "self_attention_cache", cache)
        get_seq_length = None if self_attention_cache is None else getattr(self_attention_cache, "get_seq_length", None)
        if cache is not None and (
            (get_seq_length is not None and get_seq_length() == 0)
            or (get_seq_length is None and len(self_attention_cache) == 0)
        ):
            model.audio_past_key_values = None
        elif self_attention_cache is not cache:
            model.audio_past_key_values = self_attention_cache
        return original(*args, **kwargs)

    wrapped._xh_empty_cache_patched = True
    model.get_audio_embedding_streaming = wrapped


def create_llm_wraped_cls(cls):
    class _MiniCPMOLLMWrapped(cls):
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
                    past_length = getattr(past_key_values, "seen_tokens", cache_length)
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
                model_inputs = {
                    "input_ids": input_ids.clone(memory_format=torch.contiguous_format),
                    "inputs_embeds": None,
                }
                if attention_mask is not None and position_ids is None:
                    # create position_ids on the fly for batch generation
                    position_ids = attention_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(attention_mask == 0, 1)
                    position_ids = position_ids[:, -1:]

            if isinstance(past_key_values, StaticCache) and attention_mask is not None and attention_mask.ndim == 2:
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
                    # Forward generation-time flags to the per-step model call.
                    # Without this, generate(output_hidden_states=True) reaches the
                    # HMONNX forward with None, which falls back to the config
                    # default (False), so every decode step records None hidden
                    # states and non-streaming TTS alignment crashes.
                    "output_hidden_states": kwargs.get("output_hidden_states"),
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

            del labels, output_attentions, position_ids, attention_mask, cache_position, logits_to_keep, kwargs
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("You must specify either input_ids or inputs_embeds")
                inputs_embeds = self.embed_tokens(input_ids)
            out = self._llm_model.forward_hf(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache if use_cache is not None else True,
                output_hidden_states=output_hidden_states if output_hidden_states is not None else False,
                return_dict=True,
            )

            self._past_seq_length += seq_length
            if self.prefill:
                self.prefill = False
            return out

        def generate(self, *args, **kwargs):
            self.prefill = True
            self._past_seq_length = 0
            self._llm_model.set_num_logits_to_keep(1)
            kwargs["num_beams"] = 1
            kwargs["num_return_sequences"] = 1
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

            del attention_mask, position_ids, labels, output_attentions, cache_position, logits_to_keep, kwargs
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            use_cache = use_cache if use_cache is not None else self.config.use_cache
            if inputs_embeds is None:
                if input_ids is None:
                    raise ValueError("You must specify either input_ids or inputs_embeds")
                inputs_embeds = self.embed_tokens(input_ids)
            return self._llm_model.forward_hf(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_hidden_states=output_hidden_states,
                return_dict=True,
            )

    return _MiniCPMOLLMWrapped


def create_vision_wraped_cls(cls):
    class _SiglipVisionTransformer(cls):
        def forward(
            self,
            data=None,
            pixel_values=None,
            patch_attention_mask=None,
            tgt_sizes=None,
            **kwargs,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
            del kwargs
            if isinstance(data, torch.Tensor):
                pixel_values = data
                prepare_inputs = getattr(self._vision_model, "prepare_inputs", None)
                if prepare_inputs is None:
                    raise RuntimeError("HMONNX vision runtime cannot prepare official tensor inputs")
                prepared = prepare_inputs(
                    {"pixel_values": [[image] for image in pixel_values], "tgt_sizes": [tgt_sizes]}
                )
                vision_embedding = self._vision_model._wrap_model(*prepared[:4])
                return BaseModelOutputWithPooling(last_hidden_state=vision_embedding)
            if data is None:
                data = {
                    "pixel_values": pixel_values,
                    "patch_attention_mask": patch_attention_mask,
                    "tgt_sizes": tgt_sizes,
                }
            all_pixel_values, position_ids, attention_mask, tgt_sizes, imgs_cnt = self._vision_model.prepare_inputs(
                data
            )
            if isinstance(tgt_sizes, torch.Tensor):
                if tgt_sizes.dim() == 1:
                    tgt_sizes = tgt_sizes.unsqueeze(0)
                elif tgt_sizes.dim() > 2:
                    tgt_sizes = tgt_sizes.reshape(tgt_sizes.shape[0], -1)
                if tgt_sizes.shape[-1] < 2:
                    tgt_sizes = tgt_sizes.repeat(1, 2)
                elif tgt_sizes.shape[-1] > 2:
                    tgt_sizes = tgt_sizes[:, :2]

            vision_embedding = self._vision_model._wrap_model(
                all_pixel_values,
                position_ids,
                attention_mask,
                tgt_sizes,
            )

            start = 0
            vision_hidden_states = []
            for img_cnt in imgs_cnt:
                if img_cnt > 0:
                    vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                    start += img_cnt
                else:
                    vision_hidden_states.append(torch.tensor([]))
            return BaseModelOutputWithPooling(last_hidden_state=vision_hidden_states)

    return _SiglipVisionTransformer


def create_audio_wraped_cls(cls):
    class _MiniCPMWhisperEncoderWrapper(cls):
        def forward(
            self,
            input_features,
            attention_mask=None,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            past_key_values=None,
            use_cache=None,
            use_extra_context=False,
            prefix_extra_frames=1,
            suffix_extra_frames=1,
            cnn_min_length=None,
            valid_mel_length=None,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
            del head_mask, output_attentions, return_dict, cnn_min_length
            if use_cache or past_key_values is not None:
                # A mid-stream cache reset (official drops past_key_values while the chunk
                # still carries later-chunk CNN overlap) must keep the later prefix; only a
                # truly empty cache is a first chunk.
                expected_prefix = (
                    self._audio_model.prefix_overlap_first
                    if self._audio_model.streaming_cache_length == 0
                    else self._audio_model.prefix_overlap_later
                )
                if use_extra_context and prefix_extra_frames != expected_prefix:
                    raise RuntimeError(
                        f"Streaming Audio prefix overlap mismatch: expected {expected_prefix}, "
                        f"got {prefix_extra_frames}"
                    )
                if use_extra_context and suffix_extra_frames != self._audio_model.suffix_overlap:
                    raise RuntimeError(
                        f"Streaming Audio suffix overlap mismatch: expected {self._audio_model.suffix_overlap}, "
                        f"got {suffix_extra_frames}"
                    )
                # Do not reset here: forward_streaming must observe the populated cache with a
                # dropped past_key_values to detect a mid-stream reset and keep decode geometry.
                output = self._audio_model.forward_streaming(
                    input_features,
                    valid_mel_length=(
                        getattr(self, "_streaming_audio_feature_lens", input_features.shape[-1])
                        if valid_mel_length is None
                        else valid_mel_length
                    ),
                    past_key_values=past_key_values,
                    use_extra_context=use_extra_context,
                )
                if output_hidden_states is False:
                    output.hidden_states = None
                return output
            return self._audio_model.forward(input_features, attention_mask)

    return _MiniCPMWhisperEncoderWrapper


def patch_audio_attention_return_compat(minicpm_model):
    apm = getattr(minicpm_model, "apm", None)
    layers = getattr(apm, "layers", None)
    if not layers:
        return

    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        if self_attn is None or getattr(self_attn, "_xh_return_compat_patched", False):
            continue

        original_forward = self_attn.forward

        def forward_with_compat(original_forward=original_forward, *args, **kwargs):
            outputs = original_forward(*args, **kwargs)
            if isinstance(outputs, tuple) and len(outputs) == 2:
                cache = kwargs.get("past_key_values", kwargs.get("past_key_value"))
                return outputs[0], outputs[1], cache
            return outputs

        self_attn.forward = forward_with_compat
        self_attn._xh_return_compat_patched = True


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
        def _normalize_tts_logits(self, model_output):
            """Normalize wrapped model outputs to logits tensor with shape [B, S, C, VQ]."""
            if isinstance(model_output, torch.Tensor):
                logits = model_output
            elif hasattr(model_output, "logits") and model_output.logits is not None:
                logits = model_output.logits
            elif hasattr(model_output, "last_hidden_state") and model_output.last_hidden_state is not None:
                hidden_states = model_output.last_hidden_state
                if not hasattr(self, "head_code"):
                    raise RuntimeError("TTS output is hidden_states but head_code is missing.")
                logits_per_vq = [self._head_code_hmonnx(hidden_states) for _ in range(self.num_vq)]
                logits = torch.stack(logits_per_vq, dim=-1)
            elif isinstance(model_output, (tuple, list)) and len(model_output) > 0:
                logits = model_output[0]
            else:
                raise TypeError(f"Unsupported TTS model output type: {type(model_output)}")

            if logits.dim() == 4:
                return logits
            if logits.dim() == 3 and logits.shape[-1] == self.config.num_audio_tokens and self.num_vq == 1:
                return logits.unsqueeze(-1)
            if logits.dim() == 3 and hasattr(self, "head_code"):
                logits_per_vq = [self._head_code_hmonnx(logits) for _ in range(self.num_vq)]
                return torch.stack(logits_per_vq, dim=-1)
            raise RuntimeError(f"Unexpected TTS logits shape {tuple(logits.shape)}; expected [B, S, C, VQ].")

        def _head_code_hmonnx(self, hidden_states):
            """Route tts.head_code through the HMONNX head-code graph (num_vq == 1)."""
            tts_llama_model = getattr(self, "model", None)
            project = getattr(tts_llama_model, "project_head_code", None)
            if project is None:
                raise RuntimeError("TTS head_code HMONNX graph is not available")
            return project(hidden_states)

        @torch.inference_mode()
        def generate_chunk(
            self,
            inputs_embeds: torch.Tensor,
            temperature: torch.Tensor,
            repetition_penalty: float,
            eos_token: Union[int, torch.Tensor],
            force_no_stop=False,
            max_new_token=500,
            min_new_tokens=0,
            past_key_values=None,
            logits_processors=None,
            text_start_pos=None,
        ):
            # Match the official generate_chunk: it rebuilds warpers/processors via gen_logits
            # and ignores the caller-provided logits_processors (duplex forwards the whole
            # gen_logits() tuple there, which must not be iterated as processors).
            official_warpers, official_processors = _official_tts_gen_logits(
                self.config.num_audio_tokens,
                repetition_penalty,
            )
            del logits_processors
            processors = official_processors
            eos_token = eos_token.to(inputs_embeds.device)
            temperature = temperature.unsqueeze(0).expand(inputs_embeds.shape[0], -1).contiguous().view(-1, 1)
            condition_length = inputs_embeds.shape[1]
            generated = torch.zeros(
                inputs_embeds.shape[0],
                max_new_token,
                self.num_vq,
                device=inputs_embeds.device,
                dtype=torch.long,
            )
            finished = torch.zeros(inputs_embeds.shape[0], dtype=torch.bool, device=inputs_embeds.device)
            start = 0 if text_start_pos is None else text_start_pos
            for step in range(max_new_token):
                if step == 0:
                    current_embeds = inputs_embeds
                    position_ids = torch.arange(
                        start,
                        start + condition_length,
                        device=inputs_embeds.device,
                    ).unsqueeze(0)
                else:
                    current_embeds = self.emb_code[0](generated[:, step - 1 : step, 0])
                    position_ids = torch.tensor(
                        [[start + condition_length + step - 1]],
                        dtype=torch.long,
                        device=inputs_embeds.device,
                    )
                outputs = self._tts_llama_model.forward_hf(
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=current_embeds,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = outputs.past_key_values
                logits = self._normalize_tts_logits(outputs)[:, -1].float().permute(0, 2, 1)
                logits = logits.reshape(-1, logits.size(2)) / temperature
                if step > 0:
                    previous = generated[:, :step].permute(0, 2, 1).reshape(-1, step)
                    for processor in processors:
                        logits = processor(previous, logits)
                if force_no_stop or step < min_new_tokens:
                    logits[:, eos_token] = -torch.inf
                next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).view(-1, self.num_vq)
                finished.logical_or_(next_token.eq(eos_token).any(1))
                generated[:, step] = next_token
                if finished.all():
                    break
            return generated[:, :step, :], past_key_values

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
            if hasattr(self, "merge_inputs_embeds"):
                # 2.6: ConditionalChatTTS has merge_inputs_embeds
                inputs_embeds = self.merge_inputs_embeds(
                    input_ids=input_ids,
                    lm_spk_emb_last_hidden_states=lm_spk_emb_last_hidden_states,
                )
            else:
                # 4.5: MiniCPMTTS - embed text directly
                inputs_embeds = self.emb_text(input_ids)

            self._tts_llama_model.forward_hf(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
                return_dict=True,
            )
            past_key_caches = self._tts_llama_model.past_key_caches
            past_value_caches = self._tts_llama_model.past_value_caches

            # Get model updated KV Cache
            # past_key_values_for_prefill_updated = outputs_prefill.past_key_values

            # # Update generated KV Cache to input `past_key_values`
            for layer_idx in range(len(past_key_values)):
                # Update keys
                past_key_values[layer_idx][0][:, :, position_ids[:, 0] : position_ids[:, -1] + 1, :] = past_key_caches[
                    layer_idx
                ][:, :, position_ids[:, 0] : position_ids[:, -1] + 1].clone()
                # Update values
                past_key_values[layer_idx][1][:, :, position_ids[:, 0] : position_ids[:, -1] + 1, :] = (
                    past_value_caches[layer_idx][:, :, position_ids[:, 0] : position_ids[:, -1] + 1].clone()
                )

            return past_key_values

        @torch.inference_mode()
        def generate(
            self,
            input_ids: torch.Tensor = None,
            past_key_values: List[Tuple[torch.Tensor, torch.Tensor]] = None,
            temperature: torch.Tensor = None,
            eos_token: Union[int, torch.Tensor] = None,
            streaming_tts_text_mask=None,
            force_no_stop=False,
            min_new_token=10,
            max_new_token=50,
            logits_warpers=None,
            logits_processors=None,
            show_tqdm=False,
            # 4.5 MiniCPMTTS non-streaming args
            inputs_embeds: torch.Tensor = None,
            sampling_params=None,
            **kwargs,
        ):
            if inputs_embeds is not None:
                # 4.5 MiniCPMTTS non-streaming path: generate from inputs_embeds
                return self._generate_non_streaming(
                    inputs_embeds=inputs_embeds,
                    eos_token=eos_token,
                    force_no_stop=force_no_stop,
                    max_new_token=max_new_token,
                    sampling_params=sampling_params,
                    show_tqdm=show_tqdm,
                    **kwargs,
                )

            if logits_warpers is None:
                logits_warpers = []
            if logits_processors is None:
                logits_processors = []

            # streaming path: generate from input_ids with past_key_values
            # We only support batch size `1` for now
            assert input_ids.shape[0] == 1
            assert past_key_values is not None

            # fix: this should not be `input_ids.shape[1]`
            # start_idx = input_ids.shape[1]
            _num_spk_embs = getattr(self, "num_spk_embs", 0)
            _use_spk_emb = getattr(self, "use_speaker_embedding", False)
            start_idx = 1 + _num_spk_embs * _use_spk_emb + self.streaming_text_reserved_len + 1

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

            condition_length = 1 + _num_spk_embs * _use_spk_emb + self.streaming_text_reserved_len + 1

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

                current_input_length = torch.tensor([position_ids.shape[-1]], dtype=torch.int32).to(
                    inputs_embeds.device
                )

                past_key_caches = self._tts_llama_model.past_key_caches
                past_value_caches = self._tts_llama_model.past_value_caches

                self._tts_llama_model.set_input_sequence_length(current_input_length.cpu().item())

                attention_mask = (
                    torch.ones((self._tts_llama_model.max_sequence_length,), dtype=torch.float16).to(
                        inputs_embeds.device
                    )
                    * torch.finfo(torch.float16).min
                )

                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                attention_mask[:, :, :, : causal_mask.shape[-1]] = causal_mask

                outputs = self._tts_llama_model.forward_hf(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                logits = outputs.last_hidden_state
                past_key_values = outputs.past_key_values
                logits = self._normalize_tts_logits(logits)

                for layer_idx in range(len(past_key_values)):
                    past_key_values[layer_idx] = (
                        torch.cat(
                            (
                                past_key_values[layer_idx][0],
                                past_key_caches[layer_idx][:, :, position_ids[:, 0] : position_ids[:, -1] + 1].clone(),
                            ),
                            dim=2,
                        ),
                        torch.cat(
                            (
                                past_key_values[layer_idx][1],
                                past_value_caches[layer_idx][
                                    :, :, position_ids[:, 0] : position_ids[:, -1] + 1
                                ].clone(),
                            ),
                            dim=2,
                        ),
                    )

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

        @torch.inference_mode()
        def _generate_non_streaming(
            self,
            inputs_embeds: torch.Tensor,
            eos_token: Union[int, torch.Tensor] = None,
            force_no_stop=False,
            max_new_token=2048,
            sampling_params=None,
            show_tqdm=False,
            min_new_token=50,
            **kwargs,
        ):
            """Non-streaming TTS generation for 4.5 MiniCPMTTS.

            Takes pre-built inputs_embeds (including text embeds + audio_bos) and
            generates audio tokens autoregressively using the xhquant TTS model.
            """
            # Setup sampling parameters
            if sampling_params is not None:
                _temperature = getattr(sampling_params, "temperature", 0.1)
                _top_p = getattr(sampling_params, "top_p", 0.7)
                _top_k = getattr(sampling_params, "top_k", 20)
                _rep_penalty = getattr(sampling_params, "repetition_penalty", 1.0)
            else:
                _temperature = 0.1
                _top_p = 0.7
                _top_k = 20
                _rep_penalty = 1.0

            temperature = torch.tensor([_temperature] * self.num_vq, dtype=torch.float, device=self.device)
            temperature = temperature.unsqueeze(0).expand(1, -1).contiguous().view(-1, 1)

            logits_warpers, logits_processors = _official_tts_gen_logits(
                self.config.num_audio_tokens,
                _rep_penalty,
            )

            assert inputs_embeds.shape[0] == 1
            if eos_token is not None:
                eos_token = eos_token.to(inputs_embeds.device)
            finish = torch.zeros(1, device=inputs_embeds.device).bool()

            condition_length = inputs_embeds.shape[1]
            new_tokens = torch.zeros(
                1,
                max_new_token,
                self.num_vq,
                device=inputs_embeds.device,
                dtype=torch.long,
            )

            pbar = None
            if show_tqdm:
                pbar = tqdm(
                    total=max_new_token,
                    desc="code",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}(max) [{elapsed}, {rate_fmt}{postfix}]",
                )

            # Prefill with full inputs_embeds
            mask_dtype = inputs_embeds.dtype
            # 构建正确的 causal attention mask：
            # 1. mask 掉 KV cache 中 condition_length 之后的无效位置（垃圾值）
            # 2. 在有效位置内使用因果 (lower-triangular) masking
            # 原始全零 mask 导致所有位置双向关注（包括 text_eos/audio_bos），
            # 严重破坏 decoder-only 模型行为，导致 TTS 过早生成 EOS。
            # 原生 TTS 模型传 attention_mask=None 给 LlamaModel，内部自动构建 causal mask。
            _finfo_min = torch.finfo(mask_dtype).min
            _max_seq = self._tts_llama_model.max_sequence_length
            # 构建 [condition_length, max_seq] 的因果 mask
            attention_mask = torch.full(
                (condition_length, _max_seq),
                _finfo_min,
                dtype=mask_dtype,
                device=inputs_embeds.device,
            )
            # 因果: position i 只能关注 positions 0..i
            for i in range(condition_length):
                attention_mask[i, : i + 1] = 0.0
            attention_mask = attention_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, cond_len, max_seq]

            self._tts_llama_model.set_input_sequence_length(condition_length)
            outputs = self._tts_llama_model.forward_hf(
                inputs_embeds=inputs_embeds,
                past_key_values=None,
                attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            logits = outputs.last_hidden_state
            logits = self._normalize_tts_logits(logits)

            progress = condition_length

            for t in range(max_new_token):
                # Get logits from last position: [batch, seq_len, num_audio_tokens, num_vq]
                logits_t = logits.narrow(1, -1, 1).squeeze_(1).float()
                # [batch, num_audio_tokens, num_vq] -> [batch*num_vq, num_audio_tokens]
                logits_t = logits_t.permute(0, 2, 1).reshape(-1, logits_t.size(1))
                logits_t /= temperature

                if t > 0:
                    logits_token = new_tokens[:, :t].permute(0, 2, 1).reshape(-1, t).to(self.device)
                    for lp in logits_processors:
                        logits_t = lp(logits_token, logits_t)
                    for lw in logits_warpers:
                        logits_t = lw(logits_token, logits_t)
                    del logits_token

                if t < min_new_token and eos_token is not None:
                    logits_t[:, eos_token] = -torch.inf
                if force_no_stop and eos_token is not None:
                    logits_t[:, eos_token] = -torch.inf

                scores = F.softmax(logits_t, dim=-1)
                del logits_t
                idx_next = torch.multinomial(scores, num_samples=1).view(-1, self.num_vq)
                del scores

                if eos_token is not None:
                    finish.logical_or_(idx_next.eq(eos_token).any(1))
                new_tokens[:, t] = idx_next

                if finish.all():
                    break

                # Embed next token for decode step
                code_emb = [self.emb_code[i](idx_next[:, i : i + 1]) for i in range(self.num_vq)]
                next_embeds = torch.stack(code_emb, 3).sum(3)
                del idx_next

                # Decode mask：只关注 0..progress 的有效 KV 位置，mask 掉后续未初始化位置
                attention_mask = torch.full(
                    (self._tts_llama_model.max_sequence_length,),
                    _finfo_min,
                    dtype=mask_dtype,
                    device=inputs_embeds.device,
                )
                attention_mask[: progress + 1] = 0.0  # 位置 0..progress 有效
                attention_mask = attention_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)

                self._tts_llama_model.set_input_sequence_length(1)
                outputs = self._tts_llama_model.forward_hf(
                    inputs_embeds=next_embeds,
                    past_key_values=outputs.past_key_values,
                    attention_mask=attention_mask,
                    use_cache=True,
                    return_dict=True,
                )
                logits = outputs.last_hidden_state
                logits = self._normalize_tts_logits(logits)
                del next_embeds
                progress += 1

                if pbar is not None:
                    pbar.update(1)

            if pbar is not None:
                pbar.close()

            # Trim to actual generated length
            actual_len = t + 1 if not finish.all() else t
            generated = new_tokens[:, :actual_len]

            return ConditionalChatTTSGenerationOutput(
                new_ids=generated,
                audio_input_ids=None,
                past_key_values=None,
                finished=finish.all(),
            )

    return _ChatTTSModel


def create_minicpo_wraped_cls(cls):
    class _MiniCPMO(cls):
        def get_audio_embedding_streaming(
            self,
            data,
            use_extra_context=False,
            prefix_extra_frames=1,
            suffix_extra_frames=1,
            cnn_min_length=None,
        ):
            length_tensors = [
                length
                for item in data["audio_feature_lens"]
                for length in (item if isinstance(item, (list, tuple)) else (item,))
            ]
            if not length_tensors:
                return self._old_get_audio_embedding_streaming(
                    data,
                    use_extra_context=use_extra_context,
                    prefix_extra_frames=prefix_extra_frames,
                    suffix_extra_frames=suffix_extra_frames,
                    cnn_min_length=cnn_min_length,
                )
            audio_feature_lens = torch.hstack(length_tensors)
            self.apm._streaming_audio_feature_lens = audio_feature_lens
            try:
                return self._old_get_audio_embedding_streaming(
                    data,
                    use_extra_context=use_extra_context,
                    prefix_extra_frames=prefix_extra_frames,
                    suffix_extra_frames=suffix_extra_frames,
                    cnn_min_length=cnn_min_length,
                )
            finally:
                del self.apm._streaming_audio_feature_lens

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
                apm_out = self.apm(wavforms.to(dtype), audio_attention_mask)
                if isinstance(apm_out, torch.Tensor):
                    audio_embeds = apm_out
                elif hasattr(apm_out, "hidden_states") and apm_out.hidden_states is not None:
                    audio_states = apm_out.hidden_states[self.audio_encoder_layer]
                    audio_embeds = self.audio_projection_layer(audio_states)
                    audio_embeds = audio_embeds.transpose(1, 2)
                    audio_embeds = self.audio_avg_pooler(audio_embeds)
                    audio_embeds = audio_embeds.transpose(1, 2)
                elif hasattr(apm_out, "last_hidden_state") and apm_out.last_hidden_state is not None:
                    audio_states = apm_out.last_hidden_state
                    audio_embeds = self.audio_projection_layer(audio_states)
                    audio_embeds = audio_embeds.transpose(1, 2)
                    audio_embeds = self.audio_avg_pooler(audio_embeds)
                    audio_embeds = audio_embeds.transpose(1, 2)
                elif isinstance(apm_out, (tuple, list)) and len(apm_out) > 0:
                    audio_embeds = apm_out[0]
                else:
                    raise TypeError(f"Unexpected apm output type for audio embedding: {type(apm_out)}")
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
            if "vision_hidden_states" not in data:
                pixel_values_list = data["pixel_values"]
                vision_hidden_states = []
                all_pixel_values = []
                img_cnt = []
                for pixel_values in pixel_values_list:
                    img_cnt.append(len(pixel_values))
                    all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

                # exist image
                if all_pixel_values:
                    vision_hidden_states = self.vpm(data).last_hidden_state
                else:  # no image
                    if self.training:
                        raise RuntimeError("Training mode must have image.")
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
                i.to(device=vllm_embedding.device, dtype=vllm_embedding.dtype) if isinstance(i, torch.Tensor) else i
                for i in vision_hidden_states
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


class MiniCPMO_HFCompatible:  # noqa: N801
    def __init__(self, *args, **kwargs):
        """Initializing a dynamic module is not allowed!"""
        raise RuntimeError("DynamicModule cannot be initialized directly; use convert instead!")

    @classmethod
    def to_hf_compatible(
        cls,
        hf_model,  # MiniCPMO (2.6 or 4.5)
        vision_model=None,
        llm_model=None,
        audio_model=None,
        tts_llama_model=None,
    ):
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        original_audio_streaming = (
            hf_model.get_audio_embedding_streaming
            if audio_model is not None and hasattr(hf_model, "get_audio_embedding_streaming")
            else None
        )

        def _ensure_cache_seen_tokens(past_key_values):
            if isinstance(past_key_values, Cache) and not hasattr(past_key_values, "seen_tokens"):
                try:
                    past_key_values.seen_tokens = past_key_values.get_seq_length()
                except Exception:
                    pass

        if llm_model is not None:
            minicpm_llm = hf_model.llm
            _llm_cls = create_llm_wraped_cls(type(minicpm_llm))
            minicpm_llm.__class__ = _llm_cls
            minicpm_llm._modules.pop("_llm_model", None)
            object.__setattr__(minicpm_llm, "_llm_model", llm_model)
            minicpm_llm.embed_tokens = hf_model.llm.model.embed_tokens
            # In upstream modeling, `self.llm.prepare_inputs_for_generation` is monkey-patched on the instance.
            # An instance attribute shadows the class method, so our wrapper's implementation won't be invoked.
            # Remove the instance-bound attribute to fall back to the class method we define here.
            if "prepare_inputs_for_generation" in minicpm_llm.__dict__:
                delattr(minicpm_llm, "prepare_inputs_for_generation")

            patch_audio_attention_return_compat(hf_model)

        minicpm_llm = getattr(hf_model, "llm", None)
        if minicpm_llm is not None and hasattr(minicpm_llm, "prepare_inputs_for_generation"):
            _orig_prepare_inputs_for_generation = minicpm_llm.prepare_inputs_for_generation

            def _prepare_inputs_for_generation_compat(
                self,
                input_ids=None,
                past_key_values=None,
                attention_mask=None,
                inputs_embeds=None,
                cache_position=None,
                **kwargs,
            ):
                _ensure_cache_seen_tokens(past_key_values)

                model_inputs = _orig_prepare_inputs_for_generation(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    cache_position=cache_position,
                    **kwargs,
                )

                if isinstance(model_inputs, dict):
                    _ensure_cache_seen_tokens(model_inputs.get("past_key_values", None))
                return model_inputs

            minicpm_llm.prepare_inputs_for_generation = MethodType(_prepare_inputs_for_generation_compat, minicpm_llm)

        if vision_model is not None or audio_model is not None or llm_model is not None or tts_llama_model is not None:
            minicpmo_cls = create_minicpo_wraped_cls(type(hf_model))
            hf_model.__class__ = minicpmo_cls

        if vision_model is not None:
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
            hf_model.resampler = _HMONNXVisionResampler()

        if audio_model is not None:
            minicpo_apm = hf_model.apm
            _audio_cls = create_audio_wraped_cls(type(minicpo_apm))
            minicpo_apm.__class__ = _audio_cls
            minicpo_apm._audio_model = audio_model
            hf_model.audio_projection_layer = _HMONNXAudoProjectionIdentity()
            hf_model.audio_avg_pooler = _HMONNXAudioPoolerIdentity()
            hf_model._old_get_audio_embedding = hf_model.get_audio_embedding
            hf_model.get_audio_embedding = hf_model._get_audio_embedding
            if original_audio_streaming is not None:
                hf_model._old_get_audio_embedding_streaming = MethodType(
                    original_audio_streaming.__func__
                    if hasattr(original_audio_streaming, "__func__")
                    else original_audio_streaming,
                    hf_model,
                )

        if tts_llama_model is not None:
            minicpo_tts_model = hf_model.tts
            _tts_cls = create_tts_model_wraped_cls(type(minicpo_tts_model))
            minicpo_tts_model.__class__ = _tts_cls
            minicpo_tts_model._tts_llama_model = tts_llama_model
            minicpo_tts_model.model = tts_llama_model
            if getattr(tts_llama_model, "projector_semantic_session", None) is not None:
                from .runtime_tts import ProjectorSemanticGraph

                minicpo_tts_model.projector_semantic = ProjectorSemanticGraph(tts_llama_model)

            patch_audio_attention_return_compat(hf_model)
        return hf_model


MiniCPMOHFCompatible = MiniCPMO_HFCompatible
