from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from torch import Tensor


if TYPE_CHECKING:
    from collections.abc import Sequence


def build_streaming_attention_mask(
    past_seq_length: Tensor,
    current_input_length: Tensor,
    query_capacity: int,
    cache_capacity: int,
    reference: Tensor,
) -> Tensor:
    cache_positions = torch.arange(cache_capacity).to(reference)
    valid_cache_length = past_seq_length + current_input_length
    valid_columns = cache_positions.reshape(1, 1, 1, cache_capacity) < valid_cache_length.reshape(1, 1, 1, 1)
    mask = torch.where(
        valid_columns,
        reference.new_zeros(()),
        reference.new_full((), float("-inf")),
    )
    return mask.expand(1, 1, query_capacity, cache_capacity)


def streaming_position_ids(output_capacity: int, past_seq_length: Tensor, reference: Tensor) -> Tensor:
    del reference
    return torch.arange(output_capacity).to(past_seq_length) + past_seq_length


class StreamingWhisperEncoderMixin:
    def forward_streaming(
        self,
        input_features: Tensor,
        valid_mel_length: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        attention_mask: Tensor,
        past_key_caches: Sequence[Tensor],
        past_value_caches: Sequence[Tensor],
    ) -> tuple[Tensor, ...]:
        del valid_mel_length
        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        prefix_remove = (self.streaming_prefix_extra_frames + 1) // 2
        suffix_remove = (self.streaming_suffix_extra_frames + 1) // 2
        if prefix_remove:
            inputs_embeds = inputs_embeds[:, :, prefix_remove:]
        if suffix_remove:
            inputs_embeds = inputs_embeds[:, :, :-suffix_remove]
        hidden_states = inputs_embeds.permute(0, 2, 1)
        position_ids = streaming_position_ids(
            self.streaming_output_capacity,
            past_seq_length,
            hidden_states,
        )
        hidden_states = hidden_states + self.embed_positions(position_ids)
        encoder_states = ()
        present_key_caches = ()
        present_value_caches = ()
        for index, layer in enumerate(self.layers):
            encoder_states += (hidden_states,)
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                layer_head_mask=None,
                output_attentions=False,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_k_cache=past_key_caches[index],
                past_v_cache=past_value_caches[index],
            )
            hidden_states = layer_outputs[0]
            present_key_caches += (layer_outputs[-2],)
            present_value_caches += (layer_outputs[-1],)
        hidden_states = self.layer_norm(hidden_states)
        encoder_states += (hidden_states,)
        audio_states = encoder_states[self.audio_encoder_layer]
        # The streaming graphs export the audio projection AND pooling in-graph,
        # mirroring the non-streaming main graph (_AudioFullEncoder folds
        # apm -> projection -> pooling). AvgPool1d(pool_step=5) is a deterministic
        # frame downsampler; its output matches the official host path
        # (audio_avg_pooler + _get_feat_extract_output_lengths slicing) exactly
        # for any valid region, so no precision is lost by moving it into the graph.
        audio_states = self.audio_projection_layer(audio_states)
        audio_states = audio_states.transpose(1, 2)
        audio_states = self.audio_avg_pooler(audio_states)
        audio_states = audio_states.transpose(1, 2)
        return (audio_states, *present_key_caches, *present_value_caches)


def _build_streaming_export_forward(num_hidden_layers: int):
    # A fixed-position signature is required here: TorchFX/ONNX tracing names each
    # cache input explicitly (past_k_cache_0..N, past_v_cache_0..N) and the quantized
    # graph contract depends on those stable names. A `*past_caches` variadic would
    # collapse them into one un-named input and break the export, so the forward is
    # generated once per layer count instead of written out 24 times by hand.
    cache_names = [
        *(f"past_k_cache_{index}" for index in range(num_hidden_layers)),
        *(f"past_v_cache_{index}" for index in range(num_hidden_layers)),
    ]
    args = [
        "self",
        "input_features",
        "valid_mel_length",
        "past_seq_length",
        "current_input_length",
        "attention_mask",
        *cache_names,
    ]
    keys = cache_names[:num_hidden_layers]
    values = cache_names[num_hidden_layers:]
    source = (
        f"def forward({', '.join(args)}):\n"
        "    return self.encoder.forward_streaming(\n"
        "        input_features, valid_mel_length, past_seq_length, current_input_length, attention_mask,\n"
        f"        [{', '.join(keys)}],\n"
        f"        [{', '.join(values)}],\n"
        "    )\n"
    )
    namespace: dict[str, Any] = {}
    exec(source, {}, namespace)
    return namespace["forward"]


def StreamingAudioExportAdapter(encoder: nn.Module, num_hidden_layers: int) -> nn.Module:  # noqa: N802
    class _StreamingAudioExportAdapter(nn.Module):
        def __init__(self, core: nn.Module) -> None:
            super().__init__()
            self.encoder = core

    _StreamingAudioExportAdapter.forward = _build_streaming_export_forward(num_hidden_layers)
    return _StreamingAudioExportAdapter(encoder)


__all__ = [
    "StreamingAudioExportAdapter",
    "StreamingWhisperEncoderMixin",
    "build_streaming_attention_mask",
    "streaming_position_ids",
]
