"""MiniCPM extensions for the repository's text HMONNX runtime.

The exported MiniCPM decoder graphs use the normal prefill/decode/KV contract,
so session ownership, graph loading, device movement and KV allocation stay in
``TextLLMHMONNXModel`` / ``KVCacheMixin``.  The only non-standard part is that
the upstream remote code requires a fixed-capacity HF ``DynamicCache`` view of
those graph buffers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from ...hmonnx import TextLLMHMONNXModel
from ...types import LLMModelMeta
from .runtime_cache import FixedCapacityKVCacheMixin


DecoderOutput = Tensor | Sequence[Tensor]
TensorSession = Callable[..., DecoderOutput]


def _load_decoder_child_meta(
    root: Path,
    component: dict[str, Any],
) -> LLMModelMeta:
    """Load the required repository-standard decoder metadata.

    Release artifacts always contain a complete ``LLMModelMeta`` child for
    both LLM and TTS.  Reject incomplete artifacts here instead of rebuilding
    a second metadata contract from the root manifest.
    """
    metadata = component.get("metadata")
    if not metadata:
        raise ValueError("MiniCPM decoder component must provide standard child metadata")
    meta_path = root / str(metadata)
    if not meta_path.is_file():
        raise FileNotFoundError(f"MiniCPM decoder metadata does not exist: {meta_path}")
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    required = {"model_config", "kv_cache", "prefill_hmonnx", "decode_hmonnx"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(
            f"MiniCPM decoder metadata is not a complete LLMModelMeta artifact; missing: {', '.join(missing)}"
        )
    payload["_meta_path_"] = str(meta_path)
    return LLMModelMeta.from_dict(payload)


def _decoder_inputs(
    inputs_embeds: Tensor,
    past_seq_length: int,
    current_input_length: int,
    past_key_caches: Sequence[Tensor],
    past_value_caches: Sequence[Tensor],
    attention_mask: Tensor | None = None,
) -> list[Tensor]:
    inputs = [
        inputs_embeds,
        torch.tensor([past_seq_length], dtype=torch.int32, device=inputs_embeds.device),
        torch.tensor([current_input_length], dtype=torch.int32, device=inputs_embeds.device),
        *past_key_caches,
        *past_value_caches,
    ]
    if attention_mask is not None:
        inputs.append(attention_mask)
    return inputs


def _iter_decoder_graph_outputs(
    prefill_session: TensorSession,
    decode_session: TensorSession,
    inputs_embeds: Tensor,
    *,
    past_seq_length: int,
    current_input_length: int,
    past_key_caches: Sequence[Tensor],
    past_value_caches: Sequence[Tensor],
    prefill_length: int,
    attention_mask: Tensor | None = None,
    on_graph_call: Callable[[int], None] | None = None,
) -> list[tuple[DecoderOutput, int]]:
    """Traverse MiniCPM's fixed decoder graphs and return valid chunks."""
    if inputs_embeds.shape[1] <= 1:
        output = decode_session(
            *_decoder_inputs(
                inputs_embeds, past_seq_length, current_input_length, past_key_caches, past_value_caches, attention_mask
            )
        )
        if on_graph_call is not None:
            on_graph_call(past_seq_length + current_input_length)
        return [(output, current_input_length)]
    processed = 0
    outputs: list[tuple[DecoderOutput, int]] = []
    while processed < inputs_embeds.shape[1]:
        real_length = min(prefill_length, inputs_embeds.shape[1] - processed)
        chunk = inputs_embeds[:, processed : processed + real_length]
        chunk_mask = None if attention_mask is None else attention_mask[:, :, processed : processed + real_length]
        if real_length < prefill_length:
            chunk = torch.nn.functional.pad(chunk, (0, 0, 0, prefill_length - real_length))
            if chunk_mask is not None:
                padding = torch.full(
                    (*chunk_mask.shape[:2], prefill_length - real_length, chunk_mask.shape[3]),
                    torch.finfo(chunk_mask.dtype).min,
                    dtype=chunk_mask.dtype,
                    device=chunk_mask.device,
                )
                chunk_mask = torch.cat((chunk_mask, padding), dim=2)
        output = prefill_session(
            *_decoder_inputs(chunk, past_seq_length, real_length, past_key_caches, past_value_caches, chunk_mask)
        )
        processed += real_length
        past_seq_length += real_length
        if on_graph_call is not None:
            on_graph_call(past_seq_length)
        outputs.append((output, real_length))
    return outputs


class MiniCPMO45FixedCacheTextLLMHMONNXModel(TextLLMHMONNXModel):
    """Shared text-decoder lifecycle with MiniCPM's fixed HF-cache bridge.

    This is intentionally not a new session or KV implementation.  The parent
    owns ``HMONNXModel`` prefill/decode sessions and ``KVCacheMixin`` owns the
    backing ``CacheTensor`` lists.  This subclass replaces only the generic
    cache helper with the view adapter required by MiniCPM's published graph
    ABI.
    """

    cache_component = "decoder"

    def __init__(
        self,
        meta: LLMModelMeta,
        *,
        enable_cuda_graph: bool = False,
        enable_auto_offload: bool = False,
        enable_golden: bool = False,
        device_map: str | torch.device | list[str | torch.device] | None = None,
    ) -> None:
        super().__init__(
            meta,
            enable_cuda_graph=enable_cuda_graph,
            enable_auto_offload=enable_auto_offload,
            enable_golden=enable_golden,
            device_map=device_map,
        )
        self.prefill_length = int(meta.model_config.prefill_chunk_length)
        self.input_sequence_length = self.prefill_length
        # Keep the shared allocator but expose its fixed buffers through the
        # one HF cache object the official MiniCPM code retains between calls.
        self._kvcache_mixin = FixedCapacityKVCacheMixin(self.kvcache_config, self.cache_component)
        self._sync_page_attention_mode_to_kvcache()
        self._kvcache_mixin.prepare_fixed_cache(self.device)

    @property
    def hf_cache(self):
        return self._kvcache_mixin.hf_cache

    @property
    def max_sequence_length(self) -> int:
        return self._kvcache_mixin.cache_capacity

    def _set_device(self, device: torch.device):
        super()._set_device(device)
        self._kvcache_mixin.prepare_fixed_cache(device)
        return self

    def reset_kvcache(self) -> None:
        self._kvcache_mixin.reset_fixed_cache()

    def reset_state(self) -> None:
        self.reset_kvcache()

    def release_state(self) -> None:
        """Drop only MiniCPM cache state; the top-level runtime releases graphs."""
        self._kvcache_mixin.release_fixed_cache()

    def set_input_sequence_length(self, value: int) -> None:
        self.input_sequence_length = int(value)

    def get_input_sequence_length(self) -> int:
        return int(getattr(self, "input_sequence_length", self.prefill_length))

    def run_decoder_graph(
        self,
        inputs_embeds: Tensor,
        *,
        past_seq_length: int,
        current_input_length: int,
        past_key_caches: Sequence[Tensor] | None = None,
        past_value_caches: Sequence[Tensor] | None = None,
        attention_mask: Tensor | None = None,
        on_graph_call: Callable[[int], None] | None = None,
    ) -> list[tuple[DecoderOutput, int]]:
        """Run the shared prefill/decode traversal for this fixed graph ABI.

        ``attention_mask`` is deliberately an optional extension: ordinary
        text graphs use the base input ABI, while MiniCPM TTS has this extra
        final input.  Both still share the same graph sessions and cache lists.
        """
        return _iter_decoder_graph_outputs(
            self.prefill_model,
            self.decode_model,
            inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            past_key_caches=self.past_key_caches if past_key_caches is None else past_key_caches,
            past_value_caches=self.past_value_caches if past_value_caches is None else past_value_caches,
            prefill_length=self.prefill_length,
            attention_mask=attention_mask,
            on_graph_call=on_graph_call,
        )


__all__ = ["MiniCPMO45FixedCacheTextLLMHMONNXModel"]
