from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers.modeling_outputs import CausalLMOutputWithPast

from ...types import LLMModelMeta
from .runtime_text_decoder import MiniCPMO45FixedCacheTextLLMHMONNXModel, _load_decoder_child_meta


def _cache_seq_length(past_key_values: Any) -> int:
    """Best-effort sequence length of an external HF Cache object."""
    if hasattr(past_key_values, "get_seq_length"):
        try:
            return int(past_key_values.get_seq_length())
        except Exception:
            return 0
    return 0


def _collect_llm_decoder_outputs(
    outputs: Sequence[tuple[Tensor | Sequence[Tensor], int]], input_length: int
) -> Tensor | Sequence[Tensor]:
    """Apply MiniCPM's logits-plus-hidden output contract to shared chunks."""
    hidden_parts: list[Tensor] = []
    logits_parts: list[Tensor] = []
    last_logits: Tensor | None = None
    for output, real_length in outputs:
        if input_length <= 1:
            return output
        if isinstance(output, Tensor):
            last_logits = output
            hidden = output
        else:
            last_logits = output[0]
            hidden = output[1] if len(output) > 1 else output[0]
        if last_logits.shape[1] >= real_length:
            logits_parts.append(last_logits[:, :real_length])
        hidden_parts.append(hidden[:, :real_length])
    if last_logits is None:
        raise RuntimeError("LLM Prefill produced no output")
    logits = torch.cat(logits_parts, dim=1) if len(logits_parts) == len(hidden_parts) else last_logits
    return logits, torch.cat(hidden_parts, dim=1)


def _resolve_llm_meta(root: Path, component: dict[str, Any]) -> LLMModelMeta:
    """Load the standard LLM child metadata emitted by the exporter."""
    return _load_decoder_child_meta(root, component)


class MiniCPMO45LLMHMONNXRuntime(MiniCPMO45FixedCacheTextLLMHMONNXModel):
    """MiniCPM's thin shared-LLM runtime plus its required HF-cache bridge.

    QwenVL uses this same ``BaseLLMHMONNXModel`` session and KV ownership
    path.  MiniCPM keeps only two model-specific pieces: its graph emits both
    logits and hidden states for TTS, and official remote code requires a
    fixed-capacity ``DynamicCache`` view over the graph buffers.
    """

    def __init__(
        self,
        root: Path,
        meta: dict[str, Any],
        *,
        enable_cuda_graph: bool = False,
        enable_auto_offload: bool = False,
        enable_golden: bool = False,
        device_map: str | torch.device | list[str | torch.device] | None = None,
    ) -> None:
        llm_meta = _resolve_llm_meta(root, meta)
        super().__init__(
            llm_meta,
            enable_cuda_graph=enable_cuda_graph,
            enable_auto_offload=enable_auto_offload,
            enable_golden=enable_golden,
            device_map=device_map,
        )

    @staticmethod
    def _build_embed_tokens_from_meta(meta: LLMModelMeta) -> torch.nn.Embedding:
        """Load MiniCPM's exported embedding without importing remote code.

        The shared implementation asks ``AutoConfig(..., trust_remote_code=True)``
        for vocabulary dimensions.  MiniCPM's portable export intentionally
        copies configuration files only, not remote Python source; the exported
        embedding state dict already contains both dimensions.  Reading those
        dimensions directly preserves the shared runtime while avoiding a
        hidden dependency on the original HF checkout.
        """
        try:
            state = torch.load(meta.quant_embedding, map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(meta.quant_embedding, map_location="cpu", weights_only=False)
        if isinstance(state, torch.nn.Embedding):
            return state
        weight = state["weight"]
        embedding = torch.nn.Embedding(weight.shape[0], weight.shape[1], dtype=weight.dtype)
        embedding.load_state_dict(state)
        return embedding

    def set_num_logits_to_keep(self, value: int) -> None:
        # The exported graph has a fixed output ABI.  HF calls this setter only
        # to request decode behaviour; output selection remains in the wrapper.
        # Preserve all values for HF features such as speculative decoding,
        # even though this graph cannot change its output arity.
        self._num_logits_to_keep = int(value)

    def forward(
        self,
        inputs_embeds: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        past_key_caches: Sequence[Tensor],
        past_value_caches: Sequence[Tensor],
    ) -> Tensor | Sequence[Tensor]:
        outputs = self.run_decoder_graph(
            inputs_embeds,
            past_seq_length=int(past_seq_length.reshape(-1)[0]),
            current_input_length=int(current_input_length.reshape(-1)[0]),
            past_key_caches=past_key_caches,
            past_value_caches=past_value_caches,
        )
        return _collect_llm_decoder_outputs(outputs, inputs_embeds.shape[1])

    def forward_hf(
        self,
        *,
        inputs_embeds: Tensor | None = None,
        past_key_values: Any = None,
        use_cache: bool = True,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        del kwargs
        if inputs_embeds is None:
            raise ValueError("HMONNX LLM forward requires inputs_embeds")
        if not return_dict:
            raise ValueError("HMONNX LLM forward requires return_dict=True")
        # transformers generate() creates its own empty Cache before the first
        # forward and feeds it back afterwards. The HMONNX state always lives in
        # the runtime-owned hf_cache, so an external cache is accepted only while
        # it is empty (fresh generation); a non-empty foreign cache is rejected.
        if past_key_values is not None and past_key_values is not self.hf_cache:
            external_length = _cache_seq_length(past_key_values)
            if external_length > 0:
                raise ValueError("past_key_values must be the runtime-owned hf_cache")

        cache = self.hf_cache if use_cache else None
        if cache is not None and past_key_values is None and cache.get_seq_length() > 0:
            self.reset_kvcache()
        past_length = cache.get_seq_length() if cache is not None else 0
        current_length = int(inputs_embeds.shape[1])
        if past_length + current_length > self.max_sequence_length:
            raise RuntimeError(
                "llm cache capacity exceeded: "
                f"current={past_length}, requested={current_length}, capacity={self.max_sequence_length}"
            )

        def commit(valid_length: int) -> None:
            if cache is not None:
                cache.commit_length(valid_length)

        output = _collect_llm_decoder_outputs(
            self.run_decoder_graph(
                inputs_embeds,
                past_seq_length=past_length,
                current_input_length=current_length,
                on_graph_call=commit,
            ),
            current_length,
        )
        if isinstance(output, Tensor):
            logits = output
            hidden = None
        else:
            logits = output[0]
            hidden = output[1] if len(output) > 1 else None
        hidden_states = (hidden,) if output_hidden_states and hidden is not None else None
        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=cache,
            hidden_states=hidden_states,
        )


__all__ = ["MiniCPMO45LLMHMONNXRuntime", "_resolve_llm_meta"]
