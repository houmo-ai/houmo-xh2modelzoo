from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4ForConditionalGeneration as XHGemma4ForConditionalGeneration,
)

from xhquant import nn as xhnn
from xhquant.utils.registry import _DMRegistryCls

from ...kv_cache_mixin import KVCacheMixin
from ...text_llm_hf_compatible import TextLLMHFCompatible
from ...types import KVCacheConfig
from .data_preprocess import Gemma4DataPreprocess


@dataclass(frozen=True)
class Gemma4PrefillChunk:
    """One fixed-shape Gemma4 Series prefill invocation."""

    start: int
    end: int
    graph_length: int
    graph_name: str


def plan_gemma4_atomic_prefill_chunks(
    seq_length: int,
    mm_token_type_ids: Optional[torch.Tensor],
    *,
    prefill_chunk_length: int,
) -> list[Gemma4PrefillChunk]:
    """Plan fixed-shape prefill chunks without splitting visual atomic ranges.

    Text tokens are splittable and are greedily packed into the same prefill
    graph as complete image/frame ranges.  Continuous ``mm_token_type_ids > 0``
    ranges are atomic because Gemma4 applies bidirectional visual overlay inside
    sliding-attention masks; one such range must be present in a single prefill
    invocation so future visual K/V is available to earlier visual queries.
    """

    seq_length = int(seq_length)
    prefill_chunk_length = int(prefill_chunk_length)
    if seq_length <= 0:
        return []
    if prefill_chunk_length <= 0:
        raise ValueError(
            "Gemma4 Series prefill_chunk_length must be positive; "
            f"got {prefill_chunk_length}."
        )

    if mm_token_type_ids is None:
        mm = torch.zeros(seq_length, dtype=torch.long)
    else:
        mm = mm_token_type_ids.detach().flatten().to(device="cpu")[:seq_length]
        if mm.numel() < seq_length:
            mm = torch.cat([mm, torch.zeros(seq_length - mm.numel(), dtype=mm.dtype)])

    ranges: list[tuple[int, int]] = []
    start: int | None = None
    for idx, token_type in enumerate(mm.tolist()):
        is_mm = int(token_type) > 0
        if is_mm and start is None:
            start = idx
        if start is not None and (idx == seq_length - 1 or int(mm[idx + 1].item()) <= 0):
            ranges.append((start, idx + 1))
            start = None

    segments: list[tuple[int, int, bool]] = []
    cursor = 0
    for range_start, range_end in ranges:
        if cursor < range_start:
            segments.append((cursor, range_start, False))
        segments.append((range_start, range_end, True))
        cursor = range_end
    if cursor < seq_length:
        segments.append((cursor, seq_length, False))

    chunks: list[Gemma4PrefillChunk] = []
    chunk_start: int | None = None
    chunk_end = 0

    def emit_chunk() -> None:
        nonlocal chunk_start, chunk_end
        if chunk_start is not None and chunk_end > chunk_start:
            chunks.append(Gemma4PrefillChunk(chunk_start, chunk_end, prefill_chunk_length, "prefill"))
        chunk_start = None
        chunk_end = 0

    def ensure_chunk(pos: int) -> None:
        nonlocal chunk_start, chunk_end
        if chunk_start is None:
            chunk_start = pos
            chunk_end = pos

    for seg_start, seg_end, is_atomic in segments:
        if is_atomic:
            seg_len = seg_end - seg_start
            if seg_len > prefill_chunk_length:
                raise ValueError(
                    "Gemma4 Series multimodal token range length exceeds prefill_chunk_length: "
                    f"range=[{seg_start}, {seg_end}), length={seg_len}, "
                    f"prefill_chunk_length={prefill_chunk_length}. Increase export.model.prefill_chunk_length."
                )
            ensure_chunk(seg_start)
            if chunk_end != seg_start:
                raise ValueError(
                    "Gemma4 Series atomic prefill planner encountered non-contiguous segments: "
                    f"chunk_end={chunk_end}, segment_start={seg_start}."
                )
            if (chunk_end - chunk_start) + seg_len > prefill_chunk_length:
                emit_chunk()
                ensure_chunk(seg_start)
            chunk_end = seg_end
            if chunk_end - chunk_start == prefill_chunk_length:
                emit_chunk()
            continue

        pos = seg_start
        while pos < seg_end:
            ensure_chunk(pos)
            if chunk_end != pos:
                raise ValueError(
                    "Gemma4 Series text prefill planner encountered non-contiguous segments: "
                    f"chunk_end={chunk_end}, text_pos={pos}."
                )
            capacity = prefill_chunk_length - (chunk_end - chunk_start)
            take = min(capacity, seg_end - pos)
            chunk_end += take
            pos += take
            if chunk_end - chunk_start == prefill_chunk_length:
                emit_chunk()

    emit_chunk()
    return chunks

def _gemma4_runtime_prefill_length(llm_model: Any) -> int:
    """Resolve the single Gemma4 prefill width from config/runtime surfaces.

    Float/export models expose ``config`` or ``wrap_cfg`` while HMONNX runtime
    models expose the value through ``meta_info.model_config``.  Generation must
    not depend on only one of those surfaces because the HF-compatible wrapper is
    shared by both paths.
    """

    candidates = (
        getattr(llm_model, "config", None),
        getattr(getattr(llm_model, "meta_info", None), "model_config", None),
        getattr(llm_model, "wrap_cfg", None),
    )

    def pick_positive(name: str, default: int | None = None) -> int | None:
        for candidate in candidates:
            value = getattr(candidate, name, None) if candidate is not None else None
            try:
                parsed = int(value)
            except (TypeError, ValueError):
                continue
            if parsed > 0:
                return parsed
        return default

    return pick_positive("prefill_chunk_length", 320) or 320


def _copy_model_shared_params(model: nn.Module) -> nn.Module:
    """Deep-copy model structure while sharing every tensor object with the original."""

    def _share_tensors(obj: Any, memo: dict[int, Any], seen: set[int]) -> None:
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(obj, nn.Parameter):
            memo.setdefault(obj_id, nn.Parameter(obj.data, requires_grad=obj.requires_grad))
            return
        if torch.is_tensor(obj):
            memo.setdefault(obj_id, obj)
            return
        if isinstance(obj, nn.Module):
            for value in obj.__dict__.values():
                _share_tensors(value, memo, seen)
            return
        if isinstance(obj, dict):
            for key, value in obj.items():
                _share_tensors(key, memo, seen)
                _share_tensors(value, memo, seen)
            return
        if isinstance(obj, (list, tuple, set)):
            for item in obj:
                _share_tensors(item, memo, seen)

    memo: dict[int, Any] = {}
    _share_tensors(model, memo, set())
    return copy.deepcopy(model, memo)


def _gemma4_cache_seq_len_for_layer(
    *,
    layer_type: str | None,
    context_max_length: int,
    sliding_window: int,
    input_seq_len: int,
    sliding_kv_cache_input_mode: str = "slice_window",
) -> int:
    """Return the static cache input length for a Gemma4 layer.

    ``legacy_full`` preserves the original full-context backing tensor used
    before compiler/runtime support for sliced sliding-layer KV inputs.
    ``slice_window`` exports sliding layers with only
    ``sliding_window + input_seq_len`` entries, aligned to 16.
    """

    if layer_type == "sliding_attention" and sliding_window > 0:
        mode = str(sliding_kv_cache_input_mode or "slice_window").lower()
        if mode == "slice_window":
            return Gemma4DataPreprocess._aligned(int(sliding_window) + int(input_seq_len), 16)
        if mode == "legacy_full":
            sliding_output_len = Gemma4DataPreprocess._aligned(sliding_window + input_seq_len - 1, 16)
            max_prefill_past = max(0, int(context_max_length) - int(input_seq_len))
            max_prefill_start = max(0, max_prefill_past - int(sliding_window) + 1)
            return max(int(context_max_length), max_prefill_start + sliding_output_len)
        raise ValueError(
            "Unsupported Gemma4 sliding_kv_cache_input_mode: "
            f"{sliding_kv_cache_input_mode!r}"
        )
    return int(context_max_length)


class Gemma4KVCacheMixin(KVCacheMixin):
    def __init__(self, kv_cache_config: KVCacheConfig):
        super().__init__(kv_cache_config)
        self.layer_kv_shapes: list[list[int]] = []

    def set_layer_kv_shapes(self, layer_kv_shapes: list[list[int]]):
        self.layer_kv_shapes = layer_kv_shapes
        self.kvcache_config.num_layers = len(layer_kv_shapes)
        if layer_kv_shapes:
            self.kvcache_config.kv_cache_shape = layer_kv_shapes[0]

    def prepare_kv_cache(self, dtype=torch.float16):
        if not self.use_cache:
            return
        self.past_key_caches.clear()
        self.past_value_caches.clear()
        for shape in self.layer_kv_shapes:
            self.past_key_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(shape, dtype=dtype)))
            self.past_value_caches.append(self.CACHCE_TENSOR_TYPE(torch.zeros(shape, dtype=dtype)))


class _Gemma4TextExportBridgeBase(nn.Module):
    def __init__(
        self,
        hf_model: XHGemma4ForConditionalGeneration,
        num_logits_to_keep: int | None = 0,
        *,
        language_model_keeps_last_logit: bool = False,
        language_model_returns_tensor: bool = False,
        enable_mtp_outputs: bool = False,
    ):
        super().__init__()
        self.config = hf_model.config
        self.language_model = hf_model.model.language_model
        self.lm_head = hf_model.lm_head
        self.num_logits_to_keep = int(num_logits_to_keep or 0)
        self.language_model_returns_tensor = bool(language_model_returns_tensor)
        self.language_model_keeps_last_logit = bool(language_model_keeps_last_logit)
        self.enable_mtp_outputs = bool(
            enable_mtp_outputs
            or getattr(self.language_model, "enable_mtp_outputs", False)
            or getattr(hf_model.config, "enable_mtp_outputs", False)
            or getattr(getattr(hf_model.config, "text_config", None), "enable_mtp_outputs", False)
        )
        # Real Gemma4 text graphs perform the valid-token gather inside
        # _Gemma4TextModel so FX tracing sees a fixed graph without Python
        # shape checks.  Unit-test dummy language models can still exercise
        # the bridge-local DynamicSlice path by leaving this flag false.
        self.valid_logits_slice = (
            xhnn.DynamicSlice([1], [1], [1])
            if self.num_logits_to_keep == 1 and not language_model_keeps_last_logit
            else None
        )

    def _update_cfg(self, cfg=None):
        if cfg is None:
            return self
        if hasattr(cfg, "get"):
            num_logits_to_keep = int(cfg.get("num_logits_to_keep", self.num_logits_to_keep) or 0)
            input_sequence_length = int(cfg.get("input_sequence_length", 1) or 1)
        else:
            num_logits_to_keep = int(getattr(cfg, "num_logits_to_keep", self.num_logits_to_keep) or 0)
            input_sequence_length = int(getattr(cfg, "input_sequence_length", 1) or 1)
        self.num_logits_to_keep = num_logits_to_keep
        self.valid_logits_slice = (
            xhnn.DynamicSlice([1], [1], [1])
            if self.num_logits_to_keep == 1 and not self.language_model_keeps_last_logit
            else None
        )
        if self.valid_logits_slice is not None:
            self.valid_logits_slice.valid_length = [1]
            if hasattr(self.valid_logits_slice, "_update_cfg"):
                self.valid_logits_slice._update_cfg(cfg)
        if hasattr(self.language_model, "_update_cfg"):
            self.language_model._update_cfg(cfg)
        elif hasattr(self.language_model, "llm_gather"):
            self.language_model.llm_gather.update_offset_indices(1, input_sequence_length)
        return self

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def _run(
        self,
        *,
        inputs_embeds,
        past_seq_length,
        current_input_length,
        full_attention_mask,
        sliding_attention_mask,
        past_key_cache,
        past_value_cache,
        per_layer_inputs=None,
        accepted_count=None,
    ):
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            full_attention_mask=full_attention_mask,
            sliding_attention_mask=sliding_attention_mask,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
            per_layer_inputs=per_layer_inputs,
            accepted_count=accepted_count,
        )
        if self.enable_mtp_outputs:
            hidden_states = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        elif self.language_model_returns_tensor:
            hidden_states = outputs
        else:
            hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs
        if self.valid_logits_slice is not None:
            hidden_states = self.valid_logits_slice(hidden_states, current_input_length - 1)
        logits = self.lm_head(hidden_states)
        final_logit_softcapping = getattr(self.config.text_config, "final_logit_softcapping", None)
        if final_logit_softcapping is not None:
            logits = logits / final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * final_logit_softcapping
        if self.enable_mtp_outputs:
            return logits, hidden_states
        return logits


class _Gemma4DecodeNoFullMaskBridge(_Gemma4TextExportBridgeBase):
    """Text graph adapter for Gemma4 variants whose full layers use MaskedSoftmax.

    Sliding layers still consume the explicit sliding_attention_mask.  Full
    layers receive None and therefore execute _Gemma4TextAttention's
    xhquant.nn.MaskedSoftmax causal branch.
    """

    def forward(
        self,
        inputs_embeds,
        past_seq_length,
        current_input_length,
        sliding_attention_mask,
        past_key_cache=None,
        past_value_cache=None,
    ):
        return self._run(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            full_attention_mask=None,
            sliding_attention_mask=sliding_attention_mask,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
        )


class _Gemma4DecodeNoFullMaskMTPBridge(_Gemma4TextExportBridgeBase):
    """MTP decode adapter with the extra accepted_count KV-cache input."""

    def forward(
        self,
        inputs_embeds,
        past_seq_length,
        current_input_length,
        sliding_attention_mask,
        accepted_count=None,
        past_key_cache=None,
        past_value_cache=None,
    ):
        return self._run(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            full_attention_mask=None,
            sliding_attention_mask=sliding_attention_mask,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
            accepted_count=accepted_count,
        )


class _Gemma4TextExportBridgePLE(_Gemma4TextExportBridgeBase):
    """Text-only export bridge for E4B PLE.

    ``per_layer_inputs`` contains scaled ``embed_tokens_per_layer`` lookup
    output.  E4B does not export a separate full-attention mask; its full
    layers use MaskedSoftmax while the explicit input remains the sliding mask.
    """

    def forward(
        self,
        inputs_embeds,
        past_seq_length,
        current_input_length,
        sliding_attention_mask,
        per_layer_inputs,
        past_key_cache=None,
        past_value_cache=None,
    ):
        return self._run(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            full_attention_mask=None,
            sliding_attention_mask=sliding_attention_mask,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
            per_layer_inputs=per_layer_inputs,
        )


class _Gemma4TextExportBridgePLEMTPDecode(_Gemma4TextExportBridgePLE):
    """PLE decode adapter with accepted_count as a target verify-only input."""

    def forward(
        self,
        inputs_embeds,
        past_seq_length,
        current_input_length,
        sliding_attention_mask,
        per_layer_inputs,
        accepted_count=None,
        past_key_cache=None,
        past_value_cache=None,
    ):
        return self._run(
            inputs_embeds=inputs_embeds,
            past_seq_length=past_seq_length,
            current_input_length=current_input_length,
            full_attention_mask=None,
            sliding_attention_mask=sliding_attention_mask,
            past_key_cache=past_key_cache,
            past_value_cache=past_value_cache,
            per_layer_inputs=per_layer_inputs,
            accepted_count=accepted_count,
        )


def _make_text_export_bridge_if_needed(
    hf_model: XHGemma4ForConditionalGeneration,
    num_logits_to_keep: int | None = 0,
    *,
    enable_mtp_outputs: bool = False,
):
    text_config = hf_model.config.get_text_config()
    if getattr(text_config, "hidden_size_per_layer_input", 0):
        return _Gemma4TextExportBridgePLE(
            hf_model,
            num_logits_to_keep=num_logits_to_keep,
            language_model_keeps_last_logit=(int(num_logits_to_keep or 0) == 1),
            language_model_returns_tensor=True,
            enable_mtp_outputs=enable_mtp_outputs,
        )
    return hf_model


class _Gemma4HFCompatible(TextLLMHFCompatible):
    """HF-compatible wrapper for Gemma4. Redirects forward() to our HMONNX/quanted model
    while keeping the HF generate() loop happy."""

    def _setup(self: XHGemma4ForConditionalGeneration, text_llm_model: "XHGemma4Model"):
        model = super()._setup(text_llm_model)
        if model is not None:
            for attr in ("model",):
                if hasattr(model, attr):
                    delattr(model, attr)
            if hasattr(model, "lm_head"):
                delattr(model, "lm_head")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._gemma4_pixel_values = None
        self._gemma4_pixel_position_ids = None
        self._gemma4_image_position_ids = None
        self._gemma4_pooling_matrix = None
        self._gemma4_visual_attention_mask = None
        self._gemma4_image_soft_token_count = None
        self._gemma4_pixel_values_videos = None
        self._gemma4_video_position_ids = None
        self._gemma4_video_pixel_position_ids = None
        self._gemma4_video_pooling_matrix = None
        self._gemma4_video_visual_attention_mask = None
        self._gemma4_video_soft_token_count = None
        self._gemma4_input_features = None
        self._gemma4_input_features_mask = None
        self._gemma4_audio_attention_mask = None
        self._gemma4_mm_token_type_ids = None
        return model

    @staticmethod
    def _flatten_features(features: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if features is None:
            return None
        if features.ndim == 2:
            return features
        return features.reshape(-1, features.shape[-1])

    @staticmethod
    def _split_features_by_chunk(
        input_ids: torch.Tensor,
        *,
        token_id: int,
        features: Optional[torch.Tensor],
        prefill_chunks: list[Gemma4PrefillChunk],
    ) -> list[Optional[torch.Tensor]]:
        if token_id < 0 or features is None:
            return [None] * len(prefill_chunks)
        flat_features = _Gemma4HFCompatible._flatten_features(features)
        assert flat_features is not None
        total_token_count = int((input_ids == token_id).sum().item())
        if total_token_count != flat_features.shape[0]:
            raise ValueError(
                f"Feature count does not match token count for token id {token_id}: "
                f"{flat_features.shape[0]} vs {total_token_count}"
            )
        feature_chunks: list[Optional[torch.Tensor]] = []
        cursor = 0
        for chunk in prefill_chunks:
            count = int((input_ids[:, chunk.start : chunk.end] == token_id).sum().item())
            feature_chunks.append(flat_features[cursor : cursor + count] if count else None)
            cursor += count
        return feature_chunks

    def _activate_prefill_graph(self, graph_name: str, graph_length: int) -> None:
        for attr_name in ("_quanted_model", "_frontend_model", "_inference_model"):
            switcher = getattr(self._llm_model, attr_name, None)
            if hasattr(switcher, "set_activate_model") and graph_name in switcher:
                switcher.set_activate_model(graph_name)
        set_input_sequence_length = getattr(self._llm_model, "set_input_sequence_length", None)
        if set_input_sequence_length is not None:
            set_input_sequence_length(int(graph_length))

    def _run_padded_visual(
        self,
        pixel_values: torch.Tensor,
        pixel_position_ids: torch.Tensor,
        pooling_matrix: torch.Tensor,
        visual_attention_mask: torch.Tensor,
        soft_token_count: Optional[torch.Tensor | list | tuple | int],
        visual_model: Any | None = None,
    ) -> torch.Tensor:
        visual = visual_model or self._llm_model.visual
        pixel_values = pixel_values.reshape(-1, pixel_values.shape[-2], pixel_values.shape[-1])
        pixel_position_ids = pixel_position_ids.reshape(-1, pixel_position_ids.shape[-2], pixel_position_ids.shape[-1])
        pooling_matrix = pooling_matrix.reshape(-1, pooling_matrix.shape[-2], pooling_matrix.shape[-1])
        visual_attention_mask = visual_attention_mask.reshape(
            -1,
            visual_attention_mask.shape[-3],
            visual_attention_mask.shape[-2],
            visual_attention_mask.shape[-1],
        )
        if soft_token_count is None:
            soft_counts = [None] * pixel_values.shape[0]
        elif torch.is_tensor(soft_token_count):
            soft_counts = [int(v.item()) for v in soft_token_count.flatten()]
        elif isinstance(soft_token_count, (list, tuple)):
            soft_counts = [int(v.item()) if torch.is_tensor(v) else int(v) for v in soft_token_count]
        else:
            soft_counts = [int(soft_token_count)]
        if len(soft_counts) == 1 and pixel_values.shape[0] > 1:
            soft_counts = soft_counts * pixel_values.shape[0]

        outputs = []
        for idx in range(pixel_values.shape[0]):
            embeds = visual.forward(
                pixel_values[idx : idx + 1].to(dtype=visual.dtype, device=visual.device),
                pixel_position_ids[idx : idx + 1].to(dtype=torch.int32, device=visual.device),
                pooling_matrix[idx : idx + 1].to(dtype=visual.dtype, device=visual.device),
                visual_attention_mask[idx : idx + 1].to(device=visual.device),
            )
            if isinstance(embeds, (tuple, list)):
                embeds = embeds[0]
            embeds = embeds.to(device=self._llm_model.device, dtype=self._llm_model.dtype)
            if embeds.dim() == 3 and embeds.shape[0] == 1:
                embeds = embeds[0]
            count = soft_counts[idx] if idx < len(soft_counts) else None
            if count is not None:
                embeds = embeds[:count]
            outputs.append(embeds)
        return torch.cat(outputs, dim=0) if outputs else torch.empty(0, device=self._llm_model.device)

    def _run_audio(
        self,
        input_features: torch.Tensor,
        input_features_mask: torch.Tensor | None,
        audio_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if getattr(self._llm_model, "audio", None) is None:
            raise ValueError("Gemma4 audio inputs were provided, but this preset has no public audio submodel.")
        audio = self._llm_model.audio
        audio_args = [
            input_features.to(dtype=audio.dtype, device=audio.device),
            input_features_mask.to(device=audio.device) if input_features_mask is not None else None,
        ]
        if audio_attention_mask is not None:
            audio_args.append(audio_attention_mask.to(device=audio.device))
        outputs = audio.forward(*audio_args)
        if isinstance(outputs, (tuple, list)):
            audio_embeds = outputs[0]
            audio_mask = outputs[1] if len(outputs) > 1 else None
        else:
            audio_embeds = outputs
            audio_mask = None
        audio_embeds = audio_embeds.to(device=self._llm_model.device, dtype=self._llm_model.dtype)
        if audio_mask is not None:
            audio_mask = audio_mask.to(device=audio_embeds.device).bool()
            return audio_embeds[audio_mask]
        return audio_embeds.reshape(-1, audio_embeds.shape[-1])

    def _run_llm_from_processed(self, data_input):
        processed = list(data_input)
        data_processor: Gemma4DataPreprocess = self._llm_model.get_data_preprocessor()
        if getattr(data_processor, "emit_accepted_count_input", False):
            accepted_count = processed[-3]
            past_key_caches = processed[-2]
            past_value_caches = processed[-1]
            model_args = processed[:-3] + [accepted_count] + list(past_key_caches) + list(past_value_caches)
        else:
            past_key_caches = processed[-2]
            past_value_caches = processed[-1]
            model_args = processed[:-2] + list(past_key_caches) + list(past_value_caches)
        logits = self._llm_model.forward(*model_args)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        return logits

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Retrieve vision args stored by generate()
        pixel_values = kwargs.pop("pixel_values", None)
        if pixel_values is None:
            pixel_values = self._gemma4_pixel_values
        pixel_position_ids = kwargs.pop("pixel_position_ids", None)
        if pixel_position_ids is None:
            pixel_position_ids = self._gemma4_pixel_position_ids
        image_position_ids = kwargs.pop("image_position_ids", None)
        if image_position_ids is None:
            image_position_ids = self._gemma4_image_position_ids
        pooling_matrix = kwargs.pop("pooling_matrix", None)
        if pooling_matrix is None:
            pooling_matrix = self._gemma4_pooling_matrix
        visual_attention_mask = kwargs.pop("visual_attention_mask", None)
        if visual_attention_mask is None:
            visual_attention_mask = self._gemma4_visual_attention_mask
        image_soft_token_count = kwargs.pop("image_soft_token_count", None)
        if image_soft_token_count is None:
            image_soft_token_count = kwargs.pop("num_soft_tokens_per_image", None)
        if image_soft_token_count is None:
            image_soft_token_count = self._gemma4_image_soft_token_count
        pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        if pixel_values_videos is None:
            pixel_values_videos = self._gemma4_pixel_values_videos
        video_position_ids = kwargs.pop("video_position_ids", None)
        if video_position_ids is None:
            video_position_ids = self._gemma4_video_position_ids
        video_pixel_position_ids = kwargs.pop("video_pixel_position_ids", None)
        if video_pixel_position_ids is None:
            video_pixel_position_ids = self._gemma4_video_pixel_position_ids
        video_pooling_matrix = kwargs.pop("video_pooling_matrix", None)
        if video_pooling_matrix is None:
            video_pooling_matrix = self._gemma4_video_pooling_matrix
        video_visual_attention_mask = kwargs.pop("video_visual_attention_mask", None)
        if video_visual_attention_mask is None:
            video_visual_attention_mask = self._gemma4_video_visual_attention_mask
        video_soft_token_count = kwargs.pop("video_soft_token_count", None)
        if video_soft_token_count is None:
            video_soft_token_count = self._gemma4_video_soft_token_count
        input_features = kwargs.pop("input_features", None)
        if input_features is None:
            input_features = self._gemma4_input_features
        input_features_mask = kwargs.pop("input_features_mask", None)
        if input_features_mask is None:
            input_features_mask = self._gemma4_input_features_mask
        audio_attention_mask = kwargs.pop("audio_attention_mask", None)
        if audio_attention_mask is None:
            audio_attention_mask = self._gemma4_audio_attention_mask
        mm_token_type_ids = kwargs.pop("mm_token_type_ids", None)
        if mm_token_type_ids is None:
            mm_token_type_ids = self._gemma4_mm_token_type_ids

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        # Vision: run visual model on first call (prefill) only
        image_embeds = None
        video_embeds = None
        audio_embeds = None
        if pixel_values is not None:
            if pixel_position_ids is None and image_position_ids is not None:
                if image_position_ids.dim() == 2:
                    image_position_ids = image_position_ids.unsqueeze(0)
                pixel_position_ids = image_position_ids.clone()
                pixel_position_ids[(pixel_position_ids == -1).all(dim=-1)] = 0
            if pixel_position_ids is None or pooling_matrix is None or visual_attention_mask is None:
                raise ValueError(
                    "Gemma4 padded visual inference requires pixel_position_ids, "
                    "pooling_matrix and visual_attention_mask with pixel_values."
                )
            image_embeds = self._run_padded_visual(
                pixel_values,
                pixel_position_ids,
                pooling_matrix,
                visual_attention_mask,
                image_soft_token_count,
            )
            # Offload visual model to free GPU memory for text prefill
            self._llm_model.visual.to("cpu")
            torch.cuda.empty_cache()
            # Clear so vision is not re-run on decode steps
            self._gemma4_pixel_values = None
            self._gemma4_pixel_position_ids = None
            self._gemma4_image_position_ids = None
            self._gemma4_pooling_matrix = None
            self._gemma4_visual_attention_mask = None
            self._gemma4_image_soft_token_count = None

        if pixel_values_videos is not None:
            if video_pixel_position_ids is None and video_position_ids is not None:
                video_pixel_position_ids = video_position_ids.clone()
                video_pixel_position_ids[(video_pixel_position_ids == -1).all(dim=-1)] = 0
            if video_pixel_position_ids is None or video_pooling_matrix is None or video_visual_attention_mask is None:
                raise ValueError(
                    "Gemma4 padded video inference requires video_pixel_position_ids, "
                    "video_pooling_matrix and video_visual_attention_mask with pixel_values_videos."
                )
            video_embeds = self._run_padded_visual(
                pixel_values_videos,
                video_pixel_position_ids,
                video_pooling_matrix,
                video_visual_attention_mask,
                video_soft_token_count,
                getattr(self._llm_model, "video_visual", None) or self._llm_model.visual,
            )
            (getattr(self._llm_model, "video_visual", None) or self._llm_model.visual).to("cpu")
            torch.cuda.empty_cache()
            self._gemma4_pixel_values_videos = None
            self._gemma4_video_position_ids = None
            self._gemma4_video_pixel_position_ids = None
            self._gemma4_video_pooling_matrix = None
            self._gemma4_video_visual_attention_mask = None
            self._gemma4_video_soft_token_count = None

        if input_features is not None:
            audio_embeds = self._run_audio(input_features, input_features_mask, audio_attention_mask)
            if getattr(self._llm_model, "audio", None) is not None:
                self._llm_model.audio.to("cpu")
            torch.cuda.empty_cache()
            self._gemma4_input_features = None
            self._gemma4_input_features_mask = None
            self._gemma4_audio_attention_mask = None

        seq_length = inputs_embeds.shape[1]
        device = inputs_embeds.device
        prefill_len = _gemma4_runtime_prefill_length(self._llm_model)
        if mm_token_type_ids is None:
            mm_full = torch.zeros(seq_length, dtype=torch.long, device=device)
        else:
            mm_full = mm_token_type_ids.to(device).flatten()[:seq_length]

        chunks = plan_gemma4_atomic_prefill_chunks(
            seq_length,
            mm_full,
            prefill_chunk_length=prefill_len,
        )
        data_processor0: Gemma4DataPreprocess = self._llm_model.get_data_preprocessor()
        image_chunks = self._split_features_by_chunk(
            input_ids, token_id=data_processor0.image_token_id, features=image_embeds, prefill_chunks=chunks
        )
        video_chunks = self._split_features_by_chunk(
            input_ids, token_id=data_processor0.video_token_id, features=video_embeds, prefill_chunks=chunks
        )
        audio_chunks = self._split_features_by_chunk(
            input_ids, token_id=data_processor0.audio_token_id, features=audio_embeds, prefill_chunks=chunks
        )

        running_past_seq = self._past_seq_length
        outputs_logits = []
        try:
            for idx, chunk in enumerate(chunks):
                self._activate_prefill_graph(chunk.graph_name, chunk.graph_length)
                data_processor: Gemma4DataPreprocess = self._llm_model.get_data_preprocessor()
                sub_current_len = chunk.end - chunk.start
                data_batch = {
                    "input_ids": input_ids[:, chunk.start : chunk.end],
                    "image_embeds": image_chunks[idx],
                    "video_embeds": video_chunks[idx],
                    "audio_embeds": audio_chunks[idx],
                    "past_seq_length": running_past_seq,
                    "mm_token_type_ids": mm_full[chunk.start : chunk.end].unsqueeze(0),
                }
                chunk_logits = self._run_llm_from_processed(data_processor(data_batch))
                outputs_logits.append(chunk_logits)
                running_past_seq += sub_current_len
        finally:
            self._activate_prefill_graph("prefill", prefill_len)

        last_valid = chunks[-1].end - chunks[-1].start
        logits = outputs_logits[-1][:, :last_valid, :]

        return CausalLMOutputWithPast(
            logits=logits,
            past_key_values=past_key_values,
        )

    def generate(self, *args, **kwargs):
        # Extract Gemma4-specific kwargs before HF's generate validates them
        self._gemma4_pixel_values = kwargs.pop("pixel_values", None)
        self._gemma4_pixel_position_ids = kwargs.pop("pixel_position_ids", None)
        self._gemma4_image_position_ids = kwargs.pop("image_position_ids", None)
        self._gemma4_pooling_matrix = kwargs.pop("pooling_matrix", None)
        self._gemma4_visual_attention_mask = kwargs.pop("visual_attention_mask", None)
        self._gemma4_image_soft_token_count = kwargs.pop("image_soft_token_count", None)
        if self._gemma4_image_soft_token_count is None:
            self._gemma4_image_soft_token_count = kwargs.pop("num_soft_tokens_per_image", None)
        self._gemma4_pixel_values_videos = kwargs.pop("pixel_values_videos", None)
        self._gemma4_video_position_ids = kwargs.pop("video_position_ids", None)
        self._gemma4_video_pixel_position_ids = kwargs.pop("video_pixel_position_ids", None)
        self._gemma4_video_pooling_matrix = kwargs.pop("video_pooling_matrix", None)
        self._gemma4_video_visual_attention_mask = kwargs.pop("video_visual_attention_mask", None)
        self._gemma4_video_soft_token_count = kwargs.pop("video_soft_token_count", None)
        self._gemma4_input_features = kwargs.pop("input_features", None)
        self._gemma4_input_features_mask = kwargs.pop("input_features_mask", None)
        self._gemma4_audio_attention_mask = kwargs.pop("audio_attention_mask", None)
        self._gemma4_mm_token_type_ids = kwargs.pop("mm_token_type_ids", None)
        return super().generate(*args, **kwargs)

    def set_experts_implementation(self, *args, **kwargs):
        """No-op: Gemma4 has no MoE experts; prevents HF generate crash."""


def build_gemma4_hf_compatible_model(
    hf_model: XHGemma4ForConditionalGeneration,
    xh_model: "XHGemma4Model",
):
    llm_compatible_modules = _DMRegistryCls("XHCompatible")
    hf_model_cls = type(hf_model)
    if hf_model_cls not in llm_compatible_modules:
        llm_compatible_modules.register_module({hf_model_cls: hf_model_cls.__name__}, _Gemma4HFCompatible)
    return llm_compatible_modules.convert(hf_model, text_llm_model=xh_model)
