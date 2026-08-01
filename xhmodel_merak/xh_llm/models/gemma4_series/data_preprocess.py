from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch
import torch.nn as nn

from ...llm_data_processor import BaseInputProcessorConfig, BaseLLMInputProcessor
from .attention_visibility import (
    Gemma4AttentionVisibilitySpec,
    resolve_gemma4_attention_visibility_spec,
)


IMAGE_MM_TOKEN_TYPE_ID = 1
VIDEO_MM_TOKEN_TYPE_ID = 2


def _is_vision_token_type(mm_token_type_ids: torch.Tensor) -> torch.Tensor:
    """Return only image/video tokens; audio remains strictly causal."""

    return (mm_token_type_ids == IMAGE_MM_TOKEN_TYPE_ID) | (
        mm_token_type_ids == VIDEO_MM_TOKEN_TYPE_ID
    )


class Gemma4PerLayerInputEmbedding(nn.Module):
    """Host-side raw PLE embedding lookup for Gemma4 E4B.

    This intentionally owns only ``embed_tokens_per_layer``.  Projection,
    RMSNorm, add and scaling stay inside the main text ONNX via
    ``Gemma4TextModel.project_per_layer_inputs`` so the NPU graph keeps the
    non-embedding part of the per-layer embedding flow.
    """

    def __init__(
        self,
        *,
        vocab_size_per_layer_input: int,
        num_hidden_layers: int,
        hidden_size_per_layer_input: int,
        pad_token_id: int,
        embedding_scale: float = 1.0,
    ):
        super().__init__()
        self.vocab_size_per_layer_input = int(vocab_size_per_layer_input)
        self.num_hidden_layers = int(num_hidden_layers)
        self.hidden_size_per_layer_input = int(hidden_size_per_layer_input)
        self.pad_token_id = int(pad_token_id)
        self.embedding_scale = float(embedding_scale)
        self.embed_tokens_per_layer = nn.Embedding(
            self.vocab_size_per_layer_input,
            self.num_hidden_layers * self.hidden_size_per_layer_input,
            self.pad_token_id,
        )

    @classmethod
    def from_language_model(cls, language_model) -> "Gemma4PerLayerInputEmbedding":
        config = language_model.config
        embedding = cls(
            vocab_size_per_layer_input=config.vocab_size_per_layer_input,
            num_hidden_layers=config.num_hidden_layers,
            hidden_size_per_layer_input=config.hidden_size_per_layer_input,
            pad_token_id=config.pad_token_id,
            embedding_scale=(config.hidden_size_per_layer_input**0.5) * (2.0**-0.5),
        )
        embedding.embed_tokens_per_layer.load_state_dict(language_model.embed_tokens_per_layer.state_dict())
        embedding.embed_tokens_per_layer.weight.data.mul_(embedding.embedding_scale)
        embedding.to(dtype=language_model.embed_tokens_per_layer.weight.dtype)
        return embedding.eval()

    @classmethod
    def from_artifact(cls, artifact_path: str | Path) -> "Gemma4PerLayerInputEmbedding":
        try:
            saved = torch.load(str(artifact_path), map_location="cpu", weights_only=True)
        except Exception:
            saved = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
        config = saved["config"]
        embedding = cls(**config)
        embedding.load_state_dict(saved["state_dict"])
        sample_tensor = next(iter(saved["state_dict"].values()))
        embedding.to(dtype=sample_tensor.dtype)
        return embedding.eval()

    def save_artifact(self, artifact_path: str | Path):
        payload = {
            "config": {
                "vocab_size_per_layer_input": self.vocab_size_per_layer_input,
                "num_hidden_layers": self.num_hidden_layers,
                "hidden_size_per_layer_input": self.hidden_size_per_layer_input,
                "pad_token_id": self.pad_token_id,
                "embedding_scale": self.embedding_scale,
            },
            "state_dict": {name: tensor.detach().cpu() for name, tensor in self.state_dict().items()},
        }
        torch.save(payload, str(artifact_path))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_length = input_ids.shape
        per_layer_inputs = self.embed_tokens_per_layer(input_ids.to(torch.long))
        return per_layer_inputs.reshape(
            batch_size,
            seq_length,
            self.num_hidden_layers,
            self.hidden_size_per_layer_input,
        )


class Gemma4DataPreprocess(BaseLLMInputProcessor):
    def __init__(
        self,
        *,
        token_embedding: nn.Embedding,
        input_sequence_length: int,
        context_length: int,
        past_key_caches=None,
        past_value_caches=None,
        pad_token_id: int = 0,
        image_token_id: int = -1,
        audio_token_id: int = -1,
        video_token_id: int = -1,
        per_layer_input_embedding: Gemma4PerLayerInputEmbedding | None = None,
        sliding_window: int = 1024,
        bidirectional_vision_attention: bool = False,
        attention_contract_version: int = 1,
        attention_lowering: str | None = None,
        max_mm_ranges_per_chunk: int = 1,
        attention_visibility_spec: Gemma4AttentionVisibilitySpec | Mapping | None = None,
        layer_types: Sequence[str] | None = None,
        emit_full_attention_mask: bool | None = None,
        emit_accepted_count_input: bool = False,
    ):
        super().__init__(
            BaseInputProcessorConfig(
                embed_tokens=token_embedding,
                input_sequence_length=input_sequence_length,
                past_key_caches=past_key_caches,
                past_value_caches=past_value_caches,
                pad_token_id=pad_token_id,
            )
        )
        self.token_embedding = token_embedding
        self.context_length = context_length
        self.image_token_id = image_token_id
        self.audio_token_id = audio_token_id
        self.video_token_id = video_token_id
        self.per_layer_input_embedding = per_layer_input_embedding
        self.attention_contract_version = int(attention_contract_version)
        if self.attention_contract_version not in (1, 2):
            raise ValueError("Gemma4 attention_contract_version must be 1 or 2")
        self.attention_lowering = attention_lowering or (
            "flash_attention" if self.attention_contract_version >= 2 else "legacy_attention"
        )
        if self.attention_lowering not in {
            "legacy_attention",
            "flash_attention",
            "full_flash_attention",
        }:
            raise ValueError(f"Unsupported Gemma4 attention_lowering: {self.attention_lowering!r}")
        self.attention_visibility_spec = resolve_gemma4_attention_visibility_spec(
            attention_visibility_spec,
            layer_types=layer_types or ("sliding_attention", "full_attention"),
            sliding_window=sliding_window,
            bidirectional_vision_attention=bidirectional_vision_attention,
            max_mm_ranges_per_chunk=max_mm_ranges_per_chunk,
        )
        # Compatibility fields remain available to old runtime code, but all
        # three are now projections of the version-independent semantic spec.
        self.sliding_window = self.attention_visibility_spec.sliding_window
        self.bidirectional_vision_attention = self.attention_visibility_spec.bidirectional_vision_attention
        self.max_mm_ranges_per_chunk = self.attention_visibility_spec.max_mm_ranges_per_chunk
        if emit_full_attention_mask:
            raise ValueError(
                "Gemma4 Series no longer emits full_attention_mask; full-attention layers use "
                "xhquant.nn.MaskedSoftmax with attention_mask=None."
            )
        self.emit_full_attention_mask = False
        self.emit_accepted_count_input = bool(emit_accepted_count_input)

    def to(self, *args, **kwargs) -> "Gemma4DataPreprocess":
        super().to(*args, **kwargs)
        if self.per_layer_input_embedding is not None:
            self.per_layer_input_embedding.to(device=self._device, dtype=self._dtype)
        return self

    @staticmethod
    def _aligned(size: int, alignment: int = 16) -> int:
        return ((size + alignment - 1) // alignment) * alignment

    def _pad_1d(self, tensor: torch.Tensor, value: int = 0):
        if tensor.shape[-1] >= self.input_sequence_length:
            return tensor[..., : self.input_sequence_length]
        pad = torch.full(
            (*tensor.shape[:-1], self.input_sequence_length - tensor.shape[-1]),
            value,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, pad], dim=-1)

    def _build_llm_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        llm_input_ids = input_ids.clone()
        for token_id in (self.image_token_id, self.audio_token_id, self.video_token_id):
            if token_id is not None and token_id >= 0:
                llm_input_ids[input_ids == token_id] = self.pad_token_id
        return llm_input_ids

    @staticmethod
    def _flatten_features(features: torch.Tensor) -> torch.Tensor:
        if features.ndim == 2:
            return features
        return features.reshape(-1, features.shape[-1])

    def _scatter_features(
        self,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        *,
        token_id: int,
        features: torch.Tensor | None,
        feature_name: str,
    ) -> torch.Tensor:
        if token_id is None or token_id < 0 or features is None:
            return inputs_embeds

        flat_features = self._flatten_features(features).to(inputs_embeds.device, inputs_embeds.dtype)
        token_positions = (input_ids == token_id).nonzero(as_tuple=False)
        if token_positions.numel() == 0:
            if flat_features.shape[0] != 0:
                raise ValueError(
                    f"Received {feature_name} features for token id {token_id}, but prompt does not contain that token."
                )
            return inputs_embeds
        if token_positions.shape[0] != flat_features.shape[0]:
            raise ValueError(
                f"{feature_name} features and token count do not match: "
                f"tokens={token_positions.shape[0]}, features={flat_features.shape[0]}"
            )

        updated = inputs_embeds.clone()
        updated[token_positions[:, 0], token_positions[:, 1]] = flat_features
        return updated

    def _build_attention_masks(
        self,
        current_input_length: int,
        past_seq_length: int,
        mm_token_type_ids: torch.Tensor,
        device: torch.device,
    ):
        q_len = self.input_sequence_length
        neg = torch.tensor(torch.finfo(torch.float16).min, dtype=torch.float16, device=device)
        full_ctx = self.context_length
        # A full-only legacy graph keeps the historical sliding-mask ABI even
        # though no layer consumes it.  Use full context only to construct that
        # inert compatibility input; it is not part of the visibility spec.
        sw = self.sliding_window or full_ctx

        # ── Full attention mask (width = context_length) ──
        full_mask = torch.full((1, 1, q_len, full_ctx), neg, dtype=torch.float16, device=device)
        valid_k = min(full_ctx, max(1, past_seq_length + current_input_length))
        for q in range(q_len):
            if q < current_input_length:
                full_end = min(valid_k, past_seq_length + q + 1)
                full_mask[0, 0, q, :full_end] = 0
            else:
                full_mask[0, 0, q, 0] = 0

        # ── Sliding attention mask (width = LLMCache output size) ──
        # LLMCache output width is aligned(sliding_window + q_len - 1, 16).
        # MTP decode still receives the larger physical slice-window cache
        # tensor (sliding_window + prefill_input_length), but the attention
        # consumes LLMCache's compact output, e.g. 1024 + verify_len(5) - 1
        # -> aligned 1040.  The additive mask follows those compact output
        # coordinates.
        slide_ctx = self._aligned(sw + q_len - 1, 16)
        sliding_mask = torch.full((1, 1, q_len, slide_ctx), neg, dtype=torch.float16, device=device)
        local_valid = min(max(int(past_seq_length), 0), int(slide_ctx))
        concat_pos = min(local_valid, max(0, int(sw) - 1))
        for q in range(q_len):
            if q < current_input_length:
                cache_pos = concat_pos + q
                causal_end = min(slide_ctx, cache_pos + 1)
                sw_start = max(0, cache_pos - sw + 1) if sw > 0 else 0
                sliding_mask[0, 0, q, sw_start:causal_end] = 0
            else:
                sliding_mask[0, 0, q, 0] = 0

        # ── Vision token bidirectional attention (Gemma4 31B/26B only).
        # E4B's text_config does not opt into HF's vision-bidirectional mask;
        # keep it purely causal even when visual/audio/video soft tokens exist.
        # The bidirectional overlay belongs only to sliding attention.  Full
        # attention layers receive attention_mask=None at export/runtime and use
        # xhquant.nn.MaskedSoftmax's causal path instead of this helper mask.
        if self.bidirectional_vision_attention and mm_token_type_ids.numel() > 0:
            mm = mm_token_type_ids[:current_input_length]
            is_vision = _is_vision_token_type(mm)
            # Offset to convert absolute positions to sliding-cache coordinates.
            # The first retained absolute token maps to local coordinate 0.
            cache_offset = max(0, past_seq_length - concat_pos)
            group_start = None
            for idx in range(current_input_length):
                if bool(is_vision[idx]) and group_start is None:
                    group_start = idx
                if group_start is not None and (idx == current_input_length - 1 or not bool(is_vision[idx + 1])):
                    group_end = idx + 1
                    abs_start = past_seq_length + group_start
                    abs_end = past_seq_length + group_end
                    # Sliding mask: convert to cache-relative coordinates
                    c_start = max(0, abs_start - cache_offset)
                    c_end = min(slide_ctx, abs_end - cache_offset)
                    if c_start < slide_ctx and c_end > 0:
                        sliding_mask[0, 0, group_start:group_end, c_start:c_end] = 0
                    group_start = None

        return full_mask, sliding_mask

    def _build_compact_attention_metadata(
        self,
        current_input_length: int,
        past_seq_length: int,
        mm_token_type_ids: torch.Tensor,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        p = max(0, int(past_seq_length))
        c = max(0, int(current_input_length))
        q = int(self.input_sequence_length)
        if not 0 <= c <= q:
            raise ValueError(f"current_input_length must be in [0, {q}], got {c}")
        if self.attention_visibility_spec.has_sliding_attention:
            if type(self.sliding_window) is not int or self.sliding_window <= 0:
                raise ValueError(f"sliding_window must be positive, got {self.sliding_window}")
            if self.sliding_window != self.attention_visibility_spec.sliding_window:
                raise ValueError(
                    "sliding_window compatibility field must match attention_visibility_spec: "
                    f"{self.sliding_window} != {self.attention_visibility_spec.sliding_window}"
                )
            w = self.sliding_window
            retained = min(p, w - 1)
            start = p - retained
            width = self._aligned(w + q - 1, 16)
            valid = min(width, retained + c)
        else:
            # Full-only attention retains absolute key coordinates.  Flash's
            # fixed metadata slots remain populated for ABI compatibility even
            # though no sliding layer consumes a compact KV base.
            start = 0
            valid = min(int(self.context_length), p + c)

        ranges: list[tuple[int, int]] = []
        if self.attention_visibility_spec.requires_mm_prefix_ranges:
            mm = _is_vision_token_type(mm_token_type_ids[:c])
            range_start = None
            for idx in range(c):
                if bool(mm[idx]) and range_start is None:
                    range_start = idx
                if range_start is not None and (idx == c - 1 or not bool(mm[idx + 1])):
                    ranges.append((p + range_start, p + idx))
                    range_start = None

        if len(ranges) > self.max_mm_ranges_per_chunk:
            raise ValueError(
                f"Gemma4 current chunk has {len(ranges)} multimodal ranges, "
                f"exceeding max_mm_ranges_per_chunk={self.max_mm_ranges_per_chunk}"
            )
        padded_ranges = ranges + [(0, 0)] * (self.max_mm_ranges_per_chunk - len(ranges))
        return (
            torch.tensor([start], dtype=torch.int64, device=device),
            torch.tensor([valid], dtype=torch.int64, device=device),
            torch.tensor([padded_ranges], dtype=torch.int32, device=device),
        )

    def forward(self, data: dict | tuple | list):
        assert isinstance(data, dict)
        device = self._device
        input_ids = data["input_ids"].to(device)
        assert input_ids.shape[0] == 1, "Batch size should be 1 in inference mode."
        seq_length = input_ids.shape[1]
        assert seq_length <= self.input_sequence_length, (
            f"Input sequence length is too long. "
            f"max input sequence length is {self.input_sequence_length} but got {seq_length}"
        )
        input_ids = self._pad_1d(input_ids, value=self.pad_token_id)
        llm_input_ids = self._build_llm_input_ids(input_ids)
        inputs_embeds = self.token_embedding.to(device)(llm_input_ids)

        inputs_embeds = self._scatter_features(
            inputs_embeds,
            input_ids,
            token_id=self.image_token_id,
            features=data.get("image_embeds", None),
            feature_name="Image",
        )
        inputs_embeds = self._scatter_features(
            inputs_embeds,
            input_ids,
            token_id=self.video_token_id,
            features=data.get("video_embeds", None),
            feature_name="Video",
        )
        inputs_embeds = self._scatter_features(
            inputs_embeds,
            input_ids,
            token_id=self.audio_token_id,
            features=data.get("audio_embeds", None),
            feature_name="Audio",
        )

        per_layer_inputs = None
        if self.per_layer_input_embedding is not None:
            self.per_layer_input_embedding.to(device=device)
            per_layer_inputs = self.per_layer_input_embedding(llm_input_ids.to(torch.long)).to(
                inputs_embeds.device, inputs_embeds.dtype
            )

        mm_token_type_ids = data.get("mm_token_type_ids", None)
        if mm_token_type_ids is None:
            mm_token_type_ids = torch.zeros((1, seq_length), dtype=torch.long, device=device)
        else:
            mm_token_type_ids = mm_token_type_ids.to(device)
        mm_token_type_ids = self._pad_1d(mm_token_type_ids, value=0)[0]

        past_seq_length = int(data["past_seq_length"])
        current_input_length = int(seq_length)

        past_seq_tensor = torch.tensor([past_seq_length], dtype=torch.int32, device=device)
        current_len_tensor = torch.tensor([current_input_length], dtype=torch.int32, device=device)
        output = [inputs_embeds, past_seq_tensor, current_len_tensor]

        if self.enable_page_attention and self.attention_lowering not in {
            "flash_attention",
            "full_flash_attention",
        }:
            raise RuntimeError("Gemma4 PageAttention requires FlashAttention lowering")

        if self.attention_lowering == "flash_attention":
            kv_window_start_abs, kv_valid_length, mm_prefix_ranges = self._build_compact_attention_metadata(
                current_input_length=current_input_length,
                past_seq_length=past_seq_length,
                mm_token_type_ids=mm_token_type_ids,
                device=device,
            )
            if self.attention_visibility_spec.requires_mm_prefix_ranges:
                output.append(mm_prefix_ranges)
            output.extend([kv_window_start_abs, kv_valid_length])
        else:
            full_attention_mask, sliding_attention_mask = self._build_attention_masks(
                current_input_length=current_input_length,
                past_seq_length=past_seq_length,
                mm_token_type_ids=mm_token_type_ids,
                device=device,
            )
            if self.emit_full_attention_mask:
                output.append(full_attention_mask)
            output.append(sliding_attention_mask)
        if per_layer_inputs is not None:
            output.append(per_layer_inputs)
        if self.attention_lowering in {"legacy_attention", "full_flash_attention"} and self.emit_accepted_count_input:
            accepted_count = data.get("accepted_count", 0)
            if torch.is_tensor(accepted_count):
                accepted_count = accepted_count.to(device=device, dtype=torch.int32).reshape(1)
            else:
                accepted_count = torch.tensor([int(accepted_count)], dtype=torch.int32, device=device)
            output.append(accepted_count)
        if self.enable_page_attention:
            # Flash-to-Page removes only the continuous KV-cache graph inputs.
            # Visibility metadata remains in the converted graph as fixed slots
            # and is authoritative over the legacy context fallback.
            return tuple(output)
        output.extend([self.past_key_caches, self.past_value_caches])
        return tuple(output)


class Gemma4MoeDataPreprocess(Gemma4DataPreprocess):
    """Public Gemma4 MoE preprocessor with the same multimodal contract.

    The unified Gemma4 Series graph intentionally keeps one public text input
    order for Dense, MoE and E4B:
    ``inputs_embeds, past_seq_length, current_input_length, sliding_mask,
    caches...``.

    Legacy ``gemma4_moe`` used local/global mask order because that graph had a
    separate with-mask wrapper.  Reintroducing that order here silently feeds
    a full mask into sliding-attention layers, while 26B-A4B's
    sliding KV window is only ``sliding_window + input_sequence_length - 1``
    (1344 for 1024 + 320).  Keep the series contract canonical and let full
    layers use the standard causal masksoftmax path.
    """

    def __init__(self, *, sliding_window_cfg: dict, **kwargs):
        sliding_window = sliding_window_cfg.get("sliding_window")
        if sliding_window is None:
            sliding_window = kwargs.pop("sliding_window", 1024)
        else:
            kwargs.pop("sliding_window", None)
        super().__init__(sliding_window=int(sliding_window), **kwargs)
        self.sliding_window_cfg = sliding_window_cfg

    def forward(self, data: dict | tuple | list):
        return super().forward(data)
