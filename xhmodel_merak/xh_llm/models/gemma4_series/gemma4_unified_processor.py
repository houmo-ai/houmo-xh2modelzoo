"""Encoder-free Gemma4 Unified processor adapter."""

from __future__ import annotations

import copy
import os
import wave
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers.models.gemma4_unified.processing_gemma4_unified import (
    Gemma4UnifiedProcessor,
)

from .modality_contract import Gemma4SeriesModalityContract


class XHGemma4UnifiedProcessor(Gemma4UnifiedProcessor):
    """HF Unified processor with checkpoint-owned fixed graph padding.

    The upstream processor already produces merged 48x48 RGB patches and raw
    640-sample audio frames.  This adapter only enforces the static XH graph
    bounds and publishes valid-token counts; it deliberately does not create
    tower-only pooling or attention-mask inputs.
    """

    def __init__(
        self,
        feature_extractor,
        image_processor,
        tokenizer,
        video_processor,
        chat_template=None,
        image_seq_length: int = 280,
        audio_seq_length: int = 750,
        audio_ms_per_token: int = 40,
        *,
        modality_contract: Gemma4SeriesModalityContract,
        **kwargs,
    ):
        super().__init__(
            feature_extractor=feature_extractor,
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            image_seq_length=image_seq_length,
            audio_seq_length=audio_seq_length,
            audio_ms_per_token=audio_ms_per_token,
            **kwargs,
        )
        self.modality_contract = modality_contract
        # Existing Gemma4 runtime code reads processor.config for lightweight
        # runtime facts.  Keep that compatibility surface without importing the
        # legacy tower processor configuration.
        self.config = type("Gemma4UnifiedProcessorConfig", (), {})()
        self.config.frontend_kind = modality_contract.frontend_kind
        self.config.sampling_rate = modality_contract.sampling_rate
        self.config.audio_feature_length = modality_contract.audio_soft_tokens

    @classmethod
    def from_hf_processor(
        cls,
        processor: Gemma4UnifiedProcessor,
        *,
        model_dir: str,
    ) -> "XHGemma4UnifiedProcessor":
        contract = Gemma4SeriesModalityContract.from_pretrained(model_dir)
        return cls(
            feature_extractor=processor.feature_extractor,
            image_processor=processor.image_processor,
            tokenizer=processor.tokenizer,
            video_processor=processor.video_processor,
            chat_template=getattr(processor, "chat_template", None),
            image_seq_length=contract.image_soft_tokens,
            audio_seq_length=contract.audio_soft_tokens,
            audio_ms_per_token=getattr(processor, "audio_ms_per_token", 40),
            modality_contract=contract,
        )

    @staticmethod
    def _audio_lengths(audio: Any) -> list[int]:
        if audio is None:
            return []
        if isinstance(audio, np.ndarray):
            if audio.ndim == 1:
                return [int(audio.shape[0])]
            return [int(item.shape[0]) for item in audio]
        if torch.is_tensor(audio):
            if audio.ndim == 1:
                return [int(audio.shape[0])]
            return [int(item.shape[0]) for item in audio]
        if isinstance(audio, Sequence) and not isinstance(audio, (str, bytes)):
            if not audio:
                return []
            first = audio[0]
            if isinstance(first, (float, int, np.number)):
                return [len(audio)]
            return [int(np.asarray(item).shape[0]) for item in audio]
        return [int(np.asarray(audio).shape[0])]

    def _validate_call_contract(self, kwargs: dict[str, Any]) -> None:
        for name in ("max_soft_tokens", "image_seq_length", "audio_seq_length"):
            if name in kwargs:
                raise ValueError(f"Gemma4 Unified {name} is checkpoint-owned and cannot be overridden at runtime.")
        for scope in ("images_kwargs", "videos_kwargs"):
            scoped_kwargs = kwargs.get(scope)
            if isinstance(scoped_kwargs, Mapping) and "max_soft_tokens" in scoped_kwargs:
                raise ValueError(
                    f"Gemma4 Unified {scope}.max_soft_tokens is checkpoint-owned "
                    "and cannot be overridden at runtime."
                )

        audio = kwargs.get("audio")
        max_samples = self.modality_contract.max_audio_samples
        for num_samples in self._audio_lengths(audio):
            if num_samples > max_samples:
                raise ValueError(
                    "Gemma4 Unified audio exceeds the checkpoint audio limit: "
                    f"got {num_samples} samples, maximum is {max_samples} "
                    f"({self.modality_contract.audio_soft_tokens} tokens)."
                )

        sampling_rate = kwargs.get("sampling_rate")
        if audio is not None and sampling_rate is not None:
            if int(sampling_rate) != self.modality_contract.sampling_rate:
                raise ValueError(
                    "Gemma4 Unified audio must be sampled at "
                    f"{self.modality_contract.sampling_rate} Hz, got {sampling_rate}."
                )

        videos = kwargs.get("videos")
        if videos is not None and not self._videos_are_paths(videos):
            metadata = kwargs.get("video_metadata")
            if metadata is None:
                raise ValueError(
                    "Gemma4 Unified pre-sampled video frames require video_metadata with fps, "
                    "total_num_frames and frames_indices; the Transformers fps=24 fallback is disabled."
                )
            metadata_items = metadata if isinstance(metadata, Sequence) else [metadata]
            for item in metadata_items:
                fps = item.get("fps") if isinstance(item, dict) else getattr(item, "fps", None)
                total = (
                    item.get("total_num_frames") if isinstance(item, dict) else getattr(item, "total_num_frames", None)
                )
                indices = (
                    item.get("frames_indices") if isinstance(item, dict) else getattr(item, "frames_indices", None)
                )
                if fps is None or float(fps) <= 0 or total is None or indices is None:
                    raise ValueError(
                        "Gemma4 Unified video_metadata must provide positive fps, total_num_frames "
                        "and frames_indices for pre-sampled frames."
                    )

    @classmethod
    def _videos_are_paths(cls, videos: Any) -> bool:
        if isinstance(videos, (str, bytes, os.PathLike)):
            return True
        if isinstance(videos, Sequence) and not isinstance(videos, (np.ndarray, torch.Tensor)):
            return bool(videos) and all(cls._videos_are_paths(item) for item in videos)
        return False

    def __call__(self, *args, **kwargs):
        validation_kwargs = dict(kwargs)
        if "videos" not in validation_kwargs and len(args) >= 3:
            validation_kwargs["videos"] = args[2]
        if "audio" not in validation_kwargs and len(args) >= 4:
            validation_kwargs["audio"] = args[3]
        self._validate_call_contract(validation_kwargs)
        model_inputs = super().__call__(*args, **kwargs)

        if "pixel_values" in model_inputs:
            positions = model_inputs["image_position_ids"]
            if positions.shape[-2] != self.modality_contract.image_soft_tokens:
                raise ValueError(
                    "Gemma4 Unified image processor violated the checkpoint contract: "
                    f"expected {self.modality_contract.image_soft_tokens} tokens, "
                    f"got {positions.shape[-2]}."
                )
            model_inputs["image_soft_token_count"] = (~(positions == -1).all(dim=-1)).sum(dim=-1)

        if "pixel_values_videos" in model_inputs:
            positions = model_inputs["video_position_ids"]
            if positions.shape[-2] != self.modality_contract.video_soft_tokens_per_frame:
                raise ValueError(
                    "Gemma4 Unified video processor violated the checkpoint contract: "
                    f"expected {self.modality_contract.video_soft_tokens_per_frame} tokens per frame, "
                    f"got {positions.shape[-2]}."
                )
            model_inputs["video_soft_token_count"] = (~(positions == -1).all(dim=-1)).sum(dim=-1)

        if "input_features" in model_inputs:
            features = model_inputs["input_features"]
            mask = model_inputs["input_features_mask"].to(torch.bool)
            current_tokens = int(features.shape[-2])
            max_tokens = self.modality_contract.audio_soft_tokens
            if current_tokens > max_tokens:
                raise ValueError(
                    "Gemma4 Unified audio exceeds the checkpoint audio limit: "
                    f"got {current_tokens} tokens, maximum is {max_tokens}."
                )
            if features.shape[-1] != self.modality_contract.audio_feature_dim:
                raise ValueError(
                    "Gemma4 Unified audio feature width does not match the checkpoint: "
                    f"got {features.shape[-1]}, expected {self.modality_contract.audio_feature_dim}."
                )
            pad_tokens = max_tokens - current_tokens
            if pad_tokens:
                features = F.pad(features, (0, 0, 0, pad_tokens), value=0.0)
                mask = F.pad(mask, (0, pad_tokens), value=False)
            model_inputs["input_features"] = features
            model_inputs["input_features_mask"] = mask
            model_inputs["audio_soft_token_count"] = mask.sum(dim=-1)

        return model_inputs

    def apply_chat_template(self, messages, add_generation_prompt: bool = True, **kwargs):
        # Do not mutate caller-owned messages when ProcessorMixin resolves local
        # media entries and metadata.
        messages = copy.deepcopy(messages)
        wav_rates: set[int] = set()
        video_metadata = []
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video" and item.get("video_metadata") is not None:
                    video_metadata.append(item.pop("video_metadata"))
                if not isinstance(item, dict) or item.get("type") != "audio":
                    continue
                audio = item.get("audio", item.get("path"))
                if not isinstance(audio, (str, os.PathLike)):
                    continue
                path = os.fspath(audio)
                if not path.lower().endswith(".wav") or not os.path.isfile(path):
                    continue
                audio_array, sampling_rate = self._load_local_wav(path)
                item["audio"] = audio_array
                item.pop("path", None)
                wav_rates.add(sampling_rate)
        if wav_rates:
            if wav_rates != {self.modality_contract.sampling_rate}:
                raise ValueError(
                    "Gemma4 Unified local WAV files must be sampled at "
                    f"{self.modality_contract.sampling_rate} Hz, got {sorted(wav_rates)}."
                )
            processor_kwargs = kwargs.setdefault("processor_kwargs", {})
            if not isinstance(processor_kwargs, dict):
                raise TypeError("processor_kwargs must be a mapping when applying the Gemma4 Unified chat template")
            processor_kwargs.setdefault("sampling_rate", self.modality_contract.sampling_rate)
        if video_metadata:
            processor_kwargs = kwargs.setdefault("processor_kwargs", {})
            if not isinstance(processor_kwargs, dict):
                raise TypeError("processor_kwargs must be a mapping when applying the Gemma4 Unified chat template")
            processor_kwargs.setdefault("video_metadata", video_metadata)
            processor_kwargs.setdefault("do_sample_frames", False)
        kwargs.setdefault("tokenize", True)
        kwargs.setdefault("return_dict", True)
        kwargs.setdefault("return_tensors", "pt")
        return super().apply_chat_template(
            messages,
            add_generation_prompt=add_generation_prompt,
            **kwargs,
        )

    @staticmethod
    def _load_local_wav(path: str) -> tuple[np.ndarray, int]:
        """Load PCM WAV without the optional TorchCodec/FFmpeg dependency."""

        with wave.open(path, "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sampling_rate = wav_file.getframerate()
            frames = wav_file.readframes(wav_file.getnframes())
        if sample_width == 1:
            audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        elif sample_width == 2:
            audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
        elif sample_width == 4:
            audio = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sample_width}")
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        return audio, sampling_rate


__all__ = ["XHGemma4UnifiedProcessor"]
