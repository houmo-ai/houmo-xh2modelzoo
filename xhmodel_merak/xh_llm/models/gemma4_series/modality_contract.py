"""Checkpoint-derived multimodal contracts for Gemma4 Series frontends."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Gemma4SeriesModalityContract:
    """Static frontend shapes derived from checkpoint-owned configuration."""

    frontend_kind: str
    image_soft_tokens: int
    video_soft_tokens_per_frame: int
    audio_soft_tokens: int
    vision_patch_dim: int
    audio_feature_dim: int
    position_capacity: int
    sampling_rate: int
    config_hash: str

    @property
    def max_audio_samples(self) -> int:
        return self.audio_soft_tokens * self.audio_feature_dim

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "Gemma4SeriesModalityContract":
        root = Path(model_dir)
        config_path = root / "config.json"
        processor_path = root / "processor_config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        processor = json.loads(processor_path.read_text(encoding="utf-8"))

        architectures = config.get("architectures") or []
        is_unified = config.get("model_type") == "gemma4_unified" or (
            architectures and architectures[0] == "Gemma4UnifiedForConditionalGeneration"
        )
        if not is_unified:
            raise ValueError(
                "Gemma4SeriesModalityContract currently describes only the encoder-free "
                f"Gemma4 Unified frontend, got model_type={config.get('model_type')!r}."
            )

        vision = config["vision_config"]
        audio = config["audio_config"]
        image_processor = processor["image_processor"]
        video_processor = processor["video_processor"]
        feature_extractor = processor["feature_extractor"]
        # ``model_patch_size`` was a redundant serialized field in the
        # pre-5.13 Gemma4 Unified config.  Transformers 5.13 exposes it as the
        # product of the encoder patch and pooling sizes, so a freshly saved
        # (including GPTQModel-quantized) checkpoint no longer writes that
        # legacy key.  Accept both representations, but keep validating the
        # result against the processor-owned contract below.
        if "model_patch_size" in vision:
            patch_size = int(vision["model_patch_size"])
        else:
            patch_size = int(vision["patch_size"]) * int(vision["pooling_kernel_size"])
        processor_patch_size = int(image_processor["patch_size"]) * int(image_processor["pooling_kernel_size"])
        # Likewise, Transformers 5.13 derives ``audio_samples_per_token`` from
        # ``audio_embed_dim`` and omits the former when serializing a canonical
        # config.
        if "audio_samples_per_token" in audio:
            audio_samples_per_token = int(audio["audio_samples_per_token"])
        else:
            audio_samples_per_token = int(audio["audio_embed_dim"])
        checkpoint_pairs = {
            "vision model_patch_size": (patch_size, processor_patch_size),
            "image soft tokens": (int(vision["num_soft_tokens"]), int(image_processor["max_soft_tokens"])),
            "processor image_seq_length": (int(vision["num_soft_tokens"]), int(processor["image_seq_length"])),
            "audio feature width": (
                audio_samples_per_token,
                int(feature_extractor["feature_size"]),
            ),
        }
        inconsistent = {name: values for name, values in checkpoint_pairs.items() if values[0] != values[1]}
        if inconsistent:
            details = ", ".join(f"{name}={left}/{right}" for name, (left, right) in inconsistent.items())
            raise ValueError(f"Gemma4 Unified config/processor contract is inconsistent: {details}")
        digest = hashlib.sha256(config_path.read_bytes() + processor_path.read_bytes()).hexdigest()

        contract = cls(
            frontend_kind="encoder_free",
            image_soft_tokens=int(vision["num_soft_tokens"]),
            video_soft_tokens_per_frame=int(video_processor["max_soft_tokens"]),
            audio_soft_tokens=int(processor["audio_seq_length"]),
            vision_patch_dim=patch_size * patch_size * 3,
            audio_feature_dim=audio_samples_per_token,
            position_capacity=int(vision["mm_posemb_size"]),
            sampling_rate=int(feature_extractor["sampling_rate"]),
            config_hash=digest,
        )
        contract.validate_supported()
        return contract

    def validate_supported(self) -> None:
        expected = {
            "image_soft_tokens": 280,
            "video_soft_tokens_per_frame": 70,
            "audio_soft_tokens": 750,
            "vision_patch_dim": 6912,
            "audio_feature_dim": 640,
        }
        mismatches = {
            name: (getattr(self, name), value) for name, value in expected.items() if getattr(self, name) != value
        }
        if mismatches:
            details = ", ".join(
                f"{name}={actual} (supported {supported})" for name, (actual, supported) in mismatches.items()
            )
            raise ValueError(f"Unsupported Gemma4 Unified modality contract: {details}")


__all__ = ["Gemma4SeriesModalityContract"]
