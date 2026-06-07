import re
from typing import Any

import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.models.gemma4.processing_gemma4 import Gemma4Processor
from xhmodel_merak.configuration_utils import BaseConfig


DEFAULT_VISUAL_MAX_SOFT_TOKENS = 280
FULL_VISUAL_POOLING_KERNEL_SIZE = 3
COMPACT_VISUAL_POOLING_KERNEL_SIZE = 1


class Gemma4ProcessorConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.max_size_h: int = 448
        self.max_size_w: int = 448
        self.patch_size: int = 16
        self.sampling_rate: int = 16000
        self.audio_feature_length: int | None = None
        self.export_mode: str = "full"
        self.enforce_fixed_image_size: bool = False


class XHGemma4Processor(Gemma4Processor):
    def __init__(self, image_processor=None, tokenizer=None, feature_extractor=None, video_processor=None, **kwargs):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            video_processor=video_processor,
            **kwargs,
        )
        self.config = Gemma4ProcessorConfig()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, trust_remote_code: bool = True, **kwargs):
        processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        if isinstance(processor, cls):
            return processor
        return cls(
            feature_extractor=processor.feature_extractor,
            image_processor=processor.image_processor,
            tokenizer=processor.tokenizer,
            video_processor=processor.video_processor,
            chat_template=getattr(processor, "chat_template", None),
            image_seq_length=getattr(processor, "image_seq_length", 280),
            audio_seq_length=getattr(processor, "audio_seq_length", 750),
            audio_ms_per_token=getattr(processor, "audio_ms_per_token", 40),
        )

    def _resize_image_for_export_contract(self, image: Any) -> Any:
        if not self.config.enforce_fixed_image_size or not isinstance(image, Image.Image):
            return image
        target_size = (self.config.max_size_w, self.config.max_size_h)
        if image.size == target_size:
            return image
        return image.convert("RGB").resize(target_size, Image.Resampling.BICUBIC)

    def _prepare_vision_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        if "images" not in kwargs or kwargs["images"] is None:
            return kwargs

        prepared = dict(kwargs)
        images = prepared["images"]

        def _resize_recursive(item: Any) -> Any:
            if isinstance(item, Image.Image):
                return self._resize_image_for_export_contract(item)
            if isinstance(item, (list, tuple)):
                return type(item)(_resize_recursive(subitem) for subitem in item)
            return item

        prepared["images"] = _resize_recursive(images)
        return prepared

    def __call__(self, *args, **kwargs):
        kwargs = self._prepare_vision_kwargs(kwargs)
        model_inputs = super().__call__(*args, **kwargs)
        model_inputs = self._trim_compact_image_patches_if_needed(model_inputs)
        feature_length = self.config.audio_feature_length
        if (
            feature_length is not None
            and feature_length > 0
            and "input_features" in model_inputs
            and "input_features_mask" in model_inputs
        ):
            input_features = model_inputs["input_features"]
            input_features_mask = model_inputs["input_features_mask"]
            current_length = input_features.shape[1]
            if current_length > feature_length:
                model_inputs["input_features"] = input_features[:, :feature_length, :]
                model_inputs["input_features_mask"] = input_features_mask[:, :feature_length]
            elif current_length < feature_length:
                pad_length = feature_length - current_length
                feature_pad = torch.zeros(
                    (input_features.shape[0], pad_length, input_features.shape[2]),
                    dtype=input_features.dtype,
                    device=input_features.device,
                )
                mask_pad = torch.zeros(
                    (input_features_mask.shape[0], pad_length),
                    dtype=input_features_mask.dtype,
                    device=input_features_mask.device,
                )
                model_inputs["input_features"] = torch.cat([input_features, feature_pad], dim=1)
                model_inputs["input_features_mask"] = torch.cat([input_features_mask, mask_pad], dim=1)
            model_inputs = self._retokenize_audio_placeholders_if_needed(model_inputs, kwargs)
        return model_inputs

    def _trim_compact_image_patches_if_needed(self, model_inputs):
        if self.config.export_mode != "compact":
            return model_inputs
        pixel_values = model_inputs.get("pixel_values")
        image_position_ids = model_inputs.get("image_position_ids")
        if pixel_values is None or image_position_ids is None:
            return model_inputs

        if image_position_ids.dim() == 2:
            valid_positions = ~(image_position_ids == -1).all(dim=-1)
            real_patch_count = int(valid_positions.sum().item())
        else:
            valid_positions = ~(image_position_ids == -1).all(dim=-1)
            real_patch_counts = valid_positions.to(torch.int64).sum(dim=1)
            max_real_patch_count = int(real_patch_counts.max().item())
            min_real_patch_count = int(real_patch_counts.min().item())
            if min_real_patch_count != max_real_patch_count:
                raise ValueError(
                    "Compact Gemma4 vision export requires a uniform real patch count across the batch, "
                    f"got {real_patch_counts.tolist()}."
                )
            real_patch_count = max_real_patch_count

        if pixel_values.shape[1] == real_patch_count:
            return model_inputs

        model_inputs["pixel_values"] = pixel_values[:, :real_patch_count, :]
        if image_position_ids.dim() == 2:
            model_inputs["image_position_ids"] = image_position_ids[:real_patch_count, :]
        else:
            model_inputs["image_position_ids"] = image_position_ids[:, :real_patch_count, :]
        return model_inputs

    def _compute_audio_soft_token_count_from_feature_frames(self, num_feature_frames: int) -> int:
        if num_feature_frames <= 0:
            return 0
        tokens = num_feature_frames
        for _ in range(2):
            tokens = (tokens + 2 - 3) // 2 + 1
        return min(tokens, self.audio_seq_length)

    def _retokenize_audio_placeholders_if_needed(self, model_inputs, kwargs):
        text = kwargs.get("text")
        input_ids = model_inputs.get("input_ids")
        input_features_mask = model_inputs.get("input_features_mask")
        audio_token_id = getattr(self.tokenizer, "audio_token_id", None)

        if (
            text is None
            or input_ids is None
            or input_features_mask is None
            or audio_token_id is None
            or self.audio_token is None
            or self.boa_token is None
            or self.eoa_token is None
        ):
            return model_inputs

        if input_features_mask.ndim == 1:
            feature_masks = [input_features_mask]
        else:
            feature_masks = list(input_features_mask)
        expected_counts = [
            self._compute_audio_soft_token_count_from_feature_frames(int(feature_mask.to(torch.int64).sum().item()))
            for feature_mask in feature_masks
        ]
        actual_audio_token_count = int((input_ids == audio_token_id).sum().item())
        if actual_audio_token_count == sum(expected_counts):
            return model_inputs

        texts = [text] if isinstance(text, str) else list(text)
        placeholder_count = sum(prompt.count(self.audio_token) for prompt in texts)
        if placeholder_count != len(expected_counts):
            raise ValueError(
                f"Audio placeholder count does not match audio inputs: {placeholder_count} vs {len(expected_counts)}"
            )

        replacements = iter(
            f"{self.boa_token}{self.audio_token * token_count}{self.eoa_token}" for token_count in expected_counts
        )
        audio_pattern = re.escape(self.audio_token)
        adjusted_text = [
            re.sub(audio_pattern, lambda _: next(replacements), prompt)
            for prompt in texts
        ]
        if isinstance(text, str):
            adjusted_text = adjusted_text[0]

        retokenize_kwargs = {}
        for key in (
            "padding",
            "truncation",
            "max_length",
            "pad_to_multiple_of",
            "return_attention_mask",
            "return_token_type_ids",
            "return_tensors",
        ):
            if key in kwargs:
                retokenize_kwargs[key] = kwargs[key]

        # The parent processor expands `<|image|>` into `image_seq_length` soft
        # tokens only when `images` is provided. If we re-tokenize with text
        # alone, the image placeholder collapses back to a single token while
        # the already-computed `pixel_values` still hold all image features,
        # producing a "Feature count does not match token count" error in the
        # downstream scatter. Forward `images` (and skip audio so we don't
        # overwrite the just-adjusted audio features) so both placeholders
        # expand consistently.
        if "images" in kwargs and kwargs["images"] is not None:
            retokenize_kwargs["images"] = kwargs["images"]

        text_only_inputs = super().__call__(text=adjusted_text, **retokenize_kwargs)
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key in text_only_inputs:
                model_inputs[key] = text_only_inputs[key]
        return model_inputs

    def _normalize_role(self, role: str) -> str:
        if role == "assistant":
            return "model"
        return role

    def _render_messages_fallback(self, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
        rendered_messages: list[str] = []
        bos_token = self.tokenizer.bos_token or ""
        if bos_token:
            rendered_messages.append(bos_token)

        for message in messages:
            role = self._normalize_role(message.get("role", "user"))
            content = message.get("content", "")
            if isinstance(content, str):
                body = content.strip()
            else:
                parts: list[str] = []
                for item in content:
                    item_type = item.get("type", "text")
                    if item_type in ("image", "image_url"):
                        parts.append("<|image|>")
                    elif item_type == "audio":
                        parts.append("<|audio|>")
                    elif item_type == "video":
                        parts.append("<|video|>")
                    else:
                        parts.append(item.get("text", "").strip())
                body = "".join(parts)
            rendered_messages.append(f"<|turn>{role}\n{body}<turn|>\n")

        if add_generation_prompt:
            rendered_messages.append("<|turn>model\n")
        return "".join(rendered_messages)

    def _render_messages(self, messages: list[dict[str, Any]], add_generation_prompt: bool) -> tuple[str, list, list, int]:
        images: list[Any] = []
        audios: list[Any] = []
        sampling_rate = self.config.sampling_rate

        for message in messages:
            content = message.get("content", "")
            if not isinstance(content, str):
                for item in content:
                    item_type = item.get("type", "text")
                    if item_type in ("image", "image_url"):
                        images.append(item.get("image", item.get("url", item.get("image_url"))))
                    elif item_type == "audio":
                        audios.append(item.get("audio"))
                        sampling_rate = int(item.get("sampling_rate", sampling_rate))
        if getattr(self, "chat_template", None):
            rendered = super().apply_chat_template(messages, add_generation_prompt=add_generation_prompt, tokenize=False)
        else:
            rendered = self._render_messages_fallback(messages, add_generation_prompt)
        return rendered, images, audios, sampling_rate

    def apply_chat_template(self, messages: list[dict[str, Any]], add_generation_prompt: bool = True, **kwargs):
        text, images, audios, sampling_rate = self._render_messages(messages, add_generation_prompt)

        processor_kwargs = {
            "text": text,
            "padding": True,
            "return_tensors": "pt",
        }
        processor_kwargs.update(kwargs)
        if images:
            processor_kwargs["images"] = images[0] if len(images) == 1 else images
        if audios:
            processor_kwargs["audio"] = audios[0] if len(audios) == 1 else audios
            processor_kwargs["sampling_rate"] = sampling_rate
        return self(**processor_kwargs)


def configure_gemma4_visual_processor(
    processor: XHGemma4Processor,
    *,
    export_mode: str,
    max_size_w: int,
    max_size_h: int,
    patch_size: int,
    image_seq_length: int,
) -> XHGemma4Processor:
    processor.config.max_size_h = max_size_h
    processor.config.max_size_w = max_size_w
    processor.config.patch_size = patch_size
    processor.config.export_mode = export_mode
    processor.config.enforce_fixed_image_size = True
    processor.image_processor.max_soft_tokens = DEFAULT_VISUAL_MAX_SOFT_TOKENS
    processor.image_processor.pooling_kernel_size = (
        COMPACT_VISUAL_POOLING_KERNEL_SIZE if export_mode == "compact" else FULL_VISUAL_POOLING_KERNEL_SIZE
    )
    processor.image_seq_length = image_seq_length
    return processor
