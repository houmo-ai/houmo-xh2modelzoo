import re
from typing import Any

import torch
from transformers.models.gemma4.processing_gemma4 import Gemma4Processor
from xhmodel_merak.configuration_utils import BaseConfig


class Gemma4ProcessorConfig(BaseConfig):
    def __init__(self):
        super().__init__()
        self.max_size_h: int = 448
        self.max_size_w: int = 448
        self.patch_size: int = 16
        self.sampling_rate: int = 16000
        self.audio_feature_length: int | None = None


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

    def __call__(self, *args, **kwargs):
        model_inputs = super().__call__(*args, **kwargs)
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

        text_only_inputs = super().__call__(text=adjusted_text, **retokenize_kwargs)
        for key in ("input_ids", "attention_mask", "mm_token_type_ids"):
            if key in text_only_inputs:
                model_inputs[key] = text_only_inputs[key]
        return model_inputs

    def _normalize_role(self, role: str) -> str:
        if role == "assistant":
            return "model"
        return role

    def _render_messages(self, messages: list[dict[str, Any]], add_generation_prompt: bool) -> tuple[str, list, list, int]:
        rendered_messages: list[str] = []
        images: list[Any] = []
        audios: list[Any] = []
        sampling_rate = self.config.sampling_rate

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
                        parts.append("\n\n<|image|>\n\n")
                        images.append(item.get("image", item.get("url", item.get("image_url"))))
                    elif item_type == "audio":
                        parts.append("<|audio|>")
                        audios.append(item.get("audio"))
                        sampling_rate = int(item.get("sampling_rate", sampling_rate))
                    elif item_type == "video":
                        parts.append("\n\n<|video|>\n\n")
                    else:
                        parts.append(item.get("text", "").strip())
                body = "".join(parts)

            rendered_messages.append(f"<|turn>{role}\n{body}<turn|>\n")

        if add_generation_prompt:
            rendered_messages.append("<|turn>model\n")

        return "".join(rendered_messages), images, audios, sampling_rate

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
