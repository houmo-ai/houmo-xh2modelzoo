# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any

from ...workflows.base import BaseLLMWorkflow
from .calibration import load_calibration_samples, load_reference_trajectories


class HunyuanOCRWorkflow(BaseLLMWorkflow):
    """Model-specific workflow selected by HunyuanOCR YAML configurations."""

    expected_model_config_cls_name = "XHHunYuanOCRModelConfig"
    expected_model_cls_name = "XHHunYuanOCRModel"

    @staticmethod
    def build_input_messages(input_messages: Any) -> list[dict[str, Any]]:
        if not isinstance(input_messages, Mapping):
            raise ValueError("HunyuanOCR input_messages must be a mapping with 'image' and 'prompt'")
        image = input_messages.get("image")
        prompt = input_messages.get("prompt", input_messages.get("text"))
        if not isinstance(image, str) or not image:
            raise ValueError("HunyuanOCR input_messages must contain a non-empty 'image'")
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("HunyuanOCR input_messages must contain a non-empty 'prompt'")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def export(self, quant_result, output_dir, device, config_overrides=None) -> Any:
        model_config = self.workflow_config.export["model"]
        quant_type = str(model_config.get("quant_scheme", {}).get("quant_type", "")).lower()
        if quant_type.startswith("w8") and not bool(model_config.get("smoke_only", True)):
            _, image_paths = load_calibration_samples(model_config.get("calibration"))
            load_reference_trajectories(
                model_config.get("calibration"),
                request_count=len(image_paths),
                image_paths=image_paths,
            )
        return super().export(
            quant_result=quant_result,
            output_dir=output_dir,
            device=device,
            config_overrides=config_overrides,
        )


__all__ = ["HunyuanOCRWorkflow"]