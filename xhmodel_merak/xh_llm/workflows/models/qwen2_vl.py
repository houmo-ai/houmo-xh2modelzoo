from collections.abc import Mapping
from typing import Any

from xhmodel_merak.xh_llm.workflows.result import ExportResult

from ..base import BaseHMONNXWorkflow
from ..result import QuantResult


class XHQwen2VLHMONNXWorkflow(BaseHMONNXWorkflow):
    expected_model_config_cls_name = "XHQwen2VLModelConfig"
    expected_model_cls_name = "XHQwen2VLModel"

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is not None:
            raise NotImplementedError(f"{type(self).__name__} does not support quant yet!")
        return QuantResult(hf_model_dir=self.hf_model_dir, skipped=True)
    
    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        export_result = super().export(
                            quant_result=quant_result,
                            output_dir=output_dir,
                            device=device,
                            config_overrides=config_overrides
                        )
        return export_result

    def build_input_message(self, input_messages: Any) -> list[dict[str, Any]]:
        if not isinstance(input_messages, Mapping):
            raise ValueError("Qwen2-VL input_messages must be a mapping with 'image' and 'text'")
        if "image" not in input_messages:
            raise ValueError("Qwen2-VL input_messages must contain 'image'")
        if "text" not in input_messages:
            raise ValueError("Qwen2-VL input_messages must contain 'text'")
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": input_messages["image"],
                    },
                    {"type": "text", "text": input_messages["text"]},
                ],
            }
        ]


__all__ = ["XHQwen2VLHMONNXWorkflow"]
