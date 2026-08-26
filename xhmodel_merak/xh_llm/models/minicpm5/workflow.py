"""Merak workflow for MiniCPM5-15B-A2.5B."""

from collections.abc import Mapping
from typing import Any

from ...workflows.base import BaseLLMWorkflow
from ...workflows.result import ExportResult, QuantResult


class MiniCPM5Workflow(BaseLLMWorkflow):
    expected_model_config_cls_name = "XHMiniCPM5ModelConfig"
    expected_model_cls_name = "XHMiniCPM5Model"

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        quant_cfg = workflow_config.quant
        if quant_cfg is None:
            return QuantResult(raw_model_dir=self.model_dir, skipped=True)
        algorithm = str(quant_cfg.get("algorithm", "gptqmodel")).lower().replace("-", "_")
        if algorithm not in {"gptq", "gptqmodel"}:
            raise NotImplementedError(
                "MiniCPM5Workflow.quant supports quant=None for the floating-point HF path, "
                "or quant.algorithm='gptqmodel' for WikiText-calibrated GPTQ."
            )
        from .quant_adapter import quantize_with_gptqmodel_api

        return quantize_with_gptqmodel_api(
            model_dir=self.model_dir,
            output_dir=output_dir,
            device=device,
            quant_cfg=quant_cfg,
        )

    def dump_golden(self, export_result: ExportResult, device: str, input_messages: Any) -> str:
        from transformers import TextStreamer

        from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
        from xhquant.utils import ContextManagers

        meta_file = self._find_golden_meta_file(export_result)
        hmonnx_model = AutoLLMHONNXModel.from_pretrained(
            meta_file,
            device_map=[device] if isinstance(device, str) else device,
            enable_golden=True,
        )
        tokenizer = hmonnx_model.get_tokenizer()
        prompt = input_messages if isinstance(input_messages, str) else input_messages.get("text", "")
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(hmonnx_model.device)
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            eos_token_id = tokenizer.eos_token_id
            if isinstance(eos_token_id, (list, tuple)):
                eos_token_id = eos_token_id[0]
            if eos_token_id is None:
                raise ValueError("MiniCPM5 tokenizer has no pad or EOS token id")
            pad_token_id = int(eos_token_id)
        with ContextManagers([LLMInferenceContextManager(hmonnx_model)]):
            hmonnx_model.generate(
                **model_inputs,
                max_new_tokens=2,
                streamer=TextStreamer(tokenizer),
                do_sample=False,
                pad_token_id=pad_token_id,
            )
        return meta_file


__all__ = ["MiniCPM5Workflow"]
