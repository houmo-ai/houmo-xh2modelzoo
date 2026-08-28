import os
from collections.abc import Mapping
from typing import Any

from ...workflows.base import BaseLLMWorkflow
from ...workflows.result import ExportResult, QuantResult


class LagunaWorkflow(BaseLLMWorkflow):
    expected_model_config_cls_name = "XHLagunaModelConfig"
    expected_model_cls_name = "XHLagunaModel"

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        from transformers import AutoConfig

        workflow_config = self.workflow_config.with_overrides(config_overrides)
        model_config = workflow_config.export["model"]
        hf_config = AutoConfig.from_pretrained(self.model_dir, trust_remote_code=True)
        self._validate_sliding_window_cache(
            context_max_length=int(model_config["context_max_length"]),
            prefill_chunk_length=int(model_config["prefill_chunk_length"]),
            sliding_window=int(getattr(hf_config, "sliding_window", 0) or 0),
        )
        return super().export(
            quant_result=quant_result,
            output_dir=output_dir,
            device=device,
            config_overrides=config_overrides,
        )

    @staticmethod
    def _validate_sliding_window_cache(
        context_max_length: int,
        prefill_chunk_length: int,
        sliding_window: int,
    ) -> None:
        if sliding_window <= 0:
            return
        minimum_context_length = sliding_window + prefill_chunk_length
        if context_max_length < minimum_context_length:
            raise ValueError(
                "Laguna sliding-window KV cache requires context_max_length >= "
                "sliding_window + prefill_chunk_length: "
                f"{context_max_length} < {sliding_window} + {prefill_chunk_length} "
                f"({minimum_context_length})."
            )

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
                "LagunaWorkflow.quant supports quant=None for the default HF floating-point path, "
                "or quant.algorithm='gptqmodel' for the optional GPTQModel path."
            )
        from .quant_adapter import quantize_with_gptqmodel_api

        return quantize_with_gptqmodel_api(
            model_dir=self.model_dir,
            output_dir=output_dir,
            device=device,
            quant_cfg=quant_cfg,
        )

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str | list[str],
        input_messages: Any,
    ) -> str:
        from transformers import TextStreamer

        from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
        from xhquant.api import get_xhquant_logger
        from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler

        meta_file = self._find_golden_meta_file(export_result)
        logger = get_xhquant_logger()
        device_map = self._normalize_device_map(device)
        self._configure_hmonnx_pipeline(device_map)
        hmonnx_model = AutoLLMHONNXModel.from_pretrained(
            meta_file,
            device_map=device_map,
            enable_auto_offload=len(device_map) > 1,
            enable_golden=True,
        )
        tokenizer = hmonnx_model.get_tokenizer()

        text = tokenizer.apply_chat_template(
            self.build_input_message(input_messages),
            tokenize=False,
            add_generation_prompt=True,
        )
        input_device = hmonnx_model.get_input_embeddings().weight.device
        model_inputs = tokenizer([text], return_tensors="pt", truncation=True).to(input_device)

        hmonnx_model.enable_golden = True
        contexts = [
            TimeProfiler("laguna_hmonnx_generate_golden", logger),
            MemoryTracker(device=input_device, name="laguna_generate_golden", logger=logger),
            LLMInferenceContextManager(hmonnx_model),
        ]
        with ContextManagers(contexts):
            generated_ids = hmonnx_model.generate(
                **model_inputs,
                max_new_tokens=2,
                streamer=TextStreamer(tokenizer),
                do_sample=False,
                pad_token_id=self._resolve_pad_token_id(tokenizer),
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]) :].tolist()
        logger.info("Laguna golden output: %s", tokenizer.decode(output_ids, skip_special_tokens=True).strip())
        return meta_file

    @staticmethod
    def _normalize_device_map(device: str | list[str]) -> list[str]:
        if isinstance(device, str):
            devices = [item.strip() for item in device.split(",") if item.strip()]
        else:
            devices = [str(item).strip() for item in device if str(item).strip()]
        if not devices:
            raise ValueError("Laguna golden device map must contain at least one device")
        return devices

    @staticmethod
    def _configure_hmonnx_pipeline(device_map: list[str]) -> None:
        if len(device_map) > 1:
            os.environ["HMONNX_PIPELINE_DEVICES"] = ",".join(device_map)

    @staticmethod
    def _resolve_pad_token_id(tokenizer: Any) -> int:
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is not None:
            return int(pad_token_id)
        eos_token_id = tokenizer.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            if not eos_token_id:
                raise ValueError("Laguna tokenizer has no pad or EOS token id")
            eos_token_id = eos_token_id[0]
        if eos_token_id is None:
            raise ValueError("Laguna tokenizer has no pad or EOS token id")
        return int(eos_token_id)

    @staticmethod
    def build_input_message(input_messages: Any) -> list[dict[str, str]]:
        if isinstance(input_messages, str):
            prompt = input_messages
        elif isinstance(input_messages, Mapping):
            prompt = input_messages.get("text")
        else:
            prompt = None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Laguna input_messages must contain non-empty text")
        return [{"role": "user", "content": prompt}]


__all__ = ["LagunaWorkflow"]
