import os
from collections.abc import Mapping
from typing import Any

from ...workflows.base import BaseHMONNXWorkflow
from ...workflows.result import ExportResult, QuantResult


_QWEN35_MODEL_CLS_NAMES = {
    "XHQwen3_5Model",
    "XHQwen3_5MoeModel",
    "XHQwen3_5VisionModel",
    "XHQwen3_5MoeVisionModel",
}
_QWEN35_MODEL_CONFIG_CLS_NAMES = {
    "XHQwen3_5ModelConfig",
    "XHQwen3_5MoeModelConfig",
    "XHQwen3_5_VisualConfig",
    "XHQwen3_5Moe_VisualConfig",
}



class Qwen35Workflow(BaseHMONNXWorkflow):
    """Merak HMONNX workflow for Qwen3.5/Qwen3.6 dense, MoE, and visual exports."""

    @classmethod
    def from_config(
        cls,
        hf_model_dir: str,
        config_path: str,
        seed: int = 1024,
        debug: bool = False,
    ) -> "Qwen35Workflow":
        return cls(hf_model_dir=hf_model_dir, config_path=config_path, seed=seed, debug=debug)

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        quant_cfg = workflow_config.quant
        if quant_cfg is None:
            if not self._is_explicit_base_quant_override(config_overrides):
                raise ValueError(
                    "Qwen35Workflow default workflow YAML must configure quantization. "
                    "Use config_overrides={'quant': None} only for explicit base-model validation."
                )
            return QuantResult(
                hf_model_dir=self.hf_model_dir,
                skipped=True,
            )

        self._validate_group_size(quant_cfg)
        algorithm = str(quant_cfg.get("algorithm") or "gptqmodel").lower().replace("-", "_")
        artifact_format = self._resolve_artifact_format(quant_cfg)
        if artifact_format != "gptqmodel_hf":
            raise ValueError(
                "Qwen35Workflow.quant requires artifact_format/output_format='gptqmodel_hf'; "
                f"got {artifact_format!r}."
            )

        if algorithm == "existing_hf":
            existing_hf_model_dir = quant_cfg.get("existing_hf_model_dir")
            if not existing_hf_model_dir:
                raise ValueError("quant.algorithm='existing_hf' requires quant.existing_hf_model_dir")
            return QuantResult(
                hf_model_dir=self.hf_model_dir,
                quanted_model_dir=self._normalize_path(existing_hf_model_dir),
            )

        export_model_cfg = workflow_config.export["model"]
        method = str(quant_cfg.get("method") or "").lower().replace("-", "_")
        if algorithm in {"autoround", "auto_round"} or (
            algorithm == "gptqmodel" and method in {"autoround", "auto_round", "mode1"}
        ):
            return self._quant_autoround_api(
                output_dir=output_dir,
                device=device,
                quant_cfg=quant_cfg,
                export_model_cfg=export_model_cfg,
            )

        if algorithm in {"gptqmodel", "gptq"}:
            return self._quant_gptqmodel_api(
                output_dir=output_dir,
                device=device,
                quant_cfg=quant_cfg,
                export_model_cfg=export_model_cfg,
            )

        raise NotImplementedError(
            "Qwen35Workflow.quant supports quant=None, quant.algorithm='existing_hf', "
            "quant.algorithm='gptqmodel' with method='gptq' or method='autoround', "
            "or legacy quant.algorithm='autoround'/'gptq', with "
            "artifact_format/output_format='gptqmodel_hf'. "
            f"Got algorithm={algorithm!r}, artifact_format={artifact_format!r}."
        )

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        self._validate_export_model(config_overrides)
        return super().export(
            quant_result=quant_result,
            output_dir=output_dir,
            device=device,
            config_overrides=config_overrides,
        )

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any,
    ) -> str:
        from transformers import TextStreamer

        from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
        from xhquant.api import get_xhquant_logger
        from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler

        meta_file = self._find_golden_meta_file(export_result)
        logger = get_xhquant_logger()
        hmonnx_model = AutoLLMHONNXModel.from_pretrained(meta_file)
        messages = self.build_input_message(input_messages)

        if self._messages_have_image(messages):
            processor = hmonnx_model.get_tf_processor()
            tokenizer = processor.tokenizer
            model_inputs = processor.apply_chat_template(messages).to(device)
            decode = processor.batch_decode
        else:
            tokenizer = hmonnx_model.get_tokenizer()
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            model_inputs = tokenizer([text], return_tensors="pt", truncation=True).to(device)
            decode = tokenizer.batch_decode

        streamer = TextStreamer(tokenizer)
        hmonnx_model.to(device)
        hmonnx_model.enable_golden = True
        logger.warning("Golden outputs should be generated in aligned precision for stability.")

        contexts = [
            TimeProfiler("hmonnx_generate_golden", logger),
            MemoryTracker(device=device, name="generate_golden", logger=logger),
            LLMInferenceContextManager(hmonnx_model),
        ]
        with ContextManagers(contexts):
            generated_ids = hmonnx_model.generate(
                **model_inputs,
                max_new_tokens=2,
                streamer=streamer,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids, strict=False)
        ]
        output_text = decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        logger.info(f"{'-' * 20} Golden output {'-' * 20}")
        logger.info(f"{output_text}")
        return meta_file

    def build_input_message(self, input_messages: Any) -> list[dict[str, Any]]:
        if isinstance(input_messages, list):
            return input_messages
        if isinstance(input_messages, str):
            text = input_messages
            if not text:
                raise ValueError("Qwen3.5 input text must be a non-empty string")
            return [{"role": "user", "content": text}]
        if not isinstance(input_messages, Mapping):
            raise ValueError("Qwen3.5 input_messages must be a string, message list, or mapping")

        if "messages" in input_messages:
            messages = input_messages["messages"]
            if not isinstance(messages, list):
                raise ValueError("Qwen3.5 input_messages['messages'] must be a list")
            return messages

        text = input_messages.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("Qwen3.5 input_messages mapping must contain non-empty 'text'")

        image = input_messages.get("image", input_messages.get("images"))
        if image is None:
            return [{"role": "user", "content": text}]

        images = image if isinstance(image, list) else [image]
        content = [{"type": "image", "image": item} for item in images]
        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _validate_export_model(self, config_overrides: Mapping[str, Any] | None) -> None:
        from ...builder import get_model_class

        workflow_config = self.workflow_config.with_overrides(config_overrides)
        export_cfg = workflow_config.build_export_dict(self.hf_model_dir)
        model_cls = get_model_class(export_cfg["model"])
        if model_cls is None:
            return
        model_cls_name = model_cls.__name__
        config_cls = getattr(model_cls, "CONFIG_CLS", None)
        config_cls_name = getattr(config_cls, "__name__", None)
        if model_cls_name not in _QWEN35_MODEL_CLS_NAMES or config_cls_name not in _QWEN35_MODEL_CONFIG_CLS_NAMES:
            raise TypeError(
                "Qwen35Workflow only supports Qwen3.5/Qwen3.6 dense, MoE, and visual model classes; "
                f"got model={model_cls_name}, config={config_cls_name}."
            )


    def _quant_autoround_api(
        self,
        output_dir: str,
        device: str,
        quant_cfg: Mapping[str, Any],
        export_model_cfg: Mapping[str, Any],
    ) -> QuantResult:
        from .quant_adapter import quantize_with_autoround_api

        return quantize_with_autoround_api(
            hf_model_dir=self.hf_model_dir,
            output_dir=output_dir,
            device=device,
            quant_cfg=quant_cfg,
            export_model_cfg=export_model_cfg,
            workflow_seed=self.seed,
        )

    def _quant_gptqmodel_api(
        self,
        output_dir: str,
        device: str,
        quant_cfg: Mapping[str, Any],
        export_model_cfg: Mapping[str, Any],
    ) -> QuantResult:
        from .quant_adapter import quantize_with_gptqmodel_api

        return quantize_with_gptqmodel_api(
            hf_model_dir=self.hf_model_dir,
            output_dir=output_dir,
            device=device,
            quant_cfg=quant_cfg,
            export_model_cfg=export_model_cfg,
            workflow_seed=self.seed,
        )

    @staticmethod
    def _resolve_artifact_format(quant_cfg: Mapping[str, Any]) -> str:
        artifact_format = quant_cfg.get("artifact_format")
        output_format = quant_cfg.get("output_format")
        if artifact_format is not None and output_format is not None and artifact_format != output_format:
            raise ValueError(
                "quant.artifact_format and quant.output_format must match when both are provided; "
                f"got artifact_format={artifact_format!r}, output_format={output_format!r}"
            )
        return artifact_format or output_format or "gptqmodel_hf"

    @staticmethod
    def _validate_group_size(quant_cfg: Mapping[str, Any]) -> None:
        group_size = quant_cfg.get("group_size")
        if group_size is not None and int(group_size) != 64:
            raise ValueError(f"Qwen3.5/Qwen3.6 quant group_size must be 64 when provided, got {group_size!r}")

    @staticmethod
    def _is_explicit_base_quant_override(config_overrides: Mapping[str, Any] | None) -> bool:
        return bool(config_overrides and "quant" in config_overrides and config_overrides["quant"] is None)

    @staticmethod
    def _messages_have_image(messages: list[dict[str, Any]]) -> bool:
        for message in messages:
            content = message.get("content") if isinstance(message, Mapping) else None
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, Mapping) and item.get("type") == "image":
                        return True
        return False

    @staticmethod
    def _normalize_path(path: str | os.PathLike[str]) -> str:
        return os.path.abspath(os.path.normpath(str(path)))


__all__ = ["Qwen35Workflow"]
