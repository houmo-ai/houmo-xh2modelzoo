import os
from collections.abc import Mapping
from pathlib import Path
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

_QWEN35MOE_ATTN_LAYER_CONFIG_PATTERNS = (
    "model.language_model.layers.*.self_attn.(q_proj|k_proj|v_proj|o_proj)",
    "model.language_model.layers.*.linear_attn.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)",
)
_QWEN35MOE_SHARED_EXPERT_LAYER_CONFIG_PATTERN = (
    "model.language_model.layers.*.mlp.shared_expert.(gate_proj|up_proj|down_proj)"
)
_QWEN35MOE_EXPERT_UP_GATE_LAYER_CONFIG_PATTERN = "model.language_model.layers.*.mlp.experts.*.(gate_proj|up_proj)"
_QWEN35MOE_EXPERT_DOWN_LAYER_CONFIG_PATTERN = "model.language_model.layers.*.mlp.experts.*.down_proj"


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
        algorithm = str(quant_cfg.get("algorithm") or "autoround")
        artifact_format = self._resolve_artifact_format(quant_cfg)

        if algorithm == "existing_hf":
            existing_hf_model_dir = quant_cfg.get("existing_hf_model_dir")
            if not existing_hf_model_dir:
                raise ValueError("quant.algorithm='existing_hf' requires quant.existing_hf_model_dir")
            return QuantResult(
                hf_model_dir=self.hf_model_dir,
                quanted_model_dir=self._normalize_path(existing_hf_model_dir),
            )

        if algorithm == "autoround" and artifact_format == "gptqmodel_hf":
            result = self._quant_autoround_gptqmodel_hf(
                output_dir=output_dir,
                device=device,
                quant_cfg=quant_cfg,
            )
            return result

        raise NotImplementedError(
            "Qwen35Workflow.quant supports quant=None, "
            "quant.algorithm='existing_hf', or "
            "quant.algorithm='autoround' with artifact_format/output_format='gptqmodel_hf'. "
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

    def _quant_autoround_gptqmodel_hf(
        self,
        output_dir: str,
        device: str,
        quant_cfg: Mapping[str, Any],
    ) -> QuantResult:
        try:
            from auto_round import AutoRound  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "Qwen35Workflow.quant with algorithm='autoround' and "
                "artifact_format/output_format='gptqmodel_hf' requires the optional AutoRound package. "
                "Install/provide AutoRound locally, or use quant=None for base export, or use "
                "quant.algorithm='existing_hf' with quant.existing_hf_model_dir for an externally "
                "quantized HF/GPTQModel model."
            ) from exc

        save_path = self._autoround_save_path(output_dir, quant_cfg)
        runtime_cfg = self._get_quant_section(quant_cfg, "runtime")
        calibration_cfg = self._get_quant_section(quant_cfg, "calibration")

        bits = int(quant_cfg.get("bits", 4))
        group_size = int(quant_cfg.get("group_size", 64))
        batch_size = int(runtime_cfg.get("batch_size", quant_cfg.get("batch_size", 8)))
        seqlen = int(calibration_cfg.get("seqlen", quant_cfg.get("seqlen", quant_cfg.get("seq_len", 2048))))
        nsamples = int(calibration_cfg.get("nsamples", quant_cfg.get("nsamples", quant_cfg.get("samples", 128))))
        trust_remote_code = self._as_bool(
            runtime_cfg.get("trust_remote_code", quant_cfg.get("trust_remote_code", True))
        )
        iters = int(quant_cfg.get("iters", 200))
        sym = self._as_bool(quant_cfg.get("sym", True))
        seed = int(runtime_cfg.get("seed", quant_cfg.get("seed", self.seed)))
        quant_nontext_module = self._as_bool(
            quant_cfg.get("quant_nontext_module", runtime_cfg.get("quant_nontext_module", False))
        )

        dataset = self._format_autoround_dataset(calibration_cfg)
        autoround_format = str(quant_cfg.get("autoround_format", "auto_gptq"))

        autoround_kwargs = {
            "model": self.hf_model_dir,
            "bits": bits,
            "group_size": group_size,
            "sym": sym,
            "batch_size": batch_size,
            "seqlen": seqlen,
            "nsamples": nsamples,
            "iters": iters,
            "dataset": dataset,
            "device": device,
            "trust_remote_code": trust_remote_code,
            "seed": seed,
            "quant_nontext_module": quant_nontext_module,
        }
        self._add_optional_autoround_kwargs(autoround_kwargs, quant_cfg, runtime_cfg, group_size)
        try:
            autoround = AutoRound(**autoround_kwargs)
        except TypeError as exc:
            raise NotImplementedError(
                "Installed AutoRound does not expose the expected AutoRound(model=..., ...) "
                "API used by Qwen35Workflow. Accepted workflow config: algorithm='autoround', "
                "artifact_format/output_format='gptqmodel_hf', group_size=64."
            ) from exc
        self._autoround_quantize_and_save(autoround, save_path, autoround_format)

        return QuantResult(hf_model_dir=self.hf_model_dir, quanted_model_dir=save_path)

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @classmethod
    def _add_optional_autoround_kwargs(
        cls,
        autoround_kwargs: dict[str, Any],
        quant_cfg: Mapping[str, Any],
        runtime_cfg: Mapping[str, Any],
        group_size: int,
    ) -> None:
        for key in ("device_map", "gradient_accumulate_steps"):
            if key in runtime_cfg:
                autoround_kwargs[key] = runtime_cfg[key]
        if "low_gpu_mem_usage" in runtime_cfg:
            autoround_kwargs["low_gpu_mem_usage"] = cls._as_bool(runtime_cfg["low_gpu_mem_usage"])

        layer_config = cls._build_autoround_layer_config(quant_cfg, group_size)
        if layer_config:
            autoround_kwargs["layer_config"] = layer_config

    @staticmethod
    def _build_autoround_layer_config(quant_cfg: Mapping[str, Any], group_size: int) -> dict[str, dict[str, Any]]:
        raw_layer_config = quant_cfg.get("layer_config")
        if raw_layer_config is not None:
            if not isinstance(raw_layer_config, Mapping):
                raise TypeError("quant.layer_config must be a mapping when provided")
            normalized_layer_config: dict[str, dict[str, Any]] = {}
            for name, value in raw_layer_config.items():
                if not isinstance(value, Mapping):
                    raise TypeError(f"quant.layer_config[{name!r}] must be a mapping")
                normalized_layer_config[str(name)] = dict(value)
            return normalized_layer_config

        moe_cfg = quant_cfg.get("moe", {})
        if moe_cfg is None:
            return {}
        if not isinstance(moe_cfg, Mapping):
            raise TypeError("quant.moe must be a mapping when provided")

        layer_config: dict[str, dict[str, Any]] = {}
        attn_bits = moe_cfg.get("attn_bits")
        if attn_bits is not None:
            value = {"bits": int(attn_bits), "group_size": group_size}
            for pattern in _QWEN35MOE_ATTN_LAYER_CONFIG_PATTERNS:
                layer_config[pattern] = value

        shared_expert_bits = moe_cfg.get("shared_expert_bits")
        if shared_expert_bits is not None:
            layer_config[_QWEN35MOE_SHARED_EXPERT_LAYER_CONFIG_PATTERN] = {
                "bits": int(shared_expert_bits),
                "group_size": group_size,
            }

        expert_bits = moe_cfg.get("expert_bits")
        expert_up_gate_bits = moe_cfg.get("expert_up_gate_bits", expert_bits)
        expert_down_bits = moe_cfg.get("expert_down_bits", expert_bits)
        if expert_up_gate_bits is not None:
            layer_config[_QWEN35MOE_EXPERT_UP_GATE_LAYER_CONFIG_PATTERN] = {
                "bits": int(expert_up_gate_bits),
                "group_size": group_size,
            }
        if expert_down_bits is not None:
            layer_config[_QWEN35MOE_EXPERT_DOWN_LAYER_CONFIG_PATTERN] = {
                "bits": int(expert_down_bits),
                "group_size": group_size,
            }
        return layer_config

    @staticmethod
    def _autoround_quantize_and_save(autoround: Any, save_path: str, autoround_format: str) -> None:
        if hasattr(autoround, "save_quantized"):
            if hasattr(autoround, "quantize"):
                autoround.quantize()
            autoround.save_quantized(save_path, format=autoround_format)
            return
        if hasattr(autoround, "quantize_and_save"):
            autoround.quantize_and_save(save_path, format=autoround_format)
            return
        raise AttributeError("AutoRound instance must expose save_quantized() or quantize_and_save()")

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
    def _get_quant_section(quant_cfg: Mapping[str, Any], section: str) -> Mapping[str, Any]:
        value = quant_cfg.get(section, {})
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError(f"quant.{section} must be a mapping when provided")
        return value

    @staticmethod
    def _format_autoround_dataset(calibration_cfg: Mapping[str, Any]) -> str:
        dataset = str(calibration_cfg.get("dataset", "NeelNanda/pile-10k"))
        split = calibration_cfg.get("split")
        if split:
            return f"{dataset}:{split}"
        return dataset

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

    def _autoround_save_path(self, output_dir: str, quant_cfg: Mapping[str, Any]) -> str:
        configured = quant_cfg.get("save_path") or quant_cfg.get("output_dir")
        if configured:
            return self._normalize_path(configured)
        return self._normalize_path(Path(output_dir) / f"{Path(self.hf_model_dir).name}-autoround-gptqmodel")


__all__ = ["Qwen35Workflow"]
