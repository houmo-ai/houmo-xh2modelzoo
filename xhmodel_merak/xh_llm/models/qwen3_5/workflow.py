import gc
import json
import os
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from ...workflows.base import BaseLLMWorkflow
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


def finalize_qwen35_merak_runtime_config(export_result: ExportResult) -> Path:
    """Write the vLLM-Merak entry config next to the exported HMONNX.

    ``export_hmonnx`` owns the graph and golden metadata while the workflow
    owns the release layout.  Keeping this finalization here ensures normal,
    MTP, and DFlash exports all receive the same runtime entry point instead
    of relying on a later manual copy.
    """

    meta_path = Path(BaseLLMWorkflow._find_golden_meta_file(export_result))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise TypeError(f"Qwen3.5 golden metadata must be an object: {meta_path}")

    model_config = meta.get("model_config")
    if not isinstance(model_config, dict):
        raise TypeError(f"Qwen3.5 golden metadata must contain an object field 'model_config': {meta_path}")

    # Page attention is a runtime capability of the exported target graph.
    # Derive it from the finalized metadata (which includes CLI overrides),
    # rather than from the workflow's original YAML.
    explicit_page_attention = model_config.get("enable_page_attention")
    if isinstance(explicit_page_attention, bool):
        enable_page_attention = explicit_page_attention
    else:
        flash_attention = model_config.get("flash_attention")
        enable_page_attention = bool(isinstance(flash_attention, dict) and flash_attention.get("enable") is True)

    config = {
        "architectures": ["MerakForCausalLM"],
        "config_format": "merak_llm",
        "load_format": "merak_llm",
        "xh_model": {
            "model_type": "hmonnx",
            "meta_info": meta_path.name,
        },
        "enable_page_attention": enable_page_attention,
        "model_type": "merak_llm",
    }
    config_path = meta_path.parent / "merak_config.json"
    temporary_path = config_path.with_name(f".{config_path.name}.tmp")
    temporary_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(config_path)
    return config_path


class Qwen35Workflow(BaseLLMWorkflow):
    """Merak HMONNX workflow for Qwen3.5/Qwen3.6 dense, MoE, and visual exports."""

    expected_model_cls_names = _QWEN35_MODEL_CLS_NAMES
    expected_model_config_cls_names = _QWEN35_MODEL_CONFIG_CLS_NAMES
    family_label = "Qwen3.5/Qwen3.6"

    @classmethod
    def from_config(
        cls,
        model_dir: str,
        config_path: str,
        seed: int = 1024,
        debug: bool = False,
    ) -> "Qwen35Workflow":
        return cls(model_dir=model_dir, config_path=config_path, seed=seed, debug=debug)

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        quant_cfg = workflow_config.quant
        if quant_cfg is None:
            return QuantResult(
                raw_model_dir=self.model_dir,
                skipped=True,
            )

        self._validate_group_size(quant_cfg)
        algorithm = str(quant_cfg.get("algorithm") or "gptqmodel").lower().replace("-", "_")
        artifact_format = self._resolve_artifact_format(quant_cfg)
        if artifact_format != "gptqmodel_hf":
            raise ValueError(
                f"Qwen35Workflow.quant requires artifact_format/output_format='gptqmodel_hf'; got {artifact_format!r}."
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
            "Qwen35Workflow.quant supports quant=None, "
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
        self._validate_lora_export(config_overrides)
        self._validate_export_model(config_overrides)
        export_result = super().export(
            quant_result=quant_result,
            output_dir=output_dir,
            device=device,
            config_overrides=config_overrides,
        )
        finalize_qwen35_merak_runtime_config(export_result)
        return export_result

    def _validate_lora_export(self, config_overrides: Mapping[str, Any] | None) -> None:
        """Fail before creating an output directory or loading the base model."""

        from .lora import inspect_lora_adapters

        workflow_config = self.workflow_config.with_overrides(config_overrides)
        model_cfg = workflow_config.export["model"]
        lora_cfg = model_cfg.get("lora")
        if lora_cfg is not None and str(model_cfg.get("model_type", "")).endswith("_visual"):
            raise ValueError("Qwen3.5 ViT/visual export does not support LoRA")
        visual_cfg = model_cfg.get("visual_config")
        if isinstance(visual_cfg, Mapping) and visual_cfg.get("lora") is not None:
            raise ValueError(
                "Qwen3.5 visual_config does not support LoRA; configure language-model adapters "
                "under export.model.lora only"
            )
        inspect_lora_adapters(lora_cfg)

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any,
        *,
        auto_offload: bool | None = None,
        device_map: list[Any] | None = None,
        use_v2: bool | None = None,
    ) -> str:
        from xhquant.api import get_xhquant_logger

        root_meta_file = self._find_golden_meta_file(export_result)
        messages = self.build_input_message(input_messages)
        logger = get_xhquant_logger()
        # Preserve the established single-device path unless the caller
        # explicitly requests a multi-device map.  The latter exists for
        # 122B golden generation; silently consuming every visible GPU for a
        # 9B/27B run would be an unexpected global behavior change.
        resolved_device_map = list(device_map) if device_map is not None else None
        cuda_device_count = sum(self._is_cuda_device_entry(entry) for entry in (resolved_device_map or ()))
        if auto_offload is None:
            auto_offload = cuda_device_count > 1
        if use_v2 is None:
            use_v2 = auto_offload or self._env_flag_enabled("ENABLE_HMINFERENCE_V2")
        if auto_offload and not use_v2:
            raise ValueError("Qwen3.5 golden auto-offload requires HMONNXInferenceV2.")

        logger.info(
            f"Qwen3.5 golden runtime: device_map={resolved_device_map}, "
            f"auto_offload={auto_offload}, inference_v2={use_v2}"
        )
        with self._hmonnx_v2_scope(use_v2):
            for meta_file in self._collect_golden_meta_files(root_meta_file):
                logger.info(f"Dumping Qwen3.5 golden for model view: {meta_file}")
                try:
                    self._dump_golden_for_meta(
                        meta_file,
                        device,
                        messages,
                        logger=logger,
                        auto_offload=auto_offload,
                        device_map=resolved_device_map,
                    )
                finally:
                    # Collect after the per-model call frame has unwound.  Its
                    # context managers and HF-compatible wrapper may otherwise
                    # keep the previous HMONNX model alive while loading LoRA.
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        return root_meta_file

    def _dump_golden_for_meta(
        self,
        meta_file: str,
        device: str,
        messages: list[dict[str, Any]],
        *,
        logger: Any,
        auto_offload: bool = False,
        device_map: list[Any] | None = None,
    ) -> None:
        from transformers import TextStreamer

        from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
        from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler

        load_kwargs: dict[str, Any] = {"enable_golden": True}
        if auto_offload:
            load_kwargs["enable_auto_offload"] = True
        if device_map is not None:
            load_kwargs["device_map"] = device_map
        else:
            requested_device = torch.device(device)
            if (
                requested_device.type == "cuda"
                and requested_device.index is None
            ):
                requested_device = torch.device(
                    f"cuda:{torch.cuda.current_device()}"
                )
            # HMONNX otherwise treats every visible GPU as an implicit device
            # map.  A normal golden run must honor --device and remain
            # single-device; only --golden-device-map opts into sharding.
            load_kwargs["device_map"] = [requested_device]
        hmonnx_model = AutoLLMHONNXModel.from_pretrained(
            meta_file,
            **load_kwargs,
        )
        try:
            if auto_offload and hasattr(hmonnx_model, "enable_auto_offload"):
                hmonnx_model.enable_auto_offload = True
            runtime_device = getattr(hmonnx_model, "device", device)
            if self._messages_have_image(messages):
                processor = hmonnx_model.get_tf_processor()
                tokenizer = processor.tokenizer
                model_inputs = processor.apply_chat_template(messages).to(runtime_device)
                decode = processor.batch_decode
            else:
                tokenizer = hmonnx_model.get_tokenizer()
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
                model_inputs = tokenizer([text], return_tensors="pt", truncation=True).to(runtime_device)
                decode = tokenizer.batch_decode

            streamer = TextStreamer(tokenizer)
            hmonnx_model.to(runtime_device)
            hmonnx_model.enable_golden = True
            logger.warning("Golden outputs should be generated in aligned precision for stability.")

            memory_devices = device_map if device_map and len(device_map) > 1 else runtime_device
            inference_context = (
                LLMInferenceContextManager(
                    hmonnx_model,
                    devices=[runtime_device],
                )
                if auto_offload
                else LLMInferenceContextManager(hmonnx_model)
            )
            contexts = [
                TimeProfiler("hmonnx_generate_golden", logger),
                MemoryTracker(device=memory_devices, name="generate_golden", logger=logger),
                inference_context,
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
        finally:
            del hmonnx_model

        self._dump_spec_decode_golden(
            meta_file,
            device,
            messages,
            logger=logger,
            auto_offload=auto_offload,
            device_map=device_map,
        )

    @staticmethod
    def _is_cuda_device_entry(device: Any) -> bool:
        if isinstance(device, int):
            return True
        if isinstance(device, torch.device):
            return device.type == "cuda"
        normalized = str(device).strip().lower()
        return normalized == "cuda" or normalized.startswith("cuda:") or normalized.isdigit()

    @staticmethod
    def _env_flag_enabled(name: str) -> bool:
        return os.getenv(name, "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    @contextmanager
    def _hmonnx_v2_scope(enabled: bool):
        if not enabled:
            yield
            return

        env_name = "ENABLE_HMINFERENCE_V2"
        previous = os.environ.get(env_name)
        os.environ[env_name] = "1"
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = previous

    @staticmethod
    def _collect_golden_meta_files(root_meta_file: str) -> list[str]:
        root_meta_path = Path(root_meta_file)
        root_meta = json.loads(root_meta_path.read_text(encoding="utf-8"))
        adapters = root_meta.get("lora_adapters", [])
        if not isinstance(adapters, list):
            raise TypeError("golden_meta_info.json field 'lora_adapters' must be a list")

        meta_files = [str(root_meta_path)]
        for entry in adapters:
            if not isinstance(entry, Mapping) or not isinstance(entry.get("meta_file"), str):
                raise TypeError("Every lora_adapters entry must contain a string meta_file")
            child_meta_path = root_meta_path.parent / entry["meta_file"]
            if not child_meta_path.is_file():
                raise FileNotFoundError(f"LoRA golden metadata does not exist: {child_meta_path}")
            meta_files.append(str(child_meta_path))
        return meta_files

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
        export_cfg = workflow_config.build_export_dict()
        # Compatibility for this validation path only. New export config
        # finalization should go through BaseLLMWorkflow._build_export_config().
        export_cfg["model"]["hf_model"] = self.model_dir
        model_cls = get_model_class(export_cfg["model"])
        if model_cls is None:
            return
        model_cls_name = model_cls.__name__
        config_cls = getattr(model_cls, "CONFIG_CLS", None)
        config_cls_name = getattr(config_cls, "__name__", None)
        if (
            model_cls_name not in self.expected_model_cls_names
            or config_cls_name not in self.expected_model_config_cls_names
        ):
            raise TypeError(
                f"{type(self).__name__} only supports {self.family_label} model classes; "
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
            model_dir=self.model_dir,
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
            model_dir=self.model_dir,
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
    def _messages_have_image(messages: list[dict[str, Any]]) -> bool:
        for message in messages:
            content = message.get("content") if isinstance(message, Mapping) else None
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, Mapping) and item.get("type") == "image":
                        return True
        return False

    def _dump_spec_decode_golden(
        self,
        meta_file: str,
        device: str,
        messages: list[dict[str, Any]],
        *,
        logger: Any,
        auto_offload: bool = False,
        device_map: list[Any] | None = None,
    ) -> None:
        mode = self._spec_decode_mode(meta_file)
        if mode not in {"mtp", "dflash"}:
            return

        from .hmonnx_validation import spec_decode_generate

        prompt = self._messages_text_prompt(messages)
        max_new_tokens = self._spec_decode_golden_max_new_tokens(meta_file, mode)
        logger.info(f"Dumping {mode} draft golden via spec_decode_generate (max_new_tokens={max_new_tokens}).")
        auto_offload_max_memory = self._auto_offload_max_memory_json(device_map) if auto_offload else None
        result = spec_decode_generate(
            meta_file=meta_file,
            prompt=prompt,
            device=device,
            exec_device=device,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            enable_thinking=True,
            warmup_runs=0,
            benchmark_runs=1,
            golden=True,
            disable_auto_offload=not auto_offload,
            auto_offload_max_memory=auto_offload_max_memory,
        )
        logger.info(f"{'-' * 20} Spec draft golden output {'-' * 20}")
        logger.info(result.output_text)

    @classmethod
    def _auto_offload_max_memory_json(
        cls,
        device_map: list[Any] | None,
    ) -> str | None:
        """Restrict standalone draft golden offload to explicit CUDA devices."""

        if not device_map:
            return None
        device_indices: list[int] = []
        for entry in device_map:
            if not cls._is_cuda_device_entry(entry):
                continue
            if isinstance(entry, int):
                index = entry
            else:
                device = torch.device(f"cuda:{entry}" if str(entry).isdigit() else entry)
                index = int(device.index) if device.index is not None else int(torch.cuda.current_device())
            if index not in device_indices:
                device_indices.append(index)
        if not device_indices:
            raise ValueError("Qwen3.5 golden auto-offload requires at least one CUDA device in device_map")

        # AutoOffloadGraphModel follows Accelerate's max_memory convention.
        # Use current free memory with headroom, and omit every unrequested
        # GPU so an explicit 122B golden run cannot spill onto other users'
        # devices.
        max_memory: dict[str, int] = {}
        for index in device_indices:
            free_bytes, _ = torch.cuda.mem_get_info(index)
            max_memory[str(index)] = max(int(free_bytes * 0.9), 1)
        return json.dumps(max_memory)

    @staticmethod
    def _spec_decode_mode(meta_file: str) -> str | None:
        meta = json.loads(Path(meta_file).read_text(encoding="utf-8"))
        spec_decode = meta.get("spec_decode")
        if isinstance(spec_decode, Mapping):
            mode = spec_decode.get("mode") or meta.get("spec_decode_mode")
        else:
            mode = meta.get("spec_decode_mode")
        return str(mode).lower() if mode else None

    @staticmethod
    def _spec_decode_golden_max_new_tokens(meta_file: str, mode: str) -> int:
        if mode != "dflash":
            return 2

        meta = json.loads(Path(meta_file).read_text(encoding="utf-8"))
        spec_decode = meta.get("spec_decode")
        if not isinstance(spec_decode, Mapping):
            return 2

        num_draft_tokens = spec_decode.get("num_draft_tokens")
        if num_draft_tokens is None:
            block_size = spec_decode.get("block_size")
            if block_size is not None:
                num_draft_tokens = max(int(block_size) - 1, 0)
        if num_draft_tokens is None:
            return 2

        # DFlash context_decode runs in the post-verify path. If all draft
        # tokens are accepted, the generate loop reaches that path only after
        # initial_token + all draft tokens have been emitted.
        return max(2, int(num_draft_tokens) + 2)

    @staticmethod
    def _messages_text_prompt(messages: list[dict[str, Any]]) -> str:
        text_parts: list[str] = []
        for message in messages:
            content = message.get("content") if isinstance(message, Mapping) else None
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, Mapping) and item.get("type") == "text":
                        text = item.get("text")
                        if isinstance(text, str):
                            text_parts.append(text)
        prompt = "\n".join(part for part in text_parts if part).strip()
        if not prompt:
            raise ValueError("Qwen3.5 spec decode golden requires text content in input_messages.")
        return prompt


__all__ = ["Qwen35Workflow"]
