import copy
import os
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...workflows.base import BaseHMONNXWorkflow
from ...workflows.config import WorkflowConfig
from ...workflows.result import ExportResult, QuantResult
from .export_plan import (
    DEFAULT_IMAGE_VISUAL_MAX_PATCHES,
    DEFAULT_IMAGE_VISUAL_SEQ_LENGTH,
    DEFAULT_QUANT_TYPE,
    DEFAULT_VIDEO_VISUAL_MAX_PATCHES,
    DEFAULT_VIDEO_VISUAL_SEQ_LENGTH,
    REQUIRED_CONTEXT_MAX_LENGTH,
    REQUIRED_INPUT_SEQUENCE_LENGTH,
    Gemma4SeriesExportPlan,
    build_gemma4_series_export_plan,
)
from .quant_adapter import (
    DEFAULT_AUTOROUND_DATASET,
    DEFAULT_DENSE_CALIBRATION_JSONL,
    DEFAULT_MOE_CALIBRATION_JSONL,
)


_GEMMA4_MODEL_CLS_NAMES = {
    # Unified public Gemma4 Series entry for E4B, 31B dense, and 26B-A4B.
    "XHGemma4SeriesModel",
    "XHGemma4Model",
    # Compatibility-only aliases for existing MoE configs/artifacts. These are
    # accepted by the unified workflow but are not a separate public workflow API.
    "XHGemma4MoeWithMaskModel",
    "XHGemma4VisionModel",
    "XHGemma4MoeVisualModel",
}
_GEMMA4_MODEL_CONFIG_CLS_NAMES = {
    "XHGemma4SeriesModelConfig",
    "XHGemma4ModelConfig",
    "XHGemma4VisualConfig",
    "XHGemma4MoeWithMaskConfig",
    "XHGemma4MoeVisualConfig",
}
_GEMMA4_TOP_LEVEL_MODEL_TYPES = {
    "Gemma4ForConditionalGeneration",
    # Compatibility-only legacy top-level alias. New YAML/demo configs should
    # use Gemma4ForConditionalGeneration. Do not expose a separate MoE workflow.
    "Gemma4ForConditionalGeneration_with_mask",
}
_GEMMA4_GOLDEN_MIN_TEXT_TOKENS = 1025
_GEMMA4_RECOMMENDED_CONFIGS = {
    "e2b": "configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml",
    "e4b": "configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml",
    "31b": "configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_full.yaml",
    "26b-a4b": "configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml",
}
_GEMMA4_QUANT_TEMPLATE: dict[str, Any] = {
    "algorithm": "gptqmodel",
    "method": "gptq",
    "preset": "full_multimodal",
    "rotation": None,
    "artifact_format": "gptqmodel_hf",
    "output_format": "gptqmodel_hf",
    "bits": 4,
    "group_size": 64,
    "sym": True,
    "iters": 200,
    "seed": 42,
    "quant_nontext_module": False,
    "calibration": {
        "jsonl": DEFAULT_DENSE_CALIBRATION_JSONL,
        "text_key": "text",
        "nsamples": 256,
        "seqlen": 512,
    },
    "runtime": {
        "batch_size": 1,
        "device_map": "auto",
        "trust_remote_code": True,
        "offload_to_disk": False,
    },
    "validation": {
        "check_quant_text_demo": True,
        "check_quant_image_demo": True,
        "check_quant_video_demo": False,
        "check_quant_audio_demo": False,
    },
}
_GEMMA4_AUTOROUND_MODE1_QUANT_TEMPLATE: dict[str, Any] = {
    "algorithm": "gptqmodel",
    "method": "autoround",
    "preset": "mode1",
    "rotation": None,
    "artifact_format": "gptqmodel_hf",
    "output_format": "gptqmodel_hf",
    "bits": 4,
    "group_size": 64,
    "sym": True,
    "iters": 200,
    "seed": 42,
    "format": "auto_gptq",
    "calibration": {
        "dataset": DEFAULT_AUTOROUND_DATASET,
        "nsamples": 128,
        "seqlen": 2048,
    },
    "runtime": {
        "batch_size": 8,
        "trust_remote_code": True,
    },
}
_GEMMA4_AUTOROUND_MOE_MODE1_QUANT_TEMPLATE: dict[str, Any] = {
    "algorithm": "gptqmodel",
    "method": "autoround",
    "preset": "mode1",
    "rotation": None,
    "artifact_format": "gptqmodel_hf",
    "output_format": "gptqmodel_hf",
    "bits": 4,
    "group_size": 64,
    "sym": True,
    "iters": 200,
    "seed": 42,
    "format": "auto_gptq",
    "calibration": {
        "dataset": DEFAULT_AUTOROUND_DATASET,
        "nsamples": 128,
        "seqlen": 2048,
    },
    "runtime": {
        "batch_size": 8,
        "dtype": "bfloat16",
        "trust_remote_code": True,
    },
    "validation": {
        "prompt": "你是谁",
        "max_new_tokens": 128,
    },
}
_GEMMA4_EXPORT_MODEL_TEMPLATE: dict[str, Any] = {
    "chip_arch": "XH2a",
    "model_type": "Gemma4ForConditionalGeneration",
    "hf_model": None,
    "model_name": "auto",
    "context_max_length": REQUIRED_CONTEXT_MAX_LENGTH,
    "prefill_chunk_length": REQUIRED_INPUT_SEQUENCE_LENGTH,
    "use_cache": True,
    "num_logits_to_keep": 1,
    "sliding_kv_cache_input_mode": "slice_window",
    "quant_scheme": {
        "quant_type": DEFAULT_QUANT_TYPE,
        "ops": {},
    },
    "visual_config": {
        "export_mode": "padded",
        "image_seq_length": DEFAULT_IMAGE_VISUAL_SEQ_LENGTH,
        "max_patches": DEFAULT_IMAGE_VISUAL_MAX_PATCHES,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "input_modality": "image",
        "quant_scheme": {
            "quant_type": DEFAULT_QUANT_TYPE,
            "ops": {},
        },
    },
    "video_visual_config": {
        "export_mode": "padded",
        "image_seq_length": DEFAULT_VIDEO_VISUAL_SEQ_LENGTH,
        "max_patches": DEFAULT_VIDEO_VISUAL_MAX_PATCHES,
        "patch_size": 16,
        "pooling_kernel_size": 3,
        "input_modality": "video",
        "quant_scheme": {
            "quant_type": DEFAULT_QUANT_TYPE,
            "ops": {},
        },
    },
    "only_first_block": False,
}
_GEMMA4_EXPORT_NAMING_TEMPLATE: dict[str, Any] = {
    "family": "gemma4",
    "variant": "e4b",
    "profile": "full",
}


def list_recommended_configs() -> dict[str, str]:
    """Return topology-named Gemma4 Series workflow YAMLs for the public API."""

    return dict(_GEMMA4_RECOMMENDED_CONFIGS)


def get_quant_config_help() -> str:
    return (
        "Gemma4 Series quant config defaults to the GPTQModel Gemma4 recipe "
        "(algorithm='gptqmodel', method='gptq') producing GPTQModel-compatible HF artifacts. "
        f"Dense defaults use IVSG calibration JSONL ({DEFAULT_DENSE_CALIBRATION_JSONL}); "
        f"26B-A4B MoE defaults use EBSS calibration JSONL ({DEFAULT_MOE_CALIBRATION_JSONL}) "
        "and routing bypass so every expert receives calibration activations. "
        "Dense E2B/E4B/31B checkpoints can alternatively use algorithm='gptqmodel', method='autoround', preset='mode1', "
        "which wraps third_party/auto-round/scripts_gemma4 LLM-only W4G64 no-rotation quantization "
        f"with dataset={DEFAULT_AUTOROUND_DATASET!r}; "
        "26B-A4B with the same preset wraps scripts_gemma4_moe/quantize_moe.py. "
        "Use config_overrides={'quant': None} only for explicit base-model validation, or replace "
        "the quant block with {'algorithm': 'existing_hf', 'artifact_format': 'gptqmodel_hf', "
        "'existing_hf_model_dir': ...} when a quantized HF directory already exists. "
        "The public group_size is fixed at 64."
    )


def get_export_config_help() -> str:
    return (
        "Gemma4 Series export configs for E4B, 31B, and 26B-A4B all use "
        "model_type='Gemma4ForConditionalGeneration' and XHGemma4ModelConfig. "
        "26B-A4B MoE routing is detected internally from the HF config; callers should not select "
        "a separate _with_mask public model type. Visual export follows the padded ViT contract: "
        "image VIT uses [1,2520,768]/[1,280,9], while video VIT is exported separately with "
        "[1,630,768]/[1,70,9] so video frames are not padded to the image VIT size."
    )


def _dump_yaml(path: str | os.PathLike[str], data: Mapping[str, Any]) -> str:
    import yaml

    path = str(path)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(dict(data), sort_keys=False), encoding="utf-8")
    return path


def dump_quant_config_template(path: str | os.PathLike[str]) -> str:
    return _dump_yaml(path, {"quant": copy.deepcopy(_GEMMA4_QUANT_TEMPLATE)})


def dump_autoround_mode1_quant_config_template(path: str | os.PathLike[str]) -> str:
    return _dump_yaml(path, {"quant": copy.deepcopy(_GEMMA4_AUTOROUND_MODE1_QUANT_TEMPLATE)})


def dump_autoround_moe_mode1_quant_config_template(path: str | os.PathLike[str]) -> str:
    return _dump_yaml(path, {"quant": copy.deepcopy(_GEMMA4_AUTOROUND_MOE_MODE1_QUANT_TEMPLATE)})


def dump_export_config_template(path: str | os.PathLike[str]) -> str:
    return _dump_yaml(
        path,
        {
            "export": {
                "naming": copy.deepcopy(_GEMMA4_EXPORT_NAMING_TEMPLATE),
                "model": copy.deepcopy(_GEMMA4_EXPORT_MODEL_TEMPLATE),
            }
        },
    )


def quant(
    *,
    hf_model_dir: str,
    config_path: str,
    output_dir: str,
    device: str,
    config_overrides: Mapping[str, Any] | None = None,
    seed: int = 1024,
    debug: bool = False,
) -> QuantResult:
    workflow = Gemma4SeriesWorkflow.from_config(hf_model_dir, config_path, seed=seed, debug=debug)
    return workflow.quant(output_dir=output_dir, device=device, config_overrides=config_overrides)


def export(
    *,
    hf_model_dir: str,
    config_path: str,
    quant_result: QuantResult,
    output_dir: str,
    device: str,
    config_overrides: Mapping[str, Any] | None = None,
    seed: int = 1024,
    debug: bool = False,
) -> ExportResult:
    workflow = Gemma4SeriesWorkflow.from_config(hf_model_dir, config_path, seed=seed, debug=debug)
    return workflow.export(
        quant_result=quant_result,
        output_dir=output_dir,
        device=device,
        config_overrides=config_overrides,
    )


class Gemma4SeriesWorkflow(BaseHMONNXWorkflow):
    """Merak HMONNX workflow for the unified Gemma4 public model API."""

    list_recommended_configs = staticmethod(list_recommended_configs)
    get_quant_config_help = staticmethod(get_quant_config_help)
    get_export_config_help = staticmethod(get_export_config_help)
    dump_quant_config_template = staticmethod(dump_quant_config_template)
    dump_autoround_mode1_quant_config_template = staticmethod(dump_autoround_mode1_quant_config_template)
    dump_autoround_moe_mode1_quant_config_template = staticmethod(dump_autoround_moe_mode1_quant_config_template)
    dump_export_config_template = staticmethod(dump_export_config_template)

    @classmethod
    def from_config(
        cls,
        hf_model_dir: str,
        config_path: str,
        seed: int = 1024,
        debug: bool = False,
    ) -> "Gemma4SeriesWorkflow":
        return cls(hf_model_dir=hf_model_dir, config_path=config_path, seed=seed, debug=debug)

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self._workflow_config_for_quant(config_overrides)
        quant_cfg = workflow_config.quant
        if quant_cfg is None:
            if not self._is_explicit_base_quant_override(config_overrides):
                raise ValueError(
                    "Gemma4SeriesWorkflow default workflow YAML must configure quantization. "
                    "Use config_overrides={'quant': None} only for explicit base-model validation."
                )
            return QuantResult(
                hf_model_dir=self.hf_model_dir,
                skipped=True,
            )

        algorithm = str(quant_cfg.get("algorithm") or "gptqmodel").lower()
        artifact_format = self._resolve_artifact_format(quant_cfg)
        if artifact_format != "gptqmodel_hf":
            raise ValueError(
                "Gemma4SeriesWorkflow.quant requires artifact_format/output_format='gptqmodel_hf'; "
                f"got {artifact_format!r}."
            )

        if algorithm == "existing_hf":
            existing_hf_model_dir = quant_cfg.get("existing_hf_model_dir")
            if not existing_hf_model_dir:
                raise ValueError("quant.algorithm='existing_hf' requires quant.existing_hf_model_dir")
            existing_hf_model_dir = self._require_existing_directory(
                existing_hf_model_dir,
                field="quant.existing_hf_model_dir",
            )
            return QuantResult(
                hf_model_dir=self.hf_model_dir,
                quanted_model_dir=existing_hf_model_dir,
            )

        method = str(quant_cfg.get("method") or "").lower().replace("-", "_")
        if algorithm in {"autoround", "auto-round", "auto_round"} or (
            algorithm == "gptqmodel" and method in {"autoround", "auto_round"}
        ):
            return self._quant_autoround_mode1(
                output_dir=output_dir,
                device=device,
                quant_cfg=quant_cfg,
                export_model_cfg=workflow_config.export["model"],
                effective_config_file=workflow_config.source,
            )

        if algorithm in {"gptqmodel", "gptq"}:
            return self._quant_gptqmodel_recipe(
                output_dir=output_dir,
                device=device,
                quant_cfg=quant_cfg,
                export_model_cfg=workflow_config.export["model"],
                effective_config_file=workflow_config.source,
            )

        raise NotImplementedError(
            "Gemma4SeriesWorkflow.quant supports quant=None, "
            "quant.algorithm='existing_hf', or "
            "quant.algorithm='gptqmodel' with method='gptq' or method='autoround', "
            "or legacy quant.algorithm='gptq'/'autoround', "
            "with artifact_format/output_format='gptqmodel_hf'. "
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
        import json

        from transformers import TextStreamer

        from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
        from xhquant.api import get_xhquant_logger
        from xhquant.utils import ContextManagers, MemoryTracker, TimeProfiler

        meta_file = self._find_golden_meta_file(export_result)
        logger = get_xhquant_logger()
        hmonnx_model = AutoLLMHONNXModel.from_pretrained(meta_file)
        with open(meta_file, encoding="utf-8") as fin:
            meta_info = json.load(fin)

        golden_root = Path(meta_file).parent
        cases = self._build_golden_message_cases(meta_info, input_messages, golden_root=golden_root)
        hmonnx_model.to(device)
        hmonnx_model.enable_golden = True
        logger.warning("Golden outputs should be generated in aligned precision for stability.")

        max_new_tokens = 2
        context_max_length = int(meta_info.get("model_config", {}).get("context_max_length", 0) or 0)
        max_prompt_length = max(context_max_length - max_new_tokens, 1) if context_max_length else None

        def _run_case(case_name: str, messages: list[dict[str, Any]]) -> None:
            logger.info(f"{'-' * 20} Golden case: {case_name} {'-' * 20}")
            hmonnx_model.to(device)
            hmonnx_model.enable_golden = True
            if self._messages_have_multimodal(messages):
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
                model_inputs = tokenizer(
                    [text],
                    return_tensors="pt",
                    truncation=bool(max_prompt_length),
                    max_length=max_prompt_length,
                ).to(device)
                decode = tokenizer.batch_decode

            streamer = TextStreamer(tokenizer)
            contexts = [
                TimeProfiler(f"hmonnx_generate_golden_{case_name}", logger),
                MemoryTracker(device=device, name=f"generate_golden_{case_name}", logger=logger),
                LLMInferenceContextManager(hmonnx_model),
            ]
            try:
                with ContextManagers(contexts):
                    generated_ids = hmonnx_model.generate(
                        **model_inputs,
                        max_new_tokens=max_new_tokens,
                        streamer=streamer,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
            except Exception:
                side_modules = {"visual", "video_visual", "audio"}
                if case_name in side_modules and self._module_has_golden(golden_root, case_name):
                    logger.exception(
                        "Golden case %s failed after %s golden files were materialized; "
                        "continuing to dump remaining side modules.",
                        case_name,
                        case_name,
                    )
                    return
                raise

            generated_ids_trimmed = [
                out_ids[len(in_ids) :]
                for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids, strict=False)
            ]
            output_text = decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            logger.info(f"{'-' * 20} Golden output: {case_name} {'-' * 20}")
            logger.info(f"{output_text}")
            self._normalize_golden_step_dirs(golden_root)

        # Run side modules first so their own step_0 golden directories are
        # materialized, then run the caller's text case last.  That preserves
        # the traditional prefill/decode golden payload while ensuring every
        # exported side HMONNX (visual/video_visual/audio) also has HM-style
        # ``<module>/step_N/*.npy`` golden files.
        for case_name, messages in cases:
            _run_case(case_name, messages)

        return meta_file

    def build_input_message(self, input_messages: Any) -> list[dict[str, Any]]:
        if isinstance(input_messages, list):
            return input_messages
        if isinstance(input_messages, str):
            text = input_messages
            if not text:
                raise ValueError("Gemma4 input text must be a non-empty string")
            return [{"role": "user", "content": text}]
        if not isinstance(input_messages, Mapping):
            raise ValueError("Gemma4 input_messages must be a string, message list, or mapping")

        if "messages" in input_messages:
            messages = input_messages["messages"]
            if not isinstance(messages, list):
                raise ValueError("Gemma4 input_messages['messages'] must be a list")
            return messages

        text = input_messages.get("text")
        if not isinstance(text, str) or not text:
            raise ValueError("Gemma4 input_messages mapping must contain non-empty 'text'")

        media_specs: list[tuple[str, str, Any]] = [
            ("image", "image", input_messages.get("image", input_messages.get("images"))),
            ("video", "video", input_messages.get("video", input_messages.get("videos"))),
            ("audio", "audio", input_messages.get("audio", input_messages.get("audios"))),
        ]
        content = []
        for content_type, key, value in media_specs:
            if value is None:
                continue
            values = value if isinstance(value, list) else [value]
            content.extend({"type": content_type, key: item} for item in values)
        if not content:
            return [{"role": "user", "content": text}]

        content.append({"type": "text", "text": text})
        return [{"role": "user", "content": content}]

    def _build_golden_message_cases(
        self,
        meta_info: Mapping[str, Any],
        input_messages: Any,
        *,
        golden_root: Path | None = None,
    ) -> list[tuple[str, list[dict[str, Any]]]]:
        """Build one golden-generation case per exported HMONNX module family.

        HM golden dumping is owned by each ``HMONNXGoldenInference`` session and
        stored next to that module as ``<module>/step_N/*.npy``.  A text-only
        generate only touches prefill/decode, so Gemma4's side modules need
        explicit modality-bearing generate passes to materialize their own
        golden directories.
        """

        requested_messages = self.build_input_message(input_messages)
        cases: list[tuple[str, list[dict[str, Any]]]] = []

        if self._has_exported_subgraph(meta_info, "visual_config") and not self._module_has_golden(golden_root, "visual"):
            cases.append(("visual", self._messages_for_golden_modality(requested_messages, "image")))
        if self._has_exported_subgraph(meta_info, "video_visual_config") and not self._module_has_golden(
            golden_root, "video_visual"
        ):
            cases.append(("video_visual", self._messages_for_golden_modality(requested_messages, "video")))
        if self._has_exported_subgraph(meta_info, "audio_config") and not self._module_has_golden(golden_root, "audio"):
            cases.append(("audio", self._messages_for_golden_modality(requested_messages, "audio")))

        # Keep the caller-supplied text/generation case last so prefill/decode
        # goldens remain compatible with the original workflow contract.  A
        # short prompt can materialize prefill/decode once and otherwise cause
        # later long-context golden runs to be skipped, so require evidence
        # that the existing text golden crosses the slice-window boundary.
        if not self._text_golden_covers_long_context(golden_root):
            self._remove_module_golden(golden_root, "prefill")
            self._remove_module_golden(golden_root, "decode")
            cases.append(("text", requested_messages))
        return cases

    @staticmethod
    def _module_has_golden(golden_root: Path | None, module_name: str) -> bool:
        if golden_root is None:
            return False
        module_dir = golden_root / module_name
        return module_dir.is_dir() and any(module_dir.glob("step_*/*.npy"))

    @classmethod
    def _text_golden_covers_long_context(cls, golden_root: Path | None) -> bool:
        if golden_root is None:
            return False
        if not (cls._module_has_golden(golden_root, "prefill") and cls._module_has_golden(golden_root, "decode")):
            return False
        return cls._prefill_golden_max_end(golden_root / "prefill") >= _GEMMA4_GOLDEN_MIN_TEXT_TOKENS

    @staticmethod
    def _remove_module_golden(golden_root: Path | None, module_name: str) -> None:
        if golden_root is None:
            return
        module_dir = golden_root / module_name
        if not module_dir.exists():
            return
        for step_dir in module_dir.glob("step_*"):
            if step_dir.is_dir():
                shutil.rmtree(step_dir)

    @staticmethod
    def _prefill_golden_max_end(prefill_dir: Path) -> int:
        import numpy as np

        max_end = 0
        for step_dir in prefill_dir.glob("step_*"):
            valid_files = list(step_dir.glob("*valid_length*_input.npy")) or list(step_dir.glob("valid_length.npy"))
            current_files = list(step_dir.glob("*current_length*_input.npy")) or list(
                step_dir.glob("current_length.npy")
            )
            if not valid_files or not current_files:
                continue
            valid_length = int(np.load(valid_files[0]).reshape(-1)[0])
            current_length = int(np.load(current_files[0]).reshape(-1)[0])
            max_end = max(max_end, valid_length + current_length)
        return max_end

    @classmethod
    def _normalize_golden_step_dirs(cls, golden_root: Path | None) -> None:
        if golden_root is None:
            return
        for module_dir in golden_root.iterdir():
            if module_dir.is_dir():
                cls._normalize_module_step_dirs(module_dir)

    @staticmethod
    def _normalize_module_step_dirs(module_dir: Path) -> None:
        step_dirs = sorted(
            [path for path in module_dir.glob("step_*") if path.is_dir() and path.name.removeprefix("step_").isdigit()],
            key=lambda path: int(path.name.removeprefix("step_")),
        )
        rename_pairs = [(src, module_dir / f"step_{idx}") for idx, src in enumerate(step_dirs)]
        if all(src == dst for src, dst in rename_pairs):
            return
        temp_pairs = []
        for src, _dst in rename_pairs:
            tmp = module_dir / f"{src.name}.tmp_renaming"
            src.rename(tmp)
            temp_pairs.append((tmp, _dst))
        for tmp, dst in temp_pairs:
            if dst.exists():
                raise FileExistsError(f"Cannot normalize golden step directory because target exists: {dst}")
            tmp.rename(dst)

    @classmethod
    def _messages_for_golden_modality(
        cls,
        requested_messages: list[dict[str, Any]],
        modality: str,
    ) -> list[dict[str, Any]]:
        existing = cls._first_media_value(requested_messages, modality)
        media_value = existing if existing is not None else cls._default_golden_media(modality)
        text = cls._first_text(requested_messages) or cls._default_golden_prompt(modality)
        key = "image" if modality == "image" else modality
        return [
            {
                "role": "user",
                "content": [
                    {"type": modality, key: media_value},
                    {"type": "text", "text": cls._default_golden_prompt(modality) if existing is None else text},
                ],
            }
        ]

    @staticmethod
    def _has_exported_subgraph(meta_info: Mapping[str, Any], key: str) -> bool:
        subgraph = meta_info.get(key)
        return isinstance(subgraph, Mapping) and bool(subgraph.get("hmonnx"))

    @staticmethod
    def _first_media_value(messages: list[dict[str, Any]], modality: str) -> Any | None:
        media_key = "image" if modality == "image" else modality
        for message in messages:
            content = message.get("content") if isinstance(message, Mapping) else None
            if not isinstance(content, list):
                continue
            for item in content:
                if isinstance(item, Mapping) and item.get("type") == modality:
                    return item.get(media_key)
        return None

    @staticmethod
    def _first_text(messages: list[dict[str, Any]]) -> str | None:
        for message in messages:
            content = message.get("content") if isinstance(message, Mapping) else None
            if isinstance(content, str) and content:
                return content
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, Mapping) and item.get("type") == "text":
                        text = item.get("text")
                        if isinstance(text, str) and text:
                            return text
        return None

    @staticmethod
    def _default_golden_prompt(modality: str) -> str:
        prompts = {
            "image": "请根据图像内容回答：画面里有哪些颜色和形状？",
            "video": "请根据这些视频帧回答：画面变化体现了什么规律？",
            "audio": "请根据音频内容回答：这段声音的整体特征是什么？",
        }
        return prompts.get(modality, "请简要回答这个测试问题。")

    @classmethod
    def _default_golden_media(cls, modality: str) -> Any:
        if modality == "image":
            return cls._default_golden_image()
        if modality == "video":
            return cls._default_golden_video()
        if modality == "audio":
            return cls._default_golden_audio()
        raise ValueError(f"Unsupported Gemma4 golden modality: {modality}")

    @staticmethod
    def _default_golden_image() -> Any:
        from PIL import Image, ImageDraw

        image = Image.new("RGB", (224, 224), color=(235, 242, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((24, 32, 96, 112), fill=(42, 100, 210))
        draw.ellipse((124, 44, 196, 116), fill=(230, 76, 76))
        draw.polygon([(56, 170), (104, 136), (152, 170)], fill=(75, 170, 95))
        draw.text((28, 190), "G4", fill=(20, 20, 20))
        return image

    @classmethod
    def _default_golden_video(cls) -> list[Any]:
        from PIL import Image, ImageDraw

        frames = []
        for idx in range(32):
            frame = Image.new("RGB", (224, 224), color=(245, 245, 245))
            draw = ImageDraw.Draw(frame)
            x0 = 16 + idx * 8
            draw.rectangle((x0, 72, x0 + 48, 120), fill=(40, 120, 220))
            draw.text((24, 24), f"frame {idx:02d}", fill=(0, 0, 0))
            frames.append(frame)
        return frames

    @staticmethod
    def _default_golden_audio() -> Any:
        import numpy as np

        sampling_rate = 16000
        seconds = 2
        t = np.linspace(0, seconds, sampling_rate * seconds, endpoint=False, dtype=np.float32)
        tone = 0.25 * np.sin(2 * np.pi * 440.0 * t)
        envelope = np.linspace(0.2, 1.0, tone.shape[0], dtype=np.float32)
        return (tone * envelope).astype(np.float32)

    def _validate_export_model(self, config_overrides: Mapping[str, Any] | None) -> None:
        from ...builder import get_model_class

        workflow_config = self.workflow_config.with_overrides(config_overrides)
        export_cfg = workflow_config.build_export_dict(self.hf_model_dir)
        model_cfg = export_cfg["model"]
        export_plan = self._build_export_plan(workflow_config)
        export_plan.validate_fixed_contract()
        model_type = model_cfg.get("model_type") if isinstance(model_cfg, Mapping) else None
        model_cls = get_model_class(model_cfg)
        if model_cls is None:
            return
        model_cls_name = model_cls.__name__
        config_cls = getattr(model_cls, "CONFIG_CLS", None)
        config_cls_name = getattr(config_cls, "__name__", None)
        if (
            model_type not in _GEMMA4_TOP_LEVEL_MODEL_TYPES
            or model_cls_name not in _GEMMA4_MODEL_CLS_NAMES
            or config_cls_name not in _GEMMA4_MODEL_CONFIG_CLS_NAMES
        ):
            raise TypeError(
                "Gemma4SeriesWorkflow supports the unified Gemma4ForConditionalGeneration public model type "
                "for dense/E4B/26B-A4B exports, with legacy _with_mask accepted only "
                "for compatibility; "
                f"got model_type={model_type!r}, model={model_cls_name}, "
                f"config={config_cls_name}, plan={export_plan.to_log_dict()}."
            )

    def _build_export_plan(self, workflow_config: WorkflowConfig) -> Gemma4SeriesExportPlan:
        export_cfg = workflow_config.build_export_dict(self.hf_model_dir)
        model_cfg = export_cfg.get("model")
        if not isinstance(model_cfg, Mapping):
            raise TypeError("Gemma4 Series workflow export.model must be a mapping")
        return build_gemma4_series_export_plan(
            hf_model_dir=self.hf_model_dir,
            export_model_cfg=model_cfg,
        )

    def _workflow_config_for_quant(self, config_overrides: Mapping[str, Any] | None) -> WorkflowConfig:
        """Apply quant overrides as a quant-stage choice, not as a strict path patch.

        WorkflowConfig.with_overrides intentionally rejects new nested keys. That
        is useful for export-model knobs, but quant mode switches such as
        gptqmodel -> existing_hf need to replace the whole quant block with keys
        that are not present in the default YAML. Keeping this local to Gemma4
        avoids weakening global workflow validation.
        """
        if not config_overrides or "quant" not in config_overrides:
            return self.workflow_config.with_overrides(config_overrides)

        other_overrides = {key: value for key, value in config_overrides.items() if key != "quant"}
        base_config = self.workflow_config.with_overrides(other_overrides) if other_overrides else self.workflow_config
        data = copy.deepcopy(base_config.data)
        data["quant"] = config_overrides["quant"]
        source = str(Path(base_config.source).with_stem(Path(base_config.source).stem + "_quant_override"))
        WorkflowConfig._validate_workflow_data(data, source)
        return WorkflowConfig(data=data, source=source)

    def _quant_gptqmodel_recipe(
        self,
        output_dir: str,
        device: str,
        quant_cfg: Mapping[str, Any],
        export_model_cfg: Mapping[str, Any],
        effective_config_file: str | None,
    ) -> QuantResult:
        from .quant_adapter import quantize_with_gptqmodel_recipe

        return quantize_with_gptqmodel_recipe(
            hf_model_dir=self.hf_model_dir,
            output_dir=output_dir,
            device=device,
            quant_cfg=quant_cfg,
            export_model_cfg=export_model_cfg,
            effective_config_file=effective_config_file,
            workflow_seed=self.seed,
        )

    def _quant_autoround_mode1(
        self,
        output_dir: str,
        device: str,
        quant_cfg: Mapping[str, Any],
        export_model_cfg: Mapping[str, Any],
        effective_config_file: str | None,
    ) -> QuantResult:
        from .quant_adapter import quantize_with_autoround_mode1

        return quantize_with_autoround_mode1(
            hf_model_dir=self.hf_model_dir,
            output_dir=output_dir,
            device=device,
            quant_cfg=quant_cfg,
            export_model_cfg=export_model_cfg,
            effective_config_file=effective_config_file,
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
    def _is_explicit_base_quant_override(config_overrides: Mapping[str, Any] | None) -> bool:
        return bool(config_overrides and "quant" in config_overrides and config_overrides["quant"] is None)

    @staticmethod
    def _messages_have_multimodal(messages: list[dict[str, Any]]) -> bool:
        for message in messages:
            content = message.get("content") if isinstance(message, Mapping) else None
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, Mapping) and item.get("type") in {"image", "video", "audio"}:
                        return True
        return False

    @staticmethod
    def _normalize_path(path: str | os.PathLike[str]) -> str:
        return os.path.abspath(os.path.normpath(str(path)))

    @classmethod
    def _require_existing_directory(cls, path: str | os.PathLike[str], *, field: str) -> str:
        raw_path = str(path)
        expanded = os.path.expanduser(os.path.expandvars(raw_path))
        if "$" in expanded or not Path(expanded).is_dir():
            raise FileNotFoundError(
                f"Gemma4 quant preflight failed: {field}={raw_path!r} "
                "does not resolve to a readable local directory. "
                "Set the corresponding environment variable or pass an existing GPTQModel-compatible HF directory."
            )
        return cls._normalize_path(expanded)


XHGemma4SeriesHMONNXWorkflow = Gemma4SeriesWorkflow
XHGemma4HMONNXWorkflow = Gemma4SeriesWorkflow


__all__ = [
    "Gemma4SeriesWorkflow",
    "XHGemma4SeriesHMONNXWorkflow",
    "XHGemma4HMONNXWorkflow",
    "dump_autoround_moe_mode1_quant_config_template",
    "dump_autoround_mode1_quant_config_template",
    "dump_export_config_template",
    "dump_quant_config_template",
    "export",
    "get_export_config_help",
    "get_quant_config_help",
    "list_recommended_configs",
    "quant",
]
