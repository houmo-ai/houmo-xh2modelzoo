import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

from xhmodel_merak.xh_other_model.workflows.base import BaseOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult

from ._export_utils import export_audio_encoder, export_audio_tower, export_decoder, export_qwen3
from ._model import register_wrap_modules
from .register import register_funaudiochat


class FunAudioChatWorkflow(BaseOtherModelWorkflow):
    SUPPORTED_COMPONENTS = {"audio_tower", "audio_encoder", "qwen3", "audio_decoder"}

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is not None:
            raise NotImplementedError("FunAudioChat has no separate quant stage; set quant: null")
        return QuantResult(raw_model_dir=self.model_dir, skipped=True)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        from xhquant.api import Config, get_root_logger, set_random_seed
        from xhmodel_merak.xh_other_model.builder import wrap_llm_model

        set_random_seed(self.seed)
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        model_dir = self._resolve_export_model_dir(quant_result)
        export_cfg = workflow_config.build_export_dict()
        target_device = str(export_cfg.get("target_device", "XH2a"))
        if target_device.lower() != "xh2a":
            raise ValueError(f"FunAudioChat only supports XH2a export, got {target_device!r}")

        components = _enabled_components(export_cfg.get("components"))
        unsupported = sorted(set(components) - self.SUPPORTED_COMPONENTS)
        if unsupported:
            raise ValueError(f"Unsupported FunAudioChat component(s): {unsupported}")
        if not components:
            raise ValueError("FunAudioChat export must enable at least one component")

        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))
        runtime_cfg = export_cfg.get("runtime") or {}
        if not isinstance(runtime_cfg, Mapping):
            raise TypeError("FunAudioChat export.runtime must be a mapping")
        audio_path = runtime_cfg.get("audio")
        if "audio_encoder" in components and not audio_path:
            raise ValueError("export.runtime.audio is required when audio_encoder is enabled")

        register_funaudiochat()
        config = AutoConfig.from_pretrained(model_dir)
        config.audio_config.crq_transformer_config["torch_dtype"] = torch.float16
        processor = AutoProcessor.from_pretrained(model_dir)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_dir,
            config=config,
            torch_dtype=torch.float16,
            device_map=device if device and str(device).startswith("cuda") else None,
        ).eval()

        context_length = int(export_cfg.get("context_length", 256))
        input_sequence_length = int(export_cfg.get("input_sequence_length", 256))
        register_wrap_modules(model)
        wrap_llm_model(model.continuous_audio_tower, Config({}))
        wrap_llm_model(model.audio_tower, Config({}))
        wrap_llm_model(model.language_model, _llm_wrap_config(context_length, input_sequence_length))
        if getattr(model, "audio_invert_tower", None) is not None:
            decoder_length = input_sequence_length * int(model.audio_invert_tower.group_size)
            wrap_llm_model(model.audio_invert_tower.crq_transformer, _llm_wrap_config(decoder_length, decoder_length))

        args = _build_export_args(export_cfg, runtime_cfg)
        logger = get_root_logger()
        artifacts: dict[str, Any] = {}
        if "audio_tower" in components:
            artifacts["audio_tower_hmonnx"] = _relative(export_audio_tower(model, args, work_dir, logger), work_dir)
        if "audio_encoder" in components:
            artifacts["audio_encoder_hmonnx"] = _relative(
                export_audio_encoder(model, processor, args, work_dir, logger), work_dir
            )
        if "qwen3" in components:
            qwen_meta = export_qwen3(model, args, work_dir)
            artifacts["qwen3_meta"] = _relative(qwen_meta, work_dir)
            qwen_data = json.loads(qwen_meta.read_text(encoding="utf-8"))
            artifacts["qwen3_prefill_hmonnx"] = str(Path("qwen3") / qwen_data["prefill_onnx"])
            artifacts["qwen3_decode_hmonnx"] = str(Path("qwen3") / qwen_data["decode_onnx"])
        if "audio_decoder" in components:
            prefill_file, decode_file = export_decoder(model, processor, args, work_dir, logger)
            artifacts["audio_decoder_prefill_hmonnx"] = _relative(prefill_file, work_dir)
            artifacts["audio_decoder_decode_hmonnx"] = _relative(decode_file, work_dir)

        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "config": _relative(Path(config_file), work_dir),
            "model_type": export_cfg["model"]["type"],
            "source_model_dir": model_dir,
            "target_device": target_device,
            "context_length": context_length,
            "input_sequence_length": input_sequence_length,
            "audio_duration_seconds": args.audio_duration_seconds,
            "components": components,
            "artifacts": artifacts,
        }
        (work_dir / "export_meta_info.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return ExportResult(work_dir=str(work_dir), config_file=config_file, meta=meta)

    def dump_golden(self, export_result: ExportResult, device: str, input_messages: Any = None) -> str:
        del input_messages
        import onnx
        from xhquant.api import CacheTensor, HMONNXGoldenInference

        work_dir = Path(export_result.work_dir)
        meta = json.loads((work_dir / "export_meta_info.json").read_text(encoding="utf-8"))
        artifacts = meta.get("artifacts") or {}
        torch_device = _golden_device(device)
        for component in ("audio_tower", "audio_encoder", "qwen3", "audio_decoder"):
            if component not in meta.get("components", []):
                continue
            component_dir = work_dir / component
            component_meta = json.loads((component_dir / "meta.json").read_text(encoding="utf-8"))
            graph_paths = _component_graphs(component, artifacts, work_dir)
            for graph_path in graph_paths:
                graph = onnx.load(str(graph_path))
                inputs = _build_golden_inputs(graph, component_meta, torch_device, CacheTensor)
                golden_dir = graph_path.parent / "golden"
                if golden_dir.exists():
                    shutil.rmtree(golden_dir)
                golden_dir.mkdir(parents=True, exist_ok=True)
                session = HMONNXGoldenInference(str(graph_path))
                session.to(torch_device)
                session.save_golden = True
                session.golden_dir = str(golden_dir)
                session.step = 0
                session(*inputs)
        return str(work_dir)


def _enabled_components(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        raise TypeError("FunAudioChat export.components must be a mapping")
    return [
        str(name)
        for name, cfg in value.items()
        if cfg is not False and cfg is not None and (not isinstance(cfg, Mapping) or cfg.get("enabled", True))
    ]


def _build_export_args(export_cfg: Mapping[str, Any], runtime_cfg: Mapping[str, Any]) -> SimpleNamespace:
    components = export_cfg["components"]
    audio_duration_seconds = float(export_cfg.get("audio_duration_seconds", 8.0))
    if audio_duration_seconds <= 0:
        raise ValueError(f"export.audio_duration_seconds must be positive, got {audio_duration_seconds}")

    def quant_type(name: str) -> str:
        cfg = components.get(name) or {}
        return str(cfg.get("quant_type", "w8a8h1_sefp")) if isinstance(cfg, Mapping) else "w8a8h1_sefp"

    return SimpleNamespace(
        audio=str(runtime_cfg.get("audio") or ""),
        system_prompt=str(runtime_cfg.get("system_prompt") or ""),
        context_length=int(export_cfg.get("context_length", 256)),
        input_sequence_length=int(export_cfg.get("input_sequence_length", 256)),
        audio_duration_seconds=audio_duration_seconds,
        audio_quant_type=quant_type("audio_encoder"),
        llm_quant_type=quant_type("qwen3"),
        decoder_quant_type=quant_type("audio_decoder"),
    )


def _llm_wrap_config(context_length: int, input_sequence_length: int):
    from xhquant.api import Config

    return Config(
        {
            "max_sequence_length": context_length,
            "input_sequence_length": input_sequence_length,
            "use_cache": True,
            "num_logits_to_keep": 0,
            "kv_cache": {"cache_axis": 2},
        }
    )


def _relative(path: Path, root: Path) -> str:
    return str(path.relative_to(root))


def _component_graphs(component: str, artifacts: Mapping[str, Any], work_dir: Path) -> list[Path]:
    if component == "audio_tower":
        return [work_dir / artifacts["audio_tower_hmonnx"]]
    if component == "audio_encoder":
        return [work_dir / artifacts["audio_encoder_hmonnx"]]
    if component == "qwen3":
        return [
            work_dir / artifacts["qwen3_prefill_hmonnx"],
            work_dir / artifacts["qwen3_decode_hmonnx"],
        ]
    return [
        work_dir / artifacts["audio_decoder_prefill_hmonnx"],
        work_dir / artifacts["audio_decoder_decode_hmonnx"],
    ]


def _build_golden_inputs(
    graph: Any, component_meta: Mapping[str, Any], device: str, cache_tensor_cls: Any
) -> list[Any]:
    type_map = {1: torch.float32, 6: torch.int32, 7: torch.int64, 10: torch.float16, 11: torch.float64}
    inputs = []
    cache_shape = component_meta.get("kv_cache", {}).get("shape")
    for value in graph.graph.input:
        tensor_type = value.type.tensor_type
        shape = []
        for index, dim in enumerate(tensor_type.shape.dim):
            size = int(dim.dim_value) if dim.dim_value else _golden_dim(value.name, index, component_meta)
            shape.append(max(size, 1))
        dtype = type_map.get(tensor_type.elem_type, torch.float16)
        if value.name.startswith(("past_key_cache_", "past_value_cache_")) and cache_shape:
            tensor = cache_tensor_cls(
                torch.zeros(tuple(int(item) for item in cache_shape), dtype=torch.float16, device=device)
            )
        else:
            tensor = torch.zeros(tuple(shape), dtype=dtype, device=device)
            if value.name == "speech_ids":
                tensor.fill_(int(component_meta.get("pad_token_id", 0)))
        inputs.append(tensor)
    return inputs


def _golden_dim(name: str, index: int, component_meta: Mapping[str, Any]) -> int:
    if name == "speech_ids":
        return int(component_meta.get("group_size", 1))
    if "sequence" in name or "length" in name:
        return int(component_meta.get("input_sequence_length", 256))
    if index == 0:
        return 1
    return 1


def _golden_device(device: str) -> str:
    return str(device) if torch.cuda.is_available() and str(device).startswith("cuda") else "cpu"
