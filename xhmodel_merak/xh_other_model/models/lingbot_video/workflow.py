from __future__ import annotations

import gc
import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from xhmodel_merak.xh_other_model.workflows.base import BaseOtherModelWorkflow
from xhmodel_merak.xh_other_model.workflows.result import ExportResult, QuantResult


class LingBotVideoWorkflow(BaseOtherModelWorkflow):
    REQUIRED_COMPONENTS = {
        "text_encoder",
        "visual_encoder",
        "transformer",
        "vae_encoder",
        "vae_decoder",
    }

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        del output_dir, device
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is not None:
            raise ValueError(
                "LingBot Video uses graph-coupled SEFP quantization. Keep top-level "
                "quant=null and configure each export.components.*.quant_type."
            )
        return QuantResult(
            raw_model_dir=self.model_dir,
            skipped=True,
            meta={"quantization": "integrated_into_each_static_graph_export"},
        )

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        from xhquant.api import set_random_seed

        set_random_seed(self.seed)
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        model_dir = Path(self._resolve_export_model_dir(quant_result))
        self._validate_model_dir(model_dir)
        export_cfg = workflow_config.build_export_dict()
        target_device = str(export_cfg.get("target_device", "XH2a"))
        components = export_cfg.get("components")
        if not isinstance(components, Mapping):
            raise TypeError("export.components must be a mapping")
        self._validate_components(components)
        geometry = export_cfg.get("geometry") or {}
        if not isinstance(geometry, Mapping):
            raise TypeError("export.geometry must be a mapping")
        self._validate_static_profile(components, geometry)

        work_dir = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        config_file = workflow_config.dump(str(work_dir / f"{workflow_config.name}.yaml"))
        runtime = _export_runtime_files(model_dir, work_dir)
        meta: dict[str, Any] = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "target_device": target_device,
            "config": str(Path(config_file).relative_to(work_dir)),
            "runtime": runtime,
            "components": [],
            "geometry": dict(geometry),
        }

        from .text_encoder import build_text_encoder

        qwen_output_dir = work_dir / "text_encoder"
        visual_cfg = _visual_export_config(
            model_dir=model_dir,
            component_cfg=components["visual_encoder"],
            geometry=geometry,
        )
        qwen_model = build_text_encoder(
            model_dir=model_dir,
            output_dir=qwen_output_dir,
            target_device=target_device,
            text_cfg=dict(components["text_encoder"]),
            visual_cfg=visual_cfg,
        )
        qwen_model.to(torch.device(device), torch.float16)
        meta["text_encoder"] = qwen_model.export_lingbot_hmonnx(qwen_output_dir)
        meta["components"].extend(["text_encoder", "visual_encoder"])
        del qwen_model
        _release_memory()

        from .transformer import export_transformer

        meta["transformer"] = export_transformer(
            model_dir=model_dir,
            output_dir=work_dir / "transformer",
            target_device=target_device,
            component_cfg=dict(components["transformer"]),
            geometry_cfg=dict(geometry),
            exec_device=device,
        )
        meta["components"].append("transformer")
        _release_memory()

        from .vae import export_vae_components

        meta["vae"] = export_vae_components(
            model_dir=model_dir,
            output_dir=work_dir / "vae",
            target_device=target_device,
            encoder_cfg=dict(components["vae_encoder"]),
            decoder_cfg=dict(components["vae_decoder"]),
            geometry_cfg=dict(geometry),
            exec_device=device,
        )
        meta["components"].extend(["vae_encoder", "vae_decoder"])
        _release_memory()

        meta_file = work_dir / "export_meta_info.json"
        meta_file.write_text(json.dumps(meta, indent=4), encoding="utf-8")
        return ExportResult(
            work_dir=str(work_dir),
            config_file=config_file,
            meta=meta,
        )

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any = None,
    ) -> str:
        del input_messages
        from xhquant.api import HMONNXGoldenInference

        work_dir = Path(export_result.work_dir)
        meta_file = work_dir / "export_meta_info.json"
        if not meta_file.is_file():
            raise FileNotFoundError(f"Missing export metadata: {meta_file}")
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        golden_root = work_dir / "golden"

        text_meta = meta["text_encoder"]
        _dump_one_golden(
            runtime_cls=HMONNXGoldenInference,
            hmonnx_file=work_dir / "text_encoder" / text_meta["language_hmonnx"],
            inputs_file=work_dir / "text_encoder" / text_meta["language_calibration_inputs"],
            golden_dir=golden_root / "text_encoder",
            device=device,
        )
        visual_hmonnx = Path(text_meta["visual"]["hmonnx"])
        if not visual_hmonnx.is_absolute():
            visual_hmonnx = work_dir / "text_encoder" / visual_hmonnx
        _dump_one_golden(
            runtime_cls=HMONNXGoldenInference,
            hmonnx_file=visual_hmonnx,
            inputs_file=work_dir / "text_encoder" / text_meta["visual_calibration_inputs"],
            golden_dir=golden_root / "visual_encoder",
            device=device,
        )

        transformer_meta = meta["transformer"]
        _dump_one_golden(
            runtime_cls=HMONNXGoldenInference,
            hmonnx_file=work_dir / "transformer" / transformer_meta["hmonnx_file"],
            inputs_file=work_dir / "transformer" / transformer_meta["calibration_inputs"],
            golden_dir=golden_root / "transformer",
            device=device,
        )

        encoder_meta = meta["vae"]["encoder"]
        _dump_one_golden(
            runtime_cls=HMONNXGoldenInference,
            hmonnx_file=work_dir / "vae" / encoder_meta["hmonnx_file"],
            inputs_file=work_dir / "vae" / encoder_meta["calibration_inputs"],
            golden_dir=golden_root / "vae_encoder",
            device=device,
        )
        decoder_meta = meta["vae"]["decoder"]
        for stage_name in ("first", "next"):
            stage_meta = decoder_meta.get(stage_name)
            if stage_meta is None:
                continue
            _dump_one_golden(
                runtime_cls=HMONNXGoldenInference,
                hmonnx_file=work_dir / "vae" / stage_meta["hmonnx_file"],
                inputs_file=work_dir / "vae" / stage_meta["calibration_inputs"],
                golden_dir=golden_root / f"vae_decoder_{stage_name}",
                device=device,
            )
        return str(golden_root)

    @classmethod
    def _validate_components(cls, components: Mapping[str, Any]) -> None:
        missing = sorted(cls.REQUIRED_COMPONENTS - set(components))
        unsupported = sorted(set(components) - cls.REQUIRED_COMPONENTS)
        if missing:
            raise ValueError(f"Missing LingBot Video components: {missing}")
        if unsupported:
            raise ValueError(f"Unsupported LingBot Video components: {unsupported}")
        disabled = [
            name
            for name in cls.REQUIRED_COMPONENTS
            if not isinstance(components[name], Mapping) or not bool(components[name].get("enabled", True))
        ]
        if disabled:
            raise ValueError(
                "The full-model workflow requires every neural component to be enabled; "
                f"disabled/invalid: {sorted(disabled)}"
            )

    @staticmethod
    def _validate_static_profile(components: Mapping[str, Any], geometry: Mapping[str, Any]) -> None:
        mode = str(geometry.get("mode", "t2i"))
        if mode not in {"t2i", "t2v", "ti2v"}:
            raise ValueError(f"Unsupported LingBot Video mode: {mode!r}")
        height = int(geometry.get("height", 480))
        width = int(geometry.get("width", 832))
        num_frames = int(geometry.get("num_frames", 1))
        if height <= 0 or width <= 0 or height % 16 or width % 16:
            raise ValueError(f"LingBot Video height and width must be positive multiples of 16; got {height}x{width}.")
        if num_frames <= 0 or (num_frames != 1 and (num_frames - 1) % 4):
            raise ValueError(f"LingBot Video num_frames must be 1 or 4n+1; got {num_frames}.")
        if mode == "t2i" and num_frames != 1:
            raise ValueError("A t2i static profile must use num_frames=1.")

        text_length = int(components["text_encoder"].get("sequence_length", 2048))
        transformer_text_length = int(components["transformer"].get("text_sequence_length", 2048))
        if text_length != transformer_text_length:
            raise ValueError(
                "text_encoder.sequence_length and transformer.text_sequence_length "
                f"must match, got {text_length} and {transformer_text_length}."
            )
        if text_length <= 0:
            raise ValueError("The static text sequence length must be positive.")

        flash_cfg = components["transformer"].get("flash_attention", {})
        if not isinstance(flash_cfg, Mapping) or not bool(flash_cfg.get("enable", False)):
            raise ValueError(
                "LingBot transformer.flash_attention.enable must be true for full-duration static video profiles."
            )
        invalid_flash_bits = {
            name: flash_cfg.get(name)
            for name in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")
            if int(flash_cfg.get(name, 0)) not in (8, 16)
        }
        if invalid_flash_bits:
            raise ValueError(f"LingBot FlashAttention q/k/v/s/p bits must be 8 or 16; got {invalid_flash_bits}.")

    @staticmethod
    def _validate_model_dir(model_dir: Path) -> None:
        required = (
            "model_index.json",
            "processor/config.json",
            "processor/preprocessor_config.json",
            "processor/tokenizer_config.json",
            "processor/video_preprocessor_config.json",
            "text_encoder/config.json",
            "transformer/config.json",
            "vae/config.json",
            "scheduler/scheduler_config.json",
        )
        missing = [relative for relative in required if not (model_dir / relative).is_file()]
        if missing:
            raise FileNotFoundError(f"Incomplete LingBot Video model directory {model_dir}; missing {missing}")


def _export_runtime_files(model_dir: Path, work_dir: Path) -> dict[str, str]:
    runtime_dir = work_dir / "runtime"
    if runtime_dir.exists():
        shutil.rmtree(runtime_dir)

    processor_dir = runtime_dir / "processor"
    processor_dir.mkdir(parents=True)
    processor_files = (
        "added_tokens.json",
        "chat_template.json",
        "config.json",
        "configuration.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
        "processor_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "video_preprocessor_config.json",
        "vocab.json",
    )
    for filename in processor_files:
        source = model_dir / "processor" / filename
        if source.is_file():
            shutil.copyfile(source, processor_dir / filename)

    files = {
        "scheduler": (
            model_dir / "scheduler" / "scheduler_config.json",
            runtime_dir / "scheduler" / "scheduler_config.json",
        ),
        "text_encoder_config": (
            model_dir / "text_encoder" / "config.json",
            runtime_dir / "text_encoder" / "config.json",
        ),
        "transformer_config": (
            model_dir / "transformer" / "config.json",
            runtime_dir / "transformer" / "config.json",
        ),
        "vae_config": (
            model_dir / "vae" / "config.json",
            runtime_dir / "vae" / "config.json",
        ),
    }
    runtime_meta = {"processor": processor_dir.relative_to(work_dir).as_posix()}
    for name, (source, destination) in files.items():
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        artifact = destination.parent if name == "scheduler" else destination
        runtime_meta[name] = artifact.relative_to(work_dir).as_posix()
    return runtime_meta


def _visual_export_config(
    *,
    model_dir: Path,
    component_cfg: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> dict[str, Any]:
    visual_cfg = dict(component_cfg)
    if str(geometry.get("mode", "t2i")) != "ti2v":
        return visual_cfg

    text_config = json.loads((model_dir / "text_encoder" / "config.json").read_text(encoding="utf-8"))
    vision_config = text_config["vision_config"]
    patch_factor = int(vision_config["patch_size"]) * int(vision_config["spatial_merge_size"])
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    visual_height, visual_width = smart_resize(
        int(geometry.get("height", 480)),
        int(geometry.get("width", 832)),
        factor=patch_factor,
    )
    visual_cfg["max_size_h"] = visual_height
    visual_cfg["max_size_w"] = visual_width
    return visual_cfg


def _dump_one_golden(
    *,
    runtime_cls: type,
    hmonnx_file: Path,
    inputs_file: Path,
    golden_dir: Path,
    device: str,
) -> None:
    if not hmonnx_file.is_file():
        raise FileNotFoundError(f"Missing HMONNX file: {hmonnx_file}")
    if not inputs_file.is_file():
        raise FileNotFoundError(f"Missing calibration inputs: {inputs_file}")
    inputs = torch.load(inputs_file, map_location="cpu", weights_only=False)
    torch_device = torch.device(device if torch.cuda.is_available() else "cpu")
    session = runtime_cls(str(hmonnx_file))
    session.to(torch_device)
    session.exec_device = torch_device
    session.initialize()
    input_names = list(session.get_input_names())
    if len(inputs) != len(input_names):
        raise ValueError(
            f"HMONNX input count mismatch for {hmonnx_file}: expected {len(input_names)}, got {len(inputs)}"
        )
    runtime_inputs = []
    for name, tensor in zip(input_names, inputs, strict=True):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"HMONNX input {name!r} must be a tensor, got {type(tensor)}")
        expected_dtype = session.get_input(name).dtype
        runtime_inputs.append(tensor.to(device=torch_device, dtype=expected_dtype).contiguous())
    session.save_golden = True
    if golden_dir.exists():
        shutil.rmtree(golden_dir)
    golden_dir.mkdir(parents=True, exist_ok=True)
    session.golden_dir = str(golden_dir)
    session(*runtime_inputs)


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
