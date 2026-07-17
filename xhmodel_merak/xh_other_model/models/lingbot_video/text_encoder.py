from __future__ import annotations

import gc
import json
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import onnx
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText

from xhquant.api import (
    ConfigDict,
    FrontendType,
    PrecisionMode,
    ptq_quantize,
    to_export_graph,
    to_export_hmonnx_v2,
    to_frontend_graph,
    to_quant_graph,
)

from ...builder import wrap_llm_model
from ._quant_utils import build_quant_config, export_torch_component, require_xh2a
from ._qwen3_vl_modeling import (
    activate_qwen3_vl_language_graph,
    register_qwen3_vl_wrappers,
)
from .qwen3_vl_preprocess import LingBotQwen3VLDataPreprocess


LANGUAGE_GRAPH_INPUT_NAMES = (
    "inputs_embeds",
    "time_position_ids",
    "height_position_ids",
    "width_position_ids",
    "past_seq_length",
    "current_input_length",
    "deepstack_image_embed_0",
    "deepstack_image_embed_1",
    "deepstack_image_embed_2",
)
VISUAL_GRAPH_OUTPUT_NAMES = (
    "image_embeds",
    "deepstack_feature_0",
    "deepstack_feature_1",
    "deepstack_feature_2",
)
TEXT_OUTPUT_SEMANTICS = "last_hidden_state_post_final_rmsnorm"


class _LanguageGraph(nn.Module):
    def __init__(self, language_model: nn.Module):
        super().__init__()
        self.language_model = language_model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        time_position_ids: torch.Tensor,
        height_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        deepstack_image_embed_0: torch.Tensor,
        deepstack_image_embed_1: torch.Tensor,
        deepstack_image_embed_2: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.language_model(
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_input_length,
            deepstack_image_embed_0,
            deepstack_image_embed_1,
            deepstack_image_embed_2,
            None,
            None,
        )
        return outputs.last_hidden_state


class LingBotQwen3VLEncoder(nn.Module):
    def __init__(
        self,
        *,
        text_encoder_dir: Path,
        target_device: str,
        text_cfg: Mapping[str, Any],
        visual_cfg: Mapping[str, Any],
    ):
        super().__init__()
        self.text_encoder_dir = Path(text_encoder_dir)
        self.target_device = str(target_device)
        self.text_cfg = dict(text_cfg)
        self.visual_cfg = dict(visual_cfg)
        self.sequence_length = int(self.text_cfg.get("sequence_length", 2048))
        self.language_quant_type = str(self.text_cfg.get("quant_type", "w8a8h1_sefp"))
        self.visual_quant_type = str(self.visual_cfg.get("quant_type", "w8a8h1_sefp"))
        self.visual_width = int(self.visual_cfg.get("max_size_w", 448))
        self.visual_height = int(self.visual_cfg.get("max_size_h", 448))
        self.language_model_name = _component_model_name(
            "lingbot_qwen3_vl",
            self.target_device,
            self.language_quant_type,
        )
        self.visual_model_name = _component_model_name(
            "lingbot_qwen3_vl",
            self.target_device,
            self.visual_quant_type,
            profile=f"{self.visual_width}x{self.visual_height}",
        )
        self.export_device = torch.device("cpu")
        self.export_dtype = torch.float16

    def to(self, *args, **kwargs):
        device, dtype = torch._C._nn._parse_to(*args, **kwargs)[:2]
        if device is not None:
            self.export_device = torch.device(device)
        if dtype is not None:
            self.export_dtype = dtype
        return self

    def export_lingbot_hmonnx(self, output_dir: Path) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        register_qwen3_vl_wrappers()
        native_model = AutoModelForImageTextToText.from_pretrained(
            str(self.text_encoder_dir),
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).eval()
        model_config = native_model.config
        language_model = native_model.model.language_model
        visual_model = native_model.model.visual
        token_embedding = language_model.embed_tokens
        language_model.embed_tokens = nn.Identity()
        native_model.model.language_model = nn.Identity()
        native_model.model.visual = nn.Identity()
        del native_model
        _release_memory()

        embedding_file = output_dir / "quant_embedding.pt"
        save_token_embedding(token_embedding, embedding_file)
        visual_dir = output_dir / "visual"
        visual_meta = _export_visual_hmonnx(
            visual_model=visual_model,
            output_dir=visual_dir,
            output_file=visual_dir / f"{self.visual_model_name}.onnx",
            target_device=self.target_device,
            component_cfg=self.visual_cfg,
            exec_device=str(self.export_device),
        )
        del visual_model
        _release_memory()

        language_inputs = _build_language_calibration_inputs(
            token_embedding=token_embedding,
            sequence_length=self.sequence_length,
            model_config=model_config,
            device=self.export_device,
            dtype=self.export_dtype,
        )
        del token_embedding
        language_wrap_cfg = _language_wrap_config(
            sequence_length=self.sequence_length,
            model_name=self.language_model_name,
            target_device=self.target_device,
        )
        wrapped_language = wrap_llm_model(language_model, language_wrap_cfg)
        activate_qwen3_vl_language_graph(wrapped_language)
        language_graph = _LanguageGraph(wrapped_language).to(self.export_device, dtype=self.export_dtype)
        language_dir = output_dir / "language"
        language_export = export_torch_component(
            model=language_graph,
            inputs=language_inputs,
            input_names=LANGUAGE_GRAPH_INPUT_NAMES,
            output_names=("prompt_embeds",),
            output_file=language_dir / f"{self.language_model_name}.onnx",
            target_device=self.target_device,
            component_cfg=self.text_cfg,
            exec_device=str(self.export_device),
        )
        del language_graph, wrapped_language, language_model
        _release_memory()

        language_hmonnx = _relative_artifact_path(language_export["hmonnx_file"], output_dir)
        language_calibration = _relative_artifact_path(language_export["calibration_inputs"], output_dir)
        meta = {
            "language_hmonnx": language_hmonnx,
            "language_quant_type": self.language_quant_type,
            "language_calibration_inputs": language_calibration,
            "visual": visual_meta,
            "visual_calibration_inputs": visual_meta["calibration_inputs"],
            "quant_embedding": embedding_file.name,
            "sequence_length": self.sequence_length,
            "hidden_size": int(model_config.text_config.hidden_size),
            "output_semantics": TEXT_OUTPUT_SEMANTICS,
            "calibration_method": "shape_sample_dynamic_sefp_no_dataset_statistics",
        }
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=4), encoding="utf-8")
        return meta


def build_text_encoder(
    *,
    model_dir: Path,
    output_dir: Path,
    target_device: str,
    text_cfg: dict[str, Any],
    visual_cfg: dict[str, Any],
) -> LingBotQwen3VLEncoder:
    del output_dir
    return LingBotQwen3VLEncoder(
        text_encoder_dir=Path(model_dir) / "text_encoder",
        target_device=target_device,
        text_cfg=text_cfg,
        visual_cfg=visual_cfg,
    )


def save_token_embedding(embedding: nn.Embedding, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().to(torch.float16) for name, value in embedding.state_dict().items()}
    torch.save(state, output_file)


def prepare_language_graph_inputs(
    processed_inputs: list[Any] | tuple[Any, ...],
) -> list[torch.Tensor]:
    flat_inputs = list(processed_inputs)
    if len(flat_inputs) != len(LANGUAGE_GRAPH_INPUT_NAMES):
        raise ValueError(
            f"LingBot Qwen preprocessor must return {len(LANGUAGE_GRAPH_INPUT_NAMES)} tensor inputs, "
            f"got {len(flat_inputs)}"
        )
    if not all(isinstance(value, torch.Tensor) for value in flat_inputs):
        raise TypeError("LingBot Qwen preprocessor outputs must all be tensors")
    return flat_inputs


def _build_language_calibration_inputs(
    *,
    token_embedding: nn.Embedding,
    sequence_length: int,
    model_config: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    token_embedding = token_embedding.to(device=device, dtype=dtype)
    preprocessor = LingBotQwen3VLDataPreprocess(
        token_embedding=token_embedding,
        input_sequence_length=sequence_length,
        image_token_id=int(model_config.image_token_id),
        video_token_id=int(model_config.video_token_id),
        vision_start_token_id=int(model_config.vision_start_token_id),
        spatial_merge_size=int(model_config.vision_config.spatial_merge_size),
    )
    input_ids = torch.randint(0, 100, (1, sequence_length), dtype=torch.long, device=device)
    return list(preprocessor({"input_ids": input_ids, "past_seq_length": 0}))


def _language_wrap_config(
    *,
    sequence_length: int,
    model_name: str,
    target_device: str,
) -> ConfigDict:
    return ConfigDict(
        {
            "model_name": model_name,
            "chip_arch": target_device,
            "batch_size": 1,
            "context_max_length": sequence_length,
            "prefill_chunk_length": sequence_length,
            "max_sequence_length": sequence_length,
            "input_sequence_length": sequence_length,
            "num_logits_to_keep": 0,
            "use_cache": False,
            "max_pe_length": 32768,
            "only_first_block": False,
            "max_layers": None,
            "kv_cache": {
                "num_layers": -1,
                "kv_cache_shape": None,
                "cache_axis": 2,
                "batch_size": 1,
                "cache_dtype": "float16",
                "use_cache": False,
            },
        }
    )


def _visual_wrap_config(component_cfg: Mapping[str, Any], model_name: str) -> ConfigDict:
    return ConfigDict(
        {
            "model_name": model_name,
            "max_size_w": int(component_cfg.get("max_size_w", 448)),
            "max_size_h": int(component_cfg.get("max_size_h", 448)),
            "max_size_t": int(component_cfg.get("max_size_t", 2)),
            "patch_size": int(component_cfg.get("patch_size", 16)),
            "temporal_patch_size": int(component_cfg.get("temporal_patch_size", 2)),
            "spatial_merge_size": int(component_cfg.get("spatial_merge_size", 2)),
        }
    )


def _export_visual_hmonnx(
    *,
    visual_model: nn.Module,
    output_dir: Path,
    output_file: Path,
    target_device: str,
    component_cfg: Mapping[str, Any],
    exec_device: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_name = output_file.stem
    wrap_cfg = _visual_wrap_config(component_cfg, model_name)
    wrap_cfg.spatial_merge_size = int(visual_model.spatial_merge_size)
    wrapped_visual = wrap_llm_model(visual_model, wrap_cfg)
    width = int(wrap_cfg.max_size_w)
    height = int(wrap_cfg.max_size_h)
    frames = int(wrap_cfg.max_size_t)
    dummy_input = torch.zeros((1, 3, frames, height, width), dtype=torch.float32)

    float_onnx_dir = output_dir / "onnx"
    float_onnx_dir.mkdir(parents=True, exist_ok=True)
    float_onnx_file = float_onnx_dir / f"{model_name}_float.onnx"
    with tempfile.TemporaryDirectory() as temporary_dir:
        temporary_file = Path(temporary_dir) / float_onnx_file.name
        torch.onnx.export(
            wrapped_visual.float().cpu(),
            (dummy_input,),
            str(temporary_file),
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["pixel_values"],
            output_names=list(VISUAL_GRAPH_OUTPUT_NAMES),
            dynamo=False,
        )
        onnx_model = onnx.load(str(temporary_file), load_external_data=True)
        import onnxsim

        simplified, valid = onnxsim.simplify(onnx_model)
        if valid:
            onnx_model = simplified
        onnx.save_model(
            onnx_model,
            str(float_onnx_file),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{float_onnx_file.stem}_external_data",
        )

    frontend_model = onnx.load(str(float_onnx_file), load_external_data=True)
    frontend_graph = to_frontend_graph(frontend_model, FrontendType.ONNX, [dummy_input])
    quant_graph = to_quant_graph(
        frontend_graph,
        require_xh2a(target_device),
        build_quant_config(target_device, component_cfg),
    )
    runtime_input = dummy_input.to(exec_device)
    ptq_quantize(
        quant_graph,
        [[runtime_input]],
        PrecisionMode.ALIGNED,
        [exec_device],
        auto_release_unused_parameters=True,
    )
    quant_graph.fixed()
    quant_graph.to("cpu")
    calibration_file = output_dir / "calibration_inputs.pt"
    torch.save([dummy_input], calibration_file)
    export_graph = to_export_graph(quant_graph, [dummy_input])
    exported_file = to_export_hmonnx_v2(
        export_graph,
        [dummy_input],
        str(output_file),
        ConfigDict(
            {
                "input_names": ["pixel_values"],
                "output_names": list(VISUAL_GRAPH_OUTPUT_NAMES),
            }
        ),
        normalize_onnx_name=True,
    )
    del export_graph, quant_graph, frontend_graph, frontend_model, wrapped_visual
    _release_memory()
    return {
        "image_size_w": width,
        "image_size_h": height,
        "hmonnx": _relative_artifact_path(exported_file, output_dir.parent),
        "onnx": _relative_artifact_path(float_onnx_file, output_dir.parent),
        "quant_type": str(component_cfg.get("quant_type", "w8a8h1_sefp")),
        "patch_size": int(wrap_cfg.patch_size),
        "max_size_t": frames,
        "temporal_patch_size": int(wrap_cfg.temporal_patch_size),
        "spatial_merge_size": int(wrap_cfg.spatial_merge_size),
        "calibration_inputs": _relative_artifact_path(calibration_file, output_dir.parent),
    }


def _component_model_name(
    prefix: str,
    target_device: str,
    quant_type: str,
    *,
    profile: str | None = None,
) -> str:
    parts = [prefix, target_device, quant_type]
    if profile:
        parts.append(profile)
    return "_".join(parts)


def _relative_artifact_path(path: str | Path, root: Path) -> str:
    artifact = Path(path).resolve()
    root = Path(root).resolve()
    try:
        return artifact.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(f"Exported artifact {artifact} is outside component root {root}") from error


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
