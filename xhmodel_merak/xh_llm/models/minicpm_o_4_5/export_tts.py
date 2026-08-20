from __future__ import annotations

import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .export_common import build_component_model, quantize_and_export
from .export_tts_projection import export_minicpm_o_4_5_tts_projection


def _write_tts_meta(
    work_dir: Path,
    *,
    prefill_graph: Path,
    decode_graph: Path,
    prefill_length: int,
    num_hidden_layers: int,
    kv_cache_shape: list[int],
    target_device: str,
) -> str:
    """Write the standard decoder metadata consumed by the shared text base.

    TTS has no standalone token embedding: it accepts embeddings produced by
    the official MiniCPM model.  Its child meta consequently leaves
    ``quant_embedding`` empty; ``MiniCPMO45TTSHMONNXRuntime`` supplies the
    parent class's unused placeholder embedding.
    """
    tts_dir = work_dir / "TTS"
    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model_config": {
            "model_name": "minicpm_o_4_5_tts",
            "chip_arch": target_device,
            "model_type": "MiniCPMO45TTSModel",
            "batch_size": int(kv_cache_shape[0]),
            "context_max_length": int(kv_cache_shape[2]),
            "prefill_chunk_length": int(prefill_length),
            "num_logits_to_keep": 1,
            "use_cache": True,
        },
        "hf_config": ".",
        "quant_embedding": "",
        "kv_cache": {
            "num_layers": int(num_hidden_layers),
            "kv_cache_shape": list(kv_cache_shape),
            "cache_axis": 2,
            "batch_size": int(kv_cache_shape[0]),
            "cache_dtype": "float16",
            "use_cache": True,
        },
        "prefill_hmonnx": str(prefill_graph.relative_to(tts_dir)),
        "decode_hmonnx": str(decode_graph.relative_to(tts_dir)),
        "pad_token_id": 0,
        "meta": {"class_name": "LLMModelMeta"},
    }
    meta_path = tts_dir / "golden_meta_info.json"
    meta_path.write_text(json.dumps(meta, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    return str(meta_path.relative_to(work_dir))


def export_minicpm_o_4_5_tts(
    *,
    work_dir: Path,
    model_dir: str,
    component_cfg: Mapping[str, Any],
    target_device: str,
    device: str,
    model_name: str | None = None,
    export_basename: str | None = None,
) -> dict[str, Any]:
    model_name = model_name or "minicpm_o_4_5"
    model = build_component_model(model_dir, component_cfg)
    host = model.get_hf_model(device_map="cpu")
    model.init_wrap_model(host)
    hidden_size = int(model.config.hidden_size)
    prefill_length = int(component_cfg["wrap_cfg"]["input_sequence_length"])
    max_length = int(component_cfg["wrap_cfg"]["max_sequence_length"])
    prefill_data = {
        "inputs_embeds": torch.zeros((1, prefill_length, hidden_size), dtype=torch.float16),
        "past_seq_length": torch.tensor([0], dtype=torch.int32),
        "current_input_length": torch.tensor([prefill_length], dtype=torch.int32),
        "attention_mask": torch.zeros((1, 1, prefill_length, max_length), dtype=torch.float16),
    }
    # The TTS component is a small (8-layer) LLM; calibrate it with real-shaped
    # hidden-state inputs like the qwen3_tts talker calibration, instead of the
    # shape-only dummy path used for vision/audio.
    calib_data = {
        "inputs_embeds": torch.randn((1, prefill_length, hidden_size), dtype=torch.float16),
        "past_seq_length": torch.tensor([0], dtype=torch.int32),
        "current_input_length": torch.tensor([prefill_length], dtype=torch.int32),
        "attention_mask": torch.zeros((1, 1, prefill_length, max_length), dtype=torch.float16),
    }
    prefill_graph = Path(
        quantize_and_export(
            model,
            prefill_data,
            work_dir / "TTS" / "Prefill",
            f"{model_name}_tts_prefill_{target_device}_{component_cfg['quant_type']}",
            device,
            calibrate=True,
            calib_data=calib_data,
        )
    )
    model.release_exported_model()
    model.set_input_sequence_length(1)
    decode_data = {
        "inputs_embeds": torch.zeros((1, 1, hidden_size), dtype=torch.float16),
        "past_seq_length": torch.tensor([prefill_length], dtype=torch.int32),
        "current_input_length": torch.tensor([1], dtype=torch.int32),
        "attention_mask": torch.zeros((1, 1, 1, max_length), dtype=torch.float16),
    }
    decode_graph = Path(
        model.to_export_onnx(
            decode_data,
            str(work_dir / "TTS" / "Decode"),
            f"{model_name}_tts_decode_{target_device}_{component_cfg['quant_type']}",
        )[0]
    )
    result = {
        "quant_type": component_cfg["quant_type"],
        "graphs": {
            "prefill": str(prefill_graph.relative_to(work_dir)),
            "decode": str(decode_graph.relative_to(work_dir)),
        },
        "prefill_input_sequence_length": prefill_length,
        "num_hidden_layers": len(model.past_key_caches),
        "kv_cache_shape": list(model.past_key_caches[0].shape),
    }
    result["metadata"] = _write_tts_meta(
        work_dir,
        prefill_graph=prefill_graph,
        decode_graph=decode_graph,
        prefill_length=prefill_length,
        num_hidden_layers=len(model.past_key_caches),
        kv_cache_shape=list(model.past_key_caches[0].shape),
        target_device=target_device,
    )
    projection = export_minicpm_o_4_5_tts_projection(
        work_dir=work_dir,
        model_dir=model_dir,
        component_cfg=component_cfg,
        target_device=target_device,
        device=device,
        model_name=model_name,
    )
    result["projection_graphs"] = projection["graphs"]
    result["projection_seq_capacity"] = projection["projection_seq_capacity"]
    result["num_vq"] = projection["num_vq"]
    return result


__all__ = ["export_minicpm_o_4_5_tts"]
