from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .export_common import build_component_model, quantize_and_export
from .resource_path import resolve_repo_resource


# Real calibration prompt for the LLM backbone (Qwen3-8B base, matching the
# reference-model practice of calibrating the LLM with real text, not zeros).
# The calibration set mirrors the qwen3_5 family convention: a plain text prompt
# is embedded through the real tokenizer/embedding so the activation statistics
# seen by the quantizer reflect genuine text inputs.
LLM_CALIBRATION_PROMPT = "请用一句话描述这段视频的内容。视频中有一个滑雪者在雪山上滑行，背景是连绵的雪山和蓝天。"


def _load_llm_calibration_prompts(component_cfg: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Load a deterministic text mix for the LLM PTQ calibration forward."""
    configured_path = component_cfg.get("calibration_jsonl")
    if not configured_path:
        prompt_sha256 = hashlib.sha256(LLM_CALIBRATION_PROMPT.encode("utf-8")).hexdigest()
        return [LLM_CALIBRATION_PROMPT], {
            "source": "builtin_prompt",
            "sample_count": 1,
            "text_sha256": prompt_sha256,
        }

    calibration_path = resolve_repo_resource(str(configured_path), description="LLM calibration jsonl")
    requested_samples = int(component_cfg.get("calibration_samples", 4))
    if requested_samples <= 0:
        raise ValueError("components.llm.calibration_samples must be positive")

    prompts: list[str] = []
    with calibration_path.open(encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            text = str(payload.get("text") or "").strip()
            if text:
                prompts.append(text)
            if len(prompts) >= requested_samples:
                break
    if not prompts:
        raise ValueError(f"LLM calibration jsonl contains no usable text: {calibration_path}")

    dataset_sha256 = hashlib.sha256(calibration_path.read_bytes()).hexdigest()
    text_sha256 = hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()
    return prompts, {
        "source": "jsonl",
        "configured_path": str(configured_path),
        "resolved_path": str(calibration_path),
        "dataset_sha256": dataset_sha256,
        "requested_samples": requested_samples,
        "sample_count": len(prompts),
        "text_sha256": text_sha256,
    }


def _file_md5(path: Path) -> str:
    """Compute the md5 of a file, mirroring xh_llm's calculate_file_md5."""
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_golden_meta_info(
    export_dir: Path,
    model_dir: str,
    prefill_graph: Path,
    decode_graph: Path,
    *,
    prefill_chunk_length: int,
    num_hidden_layers: int,
    kv_cache_shape: list[int],
    pad_token_id: int,
    target_device: str,
) -> dict[str, Any]:
    """Write a standard :class:`LLMModelMeta` child artifact.

    This is deliberately the same metadata boundary consumed by
    ``BaseLLMHMONNXModel`` in QwenVL and MiniCPM-V4.6.  The former MiniCPM-o
    export wrote only paths and checksums here, then duplicated the cache
    contract in the root metadata.  Keeping the contract in the child artifact
    lets a shared LLM runtime own sessions and ``KVCacheMixin`` directly.
    """
    hf_config_dir = export_dir / "hf_config"
    hf_config_dir.mkdir(parents=True, exist_ok=True)
    model_root = Path(model_dir)
    for cfg_file in sorted(model_root.glob("*.json")) + sorted(model_root.glob("*.jinja")):
        if cfg_file.name.endswith(".safetensors.index.json"):
            continue
        shutil.copyfile(cfg_file, hf_config_dir / cfg_file.name)
    quant_embedding = export_dir / "quant_embedding.pt"

    def rel(path: Path) -> str:
        return str(path.relative_to(export_dir))

    meta = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "model_name": export_dir.name,  # hmquant_<model_name>_<date>
        "model_config": {
            "model_name": export_dir.name,
            "chip_arch": target_device,
            "model_type": "MiniCPMO45LLMModel",
            "hf_model": str(model_dir),
            "batch_size": int(kv_cache_shape[0]),
            "context_max_length": int(kv_cache_shape[2]),
            "prefill_chunk_length": int(prefill_chunk_length),
            "num_logits_to_keep": 1,
            "use_cache": True,
        },
        "hf_config": str(hf_config_dir.relative_to(export_dir)),
        "quant_embedding": rel(quant_embedding),
        "quant_embedding_md5": _file_md5(quant_embedding),
        "kv_cache": {
            "num_layers": int(num_hidden_layers),
            "kv_cache_shape": list(kv_cache_shape),
            "cache_axis": 2,
            "batch_size": int(kv_cache_shape[0]),
            "cache_dtype": "float16",
            "use_cache": True,
        },
        "prefill_hmonnx": rel(prefill_graph),
        "prefill_hmonnx_md5": _file_md5(prefill_graph),
        "decode_hmonnx": rel(decode_graph),
        "decode_hmonnx_md5": _file_md5(decode_graph),
        "pad_token_id": int(pad_token_id),
        "meta": {"class_name": "LLMModelMeta"},
    }
    meta_file = export_dir / "golden_meta_info.json"
    meta_file.write_text(json.dumps(meta, indent=4, ensure_ascii=False) + "\n", encoding="utf-8")
    return meta


def _load_gptq_llm_weights(host: Any, gptq_model_dir: Path) -> None:
    """Load a packed GPTQ LLM into the MiniCPM host through the shared path.

    Standard GPTQ workflows load a ``gptqmodel_hf`` artifact and dequantize it
    in memory before wrapping for HMONNX export. MiniCPM differs only in that
    the packed artifact is its nested Qwen3 LLM, so the reconstructed state
    dict must be injected into ``host.llm`` rather than replacing the host.
    """
    from ...base_model import XHBaseModel

    if not gptq_model_dir.is_dir():
        raise FileNotFoundError(f"GPTQ LLM artifact directory not found: {gptq_model_dir}")
    gptq_llm = XHBaseModel._load_gptqmodel(
        str(gptq_model_dir),
        device_map="cpu",
        trust_remote_code=True,
    )
    gptq_llm = XHBaseModel._dequantize_gptqmodel_hf_model(gptq_llm)
    state = {key: value.detach().cpu() for key, value in gptq_llm.state_dict().items()}
    # The GPTQ dequantized Qwen3 checkpoint may carry bias tensors the MiniCPM
    # host LLM does not use; load only the host's keys so extra biases are ignored.
    host_keys = set(host.llm.state_dict().keys())
    missing = sorted(host_keys - set(state.keys()))
    if missing:
        raise RuntimeError(f"missing keys when loading GPTQ dequant weights: {missing[:5]}")
    filtered = {key: value for key, value in state.items() if key in host_keys}
    host.llm.load_state_dict(filtered, strict=True)
    host.llm = host.llm.to(torch.float16)


def _llm_calibration_embeds(
    tokenizer: Any,
    embedding: Any,
    prefill_length: int,
    hidden_size: int,
    device: str,
    prompts: list[str] | None = None,
) -> torch.Tensor:
    """Pack representative prompts evenly into one fixed-length calibration input."""
    prompts = prompts or [LLM_CALIBRATION_PROMPT]
    packed_ids: list[torch.Tensor] = []
    remaining = prefill_length
    for index, prompt in enumerate(prompts):
        prompt_count_left = len(prompts) - index
        token_budget = max(1, remaining // prompt_count_left)
        token_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"].reshape(-1)
        selected = token_ids[:token_budget]
        packed_ids.append(selected)
        remaining -= int(selected.numel())
        if remaining <= 0:
            break
    token_ids = torch.cat(packed_ids, dim=0).unsqueeze(0)
    embeds = embedding(token_ids.to(embedding.weight.device))
    if embeds.shape[1] > prefill_length:
        embeds = embeds[:, :prefill_length, :]
    if embeds.shape[1] < prefill_length:
        pad = torch.zeros((1, prefill_length - embeds.shape[1], hidden_size), dtype=embeds.dtype)
        embeds = torch.cat([embeds, pad], dim=1)
    return embeds.to(torch.float16).to(device)


def export_minicpm_o_4_5_llm(
    *,
    work_dir: Path,
    model_dir: str,
    component_cfg: Mapping[str, Any],
    target_device: str,
    device: str,
    model_name: str | None = None,
    export_basename: str | None = None,
    export_file_stem: str | None = None,
    gptq_llm_model_dir: str | None = None,
) -> dict[str, Any]:
    # xh_llm layout: <component_dir>/hmquant_<model_name>_<date>/prefill|decode/
    # The workflow passes both the dated directory name and the stable file
    # stem, mirroring the xh_llm naming convention without inferring one from
    # the other.
    export_dir_name = export_basename or f"hmquant_{Path(model_dir).name.lower()}"
    file_stem = export_file_stem or export_dir_name
    export_dir = work_dir / "LLM" / export_dir_name
    prefill_dir = export_dir / "prefill"
    decode_dir = export_dir / "decode"
    prefill_dir.mkdir(parents=True, exist_ok=True)
    decode_dir.mkdir(parents=True, exist_ok=True)
    model = build_component_model(model_dir, component_cfg)
    host = model.get_hf_model(device_map="cpu")
    if gptq_llm_model_dir is not None:
        _load_gptq_llm_weights(host, Path(gptq_llm_model_dir))
    token_embedding_file = export_dir / "quant_embedding.pt"
    token_embedding_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(host.llm.get_input_embeddings().state_dict(), token_embedding_file)
    model.init_wrap_model(host)
    hidden_size = int(host.llm.config.hidden_size)
    prefill_length = int(component_cfg["wrap_cfg"]["input_sequence_length"])
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    calibration_prompts, calibration_info = _load_llm_calibration_prompts(component_cfg)
    calibration_embeds = _llm_calibration_embeds(
        tokenizer,
        host.llm.get_input_embeddings(),
        prefill_length,
        hidden_size,
        device,
        prompts=calibration_prompts,
    )
    prefill_data = {
        "inputs_embeds": torch.zeros((1, prefill_length, hidden_size), dtype=torch.float16),
        "past_seq_length": 0,
    }
    calib_data = {
        "inputs_embeds": calibration_embeds,
        "past_seq_length": 0,
    }
    prefill_graph = Path(
        quantize_and_export(
            model,
            prefill_data,
            prefill_dir,
            f"{file_stem}_prefill_with_act",
            device,
            calibrate=True,
            calib_data=calib_data,
        )
    )
    model.release_exported_model()
    model.set_input_sequence_length(1)
    decode_data = {
        "inputs_embeds": torch.zeros((1, 1, hidden_size), dtype=torch.float16),
        "past_seq_length": prefill_length,
    }
    decode_graph = Path(
        model.to_export_onnx(
            decode_data,
            str(decode_dir),
            f"{file_stem}_decode_with_act",
        )[0]
    )
    cache_shape = list(model.past_key_caches[0].shape)
    golden_meta = _write_golden_meta_info(
        export_dir,
        model_dir,
        prefill_graph,
        decode_graph,
        prefill_chunk_length=prefill_length,
        num_hidden_layers=len(model.past_key_caches),
        kv_cache_shape=cache_shape,
        pad_token_id=int(model.pad_token_id),
        target_device=target_device,
    )
    return {
        "quant_type": component_cfg["quant_type"],
        "artifact_dir": str(export_dir.relative_to(work_dir)),
        "metadata": str((export_dir / "golden_meta_info.json").relative_to(work_dir)),
        "golden_meta_info": {key: value for key, value in golden_meta.items() if key != "model_name"},
        "graphs": {
            "prefill": str(prefill_graph.relative_to(work_dir)),
            "decode": str(decode_graph.relative_to(work_dir)),
        },
        "prefill_hmonnx": str(prefill_graph.relative_to(work_dir)),
        "decode_hmonnx": str(decode_graph.relative_to(work_dir)),
        "quant_embedding": str(token_embedding_file.relative_to(work_dir)),
        "hf_config": str((export_dir / "hf_config").relative_to(work_dir)),
        "token_embedding_file": str(token_embedding_file.relative_to(work_dir)),
        "prefill_input_sequence_length": prefill_length,
        "num_hidden_layers": len(model.past_key_caches),
        "kv_cache_shape": cache_shape,
        "calibration": calibration_info,
    }


__all__ = ["export_minicpm_o_4_5_llm"]
