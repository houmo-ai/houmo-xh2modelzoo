from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer
from xhquant.core import CacheTensor

from ..qwen3_5.qwen3_5_onnx_model import (
    Qwen3_5ONNXModel,
    _alloc_cache_inputs,
    _as_cache_value,
    _resolve_input_name,
    _infer_inputs_embeds_name,
)


def _build_runtime_linear_attn_mask(
    valid_len: int, total_len: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    mask = torch.zeros((1, total_len), dtype=dtype, device=device)
    if valid_len > 0:
        mask[:, :valid_len] = 1
    return mask


def _resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path

    base_candidate = (base_dir / path).resolve()
    if base_candidate.exists():
        return base_candidate

    cwd_candidate = path.resolve()
    if cwd_candidate.exists():
        return cwd_candidate

    return base_candidate


def _meta_value(meta_info: dict, keys: Tuple[str, ...], default: str) -> str:
    for key in keys:
        value = meta_info.get(key)
        if value:
            return value
    return default


_STRUCTURAL_META_KEYS = (
    "pad_token_id",
    "max_context_tokens",
    "wrap_cfg",
    "kv_cache",
    "linear_cache",
    "model_config",
)


def _load_sidecar_structural_meta(meta_file: Path, meta_info: dict) -> dict:
    """Fill release golden meta with structural runtime fields from export meta.

    Golden release metadata intentionally points at renamed ONNX files under the
    release directory.  Older release metadata omitted the runtime-only cache
    shapes and prefill length, while the work_dir-level ``meta.json`` still has
    them.  Copy only structural fields so path fields keep resolving inside the
    release package.
    """

    candidates = (meta_file.parent / "meta.json", meta_file.parent.parent / "meta.json")
    merged = dict(meta_info)
    missing_structural = any(merged.get(key) is None for key in _STRUCTURAL_META_KEYS)
    if not missing_structural:
        return merged

    for candidate in candidates:
        if candidate == meta_file or not candidate.exists():
            continue
        try:
            sidecar = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for key in _STRUCTURAL_META_KEYS:
            if merged.get(key) is None and key in sidecar:
                merged[key] = sidecar[key]
        break
    return merged


def _load_token_embedding(embed_path: Path) -> nn.Module:
    try:
        obj = torch.load(str(embed_path), map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(str(embed_path), map_location="cpu")

    if isinstance(obj, nn.Module):
        obj.eval()
        return obj

    if isinstance(obj, dict):
        if "weight" not in obj:
            raise ValueError(f"Unsupported token embedding state dict format: {embed_path}")
        embedding = nn.Embedding(obj["weight"].shape[0], obj["weight"].shape[1])
        embedding.load_state_dict(obj)
        embedding.eval()
        return embedding

    raise TypeError(f"Unsupported token embedding object type: {type(obj)}")


def _load_meta_artifacts(model_config_file: str):
    meta_file = Path(model_config_file).resolve()
    model_dir = meta_file.parent
    meta_info = json.loads(meta_file.read_text(encoding="utf-8"))
    meta_info = _load_sidecar_structural_meta(meta_file, meta_info)

    prefill_onnx = _resolve_path(
        model_dir, meta_info.get("prefill_onnx") or meta_info["prefill_onnx_file"]
    )
    decode_onnx = _resolve_path(
        model_dir, meta_info.get("decode_onnx") or meta_info["decode_onnx_file"]
    )
    hf_model_config_dir = _resolve_path(
        model_dir,
        _meta_value(meta_info, ("hf_config", "hf_config_dir", "hf_model"), "hf_config"),
    )
    token_embedding_file = _resolve_path(
        model_dir,
        _meta_value(
            meta_info,
            ("token_embedding_file", "quant_embedding"),
            "quant_embedding.pt",
        ),
    )

    tokenizer = AutoTokenizer.from_pretrained(str(hf_model_config_dir))
    token_embedding = _load_token_embedding(token_embedding_file)
    pad_token_id = meta_info.get("pad_token_id")
    if pad_token_id is None:
        pad_token_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
    if pad_token_id is None:
        pad_token_id = 0

    max_context_tokens = meta_info.get("max_context_tokens")
    if max_context_tokens is None:
        kv_cache = meta_info.get("kv_cache", {})
        shape = kv_cache.get("shape")
        if isinstance(shape, list) and len(shape) >= 3:
            max_context_tokens = int(shape[2])

    return {
        "meta_file": meta_file,
        "model_dir": model_dir,
        "meta_info": meta_info,
        "prefill_onnx": prefill_onnx,
        "decode_onnx": decode_onnx,
        "tokenizer": tokenizer,
        "token_embedding": token_embedding,
        "pad_token_id": int(pad_token_id),
        "max_context_tokens": max_context_tokens,
    }


def _resolve_prefill_input_sequence_length(meta_info: dict, prefill_config, prefill_inputs_info) -> int:
    if prefill_inputs_info is not None:
        shape = getattr(prefill_inputs_info, "shape", None)
        if shape is not None and len(shape) > 1 and shape[1] is not None:
            return int(shape[1])

    wrap_cfg = meta_info.get("wrap_cfg", {})
    if isinstance(wrap_cfg, dict) and wrap_cfg.get("input_sequence_length") is not None:
        return int(wrap_cfg["input_sequence_length"])

    model_config = meta_info.get("model_config", {})
    if isinstance(model_config, dict):
        for key in ("input_sequence_length", "prefill_chunk_length"):
            if model_config.get(key) is not None:
                return int(model_config[key])

    if isinstance(prefill_config, dict) and prefill_config.get("input_sequence_length") is not None:
        return int(prefill_config["input_sequence_length"])

    raise ValueError(
        "Cannot determine prefill input sequence length; meta.json must contain "
        "wrap_cfg.input_sequence_length when resource_tight_mode defers session loading."
    )


_BATCHED_INPUT_RE = re.compile(r"^(?P<base>.+)_batch_(?P<batch>\d+)$")


def _batch_suffix(name: str, base: str) -> Optional[int]:
    if name == base:
        return None
    match = _BATCHED_INPUT_RE.match(name)
    if match and match.group("base") == base:
        return int(match.group("batch"))
    return None


def _parse_layer_batch_suffix(name: str, prefix: str) -> Tuple[int, Optional[int]]:
    suffix = name[len(prefix):]
    batch_idx = None
    if "_batch_" in suffix:
        suffix, batch_str = suffix.rsplit("_batch_", 1)
        batch_idx = int(batch_str)
    return int(suffix), batch_idx


def _parse_conv_cache_name_moe(name: str) -> Tuple[Optional[str], int, Optional[int]]:
    suffix = name[len("past_conv_cache_"):]
    batch_idx = None
    if "_batch_" in suffix:
        suffix, batch_str = suffix.rsplit("_batch_", 1)
        batch_idx = int(batch_str)
    parts = suffix.rsplit("_", 1)
    if len(parts) == 2 and parts[0] in ("q", "k", "v"):
        return parts[0], int(parts[1]), batch_idx
    return None, int(suffix), batch_idx


def _resolve_moe_input_name(session, base_names: Tuple[str, ...], fallback=None) -> str:
    input_names = session.get_input_names()
    for base in base_names:
        if base in input_names:
            return base
        batch0 = f"{base}_batch_0"
        if batch0 in input_names:
            return batch0
    return _resolve_input_name(session, base_names, fallback=fallback)


class Qwen3_5MoeInference(Qwen3_5ONNXModel):
    """Thin MoE loader built on the dense Qwen3.5 ONNX runtime."""

    def __init__(
        self,
        model_config_file: str,
        fast_mode: bool = False,
        device: str = "cuda",
        execution_device: str = "cuda",
        auto_offload: bool = False,
        auto_offload_max_memory=None,
        prefill_auto_offload_max_memory=None,
        decode_auto_offload_max_memory=None,
        resource_tight_mode: bool = False,
        enable_cuda_graph: bool = False,
        cuda_graph_modules: Optional[Iterable[str]] = None,
        cuda_graph_warmup_runs: int = 3,
        cuda_graph_graph_warmup_runs: int = 6,
        cuda_graph_clone_outputs: bool = True,
    ):
        artifacts = _load_meta_artifacts(model_config_file)
        self.meta_path = str(artifacts["meta_file"])
        self.meta_info = artifacts["meta_info"]
        self.fast_mode = fast_mode
        self.tokenizer = artifacts["tokenizer"]
        self.batch_size = int(self.meta_info.get("wrap_cfg", {}).get("batch_size", 1))
        self._phase_prefill = True

        super().__init__(
            prefill={"onnx": str(artifacts["prefill_onnx"])},
            decode={"onnx": str(artifacts["decode_onnx"])},
            max_context_tokens=artifacts["max_context_tokens"],
            auto_offload=auto_offload,
            auto_offload_max_memory=auto_offload_max_memory,
            prefill_auto_offload_max_memory=prefill_auto_offload_max_memory,
            decode_auto_offload_max_memory=decode_auto_offload_max_memory,
            resource_tight_mode=resource_tight_mode,
            pad_token_id=artifacts["pad_token_id"],
            enable_cuda_graph=enable_cuda_graph,
            cuda_graph_modules=cuda_graph_modules,
            cuda_graph_warmup_runs=cuda_graph_warmup_runs,
            cuda_graph_graph_warmup_runs=cuda_graph_graph_warmup_runs,
            cuda_graph_clone_outputs=cuda_graph_clone_outputs,
        )

        self.set_input_embeddings(artifacts["token_embedding"])
        self.to(torch.device(device))
        self.set_exec_device(torch.device(execution_device))
        self.prefill_input_sequence_length = _resolve_prefill_input_sequence_length(
            self.meta_info, self.prefill_config, self._prefill_inputs_info
        )
        self.input_sequence_length = self.prefill_input_sequence_length

        cache_dtype = self.dtype
        kv_cache_info = self.meta_info.get("kv_cache", {})
        kv_cache_shape = kv_cache_info.get("shape", [])
        num_full_attn_layers = int(kv_cache_info.get("num_decoder_layers", 0))
        kv_cache_count = num_full_attn_layers * self.batch_size if self.batch_size > 1 else num_full_attn_layers
        self.past_key_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=cache_dtype, device=self.device))
            for _ in range(kv_cache_count)
        ]
        self.past_value_caches = [
            CacheTensor(torch.zeros(kv_cache_shape, dtype=cache_dtype, device=self.device))
            for _ in range(kv_cache_count)
        ]

        linear_cache_info = self.meta_info.get("linear_cache", {})
        self.linear_attention_layer_indices = linear_cache_info.get("layer_indices", [])
        self.past_conv_caches = []
        for layer_meta in linear_cache_info.get("layers", []):
            if "conv_shapes" in layer_meta:
                for shape in layer_meta["conv_shapes"]:
                    self.past_conv_caches.append(
                        CacheTensor(torch.zeros(shape, dtype=cache_dtype, device=self.device))
                    )
            else:
                self.past_conv_caches.append(
                    CacheTensor(torch.zeros(layer_meta["conv_shape"], dtype=cache_dtype, device=self.device))
                )
        self.past_recurrent_states = []
        for layer_meta in linear_cache_info.get("layers", []):
            recurrent_shapes = layer_meta.get("recurrent_shapes")
            if recurrent_shapes is None:
                recurrent_shapes = [layer_meta["recurrent_shape"]]
            for shape in recurrent_shapes:
                self.past_recurrent_states.append(CacheTensor(torch.zeros(shape, dtype=cache_dtype, device=self.device)))

    def _move_runtime_caches(self, *, device=None, dtype=None) -> None:
        for attr in (
            "past_key_caches",
            "past_value_caches",
            "past_conv_caches",
            "past_recurrent_states",
        ):
            caches = getattr(self, attr, None)
            if caches is None:
                continue
            for idx, cache in enumerate(caches):
                caches[idx] = cache.to(device=device, dtype=dtype)

    def _set_device(self, device):
        super()._set_device(device)
        self._move_runtime_caches(device=device)
        return self

    def _set_dtype(self, dtype):
        super()._set_dtype(dtype)
        self._move_runtime_caches(dtype=dtype)
        return self

    def _create_hmonnx_session(self, onnx_path: str, session_name: str):
        session = super()._create_hmonnx_session(onnx_path, session_name)
        if self.fast_mode and hasattr(session, "to_fast_mode"):
            session.to_fast_mode()
        return session

    def _create_prefill_session(self):
        onnx_path = self.prefill_config["onnx"] if isinstance(self.prefill_config, dict) else self.prefill_config.onnx
        self.prefill_session = self._create_hmonnx_session(onnx_path, "prefill")
        prefill_max_memory = (
            self.prefill_auto_offload_max_memory
            if self.prefill_auto_offload_max_memory is not None
            else self.auto_offload_max_memory
        )
        self._apply_auto_offload(self.prefill_session, prefill_max_memory)
        self._prefill_inputs_name = _resolve_moe_input_name(
            self.prefill_session, ("inputs_embeds", "input_1"), fallback=_infer_inputs_embeds_name
        )
        self._prefill_past_seq_name = _resolve_moe_input_name(self.prefill_session, ("past_seq_length", "valid_length"))
        self._prefill_current_seq_name = _resolve_moe_input_name(
            self.prefill_session, ("current_input_length", "current_length")
        )
        self._prefill_mask_name = _resolve_moe_input_name(
            self.prefill_session, ("linear_attn_mask", "attention_mask", "attn_mask")
        )
        self._prefill_inputs_info = self.prefill_session.get_input(self._prefill_inputs_name)
        self._prefill_mask_info = self.prefill_session.get_input(self._prefill_mask_name)
        self._prefill_past_seq_info = self.prefill_session.get_input(self._prefill_past_seq_name)
        self._prefill_current_seq_info = self.prefill_session.get_input(self._prefill_current_seq_name)

    def _create_decode_session(self):
        onnx_path = self.decode_config["onnx"] if isinstance(self.decode_config, dict) else self.decode_config.onnx
        self.decode_session = self._create_hmonnx_session(onnx_path, "decode")
        decode_max_memory = (
            self.decode_auto_offload_max_memory
            if self.decode_auto_offload_max_memory is not None
            else self.auto_offload_max_memory
        )
        self._apply_auto_offload(self.decode_session, decode_max_memory)
        self._decode_inputs_name = _resolve_moe_input_name(
            self.decode_session, ("inputs_embeds", "input_1"), fallback=_infer_inputs_embeds_name
        )
        self._decode_past_seq_name = _resolve_moe_input_name(self.decode_session, ("past_seq_length", "valid_length"))
        self._decode_current_seq_name = _resolve_moe_input_name(
            self.decode_session, ("current_input_length", "current_length")
        )
        self._decode_mask_name = _resolve_moe_input_name(
            self.decode_session, ("linear_attn_mask", "attention_mask", "attn_mask")
        )
        self._decode_inputs_info = self.decode_session.get_input(self._decode_inputs_name)
        self._decode_mask_info = self.decode_session.get_input(self._decode_mask_name)
        self._decode_past_seq_info = self.decode_session.get_input(self._decode_past_seq_name)
        self._decode_current_seq_info = self.decode_session.get_input(self._decode_current_seq_name)

    def set_phase_prefill(self, prefill: bool):
        self._phase_prefill = bool(prefill)
        self.input_sequence_length = self.prefill_input_sequence_length if prefill else 1

    def get_input_sequence_length(self) -> int:
        return int(self.input_sequence_length)

    def set_input_sequence_length(self, input_sequence_length: int):
        self.input_sequence_length = int(input_sequence_length)

    @property
    def execution_device(self) -> torch.device:
        return self.exec_device

    def prepare_inputs(
        self,
        data: dict,
        input_sequence_length: int,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        List[CacheTensor],
        List[CacheTensor],
        List[CacheTensor],
        List[CacheTensor],
    ]:
        input_ids = data["input_ids"].to(self.execution_device, dtype=torch.long)
        if input_ids.shape[0] != 1:
            raise ValueError("Batch size must be 1 in inference mode.")
        seq_length = int(input_ids.shape[1])
        if seq_length > input_sequence_length:
            raise ValueError(
                f"Input sequence length ({seq_length}) exceeds max ({input_sequence_length})"
            )

        if seq_length < input_sequence_length:
            padding = torch.full(
                (1, input_sequence_length - seq_length),
                self.pad_token_id,
                dtype=torch.long,
                device=self.execution_device,
            )
            input_ids = torch.cat([input_ids, padding], dim=-1)

        inputs_embeds = self.token_embedding(input_ids).to(
            device=self.execution_device, dtype=self.dtype
        )
        past_seq_length = int(data["past_seq_length"])
        position_ids = torch.arange(
            past_seq_length,
            past_seq_length + seq_length,
            dtype=torch.int32,
            device=self.execution_device,
        )
        if seq_length < input_sequence_length:
            last_pos = (
                position_ids[-1:]
                if seq_length > 0
                else torch.zeros(1, dtype=torch.int32, device=self.execution_device)
            )
            position_ids = torch.cat(
                [position_ids, last_pos.expand(input_sequence_length - seq_length)]
            )

        return (
            inputs_embeds,
            position_ids.unsqueeze(0),
            position_ids.unsqueeze(0),
            position_ids.unsqueeze(0),
            torch.tensor([past_seq_length], dtype=torch.int32, device=self.execution_device),
            torch.tensor([seq_length], dtype=torch.int32, device=self.execution_device),
            self.past_key_caches,
            self.past_value_caches,
            self.past_conv_caches,
            self.past_recurrent_states,
        )

    def _forward(
        self,
        inputs_embeds: torch.Tensor,
        time_position_ids: torch.Tensor,
        hight_position_ids: torch.Tensor,
        width_position_ids: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        linear_attn_mask: torch.Tensor,
        past_key_caches: List[CacheTensor],
        past_value_caches: List[CacheTensor],
        past_conv_caches: List[CacheTensor],
        past_recurrent_states: List[CacheTensor],
    ) -> torch.Tensor:
        session = self.prefill_session if self._phase_prefill else self.decode_session
        if session is None:
            if self._phase_prefill:
                self._ensure_prefill_session()
                session = self.prefill_session
            else:
                self._ensure_decode_session()
                session = self.decode_session
        if session is None:
            raise RuntimeError("HMONNX session is not initialized.")

        inputs_name = (
            self._prefill_inputs_name if self._phase_prefill else self._decode_inputs_name
        )
        past_name = (
            self._prefill_past_seq_name
            if self._phase_prefill
            else self._decode_past_seq_name
        )
        current_name = (
            self._prefill_current_seq_name
            if self._phase_prefill
            else self._decode_current_seq_name
        )
        mask_name = (
            self._prefill_mask_name if self._phase_prefill else self._decode_mask_name
        )

        feed: Dict[str, torch.Tensor] = {}
        for name in session.get_input_names():
            batch_idx = _batch_suffix(name, "inputs_embeds")
            if name == inputs_name or batch_idx is not None:
                feed[name] = inputs_embeds[batch_idx : batch_idx + 1].to(self.device) if batch_idx is not None else inputs_embeds.to(self.device)
            elif (batch_idx := _batch_suffix(name, "past_seq_length")) is not None or name == past_name:
                feed[name] = past_seq_length[batch_idx : batch_idx + 1].to(self.device) if batch_idx is not None else past_seq_length.to(self.device)
            elif (batch_idx := _batch_suffix(name, "current_input_length")) is not None or name == current_name:
                feed[name] = current_input_length[batch_idx : batch_idx + 1].to(self.device) if batch_idx is not None else current_input_length.to(self.device)
            elif (batch_idx := _batch_suffix(name, "linear_attn_mask")) is not None or name == mask_name:
                feed[name] = linear_attn_mask[batch_idx : batch_idx + 1].to(self.device) if batch_idx is not None else linear_attn_mask.to(self.device)
            elif (batch_idx := _batch_suffix(name, "time_position_ids")) is not None:
                feed[name] = time_position_ids[batch_idx : batch_idx + 1].to(self.device)
            elif (batch_idx := _batch_suffix(name, "hight_position_ids")) is not None:
                feed[name] = hight_position_ids[batch_idx : batch_idx + 1].to(self.device)
            elif (batch_idx := _batch_suffix(name, "width_position_ids")) is not None:
                feed[name] = width_position_ids[batch_idx : batch_idx + 1].to(self.device)
            elif name == inputs_name:
                feed[name] = inputs_embeds.to(self.device)
            elif name == "time_position_ids":
                feed[name] = time_position_ids.to(self.device)
            elif name == "hight_position_ids":
                feed[name] = hight_position_ids.to(self.device)
            elif name == "width_position_ids":
                feed[name] = width_position_ids.to(self.device)
            elif name.startswith("past_key_cache_"):
                layer_idx, batch_idx = _parse_layer_batch_suffix(name, "past_key_cache_")
                cache_idx = layer_idx * self.batch_size + batch_idx if batch_idx is not None else layer_idx
                feed[name] = past_key_caches[cache_idx]
            elif name.startswith("past_value_cache_"):
                layer_idx, batch_idx = _parse_layer_batch_suffix(name, "past_value_cache_")
                cache_idx = layer_idx * self.batch_size + batch_idx if batch_idx is not None else layer_idx
                feed[name] = past_value_caches[cache_idx]
            elif name.startswith("past_conv_cache_"):
                branch, idx, batch_idx = _parse_conv_cache_name_moe(name)
                if branch is None:
                    cache_idx = idx * self.batch_size + batch_idx if batch_idx is not None else idx
                else:
                    offset = {"q": 0, "k": 1, "v": 2}[branch]
                    cache_idx = idx * 3 * self.batch_size + offset * self.batch_size + batch_idx if batch_idx is not None else idx * 3 + offset
                feed[name] = past_conv_caches[cache_idx]
            elif name.startswith("past_recurrent_state_"):
                layer_idx, batch_idx = _parse_layer_batch_suffix(name, "past_recurrent_state_")
                cache_idx = layer_idx * self.batch_size + batch_idx if batch_idx is not None else layer_idx
                feed[name] = past_recurrent_states[cache_idx]
            else:
                info = session.get_input(name)
                feed[name] = torch.zeros(info.shape, dtype=info.dtype, device=self.device)

        conv_cache_names = [
            n for n in session.get_input_names() if n.startswith("past_conv_cache_")
        ]
        recurrent_cache_names = [
            n for n in session.get_input_names() if n.startswith("past_recurrent_state_")
        ]
        _, output_map = self._run_hmonnx(session, feed)
        linear_cache_state = {
            **{name: past_conv_caches[i] for i, name in enumerate(conv_cache_names)},
            **{
                name: past_recurrent_states[i]
                for i, name in enumerate(recurrent_cache_names)
            },
        }
        self._update_linear_cache_moe(linear_cache_state, output_map)
        for i, name in enumerate(conv_cache_names):
            past_conv_caches[i] = linear_cache_state[name]
        for i, name in enumerate(recurrent_cache_names):
            past_recurrent_states[i] = linear_cache_state[name]
        return self._extract_logits_moe(output_map)

    def _extract_logits_moe(self, output_map: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "logits" in output_map:
            return output_map["logits"]
        split_logits = [
            output_map[f"logits_batch_{batch_idx}"]
            for batch_idx in range(self.batch_size)
            if f"logits_batch_{batch_idx}" in output_map
        ]
        if split_logits:
            return torch.cat(split_logits, dim=0)
        return self._extract_logits(output_map)

    def _update_linear_cache_moe(
        self, cache_state: Dict[str, torch.Tensor], output_map: Dict[str, torch.Tensor]
    ) -> None:
        for name in list(cache_state.keys()):
            if name in output_map:
                cache_state[name] = _as_cache_value(cache_state[name], output_map[name])
                continue
            if name.startswith("past_conv_cache_"):
                suffix = name[len("past_conv_cache_"):]
                direct = f"conv_cache_out_{suffix}"
                candidates = [direct]
                if "_batch_" in suffix:
                    base, batch = suffix.rsplit("_batch_", 1)
                    candidates.append(f"conv_cache_out_{base}_0_batch_{batch}")
                else:
                    candidates.append(f"conv_cache_out_{suffix}_0")
                for out_name in candidates:
                    if out_name in output_map:
                        conv_out = output_map[out_name]
                        if conv_out.dim() >= 3 and conv_out.shape[-1] != cache_state[name].shape[-1]:
                            conv_out = conv_out[..., : cache_state[name].shape[-1]]
                        cache_state[name] = _as_cache_value(cache_state[name], conv_out)
                        break
            elif name.startswith("past_recurrent_state_"):
                suffix = name[len("past_recurrent_state_"):]
                candidates = [f"recurrent_state_out_{suffix}"]
                if "_batch_" in suffix:
                    base, batch = suffix.rsplit("_batch_", 1)
                    candidates.append(f"recurrent_state_out_{base}_0_batch_{batch}")
                else:
                    candidates.append(f"recurrent_state_out_{suffix}_0")
                for out_name in candidates:
                    if out_name in output_map:
                        cache_state[name] = _as_cache_value(cache_state[name], output_map[out_name])
                        break

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_seq_length: torch.Tensor,
        current_input_length: torch.Tensor,
        past_key_caches: List[CacheTensor],
        past_value_caches: List[CacheTensor],
    ) -> torch.Tensor:
        seq_len = int(inputs_embeds.shape[1])
        past_len = int(past_seq_length.item())
        position_ids = torch.arange(
            past_len, past_len + seq_len, dtype=torch.int32, device=inputs_embeds.device
        ).unsqueeze(0)
        linear_attn_mask = _build_runtime_linear_attn_mask(
            seq_len, seq_len, inputs_embeds.dtype, inputs_embeds.device
        )
        return self._forward(
            inputs_embeds,
            position_ids,
            position_ids,
            position_ids,
            past_seq_length,
            current_input_length,
            linear_attn_mask,
            past_key_caches,
            past_value_caches,
            self.past_conv_caches,
            self.past_recurrent_states,
        )

    @torch.no_grad()
    def generate(self, *args, **kwargs):
        if args and isinstance(args[0], torch.Tensor):
            return super().generate(*args, **kwargs)

        messages = args[0] if args else kwargs.pop("messages")
        enable_thinking = kwargs.pop("enable_thinking", False)
        max_new_tokens = kwargs.pop("max_new_tokens", 256)
        do_sample = kwargs.pop("do_sample", False)
        temperature = kwargs.pop("temperature", 1.0)
        top_p = kwargs.pop("top_p", 1.0)
        top_k = kwargs.pop("top_k", 0)
        repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        presence_penalty = kwargs.pop("presence_penalty", 0.0)
        stream_output = kwargs.pop("stream_output", False)
        if kwargs:
            raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs.keys())}")

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=enable_thinking,
            add_generation_prompt=True,
        )
        input_ids = self.tokenizer(
            [text] if isinstance(text, str) else text,
            padding=False,
            return_tensors="pt",
        ).input_ids
        output_text = super().generate(
            input_ids,
            self.tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            presence_penalty=presence_penalty,
            stream_output=stream_output,
        )
        output_token_ids = self.tokenizer.encode(output_text, add_special_tokens=False)
        return output_token_ids, output_text

    @torch.no_grad()
    def prefill_only(self, input_ids: torch.Tensor) -> Optional[torch.Tensor]:
        if input_ids.dim() != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"input_ids must be [1, seq], got {tuple(input_ids.shape)}")
        seq_len = int(input_ids.shape[1])
        if seq_len == 0:
            return None

        self._ensure_prefill_session()
        prefill_chunk_len = int(self._prefill_inputs_info.shape[1])
        all_logits: List[torch.Tensor] = []
        for start in range(0, seq_len, prefill_chunk_len):
            end = min(start + prefill_chunk_len, seq_len)
            chunk_ids = input_ids[:, start:end]
            valid_len = int(chunk_ids.shape[1])
            cache_state = _alloc_cache_inputs(self.prefill_session, self.device)
            prefill_feed = self._build_prefill_feed(chunk_ids, valid_len, 0, cache_state)
            _, output_map = self._run_hmonnx(self.prefill_session, prefill_feed)
            logits = self._extract_logits(output_map)
            all_logits.append(logits[:, :valid_len, :])
        return torch.cat(all_logits, dim=1) if all_logits else None


__all__ = [
    "Qwen3_5MoeInference",
    "_build_runtime_linear_attn_mask",
    "_load_meta_artifacts",
    "_load_token_embedding",
    "_resolve_prefill_input_sequence_length",
    "_resolve_path",
]
