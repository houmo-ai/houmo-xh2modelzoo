#!/usr/bin/env python3
"""Gemma4 Series MTP HMONNX verifier and greedy speculative runner.

Series-local utility (do not use legacy gemma4/gemma4e/gemma4_moe paths):

* ``verify`` checks the exported target/draft graph contract:
  target emits logits + post-norm hidden only; draft consumes shared KV tensors
  from the target runtime cache; draft KVcache nodes are read-only via
  ``only_handle_old_cache=True``; the assistant head is W4 while the draft body is W8.
* ``generate`` runs a small greedy MTP loop on HMONNX target + ONNX assistant.
  Each verify round feeds ``[current_token] + draft_tokens`` to the target decode
  graph, so the exported decode sequence length must be ``num_draft_tokens + 1``.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import ExitStack
import json
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from types import SimpleNamespace

import onnx

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXPECTED_BASE_OUTPUTS = ["logits", "target_hidden_state"]
EXPECTED_DRAFT_INPUTS = [
    "input_1",
    "valid_length",
    "current_length",
    "sliding_attention_mask",
    "full_attention_mask",
    "shared_key_cache_sliding",
    "shared_value_cache_sliding",
    "shared_key_cache_full",
    "shared_value_cache_full",
]
EXPECTED_DRAFT_OUTPUTS = ["logits", "assistant_hidden_state"]


@dataclass(frozen=True)
class Preset:
    name: str
    slug: str
    assistant_kind: str


PRESETS: dict[str, Preset] = {
    "e2b": Preset("e2b", "gemma4_e2b_unified", "ordered_embedding"),
    "e4b": Preset("e4b", "gemma4_e4b_unified", "ordered_embedding"),
    "26b-a4b": Preset("26b-a4b", "gemma4_26b_a4b_unified", "dense_lm_head"),
    "31b": Preset("31b", "gemma4_31b_unified", "dense_lm_head"),
}
DEFAULT_HF_DIRS: dict[str, tuple[str, str]] = {
    "e2b": ("weights/gemma-4-E2B-it", "weights/gemma-4-E2B-it-assistant"),
    "e4b": ("weights/gemma-4-E4B-it", "weights/gemma-4-E4B-it-assistant"),
    "26b-a4b": ("weights/gemma-4-26B-A4B-it", "weights/gemma-4-26B-A4B-it-assistant"),
    "31b": ("weights/gemma-4-31B-it", "weights/gemma-4-31B-it-assistant"),
}


def _attr_value(attr: onnx.AttributeProto) -> Any:
    value = onnx.helper.get_attribute_value(attr)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _node_attrs(node: onnx.NodeProto) -> dict[str, Any]:
    return {attr.name: _attr_value(attr) for attr in node.attribute}


def _onnx_dim(dim) -> int:
    if getattr(dim, "dim_value", 0):
        return int(dim.dim_value)
    raise AssertionError(f"Dynamic ONNX dimension is not supported in Gemma4 MTP contract checks: {dim}")


def _onnx_value_shape(model: onnx.ModelProto, name: str) -> list[int]:
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if value.name == name:
            return [_onnx_dim(dim) for dim in value.type.tensor_type.shape.dim]
    raise AssertionError(f"{name!r} not found in ONNX graph")


def _aligned(size: int, alignment: int = 16) -> int:
    return ((int(size) + int(alignment) - 1) // int(alignment)) * int(alignment)


def _resolve_relative_path(path_value: str, base_dir: Path) -> Path:
    path = Path(str(path_value))
    return path if path.is_absolute() else base_dir / path


def _meta_spec_decode(meta_dict: dict[str, Any]) -> dict[str, Any]:
    spec = dict(meta_dict.get("spec_decode") or {})
    if not spec and meta_dict.get("spec_decode_mode"):
        spec = {
            "mode": meta_dict.get("spec_decode_mode"),
            "block_size": meta_dict.get("spec_decode_block_size"),
            "verify_length": meta_dict.get("spec_decode_verify_length"),
            "hidden_output_name": meta_dict.get("spec_decode_hidden_output_name"),
            "draft_head_weight_bits": meta_dict.get("spec_decode_draft_head_weight_bits"),
        }
    return spec


def _latest_export_dir(base_export_root: Path) -> Path:
    if (base_export_root / "golden_meta_info.json").exists():
        return base_export_root
    candidates = sorted(path for path in base_export_root.iterdir() if path.is_dir())
    if not candidates:
        raise FileNotFoundError(f"No export directory found under {base_export_root}")
    return candidates[-1]


def _resolve_meta_path(preset: Preset, base_root: Path, meta: str | None) -> Path:
    if meta:
        meta_path = Path(meta).expanduser().resolve()
        if meta_path.is_dir():
            meta_path = meta_path / "golden_meta_info.json"
        if not meta_path.exists():
            raise FileNotFoundError(meta_path)
        return meta_path
    return _latest_export_dir(base_root / preset.slug / "existing-hf" / "export") / "golden_meta_info.json"


def _resolve_draft_onnx(
    preset: Preset,
    draft_root: Path,
    draft_onnx: str | None,
    *,
    meta_path: Path | None = None,
    meta_dict: dict[str, Any] | None = None,
) -> Path:
    if draft_onnx:
        path = Path(draft_onnx).expanduser().resolve()
    elif meta_path is not None and meta_dict is not None:
        spec = _meta_spec_decode(meta_dict)
        rel = (
            spec.get("draft_decode_onnx")
            or spec.get("draft_onnx")
            or meta_dict.get("draft_decode_onnx_file")
            or meta_dict.get("draft_onnx_file")
        )
        if not rel:
            raise FileNotFoundError(
                f"{preset.name}: manifest {meta_path} does not record spec_decode.draft_decode_onnx"
            )
        path = _resolve_relative_path(str(rel), meta_path.parent)
    else:
        path = draft_root / preset.name / "mtp_draft_decode" / f"gemma4_series_{preset.name}_assistant_decode.onnx"
        if not path.exists():
            path = draft_root / preset.name / "draft_onnx" / f"gemma4_series_{preset.name}_assistant_decode.onnx"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def _load_outputs(onnx_path: Path) -> list[str]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    return [out.name for out in model.graph.output]


def _load_inputs(onnx_path: Path) -> list[str]:
    model = onnx.load(str(onnx_path), load_external_data=False)
    return [inp.name for inp in model.graph.input]


def _quant_counter(model: onnx.ModelProto) -> Counter[tuple[int, int, str]]:
    counter: Counter[tuple[int, int, str]] = Counter()
    for node in model.graph.node:
        attrs = _node_attrs(node)
        if "hmfp_weight_man_bit" in attrs:
            counter[(int(attrs["hmfp_weight_man_bit"]), int(attrs.get("hmfp_weight_hidden_bit", -1)), str(attrs.get("mode", "")))] += 1
    return counter


def _w4_nodes(model: onnx.ModelProto) -> list[tuple[str, str, dict[str, Any]]]:
    hits = []
    for node in model.graph.node:
        attrs = _node_attrs(node)
        if attrs.get("hmfp_weight_man_bit") == 4:
            hits.append((node.name, node.op_type, attrs))
    return hits


def _kv_nodes(model: onnx.ModelProto) -> list[tuple[str, str, dict[str, Any], list[str]]]:
    hits = []
    for node in model.graph.node:
        if node.op_type == "KVcache" or "cache" in node.op_type.lower():
            hits.append((node.name, node.op_type, _node_attrs(node), list(node.input)))
    return hits


def _target_decode_onnx_path(meta_path: Path, meta_dict: dict[str, Any]) -> Path:
    export_dir = meta_path.parent
    if meta_dict.get("decode_hmonnx"):
        return _resolve_relative_path(meta_dict["decode_hmonnx"], export_dir)
    return next((export_dir / "decode").glob("*.onnx"))


def _first_sliding_kv_trace_info(meta_path: Path, meta_dict: dict[str, Any]) -> dict[str, Any] | None:
    decode_onnx = _target_decode_onnx_path(meta_path, meta_dict)
    decode_model = onnx.load(str(decode_onnx), load_external_data=False)
    for node_name, op_type, attrs, node_inputs in _kv_nodes(decode_model):
        attention_max_length = int(attrs.get("attention_max_length", attrs.get("attention-max-length", -1)))
        if attention_max_length <= 0:
            continue
        return {
            "decode_onnx": str(decode_onnx),
            "node_name": node_name,
            "op_type": op_type,
            "attention_max_length": attention_max_length,
            "node_inputs": node_inputs,
            "accepted_count_input_index": node_inputs.index("accepted_count") if "accepted_count" in node_inputs else None,
        }
    return None


def verify_preset(preset: Preset, base_root: Path, draft_root: Path, *, meta: str | None = None, draft_onnx: str | None = None) -> dict[str, Any]:
    meta_path = _resolve_meta_path(preset, base_root, meta)
    export_dir = meta_path.parent
    meta_dict = json.loads(meta_path.read_text(encoding="utf-8"))
    model_config = meta_dict.get("model_config", {})
    spec = _meta_spec_decode(meta_dict)
    if not model_config.get("enable_mtp_outputs"):
        raise AssertionError(f"{preset.name}: base export did not enable MTP outputs")
    if spec.get("mode") != "mtp" or meta_dict.get("spec_decode_mode") != "mtp":
        raise AssertionError(f"{preset.name}: manifest does not declare MTP spec_decode: {spec}")
    num_draft_tokens = int(spec.get("block_size") or model_config.get("num_draft_tokens") or 4)
    verify_length = int(spec.get("verify_length") or meta_dict.get("spec_decode_verify_length") or 0)
    if verify_length != num_draft_tokens + 1:
        raise AssertionError(
            f"{preset.name}: verify_length must be num_draft_tokens + 1, got {verify_length} vs {num_draft_tokens}"
        )

    prefill_onnx = _resolve_relative_path(meta_dict.get("prefill_hmonnx", ""), export_dir) if meta_dict.get("prefill_hmonnx") else next((export_dir / "prefill").glob("*.onnx"))
    decode_onnx = _target_decode_onnx_path(meta_path, meta_dict)
    prefill_model = onnx.load(str(prefill_onnx), load_external_data=False)
    decode_model = onnx.load(str(decode_onnx), load_external_data=False)
    prefill_outputs = [out.name for out in prefill_model.graph.output]
    decode_outputs = [out.name for out in decode_model.graph.output]
    if prefill_outputs != EXPECTED_BASE_OUTPUTS:
        raise AssertionError(f"{preset.name}: prefill outputs mismatch: {prefill_outputs}")
    if decode_outputs != EXPECTED_BASE_OUTPUTS:
        raise AssertionError(f"{preset.name}: decode outputs mismatch: {decode_outputs}")
    decode_input_shape = _onnx_value_shape(decode_model, "input_1")
    decode_inputs = _load_inputs(decode_onnx)
    if "accepted_count" not in decode_inputs:
        raise AssertionError(f"{preset.name}: target MTP verify decode graph is missing accepted_count input")
    decode_kv_nodes = _kv_nodes(decode_model)
    if not decode_kv_nodes:
        raise AssertionError(f"{preset.name}: target MTP verify decode graph has no KVcache nodes")
    sliding_kv_nodes = []
    full_kv_nodes = []
    for node_name, _, attrs, node_inputs in decode_kv_nodes:
        attention_max_length = int(attrs.get("attention_max_length", attrs.get("attention-max-length", -1)))
        if attention_max_length > 0:
            sliding_kv_nodes.append(node_name)
            if "accepted_count" not in node_inputs:
                raise AssertionError(
                    f"{preset.name}: target sliding decode KVcache {node_name} does not consume accepted_count"
                )
        else:
            full_kv_nodes.append(node_name)
            if "accepted_count" in node_inputs:
                raise AssertionError(
                    f"{preset.name}: target full decode KVcache {node_name} must not consume accepted_count"
                )
    if not sliding_kv_nodes:
        raise AssertionError(f"{preset.name}: target MTP verify decode graph has no sliding KVcache nodes")
    if not full_kv_nodes:
        raise AssertionError(f"{preset.name}: target MTP verify decode graph has no full-attention KVcache nodes")
    if decode_input_shape[1] != verify_length:
        raise AssertionError(
            f"{preset.name}: target decode input length must be verify_length={verify_length}, got {decode_input_shape}"
        )
    decode_logits_shape = _onnx_value_shape(decode_model, "logits")
    if decode_logits_shape[1] != verify_length:
        raise AssertionError(
            f"{preset.name}: target decode logits length must be verify_length={verify_length}, got {decode_logits_shape}"
        )
    sliding_window = int(meta_dict.get("sliding_window") or model_config.get("sliding_window") or 0)
    target_decode_sliding_width = int(
        spec.get("target_decode_sliding_output_length")
        or (_aligned(sliding_window + verify_length - 1, 16) if sliding_window > 0 else 0)
    )
    decode_sliding_mask_shape = _onnx_value_shape(decode_model, "sliding_attention_mask")
    if target_decode_sliding_width and decode_sliding_mask_shape[-1] != target_decode_sliding_width:
        raise AssertionError(
            f"{preset.name}: target decode sliding mask width must be LLMCache compact output "
            f"{target_decode_sliding_width}, got {decode_sliding_mask_shape}"
        )
    layer_shapes = meta_dict.get("layer_kv_shapes") or []
    context_length = int(model_config.get("context_max_length") or 0)
    shape_lengths = [int(shape[2]) for shape in layer_shapes if len(shape) > 2]
    sliding_candidates = [length for length in shape_lengths if context_length <= 0 or length < context_length]
    shared_sliding_len = int(spec.get("shared_sliding_cache_length") or max(sliding_candidates or shape_lengths))
    shared_full_len = int(spec.get("shared_full_cache_length") or context_length or max(shape_lengths))
    sliding_shape = next((shape for shape in layer_shapes if len(shape) > 2 and int(shape[2]) == shared_sliding_len), None)
    full_shape = next((shape for shape in layer_shapes if len(shape) > 2 and int(shape[2]) == shared_full_len), None)
    if sliding_shape is None or full_shape is None:
        raise AssertionError(
            f"{preset.name}: manifest layer_kv_shapes cannot locate shared KV lengths "
            f"sliding={shared_sliding_len}, full={shared_full_len}: {layer_shapes}"
        )
    target_shared_shapes = {
        "shared_key_cache_sliding": list(sliding_shape),
        "shared_value_cache_sliding": list(sliding_shape),
        "shared_key_cache_full": list(full_shape),
        "shared_value_cache_full": list(full_shape),
    }

    draft_path = _resolve_draft_onnx(preset, draft_root, draft_onnx, meta_path=meta_path, meta_dict=meta_dict)
    draft_model = onnx.load(str(draft_path), load_external_data=False)
    draft_inputs = [inp.name for inp in draft_model.graph.input]
    draft_outputs = [out.name for out in draft_model.graph.output]
    if draft_inputs != EXPECTED_DRAFT_INPUTS:
        raise AssertionError(f"{preset.name}: draft inputs mismatch: {draft_inputs}")
    if draft_outputs != EXPECTED_DRAFT_OUTPUTS:
        raise AssertionError(f"{preset.name}: draft outputs mismatch: {draft_outputs}")
    for name, target_shape in target_shared_shapes.items():
        draft_shape = _onnx_value_shape(draft_model, name)
        if draft_shape != target_shape:
            raise AssertionError(
                f"{preset.name}: draft {name} shape must match manifest shared KV cache for reuse; "
                f"draft={draft_shape}, target={target_shape}"
            )
    if _onnx_value_shape(draft_model, "sliding_attention_mask")[-1] != target_shared_shapes["shared_key_cache_sliding"][2]:
        raise AssertionError(f"{preset.name}: draft sliding mask width does not match shared sliding KV length")
    if _onnx_value_shape(draft_model, "full_attention_mask")[-1] != target_shared_shapes["shared_key_cache_full"][2]:
        raise AssertionError(f"{preset.name}: draft full mask width does not match shared full KV length")

    kv_nodes = _kv_nodes(draft_model)
    if len(kv_nodes) != 4:
        raise AssertionError(f"{preset.name}: expected 4 KVcache nodes, got {len(kv_nodes)}")
    for node_name, _, attrs, _ in kv_nodes:
        if attrs.get("only_handle_old_cache") != 1:
            raise AssertionError(f"{preset.name}: {node_name} missing only_handle_old_cache=1: {attrs}")
        if "passthrough" in attrs:
            raise AssertionError(f"{preset.name}: {node_name} still has legacy passthrough attr")

    w4_nodes = _w4_nodes(draft_model)
    if len(w4_nodes) != 1:
        raise AssertionError(f"{preset.name}: expected exactly one W4 head node, got {len(w4_nodes)}")
    quant_counter = _quant_counter(draft_model)
    if quant_counter[(4, 0, "ssfp")] != 1:
        raise AssertionError(f"{preset.name}: missing W4A8H0 SSFP head: {quant_counter}")
    if quant_counter[(8, 1, "sefp")] <= 0:
        raise AssertionError(f"{preset.name}: missing W8A8H1 SEFP draft body: {quant_counter}")

    return {
        "preset": preset.name,
        "assistant_kind": preset.assistant_kind,
        "manifest": str(meta_path),
        "base_export": str(export_dir),
        "verify_length": verify_length,
        "num_draft_tokens": num_draft_tokens,
        "prefill_inputs": _load_inputs(prefill_onnx),
        "decode_inputs": decode_inputs,
        "decode_input_shape": decode_input_shape,
        "decode_logits_shape": decode_logits_shape,
        "target_shared_shapes": target_shared_shapes,
        "base_outputs": decode_outputs,
        "draft_onnx": str(draft_path),
        "draft_inputs": draft_inputs,
        "draft_outputs": draft_outputs,
        "draft_shared_shapes": {name: _onnx_value_shape(draft_model, name) for name in target_shared_shapes},
        "kv_cache_attrs": [attrs for _, _, attrs, _ in kv_nodes],
        "target_kv_cache_inputs": {node_name: node_inputs for node_name, _, _, node_inputs in decode_kv_nodes},
        "w4_head_node": {"name": w4_nodes[0][0], "op_type": w4_nodes[0][1], "attrs": w4_nodes[0][2]},
        "quant_counter": {str(key): value for key, value in sorted(quant_counter.items())},
    }


def _tensor_dtype_name(dtype) -> str:
    return str(dtype).replace("torch.", "")


def _torch_dtype(dtype_name: str):
    import torch

    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    key = str(dtype_name).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_name!r}; choose one of {sorted(mapping)}")
    return mapping[key]


def _to_namespace(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _to_namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_namespace(item) for item in value]
    return value


class AssistantDraftSession:
    """Thin HMONNXGraph wrapper for the exported assistant ONNX."""

    def __init__(self, onnx_path: str | Path, device):
        from xhquant.xhonnxruntime import HMONNXGrapInference

        self.session = HMONNXGrapInference(str(onnx_path))
        if device.type != "cpu":
            self.session.to(device)
        self.session.exec_device = device
        self.input_names = self.session.get_input_names()
        self.output_names = self.session.get_output_names()
        self.input_infos = {name: self.session.get_input(name) for name in self.input_names}
        self.input_aliases = {
            "input_1": "inputs_embeds",
            "valid_length": "past_seq_length",
            "current_length": "current_length",
        }
        self.device = device

    def __call__(self, **inputs):
        import torch

        feed = {}
        for name in self.input_names:
            source_name = name if name in inputs else self.input_aliases.get(name)
            if source_name is None or source_name not in inputs:
                raise KeyError(f"Missing assistant input {name!r}; available={sorted(inputs)}")
            value = inputs[source_name]
            if isinstance(value, torch.Tensor):
                value = value.detach().to(self.device)
                expected_shape = tuple(int(dim) for dim in getattr(self.input_infos[name], "shape", ()) or ())
                if expected_shape and tuple(value.shape) != expected_shape:
                    raise RuntimeError(
                        f"Assistant input {name!r} shape mismatch: got {tuple(value.shape)}, "
                        f"expected {expected_shape}. Re-export target and draft from one manifest instead of slicing caches."
                    )
                expected_dtype = self.input_infos[name].dtype
                if value.dtype != expected_dtype:
                    value = value.to(expected_dtype)
            feed[name] = value
        outputs = self.session.run(feed)
        if isinstance(outputs, dict):
            output_map = outputs
        else:
            if not isinstance(outputs, (tuple, list)):
                outputs = (outputs,)
            output_map = {name: output for name, output in zip(self.output_names, outputs)}
        return output_map[self.output_names[0]], output_map[self.output_names[1]]


class TorchAssistantDraftSession:
    """Eager fp assistant draft session for acceptance-rate diagnosis.

    It intentionally implements the same small callable surface as
    :class:`AssistantDraftSession` so the speculative loop can isolate the
    effect of draft quantization while keeping the target prefill/verify path
    unchanged.
    """

    def __init__(
        self,
        *,
        assistant_model_dir: str | Path,
        target_model_dir: str | Path,
        meta: dict[str, Any],
        device,
        dtype,
    ):
        import torch
        from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model import Gemma4AssistantDraftModule

        model_config = meta.get("model_config", {})
        spec = _meta_spec_decode(meta)
        context_length = int(model_config.get("context_max_length") or model_config.get("max_sequence_length") or 2048)
        input_sequence_length = int(model_config.get("mtp_config", {}).get("input_sequence_length") or 1)
        self.module = Gemma4AssistantDraftModule(
            assistant_model_dir=str(assistant_model_dir),
            target_model_dir=str(target_model_dir),
            max_position_embeddings=context_length,
            input_sequence_length=input_sequence_length,
            cache_axis=int(meta.get("kv_cache", {}).get("cache_axis") or 2),
        ).to(device=device, dtype=dtype)
        self.module.eval()
        self.device = device
        self.dtype = dtype
        shared_sliding = int(
            spec.get("shared_sliding_cache_length")
            or meta.get("kv_cache", {}).get("kv_cache_shape", [0, 0, 0, 0])[2]
        )
        shared_full = int(spec.get("shared_full_cache_length") or model_config.get("context_max_length") or context_length)
        target_text = json.loads((Path(target_model_dir) / "config.json").read_text(encoding="utf-8"))["text_config"]
        full_heads = int(target_text.get("num_global_key_value_heads") or target_text["num_key_value_heads"])
        full_dim = int(target_text.get("global_head_dim") or target_text["head_dim"])
        sliding_heads = int(target_text["num_key_value_heads"])
        sliding_dim = int(target_text["head_dim"])
        hidden_size = int(self.module.backbone_hidden_size)
        self.input_names = list(EXPECTED_DRAFT_INPUTS)
        self.output_names = list(EXPECTED_DRAFT_OUTPUTS)
        self.input_infos = {
            "input_1": SimpleNamespace(shape=(1, input_sequence_length, hidden_size * 2), dtype=dtype),
            "valid_length": SimpleNamespace(shape=(1,), dtype=torch.int32),
            "current_length": SimpleNamespace(shape=(1,), dtype=torch.int32),
            "sliding_attention_mask": SimpleNamespace(shape=(1, 1, input_sequence_length, shared_sliding), dtype=dtype),
            "full_attention_mask": SimpleNamespace(shape=(1, 1, input_sequence_length, shared_full), dtype=dtype),
            "shared_key_cache_sliding": SimpleNamespace(
                shape=(1, sliding_heads, shared_sliding, sliding_dim), dtype=dtype
            ),
            "shared_value_cache_sliding": SimpleNamespace(
                shape=(1, sliding_heads, shared_sliding, sliding_dim), dtype=dtype
            ),
            "shared_key_cache_full": SimpleNamespace(shape=(1, full_heads, shared_full, full_dim), dtype=dtype),
            "shared_value_cache_full": SimpleNamespace(shape=(1, full_heads, shared_full, full_dim), dtype=dtype),
        }
        self.input_aliases = {
            "input_1": "inputs_embeds",
            "valid_length": "past_seq_length",
            "current_length": "current_length",
        }

    def __call__(self, **inputs):
        import torch

        feed = {}
        for name in self.input_names:
            source_name = name if name in inputs else self.input_aliases.get(name)
            if source_name is None or source_name not in inputs:
                raise KeyError(f"Missing fp assistant input {name!r}; available={sorted(inputs)}")
            value = inputs[source_name]
            if isinstance(value, torch.Tensor):
                expected = self.input_infos[name]
                value = value.detach().to(self.device)
                if value.dtype != expected.dtype:
                    value = value.to(expected.dtype)
                if tuple(value.shape) != tuple(expected.shape):
                    raise RuntimeError(
                        f"FP assistant input {name!r} shape mismatch: got {tuple(value.shape)}, "
                        f"expected {tuple(expected.shape)}"
                    )
            feed[name] = value
        return self.module(
            feed["input_1"],
            feed["valid_length"],
            feed["current_length"],
            feed["sliding_attention_mask"],
            feed["full_attention_mask"],
            feed["shared_key_cache_sliding"],
            feed["shared_value_cache_sliding"],
            feed["shared_key_cache_full"],
            feed["shared_value_cache_full"],
        )


class TorchTargetSession:
    """Eager fp target verifier with Gemma4 Series MTP shared-cache outputs.

    This is a diagnosis-only backend for answering whether the low acceptance
    rate comes from quantized target prefill/verify.  It keeps the speculative
    loop unchanged and adapts HF ``DynamicCache`` into the same shared
    sliding/full cache tensors consumed by the exported assistant.
    """

    def __init__(self, *, target_model_dir: str | Path, meta: dict[str, Any], device, dtype):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_dir = str(target_model_dir)
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            dtype=dtype,
            device_map=str(device),
            trust_remote_code=True,
        ).eval()
        if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] < 9:
            for cfg in (getattr(self.model, "config", None), getattr(getattr(self.model, "config", None), "text_config", None)):
                if cfg is not None and hasattr(cfg, "_experts_implementation"):
                    cfg._experts_implementation = "batched_mm"
        self.config_dict = json.loads((Path(self.model_dir) / "config.json").read_text(encoding="utf-8"))
        self.text_config = self.config_dict["text_config"]
        self.meta_info = SimpleNamespace(model_config=_to_namespace(meta.get("model_config", {})))
        self.sliding_window = int(self.text_config.get("sliding_window") or meta.get("sliding_window") or 1024)
        spec = _meta_spec_decode(meta)
        self.shared_sliding = int(
            spec.get("shared_sliding_cache_length")
            or meta.get("kv_cache", {}).get("kv_cache_shape", [0, 0, 0, 0])[2]
        )
        self.shared_full = int(spec.get("shared_full_cache_length") or meta.get("model_config", {}).get("context_max_length") or 2048)
        self._cache = None
        self._pre_verify_cache = None
        self._last_verify_ids: list[int] = []
        self._last_verify_past_seq_length = 0
        self._source_cache_indices = self._find_shared_source_cache_indices()

    def get_tokenizer(self, **kwargs):
        del kwargs
        return self.tokenizer

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def _find_shared_source_cache_indices(self) -> dict[str, int]:
        layers = self.model.model.language_model.layers
        layer_types = list(self.text_config["layer_types"])
        layer_to_cache_idx: dict[int, int] = {}
        cache_idx = 0
        source: dict[str, int] = {}
        for layer_idx, layer in enumerate(layers):
            attn = layer.self_attn
            if not bool(getattr(attn, "is_kv_shared_layer", False)):
                layer_to_cache_idx[layer_idx] = cache_idx
                cache_idx += 1
            if bool(getattr(attn, "store_full_length_kv", False)):
                layer_type = str(getattr(attn, "layer_type", None) or layer_types[layer_idx])
                source[layer_type] = layer_to_cache_idx[layer_idx]
        if "sliding_attention" not in source or "full_attention" not in source:
            # Dense 26B/31B mark the final sliding/full layers.  If a local HF
            # revision lacks the helper attrs, fall back to the last layer of
            # each type that owns a cache.
            for layer_idx, layer_type in enumerate(layer_types):
                if layer_idx in layer_to_cache_idx:
                    source[str(layer_type)] = layer_to_cache_idx[layer_idx]
        if "sliding_attention" not in source or "full_attention" not in source:
            raise RuntimeError(f"Cannot locate Gemma4 shared KV source layers: {source}")
        return source

    def _pad_cache(self, tensor, width: int):
        import torch

        out = torch.zeros((tensor.shape[0], tensor.shape[1], width, tensor.shape[3]), dtype=tensor.dtype, device=tensor.device)
        take = min(int(tensor.shape[2]), int(width))
        if take > 0:
            out[:, :, :take, :] = tensor[:, :, -take:, :]
        return out

    def _shared_from_cache(self) -> dict[str, Any]:
        if self._cache is None:
            raise RuntimeError("FP target cache is not initialized")
        sliding_layer = self._cache.layers[self._source_cache_indices["sliding_attention"]]
        full_layer = self._cache.layers[self._source_cache_indices["full_attention"]]
        return {
            "shared_key_cache_sliding": self._pad_cache(sliding_layer.keys, self.shared_sliding),
            "shared_value_cache_sliding": self._pad_cache(sliding_layer.values, self.shared_sliding),
            "shared_key_cache_full": self._pad_cache(full_layer.keys, self.shared_full),
            "shared_value_cache_full": self._pad_cache(full_layer.values, self.shared_full),
        }

    def _full_cache_length(self) -> int:
        if self._cache is None:
            return 0
        full_layer = self._cache.layers[self._source_cache_indices["full_attention"]]
        return int(full_layer.keys.shape[2])

    def run_prefill_ids(self, input_ids):
        import torch

        with torch.inference_mode():
            outputs = self.model(
                input_ids=input_ids.to(self.device),
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        self._cache = outputs.past_key_values
        return outputs.logits, outputs.hidden_states[-1][:, -1:, :], self._shared_from_cache(), int(input_ids.shape[1])

    def run_verify_ids(self, token_ids: list[int], past_seq_length: int):
        import torch

        if self._cache is not None and self._full_cache_length() > int(past_seq_length):
            self._cache.crop(int(past_seq_length))
        self._pre_verify_cache = copy.deepcopy(self._cache)
        self._last_verify_ids = list(map(int, token_ids))
        self._last_verify_past_seq_length = int(past_seq_length)
        tokens = torch.tensor([list(map(int, token_ids))], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            outputs = self.model(
                input_ids=tokens,
                past_key_values=self._cache,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )
        self._cache = outputs.past_key_values
        return outputs.logits, outputs.hidden_states[-1], self._shared_from_cache()

    def commit_cache(self, valid_length: int) -> dict[str, Any]:
        valid_length = int(valid_length)
        accepted_steps = valid_length - int(self._last_verify_past_seq_length)
        if (
            self._pre_verify_cache is not None
            and 0 <= accepted_steps < len(self._last_verify_ids)
        ):
            import torch

            self._cache = copy.deepcopy(self._pre_verify_cache)
            if accepted_steps > 0:
                tokens = torch.tensor(
                    [self._last_verify_ids[:accepted_steps]],
                    dtype=torch.long,
                    device=self.device,
                )
                with torch.inference_mode():
                    outputs = self.model(
                        input_ids=tokens,
                        past_key_values=self._cache,
                        use_cache=True,
                        output_hidden_states=False,
                        return_dict=True,
                    )
                self._cache = outputs.past_key_values
            return self._shared_from_cache()
        if self._cache is not None and self._full_cache_length() > valid_length:
            self._cache.crop(int(valid_length))
        return self._shared_from_cache()


def _load_prompt_inputs(tokenizer, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = prompt
    return tokenizer([text], return_tensors="pt").input_ids


def _read_json_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON config {path}: {exc}") from exc


def _generation_config_candidates(meta: dict[str, Any]) -> list[dict[str, Any]]:
    """Return generation/stop-token config sources in runtime order.

    Gemma4 chat models encode end-of-turn as EOS candidates in HF
    ``generation_config.json`` (for 31B this is ``[1, 106, 50]``).  The
    exported manifest keeps the original HF files under ``hf_config/`` instead
    of flattening every generation field into the top-level metadata, so the
    speculative loop must read those source files rather than relying only on
    ``tokenizer.eos_token_id``.
    """

    configs: list[dict[str, Any]] = []
    seen: set[Path] = set()

    def add_file(path: Path) -> None:
        resolved = path.expanduser()
        if not resolved.is_absolute():
            resolved = resolved.resolve()
        if resolved in seen:
            return
        seen.add(resolved)
        data = _read_json_config(resolved)
        if data:
            configs.append(data)

    meta_path_value = meta.get("_meta_path") or meta.get("_meta_path_")
    export_dir = Path(meta_path_value).expanduser().resolve().parent if meta_path_value else None
    hf_config = meta.get("hf_config")
    if export_dir is not None and hf_config:
        hf_config_dir = Path(str(hf_config))
        if not hf_config_dir.is_absolute():
            hf_config_dir = export_dir / hf_config_dir
        add_file(hf_config_dir / "generation_config.json")
        add_file(hf_config_dir / "config.json")

    model_config = meta.get("model_config") if isinstance(meta.get("model_config"), dict) else {}
    hf_model = model_config.get("hf_model") if isinstance(model_config, dict) else None
    if hf_model:
        hf_model_dir = Path(str(hf_model)).expanduser()
        add_file(hf_model_dir / "generation_config.json")
        add_file(hf_model_dir / "config.json")

    inline_generation = meta.get("generation_config")
    if isinstance(inline_generation, dict):
        configs.append(inline_generation)
    if isinstance(model_config, dict):
        configs.append(model_config)
    return configs


def _resolve_eos_token_ids(tokenizer, *configs: dict[str, Any]) -> set[int]:
    token_ids: list[int] = []
    for config in configs:
        value = config.get("eos_token_id") if isinstance(config, dict) else None
        if value is None:
            continue
        if isinstance(value, int):
            value = [value]
        for token_id in value:
            token_id = int(token_id)
            if token_id not in token_ids:
                token_ids.append(token_id)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None:
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        for token_id in eos_token_id:
            token_id = int(token_id)
            if token_id not in token_ids:
                token_ids.append(token_id)
    return set(token_ids)


def _parse_target_outputs(outputs) -> tuple[Any, Any, dict[str, Any]]:
    if not isinstance(outputs, (tuple, list)):
        raise RuntimeError("Target HMONNX must be exported with enable_mtp_outputs=True; got one output only.")
    if len(outputs) < len(EXPECTED_BASE_OUTPUTS):
        raise RuntimeError(f"Target HMONNX returned {len(outputs)} outputs, expected {len(EXPECTED_BASE_OUTPUTS)}.")
    return outputs[0], outputs[1], {}


def _scalar_debug_value(value: Any) -> int | float | str:
    try:
        import torch

        if torch.is_tensor(value):
            if value.numel() == 1:
                return int(value.detach().cpu().reshape(-1)[0].item())
            return f"tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    except Exception:
        pass
    try:
        return int(value)
    except Exception:
        return repr(value)


def _run_target_prefill(target_model, input_ids):
    import torch

    if hasattr(target_model, "run_prefill_ids"):
        return target_model.run_prefill_ids(input_ids)

    prefill_chunk_length = int(target_model.meta_info.model_config.prefill_chunk_length)
    past_seq_length = 0
    logits = hidden = shared = None
    for start in range(0, input_ids.shape[1], prefill_chunk_length):
        chunk = input_ids[:, start : start + prefill_chunk_length]
        target_model.set_prefill()
        target_model.set_input_sequence_length(prefill_chunk_length)
        processor = target_model.get_data_preprocessor()
        model_inputs = processor({"input_ids": chunk, "past_seq_length": past_seq_length})
        outputs = target_model.forward(*model_inputs)
        logits, hidden, _ = _parse_target_outputs(outputs)
        shared = _hmonnx_shared_from_cache(target_model)
        past_seq_length += int(chunk.shape[1])
    assert logits is not None and hidden is not None and shared is not None
    return logits, hidden, shared, past_seq_length


def _run_target_verify(target_model, token_ids: list[int], past_seq_length: int):
    import torch

    if hasattr(target_model, "run_verify_ids"):
        return target_model.run_verify_ids(token_ids, past_seq_length)

    if not token_ids:
        raise ValueError("target verify requires at least one token")
    tokens = torch.tensor([list(map(int, token_ids))], dtype=torch.long, device=target_model.device)
    target_model.set_decode()
    target_model.set_input_sequence_length(len(token_ids))
    processor = target_model.get_data_preprocessor()
    verify_accepted_count = int(getattr(target_model, "_mtp_next_verify_accepted_count", 0))
    model_inputs = processor(
        {
            "input_ids": tokens,
            "past_seq_length": int(past_seq_length),
            "accepted_count": verify_accepted_count,
        }
    )
    if getattr(target_model, "_mtp_trace_accepted_count", False):
        input_names = list(getattr(getattr(target_model, "llm", None), "input_names", []) or [])
        if not input_names and hasattr(target_model, "get_input_names"):
            try:
                input_names = list(target_model.get_input_names())
            except Exception:
                input_names = []
        accepted_tensor = model_inputs[-1] if model_inputs else verify_accepted_count
        print(
            "[MTP accepted_count trace] "
            f"verify_round={int(getattr(target_model, '_mtp_verify_round_index', 0))} "
            f"target_decode_input accepted_count={_scalar_debug_value(accepted_tensor)} "
            f"expected_previous_accepted={verify_accepted_count} "
            f"past_seq_length={int(past_seq_length)} "
            f"verify_tokens={list(map(int, token_ids))}"
        )
        if input_names:
            print(f"[MTP accepted_count trace] target graph inputs={input_names}")
    outputs = target_model.forward(*model_inputs)
    logits, hidden, _ = _parse_target_outputs(outputs)
    return logits, hidden, _hmonnx_shared_from_cache(target_model)


def _cache_to_tensor(cache):
    import torch

    if hasattr(cache, "data") and torch.is_tensor(cache.data):
        return cache.data
    if torch.is_tensor(cache):
        return cache
    raise TypeError(f"Unsupported cache object: {type(cache)!r}")


def _clone_cache_tensor(cache):
    import torch
    from xhquant.core import CacheTensor

    if hasattr(cache, "data") and torch.is_tensor(cache.data):
        cloned = type(cache)(cache.data.detach().clone())
        if hasattr(cache, "cache_valid_len"):
            cloned.cache_valid_len = int(cache.cache_valid_len)
        return cloned
    if torch.is_tensor(cache):
        return CacheTensor(cache.detach().clone())
    raise TypeError(f"Unsupported cache object: {type(cache)!r}")


def _snapshot_hmonnx_kv_cache(target_model):
    return (
        [_clone_cache_tensor(cache) for cache in target_model.past_key_caches],
        [_clone_cache_tensor(cache) for cache in target_model.past_value_caches],
    )


def _hmonnx_shared_cache_indices(target_model) -> dict[str, int]:
    source: dict[str, int] = {}
    layer_cache_types = list(getattr(target_model, "layer_cache_types", []) or [])
    if layer_cache_types:
        for cache_idx, layer_type in enumerate(layer_cache_types):
            if layer_type in {"sliding_attention", "full_attention"}:
                source[str(layer_type)] = cache_idx
    else:
        layer_kv_shapes = list(
            getattr(getattr(target_model, "_kvcache_mixin", None), "layer_kv_shapes", None)
            or getattr(getattr(target_model, "meta_info", None), "layer_kv_shapes", None)
            or []
        )
        full_cache_len = int(getattr(getattr(target_model, "kvcache_config", None), "max_sequence_length", 0) or 0)
        if full_cache_len <= 0 and layer_kv_shapes:
            full_cache_len = max(int(shape[2]) for shape in layer_kv_shapes if len(shape) > 2)
        for cache_idx, shape in enumerate(layer_kv_shapes):
            if len(shape) <= 2:
                continue
            cache_len = int(shape[2])
            layer_type = "sliding_attention" if full_cache_len > 0 and cache_len < full_cache_len else "full_attention"
            source[layer_type] = cache_idx
        if not source:
            layer_types = list(getattr(target_model, "layer_types", []) or [])
            cache_count = len(getattr(target_model, "past_key_caches", []) or [])
            # Safe compatibility fallback only when layer_types already describes
            # the cache list.  Do not use full model layer indices when shared-KV
            # layers were filtered out of past_key_caches.
            if layer_types and cache_count and len(layer_types) == cache_count:
                for cache_idx, layer_type in enumerate(layer_types):
                    if layer_type in {"sliding_attention", "full_attention"}:
                        source[str(layer_type)] = cache_idx
    if "sliding_attention" not in source or "full_attention" not in source:
        raise RuntimeError(
            "Cannot locate Gemma4 HMONNX shared KV cache indices from cache metadata: "
            f"layer_cache_types={layer_cache_types}, source={source}"
        )
    return source


def _hmonnx_shared_from_cache(target_model) -> dict[str, Any]:
    source = _hmonnx_shared_cache_indices(target_model)
    sliding_idx = source["sliding_attention"]
    full_idx = source["full_attention"]
    return {
        "shared_key_cache_sliding": _cache_to_tensor(target_model.past_key_caches[sliding_idx]),
        "shared_value_cache_sliding": _cache_to_tensor(target_model.past_value_caches[sliding_idx]),
        "shared_key_cache_full": _cache_to_tensor(target_model.past_key_caches[full_idx]),
        "shared_value_cache_full": _cache_to_tensor(target_model.past_value_caches[full_idx]),
    }


def _commit_one_hmonnx_cache(before_cache, verified_cache, past_seq_length: int, commit_length: int, verify_length: int):
    del before_cache
    committed_cache = _clone_cache_tensor(verified_cache)
    committed_tensor = _cache_to_tensor(committed_cache)

    if hasattr(committed_cache, "cache_valid_len"):
        verified_valid = min(
            max(int(getattr(verified_cache, "cache_valid_len", 0)), 0),
            int(committed_tensor.shape[2]),
        )
        rollback = max(0, int(verify_length) - int(commit_length))
        new_valid = max(0, verified_valid - rollback)
        if new_valid < int(committed_tensor.shape[2]):
            committed_tensor[:, :, new_valid:, :].zero_()
        committed_cache.cache_valid_len = new_valid
        return committed_cache

    start = max(int(past_seq_length), 0)
    accepted_end = min(int(committed_tensor.shape[2]), start + int(commit_length))
    verify_end = min(int(committed_tensor.shape[2]), start + int(verify_length))
    if accepted_end < verify_end:
        committed_tensor[:, :, accepted_end:verify_end, :].zero_()
    return committed_cache


def _commit_hmonnx_verified_cache(
    target_model,
    snapshot,
    *,
    past_seq_length: int,
    commit_length: int,
    verify_length: int,
) -> dict[str, Any]:
    before_key_caches, before_value_caches = snapshot
    committed_key_caches = [
        _commit_one_hmonnx_cache(before, verified, past_seq_length, commit_length, verify_length)
        for before, verified in zip(before_key_caches, target_model.past_key_caches)
    ]
    committed_value_caches = [
        _commit_one_hmonnx_cache(before, verified, past_seq_length, commit_length, verify_length)
        for before, verified in zip(before_value_caches, target_model.past_value_caches)
    ]
    target_model._kvcache_mixin.past_key_caches.clear()
    target_model._kvcache_mixin.past_key_caches.extend(committed_key_caches)
    target_model._kvcache_mixin.past_value_caches.clear()
    target_model._kvcache_mixin.past_value_caches.extend(committed_value_caches)
    return _hmonnx_shared_from_cache(target_model)


def _target_needs_manual_cache_commit(target_model) -> bool:
    return (
        not hasattr(target_model, "commit_cache")
        and hasattr(target_model, "past_key_caches")
        and hasattr(target_model, "past_value_caches")
        and hasattr(target_model, "_kvcache_mixin")
    )


def _build_draft_masks(
    target_model,
    assistant_session: AssistantDraftSession,
    cache_valid_length: int | None = None,
    *,
    full_valid_length: int | None = None,
    sliding_valid_length: int | None = None,
):
    import torch

    dtype = target_model.dtype
    device = target_model.device
    neg = torch.tensor(torch.finfo(torch.float16).min, dtype=torch.float16, device=device).to(dtype=dtype)
    model_config = target_model.meta_info.model_config
    sliding_window = int(getattr(target_model, "sliding_window", getattr(model_config, "sliding_window", 1024)) or 1024)
    sliding_width = int(assistant_session.input_infos["sliding_attention_mask"].shape[-1])
    full_width = int(assistant_session.input_infos["full_attention_mask"].shape[-1])
    full_mask = torch.full((1, 1, 1, full_width), neg, dtype=dtype, device=device)
    sliding_mask = torch.full((1, 1, 1, sliding_width), neg, dtype=dtype, device=device)
    if full_valid_length is None:
        full_valid_length = int(cache_valid_length or 0)
    if sliding_valid_length is None:
        sliding_valid_length = int(cache_valid_length or 0)

    # Full/shared caches only expose valid prefix positions.  Any padded tail
    # must remain masked or the assistant can attend to zeros as if they were KV.
    full_valid = min(full_width, max(1, int(full_valid_length)))
    full_mask[0, 0, 0, :full_valid] = 0

    # Sliding shared KV is already in LLMCache's compact local coordinates.
    # Expose the suffix ending at its HybridCacheTensor cache_valid_len.  This
    # value is not always the absolute full-cache length after sliding compacts
    # a saturated window during MTP verify.
    local_valid = min(sliding_width, max(1, int(sliding_valid_length)))
    sliding_end = local_valid
    sliding_start = max(0, sliding_end - sliding_window)
    if sliding_end <= sliding_start:
        sliding_mask[0, 0, 0, 0] = 0
    else:
        sliding_mask[0, 0, 0, sliding_start:sliding_end] = 0
    return full_mask, sliding_mask


def _hmonnx_shared_cache_valid_lengths(target_model, fallback_full_valid: int) -> tuple[int, int]:
    try:
        source = _hmonnx_shared_cache_indices(target_model)
        sliding_cache = target_model.past_key_caches[source["sliding_attention"]]
        sliding_valid = int(getattr(sliding_cache, "cache_valid_len", fallback_full_valid) or fallback_full_valid)
    except Exception:
        sliding_valid = int(fallback_full_valid)
    return int(fallback_full_valid), int(sliding_valid)


def _build_assistant_inputs(
    target_model,
    assistant_session: AssistantDraftSession,
    last_token_id: int,
    current_hidden,
    shared: dict[str, Any],
    position_index: int,
    cache_valid_length: int | None = None,
    *,
    full_valid_length: int | None = None,
    sliding_valid_length: int | None = None,
):
    import torch

    device = target_model.device
    token = torch.tensor([[int(last_token_id)]], dtype=torch.long, device=device)
    token_embed = target_model.get_input_embeddings().to(device)(token).to(dtype=current_hidden.dtype)
    inputs_embeds = torch.cat([token_embed, current_hidden], dim=-1)
    full_mask, sliding_mask = _build_draft_masks(
        target_model,
        assistant_session,
        cache_valid_length,
        full_valid_length=full_valid_length,
        sliding_valid_length=sliding_valid_length,
    )
    return {
        "inputs_embeds": inputs_embeds,
        # Gemma4 assistant export uses DynamicSlice on RoPE tables keyed by
        # this integer.  Match vLLM's MTP forward contract: each draft step
        # receives the position of the token currently fed to the draft model.
        "past_seq_length": torch.tensor([int(position_index)], dtype=torch.int32, device=device),
        "current_length": torch.tensor([1], dtype=torch.int32, device=device),
        "sliding_attention_mask": sliding_mask,
        "full_attention_mask": full_mask,
        **shared,
    }


def generate_with_mtp(
    target_model,
    assistant_session,
    tokenizer,
    meta: dict[str, Any],
    prompt: str,
    max_new_tokens: int,
    num_draft_tokens: int,
    *,
    trace_accepted_count: bool = False,
    trace_kv_input_info: dict[str, Any] | None = None,
):
    import torch

    input_ids = _load_prompt_inputs(tokenizer, prompt).to(target_model.device)
    output_ids = input_ids[0].tolist()
    prompt_length = int(input_ids.shape[1])
    eos_token_ids = _resolve_eos_token_ids(tokenizer, *_generation_config_candidates(meta))
    spec = _meta_spec_decode(meta)
    verify_length = int(spec.get("verify_length") or meta.get("spec_decode_verify_length") or (num_draft_tokens + 1))

    logits, last_hidden, shared, past_seq_length = _run_target_prefill(target_model, input_ids)
    current_token_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
    output_ids.append(current_token_id)
    generated = 1
    stats = {
        "preset_prompt_tokens": prompt_length,
        "verify_rounds": 0,
        "draft_proposed": 0,
        "draft_accepted": 0,
        "committed_tokens": 1,
        "accepted_per_round": [],
        "accepted_count_inputs": [],
    }
    if current_token_id in eos_token_ids or generated >= max_new_tokens:
        text = tokenizer.decode(output_ids[prompt_length:], skip_special_tokens=True).strip()
        stats["generated_tokens"] = int(len(output_ids) - prompt_length)
        stats["acceptance_rate"] = 0.0
        return text, output_ids, stats

    while generated < max_new_tokens:
        draft_tokens: list[int] = []
        assistant_hidden = last_hidden
        assistant_last_token_id = int(current_token_id)
        # The draft RoPE position is the position of the token currently fed to
        # the assistant.  The target cache contains the prefix before that token,
        # so after prefill length N the first sampled token is at position N.
        base_position = max(int(past_seq_length), 0)
        full_valid_length, sliding_valid_length = _hmonnx_shared_cache_valid_lengths(target_model, past_seq_length)
        for draft_step in range(num_draft_tokens):
            round_inputs = _build_assistant_inputs(
                target_model=target_model,
                assistant_session=assistant_session,
                last_token_id=assistant_last_token_id,
                current_hidden=assistant_hidden,
                shared=shared,
                position_index=base_position + draft_step,
                full_valid_length=full_valid_length,
                sliding_valid_length=sliding_valid_length,
            )
            draft_logits, assistant_hidden = assistant_session(**round_inputs)
            assistant_token = int(torch.argmax(draft_logits[:, -1, :], dim=-1).item())
            draft_tokens.append(assistant_token)
            assistant_last_token_id = assistant_token
            if assistant_token in eos_token_ids:
                break

        if not draft_tokens:
            break

        stats["verify_rounds"] += 1
        stats["draft_proposed"] += len(draft_tokens)
        verify_ids = [int(current_token_id)] + draft_tokens
        if len(verify_ids) > verify_length:
            raise RuntimeError(f"verify_ids length {len(verify_ids)} exceeds exported verify_length={verify_length}")
        if len(verify_ids) < verify_length:
            verify_ids = verify_ids + [verify_ids[-1]] * (verify_length - len(verify_ids))
        round_past_seq_length = int(past_seq_length)
        verify_input_accepted_count = int(getattr(target_model, "_mtp_next_verify_accepted_count", 0))
        setattr(target_model, "_mtp_trace_accepted_count", bool(trace_accepted_count))
        setattr(target_model, "_mtp_verify_round_index", int(stats["verify_rounds"]))
        cache_snapshot = (
            _snapshot_hmonnx_kv_cache(target_model)
            if _target_needs_manual_cache_commit(target_model)
            else None
        )
        verify_logits, verify_hidden, verify_shared = _run_target_verify(target_model, verify_ids, round_past_seq_length)
        predicted_tokens = [
            int(torch.argmax(verify_logits[:, idx : idx + 1, :], dim=-1)[0, 0].item())
            for idx in range(len(verify_ids))
        ]

        accepted_count = 0
        for predicted_token, draft_token in zip(predicted_tokens, draft_tokens):
            if int(predicted_token) != int(draft_token):
                break
            accepted_count += 1
        setattr(target_model, "_mtp_next_verify_accepted_count", int(accepted_count))
        stats["accepted_count_inputs"].append(verify_input_accepted_count)
        stats["draft_accepted"] += accepted_count
        stats["accepted_per_round"].append(accepted_count)
        if trace_accepted_count:
            kv_desc = trace_kv_input_info or {}
            print(
                "[MTP accepted_count trace] "
                f"verify_round={int(stats['verify_rounds'])} "
                f"first_sliding_kvcache={kv_desc.get('node_name')} "
                f"accepted_count_input_index={kv_desc.get('accepted_count_input_index')} "
                f"input_accepted_count={verify_input_accepted_count} "
                f"current_round_accepted_tokens={accepted_count} "
                f"next_verify_expected_accepted_count={accepted_count} "
                f"match_previous={verify_input_accepted_count == (stats['accepted_per_round'][-2] if len(stats['accepted_per_round']) > 1 else 0)}"
            )

        for token in draft_tokens[:accepted_count]:
            if generated >= max_new_tokens:
                break
            output_ids.append(int(token))
            generated += 1
            stats["committed_tokens"] += 1
            if int(token) in eos_token_ids:
                break
        if output_ids[-1] in eos_token_ids or generated >= max_new_tokens:
            break

        if accepted_count < len(draft_tokens):
            next_token_id = int(predicted_tokens[accepted_count])
        else:
            next_token_id = int(predicted_tokens[len(draft_tokens)])

        output_ids.append(next_token_id)
        generated += 1
        stats["committed_tokens"] += 1
        accepted_steps = accepted_count + 1
        past_seq_length += accepted_steps
        if hasattr(target_model, "commit_cache"):
            shared = target_model.commit_cache(past_seq_length)
        elif cache_snapshot is not None and accepted_steps < verify_length:
            # HMONNX LLMCache mutates KV in-place for the entire exported
            # verify block.  Speculative decoding may accept only a prefix, so
            # rebuild the compact cache from the pre-verify snapshot plus the
            # accepted prefix of the verified cache.  This is equivalent to a
            # fixed-length replay with current_length=accepted_steps, but does
            # not spend another target decode pass.
            shared = _commit_hmonnx_verified_cache(
                target_model,
                cache_snapshot,
                past_seq_length=round_past_seq_length,
                commit_length=accepted_steps,
                verify_length=verify_length,
            )
        else:
            shared = verify_shared
        if verify_hidden is not None and getattr(verify_hidden, "ndim", 0) >= 3:
            last_hidden = verify_hidden[:, accepted_count : accepted_count + 1, :]
        current_token_id = next_token_id
        if current_token_id in eos_token_ids:
            break

    text = tokenizer.decode(output_ids[prompt_length:], skip_special_tokens=True).strip()
    stats["generated_tokens"] = int(len(output_ids) - prompt_length)
    stats["acceptance_rate"] = (stats["draft_accepted"] / stats["draft_proposed"]) if stats["draft_proposed"] else 0.0
    return text, output_ids, stats


def run_verify(args: argparse.Namespace) -> None:
    names = sorted(PRESETS) if args.all else [args.preset]
    results = [
        verify_preset(
            PRESETS[name],
            Path(args.base_root),
            Path(args.draft_root),
            meta=args.meta if len(names) == 1 else None,
            draft_onnx=args.draft_onnx if len(names) == 1 else None,
        )
        for name in names
    ]
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
    for result in results:
        print(
            f"{result['preset']}: OK | {result['assistant_kind']} | "
            f"base_outputs={result['base_outputs']} | draft_outputs={result['draft_outputs']} | "
            f"w4={result['w4_head_node']['name']} | quant={result['quant_counter']}"
        )


def run_generate(args: argparse.Namespace) -> None:
    import torch
    from xhmodel_merak.xh_llm import AutoLLMHONNXModel, LLMInferenceContextManager
    from xhquant.api import get_xhquant_logger, xhquant_init
    from xhquant.utils import MemoryTracker, TimeProfiler

    preset = PRESETS[args.preset]
    meta_path = _resolve_meta_path(preset, Path(args.base_root), args.meta)
    meta_for_draft = json.loads(meta_path.read_text(encoding="utf-8"))
    draft_onnx = None
    if args.draft_backend == "hmonnx":
        draft_onnx = _resolve_draft_onnx(
            preset, Path(args.draft_root), args.draft_onnx, meta_path=meta_path, meta_dict=meta_for_draft
        )
        verify_preset(preset, Path(args.base_root), Path(args.draft_root), meta=str(meta_path), draft_onnx=str(draft_onnx))
    elif args.draft_onnx:
        # Keep the target/draft ONNX contract check available even when the
        # current run uses fp draft.  This catches stale manifests before a
        # mixed-backend diagnosis compares against the wrong exported graph.
        verify_preset(preset, Path(args.base_root), Path(args.draft_root), meta=str(meta_path), draft_onnx=args.draft_onnx)

    log_file = meta_path.parent / "gemma4_series_mtp_hmonnx_generate.log"
    xhquant_init(str(log_file), args.debug)
    logger = get_xhquant_logger()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["_meta_path"] = str(meta_path)

    device = torch.device(args.device)
    default_target_dir, default_assistant_dir = DEFAULT_HF_DIRS[args.preset]
    target_model_dir = args.target_model_dir or str(meta.get("model_config", {}).get("hf_model") or default_target_dir)
    assistant_model_dir = args.assistant_model_dir or default_assistant_dir
    if args.target_backend == "hmonnx":
        target_model = AutoLLMHONNXModel.from_pretrained(str(meta_path))
        target_model.to(device)
        tokenizer = target_model.get_tokenizer(trust_remote_code=True)
    else:
        target_model = TorchTargetSession(
            target_model_dir=target_model_dir,
            meta=meta,
            device=device,
            dtype=_torch_dtype(args.target_dtype),
        )
        tokenizer = target_model.get_tokenizer(trust_remote_code=True)
    if args.draft_backend == "hmonnx":
        if draft_onnx is None:
            draft_onnx = _resolve_draft_onnx(
                preset, Path(args.draft_root), args.draft_onnx, meta_path=meta_path, meta_dict=meta_for_draft
            )
        assistant_session = AssistantDraftSession(draft_onnx, device)
        draft_desc = str(draft_onnx)
    else:
        assistant_session = TorchAssistantDraftSession(
            assistant_model_dir=assistant_model_dir,
            target_model_dir=target_model_dir,
            meta=meta,
            device=device,
            dtype=_torch_dtype(args.draft_dtype),
        )
        draft_desc = f"torch:{assistant_model_dir}"

    logger.info(
        "Gemma4 Series MTP generate preset=%s meta=%s target_backend=%s draft_backend=%s draft=%s",
        args.preset,
        meta_path,
        args.target_backend,
        args.draft_backend,
        draft_desc,
    )
    trace_kv_input_info = _first_sliding_kv_trace_info(meta_path, meta) if args.trace_accepted_count else None
    if args.trace_accepted_count:
        if trace_kv_input_info is None:
            raise RuntimeError(f"{args.preset}: cannot find a sliding KVcache node in target decode ONNX")
        print("[MTP accepted_count trace] first sliding KVcache node")
        print(json.dumps(trace_kv_input_info, ensure_ascii=False, indent=2))
    with TimeProfiler("gemma4_series_mtp_hmonnx_generate", logger), MemoryTracker(device=str(device), name="generate", logger=logger):
        with ExitStack() as stack:
            if args.target_backend == "hmonnx":
                stack.enter_context(LLMInferenceContextManager(target_model, [device]))
            with torch.inference_mode():
                text, output_ids, stats = generate_with_mtp(
                    target_model=target_model,
                    assistant_session=assistant_session,
                    tokenizer=tokenizer,
                    meta=meta,
                    prompt=args.prompt,
                    max_new_tokens=args.max_new_tokens,
                    num_draft_tokens=args.num_draft_tokens,
                    trace_accepted_count=args.trace_accepted_count,
                    trace_kv_input_info=trace_kv_input_info,
                )

    print(text)
    print("\n[Gemma4 MTP HMONNX stats]")
    payload = {
        "preset": args.preset,
        "meta": str(meta_path),
        "target_backend": args.target_backend,
        "draft_backend": args.draft_backend,
        "draft_onnx": str(draft_onnx) if draft_onnx is not None else None,
        "draft": draft_desc,
        **stats,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    logger.info("[MTP stats] %s", json.dumps(payload, ensure_ascii=False))


def _add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-root",
        default="work_dirs/gemma4_series_mtp_base_clean",
        help="Root containing <slug>/existing-hf/export/<export-dir> target HMONNX exports.",
    )
    parser.add_argument(
        "--draft-root",
        default="work_dirs/gemma4_series_mtp_clean",
        help="Legacy root containing mtp_draft_decode/draft_onnx; ignored when manifest records draft_decode_onnx.",
    )
    parser.add_argument("--meta", help="Explicit target golden_meta_info.json or export directory for one preset.")
    parser.add_argument("--draft-onnx", help="Explicit assistant decode ONNX path for one preset.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    # Backward compatible contract smoke: ``mtp_hmonnx_inference.py --all``.
    if not argv or argv[0] not in {"verify", "generate"}:
        argv = ["verify", *argv]

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_verify = sub.add_parser("verify", help="Verify exported Gemma4 Series MTP target/draft graph contract.")
    p_verify.add_argument("--preset", choices=sorted(PRESETS), default="e2b")
    p_verify.add_argument("--all", action="store_true", help="Verify all Gemma4 Series MTP presets.")
    _add_common_paths(p_verify)
    p_verify.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    p_verify.set_defaults(func=run_verify)

    p_gen = sub.add_parser("generate", help="Run greedy HMONNX target + ONNX assistant MTP decoding.")
    p_gen.add_argument("--preset", choices=sorted(PRESETS), default="e2b")
    _add_common_paths(p_gen)
    p_gen.add_argument("--prompt", default="用一句话解释 Gemma4 MTP。")
    p_gen.add_argument("--max-new-tokens", type=int, default=32)
    p_gen.add_argument("--num-draft-tokens", type=int, default=4)
    p_gen.add_argument("--device", default="cuda:0")
    p_gen.add_argument("--target-backend", choices=["hmonnx", "torch"], default="hmonnx")
    p_gen.add_argument("--target-dtype", default="float16", help="dtype for --target-backend torch")
    p_gen.add_argument("--draft-backend", choices=["hmonnx", "torch"], default="hmonnx")
    p_gen.add_argument("--draft-dtype", default="float16", help="dtype for --draft-backend torch")
    p_gen.add_argument("--target-model-dir", help="HF target dir for --draft-backend torch embedding/config.")
    p_gen.add_argument("--assistant-model-dir", help="HF assistant dir for --draft-backend torch.")
    p_gen.add_argument(
        "--trace-accepted-count",
        action="store_true",
        help="Print first sliding KVcache accepted_count inputs and per-verify accepted-token alignment.",
    )
    p_gen.add_argument("--debug", action="store_true")
    p_gen.set_defaults(func=run_generate)

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
