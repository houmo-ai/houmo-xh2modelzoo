# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from safetensors import safe_open
from torch import Tensor

from xhquant.nn import RMSNorm


def _load_shared_dflash_decoder_layer():
    module_path = Path(__file__).parents[1] / "qwen3_5" / "_dflash_model_impl.py"
    spec = importlib.util.spec_from_file_location(
        "xhmodel_merak.xh_llm.models._shared_qwen3_5_dflash_impl",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load shared Qwen3.5 DFlash operators from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DFlashDecoderLayer


DFlashDecoderLayer = _load_shared_dflash_decoder_layer()

DEPLOYMENT_DTYPE = torch.float16
REFERENCE_DTYPE = torch.float32
EMBEDDING_KEYS = (
    "model.language_model.embed_tokens.weight",
    "model.embed_tokens.weight",
)
LM_HEAD_KEY = "lm_head.weight"


@dataclass(frozen=True)
class HunyuanOCRDFlashConfig:
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    num_target_layers: int
    target_layer_ids: tuple[int, ...]
    vocab_size: int
    block_size: int
    mask_token_id: int
    draft_eos_token_id: int
    rope_theta: float
    max_position_embeddings: int
    rms_norm_eps: float


@dataclass(frozen=True)
class HunyuanOCRDFlashOutputHead:
    weight: Tensor
    embedding_weight: Tensor
    tie_word_embeddings: bool
    source_key: str
    embedding_source_key: str
    source_shape: tuple[int, ...]
    source_dtype: str
    deployment_dtype: str
    sha256: str
    embedding_sha256: str


@dataclass(frozen=True)
class HunyuanOCRDFlashCheckpoint:
    config: HunyuanOCRDFlashConfig
    draft_state_dict: Mapping[str, Tensor]
    output_head: HunyuanOCRDFlashOutputHead
    config_sha256: str


@dataclass(frozen=True)
class _PendingDraftTransaction:
    transaction_id: int
    committed_length: int
    accepted_prefix_length: int | None = None


def _require_int(data: Mapping[str, Any], field: str, *, positive: bool = True) -> int:
    value = data.get(field)
    if type(value) is not int or (positive and value <= 0):
        qualifier = "a positive integer" if positive else "an integer"
        raise ValueError(f"HunyuanOCR DFlash config {field} must be {qualifier}, got {value!r}")
    return value


def _load_config(path: Path) -> tuple[HunyuanOCRDFlashConfig, str]:
    if not path.is_file():
        raise FileNotFoundError(f"HunyuanOCR DFlash config does not exist: {path}")
    config_bytes = path.read_bytes()
    raw = json.loads(config_bytes)
    if not isinstance(raw, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    dflash = raw.get("dflash_config")
    if not isinstance(dflash, Mapping):
        raise ValueError("HunyuanOCR DFlash config must declare dflash_config")
    target_layer_ids = dflash.get("target_layer_ids")
    if not isinstance(target_layer_ids, list) or not target_layer_ids:
        raise ValueError("dflash_config.target_layer_ids must be a non-empty integer list")
    if any(type(layer_id) is not int for layer_id in target_layer_ids):
        raise ValueError("dflash_config.target_layer_ids must contain only integers")
    if len(set(target_layer_ids)) != len(target_layer_ids) or any(
        left >= right for left, right in zip(target_layer_ids, target_layer_ids[1:], strict=False)
    ):
        raise ValueError("dflash_config.target_layer_ids must be unique and strictly increasing")
    num_target_layers = _require_int(raw, "num_target_layers")
    if target_layer_ids[0] < 0 or target_layer_ids[-1] >= num_target_layers:
        raise ValueError(
            "dflash_config.target_layer_ids contains an id outside num_target_layers: "
            f"ids={target_layer_ids}, num_target_layers={num_target_layers}"
        )
    block_size = _require_int(raw, "block_size")
    if block_size != 16:
        raise ValueError(f"HunyuanOCR DFlash block_size must be 16, got {block_size}")
    mask_token_id = dflash.get("mask_token_id")
    if type(mask_token_id) is not int:
        raise ValueError("dflash_config.mask_token_id must be an integer")
    rope_theta = raw.get("rope_theta")
    if not isinstance(rope_theta, (int, float)) or float(rope_theta) <= 0:
        raise ValueError(f"HunyuanOCR DFlash rope_theta must be positive, got {rope_theta!r}")
    if str(raw.get("dtype")) not in ("float32", "torch.float32"):
        raise ValueError("HunyuanOCR DFlash reference checkpoint dtype must be float32")
    config = HunyuanOCRDFlashConfig(
        hidden_size=_require_int(raw, "hidden_size"),
        intermediate_size=_require_int(raw, "intermediate_size"),
        num_attention_heads=_require_int(raw, "num_attention_heads"),
        num_key_value_heads=_require_int(raw, "num_key_value_heads"),
        head_dim=_require_int(raw, "head_dim"),
        num_hidden_layers=_require_int(raw, "num_hidden_layers"),
        num_target_layers=num_target_layers,
        target_layer_ids=tuple(target_layer_ids),
        vocab_size=_require_int(raw, "vocab_size"),
        block_size=block_size,
        mask_token_id=mask_token_id,
        draft_eos_token_id=_require_int(raw, "eos_token_id", positive=False),
        rope_theta=float(rope_theta),
        max_position_embeddings=_require_int(raw, "max_position_embeddings"),
        rms_norm_eps=float(raw.get("rms_norm_eps", 0.0)),
    )
    if config.num_hidden_layers != 5:
        raise ValueError(f"HunyuanOCR DFlash num_hidden_layers must be 5, got {config.num_hidden_layers}")
    if config.mask_token_id < 0 or config.mask_token_id >= config.vocab_size:
        raise ValueError("dflash_config.mask_token_id must be inside the draft vocabulary")
    if config.rms_norm_eps <= 0:
        raise ValueError("HunyuanOCR DFlash rms_norm_eps must be positive")
    return config, hashlib.sha256(config_bytes).hexdigest()


def _expected_draft_keys(config: HunyuanOCRDFlashConfig) -> set[str]:
    keys = {"fc.weight", "hidden_norm.weight", "norm.weight"}
    layer_suffixes = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    }
    for layer_id in range(config.num_hidden_layers):
        keys.update(f"layers.{layer_id}.{suffix}" for suffix in layer_suffixes)
    return keys


def _load_safetensors(path: Path) -> dict[str, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"HunyuanOCR DFlash checkpoint does not exist: {path}")
    with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
        return {key: checkpoint.get_tensor(key) for key in checkpoint.keys()}


def _checkpoint_weight_map(root: Path) -> dict[str, Path]:
    index_path = root / "model.safetensors.index.json"
    if index_path.is_file():
        raw = json.loads(index_path.read_text(encoding="utf-8"))
        weight_map = raw.get("weight_map")
        if not isinstance(weight_map, Mapping):
            raise ValueError(f"Invalid safetensors weight_map in {index_path}")
        return {str(key): root / str(relative_path) for key, relative_path in weight_map.items()}
    result: dict[str, Path] = {}
    for checkpoint_path in sorted(root.glob("*.safetensors")):
        with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
            for key in checkpoint.keys():
                if key in result:
                    raise ValueError(f"Duplicate target checkpoint tensor {key!r}")
                result[key] = checkpoint_path
    return result


def _load_weight(weight_map: Mapping[str, Path], key: str) -> Tensor | None:
    checkpoint_path = weight_map.get(key)
    if checkpoint_path is None:
        return None
    with safe_open(str(checkpoint_path), framework="pt", device="cpu") as checkpoint:
        return checkpoint.get_tensor(key)


def _tensor_sha256(tensor: Tensor) -> str:
    contiguous = tensor.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.view(torch.uint8).numpy().tobytes()).hexdigest()


def _load_output_head(target_root: Path, config: HunyuanOCRDFlashConfig) -> HunyuanOCRDFlashOutputHead:
    target_config_path = target_root / "config.json"
    if not target_config_path.is_file():
        raise FileNotFoundError(f"HunyuanOCR target config does not exist: {target_config_path}")
    target_config = json.loads(target_config_path.read_text(encoding="utf-8"))
    text_config = target_config.get("text_config", target_config)
    if not isinstance(text_config, Mapping) or type(text_config.get("tie_word_embeddings")) is not bool:
        raise ValueError("Target text_config.tie_word_embeddings must be explicitly declared")
    tie_word_embeddings = text_config["tie_word_embeddings"]
    if int(text_config.get("hidden_size", 0)) != config.hidden_size:
        raise ValueError("Target hidden_size does not match the HunyuanOCR DFlash checkpoint")
    if int(text_config.get("vocab_size", 0)) != config.vocab_size:
        raise ValueError("Target vocab_size does not match the HunyuanOCR DFlash checkpoint")
    weight_map = _checkpoint_weight_map(target_root)
    embedding_keys = [key for key in EMBEDDING_KEYS if key in weight_map]
    if len(embedding_keys) != 1:
        raise ValueError(
            "Target checkpoint must contain exactly one canonical embed_tokens.weight key, "
            f"found={embedding_keys}"
        )
    embedding_key = embedding_keys[0]
    embedding = _load_weight(weight_map, embedding_key)
    assert embedding is not None
    expected_shape = (config.vocab_size, config.hidden_size)
    if tuple(embedding.shape) != expected_shape:
        raise ValueError(
            f"Target {embedding_key} shape mismatch: expected={expected_shape}, got={tuple(embedding.shape)}"
        )
    explicit_head = _load_weight(weight_map, LM_HEAD_KEY)
    if tie_word_embeddings:
        if explicit_head is not None and (
            explicit_head.shape != embedding.shape or not torch.equal(explicit_head, embedding)
        ):
            raise ValueError(
                "Target lm_head.weight must equal embed_tokens.weight before dtype conversion when "
                "tie_word_embeddings=true"
            )
        source_key = embedding_key
        source = embedding
    else:
        if explicit_head is None:
            raise ValueError("Target lm_head.weight is required when tie_word_embeddings=false")
        if tuple(explicit_head.shape) != expected_shape:
            raise ValueError(
                f"Target lm_head.weight shape mismatch: expected={expected_shape}, got={tuple(explicit_head.shape)}"
            )
        source_key = LM_HEAD_KEY
        source = explicit_head
    source_dtype = str(source.dtype).removeprefix("torch.")
    return HunyuanOCRDFlashOutputHead(
        weight=source.to(dtype=DEPLOYMENT_DTYPE),
        embedding_weight=embedding.to(dtype=DEPLOYMENT_DTYPE),
        tie_word_embeddings=tie_word_embeddings,
        source_key=source_key,
        embedding_source_key=embedding_key,
        source_shape=tuple(source.shape),
        source_dtype=source_dtype,
        deployment_dtype=str(DEPLOYMENT_DTYPE).removeprefix("torch."),
        sha256=_tensor_sha256(source),
        embedding_sha256=_tensor_sha256(embedding),
    )


def load_hunyuan_ocr_dflash_checkpoint(
    dflash_root: str | Path,
    target_root: str | Path,
) -> HunyuanOCRDFlashCheckpoint:
    dflash_root = Path(dflash_root)
    target_root = Path(target_root)
    config, config_sha256 = _load_config(dflash_root / "config.json")
    draft_state = _load_safetensors(dflash_root / "model.safetensors")
    expected_keys = _expected_draft_keys(config)
    actual_keys = set(draft_state)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    if missing or unexpected:
        raise ValueError(
            "HunyuanOCR DFlash checkpoint tensor mismatch: "
            f"missing={missing}, unexpected={unexpected}, expected_count={len(expected_keys)}, "
            f"actual_count={len(actual_keys)}"
        )
    if any(tensor.dtype != REFERENCE_DTYPE for tensor in draft_state.values()):
        wrong = sorted(key for key, tensor in draft_state.items() if tensor.dtype != REFERENCE_DTYPE)
        raise ValueError(f"HunyuanOCR DFlash reference tensors must all be float32, wrong_dtype={wrong}")
    expected_fc_shape = (config.hidden_size, len(config.target_layer_ids) * config.hidden_size)
    if tuple(draft_state["fc.weight"].shape) != expected_fc_shape:
        raise ValueError(
            f"HunyuanOCR DFlash fc.weight shape mismatch: expected={expected_fc_shape}, "
            f"got={tuple(draft_state['fc.weight'].shape)}"
        )
    output_head = _load_output_head(target_root, config)
    return HunyuanOCRDFlashCheckpoint(
        config=config,
        draft_state_dict=draft_state,
        output_head=output_head,
        config_sha256=config_sha256,
    )


def build_dflash_noise_embedding(
    *,
    current_token_id: int,
    embedding_weight: Tensor,
    mask_token_id: int,
    block_size: int = 16,
) -> Tensor:
    if embedding_weight.ndim != 2:
        raise ValueError(f"embedding_weight must be rank 2, got shape={tuple(embedding_weight.shape)}")
    if embedding_weight.dtype != DEPLOYMENT_DTYPE:
        raise ValueError(f"embedding_weight must use float16 deployment dtype, got {embedding_weight.dtype}")
    if block_size != 16:
        raise ValueError(f"HunyuanOCR DFlash block_size must be 16, got {block_size}")
    vocab_size = int(embedding_weight.shape[0])
    for field, token_id in (("current_token_id", current_token_id), ("mask_token_id", mask_token_id)):
        if type(token_id) is not int or token_id < 0 or token_id >= vocab_size:
            raise ValueError(f"{field} must be inside [0, {vocab_size}), got {token_id!r}")
    token_ids = torch.tensor(
        [[current_token_id, *([mask_token_id] * (block_size - 1))]],
        dtype=torch.long,
        device=embedding_weight.device,
    )
    return embedding_weight[token_ids]


class HunyuanOCRDraftCacheController:
    """Own Draft logical cache length and provisional-write transaction state."""

    def __init__(self, *, capacity: int, block_size: int = 16) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError(f"capacity must be a positive integer, got {capacity!r}")
        if block_size != 16:
            raise ValueError(f"HunyuanOCR DFlash block_size must be 16, got {block_size}")
        self.capacity = capacity
        self.block_size = block_size
        self._committed_length = 0
        self._pending: _PendingDraftTransaction | None = None
        self._poisoned = False
        self._next_transaction_id = 1

    @property
    def committed_length(self) -> int:
        return self._committed_length

    @property
    def pending_transaction_id(self) -> int | None:
        return self._pending.transaction_id if self._pending is not None else None

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def reset(self) -> None:
        self._committed_length = 0
        self._pending = None
        self._poisoned = False

    def restore_committed_length(self, committed_length: int) -> None:
        self._require_idle()
        self._validate_append(committed_length=0, current_input_length=committed_length)
        self._committed_length = committed_length

    def begin_context(self, *, current_input_length: int) -> int:
        self._require_idle()
        self._validate_append(self._committed_length, current_input_length)
        return self._committed_length

    def commit_context(self, *, current_input_length: int) -> int:
        self.begin_context(current_input_length=current_input_length)
        self._committed_length += current_input_length
        return self._committed_length

    def begin_decode(self) -> int:
        self._require_idle()
        if self._committed_length + self.block_size > self.capacity:
            raise ValueError(
                "Draft decode requires committed_length + block_size <= capacity: "
                f"{self._committed_length} + {self.block_size} > {self.capacity}"
            )
        transaction_id = self._next_transaction_id
        self._next_transaction_id += 1
        self._pending = _PendingDraftTransaction(transaction_id, self._committed_length)
        return transaction_id

    def begin_context_decode(self, *, accepted_draft_count: int, transaction_id: int) -> int:
        pending = self._require_pending(transaction_id)
        if type(accepted_draft_count) is not int or not 0 <= accepted_draft_count < self.block_size:
            raise ValueError(
                f"accepted_draft_count must satisfy 0 <= count < {self.block_size}, got {accepted_draft_count!r}"
            )
        accepted_prefix_length = accepted_draft_count + 1
        self._validate_append(pending.committed_length, accepted_prefix_length)
        self._pending = _PendingDraftTransaction(
            pending.transaction_id,
            pending.committed_length,
            accepted_prefix_length,
        )
        return accepted_prefix_length

    def commit_context_decode(self, *, transaction_id: int) -> int:
        pending = self._require_pending(transaction_id)
        if pending.accepted_prefix_length is None:
            raise RuntimeError("context_decode must execute before committing the Draft transaction")
        self._committed_length = pending.committed_length + pending.accepted_prefix_length
        self._pending = None
        return self._committed_length

    def discard_decode(self, *, transaction_id: int) -> None:
        self._require_pending(transaction_id)
        self._pending = None

    def mark_execution_failed(self, *, transaction_id: int) -> None:
        self._require_pending(transaction_id)
        self._poisoned = True

    def _validate_append(self, committed_length: int, current_input_length: int) -> None:
        if type(current_input_length) is not int or current_input_length < 0:
            raise ValueError(f"current_input_length must be a non-negative integer, got {current_input_length!r}")
        if committed_length + current_input_length > self.capacity:
            raise ValueError(
                "Draft cache append exceeds capacity: "
                f"committed_length={committed_length}, current_input_length={current_input_length}, "
                f"capacity={self.capacity}"
            )

    def _require_available(self) -> None:
        if self._poisoned:
            raise RuntimeError("Draft cache request is poisoned; reset and rebuild context before continuing")

    def _require_idle(self) -> None:
        self._require_available()
        if self._pending is not None:
            raise RuntimeError("Draft cache transaction pending; resolve it before another graph execution")

    def _require_pending(self, transaction_id: int) -> _PendingDraftTransaction:
        self._require_available()
        if self._pending is None or self._pending.transaction_id != transaction_id:
            expected = self._pending.transaction_id if self._pending is not None else None
            raise RuntimeError(f"stale transaction: got transaction_id={transaction_id}, expected={expected}")
        return self._pending


def build_draft_position_ids(*, committed_length: int, input_length: int) -> Tensor:
    if type(committed_length) is not int or committed_length < 0:
        raise ValueError(f"committed_length must be a non-negative integer, got {committed_length!r}")
    if type(input_length) is not int or input_length <= 0:
        raise ValueError(f"input_length must be a positive integer, got {input_length!r}")
    return torch.arange(committed_length, committed_length + input_length, dtype=torch.long).unsqueeze(0)


class HunyuanOCRDFlashModel(nn.Module):
    """Hunyuan-owned DFlash core with explicit three-graph boundaries."""

    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        num_hidden_layers: int,
        target_layer_ids: tuple[int, ...],
        vocab_size: int,
        rms_norm_eps: float,
        rope_theta: float,
        max_position_embeddings: int,
        max_sequence_length: int,
        input_sequence_length: int = 256,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.target_layer_ids = target_layer_ids
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.fc = nn.Linear(len(target_layer_ids) * hidden_size, hidden_size, bias=False)
        self.hidden_norm = RMSNorm(hidden_size, rms_norm_eps)
        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    num_key_value_heads=num_key_value_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=rms_norm_eps,
                    input_sequence_length=input_sequence_length,
                    max_pe_length=max_position_embeddings,
                    rope_theta=rope_theta,
                    use_cache=True,
                )
                for _ in range(num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size, rms_norm_eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    @classmethod
    def from_config(
        cls,
        raw: Mapping[str, Any],
        *,
        max_sequence_length: int,
        input_sequence_length: int = 256,
    ) -> HunyuanOCRDFlashModel:
        dflash = raw.get("dflash_config")
        if not isinstance(dflash, Mapping):
            raise ValueError("HunyuanOCR DFlash config must declare dflash_config")
        target_layer_ids = dflash.get("target_layer_ids")
        if not isinstance(target_layer_ids, list):
            raise ValueError("dflash_config.target_layer_ids must be a list")
        rope_theta = raw.get("rope_theta")
        if not isinstance(rope_theta, (int, float)) or float(rope_theta) <= 0:
            raise ValueError("HunyuanOCR DFlash rope_theta must be positive")
        if int(raw.get("block_size", 0)) != 16:
            raise ValueError("HunyuanOCR DFlash block_size must be 16")
        return cls(
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["head_dim"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            target_layer_ids=tuple(int(layer_id) for layer_id in target_layer_ids),
            vocab_size=int(raw["vocab_size"]),
            rms_norm_eps=float(raw["rms_norm_eps"]),
            rope_theta=float(rope_theta),
            max_position_embeddings=int(raw["max_position_embeddings"]),
            max_sequence_length=max_sequence_length,
            input_sequence_length=input_sequence_length,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: HunyuanOCRDFlashCheckpoint,
        *,
        max_sequence_length: int,
        input_sequence_length: int,
        deployment_dtype: torch.dtype = DEPLOYMENT_DTYPE,
    ) -> HunyuanOCRDFlashModel:
        config = checkpoint.config
        model = cls(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            num_hidden_layers=config.num_hidden_layers,
            target_layer_ids=config.target_layer_ids,
            vocab_size=config.vocab_size,
            rms_norm_eps=config.rms_norm_eps,
            rope_theta=config.rope_theta,
            max_position_embeddings=config.max_position_embeddings,
            max_sequence_length=max_sequence_length,
            input_sequence_length=input_sequence_length,
        )
        model.load_draft_state_dict(checkpoint.draft_state_dict)
        model.load_output_head(checkpoint.output_head.weight)
        return model.to(dtype=deployment_dtype)

    def load_draft_state_dict(self, state_dict: Mapping[str, Tensor]) -> None:
        missing, unexpected = self.load_state_dict(dict(state_dict), strict=False)
        missing = [name for name in missing if name != "lm_head.weight"]
        if missing or unexpected:
            raise ValueError(f"HunyuanOCR DFlash model load mismatch: missing={missing}, unexpected={unexpected}")

    def load_output_head(self, weight: Tensor) -> None:
        if tuple(weight.shape) != tuple(self.lm_head.weight.shape):
            raise ValueError(
                f"HunyuanOCR DFlash output head shape mismatch: expected={tuple(self.lm_head.weight.shape)}, "
                f"got={tuple(weight.shape)}"
            )
        self.lm_head.weight.data.copy_(weight)

    def _split_cache_tensors(self, cache_tensors: tuple[Tensor, ...]) -> tuple[list[Tensor], list[Tensor]]:
        expected = self.num_hidden_layers * 2
        if len(cache_tensors) != expected:
            raise ValueError(f"Expected {expected} Draft cache tensors, got {len(cache_tensors)}")
        return (
            list(cache_tensors[: self.num_hidden_layers]),
            list(cache_tensors[self.num_hidden_layers :]),
        )

    def forward_context(
        self,
        target_hidden: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        *cache_tensors: Tensor,
    ) -> tuple[Tensor, ...]:
        target_projection = self.hidden_norm(self.fc(target_hidden))
        past_keys, past_values = self._split_cache_tensors(cache_tensors)
        present_keys: list[Tensor] = []
        present_values: list[Tensor] = []
        for layer_index, layer in enumerate(self.layers):
            present_key, present_value = layer.self_attn.build_target_kv(
                target_projection,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                past_key_cache=past_keys[layer_index],
                past_value_cache=past_values[layer_index],
            )
            present_keys.append(present_key)
            present_values.append(present_value)
        return tuple(present_keys + present_values)

    def forward_context_decode(
        self,
        target_hidden: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        *cache_tensors: Tensor,
    ) -> tuple[Tensor, ...]:
        return self.forward_context(target_hidden, past_seq_length, current_input_length, *cache_tensors)

    def forward_decode(
        self,
        noise_embedding: Tensor,
        past_seq_length: Tensor,
        current_input_length: Tensor,
        attn_mask: Tensor,
        *cache_tensors: Tensor,
    ) -> Tensor:
        past_keys, past_values = self._split_cache_tensors(cache_tensors)
        hidden_states = noise_embedding
        for layer_index, layer in enumerate(self.layers):
            hidden_states = layer.forward_decode(
                hidden_states,
                past_seq_length=past_seq_length,
                current_input_length=current_input_length,
                target_key_cache=past_keys[layer_index],
                target_value_cache=past_values[layer_index],
                attn_mask=attn_mask,
            )
        logits = self.lm_head(self.norm(hidden_states))
        return logits[:, 1:16]


def dflash_graph_io_contract(*, mode: str, num_hidden_layers: int) -> dict[str, list[str]]:
    if type(num_hidden_layers) is not int or num_hidden_layers <= 0:
        raise ValueError(f"num_hidden_layers must be a positive integer, got {num_hidden_layers!r}")
    cache_input_names = [
        *[f"past_key_cache_{index}" for index in range(num_hidden_layers)],
        *[f"past_value_cache_{index}" for index in range(num_hidden_layers)],
    ]
    if mode in ("context", "context_decode"):
        input_names = ["target_hidden", "past_seq_length", "current_input_length", *cache_input_names]
        output_names = [
            *[f"present_key_cache_{index}" for index in range(num_hidden_layers)],
            *[f"present_value_cache_{index}" for index in range(num_hidden_layers)],
        ]
    elif mode == "decode":
        input_names = [
            "noise_embedding",
            "past_seq_length",
            "current_input_length",
            "attn_mask",
            *cache_input_names,
        ]
        output_names = ["draft_logits"]
    else:
        raise ValueError(f"Unsupported HunyuanOCR DFlash mode: {mode!r}")
    return {
        "input_names": input_names,
        "output_names": output_names,
        "cache_input_names": cache_input_names,
    }


def build_hunyuan_ocr_dflash_contract(
    checkpoint: HunyuanOCRDFlashCheckpoint,
    *,
    generation_eos_token_id: int | list[int],
    cache_capacity: int,
) -> dict[str, Any]:
    if type(cache_capacity) is not int or cache_capacity <= 0:
        raise ValueError(f"cache_capacity must be a positive integer, got {cache_capacity!r}")
    if type(generation_eos_token_id) is int:
        generation_eos: int | list[int] = generation_eos_token_id
    elif isinstance(generation_eos_token_id, list) and generation_eos_token_id and all(
        type(token_id) is int for token_id in generation_eos_token_id
    ):
        generation_eos = list(dict.fromkeys(generation_eos_token_id))
    else:
        raise ValueError("generation_eos_token_id must be an integer or a non-empty integer list")
    config = checkpoint.config
    head = checkpoint.output_head
    cache_names = dflash_graph_io_contract(
        mode="context",
        num_hidden_layers=config.num_hidden_layers,
    )["cache_input_names"]
    return {
        "block_size": config.block_size,
        "num_draft_tokens": config.block_size - 1,
        "candidate_hidden_offset": 1,
        "logits_layout": "candidate_order",
        "mask_token_id": config.mask_token_id,
        "noise_layout": "current_token_embedding_then_15_mask_embeddings",
        "position_mode": "draft_cache_length_linear",
        "rope_theta": config.rope_theta,
        "max_position_capacity": cache_capacity,
        "draft_eos_token_id": config.draft_eos_token_id,
        "generation_eos_token_id": generation_eos,
        "reference_dtype": "float32",
        "deployment_dtype": "float16",
        "cache": {
            "owner": "request_local_runtime_shared",
            "binding": "context_context_decode_decode_shared_inputs",
            "input_names": cache_names,
            "output_names": [name.replace("past_", "present_") for name in cache_names],
            "num_layers": config.num_hidden_layers,
            "tensor_count": config.num_hidden_layers * 2,
            "shape": [1, config.num_key_value_heads, cache_capacity, config.head_dim],
            "dtype": "float16",
            "sequence_axis": 2,
            "capacity": cache_capacity,
        },
        "output_head": {
            "tie_word_embeddings": head.tie_word_embeddings,
            "source_key": head.source_key,
            "source_shape": list(head.source_shape),
            "source_dtype": head.source_dtype,
            "deployment_dtype": head.deployment_dtype,
            "sha256": head.sha256,
        },
        "embedding": {
            "source_key": head.embedding_source_key,
            "sha256": head.embedding_sha256,
            "dtype": head.deployment_dtype,
        },
    }


def build_hunyuan_ocr_dflash_export_adapter(
    core_model: nn.Module,
    *,
    mode: str,
    num_hidden_layers: int,
) -> nn.Module:
    contract = dflash_graph_io_contract(mode=mode, num_hidden_layers=num_hidden_layers)
    core_method = {
        "context": "forward_context",
        "context_decode": "forward_context_decode",
        "decode": "forward_decode",
    }[mode]
    arg_names = contract["input_names"]
    namespace: dict[str, Any] = {}
    exec(
        f"def forward(self, {', '.join(arg_names)}):\n"
        f"    return self.core.{core_method}({', '.join(arg_names)})\n",
        {},
        namespace,
    )

    class _HunyuanOCRDFlashExportAdapter(nn.Module):
        def __init__(self, core: nn.Module) -> None:
            super().__init__()
            self.core = core

    _HunyuanOCRDFlashExportAdapter.forward = namespace["forward"]
    return _HunyuanOCRDFlashExportAdapter(core_model)


__all__ = [
    "HunyuanOCRDFlashCheckpoint",
    "HunyuanOCRDFlashConfig",
    "HunyuanOCRDFlashModel",
    "HunyuanOCRDFlashOutputHead",
    "HunyuanOCRDraftCacheController",
    "build_draft_position_ids",
    "build_dflash_noise_embedding",
    "build_hunyuan_ocr_dflash_contract",
    "build_hunyuan_ocr_dflash_export_adapter",
    "dflash_graph_io_contract",
    "load_hunyuan_ocr_dflash_checkpoint",
]