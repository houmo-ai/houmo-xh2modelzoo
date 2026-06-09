"""Unified evaluation backends for float and HMONNX inference.

Provides a common interface for model loading and generation across:
- FloatBackend: HuggingFace model.generate()
- HMONNXBackend: ONNX runtime via LLMWithMaskONNXModel or Qwen3NextONNXModel
"""

from __future__ import annotations

import json
import logging
import os
import re
import base64
import tempfile
from abc import ABC, abstractmethod
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
from PIL import Image

from .hmonnx_meta import hmonnx_meta_dict_has_embedded_vision

logger = logging.getLogger(__name__)

_MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\((?P<target>data:image/[^)\s]+|[^)\s]+)\)")


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def strip_think_content(text: Optional[str]) -> str:
    if text is None:
        return ""
    cleaned = text.replace("<|im_end|>", "").strip()
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1].strip()
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned


def _resolve_path(base_dir: Path, path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _normalize_onnx_model_type(onnx_model_type: str) -> str:
    aliases = {
        "": "LLMWithMaskONNXModel",
        "llm_with_mask": "LLMWithMaskONNXModel",
        "LLMWithMaskONNXModel": "LLMWithMaskONNXModel",
        "qwen3_next": "Qwen3NextONNXModel",
        "Qwen3NextONNXModel": "Qwen3NextONNXModel",
    }
    return aliases.get(onnx_model_type, onnx_model_type)


def _load_token_embedding(embed_path: Path, dtype: torch.dtype = torch.float16) -> nn.Module:
    torch.serialization.add_safe_globals([nn.Embedding])
    try:
        try:
            token_embedding = torch.load(str(embed_path), map_location="cpu", weights_only=False)
        except TypeError:
            token_embedding = torch.load(str(embed_path), map_location="cpu")
    finally:
        torch.serialization.clear_safe_globals()
    if isinstance(token_embedding, dict):
        if "weight" not in token_embedding:
            raise ValueError(f"Unsupported embedding state dict: {embed_path}")
        embedding = nn.Embedding(
            token_embedding["weight"].shape[0], token_embedding["weight"].shape[1]
        )
        embedding.load_state_dict(token_embedding)
        token_embedding = embedding
    token_embedding.eval()
    return token_embedding.to(dtype=dtype)


def _get_model_device(model: Any) -> torch.device:
    if hasattr(model, "device"):
        return torch.device(model.device)
    if hasattr(model, "hf_device_map") and model.hf_device_map:
        first_device = next(iter(model.hf_device_map.values()))
        if isinstance(first_device, int):
            return torch.device(f"cuda:{first_device}")
        if isinstance(first_device, str):
            return torch.device(first_device)
    first_param = next(model.parameters(), None)
    if first_param is not None:
        return first_param.device
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    return {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
    }


def _content_attr(item: Any, key: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _content_type_name(item: Any) -> str:
    raw_type = _content_attr(item, "type", "text")
    if hasattr(raw_type, "value"):
        raw_type = raw_type.value
    return str(raw_type).lower().split(".")[-1]


def _decode_image_payload(image_payload: Any) -> Image.Image:
    if isinstance(image_payload, Image.Image):
        return image_payload.convert("RGB")

    if isinstance(image_payload, str):
        if image_payload.startswith("data:image"):
            _, encoded = image_payload.split(",", 1)
            return Image.open(BytesIO(base64.b64decode(encoded))).convert("RGB")
        path = Path(image_payload)
        if path.is_file():
            return Image.open(path).convert("RGB")

    raise ValueError("Unsupported image payload in multimodal message")


def _markdown_images_to_content_parts(text: str) -> str | list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    last_end = 0
    for match in _MARKDOWN_IMAGE_RE.finditer(text):
        if match.start() > last_end:
            parts.append({"type": "text", "text": text[last_end:match.start()]})
        parts.append({"type": "image", "image": match.group("target")})
        last_end = match.end()
    if not parts:
        return text
    if last_end < len(text):
        parts.append({"type": "text", "text": text[last_end:]})
    return parts


def _normalize_messages_with_markdown_images(messages: Sequence[Dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            normalized_content = _markdown_images_to_content_parts(content)
        elif isinstance(content, list):
            normalized_content = []
            for item in content:
                if _content_type_name(item) == "text":
                    parsed_parts = _markdown_images_to_content_parts(str(_content_attr(item, "text", "")))
                    if isinstance(parsed_parts, list):
                        normalized_content.extend(parsed_parts)
                    else:
                        normalized_content.append(item)
                else:
                    normalized_content.append(item)
        else:
            normalized_content = content
        normalized_messages.append({**message, "content": normalized_content})
    return normalized_messages


def _content_parts_to_hf(parts: Sequence[Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in parts:
        item_type = _content_type_name(item)
        if item_type == "text":
            normalized.append({"type": "text", "text": str(_content_attr(item, "text", ""))})
        elif item_type == "image":
            normalized.append({"type": "image", "image": _decode_image_payload(_content_attr(item, "image"))})
        elif item_type == "audio":
            audio_part = {"type": "audio", "audio": _content_attr(item, "audio")}
            sampling_rate = _content_attr(item, "sampling_rate")
            if sampling_rate is not None:
                audio_part["sampling_rate"] = sampling_rate
            normalized.append(audio_part)
    return normalized


def _messages_to_processor_messages(messages: Sequence[Dict[str, Any]]) -> list[dict[str, Any]]:
    processor_messages: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            normalized_content = _content_parts_to_hf(content)
        else:
            normalized_content = [{"type": "text", "text": str(content)}]
        processor_messages.append({"role": message.get("role", "user"), "content": normalized_content})
    return processor_messages


def _messages_have_multimodal_content(messages: Sequence[Dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content", "")
        if not isinstance(content, list):
            continue
        for item in content:
            if _content_type_name(item) in {"image", "audio", "video"}:
                return True
    return False


def _content_parts_to_text(parts: Sequence[Any], image_placeholder: str = "<|image|>") -> str:
    chunks: list[str] = []
    for item in parts:
        item_type = _content_type_name(item)
        if item_type == "text":
            chunks.append(str(_content_attr(item, "text", "")))
        elif item_type == "image":
            chunks.append(f"\n\n{image_placeholder}\n\n")
    return "".join(chunks)


def _messages_to_text_and_images(messages: Sequence[Dict[str, Any]]) -> tuple[list[dict[str, str]], list[Image.Image]]:
    rendered_messages: list[dict[str, str]] = []
    images: list[Image.Image] = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                item_type = _content_type_name(item)
                if item_type == "text":
                    chunks.append(str(_content_attr(item, "text", "")))
                elif item_type == "image":
                    chunks.append("\n\n<|image|>\n\n")
                    images.append(_decode_image_payload(_content_attr(item, "image")))
            rendered_content = "".join(chunks)
        else:
            rendered_content = str(content)
        rendered_messages.append({"role": message.get("role", "user"), "content": rendered_content})
    return rendered_messages, images


def _save_temp_image(image: Image.Image) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="hm_eval_vlm_", suffix=".png", delete=False)
    path = Path(handle.name)
    handle.close()
    image.save(path)
    return path


def _maybe_disable_thinking(tokenizer: Any) -> None:
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        return
    old = tokenizer.chat_template
    new = old.replace("<|im_start|>assistant\n<think>\n", "<|im_start|>assistant\n</think>\n")
    if new != old:
        tokenizer.chat_template = new


def apply_chat_template(
    tokenizer_or_processor: Any,
    messages: Sequence[Dict[str, Any]],
    disable_thinking: bool = True,
) -> Dict[str, Any]:
    """Apply chat template and return tokenized inputs."""
    if disable_thinking:
        tok = getattr(tokenizer_or_processor, "tokenizer", tokenizer_or_processor)
        _maybe_disable_thinking(tok)

    # Check if it's a processor (Gemma-style) or tokenizer (Qwen-style)
    if hasattr(tokenizer_or_processor, "image_processor"):
        # Processor: normalize messages for Gemma format
        gemma_msgs = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                gemma_msgs.append({"role": m["role"], "content": _content_parts_to_hf(content)})
            else:
                gemma_msgs.append({"role": m["role"], "content": [{"type": "text", "text": str(content)}]})
        try:
            return tokenizer_or_processor.apply_chat_template(
                gemma_msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer_or_processor.apply_chat_template(
                gemma_msgs,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
    else:
        # Tokenizer: render chat prompt as string then tokenize
        text_messages = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                content = _content_parts_to_text(content)
            text_messages.append({**message, "content": content})
        try:
            prompt = tokenizer_or_processor.apply_chat_template(
                text_messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            prompt = tokenizer_or_processor.apply_chat_template(
                text_messages, tokenize=False, add_generation_prompt=True,
            )
        inputs = tokenizer_or_processor(prompt, return_tensors="pt")
        inputs["_prompt"] = prompt
        return inputs


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class EvalBackend(ABC):
    """Abstract evaluation backend."""

    backend_type: str = ""

    @abstractmethod
    def generate(self, messages: Sequence[Dict[str, Any]], max_tokens: int) -> str:
        """Generate text from a list of chat messages. Returns the output string."""

    def cleanup(self) -> None:
        """Release resources."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Float (HuggingFace) backend
# ---------------------------------------------------------------------------

class FloatBackend(EvalBackend):
    """Official HuggingFace float-point backend."""

    backend_type = "float"

    def __init__(
        self,
        hf_model_dir: str,
        model_class: str = "",
        processor_class: str = "AutoTokenizer",
        dtype: str = "bfloat16",
        device_map: str = "auto",
        experts_implementation: str = "eager",
        disable_thinking: bool = True,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.disable_thinking = disable_thinking
        is_gemma4 = bool(model_class and "Gemma4" in model_class)
        dtype_map = {
            "auto": "auto", "float16": torch.float16,
            "bfloat16": torch.bfloat16, "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # Load processor/tokenizer
        if processor_class == "AutoProcessor":
            from transformers import AutoProcessor

            processor_kwargs: dict[str, Any] = {}
            if not is_gemma4:
                processor_kwargs["trust_remote_code"] = True
            self.processor = AutoProcessor.from_pretrained(hf_model_dir, **processor_kwargs)
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            if self.tokenizer is None:
                tokenizer_kwargs: dict[str, Any] = {}
                if not is_gemma4:
                    tokenizer_kwargs["trust_remote_code"] = True
                self.tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, **tokenizer_kwargs)
        else:
            tokenizer_kwargs: dict[str, Any] = {}
            if not is_gemma4:
                tokenizer_kwargs["trust_remote_code"] = True
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, **tokenizer_kwargs)
            self.processor = self.tokenizer

        # Load model
        load_kwargs: dict[str, Any] = {
            "torch_dtype": torch_dtype,
            "device_map": device_map,
        }
        if not is_gemma4:
            load_kwargs["trust_remote_code"] = True

        if is_gemma4:
            from transformers import Gemma4ForConditionalGeneration
            load_kwargs["experts_implementation"] = experts_implementation
            self.model = Gemma4ForConditionalGeneration.from_pretrained(
                hf_model_dir, **load_kwargs
            )
            if hasattr(self.model, "set_experts_implementation"):
                self.model.set_experts_implementation(experts_implementation)
        elif model_class == "Qwen3_5ForConditionalGeneration":
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5ForConditionalGeneration,
            )

            self.model = Qwen3_5ForConditionalGeneration.from_pretrained(
                hf_model_dir,
                **load_kwargs,
            )
        elif model_class == "Qwen3_5MoeForConditionalGeneration":
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
                Qwen3_5MoeForConditionalGeneration,
            )

            load_kwargs["experts_implementation"] = experts_implementation
            self.model = Qwen3_5MoeForConditionalGeneration.from_pretrained(
                hf_model_dir,
                **load_kwargs,
            )
            if hasattr(self.model, "set_experts_implementation"):
                self.model.set_experts_implementation(experts_implementation)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(hf_model_dir, **load_kwargs)

        self.model.eval()
        logger.info(
            "Loaded float backend: %s dtype=%s device_map=%s experts_implementation=%s",
            hf_model_dir,
            dtype,
            device_map,
            experts_implementation,
        )

    def generate(self, messages: Sequence[Dict[str, Any]], max_tokens: int) -> str:
        messages = _normalize_messages_with_markdown_images(messages)
        device = _get_model_device(self.model)
        inputs = apply_chat_template(self.processor, messages, disable_thinking=self.disable_thinking)
        prompt = inputs.pop("_prompt", "")
        inputs = _to_device(inputs, device)

        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[-1]
        output_text = self.tokenizer.decode(generation[0][input_len:], skip_special_tokens=True)
        return output_text


# ---------------------------------------------------------------------------
# HMONNX backend
# ---------------------------------------------------------------------------

class HMONNXBackend(EvalBackend):
    """HMONNX ONNX runtime backend (supports both LLMWithMaskONNXModel and Qwen3NextONNXModel)."""

    backend_type = "hmonnx"

    def __init__(
        self,
        export_meta_info_path: str,
        onnx_model_type: str = "LLMWithMaskONNXModel",
        device: str = "cuda",
        exec_device: str = "cuda",
        auto_offload: bool = False,
        resource_tight_mode: bool = False,
        disable_thinking: bool = True,
        vision_export_meta_info_path: str = "",
    ) -> None:
        from transformers import AutoTokenizer, AutoProcessor

        self.disable_thinking = disable_thinking
        self.onnx_model_type = _normalize_onnx_model_type(onnx_model_type)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.exec_device = torch.device(exec_device if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16
        self._merak_model = None
        self._merak_tokenizer = None
        self._merak_processor = None
        self._merak_runtime_meta_path: Optional[Path] = None
        self._merak_runtime_meta: Optional[dict[str, Any]] = None
        self._merak_prefill_chunk_length: Optional[int] = None
        self._vision_meta_path = Path(vision_export_meta_info_path).resolve() if vision_export_meta_info_path else None
        self._vision_meta: Optional[dict[str, Any]] = None
        if self._vision_meta_path:
            with open(self._vision_meta_path, "r", encoding="utf-8") as f:
                self._vision_meta = json.load(f)

        meta_path = Path(export_meta_info_path).resolve()
        self.meta_path = meta_path
        model_dir = meta_path.parent
        with open(meta_path, "r", encoding="utf-8") as f:
            self.meta_info = json.load(f)

        merak_meta_path = self._resolve_merak_meta_path(meta_path, self.meta_info)
        if merak_meta_path is None and self._vision_meta_path is not None:
            merak_meta_path = self._resolve_legacy_multimodal_merak_meta(meta_path, self.meta_info)
        if merak_meta_path is not None:
            self._init_merak_hmonnx(merak_meta_path)
            logger.info("Loaded xhmodel_merak HMONNX backend from %s", merak_meta_path)
            return

        self.max_context_tokens = self._infer_max_context_tokens()

        hf_config_dir = _resolve_path(model_dir, self.meta_info["hf_config"])
        token_embedding_file = _resolve_path(model_dir, self.meta_info["token_embedding_file"])

        # Load tokenizer/processor
        try:
            self.processor = AutoProcessor.from_pretrained(str(hf_config_dir), trust_remote_code=True)
            self.tokenizer = getattr(self.processor, "tokenizer", None)
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(str(hf_config_dir), trust_remote_code=True)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(str(hf_config_dir), trust_remote_code=True)
            self.processor = self.tokenizer

        token_embedding = _load_token_embedding(token_embedding_file, self.dtype)

        if self.onnx_model_type == "Qwen3NextONNXModel":
            self._init_qwen3next(model_dir, token_embedding, auto_offload, resource_tight_mode)
        else:
            self._init_llm_with_mask(model_dir, token_embedding)

        logger.info("Loaded HMONNX backend (%s) from %s", onnx_model_type, meta_path)

    @staticmethod
    def _resolve_merak_meta_path(meta_path: Path, meta_info: dict[str, Any]) -> Optional[Path]:
        if meta_info.get("model_config", {}).get("model_type"):
            return meta_path

        exported_dir = meta_info.get("exported_dir")
        if exported_dir:
            golden_meta_path = (meta_path.parent / exported_dir / "golden_meta_info.json").resolve()
            if HMONNXBackend._is_valid_merak_meta_path(golden_meta_path):
                return golden_meta_path
        return None

    @staticmethod
    def _is_valid_merak_meta_path(meta_path: Path) -> bool:
        if not meta_path.exists():
            return False
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta_info = json.load(f)
        except Exception:
            return False
        return bool(meta_info.get("model_config", {}).get("model_type"))

    @staticmethod
    def _resolve_legacy_multimodal_merak_meta(meta_path: Path, meta_info: dict[str, Any]) -> Optional[Path]:
        config_name = str(meta_info.get("config") or meta_path.stem)
        family_key = Path(config_name).stem.lower().split("_llm_", 1)[0]
        if not family_key.startswith("gemma4_moe_with_mask"):
            return None

        work_dirs_root = Path(__file__).resolve().parents[2] / "work_dirs"
        if not work_dirs_root.exists():
            return None

        legacy_hf_basename = Path(str(meta_info.get("hf_model") or "")).name
        expect_mtp = "_mtp_" in f"_{family_key}_"
        best_score = -1
        best_path: Optional[Path] = None

        for candidate_root in work_dirs_root.iterdir():
            if not candidate_root.is_dir():
                continue
            candidate_name = candidate_root.name.lower()
            if not candidate_name.startswith(family_key):
                continue

            for candidate_meta_path in candidate_root.glob("**/golden_meta_info.json"):
                if not HMONNXBackend._is_valid_merak_meta_path(candidate_meta_path):
                    continue
                try:
                    with open(candidate_meta_path, "r", encoding="utf-8") as f:
                        candidate_meta = json.load(f)
                except Exception:
                    continue

                model_config = candidate_meta.get("model_config", {})
                visual_config = model_config.get("visual_config", {}) or {}
                if model_config.get("model_type") != "Gemma4ForConditionalGeneration_with_mask":
                    continue
                if visual_config.get("model_type") != "Gemma4ForConditionalGeneration_visual":
                    continue

                score = 0
                score += 3 if ("mtp" in candidate_name) == expect_mtp else 0
                score += 2 if "valid" not in candidate_name else 0
                score += 2 if not model_config.get("only_first_block", False) else 0
                score += 1 if model_config.get("prefill_chunk_length") == meta_info.get("input_sequence_length") else 0
                score += 1 if model_config.get("num_hidden_layers") == meta_info.get("num_hidden_layers") else 0

                candidate_hf_model = Path(str(model_config.get("hf_model") or "")).name
                if legacy_hf_basename and candidate_hf_model == legacy_hf_basename:
                    score += 1

                if score > best_score:
                    best_score = score
                    best_path = candidate_meta_path.resolve()

        if best_path is not None:
            logger.info(
                "Resolved legacy Gemma4 multimodal export %s to workspace runtime meta %s",
                meta_path,
                best_path,
            )
        return best_path

    def _init_merak_hmonnx(self, runtime_meta_path: Path) -> None:
        from xhmodel_merak.xh_llm import AutoLLMHONNXModel

        with open(runtime_meta_path, "r", encoding="utf-8") as f:
            self._merak_runtime_meta = json.load(f)
        self._merak_runtime_meta_path = runtime_meta_path
        hmonnx_model = AutoLLMHONNXModel.from_pretrained(str(runtime_meta_path))
        hmonnx_model.to(self.device)
        self.dtype = getattr(hmonnx_model, "dtype", self.dtype) or self.dtype
        self._merak_model = hmonnx_model
        self._merak_tokenizer = hmonnx_model.get_tokenizer(trust_remote_code=True)
        if hasattr(hmonnx_model, "get_tf_processor"):
            try:
                self._merak_processor = hmonnx_model.get_tf_processor()
            except Exception:
                logger.debug("Merak HMONNX model does not expose a usable TF processor", exc_info=True)
        self.max_context_tokens = self._infer_merak_max_context_tokens()
        self._merak_prefill_chunk_length = self._infer_merak_prefill_chunk_length()

    def _is_unified_gemma4_merak_runtime(self) -> bool:
        if self._merak_runtime_meta is None:
            return False
        model_config = self._merak_runtime_meta.get("model_config", {})
        return isinstance(model_config, dict) and model_config.get("model_type") == "Gemma4ForConditionalGeneration"

    def _infer_merak_max_context_tokens(self) -> Optional[int]:
        if self._merak_runtime_meta is None:
            return None

        candidates: list[int] = []
        model_config = self._merak_runtime_meta.get("model_config", {})
        if isinstance(model_config, dict):
            context_length = model_config.get("context_max_length")
            if isinstance(context_length, int) and context_length > 0:
                candidates.append(context_length)

        kv_cache = self._merak_runtime_meta.get("kv_cache", {})
        if isinstance(kv_cache, dict):
            kv_cache_shape = kv_cache.get("kv_cache_shape")
            if isinstance(kv_cache_shape, list) and len(kv_cache_shape) >= 3:
                candidates.append(int(kv_cache_shape[2]))

        kv_cache_shapes_per_layer = self._merak_runtime_meta.get("kv_cache_shapes_per_layer")
        if isinstance(kv_cache_shapes_per_layer, list):
            for shape in kv_cache_shapes_per_layer:
                if isinstance(shape, list) and len(shape) >= 3:
                    candidates.append(int(shape[2]))

        return min(candidates) if candidates else None

    def _infer_merak_prefill_chunk_length(self) -> Optional[int]:
        if self._merak_runtime_meta is None:
            return None
        model_config = self._merak_runtime_meta.get("model_config", {})
        if isinstance(model_config, dict):
            prefill_chunk_length = model_config.get("prefill_chunk_length")
            if isinstance(prefill_chunk_length, int) and prefill_chunk_length > 0:
                return prefill_chunk_length
        input_sequence_length = self._merak_runtime_meta.get("input_sequence_length")
        if isinstance(input_sequence_length, int) and input_sequence_length > 0:
            return input_sequence_length
        return None

    def _merak_max_prefill_tokens(self) -> Optional[int]:
        if self.max_context_tokens is None:
            return None
        max_prefill_tokens = self.max_context_tokens - 1
        if self._merak_prefill_chunk_length is not None and self._merak_prefill_chunk_length > 1:
            max_prefill_tokens = min(max_prefill_tokens, self.max_context_tokens - self._merak_prefill_chunk_length)
        return max(1, max_prefill_tokens)

    def _effective_merak_max_new_tokens(self, input_length: int, requested_max_tokens: int) -> int:
        if self.max_context_tokens is None:
            return max(1, requested_max_tokens)
        available_tokens = max(1, self.max_context_tokens - input_length)
        effective_tokens = max(1, min(requested_max_tokens, available_tokens))
        if effective_tokens != requested_max_tokens:
            logger.warning(
                "Merak HMONNX max_new_tokens clipped from %s to %s because exported context window is %s "
                "and prompt length is %s (export_meta=%s).",
                requested_max_tokens,
                effective_tokens,
                self.max_context_tokens,
                input_length,
                self._merak_runtime_meta_path or self.meta_path,
            )
        return effective_tokens

    def _truncate_merak_text_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        max_prefill_tokens = self._merak_max_prefill_tokens()
        if max_prefill_tokens is None or input_ids.shape[-1] <= max_prefill_tokens:
            return input_ids
        truncated = input_ids[:, -max_prefill_tokens:].contiguous()
        logger.warning(
            "Merak HMONNX prompt exceeds exported prefill/cache window; truncating left context from %s to %s "
            "tokens (context=%s, prefill_chunk=%s, export_meta=%s).",
            input_ids.shape[-1],
            max_prefill_tokens,
            self.max_context_tokens,
            self._merak_prefill_chunk_length,
            self._merak_runtime_meta_path or self.meta_path,
        )
        return truncated

    def _truncate_merak_multimodal_inputs(
        self,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_prefill_tokens = self._merak_max_prefill_tokens()
        if max_prefill_tokens is None or input_ids.shape[-1] <= max_prefill_tokens:
            return input_ids, mm_token_type_ids, image_embeds

        original_length = input_ids.shape[-1]
        cut_index = original_length - max_prefill_tokens
        mm_values = mm_token_type_ids[0].tolist()
        if 0 < cut_index < original_length and mm_values[cut_index] != 0:
            while cut_index < original_length and mm_values[cut_index] != 0:
                cut_index += 1
        if cut_index >= original_length:
            cut_index = original_length - max_prefill_tokens

        keep_indices = torch.arange(cut_index, original_length, dtype=torch.long)
        truncated_input_ids = input_ids.index_select(1, keep_indices).contiguous()
        truncated_mm_token_type_ids = mm_token_type_ids.index_select(1, keep_indices).contiguous()

        image_positions = torch.nonzero(mm_token_type_ids[0] != 0, as_tuple=False).flatten()
        kept_image_positions = torch.nonzero(truncated_mm_token_type_ids[0] != 0, as_tuple=False).flatten() + cut_index
        if kept_image_positions.numel() == 0:
            truncated_image_embeds = image_embeds[:0]
        else:
            image_row_lookup = {int(pos.item()): row for row, pos in enumerate(image_positions)}
            kept_rows = [image_row_lookup[int(pos.item())] for pos in kept_image_positions]
            truncated_image_embeds = image_embeds.index_select(
                0,
                torch.tensor(kept_rows, dtype=torch.long, device=image_embeds.device),
            ).contiguous()

        logger.warning(
            "Merak HMONNX multimodal prompt exceeds exported prefill/cache window; truncating left context "
            "from %s to %s tokens and image tokens from %s to %s (context=%s, prefill_chunk=%s, export_meta=%s).",
            original_length,
            truncated_input_ids.shape[-1],
            int(mm_token_type_ids.sum().item()),
            int(truncated_mm_token_type_ids.sum().item()),
            self.max_context_tokens,
            self._merak_prefill_chunk_length,
            self._merak_runtime_meta_path or self.meta_path,
        )
        return truncated_input_ids, truncated_mm_token_type_ids, truncated_image_embeds

    def _infer_max_context_tokens(self) -> Optional[int]:
        candidates: list[int] = []

        kv_cache_shape = self.meta_info.get("kv_cache_shape")
        if isinstance(kv_cache_shape, list) and len(kv_cache_shape) >= 3:
            candidates.append(int(kv_cache_shape[2]))

        kv_cache_shapes_per_layer = self.meta_info.get("kv_cache_shapes_per_layer")
        if isinstance(kv_cache_shapes_per_layer, list):
            for shape in kv_cache_shapes_per_layer:
                if isinstance(shape, list) and len(shape) >= 3:
                    candidates.append(int(shape[2]))

        sliding_window_cfg = self.meta_info.get("sliding_window_cfg", {})
        if isinstance(sliding_window_cfg, dict):
            global_window = sliding_window_cfg.get("global_attention_window_size")
            if isinstance(global_window, int) and global_window > 0:
                candidates.append(global_window)

        if not candidates:
            return None
        return max(candidates)

    def _truncate_input_ids_for_context(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        if self.max_context_tokens is None:
            return input_ids

        input_length = input_ids.shape[-1]
        decode_budget = max(1, min(max_new_tokens, self.max_context_tokens - 1))
        max_prompt_tokens = max(1, self.max_context_tokens - decode_budget)
        if input_length <= max_prompt_tokens:
            return input_ids

        truncated = input_ids[:, -max_prompt_tokens:].contiguous()
        logger.warning(
            "HMONNX prompt exceeds exported context window; truncating left context from %s to %s tokens "
            "to preserve %s max_new_tokens slots (export_meta=%s). The UI max_tokens value is treated as "
            "max_new_tokens and does not include prompt tokens.",
            input_length,
            max_prompt_tokens,
            decode_budget,
            self.meta_path,
        )
        return truncated

    def _init_qwen3next(
        self,
        model_dir: Path,
        token_embedding: nn.Module,
        auto_offload: bool,
        resource_tight_mode: bool,
    ) -> None:
        """Initialize Qwen3NextONNXModel (has built-in generate)."""
        from xh_model_zoo.api import Config, ConfigDict
        from xh_model_zoo.xh_llm.models.builder import MODELS
        import xh_model_zoo.xh_llm.models.qwen3_next  # noqa: F401

        prefill_onnx_key = "prefill_onnx_file" if "prefill_onnx_file" in self.meta_info else "prefill_onnx"
        decode_onnx_key = "decode_onnx_file" if "decode_onnx_file" in self.meta_info else "decode_onnx"
        prefill_onnx = _resolve_path(model_dir, self.meta_info[prefill_onnx_key])
        decode_onnx = _resolve_path(model_dir, self.meta_info[decode_onnx_key])

        max_context_tokens = None
        kv_cache_shape = self.meta_info.get("kv_cache_shape")
        if kv_cache_shape is None:
            kv_cache_cfg = self.meta_info.get("kv_cache", {})
            if isinstance(kv_cache_cfg, dict):
                kv_cache_shape = kv_cache_cfg.get("shape")
        if isinstance(kv_cache_shape, list) and len(kv_cache_shape) >= 3:
            max_context_tokens = int(kv_cache_shape[2])
        elif isinstance(self.meta_info.get("max_context_tokens"), int):
            max_context_tokens = int(self.meta_info["max_context_tokens"])

        cfg = Config()
        cfg.model = ConfigDict()
        cfg.model.type = "Qwen3NextONNXModel"
        cfg.model.prefill = ConfigDict(onnx=str(prefill_onnx))
        cfg.model.decode = ConfigDict(onnx=str(decode_onnx))
        cfg.model.max_context_tokens = max_context_tokens
        cfg.model.auto_offload = auto_offload
        cfg.model.resource_tight_mode = resource_tight_mode

        self.model = MODELS.build(cfg.model)
        self.model.set_input_embeddings(token_embedding)
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_id is not None:
            self.model.set_pad_token_id(pad_id)
        self.model.to(self.device)
        self.model.set_exec_device(self.exec_device)
        self.model.to(self.dtype)

    def _init_llm_with_mask(self, model_dir: Path, token_embedding: nn.Module) -> None:
        """Initialize LLMWithMaskONNXModel (manual prefill/decode loop)."""
        from xh_model_zoo.xh_llm.models.builder import MODELS
        from xh_model_zoo.xh_llm.models.llm_with_mask_onnx_model import LLMWithMaskONNXModel

        prefill_onnx = _resolve_path(model_dir, self.meta_info["prefill_onnx"])
        decode_onnx = _resolve_path(model_dir, self.meta_info["decode_onnx"])

        hmonnx_cfg = dict(
            type="LLMWithMaskONNXModel",
            prefill=dict(
                onnx=str(prefill_onnx),
                input_sequence_length=self.meta_info.get("input_sequence_length", 256),
            ),
            decode=dict(onnx=str(decode_onnx)),
            kv_cache=dict(
                num_hidden_layers=self.meta_info["num_hidden_layers"],
                shape=self.meta_info["kv_cache_shape"],
            ),
            sliding_window_cfg=self.meta_info.get("sliding_window_cfg", {}),
        )
        self.model: LLMWithMaskONNXModel = MODELS.build(hmonnx_cfg)
        self.model.set_input_embeddings(token_embedding)

        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_id is not None:
            self.model.pad_token_id = pad_id

        # Setup heterogeneous KV cache if needed
        kv_cache_shapes_per_layer = self.meta_info.get("kv_cache_shapes_per_layer")
        if kv_cache_shapes_per_layer is not None:
            from xhquant.core import HybridCacheTensor
            for index, shape in enumerate(kv_cache_shapes_per_layer):
                setattr(
                    self.model, f"past_k_cache_{index}",
                    HybridCacheTensor(torch.zeros(shape, dtype=self.dtype, device=self.device)),
                )
                setattr(
                    self.model, f"past_v_cache_{index}",
                    HybridCacheTensor(torch.zeros(shape, dtype=self.dtype, device=self.device)),
                )

        self.model.eval()
        self.model.to(self.device)
        self.model.set_exec_device(self.exec_device)
        self.model.to(self.dtype)

    def generate(self, messages: Sequence[Dict[str, Any]], max_tokens: int) -> str:
        messages = _normalize_messages_with_markdown_images(messages)
        if self._merak_model is not None:
            return self._generate_merak(messages, max_tokens)
        if self.onnx_model_type == "Qwen3NextONNXModel":
            return self._generate_qwen3next(messages, max_tokens)
        return self._generate_llm_with_mask(messages, max_tokens)

    def _generate_merak(self, messages: Sequence[Dict[str, Any]], max_tokens: int) -> str:
        if self._merak_model is None or self._merak_tokenizer is None:
            raise RuntimeError("xhmodel_merak HMONNX backend is not initialized")
        from xhmodel_merak.xh_llm import LLMInferenceContextManager

        has_multimodal_content = _messages_have_multimodal_content(messages)
        has_embedded_vision_runtime = hmonnx_meta_dict_has_embedded_vision(
            self._merak_runtime_meta or {}
        )

        if has_multimodal_content and has_embedded_vision_runtime:
            if self._merak_processor is not None:
                return self._generate_merak_processor_multimodal(messages, max_tokens)

            model_type = (
                (self._merak_runtime_meta or {}).get("model_config", {}).get("model_type")
                or "current"
            )
            if self._is_unified_gemma4_merak_runtime():
                raise RuntimeError(
                    "Gemma4 unified HMONNX meta includes embedded vision runtime, but the runtime processor "
                    "could not be initialized. Please check xhmodel_merak Gemma4 processor support."
                )
            raise RuntimeError(
                f"{model_type} HMONNX meta includes embedded vision runtime, but the runtime processor "
                "could not be initialized. Please check xhmodel_merak processor support."
            )

        rendered_messages, images = _messages_to_text_and_images(messages)
        if images:
            return self._generate_merak_multimodal(rendered_messages, images, max_tokens)

        inputs = apply_chat_template(self._merak_tokenizer, rendered_messages, disable_thinking=self.disable_thinking)
        inputs.pop("_prompt", None)
        inputs["input_ids"] = self._truncate_merak_text_input_ids(inputs["input_ids"])
        if "attention_mask" in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"], dtype=torch.long)
        inputs = _to_device(inputs, self.device)
        input_length = inputs["input_ids"].shape[-1]
        pad_token_id = self._merak_tokenizer.pad_token_id or self._merak_tokenizer.eos_token_id
        effective_max_tokens = self._effective_merak_max_new_tokens(input_length, max_tokens)

        with torch.inference_mode(), LLMInferenceContextManager(self._merak_model):
            generated_ids = self._merak_model.generate(
                **inputs,
                max_new_tokens=effective_max_tokens,
                do_sample=False,
                temperature=0.0,
                pad_token_id=pad_token_id,
            )

        output_ids = generated_ids[0][input_length:]
        output_text = self._merak_tokenizer.decode(output_ids, skip_special_tokens=True)
        return output_text

    def _generate_merak_processor_multimodal(
        self,
        messages: Sequence[Dict[str, Any]],
        max_tokens: int,
    ) -> str:
        if self._merak_model is None or self._merak_tokenizer is None or self._merak_processor is None:
            raise RuntimeError("xhmodel_merak processor backend is not initialized")
        from xhmodel_merak.xh_llm import LLMInferenceContextManager

        processor_messages = _messages_to_processor_messages(messages)
        try:
            model_inputs = self._merak_processor.apply_chat_template(
                processor_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=not self.disable_thinking,
            )
        except TypeError:
            model_inputs = self._merak_processor.apply_chat_template(
                processor_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )

        if hasattr(model_inputs, "to"):
            model_inputs = model_inputs.to(self.device)
        else:
            model_inputs = _to_device(dict(model_inputs), self.device)

        # Truncate multimodal inputs if they exceed the context window.
        # Unlike the text-only path which truncates ``input_ids`` directly,
        # multimodal inputs carry ``pixel_values`` and side tensors whose
        # image-token slices must stay in sync with ``input_ids``.  When the
        # prompt is short enough the call is a no-op.
        model_inputs = self._truncate_merak_processor_multimodal_inputs(model_inputs)

        input_length = model_inputs["input_ids"].shape[-1]
        pad_token_id = self._merak_tokenizer.pad_token_id or self._merak_tokenizer.eos_token_id
        effective_max_tokens = self._effective_merak_max_new_tokens(input_length, max_tokens)
        generation_kwargs = dict(model_inputs)
        generation_kwargs.update(
            max_new_tokens=effective_max_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=pad_token_id,
        )

        with torch.inference_mode(), LLMInferenceContextManager(self._merak_model):
            generated_ids = self._merak_model.generate(**generation_kwargs)

        output_ids = generated_ids[0][input_length:]
        output_text = self._merak_tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text

    def _truncate_merak_processor_multimodal_inputs(
        self,
        model_inputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Left-truncate processor-style multimodal inputs to fit the export window.

        When the total input length exceeds the exported prefill / cache window
        we need to remove left-context tokens from input_ids, attention_mask and
        mm_token_type_ids. We also shrink pixel_values and image_position_ids
        so that the number of remaining image tokens stays consistent.
        """
        max_prefill_tokens = self._merak_max_prefill_tokens()
        if max_prefill_tokens is None:
            return model_inputs

        input_ids = model_inputs.get("input_ids")
        if input_ids is None or input_ids.shape[-1] <= max_prefill_tokens:
            return model_inputs

        original_length = input_ids.shape[-1]
        cut_index = original_length - max_prefill_tokens
        mm_token_type_ids = model_inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_values = mm_token_type_ids[0].tolist()
            # Avoid cutting in the middle of an image token span
            if 0 < cut_index < original_length and mm_values[cut_index] != 0:
                while cut_index < original_length and mm_values[cut_index] != 0:
                    cut_index += 1
            if cut_index >= original_length:
                cut_index = original_length - max_prefill_tokens

        keep_indices = torch.arange(cut_index, original_length, dtype=torch.long)
        truncated = dict(model_inputs)
        truncated["input_ids"] = input_ids.index_select(1, keep_indices).contiguous()
        if "attention_mask" in model_inputs:
            truncated["attention_mask"] = model_inputs["attention_mask"].index_select(1, keep_indices).contiguous()
        if mm_token_type_ids is not None:
            truncated["mm_token_type_ids"] = mm_token_type_ids.index_select(1, keep_indices).contiguous()

        pixel_values = model_inputs.get("pixel_values")
        image_position_ids = model_inputs.get("image_position_ids")
        if pixel_values is not None and mm_token_type_ids is not None:
            image_positions = torch.nonzero(mm_token_type_ids[0] != 0, as_tuple=False).flatten()
            kept_image_positions = torch.nonzero(truncated["mm_token_type_ids"][0] != 0, as_tuple=False).flatten() + cut_index
            if kept_image_positions.numel() == 0:
                truncated["pixel_values"] = pixel_values[:0]
                if image_position_ids is not None:
                    truncated["image_position_ids"] = image_position_ids[:0]
            else:
                image_row_lookup = {int(pos.item()): row for row, pos in enumerate(image_positions)}
                kept_rows = [image_row_lookup[int(pos.item())] for pos in kept_image_positions]
                truncated["pixel_values"] = pixel_values.index_select(
                    0, torch.tensor(kept_rows, dtype=torch.long, device=pixel_values.device),
                ).contiguous()
                if image_position_ids is not None:
                    truncated["image_position_ids"] = image_position_ids.index_select(
                        0, torch.tensor(kept_rows, dtype=torch.long, device=image_position_ids.device),
                    ).contiguous()

        logger.warning(
            "Merak HMONNX multimodal prompt exceeds exported prefill/cache window; truncating left context "
            "from %s to %s tokens (context=%s, prefill_chunk=%s, export_meta=%s).",
            original_length,
            truncated["input_ids"].shape[-1],
            self.max_context_tokens,
            self._merak_prefill_chunk_length,
            self._merak_runtime_meta_path or self.meta_path,
        )
        return truncated

    def _generate_merak_multimodal(
        self,
        rendered_messages: Sequence[Dict[str, str]],
        images: Sequence[Image.Image],
        max_tokens: int,
    ) -> str:
        if self._vision_meta_path is None or self._vision_meta is None:
            raise ValueError("多模态 HMONNX 评测需要同时提供 vision export_meta_info.json。")
        if self._merak_runtime_meta_path is None or self._merak_runtime_meta is None:
            raise RuntimeError("Merak HMONNX runtime meta is not initialized")

        from examples_merak.llm.gemma4_moe.gemma4_moe_with_mask_xh_vlm_generate import (
            build_image_embeds,
            build_vision_bidirectional_mask,
            resolve_eos_token_id,
            resolve_image_token_id,
        )
        from xhmodel_merak.xh_llm import LLMInferenceContextManager

        image_embeds_list: list[torch.Tensor] = []
        image_token_counts: list[int] = []
        temp_paths: list[Path] = []
        try:
            for image in images:
                temp_path = _save_temp_image(image)
                temp_paths.append(temp_path)
                image_embeds = build_image_embeds(
                    image_path=temp_path,
                    llm_runtime_meta=self._merak_runtime_meta,
                    vision_meta=self._vision_meta,
                    vision_meta_path=self._vision_meta_path,
                    device=str(self.device),
                )
                if image_embeds.dim() == 3 and image_embeds.shape[0] == 1:
                    image_embeds = image_embeds[0]
                image_embeds_list.append(image_embeds)
                image_token_counts.append(int(image_embeds.shape[0]))
        finally:
            for temp_path in temp_paths:
                temp_path.unlink(missing_ok=True)

        prompt_text = self._render_chat_prompt(rendered_messages)
        prompt_input_ids = self._merak_tokenizer(
            prompt_text,
            return_tensors="pt",
            add_special_tokens=False,
        ).input_ids
        image_token_id = resolve_image_token_id(None, self._merak_runtime_meta, self._merak_runtime_meta_path)
        input_ids, mm_token_type_ids = self._expand_image_placeholders(
            prompt_input_ids,
            image_token_id,
            image_token_counts,
        )
        image_embeds = torch.cat(image_embeds_list, dim=0)
        if int(mm_token_type_ids.sum().item()) != int(image_embeds.shape[0]):
            raise ValueError(
                f"Expanded image token count mismatch: prompt={int(mm_token_type_ids.sum().item())}, "
                f"image_embeds={int(image_embeds.shape[0])}"
            )
        input_ids, mm_token_type_ids, image_embeds = self._truncate_merak_multimodal_inputs(
            input_ids,
            mm_token_type_ids,
            image_embeds,
        )
        if int(mm_token_type_ids.sum().item()) != int(image_embeds.shape[0]):
            raise ValueError(
                f"Truncated image token count mismatch: prompt={int(mm_token_type_ids.sum().item())}, "
                f"image_embeds={int(image_embeds.shape[0])}"
            )

        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        pad_token_id = self._merak_tokenizer.pad_token_id or self._merak_tokenizer.eos_token_id
        effective_max_tokens = self._effective_merak_max_new_tokens(input_ids.shape[-1], max_tokens)
        generation_kwargs: dict[str, Any] = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "mm_token_type_ids": mm_token_type_ids.to(self.device),
            "image_embeds": image_embeds.to(device=self.device, dtype=getattr(self._merak_model, "dtype", self.dtype)),
            "max_new_tokens": effective_max_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "pad_token_id": pad_token_id,
        }
        eos_token_id = resolve_eos_token_id(self._merak_tokenizer, self._merak_runtime_meta, self._merak_runtime_meta_path)
        if eos_token_id is not None:
            generation_kwargs["eos_token_id"] = eos_token_id

        bidirectional_mask = build_vision_bidirectional_mask(mm_token_type_ids)
        if bidirectional_mask.shape[-1] != input_ids.shape[-1]:
            raise ValueError("Vision bidirectional mask shape does not match input_ids length.")

        with torch.inference_mode(), LLMInferenceContextManager(self._merak_model):
            generated_ids = self._merak_model.generate(**generation_kwargs)

        output_ids = generated_ids[0][input_ids.shape[-1]:]
        output_text = self._merak_tokenizer.decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text

    def _render_chat_prompt(self, rendered_messages: Sequence[Dict[str, str]]) -> str:
        kwargs = {"tokenize": False, "add_generation_prompt": True}
        if not self.disable_thinking:
            kwargs["enable_thinking"] = True
        else:
            kwargs["enable_thinking"] = False
        try:
            return self._merak_tokenizer.apply_chat_template(list(rendered_messages), **kwargs)
        except TypeError:
            kwargs.pop("enable_thinking", None)
            return self._merak_tokenizer.apply_chat_template(list(rendered_messages), **kwargs)

    @staticmethod
    def _expand_image_placeholders(
        input_ids: torch.Tensor,
        image_token_id: int,
        image_token_counts: Sequence[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected input_ids shape (1, seq), but got {tuple(input_ids.shape)}")

        expanded_ids: list[int] = []
        mm_token_type_ids: list[int] = []
        placeholder_index = 0
        for token_id in input_ids[0].tolist():
            if token_id == image_token_id:
                if placeholder_index >= len(image_token_counts):
                    raise ValueError("Prompt contains more image placeholders than provided images")
                token_count = int(image_token_counts[placeholder_index])
                expanded_ids.extend([image_token_id] * token_count)
                mm_token_type_ids.extend([1] * token_count)
                placeholder_index += 1
            else:
                expanded_ids.append(token_id)
                mm_token_type_ids.append(0)
        if placeholder_index != len(image_token_counts):
            raise ValueError(
                f"Prompt image placeholder count mismatch: prompt={placeholder_index}, images={len(image_token_counts)}"
            )
        return (
            torch.tensor([expanded_ids], dtype=torch.long),
            torch.tensor([mm_token_type_ids], dtype=torch.long),
        )

    def _generate_qwen3next(self, messages: Sequence[Dict[str, Any]], max_tokens: int) -> str:
        """Generate using Qwen3NextONNXModel's built-in generate."""
        inputs = apply_chat_template(self.tokenizer, messages, disable_thinking=self.disable_thinking)
        input_ids = inputs["input_ids"].to(self.device)
        raw_output = self.model.generate(
            input_ids=input_ids,
            tokenizer=self.tokenizer,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.0,
            stream_output=False,
        )
        return raw_output

    def _generate_llm_with_mask(self, messages: Sequence[Dict[str, Any]], max_tokens: int) -> str:
        """Generate using LLMWithMaskONNXModel manual prefill/decode loop."""
        from xh_model_zoo.api import decode_next_token

        if max_tokens <= 0:
            return ""

        # Reset KV cache
        num_layers = self.meta_info["num_hidden_layers"]
        for index in range(num_layers):
            k_cache = getattr(self.model, f"past_k_cache_{index}", None)
            v_cache = getattr(self.model, f"past_v_cache_{index}", None)
            if k_cache is not None:
                k_cache.reset()
            if v_cache is not None:
                v_cache.reset()

        # Reset sessions
        for session in [self.model.prefill_session, self.model.decode_session]:
            if session is not None:
                if hasattr(session, "step"):
                    session.step = 0
                if hasattr(session, "save_golden"):
                    session.save_golden = False

        self.model.to(self.device)
        self.model.set_exec_device(self.exec_device)
        self.model.to(self.dtype)

        inputs = apply_chat_template(self.processor, messages, disable_thinking=self.disable_thinking)
        inputs.pop("_prompt", None)
        input_ids = inputs["input_ids"].to(self.device)
        input_ids = self._truncate_input_ids_for_context(input_ids, max_tokens)
        input_length = input_ids.shape[-1]

        self.model.init_prefill()
        try:
            with torch.inference_mode():
                prefill_logits = self.model.prefill({"input_ids": input_ids, "past_seq_length": 0})
                next_token_id, next_token_text = decode_next_token(self.tokenizer, prefill_logits)
        except (AssertionError, RuntimeError) as exc:
            self.model.release_prefill_session()
            message = str(exc)
            if "shape mismatch" in message or "must match the size of tensor" in message:
                raise RuntimeError(
                    "HMONNX prefill 形状不匹配，当前导出物与 llm_with_mask 运行包装不兼容。"
                    f" input_length={input_length}, export_meta={self.meta_path}。原始错误: {message}"
                ) from exc
            raise
        self.model.release_prefill_session()

        self.model.init_decode()

        output_chunks = list(next_token_text)
        eos_token_id = self.tokenizer.eos_token_id
        past_seq_length = input_length
        data_batch = {"input_ids": next_token_id.to(self.device), "past_seq_length": past_seq_length}

        decode_steps = 1
        while decode_steps < max_tokens:
            if self.max_context_tokens is not None and past_seq_length >= self.max_context_tokens:
                logger.warning(
                    "HMONNX decode reached exported context limit %s after %s generated tokens; "
                    "stopping early for export %s",
                    self.max_context_tokens,
                    decode_steps,
                    self.meta_path,
                )
                break
            with torch.inference_mode():
                decode_logits = self.model.decode(data_batch)
                next_token_id, next_token_text = decode_next_token(self.tokenizer, decode_logits)

            token_id = next_token_id[0][0].item()
            if eos_token_id is not None and token_id == eos_token_id:
                break

            output_chunks.extend(next_token_text)
            past_seq_length += 1
            decode_steps += 1
            data_batch = {"input_ids": next_token_id.to(self.device), "past_seq_length": past_seq_length}

        self.model.release_decode_session()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "".join(output_chunks)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_backend(
    backend_type: str,
    model_config: Any,
) -> EvalBackend:
    """Create an evaluation backend from a ModelConfig."""
    from .model_registry import ModelConfig, BackendConfig

    backend_cfg: BackendConfig = model_config.backends.get(backend_type)
    if backend_cfg is None:
        raise ValueError(f"Backend '{backend_type}' not configured for {model_config.display_name}")

    if backend_type == "float":
        return FloatBackend(
            hf_model_dir=model_config.hf_model_dir,
            model_class=model_config.model_class,
            processor_class=model_config.processor_class,
            dtype=backend_cfg.dtype,
            device_map=backend_cfg.device_map,
            experts_implementation=backend_cfg.experts_implementation,
            disable_thinking=model_config.disable_thinking,
        )
    elif backend_type == "hmonnx":
        return HMONNXBackend(
            export_meta_info_path=backend_cfg.export_meta_info,
            onnx_model_type=backend_cfg.onnx_model_type,
            auto_offload=backend_cfg.auto_offload,
            resource_tight_mode=backend_cfg.resource_tight_mode,
            disable_thinking=model_config.disable_thinking,
            vision_export_meta_info_path=backend_cfg.vision_export_meta_info,
        )
    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
