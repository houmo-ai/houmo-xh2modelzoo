"""Core logic for product-level case analysis (qualitative evaluation).

Handles single-case inference with multimodal inputs (text, image, audio, PDF).
Stores results in a JSON-based case store for future badcase collection.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

logger = logging.getLogger(__name__)

_CASE_STORE_DIR = Path(__file__).resolve().parent.parent / "case_store"


@dataclass
class CaseRecord:
    case_id: str
    model_path: str
    backend_type: str
    gpu_id: str
    prompt_text: str
    input_files: list[str] = field(default_factory=list)
    response: str = ""
    elapsed_seconds: float = 0.0
    created_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    label: str = ""  # for future badcase tagging: good / bad / uncertain


def _ensure_case_store() -> Path:
    _CASE_STORE_DIR.mkdir(parents=True, exist_ok=True)
    return _CASE_STORE_DIR


def save_case(record: CaseRecord) -> Path:
    store_dir = _ensure_case_store()
    path = store_dir / f"{record.case_id}.json"
    path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def list_cases(limit: int = 100) -> list[CaseRecord]:
    store_dir = _ensure_case_store()
    cases = []
    for p in sorted(store_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        if len(cases) >= limit:
            break
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            cases.append(CaseRecord(**{k: v for k, v in data.items() if k in CaseRecord.__dataclass_fields__}))
        except Exception:
            continue
    return cases


def pdf_to_images(pdf_path: str, max_pages: int = 10) -> list[Any]:
    """Convert PDF pages to PIL Images using PyMuPDF."""
    import fitz
    from PIL import Image
    from io import BytesIO

    doc = fitz.open(pdf_path)
    images = []
    for page_idx in range(min(len(doc), max_pages)):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=150)
        img_data = pix.tobytes("png")
        img = Image.open(BytesIO(img_data)).convert("RGB")
        images.append(img)
    doc.close()
    return images


def build_messages_from_inputs(
    prompt_text: str,
    image_paths: list[str],
    audio_paths: list[str],
    pdf_paths: list[str],
    max_pdf_pages: int = 5,
) -> list[dict[str, Any]]:
    """Build chat messages with multimodal content parts."""
    content_parts: list[dict[str, Any]] = []

    for pdf_path in pdf_paths:
        try:
            images = pdf_to_images(pdf_path, max_pages=max_pdf_pages)
            for img in images:
                content_parts.append({"type": "image", "image": img})
        except Exception as e:
            logger.warning("Failed to load PDF %s: %s", pdf_path, e)
            content_parts.append({"type": "text", "text": f"[PDF加载失败: {pdf_path}]"})

    for img_path in image_paths:
        from PIL import Image
        try:
            img = Image.open(img_path).convert("RGB")
            content_parts.append({"type": "image", "image": img})
        except Exception as e:
            logger.warning("Failed to load image %s: %s", img_path, e)
            content_parts.append({"type": "text", "text": f"[图片加载失败: {img_path}]"})

    for audio_path in audio_paths:
        try:
            import soundfile as sf
            audio_data, sample_rate = sf.read(audio_path)
            content_parts.append({
                "type": "audio",
                "audio": audio_data,
                "sampling_rate": sample_rate,
            })
        except Exception as e:
            logger.warning("Failed to load audio %s: %s", audio_path, e)
            content_parts.append({"type": "text", "text": f"[音频加载失败: {audio_path}]"})

    if prompt_text.strip():
        content_parts.append({"type": "text", "text": prompt_text.strip()})

    if not content_parts:
        content_parts.append({"type": "text", "text": "请描述这些内容。"})

    messages = [{"role": "user", "content": content_parts}]
    return messages


def run_case_inference(
    backend_type: str,
    model_path: str,
    gpu_id: str,
    prompt_text: str,
    image_paths: list[str],
    audio_paths: list[str],
    pdf_paths: list[str],
    max_tokens: int = 2048,
    onnx_model_type: str = "",
    vision_meta_path: str = "",
    max_pdf_pages: int = 5,
) -> CaseRecord:
    """Run inference on a single case and return the result record."""
    from datetime import datetime

    case_id = f"case_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    messages = build_messages_from_inputs(
        prompt_text, image_paths, audio_paths, pdf_paths, max_pdf_pages=max_pdf_pages,
    )

    all_files = list(image_paths) + list(audio_paths) + list(pdf_paths)

    t0 = time.time()
    try:
        if backend_type == "float":
            response = _run_float_inference(model_path, messages, max_tokens)
        else:
            response = _run_hmonnx_inference(
                model_path, messages, max_tokens,
                onnx_model_type=onnx_model_type,
                vision_meta_path=vision_meta_path,
                gpu_id=gpu_id,
            )
    except Exception as e:
        logger.exception("Case inference failed")
        response = f"[推理失败] {type(e).__name__}: {e}"
    elapsed = time.time() - t0

    record = CaseRecord(
        case_id=case_id,
        model_path=model_path,
        backend_type=backend_type,
        gpu_id=gpu_id,
        prompt_text=prompt_text,
        input_files=all_files,
        response=response,
        elapsed_seconds=round(elapsed, 2),
        created_at=datetime.now().isoformat(),
    )
    save_case(record)
    return record


def _run_float_inference(model_path: str, messages: list[dict], max_tokens: int) -> str:
    from .backends import FloatBackend

    model_path_p = Path(model_path)
    model_class, processor_class = _detect_model_class(model_path_p)

    backend = FloatBackend(
        hf_model_dir=model_path,
        model_class=model_class,
        processor_class=processor_class,
        dtype="bfloat16",
        device_map="auto",
    )
    try:
        result = backend.generate(messages, max_tokens)
    finally:
        backend.cleanup()
        del backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _run_hmonnx_inference(
    meta_path: str,
    messages: list[dict],
    max_tokens: int,
    onnx_model_type: str = "",
    vision_meta_path: str = "",
    gpu_id: str = "0",
) -> str:
    from .backends import HMONNXBackend

    if not onnx_model_type:
        onnx_model_type = _detect_onnx_model_type(meta_path)

    backend = HMONNXBackend(
        export_meta_info_path=meta_path,
        onnx_model_type=onnx_model_type,
        device=f"cuda:0",
        exec_device=f"cuda:0",
        vision_export_meta_info_path=vision_meta_path,
    )
    try:
        result = backend.generate(messages, max_tokens)
    finally:
        backend.cleanup()
        del backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return result


def _detect_model_class(model_path: Path) -> tuple[str, str]:
    """Detect model class from config.json in model directory."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        return "", "AutoTokenizer"

    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return "", "AutoTokenizer"

    architectures = config.get("architectures", [])
    model_type = config.get("model_type", "")

    if "Gemma4ForConditionalGeneration" in architectures:
        return "Gemma4ForConditionalGeneration", "AutoProcessor"
    if "Qwen3_5ForConditionalGeneration" in architectures:
        return "Qwen3_5ForConditionalGeneration", "AutoProcessor"
    if "Qwen3_5MoeForConditionalGeneration" in architectures:
        return "Qwen3_5MoeForConditionalGeneration", "AutoProcessor"
    if model_type in ("gemma4", "gemma4_text"):
        return "Gemma4ForConditionalGeneration", "AutoProcessor"
    if model_type == "qwen3_5_moe":
        return "Qwen3_5MoeForConditionalGeneration", "AutoProcessor"
    if model_type == "qwen3_5":
        return "Qwen3_5ForConditionalGeneration", "AutoProcessor"

    return "", "AutoProcessor" if any("vision" in a.lower() or "conditional" in a.lower() for a in architectures) else "AutoTokenizer"


def _detect_onnx_model_type(meta_path: str) -> str:
    """Detect ONNX model type from meta info."""
    try:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
    except Exception:
        return "LLMWithMaskONNXModel"

    model_config = meta.get("model_config", {})
    model_type = model_config.get("model_type", "")

    if "qwen3" in model_type.lower() and "next" in model_type.lower():
        return "Qwen3NextONNXModel"
    if "qwen3_5" in model_type.lower():
        return "Qwen3NextONNXModel"

    return "LLMWithMaskONNXModel"
