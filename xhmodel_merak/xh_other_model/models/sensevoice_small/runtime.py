# Copyright 2025 HOUMO AI
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .frontend import SenseVoiceFrontend


LANGUAGE_MAP: dict[str, int] = {
    "auto": 0,
    "zh": 3,
    "en": 4,
    "yue": 7,
    "ja": 11,
    "ko": 12,
    "nospeech": 13,
}
TEXTNORM_MAP: dict[str, int] = {"withitn": 14, "woitn": 15}

_ONNX_SESSION_CACHE: dict[str, Any] = {}
_HMONNX_SESSION_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class Sample:
    audio: Any
    text: str
    language: str = "auto"
    textnorm: str = "woitn"
    audio_id: str = ""


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dtype: Any
    shape: tuple[int, ...]


def resolve_tag(value: str, mapping: dict[str, int]) -> int:
    if value.isdigit():
        return int(value)
    key = value.lower().strip()
    if key not in mapping:
        raise ValueError(f"Unsupported value {value!r}; supported values: {sorted(mapping)}")
    return int(mapping[key])


def read_manifest_jsonl(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            audio = item.get("audio") or item.get("source")
            if not audio:
                raise ValueError(f"Manifest line is missing audio/source: {item}")
            samples.append(
                Sample(
                    audio=str(audio),
                    text=str(item.get("text") or item.get("target") or ""),
                    language=str(item.get("language") or "auto"),
                    textnorm=str(item.get("textnorm") or item.get("textnorm_type") or "woitn"),
                    audio_id=str(item.get("id") or audio),
                )
            )
    return samples


def read_wav_scp_text(wav_scp: Path, text: Path) -> list[Sample]:
    wav_map: dict[str, str] = {}
    with wav_scp.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split(maxsplit=1)
                wav_map[key] = value

    text_map: dict[str, str] = {}
    with text.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split(maxsplit=1)
                text_map[key] = value

    keys = sorted(set(wav_map) & set(text_map))
    return [Sample(audio=wav_map[key], text=text_map[key], audio_id=key) for key in keys]


def load_hf_dataset(
    dataset: str,
    config: str,
    split: str,
    limit: int,
    streaming: bool = False,
    audio_field: str = "audio",
    text_field: str = "",
) -> list[Sample]:
    """Load HF rows without asking datasets/torchcodec to decode audio.

    ``Audio(decode=False)`` keeps encoded bytes or a local path in each row.  The
    runtime then decodes with libsndfile/librosa, so FFmpeg and torchcodec are not
    hard dependencies of calibration or evaluation.
    """

    from datasets import Audio, load_dataset

    kwargs: dict[str, Any] = {"path": dataset, "split": split, "streaming": bool(streaming)}
    if config:
        kwargs["name"] = config
    rows = load_dataset(**kwargs)
    try:
        rows = rows.cast_column(audio_field, Audio(decode=False))
    except (KeyError, TypeError, ValueError):
        # Older datasets versions may already expose undecoded dictionaries.
        pass

    samples: list[Sample] = []
    for index, row in enumerate(rows):
        if limit > 0 and len(samples) >= limit:
            break
        if audio_field not in row:
            continue
        if text_field:
            text = row.get(text_field, "")
        else:
            text = row.get("text") or row.get("sentence") or row.get("transcript") or row.get("transcription") or ""
        audio_id = str(row.get("id") or row.get("file") or row.get("__key__") or f"{dataset}:{split}:{index}")
        samples.append(Sample(audio=row[audio_field], text=str(text), audio_id=audio_id))
    return samples


def _resample(waveform: Any, sample_rate: int, target_sr: int) -> Any:
    import numpy as np

    wav = np.asarray(waveform, dtype=np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1 if wav.shape[1] <= wav.shape[0] else 0)
    if sample_rate != target_sr:
        import librosa

        wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=target_sr)
    return np.asarray(wav, dtype=np.float32)


def load_audio_any(sample: Sample, target_sr: int) -> Any:
    audio = sample.audio
    if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        return _resample(audio["array"], int(audio["sampling_rate"]), target_sr)

    if isinstance(audio, dict) and audio.get("bytes") is not None:
        import soundfile as sf

        waveform, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=False)
        return _resample(waveform, int(sample_rate), target_sr)

    if isinstance(audio, dict) and audio.get("path"):
        audio_path = str(audio["path"])
    else:
        audio_path = str(audio)

    import librosa

    waveform, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    return waveform


def build_frontend(model_dir: Path) -> SenseVoiceFrontend:
    return SenseVoiceFrontend.from_model_dir(model_dir)


def extract_features(frontend: SenseVoiceFrontend, waveform: Any) -> tuple[Any, int]:
    return frontend.extract(waveform)


def edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    hypothesis_length = len(hypothesis)
    row = list(range(hypothesis_length + 1))
    for ref_index in range(1, len(reference) + 1):
        diagonal = row[0]
        row[0] = ref_index
        for hyp_index in range(1, hypothesis_length + 1):
            previous = row[hyp_index]
            cost = 0 if reference[ref_index - 1] == hypothesis[hyp_index - 1] else 1
            row[hyp_index] = min(row[hyp_index] + 1, row[hyp_index - 1] + 1, diagonal + cost)
            diagonal = previous
    return row[hypothesis_length]


def load_tokens(tokens_path: Path) -> list[str] | None:
    if not tokens_path.is_file():
        return None
    if tokens_path.suffix == ".json":
        value = json.loads(tokens_path.read_text(encoding="utf-8"))
        return [str(item) for item in value] if isinstance(value, list) else None

    import sentencepiece as spm

    processor = spm.SentencePieceProcessor(model_file=str(tokens_path))
    return [processor.id_to_piece(index) for index in range(processor.vocab_size())]


def find_token_file(assets_dir: Path, tokens_path: str | Path | None = None) -> Path:
    if tokens_path:
        path = Path(tokens_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Token file not found: {path}")
        return path
    json_path = assets_dir / "tokens.json"
    if json_path.is_file():
        return json_path
    sentencepiece_files = sorted(assets_dir.glob("*.model"))
    if sentencepiece_files:
        return sentencepiece_files[0]
    raise FileNotFoundError(f"No tokens.json or SentencePiece .model found under {assets_dir}")


def decode_token_ids(token_ids: list[int], token_list: list[str] | None) -> str:
    if not token_list:
        return " ".join(str(item) for item in token_ids)
    tokens = [token_list[index] if 0 <= index < len(token_list) else "" for index in token_ids]
    text = "".join(tokens).replace("▁", " ").strip()
    return re.sub(r"\s+", " ", text)


def ctc_greedy_decode(logits: Any, output_length: int, blank_id: int = 0) -> list[int]:
    import torch

    values = torch.as_tensor(logits)[:output_length]
    values = torch.unique_consecutive(values.argmax(dim=-1), dim=-1)
    values = values[values != blank_id]
    return [int(value) for value in values.cpu().tolist()]


def strip_rich_tags(text: str) -> str:
    return re.sub(r"<\|.*?\|>", "", text)


def _onnx_providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [provider for provider in preferred if provider in available]
    return providers or list(available)


def _pad_or_truncate_speech(tensor: Any, target_length: int) -> Any:
    import torch
    import torch.nn.functional as functional

    value = torch.as_tensor(tensor)
    current_length = value.shape[1]
    if current_length < target_length:
        return functional.pad(value, (0, 0, 0, target_length - current_length))
    if current_length > target_length:
        return value[:, :target_length, :]
    return value


def run_onnx(onnx_path: Path, inputs: dict[str, Any]) -> tuple[Any, Any]:
    import numpy as np
    import onnxruntime as ort

    cache_key = str(onnx_path.resolve())
    session = _ONNX_SESSION_CACHE.get(cache_key)
    if session is None:
        session = ort.InferenceSession(str(onnx_path), providers=_onnx_providers())
        _ONNX_SESSION_CACHE[cache_key] = session

    input_info = {item.name: item for item in session.get_inputs()}
    feed: dict[str, Any] = {}
    for name, value in inputs.items():
        array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        info = input_info.get(name)
        if name == "speech" and info is not None and len(info.shape) == 3 and isinstance(info.shape[1], int):
            array = _pad_or_truncate_speech(array, int(info.shape[1])).numpy()
        if name == "speech_lengths" and info is not None:
            speech_info = input_info.get("speech")
            if speech_info is not None and len(speech_info.shape) == 3 and isinstance(speech_info.shape[1], int):
                array = np.minimum(array, int(speech_info.shape[1]))
        feed[name] = np.asarray(array)
    outputs = session.run(None, feed)
    return outputs[0], outputs[1]


def align_hmonnx_inputs(input_info: Sequence[Any], inputs: dict[str, Any], device: str) -> list[Any]:
    import torch

    target_device = torch.device(device)
    aligned: list[Any] = []
    for info in input_info:
        if info.name not in inputs:
            raise KeyError(f"Missing HMONNX input {info.name!r}; available inputs: {sorted(inputs)}")
        tensor = torch.as_tensor(inputs[info.name])
        if tensor.dtype != info.dtype:
            tensor = tensor.to(dtype=info.dtype)
        if info.name == "speech" and len(info.shape) == 3 and tensor.ndim == 3:
            tensor = _pad_or_truncate_speech(tensor, int(info.shape[1]))
        if info.name == "speech_lengths":
            speech_info = next((item for item in input_info if item.name == "speech"), None)
            if speech_info is not None and len(speech_info.shape) == 3:
                tensor = torch.clamp(tensor, max=int(speech_info.shape[1]))
        aligned.append(tensor.to(target_device))
    return aligned


def read_hmonnx_input_info(hmonnx_path: Path) -> list[TensorInfo]:
    import onnx
    import torch

    dtype_map = {
        onnx.TensorProto.FLOAT: torch.float32,
        onnx.TensorProto.INT32: torch.int32,
        onnx.TensorProto.INT64: torch.int64,
        onnx.TensorProto.BOOL: torch.bool,
        onnx.TensorProto.FLOAT16: torch.float16,
        onnx.TensorProto.BFLOAT16: torch.bfloat16,
    }
    model = onnx.load(str(hmonnx_path), load_external_data=False)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    result: list[TensorInfo] = []
    for value in model.graph.input:
        if value.name in initializer_names:
            continue
        tensor_type = value.type.tensor_type
        if tensor_type.elem_type not in dtype_map:
            raise TypeError(f"Unsupported HMONNX input dtype {tensor_type.elem_type} for {value.name!r}")
        shape = []
        for dim in tensor_type.shape.dim:
            if not dim.HasField("dim_value"):
                raise ValueError(f"SenseVoice HMONNX input {value.name!r} must have a static shape")
            shape.append(int(dim.dim_value))
        result.append(TensorInfo(name=value.name, dtype=dtype_map[tensor_type.elem_type], shape=tuple(shape)))
    return result


def run_hmonnx(
    hmonnx_path: Path,
    inputs: dict[str, Any],
    device: str,
    fast: bool = False,
) -> tuple[Any, Any]:
    import torch

    from xhquant.api import HMONNXInference

    cache_key = str(hmonnx_path.resolve())
    session = _HMONNX_SESSION_CACHE.get(cache_key)
    target_device = torch.device(device)
    if session is None:
        session = HMONNXInference(str(hmonnx_path))
        session.to(target_device)
        _HMONNX_SESSION_CACHE[cache_key] = session
    elif getattr(session, "device", target_device) != target_device:
        session.to(target_device)
    if fast:
        session.to_fast_mode()

    aligned = align_hmonnx_inputs(session.inputs, inputs, str(target_device))
    feed = {info.name: value for info, value in zip(session.inputs, aligned)}
    outputs = session.run(feed)
    return outputs[0], outputs[1]


def make_inputs_for_sample(
    feat: Any,
    feat_len: int,
    language: str,
    textnorm: str,
) -> dict[str, Any]:
    import numpy as np

    return {
        "speech": feat[None, :, :].astype("float32"),
        "speech_lengths": np.asarray([feat_len], dtype="int32"),
        "language": np.asarray([resolve_tag(language, LANGUAGE_MAP)], dtype="int32"),
        "textnorm": np.asarray([resolve_tag(textnorm, TEXTNORM_MAP)], dtype="int32"),
    }


def load_export_meta(export_dir: str | Path) -> tuple[Path, dict[str, Any]]:
    work_dir = Path(export_dir).expanduser().resolve()
    meta_file = work_dir / "export_meta_info.json"
    if not meta_file.is_file():
        raise FileNotFoundError(f"SenseVoice export metadata not found: {meta_file}")
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    if not isinstance(meta, dict) or "sensevoice" not in meta:
        raise ValueError(f"Invalid SenseVoice export metadata: {meta_file}")
    return work_dir, meta


def resolve_export_artifacts(export_dir: str | Path) -> dict[str, Any]:
    work_dir, meta = load_export_meta(export_dir)
    component = meta["sensevoice"]
    result: dict[str, Any] = {"work_dir": work_dir, "meta": meta}
    for key in ("onnx_file", "hmonnx_file"):
        if component.get(key):
            result[key] = work_dir / component[key]
    assets_value = meta.get("assets_dir")
    result["assets_dir"] = work_dir / assets_value if assets_value else Path(meta["source_model_dir"])
    return result


def close_hf_streaming() -> None:
    try:
        import fsspec.asyn

        fsspec.asyn.close()
    except Exception:
        pass


__all__ = [
    "LANGUAGE_MAP",
    "TEXTNORM_MAP",
    "Sample",
    "TensorInfo",
    "align_hmonnx_inputs",
    "build_frontend",
    "close_hf_streaming",
    "ctc_greedy_decode",
    "decode_token_ids",
    "edit_distance",
    "extract_features",
    "find_token_file",
    "load_audio_any",
    "load_export_meta",
    "load_hf_dataset",
    "load_tokens",
    "make_inputs_for_sample",
    "read_manifest_jsonl",
    "read_hmonnx_input_info",
    "read_wav_scp_text",
    "resolve_export_artifacts",
    "run_hmonnx",
    "run_onnx",
    "strip_rich_tags",
]
