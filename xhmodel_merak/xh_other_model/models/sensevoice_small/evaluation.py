# Copyright 2025 HOUMO AI
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from . import runtime


def load_samples(
    *,
    manifest_jsonl: str = "",
    wav_scp: str = "",
    text: str = "",
    hf_dataset: str = "",
    hf_config: str = "",
    hf_split: str = "test",
    hf_streaming: bool = False,
    hf_audio_field: str = "audio",
    hf_text_field: str = "",
    limit: int = 0,
) -> list[runtime.Sample]:
    selected = sum(bool(value) for value in (manifest_jsonl, wav_scp, hf_dataset))
    if selected != 1:
        raise ValueError("Choose exactly one data source: manifest_jsonl, wav_scp, or hf_dataset")
    if manifest_jsonl:
        samples = runtime.read_manifest_jsonl(Path(manifest_jsonl).expanduser().resolve())
    elif wav_scp:
        if not text:
            raise ValueError("text is required when wav_scp is used")
        samples = runtime.read_wav_scp_text(
            Path(wav_scp).expanduser().resolve(),
            Path(text).expanduser().resolve(),
        )
    else:
        samples = runtime.load_hf_dataset(
            dataset=hf_dataset,
            config=hf_config,
            split=hf_split,
            limit=int(limit),
            streaming=bool(hf_streaming),
            audio_field=hf_audio_field,
            text_field=hf_text_field,
        )
    return samples[:limit] if limit > 0 else samples


def evaluate_export(
    *,
    export_dir: str,
    backend: str,
    report_path: str,
    device: str = "cuda:0",
    assets_dir: str = "",
    tokens: str = "",
    manifest_jsonl: str = "",
    wav_scp: str = "",
    text: str = "",
    hf_dataset: str = "",
    hf_config: str = "",
    hf_split: str = "test",
    hf_streaming: bool = False,
    hf_audio_field: str = "audio",
    hf_text_field: str = "",
    hf_text_path: str = "",
    limit: int = 0,
    strip_tags: bool = True,
    fast: bool = False,
) -> dict[str, Any]:
    if backend not in {"onnx", "hmonnx"}:
        raise ValueError(f"Unsupported SenseVoice evaluation backend: {backend!r}")
    artifacts = runtime.resolve_export_artifacts(export_dir)
    model_file_key = "onnx_file" if backend == "onnx" else "hmonnx_file"
    if model_file_key not in artifacts:
        raise FileNotFoundError(f"The export metadata does not contain {model_file_key}")
    model_file = Path(artifacts[model_file_key])
    runtime_assets = Path(assets_dir).expanduser().resolve() if assets_dir else Path(artifacts["assets_dir"])
    frontend = runtime.build_frontend(runtime_assets)
    token_file = runtime.find_token_file(runtime_assets, tokens or None)
    token_list = runtime.load_tokens(token_file)

    samples = load_samples(
        manifest_jsonl=manifest_jsonl,
        wav_scp=wav_scp,
        text=text,
        hf_dataset=hf_dataset,
        hf_config=hf_config,
        hf_split=hf_split,
        hf_streaming=hf_streaming,
        hf_audio_field=hf_audio_field,
        hf_text_field=hf_text_field,
        limit=limit,
    )
    reference_map = _read_reference_map(hf_text_path)

    total_cer_distance = 0
    total_cer_length = 0
    total_wer_distance = 0
    total_wer_length = 0
    per_sample: list[dict[str, Any]] = []
    start_time = time.time()
    for index, sample in enumerate(samples, start=1):
        reference = sample.text
        if reference_map and sample.audio_id:
            reference = reference_map.get(sample.audio_id.split("/")[-1], reference)
        if not reference.strip():
            continue

        waveform = runtime.load_audio_any(sample, target_sr=int(frontend.cfg.fs))
        feat, feat_len = runtime.extract_features(frontend, waveform)
        inputs = runtime.make_inputs_for_sample(feat, feat_len, sample.language, sample.textnorm)
        if backend == "onnx":
            logits, output_lengths = runtime.run_onnx(model_file, inputs)
        else:
            logits, output_lengths = runtime.run_hmonnx(model_file, inputs, device=device, fast=fast)
        output_length = int(output_lengths[0]) if hasattr(output_lengths, "__len__") else int(output_lengths)
        token_ids = runtime.ctc_greedy_decode(logits[0], output_length)
        hypothesis = runtime.decode_token_ids(token_ids, token_list)

        normalized_reference = runtime.strip_rich_tags(reference) if strip_tags else reference
        normalized_hypothesis = runtime.strip_rich_tags(hypothesis) if strip_tags else hypothesis
        normalized_reference = normalized_reference.lower()
        normalized_hypothesis = normalized_hypothesis.lower()
        reference_chars = list(normalized_reference.replace(" ", ""))
        hypothesis_chars = list(normalized_hypothesis.replace(" ", ""))
        reference_words = [word for word in normalized_reference.lower().split() if word]
        hypothesis_words = [word for word in normalized_hypothesis.lower().split() if word]
        cer_distance = runtime.edit_distance(reference_chars, hypothesis_chars)
        wer_distance = runtime.edit_distance(reference_words, hypothesis_words)
        total_cer_distance += cer_distance
        total_cer_length += len(reference_chars)
        total_wer_distance += wer_distance
        total_wer_length += len(reference_words)
        per_sample.append(
            {
                "audio": sample.audio_id or str(sample.audio),
                "ref": reference,
                "hyp": hypothesis,
                "token_ids": token_ids,
                "cer": cer_distance / max(1, len(reference_chars)),
                "wer": wer_distance / max(1, len(reference_words)),
                "feat_len": int(feat_len),
                "out_len": int(output_length),
                "language": sample.language,
                "textnorm": sample.textnorm,
            }
        )
        if index % 10 == 0:
            print(f"Processed {index}/{len(samples)} samples", flush=True)

    summary = {
        "num_samples": len(per_sample),
        "cer_avg": total_cer_distance / max(1, total_cer_length),
        "wer_avg": total_wer_distance / max(1, total_wer_length),
        "elapsed_seconds": time.time() - start_time,
    }
    report = {
        "summary": summary,
        "per_sample": per_sample,
        "backend": backend,
        "model_file": str(model_file),
        "export_dir": str(artifacts["work_dir"]),
        "assets_dir": str(runtime_assets),
        "tokens": str(token_file),
        "hf_dataset": hf_dataset,
        "hf_config": hf_config,
        "hf_split": hf_split,
        "hf_text_path": hf_text_path,
        "device": device if backend == "hmonnx" else None,
    }
    output_file = Path(report_path).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if hf_streaming:
        runtime.close_hf_streaming()
    return report


def _read_reference_map(path: str) -> dict[str, str]:
    if not path:
        return {}
    references: dict[str, str] = {}
    with Path(path).expanduser().resolve().open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                key, value = line.split(maxsplit=1)
                references[key] = value
    return references


__all__ = ["evaluate_export", "load_samples"]
