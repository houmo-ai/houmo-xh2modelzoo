"""GPTQModel entry point for the MiniCPM-o-4.5 Qwen3 LLM backbone.

``gptqmodel.models.definitions.minicpm_o_4_5.MiniCPMO45QModel`` knows how to
load the MiniCPM-o-4.5 checkpoint and return its nested ``Qwen3ForCausalLM``.
This adapter only resolves repository-owned calibration resources and translates
the workflow mapping into ``gptqmodel.recipes.minicpm_o_4_5``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .resource_path import resolve_repo_resource


def _as_bool(value: Any, *, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise TypeError(f"MiniCPM-o-4.5 {name} must be a boolean, got {value!r}")


def _load_calibration_samples(
    calibration_path: Path,
    *,
    text_key: str,
    nsamples: int,
) -> list[str]:
    samples: list[str] = []
    with calibration_path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError("MiniCPM-o-4.5 calibration JSONL rows must be objects")
            text = payload.get(text_key)
            if not isinstance(text, str):
                raise TypeError(f"MiniCPM-o-4.5 calibration field {text_key!r} must be a string")
            if text.strip():
                samples.append(text)
            if len(samples) >= int(nsamples):
                break
    if len(samples) < int(nsamples):
        raise ValueError(f"MiniCPM-o-4.5 calibration data has only {len(samples)} usable samples; need {int(nsamples)}")
    return samples[: int(nsamples)]


def quantize_minicpm_llm_gptq(
    *,
    model_dir: str,
    output_dir: str,
    quant_cfg: Mapping[str, Any],
    device: str,
) -> str:
    """Resolve MiniCPM-o-4.5 GPTQ inputs and call the stable recipe API."""

    if not isinstance(quant_cfg, Mapping):
        raise TypeError(f"quant_cfg must be a mapping, got {type(quant_cfg)!r}")

    cfg = dict(quant_cfg)
    bits = int(cfg.get("bits", 4))
    group_size = int(cfg.get("group_size", cfg.get("w_groupsize", 64)))
    nsamples = int(cfg.get("nsamples", 128))
    seqlen = int(cfg.get("seqlen", 1024))
    batch_size = int(cfg.get("batch_size", 1))
    sym = _as_bool(cfg.get("sym", True), name="sym")
    damp_percent = float(cfg.get("damp_percent", 0.01))
    seed = int(cfg.get("seed", 42))
    offload_to_disk = _as_bool(cfg.get("offload_to_disk", True), name="offload_to_disk")
    offload_to_disk_path = cfg.get("offload_to_disk_path")
    text_key = str(cfg.get("calibration_text_key", "text"))

    calibration_ref = cfg.get("calibration_jsonl")
    if not calibration_ref:
        raise ValueError("MiniCPM-o-4.5 GPTQ requires quant.calibration_jsonl")
    calibration_path = resolve_repo_resource(str(calibration_ref), description="LLM calibration JSONL")
    samples = _load_calibration_samples(
        calibration_path,
        text_key=text_key,
        nsamples=nsamples,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    save_path = output_path / "gptq_llm"

    from gptqmodel.recipes.minicpm_o_4_5 import quantize_minicpm_o_4_5

    result = quantize_minicpm_o_4_5(
        model_dir=model_dir,
        output_dir=str(save_path),
        bits=bits,
        group_size=group_size,
        sym=sym,
        batch_size=batch_size,
        nsamples=nsamples,
        seqlen=seqlen,
        calibration_data=samples,
        damp_percent=damp_percent,
        device=device,
        offload_to_disk=offload_to_disk,
        offload_to_disk_path=(None if offload_to_disk_path is None else str(offload_to_disk_path)),
        seed=seed,
    )
    return result.output_dir


__all__ = ["quantize_minicpm_llm_gptq"]
