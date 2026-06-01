# Copyright 2026 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Generate real HMONNX golden for every exported Qwen3-Omni module.

This follows the same pattern as examples/llm/qwen3_next and benchmark_test/
*.py: load each exported HMONNX via ``HMONNXGoldenInference``, turn on
``session.save_golden`` and run one forward with sensible inputs sourced from
``session.get_input(name)`` metadata. The HMONNX runtime itself dumps the
golden tensors (input + output) into the configured ``golden_dir`` — no manual
tensor saving or HF-side validation is involved.

Supported modules (auto-detected from ``meta*.json`` under ``--work-dir``):

- audio_encoder
- vision_encoder
- text prefill / decode
- talker prefill / decode
- talker_prediction prefill / decode
- talker_projection_hm
- code2wav
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch

from xhquant.api import CacheTensor, get_root_logger, xhquant_init

try:
    from xhquant.xhonnxruntime import HMONNXGraphGoldenInference as HMONNXGoldenInference
except ImportError:  # pragma: no cover
    from xhquant.api import HMONNXGraphGoldenInference as HMONNXGoldenInference


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _is_cache_input_name(name: str) -> bool:
    return (
        name.startswith(
            (
                "past_key_cache_",
                "past_value_cache_",
                "past_conv_cache_",
                "past_recurrent_state_",
            )
        )
        or "kcache_input" in name
        or "vcache_input" in name
    )


def _ensure_cache_tensor(tensor: torch.Tensor) -> CacheTensor:
    if isinstance(tensor, CacheTensor):
        return tensor
    return CacheTensor(tensor)


def _concrete_shape(shape) -> List[int]:
    resolved: List[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            resolved.append(int(dim))
        else:
            # Dynamic / symbolic axes — default to 1.
            resolved.append(1)
    return resolved


def _random_tensor(info, device: torch.device) -> torch.Tensor:
    shape = _concrete_shape(info.shape)
    dtype = info.dtype
    if dtype in (torch.float16, torch.float32, torch.bfloat16):
        tensor = torch.randn(shape, dtype=torch.float32).to(dtype)
    else:
        tensor = torch.zeros(shape, dtype=dtype)
    return tensor.to(device)


def _build_input_feed(
    session: HMONNXGoldenInference,
    device: torch.device,
    overrides: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, torch.Tensor]:
    overrides = overrides or {}
    feed: Dict[str, torch.Tensor] = {}
    for name in session.get_input_names():
        if name in overrides:
            tensor = overrides[name]
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor)
            tensor = tensor.to(device)
        else:
            info = session.get_input(name)
            tensor = _random_tensor(info, device)
        if _is_cache_input_name(name):
            tensor = _ensure_cache_tensor(tensor)
        feed[name] = tensor
    return feed


def _run_session_with_golden(
    onnx_path: Path,
    golden_dir: Path,
    logger,
    device: torch.device,
    overrides: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    golden_dir.mkdir(exist_ok=True, parents=True)
    logger.info(f"[golden] loading {onnx_path}")
    session = HMONNXGoldenInference(str(onnx_path))
    session.exec_device = device
    session.to(device)
    session.save_golden = True
    session.golden_dir = str(golden_dir)
    session.initialize()
    if hasattr(session, "legacy_mode"):
        session.legacy_mode = False

    feed = _build_input_feed(session, device, overrides)
    input_shapes = {k: list(v.shape) for k, v in feed.items()}

    start = time.time()
    outputs = session.run(feed)
    elapsed = time.time() - start

    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    output_names = session.get_output_names()
    output_shapes = {
        name: list(out.shape) if hasattr(out, "shape") else None
        for name, out in zip(output_names, outputs)
    }
    logger.info(
        f"[golden] {onnx_path.name} done in {elapsed:.2f}s — golden_dir={golden_dir}"
    )

    del session
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "onnx": str(onnx_path),
        "golden_dir": str(golden_dir),
        "elapsed_sec": round(elapsed, 3),
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
    }


# ---------------------------------------------------------------------------
# module dispatchers — each resolves the onnx(s), golden_dir and any overrides
# ---------------------------------------------------------------------------


@dataclass
class ModuleSpec:
    name: str
    work_dir: Path
    meta_path: Path
    meta: Dict[str, Any]


def _load_modules(work_dir: Path) -> List[ModuleSpec]:
    specs: List[ModuleSpec] = []
    for subdir in sorted(p for p in work_dir.iterdir() if p.is_dir()):
        for meta_path in sorted(subdir.glob("meta*.json")):
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            name = meta.get("module") or subdir.name
            specs.append(ModuleSpec(name=name, work_dir=subdir, meta_path=meta_path, meta=meta))
    return specs


def _resolve_rel(module: ModuleSpec, key: str) -> Path:
    rel = module.meta[key]
    return (module.work_dir / rel).resolve()


def _golden_subdir(module: ModuleSpec, sub: str) -> Path:
    return module.work_dir / "golden" / sub


def _handle_audio(module: ModuleSpec, logger, device) -> List[Dict[str, Any]]:
    onnx_path = _resolve_rel(module, "audio_encoder_onnx")
    golden_dir = _golden_subdir(module, "audio_encoder")
    return [_run_session_with_golden(onnx_path, golden_dir, logger, device)]


def _handle_vision(module: ModuleSpec, logger, device) -> List[Dict[str, Any]]:
    onnx_path = _resolve_rel(module, "vision_encoder_onnx")
    golden_dir = _golden_subdir(module, "vision_encoder")
    return [_run_session_with_golden(onnx_path, golden_dir, logger, device)]


def _handle_code2wav(module: ModuleSpec, logger, device) -> List[Dict[str, Any]]:
    onnx_path = _resolve_rel(module, "code2wav_hmonnx")
    golden_dir = _golden_subdir(module, "code2wav")
    return [_run_session_with_golden(onnx_path, golden_dir, logger, device)]


def _handle_text(module: ModuleSpec, logger, device) -> List[Dict[str, Any]]:
    results = []
    for key, sub in (("prefill_onnx", "prefill"), ("decode_onnx", "decode")):
        if key not in module.meta:
            continue
        onnx_path = _resolve_rel(module, key)
        golden_dir = _golden_subdir(module, sub)
        results.append(_run_session_with_golden(onnx_path, golden_dir, logger, device))
    return results


def _handle_talker(module: ModuleSpec, logger, device) -> List[Dict[str, Any]]:
    results = []
    for key, sub in (
        ("talker_prefill_onnx", "prefill"),
        ("talker_decode_onnx", "decode"),
    ):
        if key not in module.meta:
            continue
        onnx_path = _resolve_rel(module, key)
        golden_dir = _golden_subdir(module, sub)
        results.append(_run_session_with_golden(onnx_path, golden_dir, logger, device))
    return results


def _handle_talker_prediction(module: ModuleSpec, logger, device) -> List[Dict[str, Any]]:
    results = []
    for key, sub in (
        ("talker_prediction_prefill_onnx", "prefill"),
        ("talker_prediction_decode_onnx", "decode"),
    ):
        if key not in module.meta:
            continue
        onnx_path = _resolve_rel(module, key)
        golden_dir = _golden_subdir(module, sub)
        results.append(_run_session_with_golden(onnx_path, golden_dir, logger, device))
    return results


_DISPATCH = {
    "audio_encoder": _handle_audio,
    "vision_encoder": _handle_vision,
    "code2wav": _handle_code2wav,
    "talker_model": _handle_talker,
    "talker_prediction": _handle_talker_prediction,
}


def _handle_generic(module: ModuleSpec, logger, device) -> List[Dict[str, Any]]:
    # Text module has no "module" key in meta.json; fall through by inspecting keys.
    if "prefill_onnx" in module.meta and "decode_onnx" in module.meta:
        return _handle_text(module, logger, device)
    logger.warning(f"[golden] unrecognized module {module.meta_path}, skipping")
    return []


# ---------------------------------------------------------------------------
# entrypoint
# ---------------------------------------------------------------------------


def main(args):
    work_dir = Path(args.work_dir).resolve()
    if not work_dir.is_dir():
        raise FileNotFoundError(f"work_dir not found: {work_dir}")

    log_file = work_dir / "hmonnx_golden.log"
    xhquant_init(str(log_file), debug=args.debug)
    logger = get_root_logger()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    logger.info(f"[golden] device={device} work_dir={work_dir}")

    modules = _load_modules(work_dir)
    if args.module:
        wanted = set(args.module)
        modules = [m for m in modules if m.name in wanted or m.work_dir.name in wanted]
        if not modules:
            raise RuntimeError(f"no modules matched --module {args.module}")

    report: Dict[str, Any] = {
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "work_dir": str(work_dir),
        "device": str(device),
        "modules": {},
    }

    for module in modules:
        handler = _DISPATCH.get(module.name, _handle_generic)
        logger.info(f"[golden] ==== {module.name} ({module.work_dir.name}) ====")
        try:
            entries = handler(module, logger, device)
            report["modules"][module.work_dir.name] = {
                "name": module.name,
                "status": "ok",
                "entries": entries,
            }
        except Exception as exc:  # pragma: no cover — keep going for other modules
            logger.exception(f"[golden] FAILED for {module.name}: {exc}")
            report["modules"][module.work_dir.name] = {
                "name": module.name,
                "status": "failed",
                "error": repr(exc),
            }

    report_path = work_dir / "hmonnx_golden_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=4)
    logger.info(f"[golden] report saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run HMONNXGoldenInference on every exported Qwen3-Omni module and dump golden tensors.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="work_dirs/qwen3omni_no_projection",
        help="root dir containing per-module sub directories with meta*.json",
    )
    parser.add_argument(
        "--module",
        type=str,
        nargs="*",
        default=None,
        help="optional subset of module names or dir names to run",
    )
    parser.add_argument("--cpu", action="store_true", help="force CPU execution")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    main(args)
