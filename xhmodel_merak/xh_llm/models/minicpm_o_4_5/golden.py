from __future__ import annotations

import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer, DynamicCache

from .hf_compatible import (
    ensure_tts_sampling_config,
    patch_audio_attention_return_compat,
    patch_dynamic_cache_seen_tokens,
)
from .runtime import MiniCPMO45HMONNXRuntime
from .streaming_fixtures import (
    get_streaming_case,
    run_streaming_case,
    streaming_case_result_manifest,
)


def _load_real_runtime(root: Path, meta: Mapping[str, Any]) -> tuple[MiniCPMO45HMONNXRuntime, Any]:
    model_dir = str(meta["hf_model"])
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    host = AutoModel.from_pretrained(
        model_dir,
        config=ensure_tts_sampling_config(config),
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    ).eval()
    if hasattr(host, "init_tts"):
        host.init_tts(model_dir=str(Path(model_dir) / "assets" / "token2wav"))
    patch_dynamic_cache_seen_tokens(DynamicCache)
    patch_audio_attention_return_compat(host)
    return MiniCPMO45HMONNXRuntime(root, host), tokenizer


def _runtime_sessions(runtime: MiniCPMO45HMONNXRuntime) -> dict[str, Any]:
    sessions: dict[str, Any] = {}
    session_paths = {
        "vision": ("vision",),
        "audio": ("audio", "session"),
        "llm_prefill": ("llm", "prefill_model"),
        "llm_decode": ("llm", "decode_model"),
        "tts_prefill": ("tts", "prefill_model"),
        "tts_decode": ("tts", "decode_model"),
    }
    for name in (
        "stream_prefill_session",
        "stream_decode_session",
        "session_prefill_session",
        "session_decode_session",
    ):
        session_paths[f"audio_{name.removesuffix('_session')}"] = ("audio", name)
    for name in ("projector_semantic_session", "head_code_session"):
        session_paths[f"tts_{name.removesuffix('_session')}"] = ("tts", name)
    for name in (
        "frontend_session",
        "decoder_session",
        "hift_session",
        "flow_frontend_session",
        "flow_frontend_final_session",
        "estimator_step_session",
        "hift_stream_session",
        "hift_stream_final_session",
        "campplus_session",
        "speech_tokenizer_session",
    ):
        session_paths[f"token2wav_{name.removesuffix('_session')}"] = ("token2wav", name)
    for name, path in session_paths.items():
        owner = runtime
        for attribute in path:
            owner = getattr(owner, attribute, None)
            if owner is None:
                break
        if owner is not None:
            sessions[name] = owner
    return sessions


def _enable_runtime_golden(runtime: MiniCPMO45HMONNXRuntime, root: Path) -> dict[str, Path]:
    golden_root = root / "real_golden"
    if golden_root.exists():
        shutil.rmtree(golden_root)
    directories: dict[str, Path] = {}
    for name, session in _runtime_sessions(runtime).items():
        if session is None:
            continue
        directory = golden_root / name
        if hasattr(session, "hmonnx_session"):
            # Keep the wrapper's state in sync: its setter also resets the
            # backend step counter.  Golden files themselves belong to the
            # concrete backend session below.
            session.enable_golden = True
        backend_session = getattr(session, "hmonnx_session", session)
        if not hasattr(backend_session, "save_golden"):
            raise RuntimeError(f"{name} backend does not support golden export")
        backend_session.save_golden = True
        backend_session.golden_dir = str(directory)
        reset_step = getattr(backend_session, "reset_step", None)
        if callable(reset_step):
            reset_step()
        else:
            backend_session.step = 0
        directories[name] = directory
    return directories


def dump_real_golden(
    work_dir: Path,
    meta: Mapping[str, Any],
    device: str,
    request: Mapping[str, Any],
) -> dict[str, Any]:
    streaming_case_name = request.get("streaming_case")
    if isinstance(streaming_case_name, str):
        case = get_streaming_case(streaming_case_name)
        runtime, tokenizer = _load_real_runtime(work_dir, meta)
        runtime.set_exec_device(device)
        directories = _enable_runtime_golden(runtime, work_dir)
        runtime.reset_state()
        processor = getattr(getattr(runtime, "host_model", None), "processor", None)
        media = request.get("media")
        try:
            result = run_streaming_case(runtime, tokenizer, processor, case, media)
        finally:
            runtime.release()
        components = {
            name: {
                "directory": str(directory.relative_to(work_dir)),
                "files": [str(path.relative_to(work_dir)) for path in sorted(directory.rglob("*.npy"))],
            }
            for name, directory in directories.items()
            if directory.is_dir() and any(directory.rglob("*.npy"))
        }
        return {
            "mode": "real",
            "streaming_case": result.case_name,
            "final": result.final,
            "api_events": list(result.api_events),
            "text_chunks": list(result.text_chunks),
            "token_ids": list(result.token_ids),
            "waveform_chunks": list(result.waveform_chunks),
            "backend_counters": dict(result.backend_counters),
            "components": components,
            "streaming_manifest": streaming_case_result_manifest(meta, case),
        }
    messages = request.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("real MiniCPM-o-4.5 Golden requires non-empty messages")
    runtime, tokenizer = _load_real_runtime(work_dir, meta)
    runtime.set_exec_device(device)
    directories = _enable_runtime_golden(runtime, work_dir)
    runtime.reset_state()
    chat_kwargs = dict(request.get("chat_kwargs", {}))
    chat_kwargs.update(
        {
            "msgs": messages,
            "tokenizer": tokenizer,
            "sampling": False,
            "do_sample": False,
        }
    )
    try:
        text = runtime.chat(**chat_kwargs)
    finally:
        runtime.release()
    components = {
        name: {
            "directory": str(directory.relative_to(work_dir)),
            "files": [str(path.relative_to(work_dir)) for path in sorted(directory.rglob("*.npy"))],
        }
        for name, directory in directories.items()
        if directory.is_dir() and any(directory.rglob("*.npy"))
    }
    return {"mode": "real", "text": text if isinstance(text, str) else str(text), "components": components}


__all__ = ["dump_real_golden"]
