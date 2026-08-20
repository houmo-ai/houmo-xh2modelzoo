from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Final, TypeAlias

import numpy as np

from examples_merak.llm.minicpm_o_4_5.media_utils import flatten_audio_chunks, video_chunks
from examples_merak.llm.minicpm_o_4_5.media_utils import write_wav as _media_write_wav
from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import (
    MiniCPMO45HMONNXRuntime,
    ensure_tts_sampling_config,
    patch_audio_attention_return_compat,
    patch_dynamic_cache_seen_tokens,
)
from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
    CERTIFIED_STREAMING_CASES,
    StreamingCase,
    StreamingCaseError,
    StreamingCaseResult,
    StreamingHost,
    get_streaming_case,
    run_streaming_case,
)


CASE_CHOICES: Final[tuple[str, ...]] = tuple(case.name for case in CERTIFIED_STREAMING_CASES)
DEFAULT_OUTPUT_DIR: Final[Path] = Path("work_dirs/minicpm_o_4_5_streaming_demo")

# Media is an opaque payload forwarded to the streaming host: a deterministic
# in-memory fixture in fake mode, or a validated audio path in real mode.
Media: TypeAlias = dict[str, object]

_SAMPLE_RATE = 16000


class FakeDuplex:
    """Deterministic in-memory duplex handle for --fake/--dry-run mode."""

    def __init__(self) -> None:
        self.prepared: bool = False

    def prepare(self, **kwargs: object) -> None:
        self.prepared = True
        del kwargs

    def streaming_prefill(self, **kwargs: object) -> None:
        del kwargs

    def streaming_generate(self, **kwargs: object):
        yield {"text": "ok", "finished": True, "audio": kwargs.get("generate_audio", False)}


class FakeHost:
    """Deterministic in-memory StreamingHost for --fake/--dry-run mode.

    Never loads a model and never touches work_dirs; each method is a no-op or
    yields a fixed token frame so the official case sequence can run end to end.
    """

    def __init__(self, seed: int) -> None:
        self.seed = seed

    def reset_session(self, reset_token2wav_cache: bool = True) -> None:
        del reset_token2wav_cache

    def init_token2wav_cache(self, prompt_speech_16k: object) -> None:
        del prompt_speech_16k

    def streaming_prefill(self, session_id: str, msgs: list[dict[str, str]], **kwargs: object) -> None:
        del session_id, msgs, kwargs

    def streaming_generate(self, session_id: str, **kwargs: object):
        del session_id
        yield {"text": "ok", "finished": True, "audio": kwargs.get("generate_audio", False)}

    def as_duplex(self, **kwargs: object) -> FakeDuplex:
        del kwargs
        return FakeDuplex()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one fixed MiniCPM-o-4.5 streaming case through Task 10 run_streaming_case()"
    )
    parser.add_argument("--case", required=True, choices=CASE_CHOICES, help="certified streaming case to run")
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="where result.json and artifacts go"
    )
    parser.add_argument("--seed", type=int, default=20260807, help="determinism seed (fake mode + artifacts)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--fake",
        "--dry-run",
        dest="fake",
        action="store_true",
        help="run deterministically with an in-memory host; no model load, no work_dirs",
    )
    parser.add_argument("--model-dir", type=Path, help="required in real mode; Hugging Face model directory")
    parser.add_argument("--work-dir", type=Path, help="required in real mode; exported HMONNX artifact directory")
    parser.add_argument("--media", type=Path, help="required in real mode; prompt/reference audio file")
    parser.add_argument("--exec-device", default="cuda:0", help="execution device for real mode")
    return parser.parse_args(argv)


def resolve_case(name: str) -> StreamingCase:
    return get_streaming_case(name)


def validate_real_inputs(args: argparse.Namespace) -> None:
    if args.fake:
        return
    required = (("--model-dir", args.model_dir), ("--work-dir", args.work_dir))
    missing = [flag for flag, value in required if value is None]
    if missing:
        raise ValueError("real mode requires " + ", ".join(missing))


def _use_exec_device_as_default(device: str) -> None:
    if device.startswith("cuda"):
        import torch

        torch.cuda.set_device(device)


def _init_host(host: Any, *, model_dir: Path, work_dir: Path) -> MiniCPMO45HMONNXRuntime:
    if hasattr(host, "init_tts"):
        host.init_tts(model_dir=str(model_dir / "assets" / "token2wav"))
    return MiniCPMO45HMONNXRuntime(work_dir, host)


def _build_real_runtime(args: argparse.Namespace) -> tuple[StreamingHost, Any, Any]:
    model_dir = Path(args.model_dir)
    work_dir = Path(args.work_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"model directory not found: {model_dir}")
    if not (work_dir / "export_meta_info.json").is_file():
        raise FileNotFoundError(f"exported HMONNX artifacts missing under work-dir: {work_dir}")
    _use_exec_device_as_default(args.exec_device)
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer, DynamicCache

    # The CLI exposes --seed as the reproducibility contract for demo artifacts.
    # Real execution samples generation tokens too, so seed every RNG before any
    # model-side processing rather than limiting it to the fake fixture path.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    config = AutoConfig.from_pretrained(str(model_dir), trust_remote_code=True)
    host = AutoModel.from_pretrained(
        str(model_dir),
        config=ensure_tts_sampling_config(config),
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="cpu",
    ).eval()
    patch_dynamic_cache_seen_tokens(DynamicCache)
    patch_audio_attention_return_compat(host)
    runtime = _init_host(host, model_dir=model_dir, work_dir=work_dir)
    runtime.set_exec_device(args.exec_device)
    return runtime, tokenizer, None


def _build_runtime(args: argparse.Namespace) -> tuple[StreamingHost, Any, Any]:
    if args.fake:
        return FakeHost(args.seed), None, None
    return _build_real_runtime(args)


def _build_media(args: argparse.Namespace) -> Media:
    if args.fake:
        return {"seed": args.seed, "kind": "deterministic-fake-audio"}
    media = Path(args.media) if args.media is not None else _default_media(args.model_dir, args.case)
    if not media.is_file():
        raise FileNotFoundError(f"media file not found: {media}")
    if media.suffix.lower() == ".wav":
        import librosa

        waveform, _ = librosa.load(media, sr=_SAMPLE_RATE, mono=True)
        chunk_lengths = [
            round(duration * _SAMPLE_RATE / 1000) for duration in resolve_case(args.case).chunk_durations_ms
        ]
        chunks: list[tuple[np.ndarray, list[object]]] = []
        offset = 0
        for length in chunk_lengths:
            chunk = waveform[offset : offset + length]
            if len(chunk) < length:
                chunk = np.pad(chunk, (0, length - len(chunk)))
            chunks.append((np.asarray(chunk, dtype=np.float32), []))
            offset += length
    else:
        chunks = video_chunks(media, _SAMPLE_RATE)
    case = resolve_case(args.case)
    chunks = chunks[: len(case.chunk_durations_ms)]
    payload: dict[str, object] = {"chunks": chunks}
    if case.generate_audio:
        import librosa

        prompt_path = Path(args.model_dir) / "assets" / "system_ref_audio.wav"
        prompt_waveform, _ = librosa.load(prompt_path, sr=_SAMPLE_RATE, mono=True)
        prompt_waveform = np.asarray(prompt_waveform, dtype=np.float32)
        # The full reference prompt is passed through; the exported Token2Wav
        # streaming graphs cover the official system_ref_audio (16.84 s -> 842 mel
        # frames), and the runtime raises a clear capacity error for longer prompts.
        prompt_wav_path = Path(args.output_dir) / "prompt_token2wav_stream.wav"
        _write_prompt_wav(prompt_wav_path, prompt_waveform)
        payload["prompt_waveform"] = prompt_waveform
        payload["prompt_wav_path"] = str(prompt_wav_path)
    return payload


def _write_prompt_wav(path: Path, waveform: np.ndarray) -> None:
    _media_write_wav(path, waveform, _SAMPLE_RATE)


def _default_media(model_dir: Path | None, case_name: str) -> Path:
    if model_dir is None:
        raise ValueError("real mode requires --model-dir when --media is omitted")
    assets = model_dir / "assets"
    if case_name in {"session_audio_text", "session_audio_reply"}:
        return assets / "Skiing.mp4"
    if case_name in {"duplex_audio_text", "duplex_audio_reply"}:
        return assets / "omni_duplex1.mp4"
    if case_name == "duplex_omni_reply":
        return assets / "omni_duplex2.mp4"
    raise ValueError(f"unsupported streaming case: {case_name}")


def _build_payload(
    args: argparse.Namespace, case: StreamingCase, result: StreamingCaseResult, mode: str
) -> dict[str, object]:
    return {
        "case": case.name,
        "case_id": case.case_id,
        "status": "ok",
        "mode": mode,
        "api_family": case.api_family,
        "generate_audio": case.generate_audio,
        "omni": case.omni,
        "final": result.final,
        "seed": args.seed,
        "role_counts": dict(result.backend_counters),
        "api_events": list(result.api_events),
        "text_chunks": list(result.text_chunks),
        "real_text": "".join(result.real_text_chunks),
        "real_audio_chunk_samples": [int(wave.size) for wave in result.real_waveform_chunks],
        "token_ids": list(result.token_ids),
        "waveform_chunks": list(result.waveform_chunks),
        "elapsed_ms": list(result.elapsed_ms),
    }


def _write_tiny_wav(path: Path, seed: int, seconds: float = 0.05) -> None:
    rng = np.random.default_rng(seed)
    samples = int(_SAMPLE_RATE * seconds)
    t = np.arange(samples, dtype=np.float32) / _SAMPLE_RATE
    tone = (0.25 * np.sin(2 * np.pi * 440.0 * t) + 0.05 * rng.standard_normal(samples).astype(np.float32)).astype(
        np.float32
    )
    _media_write_wav(path, tone, _SAMPLE_RATE)


def _write_artifacts(
    output_dir: Path,
    args: argparse.Namespace,
    case: StreamingCase,
    payload: dict[str, object],
    result: StreamingCaseResult,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    real_text = "".join(result.real_text_chunks)
    if real_text:
        text = real_text
    else:
        text = "\n".join(str(chunk) for chunk in payload["text_chunks"]) + "\n"
    (output_dir / f"text_{case.name}.txt").write_text(text, encoding="utf-8")
    if case.generate_audio:
        if result.real_waveform_chunks:
            waveform = flatten_audio_chunks(result.real_waveform_chunks)
            _media_write_wav(output_dir / f"audio_{case.name}.wav", waveform, 24000)
        else:
            _write_tiny_wav(output_dir / f"audio_{case.name}.wav", args.seed)


def run_streaming_demo(args: argparse.Namespace) -> dict[str, object]:
    case = resolve_case(args.case)
    mode = "fake" if args.fake else "real"
    validate_real_inputs(args)
    host, tokenizer, processor = _build_runtime(args)
    media = _build_media(args)
    result = run_streaming_case(model=host, tokenizer=tokenizer, processor=processor, case=case, media=media)
    payload = _build_payload(args, case, result, mode)
    _write_artifacts(args.output_dir, args, case, payload, result)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        payload = run_streaming_demo(args)
    except (StreamingCaseError, FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    role_count = len(payload["role_counts"])
    event_count = len(payload["api_events"])
    print(
        f"{payload['case']} status={payload['status']} mode={payload['mode']} roles={role_count} events={event_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
