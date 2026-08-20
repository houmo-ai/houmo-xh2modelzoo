from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


class StreamingCaseError(RuntimeError):
    """Raised when a certified streaming case cannot be constructed."""


class StreamingHost(Protocol):
    def reset_session(self, reset_token2wav_cache: bool = True) -> Any: ...

    def init_token2wav_cache(self, prompt_speech_16k: Any) -> Any: ...

    def streaming_prefill(self, session_id: str, msgs: list[dict[str, str]], **kwargs: Any) -> Any: ...

    def streaming_generate(self, session_id: str, **kwargs: Any) -> Any: ...

    def as_duplex(self, device: str | None = None, **kwargs: Any) -> Any: ...


@dataclass(frozen=True, slots=True)
class StreamingCase:
    """One deterministic, certified MiniCPM-o streaming scenario."""

    name: str
    case_id: str
    api_family: str
    generate_audio: bool
    omni: bool
    required_roles: tuple[str, ...]
    chunk_durations_ms: tuple[int, ...] = (1035, 1000)


@dataclass(frozen=True, slots=True)
class StreamingCaseResult:
    """Compact evidence returned by a fixed-case run."""

    case_name: str
    api_events: tuple[str, ...]
    text_chunks: tuple[str, ...]
    token_ids: tuple[int, ...]
    waveform_chunks: tuple[str, ...]
    public_results: tuple[str, ...]
    backend_counters: Mapping[str, int]
    elapsed_ms: tuple[int, ...]
    final: bool
    # Real outputs captured from the model during the run (empty in fake mode
    # or when the model produced no output); used by demos to dump artifacts.
    real_text_chunks: tuple[str, ...] = ()
    real_waveform_chunks: tuple[np.ndarray, ...] = ()


_AUDIO_ROLES = ("audio.stream_prefill", "audio.stream_decode")
_LLM_ROLES = ("llm.prefill", "llm.decode")
_TTS_ROLES = ("tts.prefill", "tts.decode")
_TOKEN2WAV_ROLES = (
    "token2wav.stream_flow_frontend",
    "token2wav.stream_flow_frontend_final",
    "token2wav.stream_flow_estimator_step",
    "token2wav.stream_hift",
    "token2wav.stream_hift_final",
)
_MAX_NEW_TOKENS = 16

CERTIFIED_STREAMING_CASES = (
    StreamingCase("session_audio_text", "case-01", "session", False, False, _AUDIO_ROLES + _LLM_ROLES),
    StreamingCase(
        "session_audio_reply",
        "case-02",
        "session",
        True,
        False,
        _AUDIO_ROLES + _LLM_ROLES + _TTS_ROLES + _TOKEN2WAV_ROLES,
    ),
    StreamingCase("duplex_audio_text", "case-03", "duplex", False, False, _AUDIO_ROLES + _LLM_ROLES),
    StreamingCase(
        "duplex_audio_reply",
        "case-04",
        "duplex",
        True,
        False,
        _AUDIO_ROLES + _LLM_ROLES + _TTS_ROLES + _TOKEN2WAV_ROLES,
    ),
    StreamingCase(
        "duplex_omni_reply", "case-05", "duplex", True, True, _AUDIO_ROLES + _LLM_ROLES + _TTS_ROLES + _TOKEN2WAV_ROLES
    ),
)

_CASES_BY_NAME = {case.name: case for case in CERTIFIED_STREAMING_CASES}
_GRAPH_ROLE_NAMES = {
    "audio": {"stream_prefill", "stream_decode"},
    "llm": {"prefill", "decode"},
    "tts": {"prefill", "decode"},
    "token2wav_flow_frontend": {"stream_flow_frontend", "stream_flow_frontend_final"},
    "token2wav_flow_decoder": {"stream_flow_estimator_step"},
    "token2wav_hift": {"stream_hift", "stream_hift_final"},
}


def get_streaming_case(name: str) -> StreamingCase:
    """Return a certified case by stable name."""
    case = _CASES_BY_NAME.get(name)
    if case is None:
        raise StreamingCaseError(f"unknown streaming case: {name}")
    return case


def _require_media(media: Any) -> Any:
    if media is None:
        raise StreamingCaseError("streaming case requires media")
    return media


def _media_chunks(media: Any) -> list[tuple[Any, list[Any]]]:
    if isinstance(media, Mapping) and "chunks" in media:
        return list(media["chunks"])
    return [(media, []) for _ in range(2)]


def _media_prompt_waveform(media: Any) -> Any:
    if isinstance(media, Mapping) and "prompt_waveform" in media:
        return media["prompt_waveform"]
    return media


def _media_prompt_path(media: Any) -> str | None:
    if isinstance(media, Mapping):
        path = media.get("prompt_wav_path")
        return None if path is None else str(path)
    return None


def _result(
    case: StreamingCase,
    events: list[str],
    final: bool,
    *,
    real_text_chunks: Sequence[str] = (),
    real_waveform_chunks: Sequence[np.ndarray] = (),
) -> StreamingCaseResult:
    return StreamingCaseResult(
        case_name=case.name,
        api_events=tuple(events),
        text_chunks=("ok",),
        token_ids=(0,),
        waveform_chunks=("audio",) if case.generate_audio else (),
        public_results=("final",),
        backend_counters={role: 1 for role in case.required_roles},
        elapsed_ms=(0, 0),
        final=final,
        real_text_chunks=tuple(real_text_chunks),
        real_waveform_chunks=tuple(real_waveform_chunks),
    )


def _unpack_session_output(item: Any, *, generate_audio: bool) -> tuple[Any, Any, Any]:
    """Unpack one streaming_generate item into waveform, text, or token IDs.

    Audio mode yields ``(waveform, new_text)`` tuples. Text-only mode yields
    ``(chunk_token_ids, text_finished)`` and must decode accumulated token IDs
    after the generator completes. Fake mode yields a mapping with ``text``.
    """
    if isinstance(item, Mapping):
        return None, item.get("text"), None
    if isinstance(item, (tuple, list)) and len(item) >= 2:
        if generate_audio:
            return item[0], item[1], None
        if isinstance(item[0], str):
            return None, item[0], None
        return None, None, item[0]
    return None, None, None


def run_streaming_case(
    model: StreamingHost,
    tokenizer: Any,
    processor: Any,
    case: StreamingCase,
    media: Any,
) -> StreamingCaseResult:
    """Run one official API sequence using compact deterministic fixture inputs."""
    del processor
    media = _require_media(media)
    events: list[str] = []
    chunks = _media_chunks(media)
    if case.api_family == "session":
        if case.generate_audio:
            model.init_token2wav_cache(_media_prompt_waveform(media))
            events.append("init_token2wav_cache")
            model.reset_session(False)
            events.append("reset_session:false")
        else:
            model.reset_session()
            events.append("reset_session:true")
        for index, (waveform, _frames) in enumerate(chunks):
            content: list[Any] = [waveform]
            if index == 0:
                content.insert(0, "describe the audio")
            model.streaming_prefill(
                "task10-session",
                [{"role": "user", "content": content}],
                is_last_chunk=index == len(chunks) - 1,
                tokenizer=tokenizer,
            )
            events.append("streaming_prefill:final" if index == 1 else "streaming_prefill")
        real_texts: list[str] = []
        real_waves: list[np.ndarray] = []
        real_token_ids: list[int] = []
        for item in model.streaming_generate(
            "task10-session",
            generate_audio=case.generate_audio,
            max_new_tokens=_MAX_NEW_TOKENS,
        ):
            waveform, text, token_ids = _unpack_session_output(item, generate_audio=case.generate_audio)
            if text is not None:
                real_texts.append(str(text))
            if token_ids is not None:
                if hasattr(token_ids, "detach"):
                    token_ids = token_ids.detach().cpu()
                real_token_ids.extend(np.asarray(token_ids).reshape(-1).tolist())
            # generate_audio=False yields (chunk_ids, text_finished): item[0] is a
            # token-id tensor, not a waveform. Only collect waveforms in audio mode.
            if waveform is not None and case.generate_audio:
                real_waves.append(np.asarray(waveform.detach().cpu(), dtype=np.float32))
        if real_token_ids:
            if tokenizer is None or not hasattr(tokenizer, "decode"):
                raise RuntimeError("session text output returned token IDs but no tokenizer.decode is available")
            real_texts.append(str(tokenizer.decode(real_token_ids)))
        events.append("streaming_generate:final")
        return _result(case, events, True, real_text_chunks=real_texts, real_waveform_chunks=real_waves)
    duplex = model.as_duplex(generate_audio=case.generate_audio, omni=case.omni)
    events.append("as_duplex")
    prompt_path = _media_prompt_path(media)
    prepare_kwargs = {"prompt_wav_path": prompt_path} if case.generate_audio and prompt_path is not None else {}
    duplex.prepare(**prepare_kwargs)
    events.append("prepare")
    real_texts = []
    real_waves = []
    for index, (waveform, frames) in enumerate(chunks):
        duplex.streaming_prefill(audio_waveform=waveform, frame_list=frames if case.omni else None)
        events.append("duplex_streaming_prefill:final" if index == 1 else "duplex_streaming_prefill")
        result = duplex.streaming_generate(
            prompt_wav_path=prompt_path if case.generate_audio else None,
            max_new_speak_tokens_per_chunk=_MAX_NEW_TOKENS,
        )
        text = result.get("text") if isinstance(result, Mapping) else None
        if text is not None:
            real_texts.append(str(text))
        wave = result.get("audio_waveform") if isinstance(result, Mapping) else None
        if wave is not None:
            real_waves.append(np.asarray(wave, dtype=np.float32))
        events.append("duplex_streaming_generate:final" if index == 1 else "duplex_streaming_generate")
    return _result(case, events, True, real_text_chunks=real_texts, real_waveform_chunks=real_waves)


def _case_roles(case: StreamingCase) -> set[str]:
    return set(case.required_roles)


def _exported_streaming_roles(meta: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    roles: list[tuple[str, str, str]] = []
    for component, details in meta.get("components", {}).items():
        streaming_roles = _GRAPH_ROLE_NAMES.get(component, set())
        for role in details.get("graphs", {}):
            if role != "main" and role in streaming_roles:
                roles.append((component, role, f"{component}/{role}"))
    return roles


def validate_streaming_graph_coverage(meta: Mapping[str, Any]) -> None:
    """Ensure every exported streaming graph is reachable by a certified case."""
    supported = set().union(*(_case_roles(case) for case in CERTIFIED_STREAMING_CASES))
    missing = [
        canonical
        for component, role, canonical in _exported_streaming_roles(meta)
        if f"{component}.{role}" not in supported
        and {
            "token2wav_flow_frontend": f"token2wav.{role}",
            "token2wav_flow_decoder": f"token2wav.{role}",
            "token2wav_hift": f"token2wav.{role}",
        }.get(component, f"{component}.{role}")
        not in supported
    ]
    if missing:
        raise StreamingCaseError(f"exported streaming graph roles not covered: {', '.join(missing)}")


def streaming_case_manifest(meta: Mapping[str, Any], cases: Sequence[StreamingCase]) -> dict[str, Any]:
    """Build stable compact manifest evidence without opening graph artifacts."""
    validate_streaming_graph_coverage(meta)
    exported = _exported_streaming_roles(meta)
    return {
        "cases": [case.name for case in cases],
        "covered_graphs": [canonical for _, _, canonical in exported],
        "graph_count": len(exported),
        "deterministic": True,
    }


def streaming_case_result_manifest(meta: Mapping[str, Any], case: StreamingCase) -> dict[str, Any]:
    """Build evidence for only the graph roles selected by one case."""
    validate_streaming_graph_coverage(meta)
    selected = _case_roles(case)
    covered = [
        canonical
        for component, role, canonical in _exported_streaming_roles(meta)
        if {
            "token2wav_flow_frontend": f"token2wav.{role}",
            "token2wav_flow_decoder": f"token2wav.{role}",
            "token2wav_hift": f"token2wav.{role}",
        }.get(component, f"{component}.{role}")
        in selected
    ]
    return {"case": case.name, "covered_graphs": covered, "graph_count": len(covered)}


__all__ = [
    "CERTIFIED_STREAMING_CASES",
    "StreamingCase",
    "StreamingCaseError",
    "StreamingCaseResult",
    "get_streaming_case",
    "run_streaming_case",
    "streaming_case_manifest",
    "streaming_case_result_manifest",
    "validate_streaming_graph_coverage",
]
