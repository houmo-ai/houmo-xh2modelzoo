from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch


@dataclass
class FakeHost:
    events: list[tuple[str, object]]

    def reset_session(self, reset_token2wav_cache: bool = True) -> None:
        self.events.append(("reset_session", reset_token2wav_cache))

    def init_token2wav_cache(self, speech: object) -> None:
        self.events.append(("init_token2wav_cache", speech))

    def streaming_prefill(self, session_id: str, msgs: list[dict[str, str]], **kwargs: object) -> None:
        self.events.append(("streaming_prefill", (session_id, msgs, kwargs)))

    def streaming_generate(self, session_id: str, **kwargs: object):
        self.events.append(("streaming_generate", (session_id, kwargs)))
        yield {"text": "ok", "finished": True, "audio": kwargs.get("generate_audio", False)}

    def as_duplex(self, **kwargs: object) -> FakeDuplex:
        self.events.append(("as_duplex", kwargs))
        return FakeDuplex(self.events)


@dataclass
class FakeDuplex:
    events: list[tuple[str, object]]

    def prepare(self, **kwargs: object) -> None:
        self.events.append(("prepare", kwargs))

    def streaming_prefill(self, **kwargs: object) -> None:
        self.events.append(("duplex_streaming_prefill", kwargs))

    def streaming_generate(self, **kwargs: object):
        self.events.append(("duplex_streaming_generate", kwargs))
        return {"text": "ok", "finished": True, "audio": kwargs.get("prompt_wav_path") is not None}


@dataclass
class OfficialDuplex:
    events: list[tuple[str, object]]

    def prepare(self, **kwargs: object) -> None:
        self.events.append(("official_prepare", kwargs))

    def streaming_prefill(
        self,
        *,
        audio_waveform: np.ndarray | None = None,
        frame_list: list[object] | None = None,
        text_list: list[str] | None = None,
    ) -> dict[str, object]:
        self.events.append(("official_duplex_prefill", (audio_waveform, frame_list, text_list)))
        return {"success": True}

    def streaming_generate(
        self,
        *,
        prompt_wav_path: str | None = None,
        max_new_speak_tokens_per_chunk: int = 20,
    ) -> dict[str, object]:
        self.events.append(
            (
                "official_duplex_generate",
                {
                    "prompt_wav_path": prompt_wav_path,
                    "max_new_speak_tokens_per_chunk": max_new_speak_tokens_per_chunk,
                },
            )
        )
        return {"text": "ok", "audio_waveform": np.zeros(4, dtype=np.float32), "end_of_turn": True}


@dataclass
class OfficialHost(FakeHost):
    duplex: OfficialDuplex | None = None

    def as_duplex(self, **kwargs: object) -> OfficialDuplex:
        self.events.append(("as_duplex", kwargs))
        self.duplex = OfficialDuplex(self.events)
        return self.duplex


def test_duplex_fixture_uses_official_single_result_contract() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
        get_streaming_case,
        run_streaming_case,
    )

    events: list[tuple[str, object]] = []
    media = np.zeros(16000, dtype=np.float32)
    result = run_streaming_case(OfficialHost(events), None, None, get_streaming_case("duplex_audio_reply"), media)

    assert result.final is True
    assert [name for name, _ in events] == [
        "as_duplex",
        "official_prepare",
        "official_duplex_prefill",
        "official_duplex_generate",
        "official_duplex_prefill",
        "official_duplex_generate",
    ]
    prefill = events[2][1]
    assert isinstance(prefill, tuple)
    assert isinstance(prefill[0], np.ndarray)
    assert prefill[1] is None
    assert prefill[2] is None


def test_duplex_omni_fixture_passes_frames_to_official_prefill() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
        get_streaming_case,
        run_streaming_case,
    )

    events: list[tuple[str, object]] = []
    result = run_streaming_case(
        OfficialHost(events),
        None,
        None,
        get_streaming_case("duplex_omni_reply"),
        {
            "chunks": [
                (np.zeros(16000, dtype=np.float32), ["frame-0"]),
                (np.zeros(16000, dtype=np.float32), ["frame-1"]),
            ],
            "prompt_waveform": np.zeros(16000, dtype=np.float32),
            "prompt_wav_path": "prompt.wav",
        },
    )

    assert result.final is True
    prefill = events[2][1]
    assert isinstance(prefill, tuple)
    assert isinstance(prefill[1], list)
    assert len(prefill[1]) == 1


def test_five_cases_have_stable_identity_and_roles() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
        CERTIFIED_STREAMING_CASES,
        get_streaming_case,
    )

    assert tuple(case.name for case in CERTIFIED_STREAMING_CASES) == (
        "session_audio_text",
        "session_audio_reply",
        "duplex_audio_text",
        "duplex_audio_reply",
        "duplex_omni_reply",
    )
    assert len({case.case_id for case in CERTIFIED_STREAMING_CASES}) == 5
    assert get_streaming_case("duplex_omni_reply").required_roles == (
        "audio.stream_prefill",
        "audio.stream_decode",
        "llm.prefill",
        "llm.decode",
        "tts.prefill",
        "tts.decode",
        "token2wav.stream_flow_frontend",
        "token2wav.stream_flow_frontend_final",
        "token2wav.stream_flow_estimator_step",
        "token2wav.stream_hift",
        "token2wav.stream_hift_final",
    )


def test_runner_reproduces_session_audio_reply_and_final_audio_semantics() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
        get_streaming_case,
        run_streaming_case,
    )

    events: list[tuple[str, object]] = []
    result = run_streaming_case(FakeHost(events), None, None, get_streaming_case("session_audio_reply"), "tiny-audio")

    assert result.case_name == "session_audio_reply"
    assert [name for name, _ in events] == [
        "init_token2wav_cache",
        "reset_session",
        "streaming_prefill",
        "streaming_prefill",
        "streaming_generate",
    ]
    assert events[1][1] is False
    generate_args = events[-1][1]
    assert isinstance(generate_args, tuple)
    assert generate_args[1]["max_new_tokens"] == 16
    assert result.final is True
    assert result.waveform_chunks == ("audio",)


def test_session_text_decodes_token_chunks_instead_of_finished_flags() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
        get_streaming_case,
        run_streaming_case,
    )

    class TokenTextHost(FakeHost):
        def streaming_generate(self, session_id: str, **kwargs: object):
            self.events.append(("streaming_generate", (session_id, kwargs)))
            yield torch.tensor([[11, 12]]), False
            yield torch.tensor([[13]]), True

    class Tokenizer:
        def decode(self, token_ids: list[int]) -> str:
            assert token_ids == [11, 12, 13]
            return "decoded session text"

    result = run_streaming_case(
        TokenTextHost([]),
        Tokenizer(),
        None,
        get_streaming_case("session_audio_text"),
        "tiny-audio",
    )

    assert result.real_text_chunks == ("decoded session text",)


def test_session_text_keeps_predecoded_chunks_instead_of_finished_flags() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
        get_streaming_case,
        run_streaming_case,
    )

    class PredecodedTextHost(FakeHost):
        def streaming_generate(self, session_id: str, **kwargs: object):
            self.events.append(("streaming_generate", (session_id, kwargs)))
            yield "hello ", False
            yield "world", True

    result = run_streaming_case(
        PredecodedTextHost([]),
        None,
        None,
        get_streaming_case("session_audio_text"),
        "tiny-audio",
    )

    assert result.real_text_chunks == ("hello ", "world")


def test_runner_reproduces_duplex_omni_roles_and_final_signal() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
        get_streaming_case,
        run_streaming_case,
    )

    events: list[tuple[str, object]] = []
    result = run_streaming_case(
        FakeHost(events),
        None,
        None,
        get_streaming_case("duplex_omni_reply"),
        {
            "chunks": [("audio-0", ["frame-0"]), ("audio-1", ["frame-1"])],
            "prompt_waveform": "prompt-audio",
            "prompt_wav_path": "prompt.wav",
        },
    )

    assert [name for name, _ in events] == [
        "as_duplex",
        "prepare",
        "duplex_streaming_prefill",
        "duplex_streaming_generate",
        "duplex_streaming_prefill",
        "duplex_streaming_generate",
    ]
    assert result.final is True
    assert result.api_events[-1] == "duplex_streaming_generate:final"
    generate_kwargs = events[-1][1]
    assert isinstance(generate_kwargs, dict)
    assert generate_kwargs["max_new_speak_tokens_per_chunk"] == 16


def test_runner_requires_media_for_audio_cases() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
        StreamingCaseError,
        get_streaming_case,
        run_streaming_case,
    )

    with pytest.raises(StreamingCaseError, match="media"):
        run_streaming_case(FakeHost([]), None, None, get_streaming_case("session_audio_text"), None)


@pytest.mark.parametrize(
    "case_name",
    (
        "session_audio_text",
        "session_audio_reply",
        "duplex_audio_text",
        "duplex_audio_reply",
        "duplex_omni_reply",
    ),
)
def test_every_certified_case_runs_with_compact_media(case_name: str) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
        get_streaming_case,
        run_streaming_case,
    )

    events: list[tuple[str, object]] = []
    result = run_streaming_case(FakeHost(events), None, None, get_streaming_case(case_name), "tiny-audio")

    assert result.case_name == case_name
    assert result.final is True
    assert result.api_events


def test_runner_rejects_unknown_case() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
        StreamingCaseError,
        get_streaming_case,
    )

    with pytest.raises(StreamingCaseError, match="unknown streaming case"):
        get_streaming_case("not-certified")
