from __future__ import annotations

from types import SimpleNamespace


def test_runtime_delegates_session_calls() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import MiniCPMO45HMONNXRuntime

    runtime = object.__new__(MiniCPMO45HMONNXRuntime)
    runtime.host_model = SimpleNamespace(
        reset_session=lambda value=True: ("reset", value),
        init_token2wav_cache=lambda speech: ("cache", speech),
        streaming_prefill=lambda *args, **kwargs: (args, kwargs),
        streaming_generate=lambda *args, **kwargs: iter([("chunk", False)]),
        as_duplex=lambda *args, **kwargs: (args, kwargs),
    )
    runtime.audio = SimpleNamespace(reset_state=lambda: None)
    runtime.llm = SimpleNamespace(reset_state=lambda: None)
    runtime.tts = SimpleNamespace(reset_state=lambda: None)
    runtime.token2wav = SimpleNamespace(reset_state=lambda: None)

    assert runtime.reset_session(False) == ("reset", False)
    assert runtime.init_token2wav_cache("speech16k") == ("cache", "speech16k")

    session_id = "s0"
    msgs = [{"role": "user", "content": "hello"}]
    assert runtime.streaming_prefill(session_id, msgs, audio=1) == ((session_id, msgs), {"audio": 1})

    assert list(runtime.streaming_generate("s0", generate_audio=False)) == [("chunk", False)]
    runtime.reset_session()
    assert runtime.as_duplex(device="cuda:0", extra=2) == ((), {"device": "cuda:0", "extra": 2})


def test_certified_streaming_case_names() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import STREAMING_CASES

    assert STREAMING_CASES == (
        "session_audio_text",
        "session_audio_reply",
        "duplex_audio_text",
        "duplex_audio_reply",
        "duplex_omni_reply",
    )
