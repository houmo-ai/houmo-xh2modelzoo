from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


OFFICIAL_ROOT = Path(os.environ["MINICPM_O45_MODEL_DIR"]) if os.environ.get("MINICPM_O45_MODEL_DIR") else None
OFFICIAL_MODELING = OFFICIAL_ROOT / "modeling_minicpmo.py" if OFFICIAL_ROOT is not None else None


def _official_module():
    if OFFICIAL_MODELING is None:
        pytest.skip("set MINICPM_O45_MODEL_DIR to run official remote-code tests")
    if not OFFICIAL_MODELING.is_file():
        pytest.skip(f"official MiniCPM-o-4.5 source not found: {OFFICIAL_MODELING}")
    package = "task9_official_minicpmo"
    module_name = f"{package}.modeling_minicpmo"
    if module_name in sys.modules:
        return sys.modules[module_name]
    package_module = type(sys)(package)
    package_module.__path__ = [str(OFFICIAL_MODELING.parent)]
    sys.modules[package] = package_module
    spec = importlib.util.spec_from_file_location(
        module_name,
        OFFICIAL_MODELING,
        submodule_search_locations=[str(OFFICIAL_MODELING.parent)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_with_host(host: object):
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import MiniCPMO45HMONNXRuntime

    runtime = object.__new__(MiniCPMO45HMONNXRuntime)
    runtime.host_model = host
    runtime.audio = SimpleNamespace(reset_state=lambda: None)
    runtime.llm = SimpleNamespace(reset_state=lambda: None)
    runtime.tts = SimpleNamespace(reset_state=lambda: None)
    runtime.token2wav = SimpleNamespace(reset_state=lambda: None)
    return runtime


def test_official_session_and_duplex_method_identity_and_signatures() -> None:
    official = _official_module()
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import MiniCPMO45HMONNXRuntime

    host_cls = official.MiniCPMO
    duplex_cls = official.MiniCPMODuplex
    host = SimpleNamespace(
        reset_session=lambda reset_token2wav_cache=True: None,
        streaming_prefill=lambda session_id, msgs, **kwargs: "prompt",
        streaming_generate=lambda session_id, **kwargs: iter(()),
        as_duplex=lambda device=None, **kwargs: None,
    )
    runtime = _runtime_with_host(host)

    assert host_cls.streaming_prefill.__name__ == "streaming_prefill"
    assert host_cls.streaming_generate.__name__ == "streaming_generate"
    assert host_cls.as_duplex.__name__ == "as_duplex"
    assert duplex_cls.prepare.__name__ == "prepare"
    assert duplex_cls.streaming_prefill.__name__ == "streaming_prefill"
    assert duplex_cls.streaming_generate.__name__ == "streaming_generate"
    assert tuple(inspect.signature(host_cls.streaming_prefill).parameters)[:2] == ("self", "session_id")
    assert tuple(inspect.signature(host_cls.streaming_generate).parameters)[:2] == ("self", "session_id")
    assert tuple(inspect.signature(host_cls.as_duplex).parameters)[:2] == ("self", "device")
    assert tuple(inspect.signature(duplex_cls.prepare).parameters)[:2] == ("self", "prefix_system_prompt")
    assert tuple(inspect.signature(duplex_cls.streaming_prefill).parameters)[:2] == ("self", "audio_waveform")
    assert tuple(inspect.signature(duplex_cls.streaming_generate).parameters)[:1] == ("self",)
    assert isinstance(runtime, MiniCPMO45HMONNXRuntime)


def test_session_and_duplex_modes_reject_interleaving_until_full_reset() -> None:
    calls: list[str] = []
    host = SimpleNamespace(
        reset_session=lambda reset_token2wav_cache=True: calls.append(f"reset:{reset_token2wav_cache}"),
        streaming_prefill=lambda session_id, msgs, **kwargs: calls.append(f"prefill:{session_id}") or "prompt",
        streaming_generate=lambda session_id, **kwargs: iter(()),
        as_duplex=lambda device=None, **kwargs: calls.append("duplex") or {"device": device},
    )
    runtime = _runtime_with_host(host)

    assert runtime.streaming_prefill("session-a", [{"role": "user", "content": ["hi"]}]) == "prompt"
    with pytest.raises(RuntimeError, match="Session API is active"):
        runtime.as_duplex()

    runtime.reset_session(False)
    assert runtime.as_duplex(device="cpu") == {"device": "cpu"}
    with pytest.raises(RuntimeError, match="Duplex API is active"):
        runtime.streaming_prefill("session-b", [])
    assert calls == ["prefill:session-a", "reset:False", "duplex"]


def test_reset_session_clears_mode_but_preserves_token2wav_base_when_requested() -> None:
    token2wav_resets: list[str] = []
    host = SimpleNamespace(
        reset_session=lambda reset_token2wav_cache=True: None,
        streaming_prefill=lambda session_id, msgs, **kwargs: "prompt",
        streaming_generate=lambda session_id, **kwargs: iter(()),
        as_duplex=lambda device=None, **kwargs: object(),
    )
    runtime = _runtime_with_host(host)
    runtime.token2wav = SimpleNamespace(reset_state=lambda: token2wav_resets.append("reset"))

    runtime.streaming_prefill("session-a", [])
    runtime.reset_session(False)
    assert token2wav_resets == []
    runtime.as_duplex()
    runtime.reset_session(True)
    assert token2wav_resets == ["reset"]


def test_official_result_and_generator_shapes_are_forwarded_without_fallback() -> None:
    official_result = {
        "success": True,
        "reason": "",
        "terminal": False,
    }
    official_chunks = iter([("waveform", "text"), ("terminal-waveform", "")])
    host = SimpleNamespace(
        reset_session=lambda reset_token2wav_cache=True: None,
        streaming_prefill=lambda session_id, msgs, **kwargs: official_result,
        streaming_generate=lambda session_id, **kwargs: official_chunks,
        as_duplex=lambda device=None, **kwargs: {"is_listen": True, "end_of_turn": True},
    )
    runtime = _runtime_with_host(host)

    assert runtime.streaming_prefill("session-a", []) is official_result
    assert list(runtime.streaming_generate("session-a")) == [("waveform", "text"), ("terminal-waveform", "")]


def test_streaming_generate_resets_hmonnx_state_for_a_new_session() -> None:
    resets: list[str] = []
    official_resets: list[bool] = []
    host = SimpleNamespace(
        reset_session=lambda reset_token2wav_cache=True: official_resets.append(reset_token2wav_cache),
        streaming_prefill=lambda session_id, msgs, **kwargs: "prompt",
        streaming_generate=lambda session_id, **kwargs: iter([(session_id, False)]),
        as_duplex=lambda device=None, **kwargs: SimpleNamespace(model=host),
    )
    runtime = _runtime_with_host(host)
    runtime.audio.reset_state = lambda: resets.append("audio")
    runtime.llm.reset_state = lambda: resets.append("llm")
    runtime.tts.reset_state = lambda: resets.append("tts")
    runtime.token2wav.reset_state = lambda: resets.append("token2wav")

    runtime.streaming_prefill("session-a", [])
    assert list(runtime.streaming_generate("session-a")) == [("session-a", False)]
    assert list(runtime.streaming_generate("session-b")) == [("session-b", False)]

    assert official_resets == [False]
    assert resets == ["audio", "llm", "tts", "token2wav"]


def test_real_duplex_binding_resets_hmonnx_only_when_duplex_is_active() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import bind_duplex_hmonnx_reset

    calls: list[str] = []
    active = False

    class Host:
        def init_streaming_processor(self) -> str:
            calls.append("official")
            return "initialized"

    class Duplex:
        def __init__(self) -> None:
            self.model = Host()

    duplex = Duplex()
    bind_duplex_hmonnx_reset(duplex, lambda: calls.append("reset"), lambda: active)
    assert duplex.model.init_streaming_processor() == "initialized"
    assert calls == ["official"]
    active = True
    assert duplex.model.init_streaming_processor() == "initialized"
    assert calls == ["official", "reset", "official"]
    bind_duplex_hmonnx_reset(duplex, lambda: calls.append("duplicate-reset"), lambda: active)
    assert duplex.model.init_streaming_processor() == "initialized"
    assert calls == ["official", "reset", "official", "duplicate-reset", "official"]


def test_duplex_binding_preserves_official_method_identity() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import bind_duplex_hmonnx_reset

    class Host:
        def init_streaming_processor(self):
            return "official"

    host = Host()
    original = host.init_streaming_processor.__func__
    duplex = SimpleNamespace(model=host)

    bind_duplex_hmonnx_reset(duplex, lambda: None)

    assert host.init_streaming_processor.__name__ == original.__name__
    assert host.init_streaming_processor.__wrapped__.__name__ == original.__name__


def test_all_certified_cases_reach_an_official_delegate_surface() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import STREAMING_CASES

    calls: list[str] = []
    host = SimpleNamespace(
        reset_session=lambda reset_token2wav_cache=True: calls.append("reset"),
        streaming_prefill=lambda session_id, msgs, **kwargs: calls.append("session-prefill") or "prompt",
        streaming_generate=lambda session_id, **kwargs: calls.append("session-generate") or iter(()),
        as_duplex=lambda device=None, **kwargs: calls.append("duplex") or SimpleNamespace(model=host),
    )
    runtime = _runtime_with_host(host)

    runtime.streaming_prefill("session", [])
    list(runtime.streaming_generate("session"))
    runtime.reset_session(False)
    runtime.as_duplex()
    assert len(STREAMING_CASES) == 5
    assert calls == ["session-prefill", "session-generate", "reset", "duplex"]


@pytest.mark.parametrize(
    ("case_name", "expected_calls"),
    [
        ("session_audio_text", ["session-prefill", "session-generate:text"]),
        ("session_audio_reply", ["session-prefill", "session-generate:audio"]),
        ("duplex_audio_text", ["duplex", "duplex-prefill:audio", "duplex-generate:text"]),
        ("duplex_audio_reply", ["duplex", "duplex-prefill:audio", "duplex-generate:audio"]),
        ("duplex_omni_reply", ["duplex", "duplex-prefill:omni", "duplex-generate:audio"]),
    ],
)
def test_certified_case_executes_its_delegate_surface(case_name: str, expected_calls: list[str]) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import STREAMING_CASES

    calls: list[str] = []

    class Duplex:
        def __init__(self, model) -> None:
            self.model = model

        def streaming_prefill(self, *, audio_waveform=None, frame_list=None, text_list=None):
            mode = "omni" if frame_list else ("audio" if audio_waveform else "text")
            calls.append(f"duplex-prefill:{mode}")
            return {"success": True}

        def streaming_generate(self, *, prompt_wav_path=None):
            calls.append(f"duplex-generate:{'audio' if prompt_wav_path else 'text'}")
            return {"end_of_turn": True}

    host = SimpleNamespace(
        reset_session=lambda reset_token2wav_cache=True: None,
        streaming_prefill=lambda session_id, msgs, **kwargs: calls.append("session-prefill") or "prompt",
        streaming_generate=lambda session_id, **kwargs: (
            calls.append(f"session-generate:{'audio' if kwargs.get('generate_audio') else 'text'}") or iter(())
        ),
        as_duplex=lambda device=None, **kwargs: calls.append("duplex") or Duplex(host),
    )
    runtime = _runtime_with_host(host)

    assert case_name in STREAMING_CASES
    if case_name.startswith("session"):
        runtime.streaming_prefill("session", [])
        list(runtime.streaming_generate("session", generate_audio=case_name.endswith("reply")))
    else:
        duplex = runtime.as_duplex()
        duplex.streaming_prefill(
            audio_waveform=[1],
            frame_list=[1] if case_name == "duplex_omni_reply" else None,
        )
        duplex.streaming_generate(prompt_wav_path="prompt.wav" if case_name.endswith("reply") else None)

    assert calls == expected_calls


def test_as_duplex_reuses_initialized_tts_on_a_real_object_shape() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import reuse_initialized_tts

    class Tokenizer:
        name = "initialized-tokenizer"

    class Host:
        def __init__(self) -> None:
            self.tts = SimpleNamespace(audio_tokenizer=Tokenizer())

        def init_tts(self, *args, **kwargs):
            raise AssertionError("official Duplex construction must reuse initialized TTS")

        def create_duplex(self):
            return SimpleNamespace(model=self, observed_tts=self.init_tts(enable_float16=True))

    host = Host()
    duplex = reuse_initialized_tts(host, host.tts.audio_tokenizer, host.create_duplex)

    assert duplex.observed_tts.name == "initialized-tokenizer"
    assert duplex.model is host
    with pytest.raises(AssertionError, match="reuse initialized TTS"):
        host.init_tts()


def test_runtime_as_duplex_uses_initialized_tts_on_actual_delegate_surface() -> None:
    class Host:
        def __init__(self) -> None:
            self.tts = SimpleNamespace(audio_tokenizer=SimpleNamespace(name="tokenizer"))

        def reset_session(self, reset_token2wav_cache=True):
            return reset_token2wav_cache

        def streaming_prefill(self, session_id, msgs, **kwargs):
            return session_id, msgs, kwargs

        def streaming_generate(self, session_id, **kwargs):
            return iter(((session_id, kwargs),))

        def init_tts(self, *args, **kwargs):
            raise AssertionError("native TTS initialization is forbidden after attachment")

        def as_duplex(self, device=None, **kwargs):
            return SimpleNamespace(model=self, observed_tts=self.init_tts(), device=device)

    host = Host()
    runtime = _runtime_with_host(host)
    duplex = runtime.as_duplex(device="cpu")

    assert duplex.observed_tts.name == "tokenizer"
    assert duplex.model is host
