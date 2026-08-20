from __future__ import annotations

from types import SimpleNamespace


def test_minicpm_does_not_define_a_parallel_runtime_lifecycle() -> None:
    from xhmodel_merak.xh_llm.hmonnx.hmonnx_model import HMONNXBaseModel, HMONNXModel
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_audio import MiniCPMO45AudioHMONNXRuntime
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import MiniCPMO45Token2WavHMONNXRuntime
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_vision import MiniCPMO45VisionHMONNXRuntime

    assert issubclass(MiniCPMO45VisionHMONNXRuntime, HMONNXModel)
    assert issubclass(MiniCPMO45AudioHMONNXRuntime, HMONNXBaseModel)
    assert issubclass(MiniCPMO45Token2WavHMONNXRuntime, HMONNXBaseModel)


def test_non_cached_component_does_not_expose_kv_cache_lifecycle() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_vision import MiniCPMO45VisionHMONNXRuntime

    assert not hasattr(MiniCPMO45VisionHMONNXRuntime, "reset_kvcache")
    assert not hasattr(MiniCPMO45VisionHMONNXRuntime, "cache_adapter")


def test_runtime_places_all_component_groups_on_one_device() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import MiniCPMO45HMONNXRuntime

    moves: list[tuple[str, str]] = []

    def component(name: str) -> SimpleNamespace:
        return SimpleNamespace(to=lambda device: moves.append((name, device)))

    def native(name: str) -> SimpleNamespace:
        return SimpleNamespace(to=lambda device: moves.append((name, device)))

    runtime = object.__new__(MiniCPMO45HMONNXRuntime)
    runtime.vision = component("vision")
    runtime.audio = component("audio")
    runtime.llm = component("llm")
    runtime.tts = component("tts")
    runtime.token2wav = component("token2wav")
    runtime.host_model = SimpleNamespace(
        audio_projection_layer=native("native_audio"),
        llm=SimpleNamespace(embed_tokens=native("native_llm")),
        tts=SimpleNamespace(emb_text=native("native_tts")),
    )

    runtime.set_exec_device("device")

    assert moves == [
        ("vision", "device"),
        ("audio", "device"),
        ("llm", "device"),
        ("tts", "device"),
        ("token2wav", "device"),
        ("native_audio", "device"),
        ("native_llm", "device"),
        ("native_tts", "device"),
    ]


def test_top_level_release_uses_the_same_component_lifecycle() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import MiniCPMO45HMONNXRuntime

    released: list[str] = []
    runtime = object.__new__(MiniCPMO45HMONNXRuntime)
    for name in ("vision", "audio", "llm", "tts", "token2wav"):
        setattr(runtime, name, SimpleNamespace(release_state=lambda name=name: released.append(name)))
    runtime._streaming_api_mode = "session"
    runtime._session_id = "id"

    runtime.release()

    assert released == ["vision", "audio", "llm", "tts", "token2wav"]
    assert runtime._streaming_api_mode is None
    assert runtime._session_id is None
