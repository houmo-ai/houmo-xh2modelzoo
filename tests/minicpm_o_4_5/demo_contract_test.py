from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


DEMO_PATH = Path("examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_hmonnx_demo.py")
HF_DEMO_PATH = Path("examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_hf_demo.py")


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("minicpm_o_4_5_hmonnx_demo", DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_hf_demo_module():
    spec = importlib.util.spec_from_file_location("minicpm_o_4_5_hf_demo", HF_DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_demo_cli_accepts_tts_output_arguments(monkeypatch) -> None:
    demo = _load_demo_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(DEMO_PATH),
            "--model-dir",
            "/models/minicpm",
            "--work-dir",
            "/tmp/export",
            "--video",
            "/tmp/video.mp4",
            "--generate-audio",
            "--ref-audio",
            "/tmp/ref.wav",
            "--output-audio-path",
            "/tmp/output.wav",
            "--output-speech-tokens-path",
            "/tmp/speech.json",
        ],
    )

    args = demo.parse_args()

    assert args.generate_audio is True
    assert args.ref_audio == Path("/tmp/ref.wav")
    assert args.output_audio_path == Path("/tmp/output.wav")
    assert args.output_speech_tokens_path == Path("/tmp/speech.json")


def test_validate_audio_rejects_silence_and_accepts_non_silent_wav(tmp_path) -> None:
    demo = _load_demo_module()
    silent = tmp_path / "silent.wav"
    audible = tmp_path / "audible.wav"
    sf.write(silent, np.zeros(1600, dtype=np.float32), 16000)
    sf.write(audible, np.tile(np.array([0.25, -0.25], dtype=np.float32), 800), 16000)

    with pytest.raises(RuntimeError, match="invalid or silent WAV"):
        demo.validate_audio(silent)

    metrics = demo.validate_audio(audible)
    assert metrics["non_silent"] is True
    assert metrics["sample_rate"] == 16000
    assert metrics["duration"] == pytest.approx(0.1)


def test_prepare_output_paths_creates_parent_directories(tmp_path) -> None:
    demo = _load_demo_module()
    output_dir = tmp_path / "result"
    audio_path = output_dir / "audio" / "output.wav"
    tokens_path = output_dir / "tokens" / "speech.json"

    resolved_audio, resolved_tokens = demo.prepare_output_paths(output_dir, audio_path, tokens_path)

    assert resolved_audio == audio_path
    assert resolved_tokens == tokens_path
    assert output_dir.is_dir()
    assert audio_path.parent.is_dir()
    assert tokens_path.parent.is_dir()


def test_token2wav_asset_path_uses_model_directory() -> None:
    demo = _load_demo_module()

    path = demo.token2wav_asset_path(Path("/models/MiniCPM-o-4_5"))

    assert path == Path("/models/MiniCPM-o-4_5/assets/token2wav")


def test_normalize_video_uses_official_ffmpeg_path(monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import media

    calls: list[dict[str, object]] = []

    def helper(path: str, **kwargs):
        calls.append({"path": path, **kwargs})
        return ["frame"], ["audio"], None

    monkeypatch.setattr(media, "_load_official_video_helper", lambda: helper)

    contents = media.normalize_minicpmo_video(Path("/tmp/video.mp4"), include_audio=True, stack_frames=1)

    assert contents == ["frame", "audio"]
    assert [call["use_ffmpeg"] for call in calls] == [True]


def test_normalize_video_falls_back_when_ffprobe_is_unavailable(monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import media

    def helper(*args, **kwargs):
        del args, kwargs
        raise FileNotFoundError("ffprobe")

    monkeypatch.setattr(media, "_load_official_video_helper", lambda: helper)
    monkeypatch.setattr(
        media,
        "_moviepy_video_segments",
        lambda path, include_audio: ([path.name], ["audio"] if include_audio else None, None),
    )

    contents = media.normalize_minicpmo_video(Path("/tmp/video.mp4"), include_audio=True, stack_frames=1)

    assert contents == ["video.mp4", "audio"]


def test_official_video_helper_loads_without_importing_decord(monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import media

    real_decord = sys.modules.pop("decord", None)
    real_utils = sys.modules.pop("minicpmo.utils", None)
    monkeypatch.setitem(sys.modules, "decord", None)

    try:
        try:
            helper = media._load_official_video_helper()
        except ModuleNotFoundError as error:
            if error.name == "minicpmo":
                pytest.skip("official minicpmo package is not installed")
            raise
        assert sys.modules.get("decord") is None
    finally:
        sys.modules.pop("decord", None)
        sys.modules.pop("minicpmo.utils", None)
        if real_decord is not None:
            sys.modules["decord"] = real_decord
        if real_utils is not None:
            sys.modules["minicpmo.utils"] = real_utils

    assert helper.__name__ == "get_video_frame_audio_segments"


def test_hmonnx_demo_initializes_tts_before_attaching_runtime_even_for_text_only() -> None:
    demo = _load_demo_module()
    events: list[str] = []

    class Host:
        def init_tts(self, *, model_dir: str) -> None:
            events.append(f"init:{model_dir}")

    class Runtime:
        def __init__(self, work_dir: Path, host: Host) -> None:
            del work_dir, host
            events.append("runtime")

    demo.initialize_hmonnx_host(
        Host(),
        model_dir=Path("/models/MiniCPM-o-4_5"),
        work_dir=Path("/tmp/export"),
        runtime_type=Runtime,
    )

    assert events == ["init:/models/MiniCPM-o-4_5/assets/token2wav", "runtime"]


def test_hf_demo_prepares_audio_attention_compatibility_before_chat() -> None:
    demo = _load_hf_demo_module()
    events: list[str] = []

    class Model:
        pass

    model = Model()
    prepared = demo.prepare_hf_model(
        model,
        cache_type=Model,
        patch_dynamic_cache=lambda candidate: events.append("cache") if candidate is Model else None,
        patch_audio_attention=lambda candidate: events.append("patched") if candidate is model else None,
    )

    assert prepared is model
    assert events == ["cache", "patched"]

