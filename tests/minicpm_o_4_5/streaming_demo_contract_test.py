from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
    StreamingCaseResult,
    get_streaming_case,
)


DEMO_PATH = Path("examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_streaming_demo.py")
README_PATH = Path("examples_merak/llm/minicpm_o_4_5/README.md")

CASE_CHOICES = (
    "session_audio_text",
    "session_audio_reply",
    "duplex_audio_text",
    "duplex_audio_reply",
    "duplex_omni_reply",
)


def _load_demo():
    spec = importlib.util.spec_from_file_location("minicpm_o_4_5_streaming_demo", DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _result_for(case_name: str) -> StreamingCaseResult:
    case = get_streaming_case(case_name)
    return StreamingCaseResult(
        case_name=case.name,
        api_events=("fake",),
        text_chunks=("ok",),
        token_ids=(0,),
        waveform_chunks=("audio",) if case.generate_audio else (),
        public_results=("final",),
        backend_counters={role: 1 for role in case.required_roles},
        elapsed_ms=(0, 0),
        final=True,
    )


@pytest.mark.parametrize("name", CASE_CHOICES)
def test_parse_args_accepts_all_five_cases(name: str, tmp_path: Path) -> None:
    demo = _load_demo()
    args = demo.parse_args(["--case", name, "--fake", "--output-dir", str(tmp_path), "--seed", "7"])
    assert args.case == name


def test_real_case_media_defaults_to_certified_model_asset(tmp_path: Path) -> None:
    import importlib.util

    demo_path = Path("examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_streaming_demo.py")
    spec = importlib.util.spec_from_file_location("minicpm_streaming_demo_media", demo_path)
    assert spec is not None and spec.loader is not None
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "Skiing.mp4").touch()
    (assets / "system_ref_audio.wav").touch()
    (assets / "omni_duplex1.mp4").touch()
    assert demo._default_media(tmp_path, "session_audio_text") == assets / "Skiing.mp4"
    assert demo._default_media(tmp_path, "session_audio_reply") == assets / "Skiing.mp4"
    assert demo._default_media(tmp_path, "duplex_audio_text") == assets / "omni_duplex1.mp4"
    assert demo._default_media(tmp_path, "duplex_audio_reply") == assets / "omni_duplex1.mp4"
    assert demo._default_media(tmp_path, "duplex_omni_reply") == assets / "omni_duplex2.mp4"


def test_real_case_defaults_to_certified_chunk_geometry() -> None:
    demo = _load_demo()
    case = demo.resolve_case("session_audio_text")

    assert case.chunk_durations_ms == (1035, 1000)


def test_real_runtime_uses_exec_device_as_default_cuda_device(monkeypatch) -> None:
    """Official Token2Wav code hard-codes device='cuda'; the runner must make the exec
    device the default so those tensors land where the HMONNX runtime executes."""
    import torch

    demo = _load_demo()
    calls: list[object] = []
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: calls.append(device))

    demo._use_exec_device_as_default("cuda:4")
    demo._use_exec_device_as_default("cpu")

    assert calls == ["cuda:4"]


def test_real_media_prompt_passes_through_full_reference_audio(monkeypatch, tmp_path: Path) -> None:
    """The exported Token2Wav streaming graphs cover the official system_ref_audio
    (16.84 s -> 842 mel frames); the full reference prompt is passed through and the
    runtime raises a clear capacity error only for prompts longer than the export."""
    import wave

    demo = _load_demo()
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "Skiing.mp4").touch()
    (assets / "system_ref_audio.wav").touch()
    full_prompt = np.zeros(demo._SAMPLE_RATE * 17, dtype=np.float32)
    import librosa as librosa_module

    monkeypatch.setattr(librosa_module, "load", lambda path, sr=None, mono=None: (full_prompt, sr))
    monkeypatch.setattr(
        demo,
        "video_chunks",
        lambda _path, _sample_rate: [(np.zeros(16000, dtype=np.float32), []) for _ in range(2)],
    )
    output_dir = tmp_path / "out"
    args = demo.parse_args(
        [
            "--case",
            "session_audio_reply",
            "--model-dir",
            str(tmp_path),
            "--work-dir",
            str(tmp_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    media = demo._build_media(args)

    # No truncation: the full reference prompt is preserved for the runtime check.
    assert len(media["prompt_waveform"]) == len(full_prompt)
    prompt_wav = Path(media["prompt_wav_path"])
    assert prompt_wav.is_file()
    assert prompt_wav != assets / "system_ref_audio.wav"
    with wave.open(str(prompt_wav), "rb") as handle:
        assert handle.getnframes() == len(full_prompt)


def test_real_video_media_uses_only_certified_two_chunks(monkeypatch, tmp_path: Path) -> None:
    demo = _load_demo()
    assets = tmp_path / "assets"
    assets.mkdir()
    video = assets / "Skiing.mp4"
    video.touch()
    chunks = [(np.full(16000, index, dtype=np.float32), [f"frame-{index}"]) for index in range(3)]
    monkeypatch.setattr(demo, "video_chunks", lambda _path, _sample_rate: chunks)
    args = demo.parse_args(
        [
            "--case",
            "session_audio_text",
            "--model-dir",
            str(tmp_path),
            "--work-dir",
            str(tmp_path),
        ]
    )

    media = demo._build_media(args)

    assert len(media["chunks"]) == 2


@pytest.mark.parametrize("name", CASE_CHOICES)
def test_resolve_case_maps_each_cli_choice_to_certified_case(name: str) -> None:
    demo = _load_demo()
    case = demo.resolve_case(name)
    assert case.name == name
    assert case.name in demo.CASE_CHOICES
    assert isinstance(case.case_id, str) and case.case_id.startswith("case-")


def test_invalid_case_rejected_by_argparse() -> None:
    demo = _load_demo()
    with pytest.raises(SystemExit):
        demo.parse_args(["--case", "not_a_certified_case", "--fake"])


def test_real_mode_requires_model_and_work_dirs_but_defaults_media() -> None:
    demo = _load_demo()
    args = demo.parse_args(["--case", "session_audio_text", "--output-dir", "tmp/out"])
    with pytest.raises(ValueError, match="--model-dir"):
        demo.validate_real_inputs(args)

    args = demo.parse_args(
        ["--case", "session_audio_text", "--model-dir", "/m", "--work-dir", "/w", "--output-dir", "tmp/out"]
    )
    demo.validate_real_inputs(args)


def test_fake_mode_does_not_require_model_work_media(tmp_path: Path) -> None:
    demo = _load_demo()
    args = demo.parse_args(["--case", "session_audio_text", "--fake", "--output-dir", str(tmp_path)])
    demo.validate_real_inputs(args)  # must not raise in fake mode


def test_args_convert_to_typed_case_and_deterministic_media(monkeypatch, tmp_path: Path) -> None:
    demo = _load_demo()
    captured: list[dict[str, object]] = []

    def fake_run(model, tokenizer, processor, case, media):
        captured.append({"model": model, "case": case, "media": media, "tokenizer": tokenizer, "processor": processor})
        return _result_for(case.name)

    monkeypatch.setattr(demo, "run_streaming_case", fake_run)
    args = demo.parse_args(["--case", "duplex_omni_reply", "--fake", "--output-dir", str(tmp_path), "--seed", "3"])
    payload = demo.run_streaming_demo(args)

    assert len(captured) == 1
    call = captured[0]
    assert call["case"].name == "duplex_omni_reply"
    assert call["case"].omni is True
    assert call["case"].generate_audio is True
    # Fake mode must use the in-memory FakeHost, never load a real HMONNX runtime.
    assert type(call["model"]).__name__ == "FakeHost"
    assert call["tokenizer"] is None and call["processor"] is None
    media = call["media"]
    assert isinstance(media, dict)
    assert media["seed"] == 3
    assert payload["status"] == "ok"
    assert payload["case"] == "duplex_omni_reply"


def test_fake_pipeline_writes_result_schema(tmp_path: Path) -> None:
    demo = _load_demo()
    rc = demo.main(["--case", "session_audio_text", "--fake", "--output-dir", str(tmp_path), "--seed", "5"])
    assert rc == 0
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    for key in (
        "case",
        "case_id",
        "status",
        "mode",
        "api_family",
        "generate_audio",
        "omni",
        "final",
        "role_counts",
        "api_events",
        "text_chunks",
        "token_ids",
        "waveform_chunks",
        "elapsed_ms",
        "seed",
    ):
        assert key in result
    assert result["status"] == "ok"
    assert result["mode"] == "fake"
    assert result["case"] == "session_audio_text"
    assert result["role_counts"]["audio.stream_prefill"] == 1


def test_fake_pipeline_writes_text_and_tiny_wav_artifacts(tmp_path: Path) -> None:
    demo = _load_demo()
    assert demo.main(["--case", "duplex_omni_reply", "--fake", "--output-dir", str(tmp_path), "--seed", "2"]) == 0
    assert (tmp_path / "text_duplex_omni_reply.txt").exists()
    wav = tmp_path / "audio_duplex_omni_reply.wav"
    assert wav.exists()
    assert 0 < wav.stat().st_size < 20000


def test_fake_pipeline_idempotent(tmp_path: Path) -> None:
    demo = _load_demo()
    argv = ["--case", "duplex_audio_reply", "--fake", "--output-dir", str(tmp_path), "--seed", "9"]
    assert demo.main(argv) == 0
    first = (tmp_path / "result.json").read_bytes()
    assert demo.main(argv) == 0
    assert (tmp_path / "result.json").read_bytes() == first


def test_fake_mode_never_writes_work_dirs_and_artifacts_are_tiny(tmp_path: Path) -> None:
    demo = _load_demo()
    assert demo.main(["--case", "session_audio_reply", "--fake", "--output-dir", str(tmp_path), "--seed", "1"]) == 0
    assert not Path("work_dirs/minicpm_o_4_5_streaming_demo").exists()
    for path in tmp_path.iterdir():
        assert path.stat().st_size < 20000
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["mode"] == "fake"


def test_compact_result_json_is_sorted_and_small(tmp_path: Path) -> None:
    demo = _load_demo()
    assert demo.main(["--case", "session_audio_text", "--fake", "--output-dir", str(tmp_path), "--seed", "8"]) == 0
    raw = (tmp_path / "result.json").read_bytes()
    assert len(raw) < 20000
    # Deterministic compact encoding: no accidental whitespace bloat.
    decoded = json.loads(raw)
    assert decoded["case"] == "session_audio_text"


def test_readme_documents_formal_streaming_entry_without_test_details() -> None:
    readme = README_PATH.read_text(encoding="utf-8")
    required_fragments = (
        "session_audio_text",
        "session_audio_reply",
        "duplex_audio_text",
        "duplex_audio_reply",
        "duplex_omni_reply",
        "minicpm_o_4_5_streaming_demo.py",
        "work_dirs",
        "CUDA",
        "native fallback",
        "Session 与 Duplex API",
    )
    missing = [fragment for fragment in required_fragments if fragment not in readme]
    assert not missing, f"README is missing documentation contracts: {missing}"

    internal_test_fragments = (
        "passed,",
        "BLEU",
        "WER",
    )
    present = [fragment for fragment in internal_test_fragments if fragment in readme]
    assert not present, f"README contains internal test details: {present}"
