from __future__ import annotations

from pathlib import Path

import numpy as np


def test_media_module_exposes_shared_media_audio_utilities() -> None:
    from examples_merak.llm.minicpm_o_4_5 import media_utils

    assert callable(getattr(media_utils, "video_chunks", None))
    assert callable(getattr(media_utils, "write_wav", None))
    assert callable(getattr(media_utils, "flatten_audio_chunks", None))


def test_model_media_module_keeps_model_specific_video_normalization() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import media

    assert callable(getattr(media, "normalize_minicpmo_video", None))
    # Shared streaming-demo media utilities live in examples_merak media_utils,
    # not in the model package.
    assert not hasattr(media, "video_chunks")
    assert not hasattr(media, "write_wav")
    assert not hasattr(media, "flatten_audio_chunks")


def test_media_write_wav_produces_16bit_mono_pcm(tmp_path) -> None:
    import wave

    from examples_merak.llm.minicpm_o_4_5.media_utils import write_wav

    waveform = np.array([0.0, 0.5, -0.5, 1.5, -1.5], dtype=np.float32)
    out = tmp_path / "out.wav"
    write_wav(out, waveform, 16000)

    with wave.open(str(out), "rb") as handle:
        assert handle.getnchannels() == 1
        assert handle.getsampwidth() == 2
        assert handle.getframerate() == 16000
        assert handle.getnframes() == 5


def test_media_flatten_audio_chunks_concatenates() -> None:
    from examples_merak.llm.minicpm_o_4_5.media_utils import flatten_audio_chunks

    flat = flatten_audio_chunks([np.array([1.0, 2.0]), np.array([3.0])])
    assert np.array_equal(flat, np.array([1.0, 2.0, 3.0]))


def test_hf_compatible_module_exposes_consolidated_compat_patches() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import hf_compatible

    assert callable(getattr(hf_compatible, "patch_dynamic_cache_legacy_methods", None))
    assert callable(getattr(hf_compatible, "patch_remote_cache_helpers", None))
    assert callable(getattr(hf_compatible, "patch_empty_audio_cache", None))
    assert callable(getattr(hf_compatible, "patch_dynamic_cache_seen_tokens", None))


def test_patch_dynamic_cache_legacy_methods_is_superset_of_seen_tokens() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import (
        patch_dynamic_cache_legacy_methods,
    )

    class FakeCache:
        def get_seq_length(self, layer_idx=0):
            return 7

    patch_dynamic_cache_legacy_methods(FakeCache)
    assert FakeCache.seen_tokens.fget(FakeCache()) == 7
    assert FakeCache.get_usable_length(FakeCache(), 3) == 7
    assert hasattr(FakeCache, "key_cache")
    assert hasattr(FakeCache, "value_cache")


def test_hf_streaming_support_keeps_only_artifact_contract_helpers() -> None:
    import importlib.util

    root = Path("examples_merak/llm/minicpm_o_4_5")
    support = root / "minicpm_o_4_5_hf_streaming_support.py"
    assert support.is_file()

    spec = importlib.util.spec_from_file_location("minicpm_hf_streaming_support_layout", support)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Artifact-contract helpers remain.
    for name in (
        "utc_timestamp",
        "environment_versions",
        "FailurePacket",
        "make_failure_packet",
        "write_failure_packet",
        "write_json",
    ):
        assert hasattr(module, name), name

    # Media/audio utilities and compat patches have moved out.
    for name in ("video_chunks", "write_wav", "flatten_audio_chunks"):
        assert not hasattr(module, name), name
