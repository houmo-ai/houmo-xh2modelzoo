from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch
from streaming_audio_runtime_test import _runtime_with_fake_sessions


OFFICIAL_ROOT = Path(os.environ["MINICPM_O45_MODEL_DIR"]) if os.environ.get("MINICPM_O45_MODEL_DIR") else None


def _require_official_root() -> Path:
    if OFFICIAL_ROOT is None:
        pytest.skip("set MINICPM_O45_MODEL_DIR to run official remote-code tests")
    module_path = OFFICIAL_ROOT / "modeling_minicpmo.py"
    if not module_path.is_file():
        pytest.skip(f"official MiniCPM-o-4.5 source not found: {module_path}")
    return OFFICIAL_ROOT


def test_official_audio_encoder_method_is_used_when_importable() -> None:
    module_path = _require_official_root() / "modeling_minicpmo.py"
    package = ModuleType("minicpmo_official")
    package.__path__ = [str(module_path.parent)]
    sys.modules["minicpmo_official"] = package
    spec = importlib.util.spec_from_file_location(
        "minicpmo_official.modeling_minicpmo",
        module_path,
        submodule_search_locations=[str(module_path.parent)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    official = module.MiniCPMO.get_audio_embedding_streaming

    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import (
        create_audio_wraped_cls,
        create_minicpo_wraped_cls,
    )

    runtime, calls = _runtime_with_fake_sessions()

    class AudioBase:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class HostBase:
        pass

    apm = object.__new__(create_audio_wraped_cls(AudioBase))
    apm._audio_model = runtime
    apm.embed_positions = SimpleNamespace(weight=torch.zeros((8, 5)))
    apm.conv1 = SimpleNamespace(weight=torch.zeros((), dtype=torch.float16))
    host = object.__new__(create_minicpo_wraped_cls(HostBase))
    host.apm = apm
    host.audio_past_key_values = None
    host.audio_encoder_layer = -1
    projection_calls: list[torch.Tensor] = []
    pooling_calls: list[torch.Tensor] = []

    def project(value: torch.Tensor) -> torch.Tensor:
        projection_calls.append(value.clone())
        return value + 10

    def pool(value: torch.Tensor) -> torch.Tensor:
        pooling_calls.append(value.clone())
        return value

    host.audio_projection_layer = project
    host.audio_avg_pooler = pool
    host._get_feat_extract_output_lengths = lambda lengths: (lengths, lengths)
    host._old_get_audio_embedding_streaming = MethodType(official, host)

    result = host.get_audio_embedding_streaming(
        {
            "audio_features": torch.ones((1, 80, 7), dtype=torch.float16),
            "audio_feature_lens": [torch.tensor([7])],
        },
        use_extra_context=True,
        prefix_extra_frames=0,
        suffix_extra_frames=2,
        cnn_min_length=9,
    )

    assert calls[-1] == ("stream_prefill", 7, 0)
    assert len(projection_calls) == 1
    assert len(pooling_calls) == 1
    # The streaming graph already includes the official pool_step=5 downsampling.
    # With the fake prefill geometry this call therefore exposes one pooled frame.
    expected_raw = torch.arange(20, dtype=torch.float16).reshape(1, 4, 5)[:, :1, :]
    assert torch.equal(result[0][0], expected_raw[0] + 10)


def test_hf_conversion_preserves_original_streaming_audio_method() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import MiniCPMOHFCompatible

    class Host:
        def get_audio_embedding(self, data, **kwargs):
            return data, kwargs

        def get_audio_embedding_streaming(self, data, **kwargs):
            return data, kwargs

    class Audio:
        pass

    original = Host.get_audio_embedding_streaming
    host = Host()
    host.apm = Audio()

    MiniCPMOHFCompatible.to_hf_compatible(host, audio_model=SimpleNamespace())

    assert host._old_get_audio_embedding_streaming.__func__ is original


def test_official_session_audio_forwards_no_extra_context_to_runtime() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import (
        create_audio_wraped_cls,
        create_minicpo_wraped_cls,
    )

    received: list[tuple[bool, int]] = []

    class AudioBase:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class HostBase:
        pass

    class AudioRuntime:
        streaming_cache_length = 0
        prefix_overlap_first = 0
        prefix_overlap_later = 2
        suffix_overlap = 2

        def forward_streaming(self, input_features, **kwargs):
            received.append((kwargs["use_extra_context"], int(kwargs["valid_mel_length"])))
            hidden = torch.zeros((1, 3, 5), dtype=torch.float16)
            return SimpleNamespace(hidden_states=(hidden,), last_hidden_state=hidden, past_key_values=None)

    apm = object.__new__(create_audio_wraped_cls(AudioBase))
    apm._audio_model = AudioRuntime()
    apm.conv1 = SimpleNamespace(weight=torch.zeros((), dtype=torch.float16))
    host = object.__new__(create_minicpo_wraped_cls(HostBase))
    host.apm = apm
    host.audio_past_key_values = None
    host.audio_encoder_layer = -1
    host.audio_projection_layer = lambda value: value
    host.audio_avg_pooler = lambda value: value
    host._get_feat_extract_output_lengths = lambda lengths: (lengths, lengths)
    host._old_get_audio_embedding_streaming = MethodType(
        lambda self, data, **kwargs: [
            [
                self.apm(
                    data["audio_features"],
                    use_cache=True,
                    output_hidden_states=True,
                    past_key_values=self.audio_past_key_values,
                    **kwargs,
                ).hidden_states[-1]
            ]
        ],
        host,
    )

    host.get_audio_embedding_streaming(
        {
            "audio_features": torch.ones((1, 80, 5), dtype=torch.float16),
            "audio_feature_lens": [torch.tensor([3])],
        },
        use_extra_context=False,
    )

    assert received == [(False, 3)]


def test_official_streaming_audio_allows_calls_without_audio_features() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import create_minicpo_wraped_cls

    class HostBase:
        pass

    host = object.__new__(create_minicpo_wraped_cls(HostBase))
    host.apm = SimpleNamespace()
    host._old_get_audio_embedding_streaming = lambda data, **kwargs: (data, kwargs)

    data, kwargs = host.get_audio_embedding_streaming(
        {"audio_features": [], "audio_feature_lens": []},
        use_extra_context=False,
    )

    assert data == {"audio_features": [], "audio_feature_lens": []}
    assert kwargs["use_extra_context"] is False
    assert not hasattr(host.apm, "_streaming_audio_feature_lens")


def test_official_streaming_audio_preserves_processor_length_for_padded_features() -> None:
    module_path = _require_official_root() / "modeling_minicpmo.py"
    package = ModuleType("minicpmo_official_padded")
    package.__path__ = [str(module_path.parent)]
    sys.modules["minicpmo_official_padded"] = package
    spec = importlib.util.spec_from_file_location(
        "minicpmo_official_padded.modeling_minicpmo",
        module_path,
        submodule_search_locations=[str(module_path.parent)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import (
        create_audio_wraped_cls,
        create_minicpo_wraped_cls,
    )

    runtime, calls = _runtime_with_fake_sessions()

    class AudioBase:
        def __call__(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    class HostBase:
        pass

    apm = object.__new__(create_audio_wraped_cls(AudioBase))
    apm._audio_model = runtime
    apm.embed_positions = SimpleNamespace(weight=torch.zeros((8, 5)))
    apm.conv1 = SimpleNamespace(weight=torch.zeros((), dtype=torch.float16))
    host = object.__new__(create_minicpo_wraped_cls(HostBase))
    host.apm = apm
    host.audio_past_key_values = None
    host.audio_encoder_layer = -1
    host.audio_projection_layer = lambda value: value
    host.audio_avg_pooler = lambda value: value
    host._get_feat_extract_output_lengths = lambda lengths: (lengths, lengths)
    host._old_get_audio_embedding_streaming = MethodType(module.MiniCPMO.get_audio_embedding_streaming, host)

    host.get_audio_embedding_streaming(
        {
            "audio_features": torch.ones((1, 80, 7), dtype=torch.float16),
            "audio_feature_lens": [torch.tensor([5])],
        },
        use_extra_context=True,
        prefix_extra_frames=0,
        suffix_extra_frames=2,
    )
    host.get_audio_embedding_streaming(
        {
            "audio_features": torch.ones((1, 80, 6), dtype=torch.float16),
            "audio_feature_lens": [torch.tensor([6])],
        },
        use_extra_context=True,
        prefix_extra_frames=2,
        suffix_extra_frames=2,
    )

    assert [call[1] for call in calls] == [5, 6]
