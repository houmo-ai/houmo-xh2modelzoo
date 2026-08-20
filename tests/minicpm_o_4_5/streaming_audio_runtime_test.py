from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers.modeling_outputs import BaseModelOutputWithPast

from xhquant.core import CacheTensor


def _runtime_with_fake_sessions():
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_audio import (
        MiniCPMO45AudioHMONNXRuntime,
    )
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_cache import FixedCapacityKVCacheMixin
    from xhmodel_merak.xh_llm.types import KVCacheConfig

    calls: list[tuple[str, int, int]] = []
    runtime = object.__new__(MiniCPMO45AudioHMONNXRuntime)
    runtime._models = {}
    runtime._device = torch.device("cpu")
    runtime._dtype = torch.float16
    runtime._valid_devices = [torch.device("cpu")]
    runtime.num_hidden_layers = 2
    runtime.streaming_enabled = True
    runtime.audio_encoder_layer = -1
    runtime.kv_cache_shape = (1, 2, 8, 4)
    # Keep the synthetic graph geometry large enough for the exported
    # AvgPool1d(pool_step=5) contract while using short real inputs below.
    runtime.stream_prefill_frames = 15
    runtime.stream_decode_frames = 14
    runtime.prefix_overlap_first = 0
    runtime.prefix_overlap_later = 2
    runtime.suffix_overlap = 2
    runtime.pool_step = 5
    runtime._kvcache_mixin = FixedCapacityKVCacheMixin(
        KVCacheConfig(
            num_layers=runtime.num_hidden_layers,
            kv_cache_shape=list(runtime.kv_cache_shape),
            cache_axis=2,
            batch_size=1,
            cache_dtype="float16",
            use_cache=True,
        ),
        "audio",
    )
    runtime._kvcache_mixin.prepare_fixed_cache("cpu")
    runtime.session = SimpleNamespace(to=lambda _device: None)

    def session(role: str):
        def run(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
            valid_mel_length = int(inputs[1].item())
            past_length = int(inputs[2].item())
            calls.append((role, valid_mel_length, past_length))
            prefix = runtime.prefix_overlap_first if role == "stream_prefill" else runtime.prefix_overlap_later
            current_length = (valid_mel_length + 1) // 2 - (prefix + 1) // 2 - (runtime.suffix_overlap + 1) // 2
            embeddings = torch.arange(20, dtype=torch.float16).reshape(1, 4, 5)
            present_length = past_length + current_length
            present_keys = tuple(
                torch.full(runtime.kv_cache_shape, float(index + 1), dtype=torch.float16)
                for index in range(runtime.num_hidden_layers)
            )
            present_values = tuple(
                torch.full(runtime.kv_cache_shape, float(index + 11), dtype=torch.float16)
                for index in range(runtime.num_hidden_layers)
            )
            assert present_length <= runtime.cache_capacity
            return (embeddings, *present_keys, *present_values)

        return run

    prefill = session("stream_prefill")
    decode = session("stream_decode")
    prefill.to = lambda _device: None
    decode.to = lambda _device: None
    runtime.stream_prefill_session = prefill
    runtime.stream_decode_session = decode
    runtime.session_prefill_session = None
    runtime.session_decode_session = None
    runtime.session_prefill_frames = runtime.stream_prefill_frames
    runtime.session_decode_frames = runtime.stream_decode_frames
    return runtime, calls


def test_forward_streaming_routes_first_then_later_chunks_and_advances_cache() -> None:
    runtime, calls = _runtime_with_fake_sessions()

    first = runtime.forward_streaming(
        torch.ones((1, 80, 7), dtype=torch.float32),
        valid_mel_length=torch.tensor([7], dtype=torch.int32),
    )
    later = runtime.forward_streaming(
        torch.ones((1, 80, 6), dtype=torch.float32),
        valid_mel_length=torch.tensor([6], dtype=torch.int32),
        past_key_values=first.past_key_values,
    )

    assert calls == [("stream_prefill", 7, 0), ("stream_decode", 6, 3)]
    assert runtime.streaming_cache_length == 4
    assert isinstance(first, BaseModelOutputWithPast)
    assert first.last_hidden_state.shape == (1, 1, 5)
    assert len(first.hidden_states) == 1
    assert first.hidden_states[-1] is first.last_hidden_state
    assert first.past_key_values is runtime.hf_cache
    assert later.past_key_values is runtime.hf_cache
    assert torch.all(runtime.past_key_caches[1][:, :, :4, :] == 2)
    assert torch.all(runtime.past_value_caches[0][:, :, :4, :] == 11)


def test_forward_streaming_routes_session_geometry_without_extra_context() -> None:
    runtime, calls = _runtime_with_fake_sessions()
    runtime.session_prefill_frames = 15
    runtime.session_decode_frames = 15

    def session(role: str):
        def run(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
            valid_mel_length = int(inputs[1].item())
            past_length = int(inputs[2].item())
            calls.append((role, valid_mel_length, past_length))
            current_length = int(inputs[3].item())
            embeddings = torch.arange(20, dtype=torch.float16).reshape(1, 4, 5)
            present_keys = tuple(
                torch.full(runtime.kv_cache_shape, float(index + 1), dtype=torch.float16)
                for index in range(runtime.num_hidden_layers)
            )
            present_values = tuple(
                torch.full(runtime.kv_cache_shape, float(index + 11), dtype=torch.float16)
                for index in range(runtime.num_hidden_layers)
            )
            assert past_length + current_length <= runtime.cache_capacity
            return (embeddings, *present_keys, *present_values)

        return run

    runtime.session_prefill_session = session("session_prefill")
    runtime.session_decode_session = session("session_decode")

    first = runtime.forward_streaming(
        torch.ones((1, 80, 5), dtype=torch.float16),
        valid_mel_length=5,
        use_extra_context=False,
    )
    runtime.forward_streaming(
        torch.ones((1, 80, 5), dtype=torch.float16),
        valid_mel_length=5,
        past_key_values=first.past_key_values,
        use_extra_context=False,
    )

    assert calls == [("session_prefill", 5, 0), ("session_decode", 5, 3)]
    assert runtime.streaming_cache_length == 6


def test_forward_streaming_keeps_exported_query_capacity_for_host_pooling() -> None:
    runtime, _ = _runtime_with_fake_sessions()

    output = runtime.forward_streaming(
        torch.ones((1, 80, 5), dtype=torch.float16),
        valid_mel_length=5,
    )

    assert output.last_hidden_state.shape == (1, 1, 5)


def test_forward_streaming_rejects_non_default_geometry_and_batch() -> None:
    runtime, _ = _runtime_with_fake_sessions()

    with pytest.raises(RuntimeError, match=r"batch_size=1"):
        runtime.forward_streaming(
            torch.ones((2, 80, 7), dtype=torch.float16),
            valid_mel_length=torch.tensor([7, 7], dtype=torch.int32),
        )
    short = runtime.forward_streaming(
        torch.ones((1, 80, 5), dtype=torch.float16),
        valid_mel_length=torch.tensor([5], dtype=torch.int32),
    )
    assert short.last_hidden_state.shape == (1, 1, 5)


def test_forward_streaming_resets_at_capacity_and_on_explicit_empty_cache() -> None:
    runtime, calls = _runtime_with_fake_sessions()
    runtime.kv_cache_shape = (1, 2, 4, 4)
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_cache import FixedCapacityKVCacheMixin
    from xhmodel_merak.xh_llm.types import KVCacheConfig

    runtime._kvcache_mixin = FixedCapacityKVCacheMixin(
        KVCacheConfig(
            num_layers=runtime.num_hidden_layers,
            kv_cache_shape=list(runtime.kv_cache_shape),
            cache_axis=2,
            batch_size=1,
            cache_dtype="float16",
            use_cache=True,
        ),
        "audio",
    )
    runtime._kvcache_mixin.prepare_fixed_cache("cpu")
    runtime.reset_state()

    first = runtime.forward_streaming(torch.ones((1, 80, 7), dtype=torch.float16), valid_mel_length=7)
    runtime.forward_streaming(
        torch.ones((1, 80, 5), dtype=torch.float16), valid_mel_length=5, past_key_values=first.past_key_values
    )
    assert [role for role, _, _ in calls] == ["stream_prefill", "stream_prefill"]

    # A dropped cache with a populated streaming cache is a mid-stream reset: the chunk is
    # still a later (decode-geometry) chunk, so it routes to the decode graph.
    runtime.forward_streaming(torch.ones((1, 80, 5), dtype=torch.float16), valid_mel_length=5)
    assert [role for role, _, _ in calls] == ["stream_prefill", "stream_prefill", "stream_decode"]


def test_forward_streaming_rejects_bad_present_cache_before_mutation() -> None:
    runtime, _ = _runtime_with_fake_sessions()
    original = runtime.stream_prefill_session

    def bad_session(*inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        outputs = list(original(*inputs))
        outputs[1] = torch.zeros((1, 1, 8, 4), dtype=torch.float16)
        return tuple(outputs)

    runtime.stream_prefill_session = bad_session
    with pytest.raises(RuntimeError, match=r"present_k_cache_0.*heads"):
        runtime.forward_streaming(torch.ones((1, 80, 7), dtype=torch.float16), valid_mel_length=7)
    assert runtime.streaming_cache_length == 0


def test_forward_streaming_rejects_non_final_selected_encoder_layer() -> None:
    runtime, _ = _runtime_with_fake_sessions()
    runtime.audio_encoder_layer = 0
    with pytest.raises(RuntimeError, match="audio_encoder_layer=-1"):
        runtime.forward_streaming(torch.ones((1, 80, 7), dtype=torch.float16), valid_mel_length=7)


def test_streaming_cache_moves_with_runtime_and_release_resets_view() -> None:
    runtime, _ = _runtime_with_fake_sessions()
    runtime.forward_streaming(torch.ones((1, 80, 7), dtype=torch.float16), valid_mel_length=7)
    live_cache = runtime.hf_cache
    runtime.to("cpu")
    assert all(value.device.type == "cpu" for value in runtime.past_key_caches)
    assert runtime.hf_cache is live_cache
    assert runtime.hf_cache.get_seq_length() == 3
    runtime.release_state()
    assert runtime.streaming_cache_length == 0
    assert live_cache.get_seq_length() == 0


def test_streaming_runtime_uses_hmonnx_cache_tensors() -> None:
    runtime, _ = _runtime_with_fake_sessions()

    assert all(isinstance(value, CacheTensor) for value in runtime.past_key_caches)
    assert all(isinstance(value, CacheTensor) for value in runtime.past_value_caches)


def test_set_exec_device_moves_native_audio_projection_modules() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import MiniCPMO45HMONNXRuntime

    moves: list[str] = []
    runtime = object.__new__(MiniCPMO45HMONNXRuntime)
    runtime.vision = SimpleNamespace(to=lambda device: None)
    runtime.audio = SimpleNamespace(to=lambda device: None)
    runtime.llm = SimpleNamespace(to=lambda device: None)
    runtime.tts = SimpleNamespace(to=lambda device: None)
    runtime.token2wav = SimpleNamespace(to=lambda device: None)
    runtime.host_model = SimpleNamespace(
        audio_projection_layer=SimpleNamespace(to=lambda device: moves.append(device)),
        audio_avg_pooler=SimpleNamespace(to=lambda device: moves.append(device)),
        resampler=SimpleNamespace(to=lambda device: moves.append(device)),
    )

    runtime.set_exec_device("cuda:0")

    assert moves == ["cuda:0", "cuda:0", "cuda:0"]


def test_runtime_accepts_offline_only_audio_metadata_and_rejects_streaming_clearly() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_audio import MiniCPMO45AudioHMONNXRuntime

    runtime = object.__new__(MiniCPMO45AudioHMONNXRuntime)
    runtime.stream_prefill_session = None
    runtime.stream_decode_session = None
    runtime.streaming_enabled = False
    with pytest.raises(RuntimeError, match="metadata/graphs"):
        runtime.forward_streaming(torch.ones((1, 80, 7), dtype=torch.float16), valid_mel_length=7)


def test_official_audio_method_keeps_full_signature_and_resets_at_capacity() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import (
        create_audio_wraped_cls,
        create_minicpo_wraped_cls,
    )

    runtime, calls = _runtime_with_fake_sessions()
    runtime.forward_streaming(
        torch.ones((1, 80, 7), dtype=torch.float16),
        valid_mel_length=torch.tensor([7], dtype=torch.int32),
    )

    class AudioBase:
        pass

    apm = object.__new__(create_audio_wraped_cls(AudioBase))
    apm._audio_model = runtime
    apm.embed_positions = SimpleNamespace(weight=torch.zeros((6, 4)))
    apm.conv1 = SimpleNamespace(weight=torch.zeros((), dtype=torch.float16))

    class HostBase:
        pass

    host = object.__new__(create_minicpo_wraped_cls(HostBase))
    host.apm = apm
    host.audio_past_key_values = runtime.hf_cache
    host.audio_encoder_layer = -1
    host.audio_projection_layer = lambda value: value
    host.audio_avg_pooler = lambda value: value
    host._get_feat_extract_output_lengths = lambda lengths: (lengths, lengths)
    received: list[tuple[bool, int, int, int | None]] = []

    def official(data, use_extra_context=False, prefix_extra_frames=1, suffix_extra_frames=1, cnn_min_length=None):
        received.append((use_extra_context, prefix_extra_frames, suffix_extra_frames, cnn_min_length))
        current_length = (data["audio_features"].shape[-1] - 1) // 2 + 1
        if host.audio_past_key_values is not None:
            cache_length = host.audio_past_key_values[0][0].shape[2]
            if cache_length + current_length >= apm.embed_positions.weight.shape[0]:
                host.audio_past_key_values = None
        output = apm.forward(
            data["audio_features"],
            past_key_values=host.audio_past_key_values,
            use_cache=True,
            output_hidden_states=True,
            use_extra_context=use_extra_context,
            prefix_extra_frames=prefix_extra_frames,
            suffix_extra_frames=suffix_extra_frames,
            cnn_min_length=cnn_min_length,
        )
        host.audio_past_key_values = output.past_key_values
        return [[output.hidden_states[-1]]]

    host._old_get_audio_embedding_streaming = official
    # A mid-stream reset chunk is a later (decode-geometry) chunk carrying the later prefix.
    data = {
        "audio_features": torch.ones((1, 80, 6), dtype=torch.float16),
        "audio_feature_lens": [[torch.tensor(6)]],
    }

    result = host.get_audio_embedding_streaming(
        data,
        use_extra_context=True,
        prefix_extra_frames=2,
        suffix_extra_frames=2,
        cnn_min_length=9,
    )

    assert received == [(True, 2, 2, 9)]
    assert calls[-1] == ("stream_decode", 6, 0)
    assert runtime.streaming_cache_length == 1
    assert result[0][0].shape == (1, 1, 5)


def test_audio_wrapper_keeps_offline_forward_path() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.hf_compatible import create_audio_wraped_cls

    class AudioBase:
        pass

    wrapped = object.__new__(create_audio_wraped_cls(AudioBase))
    expected = torch.ones((1, 2, 3), dtype=torch.float16)
    wrapped._audio_model = SimpleNamespace(
        forward=lambda input_features, attention_mask: expected,
        forward_streaming=lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("streaming path executed")),
    )

    result = wrapped.forward(
        torch.ones((1, 80, 4), dtype=torch.float16),
        attention_mask=torch.zeros((1, 1, 2, 2), dtype=torch.float16),
        use_cache=False,
    )

    assert result is expected
