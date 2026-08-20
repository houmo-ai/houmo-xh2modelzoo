from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _flow_meta() -> dict[str, object]:
    return {
        "graphs": {
            "main": "flow.onnx",
            "stream_flow_frontend": "flow_frontend.onnx",
            "stream_flow_frontend_final": "flow_frontend_final.onnx",
            "stream_flow_estimator_step": "flow_estimator_step.onnx",
        },
        "stream_contract": {
            "prompt_token_capacity": 64,
            "prompt_mel_capacity": 80,
            "pre_lookahead_len": 3,
            "chunk_token_capacity": 28,
            "base_conformer_layers": 6,
            "cache_alignment": "right",
            "append_capacity": 56,
            "base_cache_valid_length": 50,
            "base_cache_shapes": {
                "conformer_cnn_cache": [1, 512, 6],
                "conformer_att_cache": [10, 1, 8, 150, 128],
                "estimator_cnn_cache": [16, 16, 2, 1024, 2],
                "estimator_att_cache": [16, 16, 2, 8, 150, 128],
            },
            "cache_shapes": {
                "conformer_cnn_cache": [1, 512, 6],
                "conformer_att_cache": [10, 1, 8, 150, 128],
                "estimator_cnn_cache": [16, 16, 2, 1024, 2],
                "estimator_att_cache": [16, 16, 2, 8, 150, 128],
            },
            "cache_capacity": 150,
            "cache_axes": {"conformer_att_cache": 3, "estimator_att_cache": 4},
            "prompt_cache_policy": {
                "leading_frame_count": "prompt_mel_length",
                "recent_mel_frames": 100,
                "lookahead_silence_token": 4218,
            },
            "valid_length_inputs": {
                "conformer_att_cache": "conformer_cache_valid_length",
                "estimator_att_cache": "estimator_cache_valid_length",
            },
            "valid_length_outputs": {
                "conformer_att_cache": "present_conformer_cache_valid_length",
                "estimator_att_cache": "present_estimator_cache_valid_length",
            },
            "frontend_output_names": [
                "mu",
                "spks",
                "present_conformer_cnn_cache",
                "present_conformer_att_cache",
                "present_conformer_cache_valid_length",
            ],
            "estimator_step_input_names": [
                "x_cfg",
                "mu_cfg",
                "t_cfg",
                "spks_cfg",
                "cond_cfg",
                "past_estimator_cnn_cache",
                "past_estimator_att_cache",
                "past_cache_valid_length",
                "current_frame_valid_length",
            ],
            "estimator_step_output_names": [
                "derivative_cfg",
                "present_estimator_cnn_cache",
                "present_estimator_att_cache",
            ],
        },
        "up_rate": 2,
        "n_timesteps": 1,
        "cfg_rate": 0.7,
    }


def _hift_meta() -> dict[str, object]:
    return {
        "graphs": {"stream_hift": "hift.onnx", "stream_hift_final": "hift_final.onnx"},
        "stream_contract": {
            "frame_capacity": 58,
            "mel_cache_length": 8,
            "source_cache_length": 4,
            "speech_cache_length": 4,
            "cache_axes": {"mel": 2, "source": 2},
            "first_valid_lengths": {"mel": 0, "source": 0, "speech": 0},
            "phase_noise_file": "phase.pt",
            "source_noise_file": "source.pt",
            "initial_source_cache_file": "initial.pt",
            "cache_inputs": {"mel": "past_mel", "source": "past_source"},
            "cache_outputs": {"mel": "present_mel", "source": "present_source"},
            "valid_length_outputs": {
                "mel": "present_mel_valid_length",
                "source": "present_source_valid_length",
            },
        },
        "hop_length": 2,
    }


def _components() -> dict[str, dict[str, object]]:
    return {
        "token2wav_flow_frontend": _flow_meta(),
        "token2wav_flow_decoder": _flow_meta(),
        "token2wav_hift": _hift_meta(),
    }


def test_streaming_role_loader_requires_decomposed_flow_graphs(tmp_path, monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    torch.save(torch.zeros(1, 1), tmp_path / "phase.pt")
    torch.save(torch.zeros(1, 1, 1), tmp_path / "source.pt")
    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    runtime._load_streaming_roles(tmp_path, _components(), lambda path: str(path))

    assert runtime.flow_frontend_session == str(tmp_path / "flow_frontend.onnx")
    assert runtime.flow_frontend_final_session == str(tmp_path / "flow_frontend_final.onnx")
    assert runtime.estimator_step_session == str(tmp_path / "flow_estimator_step.onnx")


def test_stream_cache_reset_restores_base_flow_and_empty_hift_state() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        MiniCPMO45Token2WavHMONNXRuntime,
    )

    runtime = object.__new__(MiniCPMO45Token2WavHMONNXRuntime)
    base = {
        name: torch.full(shape, index + 1.0)
        for index, (name, shape) in enumerate(
            (
                ("conformer_cnn_cache", (1, 2, 1)),
                ("conformer_att_cache", (1, 1, 1, 3, 1)),
                ("estimator_cnn_cache", (1, 1, 1, 2, 1)),
                ("estimator_att_cache", (1, 1, 1, 1, 3, 1)),
            )
        )
    }
    runtime._base_flow_cache = {name: value.clone() for name, value in base.items()}
    runtime._stream_state = runtime_token2wav.StreamState.from_base_cache(base, prompt_mel_length=5)
    runtime._stream_state.flow_cache["conformer_cnn_cache"].add_(10)
    runtime.reset_state()

    assert torch.equal(runtime._stream_state.flow_cache["conformer_cnn_cache"], base["conformer_cnn_cache"])
    assert all(value.shape[-1] == 0 for value in runtime._stream_state.hift_cache.values())


def test_reset_before_prompt_cache_initialization_is_a_noop() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        MiniCPMO45Token2WavHMONNXRuntime,
    )

    runtime = object.__new__(MiniCPMO45Token2WavHMONNXRuntime)
    runtime._base_flow_cache = None
    runtime._stream_state = None

    runtime.reset_stream_cache()

    assert runtime._stream_state is None


def test_stream_cfm_uses_ten_independent_estimator_cache_banks() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import run_stream_cfm

    calls: list[torch.Tensor] = []
    cnn = torch.zeros(10, 1, 1, 1, 1)
    att = torch.zeros(10, 1, 1, 1, 1, 1)

    def estimator_step(x_cfg, mu_cfg, t_cfg, spks_cfg, cond_cfg, past_cnn, past_att, _past_valid, current_valid):
        del mu_cfg, t_cfg, spks_cfg, cond_cfg, current_valid
        calls.append(past_cnn.clone())
        current_cache = past_att.new_ones((*past_att.shape[:3], x_cfg.shape[2], past_att.shape[4]))
        return (
            torch.cat((torch.ones_like(x_cfg[:1]), torch.zeros_like(x_cfg[:1]))),
            past_cnn + 1,
            torch.cat((current_cache, past_att + 1), dim=3),
        )

    output, present_cnn, present_att = run_stream_cfm(
        estimator_step,
        mu=torch.zeros(1, 1, 1),
        spks=torch.zeros(1, 1),
        cond=torch.zeros(1, 1, 1),
        noise=torch.zeros(1, 1, 1),
        n_timesteps=10,
        cfg_rate=0.7,
        estimator_cnn_banks=cnn,
        estimator_att_banks=att,
    )

    assert len(calls) == 10
    assert [int(value.reshape(()).item()) for value in calls] == [0] * 10
    assert torch.equal(present_cnn[:, 0, 0, 0, 0], torch.ones(10))
    assert torch.equal(present_att[:, 0, 0, 0, 0, 0], torch.ones(10))
    assert torch.allclose(output, torch.full_like(output, 1.7))


def test_stream_state_keeps_compact_att_caches_and_tracks_graph_capacity_separately() -> None:
    """Host state stays compact; fixed-capacity padding is only added at graph calls."""
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    base = {
        "conformer_cnn_cache": torch.zeros(1, 512, 6),
        "conformer_att_cache": torch.zeros(10, 1, 8, 842, 128),
        "estimator_cnn_cache": torch.zeros(16, 16, 2, 1024, 2),
        "estimator_att_cache": torch.zeros(16, 16, 2, 8, 842, 128),
    }
    state = runtime_token2wav.StreamState.from_base_cache(base, 842, att_cache_capacity=942)

    assert state.flow_cache["conformer_att_cache"].shape[3] == 842
    assert state.flow_cache["estimator_att_cache"].shape[4] == 842
    assert state.estimator_att_banks.shape[4] == 842
    assert state.att_cache_capacity == 942
    assert state.flow_valid_lengths == {"conformer_att_cache": 842, "estimator_att_cache": 842}

    # Chunk 1: prompt full + empty tail (nothing generated yet).
    offset = state.flow_valid_lengths["estimator_att_cache"]
    assert offset == 842
    # Chunk 2: valid length grows to 892, tail now carries 50 generated frames.
    offset = min(offset + 50, 942)
    assert offset == 892
    # Chunk 3+: valid length saturates at 942 = prompt 842 + recent 100 generated frames.
    offset = min(offset + 50, 942)
    assert offset == 942
    offset = min(offset + 50, 942)
    assert offset == 942


def test_streaming_attention_cache_keeps_prompt_prefix_and_recent_tail() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        bound_streaming_attention_cache,
    )

    cache = torch.arange(200, dtype=torch.float32).reshape(1, 1, 1, 200, 1)

    bounded = bound_streaming_attention_cache(cache, valid_length=150, capacity=150, prompt_length=50)

    assert bounded.shape == (1, 1, 1, 150, 1)
    assert bounded[0, 0, 0, :50, 0].tolist() == list(range(50))
    # prompt occupies the first 50 frames; the remaining 100 slots keep the
    # most recent 100 valid frames (50..149).
    assert bounded[0, 0, 0, 50:, 0].tolist() == list(range(50, 150))


@pytest.mark.parametrize("past_valid,current_valid", [(842, 50), (892, 50), (942, 56)])
def test_conformer_cache_pack_and_compact_preserve_base_and_up_layout(
    past_valid: int,
    current_valid: int,
) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        compact_conformer_attention_cache,
        pack_conformer_attention_cache,
    )

    base_layers = 6
    capacity = 942
    base_half = torch.arange(past_valid // 2, dtype=torch.float32).reshape(1, 1, 1, -1, 1)
    base = base_half.repeat(base_layers, 1, 1, 2, 1)
    up = torch.arange(past_valid, dtype=torch.float32).reshape(1, 1, 1, -1, 1).repeat(4, 1, 1, 1, 1) + 10_000
    logical = torch.cat((base, up), dim=0)

    packed = pack_conformer_attention_cache(
        logical,
        valid_length=past_valid,
        capacity=capacity,
        base_layer_count=base_layers,
    )

    assert packed.shape[3] == capacity
    assert torch.equal(packed[:base_layers, :, :, -past_valid // 2 :, :], base[:, :, :, : past_valid // 2, :])
    assert torch.equal(packed[base_layers:, :, :, -past_valid:, :], up)

    base_current = (
        torch.arange(current_valid // 2, dtype=torch.float32).reshape(1, 1, 1, -1, 1).repeat(base_layers, 1, 1, 1, 1)
        + 20_000
    )
    raw_base_half = torch.cat((packed[:base_layers, :, :, : capacity // 2, :], base_current), dim=3)
    raw_base = raw_base_half.repeat(1, 1, 1, 2, 1)
    up_current = torch.arange(current_valid, dtype=torch.float32).reshape(1, 1, 1, -1, 1).repeat(4, 1, 1, 1, 1) + 30_000
    raw_up = torch.cat((packed[base_layers:], up_current), dim=3)
    raw = torch.cat((raw_base, raw_up), dim=0)

    compact = compact_conformer_attention_cache(
        raw,
        past_valid_length=past_valid,
        current_valid_length=current_valid,
        input_capacity=capacity,
        base_layer_count=base_layers,
    )

    assert compact.shape[3] == past_valid + current_valid
    expected_base_half = torch.cat((base_half.repeat(base_layers, 1, 1, 1, 1), base_current), dim=3)
    assert torch.equal(compact[:base_layers], expected_base_half.repeat(1, 1, 1, 2, 1))
    assert torch.equal(compact[base_layers:, :, :, :past_valid, :], up)
    assert torch.equal(compact[base_layers:, :, :, past_valid:, :], up_current)


@pytest.mark.parametrize("past_valid,current_valid", [(842, 50), (892, 50), (942, 56)])
def test_estimator_cache_pack_and_compact_remove_both_padding_regions(
    past_valid: int,
    current_valid: int,
) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        compact_estimator_attention_cache,
        pack_streaming_attention_cache,
    )

    capacity = 942
    frame_capacity = 56
    logical = torch.arange(past_valid, dtype=torch.float32).reshape(1, 1, 1, -1, 1)
    packed = pack_streaming_attention_cache(logical, valid_length=past_valid, capacity=capacity, axis=3)
    current = torch.arange(frame_capacity, dtype=torch.float32).reshape(1, 1, 1, -1, 1) + 10_000
    raw = torch.cat((current, packed), dim=3)

    compact = compact_estimator_attention_cache(
        raw,
        past_valid_length=past_valid,
        current_valid_length=current_valid,
        input_capacity=capacity,
        frame_capacity=frame_capacity,
    )

    assert compact.shape[3] == past_valid + current_valid
    assert torch.equal(compact[:, :, :, :current_valid, :], current[:, :, :, :current_valid, :])
    assert torch.equal(compact[:, :, :, current_valid:, :], logical)


def test_streaming_cache_truncation_keeps_prompt_full_plus_recent_tail() -> None:
    """The official stream() truncation (prompt in full + recent 100) is mirrored
    on the stored caches so the next chunk feeds at most the graph capacity."""
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        _truncate_streaming_cache,
    )

    # 5-D conformer cache: 992 frames > capacity 942.
    conformer = torch.arange(992, dtype=torch.float32).reshape(1, 1, 1, 992, 1)
    truncated = _truncate_streaming_cache(conformer, prompt_length=842, tail=100, capacity=942)
    assert truncated.shape == (1, 1, 1, 942, 1)
    assert truncated[0, 0, 0, :842, 0].tolist() == list(range(842))
    assert truncated[0, 0, 0, 842:, 0].tolist() == list(range(892, 992))

    # 6-D estimator banks: same truncation on axis 4.
    banks = torch.arange(992, dtype=torch.float32).reshape(1, 1, 1, 1, 992, 1)
    truncated_banks = _truncate_streaming_cache(banks, prompt_length=842, tail=100, capacity=942)
    assert truncated_banks.shape == (1, 1, 1, 1, 942, 1)
    assert truncated_banks[0, 0, 0, 0, :842, 0].tolist() == list(range(842))
    assert truncated_banks[0, 0, 0, 0, 842:, 0].tolist() == list(range(892, 992))

    # Within capacity: unchanged.
    small = torch.arange(500, dtype=torch.float32).reshape(1, 1, 1, 500, 1)
    assert _truncate_streaming_cache(small, prompt_length=842, tail=100, capacity=942) is small


def test_present_valid_length_is_bounded_after_host_cache_truncation() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        _read_present_valid_length,
    )

    assert (
        _read_present_valid_length(
            torch.tensor([998], dtype=torch.int32),
            name="present_conformer_cache_valid_length",
            capacity=942,
        )
        == 942
    )


def test_streaming_attention_cache_rejects_valid_length_over_capacity() -> None:
    import pytest

    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        bound_streaming_attention_cache,
    )

    cache = torch.arange(200, dtype=torch.float32).reshape(1, 1, 1, 200, 1)
    with pytest.raises(RuntimeError, match="exceeds exported capacity"):
        bound_streaming_attention_cache(cache, valid_length=200, capacity=150, prompt_length=50)


def test_stream_cfm_pads_normal_role_to_estimator_capacity_and_crops_result() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import run_stream_cfm

    seen: list[torch.Size] = []

    def estimator_step(x_cfg, mu_cfg, t_cfg, spks_cfg, cond_cfg, past_cnn, past_att, _past_valid, _current_valid):
        del mu_cfg, t_cfg, spks_cfg, cond_cfg
        seen.append(x_cfg.shape)
        current_cache = past_att.new_zeros((*past_att.shape[:3], x_cfg.shape[2], past_att.shape[4]))
        return torch.zeros_like(x_cfg), past_cnn, torch.cat((current_cache, past_att), dim=3)

    output, _, _ = run_stream_cfm(
        estimator_step,
        mu=torch.zeros(1, 80, 50),
        spks=torch.zeros(1, 80),
        cond=torch.zeros(1, 80, 50),
        noise=torch.zeros(1, 80, 50),
        n_timesteps=1,
        cfg_rate=0.7,
        estimator_cnn_banks=torch.zeros(1, 1, 1, 1, 1),
        estimator_att_banks=torch.zeros(1, 1, 1, 1, 1, 1),
        frame_capacity=56,
    )

    assert seen == [torch.Size((2, 80, 56))]
    assert output.shape == (1, 80, 50)


def test_stream_cfm_pads_attention_cache_to_estimator_input_capacity() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import run_stream_cfm

    seen: list[torch.Size] = []

    def estimator_step(x_cfg, mu_cfg, t_cfg, spks_cfg, cond_cfg, past_cnn, past_att, _past_valid, _current_valid):
        del mu_cfg, t_cfg, spks_cfg, cond_cfg
        seen.append(past_att.shape)
        return torch.zeros_like(x_cfg), past_cnn, torch.zeros(1, 1, 1, 206, 1)

    run_stream_cfm(
        estimator_step,
        mu=torch.zeros(1, 1, 1),
        spks=torch.zeros(1, 1),
        cond=torch.zeros(1, 1, 1),
        noise=torch.zeros(1, 1, 1),
        n_timesteps=1,
        cfg_rate=0.7,
        estimator_cnn_banks=torch.zeros(1, 1, 1, 1, 1),
        estimator_att_banks=torch.zeros(1, 1, 1, 1, 50, 1),
        estimator_att_capacity=150,
    )

    assert seen == [torch.Size((1, 1, 1, 150, 1))]


def test_host_waveform_overlap_matches_previous_graph_policy() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        apply_streaming_waveform_overlap,
    )

    raw = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    past = torch.tensor([[10.0, 20.0, 30.0, 40.0]])
    window = torch.arange(1, 9, dtype=torch.float32)
    expected_head = raw[:, :4] * window[:4].reshape(1, -1) + past * window[4:].reshape(1, -1)
    expected = torch.cat((expected_head, raw[:, 4:]), dim=1)

    emitted, present, present_length = apply_streaming_waveform_overlap(
        raw,
        raw_valid_length=8,
        past_speech=past,
        past_speech_valid_length=4,
        speech_window=window,
        overlap=4,
        final=False,
    )

    assert torch.equal(emitted, expected[:, :-4])
    assert torch.equal(present, expected[:, -4:])
    assert present_length == 4


def test_host_waveform_overlap_matches_official_first_chunk_silence_policy() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        apply_streaming_waveform_overlap,
    )

    raw = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    emitted, present, present_length = apply_streaming_waveform_overlap(
        raw,
        raw_valid_length=8,
        past_speech=torch.zeros(1, 4),
        past_speech_valid_length=0,
        speech_window=torch.ones(8),
        overlap=4,
        final=False,
    )

    assert torch.equal(emitted, torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0]]))
    assert torch.equal(present, torch.tensor([[4.0, 5.0, 6.0, 7.0]]))
    assert present_length == 4


def test_host_waveform_overlap_flushes_full_final_waveform() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        apply_streaming_waveform_overlap,
    )

    raw = torch.arange(8, dtype=torch.float32).reshape(1, 8)
    window = torch.ones(8)

    emitted, present, present_length = apply_streaming_waveform_overlap(
        raw,
        raw_valid_length=8,
        past_speech=torch.ones(1, 4),
        past_speech_valid_length=4,
        speech_window=window,
        overlap=4,
        final=True,
    )

    assert torch.equal(emitted, torch.tensor([[1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0]]))
    assert torch.count_nonzero(present) == 0
    assert present_length == 0


def test_host_waveform_overlap_rejects_invalid_raw_length() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        apply_streaming_waveform_overlap,
    )

    with pytest.raises(RuntimeError, match="valid length"):
        apply_streaming_waveform_overlap(
            torch.zeros(1, 8),
            raw_valid_length=9,
            past_speech=torch.zeros(1, 4),
            past_speech_valid_length=0,
            speech_window=torch.ones(8),
            overlap=4,
            final=False,
        )


def test_stream_routes_final_roles_and_applies_official_overlap_policy() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    runtime.up_rate = 2
    runtime._stream_state = runtime_token2wav.StreamState.from_base_cache(
        {
            name: torch.zeros(shape)
            for name, shape in {
                "conformer_cnn_cache": (1, 2, 1),
                "conformer_att_cache": (1, 1, 1, 3, 1),
                "estimator_cnn_cache": (1, 1, 1, 2, 1),
                "estimator_att_cache": (1, 1, 1, 1, 3, 1),
            }.items()
        },
        prompt_mel_length=5,
    )
    runtime._stream_state.hift_cache["speech"] = torch.ones((1, 4))
    runtime.pre_lookahead_len = 3
    runtime.source_cache_length = 4
    runtime.speech_window = torch.ones(8)
    observed: list[str] = []
    runtime._run_flow_role = lambda tokens, embedding, final: (
        observed.append("flow_final" if final else "flow") or torch.zeros(1, 80, 2)
    )
    runtime._run_hift_role = lambda mel, final: (
        observed.append("hift_final" if final else "hift") or torch.arange(8, dtype=torch.float32).reshape(1, 8)
    )

    result = runtime.stream(torch.tensor([1, 2, 3]), torch.zeros(1, 192), last_chunk=True)

    assert observed == ["flow_final", "hift_final"]
    assert result.shape == (1, 8)


def test_install_guards_native_streaming_boundaries_without_fallback() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import install_hmonnx_token2wav

    tokenizer = SimpleNamespace(flow=SimpleNamespace(), hift=lambda *args, **kwargs: None)
    runtime = SimpleNamespace(
        up_rate=2,
        pre_lookahead_len=3,
        flow_inference=lambda *args: torch.zeros(1),
        hift_inference=lambda *args: torch.zeros(1),
        stream_hift=lambda *args: (_ for _ in ()).throw(RuntimeError("streaming cache is not initialized")),
    )

    install_hmonnx_token2wav(tokenizer, runtime)

    assert hasattr(tokenizer, "_xh_hmonnx_streaming_attached")
    assert tokenizer.flow.pre_lookahead_len == 3
    # Streaming entry (non-None cache source) must fail on uninitialized cache
    # instead of falling back to native or offline paths.
    with pytest.raises(RuntimeError, match="streaming cache is not initialized"):
        tokenizer.hift(torch.zeros(1), torch.zeros(1))


def test_adapter_setup_cache_uses_native_result_instead_of_exported_template() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    shapes = {
        "conformer_cnn_cache": (1, 2, 1),
        "conformer_att_cache": (1, 1, 1, 3, 1),
        "estimator_cnn_cache": (1, 1, 1, 2, 1),
        "estimator_att_cache": (1, 1, 1, 1, 3, 1),
    }
    exported_template = {name: torch.zeros(shape) for name, shape in shapes.items()}
    native_cache = {name: torch.full(shape, float(index + 11)) for index, (name, shape) in enumerate(shapes.items())}
    observed_native_args: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []

    class NativeFlow:
        def setup_cache(self, token, mel, spk, n_timesteps=10):
            observed_native_args.append((token, mel, spk, n_timesteps))
            return native_cache

    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    runtime.up_rate = 2
    runtime._base_flow_cache = exported_template
    runtime.stream_flow_meta = {
        "cache_shapes": {name: list(shape) for name, shape in shapes.items()},
        "cache_axes": {"conformer_att_cache": 3, "estimator_att_cache": 4},
    }
    runtime.report = runtime_token2wav.Token2WavExecutionReport()
    tokenizer = SimpleNamespace(flow=NativeFlow(), hift=SimpleNamespace())
    runtime_token2wav.install_hmonnx_token2wav(tokenizer, runtime)
    token = torch.ones(1, 2, dtype=torch.int32)
    mel = torch.ones(1, 7, 80)
    spk = torch.ones(1, 192)

    result = tokenizer.flow.setup_cache(token, mel, spk, n_timesteps=4)

    assert result is native_cache
    assert observed_native_args == [(token, mel, spk, 4)]
    assert runtime._stream_state.prompt_mel_length == 7
    assert all(
        torch.equal(runtime._stream_state.flow_cache[name], native_cache[name])
        for name in runtime_token2wav.FLOW_CACHE_NAMES
    )
    assert runtime.report.host_initialization_count == 1
    assert runtime.report.host_initialization_backend == "official_host"


def test_official_style_stream_calls_hmonnx_flow_and_hift_roles() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    class OfficialStyleTokenizer:
        def __init__(self) -> None:
            self.flow = SimpleNamespace()
            self.hift = SimpleNamespace()
            self.stream_cache = {
                name: torch.zeros(shape)
                for name, shape in {
                    "conformer_cnn_cache": (1, 2, 1),
                    "conformer_att_cache": (2, 1, 1, 2, 1),
                    "estimator_cnn_cache": (1, 1, 1, 2, 1),
                    "estimator_att_cache": (1, 1, 1, 1, 2, 1),
                }.items()
            }
            self.hift_cache_dict = {
                "mel": torch.zeros(1, 80, 0),
                "source": torch.zeros(1, 1, 0),
                "speech": torch.zeros(1, 0),
            }

        def stream(self, generated_speech_tokens, last_chunk=False):
            token = torch.tensor([generated_speech_tokens], dtype=torch.int32)
            mel, self.stream_cache = self.flow.inference_chunk(
                token=token,
                spk=torch.zeros(1, 192),
                cache=self.stream_cache,
                last_chunk=last_chunk,
                n_timesteps=1,
            )
            mel = torch.cat((self.hift_cache_dict["mel"], mel), dim=2)
            speech, source = self.hift(mel, self.hift_cache_dict["source"])
            is_first = self.hift_cache_dict["speech"].shape[-1] == 0
            self.hift_cache_dict = {
                "mel": mel[:, :, -4:],
                "source": source[:, :, -4:],
                "speech": speech[:, -4:],
            }
            if not last_chunk:
                speech = torch.cat((torch.zeros(1, 4), speech[:, :-4]), dim=1) if is_first else speech[:, :-4]
            return speech

    tokenizer = OfficialStyleTokenizer()
    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    runtime.up_rate = 2
    runtime.stream_flow_meta = {
        "frontend_cache_shapes": {
            "conformer_cnn_cache": list(tokenizer.stream_cache["conformer_cnn_cache"].shape),
            "conformer_att_cache": [2, 1, 1, 4, 1],
        },
        "roles": {
            "stream_flow_frontend": {"output_mel_capacity": 4},
            "stream_flow_frontend_final": {"output_mel_capacity": 6},
        },
    }
    runtime.stream_hift_meta = {
        "source_cache_length": 4,
        "mel_cache_length": 4,
        "speech_cache_length": 4,
        "frame_capacity": 6,
    }
    runtime.hift_phase_noise = torch.zeros(1, 1)
    runtime.hift_source_noise = torch.zeros(1, 4, 1)
    runtime.stream_hift_phase_noise = torch.zeros(1, 1)
    runtime.stream_hift_source_noise = torch.zeros(1, 10, 1)
    runtime.hift_hop_length = 1
    runtime.speech_cache_length = 4
    runtime.speech_window = torch.ones(8)
    runtime.report = runtime_token2wav.Token2WavExecutionReport()
    runtime.chunk_token_capacity = 28
    runtime.pre_lookahead_len = 0
    runtime.base_conformer_layers = 1
    runtime.append_capacity = 6
    runtime.stream_tail_frames = 2
    runtime.host_timestep_cache_banks = 1
    runtime.n_timesteps = 1
    runtime.cfg_rate = 0.7
    runtime.rand_noise = torch.zeros(1, 80, 16)
    runtime._stream_state = runtime_token2wav.StreamState.from_base_cache(
        tokenizer.stream_cache,
        2,
        att_cache_capacity=4,
        append_capacity=6,
        host_timestep_cache_banks=1,
    )
    runtime._stream_state.hift_cache = {
        "mel": torch.zeros(1, 80, 4),
        "source": torch.zeros(1, 1, 4),
        "speech": torch.zeros(1, 4),
    }
    hift_input_counts: list[int] = []

    def flow_frontend(*args):
        token_valid = int(args[1].item())
        current_mel = token_valid * 2
        return (
            torch.zeros(1, 80, current_mel),
            torch.zeros(1, 80),
            args[3],
            torch.zeros(2, 1, 1, 4 + current_mel, 1),
            torch.tensor([int(args[5].item()) + current_mel]),
        )

    runtime.flow_frontend_session = flow_frontend
    runtime.flow_frontend_final_session = runtime.flow_frontend_session
    runtime.stream_estimator_meta = {"estimator_step_cache_shapes": {"input_att": [1, 1, 1, 4, 1], "frame_capacity": 6}}

    def estimator_step(x, mu, t, spks, cond, cnn, att, past_valid, current_valid):
        del mu, t, spks, cond, past_valid, current_valid
        current_cache = att.new_zeros((*att.shape[:3], x.shape[2], att.shape[4]))
        return torch.zeros_like(x), cnn, torch.cat((current_cache, att), dim=3)

    runtime.estimator_step_session = estimator_step

    def run_hift(*args):
        hift_input_counts.append(len(args))
        return (
            torch.ones(1, 10, dtype=torch.float16),
            torch.zeros(1, 1, 10, dtype=torch.float16),
        )

    runtime.hift_stream_session = run_hift
    runtime.hift_stream_final_session = runtime.hift_stream_session

    runtime_token2wav.install_hmonnx_token2wav(tokenizer, runtime)
    normal = tokenizer.stream([1, 2], last_chunk=False)
    final = tokenizer.stream([3], last_chunk=True)

    assert normal.shape == (1, 4)
    assert final.shape == (1, 6)
    assert runtime.report.full_execution_counts["token2wav_flow_stream"] == 1
    assert runtime.report.full_execution_counts["token2wav_flow_stream_final"] == 1
    assert runtime.report.full_execution_counts["token2wav_hift_stream"] == 1
    assert runtime.report.full_execution_counts["token2wav_hift_stream_final"] == 1
    assert hift_input_counts == [8, 8]


def test_top_level_init_bridges_official_prompt_cache_before_stream() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import MiniCPMO45HMONNXRuntime

    base = {
        name: torch.zeros(shape)
        for name, shape in {
            "conformer_cnn_cache": (1, 2, 1),
            "conformer_att_cache": (1, 1, 1, 3, 1),
            "estimator_cnn_cache": (1, 1, 1, 2, 1),
            "estimator_att_cache": (1, 1, 1, 1, 3, 1),
        }.items()
    }
    init_calls: list[tuple[object, int]] = []
    runtime = object.__new__(MiniCPMO45HMONNXRuntime)
    runtime.token2wav = SimpleNamespace(
        init_stream_cache=lambda cache, length: init_calls.append((cache, length)),
        bridge_official_init=lambda result: init_calls.append((result["flow_cache_base"], result["prompt_mel_length"])),
    )
    runtime.host_model = SimpleNamespace(
        init_token2wav_cache=lambda prompt: {
            "flow_cache_base": base,
            "hift_cache_base": {
                "mel": torch.zeros(1, 80, 0),
                "source": torch.zeros(1, 1, 0),
                "speech": torch.zeros(1, 0),
            },
            "prompt_mel_length": 7,
            "prompt": prompt,
        }
    )

    result = runtime.init_token2wav_cache("prompt")

    assert result["prompt"] == "prompt"
    assert init_calls == [(base, 7)]


def test_top_level_init_parses_actual_official_none_return_and_host_cache() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import MiniCPMO45HMONNXRuntime

    flow = {
        "conformer_cnn_cache": torch.zeros(1, 2, 1),
        "conformer_att_cache": torch.zeros(1, 1, 1, 7, 1),
        "estimator_cnn_cache": torch.zeros(1, 1, 1, 2, 1),
        "estimator_att_cache": torch.zeros(1, 1, 1, 1, 7, 1),
    }
    official_cache = {
        "flow_cache_base": flow,
        "hift_cache_base": {
            "mel": torch.zeros(1, 80, 0),
            "source": torch.zeros(1, 1, 0),
            "speech": torch.zeros(1, 0),
        },
    }
    bridged: list[object] = []
    host = SimpleNamespace(token2wav_cache=None)

    def official_init(prompt: object) -> None:
        del prompt
        host.token2wav_cache = official_cache

    host.init_token2wav_cache = official_init
    runtime = object.__new__(MiniCPMO45HMONNXRuntime)
    runtime.host_model = host
    runtime.token2wav = SimpleNamespace(bridge_official_init=bridged.append)

    result = runtime.init_token2wav_cache("prompt")

    assert result is None
    assert bridged == [official_cache]


def test_reset_requires_an_immutable_prompt_base() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        MiniCPMO45Token2WavHMONNXRuntime,
    )

    runtime = object.__new__(MiniCPMO45Token2WavHMONNXRuntime)
    runtime._stream_state = SimpleNamespace()
    runtime._base_flow_cache = None

    with pytest.raises(RuntimeError, match="immutable Token2Wav stream base cache"):
        runtime.reset_stream_cache()


@pytest.mark.parametrize(
    ("component", "role"),
    (
        ("token2wav_flow_frontend", "stream_flow_frontend"),
        ("token2wav_flow_frontend", "stream_flow_frontend_final"),
        ("token2wav_flow_decoder", "stream_flow_estimator_step"),
        ("token2wav_hift", "stream_hift"),
        ("token2wav_hift", "stream_hift_final"),
    ),
)
def test_each_hmonnx_streaming_execution_role_is_required(monkeypatch, component: str, role: str) -> None:
    from pathlib import Path

    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    components = _components()
    del components[component]["graphs"][role]
    with pytest.raises(RuntimeError, match=role):
        runtime._load_streaming_roles(Path("/tmp"), components, lambda path: str(path))


def test_top_level_resets_token2wav_stream_state() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import MiniCPMO45HMONNXRuntime

    calls: list[str] = []
    runtime = object.__new__(MiniCPMO45HMONNXRuntime)
    runtime.host_model = SimpleNamespace(reset_session=lambda value=True: value)
    runtime.token2wav = SimpleNamespace(reset_state=lambda: calls.append("reset"))
    runtime.audio = SimpleNamespace(reset_state=lambda: None)
    runtime.llm = SimpleNamespace(reset_state=lambda: None)
    runtime.tts = SimpleNamespace(reset_state=lambda: None)

    runtime.reset_session()
    runtime.reset_state()

    assert calls == ["reset", "reset"]


def test_reset_session_false_does_not_reset_token2wav() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime import MiniCPMO45HMONNXRuntime

    calls: list[str] = []
    runtime = object.__new__(MiniCPMO45HMONNXRuntime)
    runtime.host_model = SimpleNamespace(reset_session=lambda value=True: value)
    runtime.token2wav = SimpleNamespace(reset_state=lambda: calls.append("reset"))
    runtime.audio = SimpleNamespace(reset_state=lambda: None)
    runtime.llm = SimpleNamespace(reset_state=lambda: None)
    runtime.tts = SimpleNamespace(reset_state=lambda: None)

    runtime.reset_session(False)

    assert calls == []


def test_host_initialization_is_reported_separately_from_hmonnx_execution() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import Token2WavExecutionReport

    report = Token2WavExecutionReport(streaming_attached=True)
    report.record_flow_frontend()
    report.record_flow_decoder()
    report.record_hift()
    report.streaming_attached = True

    report.record_host_initialization()

    assert report.host_initialization_count == 1
    assert report.host_initialization_backend == "official_host"
    assert tuple(report.streaming_execution_counts) == (
        "token2wav_flow_stream",
        "token2wav_flow_stream_final",
        "token2wav_hift_stream",
        "token2wav_hift_stream_final",
    )


def test_host_initialization_cannot_satisfy_hmonnx_streaming_roles() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import Token2WavExecutionReport

    report = Token2WavExecutionReport(streaming_attached=True)
    report.record_host_initialization()
    report.record_stream_flow(False)
    report.record_stream_flow(True)
    report.record_stream_hift(False)

    with pytest.raises(RuntimeError, match="token2wav_hift_stream_final"):
        report.require_streaming_components(tuple(report.streaming_execution_counts))


def test_stream_uses_metadata_chunk_capacity_instead_of_25() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    runtime._stream_state = runtime_token2wav.StreamState.from_base_cache(
        {
            name: torch.zeros(shape)
            for name, shape in {
                "conformer_cnn_cache": (1, 2, 1),
                "conformer_att_cache": (1, 1, 1, 3, 1),
                "estimator_cnn_cache": (1, 1, 1, 2, 1),
                "estimator_att_cache": (1, 1, 1, 1, 3, 1),
            }.items()
        },
        prompt_mel_length=5,
    )
    runtime.chunk_token_capacity = 9
    captured: list[int] = []
    runtime._run_flow_role = lambda tokens, embedding, final: captured.append(tokens.shape[1]) or torch.zeros(1, 80, 2)
    runtime._run_hift_role = lambda mel, final: torch.zeros(1, 8)

    runtime.stream(torch.arange(9).reshape(1, -1), torch.zeros(1, 192), last_chunk=False)

    assert captured == [9]


def test_stream_rejects_oversized_non_final_chunk_without_truncation() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    runtime._stream_state = runtime_token2wav.StreamState.from_base_cache(
        {
            name: torch.zeros(shape)
            for name, shape in {
                "conformer_cnn_cache": (1, 2, 1),
                "conformer_att_cache": (1, 1, 1, 3, 1),
                "estimator_cnn_cache": (1, 1, 1, 2, 1),
                "estimator_att_cache": (1, 1, 1, 1, 3, 1),
            }.items()
        },
        5,
    )
    runtime.chunk_token_capacity = 3

    with pytest.raises(RuntimeError, match="exceeds exported token capacity"):
        runtime.stream(torch.ones(1, 4), torch.zeros(1, 192), last_chunk=False)


def test_attached_hift_without_stream_role_fails_instead_of_using_offline() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    runtime = SimpleNamespace(
        up_rate=2,
        flow_inference=lambda *args: torch.zeros(1),
        hift_inference=lambda *args: torch.ones(1),
    )
    tokenizer = SimpleNamespace(flow=SimpleNamespace(), hift=SimpleNamespace())
    runtime_token2wav.install_hmonnx_token2wav(tokenizer, runtime)

    # Streaming entry (non-None cache source) with no stream role graph must
    # fail loudly instead of silently using the offline graph.
    with pytest.raises(RuntimeError, match="stream_hift"):
        tokenizer.hift(torch.zeros(1, 80, 2), torch.zeros(1))


def test_init_stream_cache_rejects_missing_or_malformed_flow_cache() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    runtime.stream_flow_meta = {
        "cache_shapes": {
            "conformer_cnn_cache": [1, 2, 1],
            "conformer_att_cache": [1, 1, 1, 3, 1],
            "estimator_cnn_cache": [1, 1, 1, 2, 1],
            "estimator_att_cache": [1, 1, 1, 1, 3, 1],
        },
        "cache_axes": {"conformer_att_cache": 3, "estimator_att_cache": 4},
    }

    with pytest.raises(RuntimeError, match="missing Flow cache"):
        runtime.init_stream_cache({}, 5)


def test_official_none_bridge_uses_prompt_mel_frame_axis() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    flow = {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in {
            "conformer_cnn_cache": (1, 2, 1),
            "conformer_att_cache": (1, 1, 1, 12, 1),
            "estimator_cnn_cache": (1, 1, 1, 2, 1),
            "estimator_att_cache": (1, 1, 1, 1, 12, 1),
        }.items()
    }
    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    runtime.stream_flow_meta = {"cache_shapes": {name: list(value.shape) for name, value in flow.items()}}
    runtime.report = runtime_token2wav.Token2WavExecutionReport()
    result = {
        "flow_cache_base": flow,
        # Official hift cache mel is zero-length by design; the prompt length is
        # inferred from the flow conformer attention cache length axis.
        "hift_cache_base": {"mel": torch.zeros(1, 80, 0), "source": torch.zeros(1, 1, 0), "speech": torch.zeros(1, 0)},
    }

    runtime.bridge_official_init(result)

    assert runtime._stream_state.prompt_mel_length == 12


def test_official_none_bridge_infers_prompt_mel_from_flow_when_hift_layout_odd() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    flow = {
        name: torch.zeros(shape, dtype=torch.float32)
        for name, shape in {
            "conformer_cnn_cache": (1, 2, 1),
            "conformer_att_cache": (1, 1, 1, 3, 1),
            "estimator_cnn_cache": (1, 1, 1, 2, 1),
            "estimator_att_cache": (1, 1, 1, 1, 3, 1),
        }.items()
    }
    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    runtime.stream_flow_meta = {"cache_shapes": {name: list(value.shape) for name, value in flow.items()}}
    runtime.report = runtime_token2wav.Token2WavExecutionReport()

    # The flow conformer cache length (3) wins over the odd hift mel layout.
    runtime.bridge_official_init({"flow_cache_base": flow, "hift_cache_base": {"mel": torch.zeros(1, 12, 80)}})
    assert runtime._stream_state.prompt_mel_length == 3


def test_offline_report_stays_three_keys_after_stream_attachment() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import Token2WavExecutionReport

    report = Token2WavExecutionReport(streaming_attached=True)

    assert tuple(report.execution_counts) == (
        "token2wav_flow_frontend",
        "token2wav_flow_decoder",
        "token2wav_hift",
    )
    assert "token2wav_flow_stream" in report.full_execution_counts


def test_full_hmonnx_requirement_accepts_single_final_chunk_streaming_execution() -> None:
    """A bounded generation may emit one final chunk only; non-streaming roles never run
    in the streaming path, so requiring them would reject every real streaming case."""
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import Token2WavExecutionReport

    report = Token2WavExecutionReport(streaming_attached=True)
    report.record_host_initialization()
    report.record_stream_flow(True)
    report.record_stream_hift(True)

    report.require_full_hmonnx_execution()


def test_full_hmonnx_requirement_rejects_streaming_execution_without_hift() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import Token2WavExecutionReport

    report = Token2WavExecutionReport(streaming_attached=True)
    report.record_host_initialization()
    report.record_stream_flow(False)

    with pytest.raises(RuntimeError, match="HiFT"):
        report.require_full_hmonnx_execution()


def test_full_hmonnx_requirement_rejects_streaming_execution_without_flow() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import Token2WavExecutionReport

    report = Token2WavExecutionReport(streaming_attached=True)
    report.record_host_initialization()
    report.record_stream_hift(False)

    with pytest.raises(RuntimeError, match="Flow"):
        report.require_full_hmonnx_execution()


def test_flow_adapter_disables_autocast_around_stream_flow(monkeypatch) -> None:
    """Official stream() enables autocast; A16 HMONNX inputs need the adapter to disable it."""
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    events: list[str] = []

    class FakeAutocast:
        def __init__(self, device_type, enabled=True, **kwargs):
            del kwargs
            self.tag = f"{device_type}:{enabled}"

        def __enter__(self):
            events.append(f"enter:{self.tag}")
            return self

        def __exit__(self, *exc):
            del exc
            events.append(f"exit:{self.tag}")
            return False

    monkeypatch.setattr(runtime_token2wav.torch, "autocast", FakeAutocast)
    runtime = SimpleNamespace(
        stream_flow=lambda token, embedding, last_chunk: events.append("stream_flow") or (torch.zeros(1), {}),
        up_rate=2,
        pre_lookahead_len=3,
    )
    adapter = runtime_token2wav._HMONNXFlowAdapter(SimpleNamespace(), runtime)

    adapter.inference_chunk(torch.zeros(1, 2, dtype=torch.int32), torch.zeros(1, 4), False)

    assert events == ["enter:cpu:False", "stream_flow", "exit:cpu:False"]


def test_hift_adapter_disables_autocast_around_stream_hift(monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    events: list[str] = []

    class FakeAutocast:
        def __init__(self, device_type, enabled=True, **kwargs):
            del kwargs
            self.tag = f"{device_type}:{enabled}"

        def __enter__(self):
            events.append(f"enter:{self.tag}")
            return self

        def __exit__(self, *exc):
            del exc
            events.append(f"exit:{self.tag}")
            return False

    monkeypatch.setattr(runtime_token2wav.torch, "autocast", FakeAutocast)
    runtime = SimpleNamespace(
        stream_hift=lambda mel, cache_source: (
            events.append(f"stream_hift:{mel.shape[-1]}:{cache_source.shape[-1]}") or (torch.zeros(1), torch.zeros(1))
        )
    )
    adapter = runtime_token2wav._HMONNXHiFTAdapter(runtime)

    # Streaming entry: cache source is non-None -> stream_hift under autocast off.
    adapter(torch.zeros(1, 80, 2), torch.zeros(1, 1, 3))

    assert events == ["enter:cpu:False", "stream_hift:2:3", "exit:cpu:False"]


def _flow_role_runtime() -> object:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    base = {
        "conformer_cnn_cache": torch.zeros(1, 2, 1),
        "conformer_att_cache": torch.zeros(2, 1, 1, 2, 1),
        "estimator_cnn_cache": torch.zeros(1, 1, 1, 2, 1),
        "estimator_att_cache": torch.zeros(1, 1, 1, 1, 2, 1),
    }
    runtime._stream_state = runtime_token2wav.StreamState.from_base_cache(
        base,
        prompt_mel_length=2,
        att_cache_capacity=4,
        append_capacity=6,
        host_timestep_cache_banks=1,
    )
    runtime.stream_flow_meta = {
        "frontend_cache_shapes": {
            "conformer_cnn_cache": [1, 2, 1],
            "conformer_att_cache": [2, 1, 1, 4, 1],
        },
        "base_cache_valid_length": 2,
        "roles": {
            "stream_flow_frontend": {"output_mel_capacity": 6},
            "stream_flow_frontend_final": {"output_mel_capacity": 6},
        },
    }
    runtime.stream_estimator_meta = {"estimator_step_cache_shapes": {"input_att": [1, 1, 1, 4, 1], "frame_capacity": 6}}
    runtime.up_rate = 2
    runtime.pre_lookahead_len = 0
    runtime.base_conformer_layers = 1
    runtime.append_capacity = 6
    runtime.stream_tail_frames = 2
    runtime.host_timestep_cache_banks = 1
    runtime.n_timesteps = 1
    runtime.cfg_rate = 0.7
    runtime.rand_noise = torch.zeros(1, 80, 16, dtype=torch.float16)
    runtime.chunk_token_capacity = 28
    runtime.report = runtime_token2wav.Token2WavExecutionReport()
    return runtime


def test_stream_hift_crops_waveform_and_source_to_logical_combined_mel_length() -> None:
    runtime = _flow_role_runtime()
    runtime.stream_hift_meta = {
        "frame_capacity": 6,
        "mel_cache_length": 2,
        "source_cache_length": 4,
        "speech_cache_length": 4,
    }
    runtime.hift_hop_length = 2
    runtime.stream_hift_phase_noise = torch.zeros(1, 1, dtype=torch.float16)
    runtime.stream_hift_source_noise = torch.zeros(1, 16, 1, dtype=torch.float16)
    runtime._pending_hift_final = False
    observed: list[tuple[int, int, int]] = []

    def hift_session(*args):
        observed.append(
            (
                int(args[1].reshape(()).item()),
                int(args[3].reshape(()).item()),
                int(args[5].reshape(()).item()),
            )
        )
        return (
            torch.arange(16, dtype=torch.float16).reshape(1, 16),
            torch.arange(16, dtype=torch.float16).reshape(1, 1, 16),
        )

    runtime.hift_stream_session = hift_session
    runtime.hift_stream_final_session = hift_session

    first_wave, first_source = runtime.stream_hift(torch.zeros(1, 80, 3), torch.zeros(1, 1, 0))
    later_wave, later_source = runtime.stream_hift(torch.zeros(1, 80, 5), torch.zeros(1, 1, 4))

    assert observed == [(3, 0, 0), (3, 2, 4)]
    assert first_wave.shape == (1, 6)
    assert first_source.shape == (1, 1, 6)
    assert later_wave.shape == (1, 10)
    assert later_source.shape == (1, 1, 10)


def test_init_stream_cache_rejects_prompt_caches_exceeding_exported_stream_capacity() -> None:
    """Official native setup_cache sizes prompt caches to the reference audio; graphs are
    exported for a bounded prompt, so oversized caches must fail with a clear error."""
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    runtime = object.__new__(runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime)
    runtime.stream_flow_meta = {
        "frontend_cache_shapes": {
            "conformer_cnn_cache": [1, 512, 6],
            "conformer_att_cache": [10, 1, 8, 150, 128],
        }
    }
    runtime.stream_hift_meta = {"mel_cache_length": 8, "source_cache_length": 4, "speech_cache_length": 4}
    runtime.report = runtime_token2wav.Token2WavExecutionReport()
    base = {
        "conformer_cnn_cache": torch.zeros(1, 512, 6),
        "conformer_att_cache": torch.zeros(10, 1, 8, 842, 128),
        "estimator_cnn_cache": torch.zeros(16, 16, 2, 1024, 2),
        "estimator_att_cache": torch.zeros(16, 16, 2, 8, 842, 128),
    }

    with pytest.raises(RuntimeError, match="prompt"):
        runtime.init_stream_cache(base, 842)


def test_streaming_flow_role_pads_short_final_chunk_to_token_capacity() -> None:
    """The exported streaming Flow graph has a fixed 28-token input; a short final chunk
    must be right-padded with the silence token rather than rejected."""
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import runtime_token2wav

    runtime = _flow_role_runtime()
    runtime.chunk_token_capacity = 28
    observed: dict[str, torch.Tensor] = {}

    def flow_session(*args):
        observed["tokens"] = args[0]
        observed["token_valid_length"] = args[1]
        return (
            torch.zeros(1, 80, 6, dtype=torch.float16),
            torch.zeros(1, 80, dtype=torch.float16),
            args[3],
            torch.zeros(2, 1, 1, 10, 1, dtype=torch.float16),
            torch.tensor([8], dtype=torch.int32),
        )

    runtime.flow_frontend_session = flow_session
    runtime.flow_frontend_final_session = flow_session

    def estimator_session(x, _mu, _t, _spks, _cond, cnn, att, _past_valid, _current_valid):
        current_cache = att.new_zeros((*att.shape[:3], x.shape[2], att.shape[4]))
        return torch.zeros_like(x), cnn, torch.cat((current_cache, att), dim=3)

    runtime.estimator_step_session = estimator_session

    tokens = torch.tensor([[1, 2, 3]], dtype=torch.int32)
    runtime._run_flow_role(tokens, torch.ones(1, 192, dtype=torch.float32), final=True)

    padded = observed["tokens"]
    assert tuple(padded.shape) == (1, 28)
    assert observed["token_valid_length"].item() == 3
    assert padded[0, :3].tolist() == [1, 2, 3]
    assert torch.all(padded[0, 3:] == runtime_token2wav.STREAM_SILENCE_TOKEN)
    assert runtime._stream_state.flow_valid_lengths["conformer_att_cache"] == 4


def test_token2wav_release_state_clears_model_specific_device_tensors() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import (
        MiniCPMO45Token2WavHMONNXRuntime,
    )

    runtime = object.__new__(MiniCPMO45Token2WavHMONNXRuntime)
    device_tensor_names = (
        "hift_phase_noise",
        "hift_source_noise",
        "stream_hift_phase_noise",
        "stream_hift_source_noise",
        "rand_noise",
        "speech_window",
    )
    for name in device_tensor_names:
        setattr(runtime, name, torch.ones(1))
    runtime._base_flow_cache = {"cache": torch.ones(1)}
    runtime._stream_state = object()

    runtime.release_state()

    assert all(getattr(runtime, name) is None for name in device_tensor_names)
    assert runtime._base_flow_cache is None
    assert runtime._stream_state is None


def test_attached_speaker_adapter_does_not_retain_released_session() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.runtime_token2wav import install_hmonnx_token2wav

    runtime = SimpleNamespace(
        up_rate=2,
        campplus_session=lambda value: torch.ones((value.shape[0], 192)),
        campplus_sequence_length=4,
    )
    tokenizer = SimpleNamespace(flow=SimpleNamespace(), hift=SimpleNamespace())
    install_hmonnx_token2wav(tokenizer, runtime)
    runtime.campplus_session = None

    with pytest.raises(RuntimeError, match="released"):
        tokenizer.spk_model.run(None, {"input": torch.ones((1, 4, 80))})


def test_streaming_flow_role_feeds_fp16_activations_to_w8a16_graphs() -> None:
    """Exported streaming Flow graphs are W8A16, so fp32 host state is cast at the boundary."""
    runtime = _flow_role_runtime()
    observed: dict[str, torch.dtype] = {}

    def flow_session(*args):
        tokens, token_valid_length, embedding, cnn, att, _valid = args
        observed["tokens"] = tokens.dtype
        observed["token_valid_length"] = token_valid_length.dtype
        observed["embedding"] = embedding.dtype
        observed["cnn_cache"] = cnn.dtype
        observed["att_cache"] = att.dtype
        return (
            torch.zeros(1, 80, 4, dtype=torch.float16),
            torch.zeros(1, 80, dtype=torch.float16),
            cnn,
            torch.zeros(2, 1, 1, 8, 1, dtype=torch.float16),
            torch.tensor([6], dtype=torch.int32),
        )

    def estimator_session(x, _mu, _t, _spks, _cond, cnn, att, past_valid, current_valid):
        observed["estimator_cnn"] = cnn.dtype
        observed["estimator_att"] = att.dtype
        observed["estimator_past_valid"] = past_valid.dtype
        observed["estimator_current_valid"] = current_valid.dtype
        current_cache = att.new_zeros((*att.shape[:3], x.shape[2], att.shape[4]))
        return torch.zeros_like(x), cnn, torch.cat((current_cache, att), dim=3)

    runtime.flow_frontend_session = flow_session
    runtime.flow_frontend_final_session = flow_session
    runtime.estimator_step_session = estimator_session

    runtime._run_flow_role(
        torch.tensor([[1, 2]], dtype=torch.int64),
        torch.ones(1, 192, dtype=torch.float32),
        final=False,
    )

    assert observed["tokens"] == torch.int32
    assert observed["token_valid_length"] == torch.int32
    assert observed["embedding"] == torch.float16
    assert observed["cnn_cache"] == torch.float16
    assert observed["att_cache"] == torch.float16
    assert observed["estimator_cnn"] == torch.float16
    assert observed["estimator_att"] == torch.float16
    assert observed["estimator_past_valid"] == torch.int32
    assert observed["estimator_current_valid"] == torch.int32


def test_streaming_hift_role_pads_mel_to_capacity_and_feeds_fp16_caches() -> None:
    """Exported streaming HiFT is w8a16 with fixed [1, 80, capacity] mel input."""
    runtime = _flow_role_runtime()
    runtime.stream_hift_meta = {
        "mel_cache_length": 8,
        "source_cache_length": 4,
        "speech_cache_length": 4,
        "frame_capacity": 6,
    }
    runtime.stream_hift_phase_noise = torch.zeros(1, 1, dtype=torch.float16)
    runtime.stream_hift_source_noise = torch.zeros(1, 4, 1, dtype=torch.float16)
    runtime.hift_hop_length = 2
    runtime.speech_cache_length = 4
    runtime.speech_window = torch.ones(8)
    runtime._stream_state.hift_cache = {
        "mel": torch.zeros(1, 80, 8),
        "source": torch.zeros(1, 1, 4),
        "speech": torch.zeros(1, 4),
    }
    observed: dict[str, object] = {}

    def hift_session(*args):
        speech_feat, speech_feat_length, past_mel, _mel_len, past_source, _source_len, _phase, _source_noise = args
        observed["speech_feat_shape"] = tuple(speech_feat.shape)
        observed["speech_feat_dtype"] = speech_feat.dtype
        observed["speech_feat_valid_length"] = int(speech_feat_length.reshape(-1)[0])
        observed["past_mel_dtype"] = past_mel.dtype
        observed["past_source_dtype"] = past_source.dtype
        return (
            torch.ones(1, 8, dtype=torch.float16),
            torch.zeros(1, 1, 8, dtype=torch.float16),
        )

    runtime.hift_stream_session = hift_session
    runtime.hift_stream_final_session = hift_session

    runtime._run_hift_role(torch.zeros(1, 80, 2, dtype=torch.float32), final=False)

    assert observed["speech_feat_shape"] == (1, 80, 6)
    assert observed["speech_feat_valid_length"] == 2
    assert observed["speech_feat_dtype"] == torch.float16
    assert observed["past_mel_dtype"] == torch.float16
    assert observed["past_source_dtype"] == torch.float16


def test_streaming_hift_role_rejects_final_mel_exceeding_capacity() -> None:
    """Dropping valid final mel frames changes speech, so reject an oversized chunk."""
    runtime = _flow_role_runtime()
    runtime.stream_hift_meta = {
        "mel_cache_length": 8,
        "source_cache_length": 4,
        "speech_cache_length": 4,
        "frame_capacity": 6,
    }
    runtime.stream_hift_phase_noise = torch.zeros(1, 1, dtype=torch.float16)
    runtime.stream_hift_source_noise = torch.zeros(1, 4, 1, dtype=torch.float16)
    runtime.speech_cache_length = 4
    runtime.speech_window = torch.ones(8)
    runtime._stream_state.hift_cache = {
        "mel": torch.zeros(1, 80, 8),
        "source": torch.zeros(1, 1, 4),
        "speech": torch.zeros(1, 4),
    }
    observed: dict[str, object] = {}

    def hift_session(*args):
        observed["speech_feat_shape"] = tuple(args[0].shape)
        observed["speech_feat_valid_length"] = int(args[1].reshape(-1)[0])
        return (
            torch.ones(1, 8, dtype=torch.float16),
            torch.zeros(1, 80, 8, dtype=torch.float16),
            torch.zeros(1, 1, 4, dtype=torch.float16),
            torch.tensor([2], dtype=torch.int32),
            torch.tensor([4], dtype=torch.int32),
        )

    runtime.hift_stream_session = hift_session
    runtime.hift_stream_final_session = hift_session

    with pytest.raises(RuntimeError, match="exceeds HiFT capacity"):
        runtime._run_hift_role(torch.zeros(1, 80, 9, dtype=torch.float16), final=True)
    assert observed == {}


def test_streaming_hift_role_uses_stream_noise_instead_of_full_noise() -> None:
    """The streaming HiFT graph expects the per-chunk stream noise, not the full-length
    non-streaming noise loaded at construction."""
    runtime = _flow_role_runtime()
    runtime.stream_hift_meta = {
        "mel_cache_length": 8,
        "source_cache_length": 4,
        "speech_cache_length": 4,
        "frame_capacity": 6,
    }
    full_source_noise = torch.full((1, 40, 1), 9.0, dtype=torch.float16)
    stream_source_noise = torch.full((1, 4, 1), 3.0, dtype=torch.float16)
    stream_phase_noise = torch.full((1, 1), 2.0, dtype=torch.float16)
    runtime.hift_phase_noise = torch.zeros(1, 1, dtype=torch.float16)
    runtime.hift_source_noise = full_source_noise
    runtime.stream_hift_phase_noise = stream_phase_noise
    runtime.stream_hift_source_noise = stream_source_noise
    runtime.hift_hop_length = 2
    runtime.speech_cache_length = 4
    runtime.speech_window = torch.ones(8)
    runtime._stream_state.hift_cache = {
        "mel": torch.zeros(1, 80, 8),
        "source": torch.zeros(1, 1, 4),
        "speech": torch.zeros(1, 4),
    }
    observed: dict[str, torch.Tensor] = {}

    def hift_session(*args):
        observed["phase_noise"] = args[6]
        observed["source_noise"] = args[7]
        return (
            torch.ones(1, 8, dtype=torch.float16),
            torch.zeros(1, 1, 8, dtype=torch.float16),
        )

    runtime.hift_stream_session = hift_session
    runtime.hift_stream_final_session = hift_session

    runtime._run_hift_role(torch.zeros(1, 80, 2, dtype=torch.float16), final=False)

    assert torch.equal(observed["source_noise"], stream_source_noise)
    assert torch.equal(observed["phase_noise"], stream_phase_noise)
