from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_real_golden_enables_every_audio_runtime_session() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.golden import _runtime_sessions

    audio_sessions = {
        name: object()
        for name in (
            "session",
            "stream_prefill_session",
            "stream_decode_session",
            "session_prefill_session",
            "session_decode_session",
        )
    }
    runtime = SimpleNamespace(
        vision=None,
        audio=SimpleNamespace(**audio_sessions),
        llm=SimpleNamespace(prefill_model=None, decode_model=None),
        tts=SimpleNamespace(
            prefill_model=None,
            decode_model=None,
            projector_semantic_session=None,
            head_code_session=None,
        ),
        token2wav=SimpleNamespace(),
    )

    sessions = _runtime_sessions(runtime)

    assert sessions == {
        "audio": audio_sessions["session"],
        "audio_stream_prefill": audio_sessions["stream_prefill_session"],
        "audio_stream_decode": audio_sessions["stream_decode_session"],
        "audio_session_prefill": audio_sessions["session_prefill_session"],
        "audio_session_decode": audio_sessions["session_decode_session"],
    }


def test_streaming_graph_roles_are_complete_and_case_manifest_is_reproducible() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
        CERTIFIED_STREAMING_CASES,
        streaming_case_manifest,
        validate_streaming_graph_coverage,
    )

    meta = {
        "components": {
            "audio": {"graphs": {"stream_prefill": "audio/p.onnx", "stream_decode": "audio/d.onnx"}},
            "llm": {"graphs": {"prefill": "llm/p.onnx", "decode": "llm/d.onnx"}},
            "tts": {"graphs": {"prefill": "tts/p.onnx", "decode": "tts/d.onnx"}},
            "token2wav_flow_frontend": {
                "graphs": {
                    "main": "flow/main.onnx",
                    "stream_flow_frontend": "flow/frontend.onnx",
                    "stream_flow_frontend_final": "flow/frontend_final.onnx",
                }
            },
            "token2wav_flow_decoder": {"graphs": {"stream_flow_estimator_step": "flow/step.onnx"}},
            "token2wav_hift": {"graphs": {"stream_hift": "hift/n.onnx", "stream_hift_final": "hift/final.onnx"}},
        }
    }

    validate_streaming_graph_coverage(meta)
    first = streaming_case_manifest(meta, CERTIFIED_STREAMING_CASES)
    second = streaming_case_manifest(json.loads(json.dumps(meta)), CERTIFIED_STREAMING_CASES)

    assert first == second
    assert first["covered_graphs"] == [
        "audio/stream_prefill",
        "audio/stream_decode",
        "llm/prefill",
        "llm/decode",
        "tts/prefill",
        "tts/decode",
        "token2wav_flow_frontend/stream_flow_frontend",
        "token2wav_flow_frontend/stream_flow_frontend_final",
        "token2wav_flow_decoder/stream_flow_estimator_step",
        "token2wav_hift/stream_hift",
        "token2wav_hift/stream_hift_final",
    ]


def test_streaming_graph_coverage_rejects_unreachable_exported_role(monkeypatch) -> None:
    import xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures as fixtures
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
        StreamingCaseError,
        validate_streaming_graph_coverage,
    )

    monkeypatch.setitem(fixtures._GRAPH_ROLE_NAMES, "audio", {"stream_prefill", "stream_extra"})
    meta = {"components": {"audio": {"graphs": {"stream_extra": "audio/p.onnx"}}}}
    with pytest.raises(StreamingCaseError, match="not covered"):
        validate_streaming_graph_coverage(meta)


def test_streaming_graph_coverage_ignores_non_streaming_audio_roles() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.streaming_fixtures import (
        validate_streaming_graph_coverage,
    )

    meta = {
        "components": {
            "audio": {
                "graphs": {
                    "main": "audio/main.onnx",
                    "stream_prefill": "audio/p.onnx",
                    "stream_decode": "audio/d.onnx",
                    "session_prefill": "audio/sp.onnx",
                    "session_decode": "audio/sd.onnx",
                }
            }
        }
    }
    validate_streaming_graph_coverage(meta)


def test_synthetic_streaming_inputs_are_metadata_driven_and_deterministic(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.synthetic_golden import synthetic_golden_inputs

    component = {
        "stream_prefill_frames": 7,
        "num_hidden_layers": 2,
        "kv_cache_shape": [1, 2, 9, 4],
        "graph_contracts": {
            "stream_prefill": {
                "input_names": [
                    "input_features",
                    "valid_mel_length",
                    "past_seq_length",
                    "past_k_cache_0",
                    "past_k_cache_1",
                    "past_v_cache_0",
                    "past_v_cache_1",
                ]
            }
        },
    }

    first = synthetic_golden_inputs(tmp_path, "audio", "stream_prefill", component)
    second = synthetic_golden_inputs(tmp_path, "audio", "stream_prefill", component)

    assert [tuple(value.shape) for value in first] == [
        (1, 80, 7),
        (1,),
        (1,),
        (1, 2, 9, 4),
        (1, 2, 9, 4),
        (1, 2, 9, 4),
        (1, 2, 9, 4),
    ]
    assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))


def test_synthetic_hift_final_inputs_include_all_cache_lengths(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.synthetic_golden import synthetic_golden_inputs

    torch.save(torch.zeros((1, 3)), tmp_path / "phase.pt")
    torch.save(torch.zeros((1, 1, 3)), tmp_path / "source.pt")
    component = {
        "stream_contract": {
            "frame_capacity": 4,
            "mel_cache_length": 2,
            "source_cache_length": 3,
            "phase_noise_file": "phase.pt",
            "source_noise_file": "source.pt",
        }
    }

    inputs = synthetic_golden_inputs(tmp_path, "token2wav_hift", "stream_hift_final", component)

    assert len(inputs) == 8
    assert inputs[-1].shape == (1, 1, 3)


def test_synthetic_flow_frontend_inputs_include_token_and_cache_valid_lengths(tmp_path: Path) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.synthetic_golden import synthetic_golden_inputs

    component = {
        "stream_contract": {
            "prompt_token_capacity": 4,
            "chunk_token_capacity": 5,
            "base_cache_shapes": {"conformer_cnn_cache": [1, 2, 1]},
            "frontend_cache_shapes": {"conformer_att_cache": [1, 1, 1, 3, 1]},
            "frontend_input_names": [
                "tokens",
                "token_valid_length",
                "embedding",
                "past_conformer_cnn_cache",
                "past_conformer_att_cache",
                "conformer_cache_valid_length",
            ],
            "roles": {"stream_flow_frontend": {"output_mel_capacity": 4}},
        }
    }

    inputs = synthetic_golden_inputs(tmp_path, "token2wav_flow_frontend", "stream_flow_frontend", component)

    assert len(inputs) == 6
    assert inputs[0].shape == (1, 5)
    assert torch.equal(inputs[1], torch.tensor([5], dtype=torch.int32))
    assert inputs[-1].shape == (1,)
    assert inputs[-1].dtype == torch.int32


def test_real_golden_streaming_request_routes_through_case_runner(monkeypatch, tmp_path: Path) -> None:
    import xhmodel_merak.xh_llm.models.minicpm_o_4_5.golden as golden

    calls: list[str] = []

    class Runtime:
        def set_exec_device(self, device: str) -> None:
            del device

        def reset_state(self) -> None:
            calls.append("reset_state")

        def release(self) -> None:
            calls.append("release")

    monkeypatch.setattr(golden, "_load_real_runtime", lambda root, meta: (Runtime(), "tokenizer"))
    monkeypatch.setattr(
        golden,
        "run_streaming_case",
        lambda *args: (
            calls.append("runner")
            or type(
                "Result",
                (),
                {
                    "case_name": "session_audio_text",
                    "api_events": ("reset_session:true",),
                    "text_chunks": ("ok",),
                    "token_ids": (0,),
                    "waveform_chunks": (),
                    "backend_counters": {},
                    "final": True,
                },
            )()
        ),
    )

    request = {"streaming_case": "session_audio_text", "media": "tiny.json"}
    output = golden.dump_real_golden(tmp_path, {"hf_model": "/models/minicpm", "components": {}}, "cpu", request)

    assert calls == ["reset_state", "runner", "release"]
    assert output["streaming_case"] == "session_audio_text"
