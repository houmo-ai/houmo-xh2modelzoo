from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch
import yaml
from onnx import TensorProto, helper

from examples_merak.audio.kokoro.bucketed_inference_demo import (
    _exercise_requests,
    _resolve_voice_path,
)
from xhmodel_merak.xh_other_model.models.kokoro import assets as assets_module
from xhmodel_merak.xh_other_model.models.kokoro import runtime as runtime_module
from xhmodel_merak.xh_other_model.models.kokoro.analysis import analyze_onnx
from xhmodel_merak.xh_other_model.models.kokoro.assets import (
    KOKORO_CHECKPOINT_SHA256,
    KOKORO_CONFIG_SHA256,
    KOKORO_SOURCE_COMMIT,
    KOKORO_ZF001_SHA256,
    resolve_model_assets,
    verify_release_assets,
)
from xhmodel_merak.xh_other_model.models.kokoro.bucketed_runtime import (
    BUCKETED_PRECISION_SPLIT_GRAPH_MODE,
    KokoroBucketedRuntime,
)
from xhmodel_merak.xh_other_model.models.kokoro.buckets import (
    AUDIO_BUCKET_SECONDS,
    BUCKET_ROUTES,
    FRAME_BUCKETS,
    TOKEN_BUCKETS,
    audio_seconds_to_frames,
    normalize_audio_seconds_buckets,
    normalize_bucket_routes,
    normalize_token_buckets,
)
from xhmodel_merak.xh_other_model.models.kokoro.graph import (
    GRAPH_ROLES,
    PrefixReversedBidirectionalLSTMStatic,
    _masked_adain,
    _masked_adain_rmsnorm,
    _masked_temporal_mean,
    _MaskedTemporalRMSNorm,
)
from xhmodel_merak.xh_other_model.models.kokoro.host import (
    ATTENTION_MASK_MIN,
    build_sine_wavs,
    duration_to_alignment,
    duration_to_frame_indices,
    load_voice_style,
    make_attention_mask,
    make_generator_rmsnorm_scales,
    make_reverse_idx,
    make_rmsnorm_scales,
    prepare_lstm_inputs,
    restore_bidirectional_outputs,
    save_voice_pack_numpy,
)
from xhmodel_merak.xh_other_model.models.kokoro.independent_split import (
    FRAME_ACOUSTIC_ROLE,
    TEXT_DURATION_ROLE,
    run_duration_alignment_host,
)
from xhmodel_merak.xh_other_model.models.kokoro.independent_split import (
    GENERATOR_ISTFT_ROLE as INDEPENDENT_GENERATOR_ISTFT_ROLE,
)
from xhmodel_merak.xh_other_model.models.kokoro.independent_split import (
    PHASE_CORE_ROLE as INDEPENDENT_PHASE_CORE_ROLE,
)
from xhmodel_merak.xh_other_model.models.kokoro.precision_split import (
    ACOUSTIC_ROLE,
    GENERATOR_ISTFT_ROLE,
    PHASE_CORE_ROLE,
    PRECISION_SPLIT_GRAPH_MODE,
    PRECISION_SPLIT_NPU_ROLES,
    AcousticStatic,
    GeneratorISTFTStatic,
    KokoroPrecisionSplitRuntime,
    PhaseCoreStatic,
    _onnx_output_max_abs_limit,
    _validate_attention_mask_feed,
)
from xhmodel_merak.xh_other_model.models.kokoro.runtime import KokoroStaticRuntime
from xhmodel_merak.xh_other_model.models.kokoro.single_graph import (
    SINGLE_GRAPH_ROLE,
    StaticAlignment,
    _prefix_mask,
    _reverse_prefix_index,
)
from xhmodel_merak.xh_other_model.models.kokoro.static_dsp import (
    StaticISTFT20,
    StaticSTFT20,
    cordic_atan2,
)
from xhmodel_merak.xh_other_model.models.kokoro.workflow import (
    XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE_ENV,
    _validate_export_config,
    _xhquant_torch_onnx_optimize_context,
)


ROOT = Path(__file__).resolve().parents[2]


def test_quant_export_assets_do_not_require_released_onnx(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_root = tmp_path / "source" / "kokoro"
    config = tmp_path / "pytorch" / "config.json"
    checkpoint = tmp_path / "pytorch" / "kokoro-v1_1-zh.pth"
    voice = tmp_path / "pytorch" / "voices" / "zf_001.pt"
    (source_root / "kokoro").mkdir(parents=True)
    config.parent.mkdir(parents=True)
    voice.parent.mkdir(parents=True)
    (source_root / "kokoro" / "model.py").write_text("", encoding="utf-8")
    config.write_text("{}\n", encoding="utf-8")
    checkpoint.write_bytes(b"checkpoint")
    voice.write_bytes(b"voice")

    assets = resolve_model_assets(tmp_path)
    assert not hasattr(assets, "reference_onnx")
    assert assets.voice_pack is None

    expected_hashes = {
        config.resolve(): KOKORO_CONFIG_SHA256,
        checkpoint.resolve(): KOKORO_CHECKPOINT_SHA256,
        voice.resolve(): KOKORO_ZF001_SHA256,
    }
    monkeypatch.setattr(
        assets_module,
        "sha256",
        lambda path: expected_hashes[Path(path).resolve()],
    )
    monkeypatch.setattr(assets_module, "git_commit", lambda path: KOKORO_SOURCE_COMMIT)
    identity = verify_release_assets(assets)
    assert identity == {
        "config_sha256": KOKORO_CONFIG_SHA256,
        "checkpoint_sha256": KOKORO_CHECKPOINT_SHA256,
        "voice_sha256": KOKORO_ZF001_SHA256,
        "source_commit": KOKORO_SOURCE_COMMIT,
    }


def test_numpy_voice_pack_is_self_contained_and_matches_upstream_pt(tmp_path: Path) -> None:
    source = tmp_path / "zf_test.pt"
    exported = tmp_path / "artifact" / "assets" / "voices" / "zf_test.npy"
    voice = torch.arange(3 * 256, dtype=torch.float32).reshape(3, 1, 256)
    torch.save(voice, source)

    assert save_voice_pack_numpy(source, exported) == exported
    assert np.load(exported, allow_pickle=False).shape == (3, 1, 256)
    assert torch.equal(
        load_voice_style(source, phoneme_count=2),
        load_voice_style(exported, phoneme_count=2),
    )
    metadata = {"runtime_assets": {"voice_pack": {"file": "assets/voices/zf_test.npy"}}}
    assert _resolve_voice_path(tmp_path / "artifact", metadata, model_dir=None) == exported


def test_torch_onnx_optimize_mode_is_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE_ENV, "previous")
    with _xhquant_torch_onnx_optimize_context(True):
        assert os.environ[XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE_ENV] == "1"
    assert os.environ[XHQUANT_TORCH_ONNX_INTERNAL_OPTIMIZE_ENV] == "previous"


def test_bucket_presets_are_the_four_approved_paired_gears() -> None:
    assert TOKEN_BUCKETS == (32, 64, 128, 256)
    assert AUDIO_BUCKET_SECONDS == (4, 8, 16, 32)
    assert FRAME_BUCKETS == (160, 320, 640, 1280)
    assert tuple(audio_seconds_to_frames(value) for value in AUDIO_BUCKET_SECONDS) == FRAME_BUCKETS
    assert tuple((route.token_max_length, route.frame_max_length) for route in BUCKET_ROUTES) == (
        (32, 160),
        (64, 320),
        (128, 640),
        (256, 1280),
    )
    assert normalize_token_buckets([256, 32, 128]) == (32, 128, 256)
    assert normalize_audio_seconds_buckets([32, 4, 16]) == (4, 16, 32)
    assert normalize_bucket_routes([256, 32], [32, 4]) == (
        BUCKET_ROUTES[0],
        BUCKET_ROUTES[3],
    )
    with pytest.raises(ValueError, match="unsupported Kokoro token buckets"):
        normalize_token_buckets([33])
    with pytest.raises(ValueError, match="unsupported Kokoro audio-seconds buckets"):
        normalize_audio_seconds_buckets([3])
    with pytest.raises(ValueError, match="must select matching"):
        normalize_bucket_routes([32], [8])


def test_bucketed_config_accepts_matching_route_fields_and_rejects_mismatches() -> None:
    export = {
        "target_device": "XH2a",
        "model": {"type": "XHKokoroModel"},
        "graph_mode": BUCKETED_PRECISION_SPLIT_GRAPH_MODE,
        "token_buckets": [32],
        "audio_seconds_buckets": [4],
        "phase_on_npu": False,
        "components": {
            TEXT_DURATION_ROLE: {"quant_type": "w16a16_sefp"},
            FRAME_ACOUSTIC_ROLE: {"quant_type": "w16a16_sefp"},
            INDEPENDENT_GENERATOR_ISTFT_ROLE: {"quant_type": "w16a16_sefp"},
        },
    }
    _validate_export_config(export)
    export["torch_onnx_internal_optimize"] = "true"
    with pytest.raises(TypeError, match="must be a boolean"):
        _validate_export_config(export)
    export["torch_onnx_internal_optimize"] = True
    export["audio_seconds_buckets"] = [8]
    with pytest.raises(ValueError, match="must select matching"):
        _validate_export_config(export)
    export["audio_seconds_buckets"] = [4]
    export["bucket_routes"] = [[32, 160]]
    with pytest.raises(ValueError, match="export.bucket_routes is not supported"):
        _validate_export_config(export)


@pytest.mark.parametrize("name", ["validate_onnx", "validate_hmonnx"])
def test_workflow_routes_numerical_validation_to_demo(name: str) -> None:
    export = {
        "target_device": "XH2a",
        "model": {"type": "XHKokoroModel"},
        "graph_mode": BUCKETED_PRECISION_SPLIT_GRAPH_MODE,
        "token_buckets": [32],
        "audio_seconds_buckets": [4],
        "components": {
            TEXT_DURATION_ROLE: {"quant_type": "w16a16_sefp"},
            FRAME_ACOUSTIC_ROLE: {"quant_type": "w16a16_sefp"},
            INDEPENDENT_GENERATOR_ISTFT_ROLE: {"quant_type": "w16a16_sefp"},
        },
        name: True,
    }
    with pytest.raises(ValueError, match="compare_backends.py"):
        _validate_export_config(export)


def test_exercise_all_buckets_runs_exactly_the_four_paired_routes() -> None:
    routes = tuple(route.as_dict() for route in BUCKET_ROUTES)
    requests = _exercise_requests(routes, token_bucket=None, audio_seconds=None)
    assert len(requests) == 4
    assert tuple((value["token_bucket"], value["frame_bucket"]) for value in requests) == (
        (32, 160),
        (64, 320),
        (128, 640),
        (256, 1280),
    )

    selected = _exercise_requests(routes, token_bucket=256, audio_seconds=None)
    assert selected == ({"exercise": "t0256_f1280", "token_bucket": 256, "frame_bucket": 1280},)
    selected = _exercise_requests(routes, token_bucket=None, audio_seconds=16)
    assert selected == ({"exercise": "t0128_f0640", "token_bucket": 128, "frame_bucket": 640},)
    with pytest.raises(ValueError, match="no unique paired route"):
        _exercise_requests(routes, token_bucket=33, audio_seconds=None)


def test_attention_mask_uses_finite_fp16_min_instead_of_infinity() -> None:
    mask = make_attention_mask(8, torch.tensor([3], dtype=torch.int32))
    assert tuple(mask.shape) == (1, 1, 8, 8)
    assert ATTENTION_MASK_MIN == -65_504.0
    assert torch.isfinite(mask).all()
    assert torch.count_nonzero(mask[..., :3]) == 0
    assert torch.count_nonzero(mask[..., 3:] == ATTENTION_MASK_MIN) == 5 * 8
    _validate_attention_mask_feed({"attention_mask": mask})
    with pytest.raises(ValueError, match="must not contain"):
        _validate_attention_mask_feed({"attention_mask": torch.full((1, 1, 2, 2), float("-inf"))})


@pytest.mark.parametrize(
    ("frame_capacity", "expected_limit"),
    (
        (120, 2e-4),
        (600, 2e-4 * 600 / 120),
        (1200, 2e-4 * 1200 / 120),
    ),
)
def test_generator_onnx_validation_gate_scales_with_static_waveform(
    frame_capacity: int,
    expected_limit: float,
) -> None:
    waveform = torch.zeros(1, frame_capacity * 600)
    assert _onnx_output_max_abs_limit(GENERATOR_ISTFT_ROLE, "waveform", waveform) == pytest.approx(expected_limit)


def test_bucketed_runtime_retries_the_next_paired_route_when_duration_overflows() -> None:
    meta = {
        "graph_mode": BUCKETED_PRECISION_SPLIT_GRAPH_MODE,
        "seed": 1234,
        "bucket_presets": {
            "policy": "duration_driven_paired_t_f",
            "routes": [
                {
                    "key": "t0032_f0160",
                    "token_max_length": 32,
                    "frame_max_length": 160,
                    "audio_seconds": 4,
                },
                {
                    "key": "t0064_f0320",
                    "token_max_length": 64,
                    "frame_max_length": 320,
                    "audio_seconds": 8,
                },
            ],
            "token": [32, 64],
            "frame": [160, 320],
            "frame_to_audio_seconds": {"160": 4, "320": 8},
        },
        "phase_boundary": {"mode": "host_fp32"},
        "components": {
            TEXT_DURATION_ROLE: {
                "buckets": {
                    "t0032": {"token_max_length": 32},
                    "t0064": {"token_max_length": 64},
                }
            },
            FRAME_ACOUSTIC_ROLE: {
                "buckets": {
                    "f0160": {"frame_max_length": 160},
                    "f0320": {"frame_max_length": 320},
                }
            },
            INDEPENDENT_PHASE_CORE_ROLE: {
                "buckets": {
                    "f0160": {"frame_max_length": 160},
                    "f0320": {"frame_max_length": 320},
                }
            },
            INDEPENDENT_GENERATOR_ISTFT_ROLE: {
                "buckets": {
                    "f0160": {"frame_max_length": 160},
                    "f0320": {"frame_max_length": 320},
                }
            },
        },
    }
    calls: list[tuple[str, str]] = []

    class FakeRunner:
        def __init__(self, role: str, key: str) -> None:
            self.role = role
            self.key = key

        def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            calls.append((self.role, self.key))
            if self.role == TEXT_DURATION_ROLE:
                text_capacity = int(self.key[1:])
                assert feed["input_ids"].shape == (1, text_capacity)
                logits = np.full((1, text_capacity, 50), -1.81528997, dtype=np.float32)
                logits[:, :4] = -1.65822808
                return {
                    "duration_features": np.zeros((1, text_capacity, 640), dtype=np.float32),
                    "text_encoded": np.zeros((1, 512, text_capacity), dtype=np.float32),
                    "duration_logits": logits,
                }
            if self.role == FRAME_ACOUSTIC_ROLE:
                assert feed["encoded"].shape == (1, 640, 320)
                return {
                    "decoder_feature": np.zeros((1, 512, 640), dtype=np.float32),
                    "f0": np.zeros((1, 640), dtype=np.float32),
                    "phase_increments": np.zeros((1, 640, 9), dtype=np.float32),
                }
            if self.role == INDEPENDENT_PHASE_CORE_ROLE:
                return {"sine": np.zeros((1, 320 * 600, 9), dtype=np.float32)}
            return {"waveform": np.zeros((1, 320 * 600), dtype=np.float32)}

    runtime = KokoroBucketedRuntime(
        ".",
        meta,
        backend="hmonnx",
        lstm_variant="native",
        device="cpu",
    )
    runtime._runner = lambda role, key: FakeRunner(role, key)  # type: ignore[method-assign]
    waveform, synthesis = runtime.synthesize(
        np.arange(28, dtype=np.int32),
        np.zeros((1, 256), dtype=np.float32),
    )
    assert waveform.shape == (200 * 600,)
    assert synthesis["route"] == "t0064_f0320"
    assert synthesis["token_bucket"] == 64
    assert synthesis["frame_bucket"] == 320
    assert len(synthesis["attempts"]) == 2
    assert synthesis["phase_boundary"] == "host_fp32"
    assert synthesis["npu_graph_count"] == 3
    assert calls == [
        (TEXT_DURATION_ROLE, "t0032"),
        (TEXT_DURATION_ROLE, "t0064"),
        (FRAME_ACOUSTIC_ROLE, "f0320"),
        (INDEPENDENT_PHASE_CORE_ROLE, "f0320"),
        (INDEPENDENT_GENERATOR_ISTFT_ROLE, "f0320"),
    ]
    runtime._runners[(TEXT_DURATION_ROLE, "t0032")] = object()  # type: ignore[assignment]
    runtime.clear_runner_cache()
    assert runtime._runners == {}


@pytest.mark.parametrize("valid_length", [1, 28, 110, 120])
def test_masked_temporal_rmsnorm_is_equivalent_to_masked_adain_variance(valid_length: int) -> None:
    channels = 8
    temporal_length = 120
    style_size = 16
    generator = torch.Generator().manual_seed(31)

    class AdaINParameters(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.eps = 1e-5
            self.fc = torch.nn.Linear(style_size, 2 * channels)

    module = AdaINParameters().eval()
    values = torch.randn(1, channels, temporal_length, generator=generator)
    style = torch.randn(1, style_size, generator=generator)
    mask = (torch.arange(temporal_length) < valid_length).float().reshape(1, 1, -1)
    rmsnorm = _MaskedTemporalRMSNorm(temporal_length, module.eps).eval()
    scales = make_rmsnorm_scales(torch.tensor([valid_length]), temporal_length)

    expected = _masked_adain(module, values, style, mask)
    actual = _masked_adain_rmsnorm(
        module,
        rmsnorm,
        values,
        style,
        mask,
        scales[:, 0:1, :],
        scales[:, 1:2, :],
    )

    # At L=1, reconstructing T/L from the independently rounded sqrt pair
    # leaves a tiny centered residual that is amplified by AdaIN's epsilon.
    torch.testing.assert_close(actual, expected, atol=5e-5, rtol=2e-4)


def test_rmsnorm_scales_are_host_computed_reciprocal_pair() -> None:
    scales = make_rmsnorm_scales(torch.tensor([110]), 120)

    assert tuple(scales.shape) == (1, 2, 1)
    assert scales.dtype == torch.float32
    torch.testing.assert_close(scales[0, 0, 0], torch.sqrt(torch.tensor(120.0 / 110.0)))
    torch.testing.assert_close(scales[0, 1, 0], torch.sqrt(torch.tensor(110.0 / 120.0)))
    torch.testing.assert_close(scales.prod(), torch.tensor(1.0), atol=1e-7, rtol=1e-7)


def test_generator_rmsnorm_scales_use_exact_stage_lengths() -> None:
    scales = make_generator_rmsnorm_scales(torch.tensor([810]), 1280)

    assert tuple(scales.shape) == (1, 2, 2, 1)
    assert scales.dtype == torch.float32
    expected = torch.tensor(
        [
            [
                [
                    [np.sqrt((20 * 1280) / (20 * 810))],
                    [np.sqrt((20 * 810) / (20 * 1280))],
                ],
                [
                    [np.sqrt((120 * 1280 + 1) / (120 * 810 + 1))],
                    [np.sqrt((120 * 810 + 1) / (120 * 1280 + 1))],
                ],
            ]
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(scales, expected)
    torch.testing.assert_close(
        scales[:, :, 0, :] * scales[:, :, 1, :],
        torch.ones(1, 2, 1),
        atol=1e-7,
        rtol=1e-7,
    )


def test_masked_temporal_mean_does_not_overflow_fp16_long_axis() -> None:
    temporal_length = 70_000
    valid_length = 66_000
    values = torch.ones(1, 1, temporal_length, dtype=torch.float16)
    mask = (torch.arange(temporal_length) < valid_length).to(torch.float16).reshape(1, 1, -1)
    scale = torch.tensor(
        [[[np.sqrt(temporal_length / valid_length)]]],
        dtype=torch.float16,
    )

    assert torch.isinf((values * mask).sum())
    actual = _masked_temporal_mean(values, mask, scale)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), torch.ones_like(actual).float(), atol=2e-3, rtol=2e-3)


def test_duration_alignment_is_exact_and_rejects_bucket_overflow() -> None:
    duration = torch.tensor([[2, 1, 3, 0]], dtype=torch.int64)
    frame_indices, indexed_frames = duration_to_frame_indices(
        duration,
        frame_max_length=8,
        valid_len=torch.tensor([3]),
    )
    alignment, frames = duration_to_alignment(
        duration,
        frame_max_length=8,
        valid_len=torch.tensor([3]),
    )
    assert frame_indices.tolist() == [0, 0, 1, 2, 2, 2]
    assert torch.equal(indexed_frames, frames)
    assert frames.tolist() == [6]
    assert alignment[0, :, :6].argmax(dim=0).tolist() == [0, 0, 1, 2, 2, 2]
    assert torch.count_nonzero(alignment[:, :, 6:]) == 0
    with pytest.raises(ValueError, match="exceeds Fmax"):
        duration_to_alignment(
            duration,
            frame_max_length=5,
            valid_len=torch.tensor([3]),
        )


def test_host_duration_expansion_gather_is_bit_exact_to_alignment_matmul(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    duration_logits = torch.full((1, 4, 6), -20.0)
    duration_logits[0, 0, :2] = 20.0
    duration_logits[0, 1, :1] = 20.0
    duration_logits[0, 2, :3] = 20.0
    duration_features = torch.arange(1, 13, dtype=torch.float32).reshape(1, 4, 3)
    text_encoded = torch.arange(1, 9, dtype=torch.float32).reshape(1, 2, 4)
    valid_len = torch.tensor([3])
    text_outputs = {
        "duration_features": duration_features,
        "text_encoded": text_encoded,
        "duration_logits": duration_logits,
    }

    duration = torch.tensor([[2, 1, 3, 0]], dtype=torch.int64)
    alignment, expected_frames = duration_to_alignment(duration, 8, valid_len)
    expected_encoded = duration_features.transpose(1, 2) @ alignment
    expected_asr = text_encoded @ alignment

    def reject_matmul(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Host duration expansion must not call torch.matmul")

    monkeypatch.setattr(torch, "matmul", reject_matmul)
    actual, actual_duration = run_duration_alignment_host(
        text_outputs,
        speed=torch.tensor([1.0]),
        valid_len=valid_len,
        frame_max_length=8,
    )

    assert torch.equal(actual_duration, duration)
    assert torch.equal(actual["valid_frames"], expected_frames)
    assert torch.equal(actual["encoded"], expected_encoded)
    assert torch.equal(actual["asr"], expected_asr)


def test_static_alignment_uses_cumsum_compare_and_matmul_semantics() -> None:
    duration = torch.tensor([[2, 1, 3, 0]], dtype=torch.float32)
    duration_features = torch.arange(1, 13, dtype=torch.float32).reshape(1, 4, 3)
    text_features = torch.arange(1, 9, dtype=torch.float32).reshape(1, 2, 4)
    encoded, asr, alignment, valid_frames = StaticAlignment(8)(
        duration_features,
        text_features,
        duration,
    )
    assert valid_frames.tolist() == [6]
    assert alignment[0, :, :6].argmax(dim=0).tolist() == [0, 0, 1, 2, 2, 2]
    assert torch.count_nonzero(alignment[:, :, 6:]) == 0
    assert torch.equal(encoded, duration_features.transpose(1, 2) @ alignment)
    assert torch.equal(asr, text_features @ alignment)


def test_static_reverse_index_clamps_an_overflow_for_safe_bucket_retry() -> None:
    assert _reverse_prefix_index(torch.tensor([11]), 8).tolist() == list(range(7, -1, -1))


def test_bidirectional_host_bridge_reverses_only_the_valid_prefix() -> None:
    values = torch.arange(1, 21, dtype=torch.float32).reshape(1, 5, 4)
    valid_len = torch.tensor([3])
    forward, backward = prepare_lstm_inputs(values, valid_len)
    assert torch.equal(forward[:, :3], values[:, :3])
    assert torch.equal(backward[:, :3], torch.flip(values[:, :3], dims=[1]))
    assert torch.count_nonzero(forward[:, 3:]) == 0
    restored = restore_bidirectional_outputs(forward, backward, valid_len)
    assert torch.equal(restored[:, :3, :4], values[:, :3])
    assert torch.equal(restored[:, :3, 4:], values[:, :3])
    assert torch.count_nonzero(restored[:, 3:]) == 0
    assert make_reverse_idx(5, valid_len).tolist() == [2, 1, 0, 3, 4]


@pytest.mark.parametrize("valid_length", [1, 3, 5])
def test_prefix_reversed_wrapper_matches_packed_bilstm_and_exports_standard_onnx(
    tmp_path: Path,
    valid_length: int,
) -> None:
    torch.manual_seed(41)
    total_length = 5
    source = torch.nn.LSTM(
        input_size=3,
        hidden_size=4,
        batch_first=True,
        bidirectional=True,
    ).eval()
    values = torch.randn(1, total_length, 3)
    valid_len = torch.tensor([valid_length], dtype=torch.int64)
    packed = torch.nn.utils.rnn.pack_padded_sequence(
        values,
        valid_len,
        batch_first=True,
        enforce_sorted=True,
    )
    packed_output, _ = source(packed)
    expected, _ = torch.nn.utils.rnn.pad_packed_sequence(
        packed_output,
        batch_first=True,
        total_length=total_length,
    )

    reverse_indices = make_reverse_idx(total_length, valid_len).to(torch.int64)
    wrapper = PrefixReversedBidirectionalLSTMStatic(source, total_length).eval()
    mask = (torch.arange(total_length) < valid_length).reshape(1, total_length, 1)
    actual = wrapper(values, reverse_indices) * mask
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=0)

    path = tmp_path / f"prefix_reversed_l{valid_length}.onnx"
    torch.onnx.export(
        wrapper,
        (values, reverse_indices),
        path,
        input_names=["values", "reverse_indices"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    lstm_nodes = [node for node in graph.graph.node if node.op_type == "LSTM"]
    assert len(lstm_nodes) == 2
    assert sum(node.op_type == "Gather" for node in graph.graph.node) >= 2
    for node in lstm_nodes:
        attributes = {
            attribute.name: helper.get_attribute_value(attribute)
            for attribute in node.attribute
        }
        assert attributes.get("direction", b"forward") == b"forward"
        assert len(node.input) < 5 or node.input[4] == ""
    session = ort.InferenceSession(
        str(path),
        providers=["CPUExecutionProvider"],
    )
    ort_output = session.run(
        None,
        {"values": values.numpy(), "reverse_indices": reverse_indices.numpy()},
    )[0]
    np.testing.assert_allclose(ort_output, wrapper(values, reverse_indices).detach().numpy(), atol=2e-6)


def test_sine_generation_is_seeded_without_changing_global_rng() -> None:
    f0 = torch.full((1, 8), 220.0)
    valid_frames = torch.tensor([4])
    state = torch.random.get_rng_state().clone()
    first = build_sine_wavs(f0, valid_frames, frame_max_length=4, seed=123)
    second = build_sine_wavs(f0, valid_frames, frame_max_length=4, seed=123)
    third = build_sine_wavs(f0, valid_frames, frame_max_length=4, seed=124)
    assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(first, second)
    assert not torch.equal(first, third)


def test_static_stft20_matches_constant_padded_torch_stft() -> None:
    generator = torch.Generator().manual_seed(17)
    waveform = torch.randn(1, 600, generator=generator)
    static = StaticSTFT20(pad_mode="constant")(waveform)
    reference_complex = torch.stft(
        waveform,
        n_fft=20,
        hop_length=5,
        win_length=20,
        window=torch.hann_window(20, periodic=True),
        center=True,
        pad_mode="constant",
        return_complex=True,
    )
    reference = torch.cat([reference_complex.abs(), torch.angle(reference_complex)], dim=1)
    magnitude_error = (static[:, :11] - reference[:, :11]).abs().max()
    assert magnitude_error < 2e-5
    phase_mask = reference[:, :11] > 1e-5
    phase_error = torch.atan2(
        torch.sin(static[:, 11:] - reference[:, 11:]),
        torch.cos(static[:, 11:] - reference[:, 11:]),
    ).abs()
    assert phase_error[phase_mask].max() < 2e-5


def test_static_stft20_zero_signal_has_finite_zero_phase() -> None:
    result = StaticSTFT20(pad_mode="constant")(torch.zeros(1, 600))
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result) == 0


def test_cordic_atan2_matches_torch_without_atan_operator() -> None:
    generator = torch.Generator().manual_seed(29)
    x = torch.randn(4, 11, 121, generator=generator)
    y = torch.randn(4, 11, 121, generator=generator)
    expected = torch.atan2(y, x)
    actual = cordic_atan2(y, x, iterations=16)
    circular_error = torch.atan2(
        torch.sin(actual - expected),
        torch.cos(actual - expected),
    ).abs()
    assert circular_error.max() < 4e-5


def test_length_aware_reflect_stft_matches_true_prefix() -> None:
    generator = torch.Generator().manual_seed(19)
    waveform = torch.randn(1, 720, generator=generator)
    valid_samples = torch.tensor([600], dtype=torch.int32)
    static = StaticSTFT20(
        pad_mode="length_aware_reflect",
        waveform_length=720,
    )(waveform, valid_samples)
    reference_complex = torch.stft(
        waveform[:, :600],
        n_fft=20,
        hop_length=5,
        win_length=20,
        window=torch.hann_window(20, periodic=True),
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    reference = torch.cat([reference_complex.abs(), torch.angle(reference_complex)], dim=1)
    valid_spectral_frames = reference.shape[-1]
    magnitude_error = (static[:, :11, :valid_spectral_frames] - reference[:, :11]).abs().max()
    assert magnitude_error < 2e-5
    phase_mask = reference[:, :11] > 1e-5
    phase_error = torch.atan2(
        torch.sin(static[:, 11:, :valid_spectral_frames] - reference[:, 11:]),
        torch.cos(static[:, 11:, :valid_spectral_frames] - reference[:, 11:]),
    ).abs()
    assert phase_error[phase_mask].max() < 2e-5


def test_length_aware_reflect_exports_integer_bounds_as_where(tmp_path: Path) -> None:
    path = tmp_path / "length_aware_reflect.onnx"
    torch.onnx.export(
        StaticSTFT20(pad_mode="length_aware_reflect", waveform_length=720),
        (torch.zeros(1, 720), torch.tensor([600], dtype=torch.int32)),
        path,
        input_names=["waveform", "valid_samples"],
        output_names=["spectrogram"],
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(path, load_external_data=False)
    assert not [node for node in graph.graph.node if node.op_type == "Clip"]
    assert len([node for node in graph.graph.node if node.op_type == "Where"]) >= 4
    assert not [
        node
        for node in graph.graph.node
        if node.op_type == "Cast"
        and any(attribute.name == "to" and attribute.i == TensorProto.FLOAT for attribute in node.attribute)
    ]


def test_prefix_mask_exports_where_without_fp32_cast(tmp_path: Path) -> None:
    class PrefixMask(torch.nn.Module):
        def forward(self, length: torch.Tensor) -> torch.Tensor:
            return _prefix_mask(length, 16, channel_axis=True)

    path = tmp_path / "prefix_mask.onnx"
    torch.onnx.export(
        PrefixMask(),
        (torch.tensor([7], dtype=torch.int32),),
        path,
        input_names=["length"],
        output_names=["mask"],
        opset_version=17,
        dynamo=False,
    )
    graph = onnx.load(path, load_external_data=False)
    assert any(node.op_type == "Where" for node in graph.graph.node)
    assert not [
        node
        for node in graph.graph.node
        if node.op_type == "Cast"
        and any(attribute.name == "to" and attribute.i == TensorProto.FLOAT for attribute in node.attribute)
    ]


def test_static_istft20_matches_torch_istft() -> None:
    frame_max_length = 1
    generator = torch.Generator().manual_seed(23)
    waveform = torch.randn(1, 600, generator=generator)
    complex_spec = torch.stft(
        waveform,
        n_fft=20,
        hop_length=5,
        win_length=20,
        window=torch.hann_window(20, periodic=True),
        center=True,
        pad_mode="reflect",
        return_complex=True,
    )
    spec_phase = torch.cat([complex_spec.abs(), torch.angle(complex_spec)], dim=1)
    static = StaticISTFT20(frame_max_length)(spec_phase)
    reference = torch.istft(
        complex_spec,
        n_fft=20,
        hop_length=5,
        win_length=20,
        window=torch.hann_window(20, periodic=True),
        center=True,
        length=600,
    )
    assert (static - reference).abs().max() < 2e-5


def test_reference_analyzer_walks_control_flow_and_lstm_contract(tmp_path: Path) -> None:
    then_graph = helper.make_graph(
        [helper.make_node("SequenceEmpty", [], ["sequence"], dtype=TensorProto.FLOAT)],
        "then",
        [],
        [helper.make_tensor_sequence_value_info("sequence", TensorProto.FLOAT, None)],
    )
    else_graph = helper.make_graph(
        [helper.make_node("SequenceEmpty", [], ["sequence"], dtype=TensorProto.FLOAT)],
        "else",
        [],
        [helper.make_tensor_sequence_value_info("sequence", TensorProto.FLOAT, None)],
    )
    graph = helper.make_graph(
        [
            helper.make_node(
                "If",
                ["condition"],
                ["result"],
                then_branch=then_graph,
                else_branch=else_graph,
            ),
            helper.make_node(
                "LSTM",
                ["x", "weights", "recurrent", "bias", "sequence_lens"],
                ["y"],
                name="bilstm",
                direction="bidirectional",
                hidden_size=2,
            ),
        ],
        "synthetic",
        [
            helper.make_tensor_value_info("condition", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 1, 4]),
            helper.make_tensor_value_info("weights", TensorProto.FLOAT, [2, 8, 4]),
            helper.make_tensor_value_info("recurrent", TensorProto.FLOAT, [2, 8, 2]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [2, 16]),
            helper.make_tensor_value_info("sequence_lens", TensorProto.INT32, [1]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2, 1, 2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    path = tmp_path / "synthetic.onnx"
    onnx.save(model, path)
    report = analyze_onnx(path)
    assert report["main_nodes"] == 2
    assert report["recursive_nodes"] == 4
    assert report["op_counts"]["SequenceEmpty"] == 2
    assert report["lstm_nodes"][0]["direction"] == "bidirectional"
    assert report["lstm_nodes"][0]["has_sequence_lens"] is True


def test_kokoro_configs_cover_all_graphs_and_runtime_rejects_missing_runner() -> None:
    config_dir = ROOT / "configs_merak/workflows/xh2a/other_models/kokoro"
    for path in sorted(config_dir.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        export = data["export"]
        if export.get("graph_mode") == "single_graph":
            expected = (SINGLE_GRAPH_ROLE,)
        elif export.get("graph_mode") == BUCKETED_PRECISION_SPLIT_GRAPH_MODE:
            expected = (
                TEXT_DURATION_ROLE,
                FRAME_ACOUSTIC_ROLE,
                INDEPENDENT_GENERATOR_ISTFT_ROLE,
            )
        elif export.get("graph_mode") == PRECISION_SPLIT_GRAPH_MODE:
            expected = PRECISION_SPLIT_NPU_ROLES
        else:
            expected = GRAPH_ROLES
        assert tuple(export["components"]) == expected
        _validate_export_config(export)
    with pytest.raises(ValueError, match="missing Kokoro graph runners"):
        KokoroStaticRuntime(
            {},
            text_max_length=64,
            frame_max_length=512,
            lstm_chunk_length=64,
        )


def test_single_graph_hmonnx_requires_native_lstm() -> None:
    export = {
        "target_device": "XH2a",
        "model": {"type": "XHKokoroModel"},
        "graph_mode": "single_graph",
        "text_max_length": 32,
        "frame_max_length": 120,
        "stft_phase_mode": "cordic",
        "convert_hmonnx": True,
        "components": {SINGLE_GRAPH_ROLE: {"quant_type": "w16a16_sefp"}},
    }
    _validate_export_config(export)
    export["decompose_lstm"] = True
    with pytest.raises(ValueError, match="must keep native LSTM"):
        _validate_export_config(export)


def test_kokoro_f0_norm_mode_rejects_unknown_value() -> None:
    export = {
        "target_device": "XH2a",
        "model": {"type": "XHKokoroModel"},
        "graph_mode": "precision_split",
        "text_max_length": 32,
        "frame_max_length": 120,
        "f0_norm_mode": "layernorm",
        "components": {
            ACOUSTIC_ROLE: {"quant_type": "w16a16_sefp"},
            GENERATOR_ISTFT_ROLE: {"quant_type": "w16a16_sefp"},
        },
    }
    with pytest.raises(ValueError, match="f0_norm_mode"):
        _validate_export_config(export)


def test_runtime_can_fallback_selected_hmonnx_roles_to_ort(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    components = {
        role: {
            "onnx_file": f"onnx/{role}.onnx",
            "hmonnx_file": f"hmonnx/{role}.onnx",
        }
        for role in GRAPH_ROLES
    }
    (tmp_path / "export_meta_info.json").write_text(
        json.dumps(
            {
                "components": components,
                "text_max_length": 64,
                "frame_max_length": 512,
                "lstm_chunk_length": 64,
                "seed": 1234,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_module, "OrtRunner", lambda path: ("ort", Path(path).name))
    monkeypatch.setattr(
        runtime_module,
        "HmonnxRunner",
        lambda path, device: ("hmonnx", Path(path).name, device),
    )

    runtime = KokoroStaticRuntime.from_export(
        tmp_path,
        backend="hmonnx",
        device="cuda:7",
        ort_fallback_roles=("shared_fwd_lstm", "shared_bwd_lstm", "f0_branch"),
    )
    assert runtime.runners["f0_branch"] == ("ort", "f0_branch.onnx")
    assert runtime.runners["generator"] == ("hmonnx", "generator.onnx", "cuda:7")
    with pytest.raises(ValueError, match="unknown Kokoro ORT fallback"):
        KokoroStaticRuntime.from_export(
            tmp_path,
            backend="hmonnx",
            ort_fallback_roles=("not_a_graph",),
        )


def test_ort_runner_filters_optional_feed_values_not_declared_by_graph() -> None:
    class FakeSession:
        def run(
            self,
            output_names: None,
            feed: dict[str, np.ndarray],
        ) -> list[np.ndarray]:
            assert output_names is None
            assert tuple(feed) == ("shared",)
            return [feed["shared"] + 1.0]

    runner = runtime_module.OrtRunner.__new__(runtime_module.OrtRunner)
    runner.session = FakeSession()
    runner.input_names = ["shared"]
    runner.output_names = ["f0"]

    outputs = runner.run(
        {
            "shared": np.asarray([1.0], dtype=np.float32),
            "norm_scales": np.asarray([2.0, 0.5], dtype=np.float32),
        }
    )

    np.testing.assert_array_equal(outputs["f0"], np.asarray([2.0], dtype=np.float32))


def test_precision_split_keeps_only_phase_core_on_host() -> None:
    class FakeSineSource(torch.nn.Module):
        def upsample_f0(self, f0: torch.Tensor) -> torch.Tensor:
            assert torch.count_nonzero(f0 == 3.0) == f0.numel()
            return torch.full((1, 18, 1), 4.0)

        def prepare_phase_increments(self, f0_up: torch.Tensor) -> torch.Tensor:
            assert tuple(f0_up.shape) == (1, 18, 1)
            return torch.full((1, 6, 9), 5.0)

        def phase_increments_to_sine(self, phase_increments: torch.Tensor) -> torch.Tensor:
            assert tuple(phase_increments.shape) == (1, 6, 9)
            return torch.full((1, 1800, 9), 6.0)

    class FakeRoot(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.sine_source = FakeSineSource()

        def forward_acoustic(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            style: torch.Tensor,
            speed: torch.Tensor,
            valid_len: torch.Tensor,
        ) -> tuple[torch.Tensor, ...]:
            del input_ids, attention_mask, style, speed, valid_len
            return (
                torch.full((1, 3, 4), -1.0),
                torch.full((1, 2, 6), 2.0),
                torch.tensor([[1, 1, 0, 0]]),
                torch.tensor([2], dtype=torch.int32),
                torch.full((1, 6), 3.0),
            )

        def forward_generator_from_sine(
            self,
            decoder_feature: torch.Tensor,
            sine: torch.Tensor,
            f0: torch.Tensor,
            style: torch.Tensor,
            valid_frames: torch.Tensor,
            generator_norm_scales: torch.Tensor,
        ) -> torch.Tensor:
            assert tuple(decoder_feature.shape) == (1, 2, 6)
            assert tuple(sine.shape) == (1, 1800, 9)
            assert torch.count_nonzero(f0 == 3.0) == f0.numel()
            assert tuple(style.shape) == (1, 256)
            assert valid_frames.tolist() == [2]
            assert tuple(generator_norm_scales.shape) == (1, 2, 2, 1)
            return torch.full((1, 1800), 7.0)

    root = FakeRoot()
    acoustic = AcousticStatic(root)
    decoder_feature, f0, phase_increments, duration, valid_frames = acoustic(
        torch.zeros(1, 4, dtype=torch.int32),
        make_attention_mask(4, torch.tensor([2], dtype=torch.int32)),
        torch.zeros(1, 256),
        torch.ones(1),
        torch.tensor([2], dtype=torch.int32),
    )
    sine = PhaseCoreStatic(root)(phase_increments)
    waveform = GeneratorISTFTStatic(root)(
        decoder_feature,
        sine,
        f0,
        torch.zeros(1, 256),
        valid_frames,
        make_generator_rmsnorm_scales(valid_frames, 3),
    )

    assert tuple(decoder_feature.shape) == (1, 2, 6)
    assert tuple(f0.shape) == (1, 6)
    assert tuple(phase_increments.shape) == (1, 6, 9)
    assert duration.tolist() == [[1, 1, 0, 0]]
    assert torch.count_nonzero(sine == 6.0) == sine.numel()
    assert torch.count_nonzero(waveform == 7.0) == waveform.numel()


def test_precision_split_runtime_keeps_only_cumsum_phase_and_sin_in_fp32() -> None:
    calls: list[tuple[str, dict[str, np.ndarray]]] = []

    class FakeRunner:
        def __init__(self, role: str) -> None:
            self.role = role

        def run(self, feed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
            calls.append((self.role, feed))
            if self.role == ACOUSTIC_ROLE:
                return {
                    "decoder_feature": np.ones((1, 2, 6), dtype=np.float32),
                    "f0": np.ones((1, 6), dtype=np.float32),
                    "phase_increments": np.ones((1, 6, 9), dtype=np.float32),
                    "duration": np.asarray([[1, 1, 0, 0]], dtype=np.float32),
                    # HmonnxRunner exposes all outputs as float32.
                    "valid_frames": np.asarray([2.0], dtype=np.float32),
                }
            if self.role == PHASE_CORE_ROLE:
                assert tuple(feed) == ("phase_increments",)
                assert feed["phase_increments"].dtype == np.float32
                return {"sine": np.ones((1, 1800, 9), dtype=np.float32)}
            assert feed["sine"].dtype == np.float32
            assert feed["f0"].dtype == np.float32
            assert feed["valid_frames"].dtype == np.int32
            assert feed["generator_norm_scales"].shape == (1, 2, 2, 1)
            return {"waveform": np.arange(1800, dtype=np.float32).reshape(1, -1)}

    runtime = KokoroPrecisionSplitRuntime(
        {role: FakeRunner(role) for role in (ACOUSTIC_ROLE, PHASE_CORE_ROLE, GENERATOR_ISTFT_ROLE)},
        text_max_length=4,
        frame_max_length=3,
        seed=1234,
    )
    waveform, synthesis = runtime.synthesize(
        [1, 2],
        np.zeros((1, 256), dtype=np.float32),
    )
    assert [role for role, _feed in calls] == [
        ACOUSTIC_ROLE,
        PHASE_CORE_ROLE,
        GENERATOR_ISTFT_ROLE,
    ]
    assert waveform.shape == (1200,)
    assert synthesis["frame_length"] == 2
    assert synthesis["duration"] == [1, 1]
    with pytest.raises(ValueError, match="fixed at export time"):
        runtime.synthesize(
            [1, 2],
            np.zeros((1, 256), dtype=np.float32),
            seed=7,
        )
