from __future__ import annotations

import numpy as np
import pytest


def test_chunk_waveform_splits_fixed_windows_and_keeps_valid_samples():
    from xhmodel_merak.xh_llm.models.emotion2vec.audio_utils import chunk_waveform

    waveform = np.arange(256000 + 16000, dtype=np.float32)
    chunks = chunk_waveform(waveform, sampling_rate=16000, window_samples=256000)

    assert len(chunks) == 2
    assert chunks[0].waveform.shape == (256000,)
    assert chunks[0].valid_samples == 256000
    assert chunks[1].waveform.shape == (256000,)
    assert chunks[1].valid_samples == 16000
    assert np.allclose(chunks[1].waveform[16000:], 0.0)


def test_chunk_waveform_can_limit_valid_samples_for_hmonnx_fp16():
    from xhmodel_merak.xh_llm.models.emotion2vec.audio_utils import chunk_waveform

    waveform = np.arange(65666, dtype=np.float32)
    chunks = chunk_waveform(
        waveform,
        sampling_rate=16000,
        window_samples=256000,
        max_valid_samples=65504,
    )

    assert [chunk.valid_samples for chunk in chunks] == [65504, 162]
    assert all(chunk.waveform.shape == (256000,) for chunk in chunks)


def test_normalize_padded_waveform_uses_only_valid_samples_in_fp32():
    from xhmodel_merak.xh_llm.models.emotion2vec.audio_utils import normalize_padded_waveform

    waveform = np.zeros(16, dtype=np.float32)
    waveform[:8] = np.arange(8, dtype=np.float32)
    normalized = normalize_padded_waveform(waveform, valid_samples=8)

    assert normalized.dtype == np.float32
    assert abs(float(normalized[:8].mean())) < 1e-6
    assert np.isclose(float(normalized[:8].var()), 1.0, atol=1e-5)
    assert np.count_nonzero(normalized[8:]) == 0


def test_validate_and_load_mono_audio_rejects_wrong_sampling_rate():
    from xhmodel_merak.xh_llm.models.emotion2vec.audio_utils import validate_audio_inputs

    with pytest.raises(ValueError, match="16 kHz"):
        validate_audio_inputs(np.zeros((16000,), dtype=np.float32), sampling_rate=8000)


def test_masked_mean_and_trim_frame_padding_mask():
    from xhmodel_merak.xh_llm.models.emotion2vec.audio_utils import masked_mean, trim_frame_padding_mask

    features = np.array([[[1.0, 2.0], [3.0, 6.0], [9.0, 12.0]]], dtype=np.float32)
    mask = np.array([[False, False, True]])

    trimmed = trim_frame_padding_mask(mask)
    pooled = masked_mean(features, mask)

    assert trimmed.tolist() == [[False, False, True]]
    assert pooled.shape == (1, 2)
    assert np.allclose(pooled, np.array([[2.0, 4.0]], dtype=np.float32))


def test_conv_output_length_matches_stride_and_kernel_contract():
    from xhmodel_merak.xh_llm.models.emotion2vec.audio_utils import conv_output_length

    assert conv_output_length(256000, kernel_size=10, stride=5, padding=0, dilation=1) > 0
    assert conv_output_length(16, kernel_size=3, stride=2, padding=1, dilation=1) == 8


def test_emotion2vec_frame_count_matches_official_feature_encoder():
    from xhmodel_merak.xh_llm.models.emotion2vec.audio_utils import emotion2vec_frame_count

    assert emotion2vec_frame_count(256000) == 799
    assert emotion2vec_frame_count(65666) == 204
