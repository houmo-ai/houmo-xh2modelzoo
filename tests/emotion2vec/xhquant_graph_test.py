from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch


def test_xhquant_frame_mask_uses_precomputed_valid_frames():
    from xhmodel_merak.xh_llm.models.emotion2vec.xhquant_graph import XHEmotion2vecFrameMask

    frame_mask = XHEmotion2vecFrameMask(frame_count=799)

    short = frame_mask(torch.tensor([204], dtype=torch.int32))
    full = frame_mask(torch.tensor([799], dtype=torch.int32))

    assert short.shape == (1, 799)
    assert short.dtype == torch.bool
    assert (~short).sum().item() == 204
    assert short.sum().item() == 595
    assert (~full).sum().item() == 799


def test_xhquant_graph_module_uses_xhquant_wrappers_for_core_layers():
    from xhmodel_merak.xh_llm.models.emotion2vec.xhquant_graph import (
        XHEmotion2vecFeatureEncoder,
        XHEmotion2vecSelfAttention,
    )
    from xhquant.nn import MatMul, Softmax, XHConv1d, XHLinear

    feature_encoder = XHEmotion2vecFeatureEncoder.from_spec(
        [(4, 3, 2), (4, 2, 2)],
        input_channels=1,
    )
    attention = XHEmotion2vecSelfAttention(embed_dim=8, num_heads=2)

    assert any(isinstance(module, XHConv1d) for module in feature_encoder.modules())
    assert isinstance(attention.qkv, XHLinear)
    assert isinstance(attention.proj, XHLinear)
    assert isinstance(attention.qk_matmul, MatMul)
    assert isinstance(attention.av_matmul, MatMul)
    assert isinstance(attention.softmax, Softmax)


def test_xhquant_graph_exports_classification_head():
    from xhmodel_merak.xh_llm.models.emotion2vec.xhquant_graph import XHEmotion2vecGraphModel

    assert "classification_head" in XHEmotion2vecGraphModel.forward.__code__.co_names


def test_xhquant_graph_expects_externally_normalized_waveform():
    from xhmodel_merak.xh_llm.models.emotion2vec.xhquant_graph import XHEmotion2vecGraphModel

    assert "normalizer" not in XHEmotion2vecGraphModel.forward.__code__.co_names


def test_alibi_shape_supports_plus_large_heads_and_dynamic_frames():
    from xhmodel_merak.xh_llm.models.emotion2vec.xhquant_graph import build_alibi_bias

    scale = torch.ones(1, 16, 1, 1)
    alibi = build_alibi_bias(frame_count=31, num_heads=16, num_extra_tokens=10, alibi_scale=scale)

    assert alibi.shape == (1, 16, 41, 41)


@pytest.mark.skipif(not Path("data/models/emotion2vec_plus_large/model.pt").exists(), reason="model missing")
def test_xhquant_graph_matches_official_plus_large_short_audio():
    from xhmodel_merak.xh_llm.models.emotion2vec.modeling_emotion2vec import (
        Emotion2vecReferenceModel,
        load_funasr_emotion2vec_model,
    )
    from xhmodel_merak.xh_llm.models.emotion2vec.xhquant_graph import XHEmotion2vecGraphModel

    native = load_funasr_emotion2vec_model("data/models/emotion2vec_plus_large").eval()
    graph = XHEmotion2vecGraphModel.from_funasr(native).eval()
    waveform, sampling_rate = sf.read("data/models/emotion2vec_plus_large/example/test.wav", always_2d=False)
    assert sampling_rate == 16000
    padded = np.zeros(256000, dtype=np.float32)
    padded[: waveform.size] = waveform
    valid_samples = torch.tensor([waveform.size], dtype=torch.int64)
    valid_frames = torch.tensor([204], dtype=torch.int32)
    graph_input = torch.nn.functional.layer_norm(torch.from_numpy(padded[: waveform.size]), (waveform.size,))
    normalized = torch.zeros_like(torch.from_numpy(padded))
    normalized[: waveform.size] = graph_input

    with torch.no_grad():
        expected, expected_mask, expected_utterance = Emotion2vecReferenceModel(native)(
            torch.from_numpy(padded).unsqueeze(0), valid_samples
        )
        actual, actual_mask, actual_utterance, actual_probabilities = graph(normalized.unsqueeze(0), valid_frames)

    expected = expected[0, ~expected_mask[0]].float()
    actual = actual[0, ~actual_mask[0]].float()
    cosine = torch.nn.functional.cosine_similarity(expected.flatten(), actual.flatten(), dim=0)
    assert expected.shape == actual.shape == (204, 1024)
    assert graph.blocks[0].attn.num_heads == 16
    assert graph.alibi_bias.shape == (1, 16, 809, 809)
    assert cosine.item() >= 0.99
    torch.testing.assert_close(actual_utterance, expected_utterance, rtol=2e-2, atol=2e-2)
    assert actual_probabilities.shape == (1, 9)
