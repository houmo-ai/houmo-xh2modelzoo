from __future__ import annotations

import numpy as np
import torch


class _FakeFunASRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.proj = torch.nn.Linear(768, 9)

    def forward(self, *, source, padding_mask, mask, features_only, remove_extra_tokens):
        self.calls.append((source.clone(), padding_mask.clone(), mask, features_only, remove_extra_tokens))
        batch = source.shape[0]
        frame_count = 4
        return {
            "x": torch.ones(batch, frame_count, 768, dtype=torch.float32),
            "padding_mask": torch.tensor([[False, False, True, True]]),
        }


def test_extract_x_from_funasr_result_supports_dict_and_tuple():
    from xhmodel_merak.xh_llm.models.emotion2vec.modeling_emotion2vec import extract_x_from_result

    tensor = torch.randn(1, 2, 3)
    assert torch.equal(extract_x_from_result({"x": tensor}), tensor)
    assert torch.equal(extract_x_from_result((tensor, torch.ones(1, 2, dtype=torch.bool))), tensor)


def test_reference_model_masks_tail_and_returns_feature_outputs():
    from xhmodel_merak.xh_llm.models.emotion2vec.modeling_emotion2vec import (
        Emotion2vecReferenceModel,
        classify_utterance_feature,
    )

    bridge = Emotion2vecReferenceModel(_FakeFunASRModel(), sampling_rate=16000, window_samples=256000)

    waveform = torch.arange(256000, dtype=torch.float32).unsqueeze(0)
    valid_samples = torch.tensor([16000], dtype=torch.int64)
    outputs = bridge(waveform, valid_samples)

    frame_features, frame_padding_mask, utterance_feature = outputs
    assert frame_features.shape == (1, 4, 768)
    assert frame_padding_mask.shape == (1, 4)
    assert frame_padding_mask.dtype == torch.bool
    assert frame_padding_mask.tolist() == [[False, False, True, True]]
    assert utterance_feature.shape == (1, 768)
    logits, probabilities = classify_utterance_feature(utterance_feature, bridge.native_model.proj)
    assert logits.shape == (1, 9)
    assert probabilities.shape == (1, 9)
    source, padding_mask, mask, features_only, remove_extra_tokens = bridge.native_model.calls[0]
    assert np.allclose(source[0, 16000:].cpu().numpy(), 0.0)
    assert abs(float(source[0, :16000].mean())) < 1e-4
    assert padding_mask[0, :16000].sum() == 0
    assert padding_mask[0, 16000:].all()
    assert mask is False
    assert features_only is True
    assert remove_extra_tokens is True


def test_reference_model_does_not_use_tensor_item_for_valid_samples():
    from xhmodel_merak.xh_llm.models.emotion2vec.modeling_emotion2vec import Emotion2vecReferenceModel

    bridge = Emotion2vecReferenceModel(_FakeFunASRModel(), sampling_rate=16000, window_samples=256000)
    traced = torch.jit.trace(
        bridge,
        (torch.zeros(1, 256000, dtype=torch.float32), torch.tensor([16000], dtype=torch.int64)),
        strict=False,
    )
    outputs = traced(torch.ones(1, 256000), torch.tensor([8000], dtype=torch.int64))
    assert outputs[0].shape == (1, 4, 768)
    assert len(outputs) == 3


def test_reference_model_validates_fixed_input_shape():
    from xhmodel_merak.xh_llm.models.emotion2vec.modeling_emotion2vec import Emotion2vecReferenceModel

    bridge = Emotion2vecReferenceModel(_FakeFunASRModel(), sampling_rate=16000, window_samples=256000)

    with torch.no_grad():
        waveform = torch.zeros(2, 256000, dtype=torch.float32)
        valid_samples = torch.tensor([256000, 256000], dtype=torch.int64)
        try:
            bridge(waveform, valid_samples)
        except ValueError as exc:
            assert "batch size 1" in str(exc)
        else:
            raise AssertionError("expected fixed-shape validation to fail")
