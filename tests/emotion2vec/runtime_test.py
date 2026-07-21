from __future__ import annotations

import numpy as np
import pytest
import torch


class _FakeSession:
    def __init__(self):
        self.calls = []

    def forward(self, waveform, valid_frames):
        self.calls.append((waveform.clone(), valid_frames.clone()))
        frame_features = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
        frame_mask = torch.tensor([[False, False, True]])
        utterance = frame_features[:, :2].mean(dim=1)
        probabilities = torch.softmax(torch.arange(9, dtype=torch.float32), dim=0).unsqueeze(0)
        return frame_features, frame_mask, utterance, probabilities


def test_runtime_extract_waveform_chunks_and_pools_frames(tmp_path):
    from xhmodel_merak.xh_llm.models.emotion2vec.configuration_emotion2vec import Emotion2vecModelMeta
    from xhmodel_merak.xh_llm.models.emotion2vec.emotion2vec_hmonnx_inference import Emotion2vecHMONNXModel

    head_path = tmp_path / "quant_embedding.pt"
    head_weight = torch.arange(72, dtype=torch.float32).reshape(9, 8)
    head_bias = torch.arange(9, dtype=torch.float32)
    torch.save({"weight": head_weight, "bias": head_bias}, head_path)
    meta = Emotion2vecModelMeta(
        hmonnx=str(tmp_path / "model.hmonnx"),
        quant_embedding=str(head_path),
        sampling_rate=16000,
        window_samples=256000,
        feature_dim=8,
    )
    model = Emotion2vecHMONNXModel(meta)
    model.session = _FakeSession()

    wav = np.linspace(0.0, 1.0, 320000, dtype=np.float32)
    result = model.extract_waveform(wav, sampling_rate=16000)

    assert result["frame_features"].shape == (4, 8)
    assert result["utterance_feature"].shape == (8,)
    assert result["logits"].shape == (9,)
    assert result["probabilities"].shape == (9,)
    torch.testing.assert_close(
        result["logits"],
        torch.nn.functional.linear(result["utterance_feature"], head_weight, head_bias),
    )
    assert result["predicted_label"] == "<unk>"
    assert result["labels"] == ["生气/angry", "开心/happy", "中立/neutral", "难过/sad", "<unk>"]
    assert len(result["scores"]) == 5
    assert result["chunk_count"] == 2
    assert result["valid_frame_count"] == 4
    assert model.session.calls[0][0].dtype == torch.float16
    assert model.session.calls[0][1].dtype == torch.int32
    assert int(model.session.calls[0][1].item()) == 799
    assert int(model.session.calls[1][1].item()) == 199
    assert torch.isfinite(model.session.calls[0][0]).all()
    assert abs(float(model.session.calls[0][0].float().mean())) < 1e-3


def test_runtime_rejects_mismatched_meta_path(tmp_path):
    from xhmodel_merak.xh_llm.models.emotion2vec.configuration_emotion2vec import Emotion2vecModelMeta
    from xhmodel_merak.xh_llm.models.emotion2vec.emotion2vec_hmonnx_inference import Emotion2vecHMONNXModel

    meta_file = tmp_path / "emotion2vec_meta.json"
    meta_file.write_text("{}", encoding="utf-8")
    meta = Emotion2vecModelMeta.from_dict({"_meta_path_": str(meta_file), "hmonnx": "model.hmonnx"})
    model = Emotion2vecHMONNXModel(meta)

    assert model.meta_info.hmonnx == str(tmp_path / "model.hmonnx")


def test_runtime_single_window_uses_hmonnx_probabilities_without_external_head(tmp_path):
    from xhmodel_merak.xh_llm.models.emotion2vec.configuration_emotion2vec import Emotion2vecModelMeta
    from xhmodel_merak.xh_llm.models.emotion2vec.emotion2vec_hmonnx_inference import Emotion2vecHMONNXModel

    meta = Emotion2vecModelMeta(hmonnx=str(tmp_path / "model.hmonnx"), feature_dim=8)
    model = Emotion2vecHMONNXModel(meta)
    model.session = _FakeSession()

    result = model.extract_waveform(np.ones(16000, dtype=np.float32), sampling_rate=16000)

    assert result["logits"] is None
    assert result["probabilities"].shape == (9,)
