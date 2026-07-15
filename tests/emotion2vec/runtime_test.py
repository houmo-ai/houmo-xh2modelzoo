from __future__ import annotations

import numpy as np
import torch


class _FakeSession:
    def __init__(self):
        self.calls = []

    def forward(self, waveform, valid_samples):
        self.calls.append((waveform.clone(), valid_samples.clone()))
        frame_features = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
        frame_mask = torch.tensor([[False, False, True]])
        return frame_features, frame_mask


def test_runtime_extract_waveform_chunks_and_pools_frames(tmp_path):
    from xhmodel_merak.xh_llm.models.emotion2vec.configuration_emotion2vec import Emotion2vecModelMeta
    from xhmodel_merak.xh_llm.models.emotion2vec.emotion2vec_hmonnx_inference import Emotion2vecHMONNXModel

    meta = Emotion2vecModelMeta(hmonnx=str(tmp_path / "model.hmonnx"), sampling_rate=16000, window_samples=256000)
    model = Emotion2vecHMONNXModel(meta)
    model.session = _FakeSession()

    wav = np.linspace(0.0, 1.0, 320000, dtype=np.float32)
    result = model.extract_waveform(wav, sampling_rate=16000)

    assert result["frame_features"].shape == (4, 8)
    assert result["utterance_feature"].shape == (8,)
    assert result["chunk_count"] == 2
    assert result["valid_frame_count"] == 4
    assert model.session.calls[0][0].dtype == torch.float16
    assert model.session.calls[0][1].dtype == torch.int32
    assert int(model.session.calls[0][1].item()) == 256000
    assert int(model.session.calls[1][1].item()) == 64000
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
