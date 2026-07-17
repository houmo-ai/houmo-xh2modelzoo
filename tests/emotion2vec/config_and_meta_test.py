from __future__ import annotations


def test_emotion2vec_config_defaults_target_plus_large_and_w8a8():
    from xhmodel_merak.xh_llm.models.emotion2vec.configuration_emotion2vec import (
        Emotion2vecModelMeta,
        XHEmotion2vecConfig,
    )

    cfg = XHEmotion2vecConfig(model_name="emotion2vec_plus_large", hf_model="/tmp/emotion2vec")

    assert cfg.model_id == "iic/emotion2vec_plus_large"
    assert cfg.sampling_rate == 16000
    assert cfg.window_samples == 256000
    assert cfg.feature_dim == 1024
    assert cfg.num_labels == 9
    assert cfg.model_type == "Emotion2vecForEmotionRecognition"
    assert cfg.quant_scheme.quant_type == "w8a8h1_sefp"

    meta = Emotion2vecModelMeta(
        hmonnx="artifacts/emotion2vec.hmonnx",
        onnx="artifacts/emotion2vec.onnx",
        quant_embedding="hmquant/quant_embedding.pt",
        quant_embedding_md5="abc123",
        sampling_rate=cfg.sampling_rate,
        window_samples=cfg.window_samples,
        model_config=cfg.to_dict(),
    )
    payload = meta.to_dict()
    assert payload["hmonnx"] == "artifacts/emotion2vec.hmonnx"
    assert payload["onnx"] == "artifacts/emotion2vec.onnx"
    assert payload["quant_embedding"] == "hmquant/quant_embedding.pt"
    assert payload["quant_embedding_md5"] == "abc123"
    assert payload["model_config"]["model_type"] == "Emotion2vecForEmotionRecognition"
    assert payload["feature_dim"] == 1024
    assert payload["num_labels"] == 9
    assert payload["labels"][0] == "生气/angry"


def test_emotion2vec_meta_resolves_relative_paths_from_meta_file(tmp_path):
    from xhmodel_merak.xh_llm.models.emotion2vec.configuration_emotion2vec import Emotion2vecModelMeta

    meta_path = tmp_path / "emotion2vec_meta.json"
    meta = Emotion2vecModelMeta.from_dict(
        {
            "_meta_path_": str(meta_path),
            "hmonnx": "export/model.hmonnx",
            "onnx": "export/model.onnx",
            "golden_dir": "golden",
            "quant_embedding": "hmquant/quant_embedding.pt",
            "model_config": {"model_type": "Emotion2vecForSequenceEmbedding"},
        }
    )

    assert meta.hmonnx == str(tmp_path / "export/model.hmonnx")
    assert meta.onnx == str(tmp_path / "export/model.onnx")
    assert meta.golden_dir == str(tmp_path / "golden")
    assert meta.quant_embedding == str(tmp_path / "hmquant/quant_embedding.pt")
    assert meta.model_config.model_type == "Emotion2vecForSequenceEmbedding"
