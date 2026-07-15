from __future__ import annotations


def test_iemocap_session_split_and_metrics_contract():
    from xhmodel_merak.xh_llm.models.emotion2vec.iemocap_protocol import (
        IEMOCAP_LABELS,
        compute_metrics,
        session_from_utterance_id,
        split_leave_one_session_out,
    )

    assert IEMOCAP_LABELS == ("ang", "hap", "neu", "sad")
    assert session_from_utterance_id("Ses03F_script02_1") == "Ses03"

    train, valid = split_leave_one_session_out(["Ses01", "Ses02", "Ses03"], leave_out="Ses02")
    assert train == ["Ses01", "Ses03"]
    assert valid == ["Ses02"]

    metrics = compute_metrics([0, 1, 2, 3], [0, 1, 1, 3], num_classes=4)
    assert metrics["wa"] == 0.75
    assert metrics["ua"] > 0.0
    assert metrics["weighted_f1"] > 0.0


def test_iemocap_official_session_boundaries():
    from xhmodel_merak.xh_llm.models.emotion2vec.iemocap_protocol import IEMOCAP_SESSION_COUNTS

    assert IEMOCAP_SESSION_COUNTS == (1085, 1023, 1151, 1031, 1241)
    assert sum(IEMOCAP_SESSION_COUNTS) == 5531


def test_official_classifier_ignores_padded_frames():
    import importlib.util
    from pathlib import Path

    import torch

    script = Path(__file__).parents[2] / "examples_merak/audio/emotion2vec/debug/iemocap_hmonnx_eval.py"
    spec = importlib.util.spec_from_file_location("emotion2vec_iemocap_eval", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    model = module.OfficialIEMOCAPClassifier(input_dim=2, output_dim=4)
    features = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]]])
    padding_mask = torch.tensor([[False, False, True]])
    logits_a = model(features, padding_mask)
    features[:, 2] = -999.0
    logits_b = model(features, padding_mask)
    assert torch.allclose(logits_a, logits_b)
