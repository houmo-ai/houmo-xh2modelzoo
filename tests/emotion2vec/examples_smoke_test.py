from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow
from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig
from xhmodel_merak.xh_llm.workflows.result import ExportResult


WORKFLOW_CONFIG = Path(
    "configs_merak/workflows/xh2a/audio_models/emotion2vec/"
    "emotion2vec_plus_large_xh2a_w8a8_16s.yaml"
)


def test_example_modules_import_and_build_parsers():
    import examples_merak.audio.emotion2vec.debug.compare_golden as compare_golden
    import examples_merak.audio.emotion2vec.debug.download_model as download_model
    import examples_merak.audio.emotion2vec.debug.iemocap_hmonnx_eval as iemocap_hmonnx_eval
    import examples_merak.audio.emotion2vec.export_hmonnx as export_hmonnx
    import examples_merak.audio.emotion2vec.hmonnx_infer as hmonnx_infer

    for module in (download_model, export_hmonnx, hmonnx_infer, compare_golden, iemocap_hmonnx_eval):
        assert hasattr(module, "build_parser")
        parser = module.build_parser()
        assert parser is not None


def test_example_defaults_target_plus_large():
    import examples_merak.audio.emotion2vec.debug.download_model as download_model
    import examples_merak.audio.emotion2vec.debug.iemocap_hmonnx_eval as iemocap_hmonnx_eval
    import examples_merak.audio.emotion2vec.export_hmonnx as export_hmonnx

    download_args = download_model.build_parser().parse_args([])
    assert download_args.model_id == "iic/emotion2vec_plus_large"
    assert download_args.output_dir == "data/models/emotion2vec_plus_large"

    export_args = export_hmonnx.build_parser().parse_args([])
    assert export_args.model_dir == "data/models/emotion2vec_plus_large"
    assert export_args.config_path == str(WORKFLOW_CONFIG)
    assert export_args.output_dir == "work_dirs/emotion2vec_plus_large_xh2a_w8a8_16s"
    assert export_args.dump_golden is False
    assert export_args.golden_audio == "data/models/emotion2vec_plus_large/example/test.wav"

    eval_args = iemocap_hmonnx_eval.build_parser().parse_args(
        ["--iemocap-root", "/tmp/iemocap", "--feature-dir", "/tmp/features"]
    )
    assert eval_args.feature_dim == 1024


def test_iemocap_classifier_supports_plus_large_feature_dim():
    import examples_merak.audio.emotion2vec.debug.iemocap_hmonnx_eval as iemocap_hmonnx_eval

    model = iemocap_hmonnx_eval.OfficialIEMOCAPClassifier(input_dim=1024)

    assert model.pre_net.in_features == 1024


def test_compare_golden_only_compares_workflow_outputs():
    import examples_merak.audio.emotion2vec.debug.compare_golden as compare_golden

    assert not hasattr(compare_golden, "generate_pytorch_golden")


def test_emotion2vec_workflow_yaml_uses_audio_model_layout_and_w8a8():
    workflow_config = WorkflowConfig.from_file(str(WORKFLOW_CONFIG))

    assert workflow_config.quant is None
    assert workflow_config.export["model"] == {
        "chip_arch": "XH2a",
        "model_type": "Emotion2vecForEmotionRecognition",
        "hf_model": None,
        "model_name": "emotion2vec_plus_large",
        "model_id": "iic/emotion2vec_plus_large",
        "sampling_rate": 16000,
        "window_samples": 256000,
        "feature_dim": 1024,
        "num_labels": 9,
        "use_cache": False,
        "quant_scheme": {"quant_type": "w8a8h1_sefp"},
    }


def test_auto_workflow_resolves_emotion2vec_workflow(tmp_path):
    from xhmodel_merak.xh_llm.models.emotion2vec.workflow import Emotion2vecWorkflow

    model_dir = tmp_path / "emotion2vec_plus_large"
    model_dir.mkdir()

    workflow = AutoLLMWorkflow.from_config(
        model_dir=str(model_dir),
        config_path=str(WORKFLOW_CONFIG),
    )

    assert type(workflow) is Emotion2vecWorkflow


def test_emotion2vec_workflow_model_config_round_trips(tmp_path):
    from xhmodel_merak.xh_llm import AutoLLMConfig

    model_dir = tmp_path / "emotion2vec_plus_large"
    model_dir.mkdir()
    workflow_config = WorkflowConfig.from_file(str(WORKFLOW_CONFIG))
    model_config = dict(workflow_config.export["model"])
    model_config["hf_model"] = str(model_dir)

    resolved = AutoLLMConfig.from_pretrained(model_config)

    assert resolved.model_type == "Emotion2vecForEmotionRecognition"
    assert resolved.window_samples == 256000
    assert resolved.use_cache is False
    assert resolved.quant_scheme.quant_type == "w8a8h1_sefp"


def test_emotion2vec_workflow_dump_golden_uses_standard_api(monkeypatch, tmp_path):
    import soundfile as sf

    import xhmodel_merak.xh_llm.models.emotion2vec.workflow as workflow_module
    from xhmodel_merak.xh_llm.models.emotion2vec.configuration_emotion2vec import Emotion2vecModelMeta
    from xhmodel_merak.xh_llm.models.emotion2vec.workflow import Emotion2vecWorkflow

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    audio_path = tmp_path / "test.wav"
    sf.write(audio_path, np.linspace(-0.1, 0.1, 1600, dtype=np.float32), 16000)
    export_dir = tmp_path / "export"
    export_dir.mkdir()
    hmonnx_file = export_dir / "model.onnx"
    hmonnx_file.touch()
    meta = Emotion2vecModelMeta(hmonnx=str(hmonnx_file), window_samples=256000, feature_dim=1024)
    (export_dir / "emotion2vec_meta.json").write_text("{}", encoding="utf-8")

    class FakeBridge(torch.nn.Module):
        def forward(self, waveform, valid_frames):
            assert waveform.device.type == "cpu"
            assert valid_frames.dtype == torch.int32
            features = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
            mask = torch.tensor([[False, False, True]])
            utterance = features[:, :2].mean(dim=1)
            return features, mask, utterance

    class FakeNativeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(8, 9)

    monkeypatch.setattr(workflow_module, "load_funasr_emotion2vec_model", lambda path: FakeNativeModel())
    monkeypatch.setattr(workflow_module, "Emotion2vecReferenceModel", lambda model, window_samples: FakeBridge())
    operator_golden_calls = []

    def fake_dump_operator_golden(*, hmonnx_file, golden_dir, device, inputs):
        operator_golden_calls.append((hmonnx_file, golden_dir, device, inputs))
        step_dir = golden_dir / "step_0"
        step_dir.mkdir(parents=True)
        np.save(step_dir / "node_less.npy", np.zeros((1, 799), dtype=np.bool_))

    monkeypatch.setattr(
        Emotion2vecWorkflow,
        "_dump_hmonnx_operator_golden",
        staticmethod(fake_dump_operator_golden),
    )
    workflow = Emotion2vecWorkflow(str(model_dir), str(WORKFLOW_CONFIG))
    export_result = ExportResult(
        work_dir=str(export_dir),
        config_file=str(WORKFLOW_CONFIG),
        meta=meta,
    )

    golden_dir = workflow.dump_golden(
        export_result=export_result,
        device="cpu",
        input_messages={"audio": str(audio_path)},
    )

    assert golden_dir == str(export_dir / "golden")
    assert len(operator_golden_calls) == 1
    assert operator_golden_calls[0][3][0].dtype == torch.float16
    assert operator_golden_calls[0][3][1].dtype == torch.int32
    assert int(operator_golden_calls[0][3][1].item()) == 4
    assert (export_dir / "golden/step_0/node_less.npy").is_file()
    reference_dir = export_dir / "golden/reference"
    assert np.load(reference_dir / "frame_features.npy").shape == (2, 8)
    assert np.load(reference_dir / "utterance_feature.npy").shape == (8,)
    assert np.load(reference_dir / "logits.npy").shape == (9,)
    assert np.load(reference_dir / "probabilities.npy").shape == (9,)


def test_emotion2vec_workflow_dump_golden_requires_audio(tmp_path):
    from xhmodel_merak.xh_llm.models.emotion2vec.workflow import Emotion2vecWorkflow

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    workflow = Emotion2vecWorkflow(str(model_dir), str(WORKFLOW_CONFIG))

    with pytest.raises(ValueError, match="audio"):
        workflow.dump_golden(
            export_result=ExportResult(
                work_dir=str(tmp_path / "export"),
                config_file=str(WORKFLOW_CONFIG),
            ),
            device="cpu",
            input_messages={},
        )
