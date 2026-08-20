from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from xhmodel_merak.workflows import AutoWorkflow
from xhmodel_merak.xh_llm.workflows.result import ExportResult, QuantResult


CONFIG_DIR = Path("configs_merak/workflows/xh2a/llm_models/minicpm_o_4_5")
CONFIG_PATH = CONFIG_DIR / "minicpm_o_4_5_xh2a_w8a8_gptq.yaml"
W4_CONFIG_PATH = Path("configs_merak/workflows/xh2a/llm_models/minicpm_o_4_5/minicpm_o_4_5_xh2a_w4a8_gptq.yaml")
WORKFLOW_DEMO_PATH = Path("examples_merak/llm/minicpm_o_4_5/minicpm_o_4_5_workflow.py")
CALIBRATION_PATH = Path("data/calib_data/minicpm_o_4_5_qwen_vl_style_mix80.jsonl")


def test_minicpm_o_4_5_example_layout_uses_llm_directory_without_eval() -> None:
    root = Path("examples_merak/llm/minicpm_o_4_5")

    assert (root / "minicpm_o_4_5_workflow.py").is_file()
    assert (root / "minicpm_o_4_5_hf_demo.py").is_file()
    assert (root / "minicpm_o_4_5_hmonnx_demo.py").is_file()
    assert not Path("examples_merak/omni/minicpm_o_4_5").exists()
    assert not (root / "eval").exists()


def test_workflow_cli_accepts_real_golden_media_arguments(monkeypatch) -> None:
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("minicpm_o_4_5_workflow_demo", WORKFLOW_DEMO_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(WORKFLOW_DEMO_PATH),
            "--model-dir",
            "/models/minicpm",
            "--dump-golden",
            "--golden-mode",
            "real",
            "--golden-video",
            "/tmp/video.mp4",
            "--golden-question",
            "describe",
        ],
    )

    args = module.parse_args()

    assert args.golden_video == Path("/tmp/video.mp4")
    assert args.golden_question == "describe"


def test_minicpm_o_4_5_yaml_declares_all_component_contracts() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    assert config["export"]["model"]["model_type"] == "MiniCPMO45Model"
    assert config["export"]["model"]["chip_arch"] == "XH2a"
    assert config["export"]["model"]["model_name"] == "minicpm_o_4_5"
    assert config["export"]["model"]["quant_scheme"]["quant_type"] == "w8a8_sefp"
    assert config["export"]["components"]["llm"]["wrap_cfg"]["max_sequence_length"] == 32768
    assert config["export"]["components"]["llm"]["wrap_cfg"]["input_sequence_length"] == 512
    assert set(config["export"]["components"]) == {
        "vision",
        "audio",
        "llm",
        "tts",
        "speaker",
        "token2wav_flow_frontend",
        "token2wav_flow_decoder",
        "token2wav_hift",
    }
    assert all(component["quant_type"] for component in config["export"]["components"].values())
    assert config["export"]["components"]["vision"]["image_slice_max_size"] == [40, 40]
    assert config["export"]["components"]["audio"]["static_batch_size"] >= 1
    assert config["export"]["components"]["audio"]["quant_type"] == "w8a16_sefp"
    assert config["export"]["components"]["tts"]["quant_type"] == "w8a8_sefp"
    assert config["export"]["components"]["audio"]["streaming"] == {
        "enabled": True,
        "first_chunk_ms": 1035,
        "chunk_ms": 1000,
        "cnn_redundancy_ms": 20,
        "sample_rate": 16000,
        "prefix_overlap_first": 0,
        "prefix_overlap_later": 2,
        "suffix_overlap": 2,
        "cache_capacity": 1500,
        "session_frames": 100,
    }
    assert config["export"]["components"]["llm"]["wrap_cfg"]["use_cache"] is True
    assert config["export"]["components"]["tts"]["wrap_cfg"]["use_cache"] is True
    assert config["export"]["components"]["token2wav_flow_frontend"]["quant_type"] == "w8a16_sefp"
    assert config["export"]["components"]["token2wav_flow_decoder"]["quant_type"] == "w8a16_sefp"
    assert config["export"]["components"]["token2wav_flow_decoder"]["mel_capacity"] == 2048
    assert config["export"]["components"]["token2wav_hift"]["frame_capacity"] == 1024


def test_w4_and_w8_configs_have_single_sources_for_export_contracts() -> None:
    expected_prefill = {CONFIG_PATH: 512, W4_CONFIG_PATH: 256}
    for config_path, prefill_length in expected_prefill.items():
        export = yaml.safe_load(config_path.read_text(encoding="utf-8"))["export"]
        assert export["model"]["model_type"] == "MiniCPMO45Model"
        assert export["model"]["chip_arch"] == "XH2a"
        assert export["model"]["model_name"] == "minicpm_o_4_5"
        assert export["components"]["llm"]["wrap_cfg"]["max_sequence_length"] == 32768
        assert export["components"]["llm"]["wrap_cfg"]["input_sequence_length"] == prefill_length
        assert export["components"]["token2wav_flow_decoder"]["noise_seed"] == 20260821
        assert all("enabled" not in component for component in export["components"].values())


def test_only_formal_g64_configs_are_shipped() -> None:
    assert {path.name for path in CONFIG_DIR.glob("*.yaml")} == {
        "minicpm_o_4_5_xh2a_w8a8_gptq.yaml",
        "minicpm_o_4_5_xh2a_w4a8_gptq.yaml",
    }

    expected = {
        CONFIG_PATH: (8, "w8a8_sefp", 512),
        W4_CONFIG_PATH: (4, "w4a8h0_ssfp", 256),
    }
    for config_path, (bits, llm_quant_type, prefill_length) in expected.items():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert config["quant"] == {
            "algorithm": "gptqmodel",
            "bits": bits,
            "group_size": 64,
            "nsamples": 80,
            "seqlen": 1024,
            "calibration_jsonl": f"xh2modelzoo://{CALIBRATION_PATH.as_posix()}",
            **({"llm_backbone_dir": None} if bits == 4 else {}),
        }
        llm = config["export"]["components"]["llm"]
        assert llm["quant_type"] == llm_quant_type
        if bits == 4:
            assert llm["nodes"] == {"lm_head": {"quant_type": "w8a8_sefp"}}
        else:
            assert "nodes" not in llm
        assert llm["wrap_cfg"]["input_sequence_length"] == prefill_length
        assert "calibration_jsonl" not in llm
        assert "calibration_samples" not in llm
        assert config["export"]["components"]["tts"]["quant_type"] == "w8a8_sefp"


def test_w4_config_overrides_only_lm_head_quant_type() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_common import quant_config

    llm = yaml.safe_load(W4_CONFIG_PATH.read_text(encoding="utf-8"))["export"]["components"]["llm"]
    config = quant_config("XH2a", llm)

    assert config["nodes_cfg"] == {"lm_head": {"quant_type": "w8a8_sefp"}}


def test_w4_and_w8_configs_use_balanced_vl_wikitext_calibration() -> None:
    records = [json.loads(line) for line in CALIBRATION_PATH.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 80
    assert [record["source"] for record in records[:5]] == ["CMMMU", "COCO", "DocVQA", "MMMU", "Wikitext"]
    counts = {
        source: sum(record["source"] == source for record in records)
        for source in set(record["source"] for record in records)
    }
    assert counts == {
        "CMMMU": 18,
        "COCO": 18,
        "DocVQA": 18,
        "MMMU": 18,
        "Wikitext": 8,
    }

    w8 = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    w4 = yaml.safe_load(W4_CONFIG_PATH.read_text(encoding="utf-8"))
    w8_llm = w8["export"]["components"]["llm"]
    w4_llm = w4["export"]["components"]["llm"]
    assert w8["quant"]["calibration_jsonl"] == f"xh2modelzoo://{CALIBRATION_PATH.as_posix()}"
    assert w8["quant"]["group_size"] == 64
    assert w8["quant"]["nsamples"] == 80
    assert "calibration_jsonl" not in w8_llm
    assert "calibration_samples" not in w8_llm
    assert w4["quant"]["calibration_jsonl"] == f"xh2modelzoo://{CALIBRATION_PATH.as_posix()}"
    assert w4["quant"]["group_size"] == 64
    assert w4["quant"]["nsamples"] == 80
    assert "calibration_jsonl" not in w4_llm
    assert "calibration_samples" not in w4_llm


def test_token2wav_streaming_derived_values_are_consistent() -> None:
    """Lock the mathematical relationships between hard-coded streaming values.

    These values are derived from the official reference audio (16.84 s ->
    842 mel frames -> 421 speech tokens + 3 silence pad) and the official
    truncation window (leading prompt-length frames + most recent 100 mel). Any
    edit to one side must keep the relationships below; otherwise the
    exported graph contracts and the runtime capacity checks drift apart.
    """
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    frontend = config["export"]["components"]["token2wav_flow_frontend"]["streaming"]
    decoder = config["export"]["components"]["token2wav_flow_decoder"]["streaming"]

    prompt_mel = frontend["prompt_mel_capacity"]
    prompt_tokens = frontend["prompt_token_capacity"]
    base_valid = frontend["base_cache_valid_length"]
    att_axis = 3
    est_axis = 4
    cache_capacity = frontend["cache_shapes"]["conformer_att_cache"][att_axis]
    append_capacity = decoder["append_capacity"]
    recent_mel_frames = decoder["prompt_cache_policy"]["recent_mel_frames"]

    # Graph cache = full prompt + most recent 100 generated mel frames.
    assert cache_capacity == prompt_mel + recent_mel_frames
    assert base_valid == prompt_mel
    # Official set_stream_cache right-pads 3 silence tokens: 421 + 3 = 424.
    assert prompt_tokens == prompt_mel // 2 + 3

    # Cache shapes must match the configured capacities on the cache axis.
    assert frontend["cache_shapes"]["conformer_att_cache"][att_axis] == cache_capacity
    assert frontend["cache_shapes"]["estimator_att_cache"][est_axis] == cache_capacity
    assert frontend["base_cache_shapes"]["conformer_att_cache"][att_axis] == base_valid
    assert frontend["base_cache_shapes"]["estimator_att_cache"][est_axis] == base_valid

    # Decoder streaming block must agree with the frontend block.
    assert decoder["base_cache_valid_length"] == base_valid
    assert decoder["prompt_cache_policy"]["recent_mel_frames"] == recent_mel_frames
    assert frontend["cache_alignment"] == "right"
    assert frontend["base_conformer_layers"] == 6

    # The estimator graph returns current-capacity frames followed by the
    # fixed-capacity past cache; Host compacts both padding regions afterwards.
    estimator = decoder["estimator_step_cache_shapes"]
    assert estimator["input_att"][3] == cache_capacity
    assert estimator["output_att"][3] == cache_capacity + append_capacity
    assert estimator["frame_capacity"] == frontend["roles"]["stream_flow_frontend_final"]["output_mel_capacity"]


def test_auto_workflow_binds_minicpm_o_4_5() -> None:
    workflow = AutoWorkflow.from_config(
        model_dir="/models/MiniCPM-o-4_5",
        config_path=str(CONFIG_PATH),
    )

    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.workflow import MiniCPMO45Workflow

    assert isinstance(workflow, MiniCPMO45Workflow)


def test_workflow_rejects_redundant_model_fields_and_component_switches() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.workflow import MiniCPMO45Workflow

    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["export"]
    missing_model_type = {
        **config,
        "model": {key: value for key, value in config["model"].items() if key != "model_type"},
    }
    with pytest.raises(ValueError, match="export.model.model_type"):
        MiniCPMO45Workflow._validate_export_config(missing_model_type)

    wrong_model_type = {**config, "model": {**config["model"], "model_type": "OtherModel"}}
    with pytest.raises(ValueError, match="must be 'MiniCPMO45Model'"):
        MiniCPMO45Workflow._validate_export_config(wrong_model_type)

    mismatched_quant_type = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["export"]
    mismatched_quant_type["components"]["llm"]["quant_type"] = "w4a8h0_ssfp"
    with pytest.raises(ValueError, match="quant_scheme.quant_type must match"):
        MiniCPMO45Workflow._validate_export_config(mismatched_quant_type)

    disabled = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["export"]
    disabled["components"]["audio"]["enabled"] = False
    with pytest.raises(ValueError, match="omit redundant enabled fields: audio"):
        MiniCPMO45Workflow._validate_export_config(disabled)

    invalid_context = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))["export"]
    invalid_context["components"]["llm"]["wrap_cfg"]["max_sequence_length"] = 0
    with pytest.raises(ValueError, match="max_sequence_length must be positive"):
        MiniCPMO45Workflow._validate_export_config(invalid_context)


def test_workflow_rejects_gptq_export_config_mismatches() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.workflow import MiniCPMO45Workflow

    data = yaml.safe_load(W4_CONFIG_PATH.read_text(encoding="utf-8"))
    export = data["export"]
    quant = data["quant"]
    MiniCPMO45Workflow._validate_export_config(export, quant)

    wrong_quant_type = yaml.safe_load(W4_CONFIG_PATH.read_text(encoding="utf-8"))["export"]
    wrong_quant_type["components"]["llm"]["quant_type"] = "w8a8_sefp"
    with pytest.raises(ValueError, match="GPTQ bits=4 requires components.llm.quant_type=w4a8h0_ssfp"):
        MiniCPMO45Workflow._validate_export_config(wrong_quant_type, quant)

    wrong_prefill = yaml.safe_load(W4_CONFIG_PATH.read_text(encoding="utf-8"))["export"]
    wrong_prefill["components"]["llm"]["wrap_cfg"]["input_sequence_length"] = 512
    with pytest.raises(ValueError, match="W4A8 export requires.*input_sequence_length=256"):
        MiniCPMO45Workflow._validate_export_config(wrong_prefill, quant)


def test_minicpm_o_4_5_uses_flat_lifecycle_component_modules() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import (
        export_audio,
        export_common,
        export_llm,
        export_token2wav,
        export_tts,
        export_vision,
        runtime,
        runtime_audio,
        runtime_llm,
        runtime_token2wav,
        runtime_tts,
        runtime_vision,
    )

    assert export_vision.export_minicpm_o_4_5_vision
    assert export_audio.export_minicpm_o_4_5_audio
    assert export_llm.export_minicpm_o_4_5_llm
    assert export_tts.export_minicpm_o_4_5_tts
    assert export_token2wav.export_minicpm_o_4_5_token2wav_hift
    assert export_common.quantize_and_export
    assert runtime.MiniCPMO45HMONNXRuntime
    assert runtime_vision.MiniCPMO45VisionHMONNXRuntime
    assert runtime_audio.MiniCPMO45AudioHMONNXRuntime
    assert runtime_llm.MiniCPMO45LLMHMONNXRuntime
    assert runtime_tts.MiniCPMO45TTSHMONNXRuntime
    assert runtime_token2wav.MiniCPMO45Token2WavHMONNXRuntime


def test_relative_graph_path_removes_relative_work_dir_prefix() -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.export_common import relative_artifact_path

    path = relative_artifact_path(
        Path("work_dirs/minicpm_o_4_5_xh2a_w8a8_gptq_export"),
        Path("work_dirs/minicpm_o_4_5_xh2a_w8a8_gptq_export/token2wav_hift/model.onnx"),
    )

    assert path == "token2wav_hift/model.onnx"


def test_export_dispatches_all_components_and_writes_relative_metadata(tmp_path, monkeypatch) -> None:
    workflow = AutoWorkflow.from_config(
        model_dir="/models/MiniCPM-o-4_5",
        config_path=str(CONFIG_PATH),
    )
    calls: list[str] = []
    seen_llm_cfg: dict[str, object] = {}
    seen_llm_kwargs: dict[str, object] = {}

    import xhmodel_merak.xh_llm.models.minicpm_o_4_5.workflow as workflow_module

    def fake_export(name: str):
        def export_component(**kwargs):
            calls.append(name)
            if name == "llm":
                seen_llm_cfg.update(kwargs["component_cfg"])
                seen_llm_kwargs.update(kwargs)
            component_dir = kwargs["work_dir"] / name
            component_dir.mkdir(parents=True, exist_ok=True)
            result = {
                "graphs": {"main": f"{name}/{name}_XH2a_w8a8_sefp.onnx"},
                "quant_type": "w8a8_sefp",
            }
            if name == "token2wav_hift":
                result["phase_noise_file"] = component_dir / "phase_noise.pt"
                result["source_noise_file"] = component_dir / "source_noise.pt"
            return result

        return export_component

    component_names = (
        "vision",
        "audio",
        "llm",
        "tts",
        "speaker",
        "token2wav_flow_frontend",
        "token2wav_flow_decoder",
        "token2wav_hift",
    )
    for name in component_names:
        if name == "speaker":
            # workflow dispatches the speaker component through the
            # campplus/speech_tokenizer sub-exporters, not a single function.
            continue
        monkeypatch.setattr(workflow_module, f"export_minicpm_o_4_5_{name}", fake_export(name))

    def fake_speaker_export(role: str):
        def export_component(**kwargs):
            calls.append(f"speaker/{role}")
            component_dir = kwargs["work_dir"] / "speaker"
            component_dir.mkdir(parents=True, exist_ok=True)
            return {
                "quant_type": "w8a16_sefp",
                "graphs": {role: f"speaker/{role}_XH2a_w8a16_sefp.onnx"},
            }

        return export_component

    monkeypatch.setattr(workflow_module, "export_minicpm_o_4_5_campplus", fake_speaker_export("campplus"))
    monkeypatch.setattr(
        workflow_module, "export_minicpm_o_4_5_speech_tokenizer", fake_speaker_export("speech_tokenizer")
    )

    result = workflow.export(
        quant_result=QuantResult(raw_model_dir="/models/MiniCPM-o-4_5", skipped=True),
        output_dir=str(tmp_path / "export"),
        device="cpu",
    )

    expected_calls = []
    for name in component_names:
        if name == "speaker":
            expected_calls += ["speaker/campplus", "speaker/speech_tokenizer"]
        else:
            expected_calls.append(name)
    assert calls == expected_calls
    meta = json.loads((Path(result.work_dir) / "export_meta_info.json").read_text(encoding="utf-8"))
    assert set(meta["components"]) == set(component_names)
    assert meta["model_name"] == "minicpm_o_4_5"
    assert meta["chip_arch"] == "XH2a"
    assert meta["export_basename"].startswith("hmquant_xh2_minicpm_o_4_5_w8a8_512_32k_")
    assert seen_llm_cfg["calibration_jsonl"] == f"xh2modelzoo://{CALIBRATION_PATH.as_posix()}"
    assert seen_llm_cfg["calibration_samples"] == 5
    assert all(not Path(path).is_absolute() for item in meta["components"].values() for path in item["graphs"].values())
    assert meta["components"]["token2wav_hift"]["phase_noise_file"] == "token2wav_hift/phase_noise.pt"
    assert meta["components"]["token2wav_hift"]["source_noise_file"] == "token2wav_hift/source_noise.pt"
    assert Path(result.config_file).exists()

    calls.clear()
    seen_llm_kwargs.clear()
    workflow.export(
        quant_result=QuantResult(
            raw_model_dir="/models/MiniCPM-o-4_5",
            quanted_model_dir="/artifacts/gptq_llm",
        ),
        output_dir=str(tmp_path / "export_gptq"),
        device="cpu",
    )
    assert calls == expected_calls
    assert seen_llm_kwargs["model_dir"] == "/models/MiniCPM-o-4_5"
    assert seen_llm_kwargs["gptq_llm_model_dir"] == "/artifacts/gptq_llm"


def test_export_writes_incremental_metadata_before_component_export(tmp_path, monkeypatch) -> None:
    workflow = AutoWorkflow.from_config(
        model_dir="/models/MiniCPM-o-4_5",
        config_path=str(CONFIG_PATH),
    )
    observed: list[dict[str, object]] = []

    import xhmodel_merak.xh_llm.models.minicpm_o_4_5.workflow as workflow_module

    def export_component(**kwargs):
        work_dir = kwargs["work_dir"]
        meta = json.loads((work_dir / "export_meta_info.json").read_text(encoding="utf-8"))
        observed.append(meta)
        name = "vision"
        component_dir = work_dir / name
        component_dir.mkdir(parents=True, exist_ok=True)
        return {"graphs": {"main": f"{name}/{name}.onnx"}, "quant_type": "w8a8_sefp"}

    monkeypatch.setattr(workflow_module, "export_minicpm_o_4_5_vision", export_component)
    for name in workflow.SUPPORTED_COMPONENTS[1:]:
        if name == "speaker":
            # speaker dispatches through campplus/speech_tokenizer sub-exporters.
            continue
        monkeypatch.setattr(
            workflow_module,
            f"export_minicpm_o_4_5_{name}",
            lambda **kwargs: {"graphs": {}, "quant_type": "w8a8_sefp"},
        )

    def noop_speaker_export(role: str):
        def export_component(**kwargs):
            return {
                "quant_type": "w8a16_sefp",
                "graphs": {role: f"speaker/{role}.onnx"},
            }

        return export_component

    monkeypatch.setattr(workflow_module, "export_minicpm_o_4_5_campplus", noop_speaker_export("campplus"))
    monkeypatch.setattr(
        workflow_module,
        "export_minicpm_o_4_5_speech_tokenizer",
        noop_speaker_export("speech_tokenizer"),
    )

    workflow.export(
        quant_result=QuantResult(raw_model_dir="/models/MiniCPM-o-4_5", skipped=True),
        output_dir=str(tmp_path / "export"),
        device="cpu",
    )

    assert observed == [
        {
            "schema_version": 1,
            "create_time": observed[0]["create_time"],
            "model_type": "MiniCPM-o-4.5",
            "model_name": "minicpm_o_4_5",
            "target_device": "XH2a",
            "chip_arch": "XH2a",
            "export_basename": observed[0]["export_basename"],
            "hf_model": "/models/MiniCPM-o-4_5",
            "config": "minicpm_o_4_5_xh2a_w8a8_gptq.yaml",
            "components": {},
        }
    ]


def test_dump_golden_uses_export_result_work_dir(tmp_path, monkeypatch) -> None:
    workflow = AutoWorkflow.from_config(
        model_dir="/models/MiniCPM-o-4_5",
        config_path=str(CONFIG_PATH),
    )
    work_dir = tmp_path / "export"
    work_dir.mkdir()
    (work_dir / "export_meta_info.json").write_text(
        json.dumps(
            {
                "components": {
                    "vision": {"graphs": {"main": "vision/model.onnx"}},
                    "audio": {"graphs": {"main": "audio/model.onnx"}},
                    "llm": {"graphs": {"prefill": "llm/prefill.onnx", "decode": "llm/decode.onnx"}},
                    "tts": {"graphs": {"prefill": "tts/prefill.onnx", "decode": "tts/decode.onnx"}},
                }
            }
        ),
        encoding="utf-8",
    )
    calls: list[str] = []

    import xhmodel_merak.xh_llm.models.minicpm_o_4_5.workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "dump_synthetic_golden",
        lambda root, meta, device: calls.extend(
            ["vision", "audio", "llm_prefill", "llm_decode", "tts_prefill", "tts_decode"]
        ),
    )

    output = workflow.dump_golden(
        ExportResult(work_dir=str(work_dir), config_file=str(work_dir / "effective.yaml")),
        device="cpu",
        input_messages={"mode": "synthetic"},
    )

    assert calls == ["vision", "audio", "llm_prefill", "llm_decode", "tts_prefill", "tts_decode"]
    assert output == str(work_dir / "golden_meta_info.json")


def test_dump_golden_routes_streaming_case_to_real_capture(tmp_path, monkeypatch) -> None:
    workflow = AutoWorkflow.from_config(
        model_dir="/models/MiniCPM-o-4_5",
        config_path=str(CONFIG_PATH),
    )
    work_dir = tmp_path / "export"
    work_dir.mkdir()
    (work_dir / "export_meta_info.json").write_text(json.dumps({"components": {}}), encoding="utf-8")
    calls: list[object] = []

    import xhmodel_merak.xh_llm.models.minicpm_o_4_5.workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "dump_real_golden",
        lambda root, meta, device, request: calls.extend([root, meta, device, request]) or {"mode": "real"},
    )

    output = workflow.dump_golden(
        ExportResult(work_dir=str(work_dir), config_file=str(work_dir / "effective.yaml")),
        device="cpu",
        input_messages={"streaming_case": "session_audio_text", "media": "tiny.json"},
    )

    assert calls[-1] == {"streaming_case": "session_audio_text", "media": "tiny.json"}
    assert json.loads(Path(output).read_text(encoding="utf-8"))["mode"] == "real"


def test_dump_synthetic_golden_writes_each_component_input_and_output(tmp_path, monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import synthetic_golden as golden_module

    meta = {
        "components": {
            "vision": {"graphs": {"main": "vision/model.onnx"}},
            "token2wav_hift": {
                "graphs": {"main": "token2wav_hift/model.onnx"},
                "frame_capacity": 4,
                "phase_noise_file": "token2wav_hift/phase_noise.pt",
                "source_noise_file": "token2wav_hift/source_noise.pt",
            },
        }
    }
    (tmp_path / "vision").mkdir()
    (tmp_path / "token2wav_hift").mkdir()
    torch.save(torch.zeros((1, 9)), tmp_path / "token2wav_hift/phase_noise.pt")
    torch.save(torch.zeros((1, 1920, 9)), tmp_path / "token2wav_hift/source_noise.pt")

    class Session:
        def __init__(self, path: str) -> None:
            self.path = path
            self.save_golden = False
            self.golden_dir = None
            self.step = None

        def to(self, device: str) -> None:
            del device

        def __call__(self, *inputs: torch.Tensor):
            assert self.save_golden is True
            assert self.golden_dir is not None
            assert self.step == 0
            return inputs[0].sum().reshape(1)

    monkeypatch.setattr(golden_module, "HMONNXGoldenInference", Session)
    golden = golden_module.dump_synthetic_golden(tmp_path, meta, "cpu")

    assert set(golden["components"]) == {"vision_main", "token2wav_hift_main"}
    for component in golden["components"].values():
        assert (tmp_path / component["golden_dir"]).is_dir()


def test_dump_synthetic_golden_covers_tts_projection_graphs(tmp_path, monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import synthetic_golden as golden_module

    meta = {
        "components": {
            "tts": {
                "graphs": {"prefill": "tts/prefill.onnx"},
                "projection_graphs": {
                    "projector_semantic": "tts/projector.onnx",
                    "head_code": "tts/head.onnx",
                },
                "prefill_input_sequence_length": 12,
                "projection_seq_capacity": 12,
                "num_hidden_layers": 1,
                "kv_cache_shape": [1, 2, 16, 4],
            }
        }
    }
    (tmp_path / "tts").mkdir()

    golden_dirs: list[str] = []

    class Session:
        def __init__(self, path: str) -> None:
            self.path = path
            self.save_golden = False
            self.golden_dir = None
            self.step = None

        def to(self, device: str) -> None:
            del device

        def __call__(self, *inputs: torch.Tensor):
            assert self.save_golden is True
            golden_dirs.append(self.golden_dir)
            return inputs[0]

    monkeypatch.setattr(golden_module, "HMONNXGoldenInference", Session)
    monkeypatch.setattr(golden_module, "validate_streaming_graph_coverage", lambda meta: None)
    monkeypatch.setattr(golden_module, "streaming_case_manifest", lambda meta, cases: {})

    golden = golden_module.dump_synthetic_golden(tmp_path, meta, "cpu")

    assert set(golden["components"]) == {"tts_prefill", "tts_projector_semantic", "tts_head_code"}
    assert len(set(golden_dirs)) == 3
    assert all(Path(path).parent.name == "golden" for path in golden_dirs)


def test_synthetic_golden_inputs_match_exported_vision_and_llm_shapes(tmp_path) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5.synthetic_golden import synthetic_golden_inputs

    vision = synthetic_golden_inputs(tmp_path, "vision", "main", {})
    llm = synthetic_golden_inputs(
        tmp_path,
        "llm",
        "prefill",
        {
            "prefill_input_sequence_length": 256,
            "num_hidden_layers": 36,
            "kv_cache_shape": [1, 8, 4096, 128],
        },
    )

    assert vision[0].shape == (1, 3, 14, 22400)
    assert llm[0].shape == (1, 256, 4096)
    assert len(llm) == 3 + 2 * 36


def test_dump_golden_real_mode_delegates_to_real_capture(tmp_path, monkeypatch) -> None:
    workflow = AutoWorkflow.from_config(
        model_dir="/models/MiniCPM-o-4_5",
        config_path=str(CONFIG_PATH),
    )
    work_dir = tmp_path / "export"
    work_dir.mkdir()
    (work_dir / "export_meta_info.json").write_text(json.dumps({"components": {}}), encoding="utf-8")
    observed: list[object] = []

    import xhmodel_merak.xh_llm.models.minicpm_o_4_5.workflow as workflow_module

    monkeypatch.setattr(
        workflow_module,
        "dump_real_golden",
        lambda root, meta, device, messages: observed.extend([root, meta, device, messages]) or {"mode": "real"},
        raising=False,
    )
    output = workflow.dump_golden(
        ExportResult(work_dir=str(work_dir), config_file=str(work_dir / "effective.yaml")),
        device="cpu",
        input_messages={"mode": "real", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert observed[-1] == {"mode": "real", "messages": [{"role": "user", "content": "hello"}]}
    assert json.loads(Path(output).read_text(encoding="utf-8"))["mode"] == "real"


def test_dump_real_golden_captures_only_executed_runtime_components(tmp_path, monkeypatch) -> None:
    from xhmodel_merak.xh_llm.models.minicpm_o_4_5 import golden as golden_module

    class Session:
        def __init__(self, name: str) -> None:
            self.name = name
            self.save_golden = False
            self.golden_dir = None
            self.step = None

    vision = Session("vision")
    audio = Session("audio")
    stale = tmp_path / "real_golden" / "audio" / "step_0"
    stale.mkdir(parents=True)
    (stale / "stale.npy").write_bytes(b"stale")

    class Runtime:
        def __init__(self) -> None:
            self.vision = vision
            self.audio = SimpleNamespace(session=audio)
            self.llm = SimpleNamespace(prefill_model=Session("llm_prefill"), decode_model=Session("llm_decode"))
            self.tts = SimpleNamespace(prefill_model=Session("tts_prefill"), decode_model=Session("tts_decode"))
            self.token2wav = SimpleNamespace(
                frontend_session=Session("flow_frontend"),
                decoder_session=Session("flow_decoder"),
                hift_session=Session("hift"),
            )

        def set_exec_device(self, device: str) -> None:
            assert device == "cpu"

        def reset_state(self) -> None:
            return None

        def chat(self, **kwargs):
            assert kwargs["msgs"] == [{"role": "user", "content": "hello"}]
            assert vision.save_golden is True
            assert audio.save_golden is True
            output = Path(vision.golden_dir) / "step_0"
            output.mkdir(parents=True)
            (output / "vision.npy").write_bytes(b"vision")
            return "ok"

        def release(self) -> None:
            return None

    runtime = Runtime()
    monkeypatch.setattr(golden_module, "_load_real_runtime", lambda root, meta: (runtime, object()))

    result = golden_module.dump_real_golden(
        tmp_path,
        {"hf_model": "/models/minicpm", "components": {}},
        "cpu",
        {"messages": [{"role": "user", "content": "hello"}]},
    )

    assert result["mode"] == "real"
    assert set(result["components"]) == {"vision"}
    assert result["text"] == "ok"
    assert not stale.exists()
