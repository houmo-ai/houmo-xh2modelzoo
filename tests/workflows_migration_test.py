import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow, BaseLLMWorkflow
from xhmodel_merak.xh_llm.workflows.result import ExportResult, QuantResult


class DummyWorkflow(BaseLLMWorkflow):
    pass


def _write_workflow_config(path: Path, model_type: str, visual_buckets: bool = False) -> Path:
    export = {
        "model": {
            "chip_arch": "XH2a",
            "model_type": model_type,
            "hf_model": None,
            "model_name": "test_model",
        }
    }
    if visual_buckets:
        export["visual_buckets"] = {
            "model": {
                "chip_arch": "XH2a",
                "model_type": "Qwen2VLForConditionalGeneration_visual",
                "patch_size": 14,
                "quant_scheme": {
                    "quant_type": "w8a16h0_ssfp",
                    "ops": {},
                },
            },
            "buckets": [
                {
                    "max_size_h": 140,
                    "max_size_w": 392,
                }
            ],
        }

    path.write_text(
        yaml.safe_dump(
            {
                "quant": None,
                "export": export,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_auto_llm_workflow_loads_declared_workflow_class(monkeypatch, tmp_path):
    class DummyModel:
        WORKFLOW_CLS = "dummy_workflow_module:DummyWorkflow"

    captured_cfg = {}

    def fake_get_model_class(cfg):
        captured_cfg.update(cfg)
        return DummyModel

    monkeypatch.setattr("xhmodel_merak.xh_llm.workflows.auto.get_model_class", fake_get_model_class)
    monkeypatch.setattr(
        "xhmodel_merak.xh_llm.workflows.auto.importlib.import_module",
        lambda module_name: SimpleNamespace(DummyWorkflow=DummyWorkflow),
    )
    hf_model_dir = tmp_path / "hf"
    hf_model_dir.mkdir()
    config_path = _write_workflow_config(tmp_path / "workflow.yaml", "Qwen2VLForConditionalGeneration")

    workflow = AutoLLMWorkflow.from_config(str(hf_model_dir), str(config_path))

    assert type(workflow) is DummyWorkflow
    assert captured_cfg["hf_model"] is None


def test_base_workflow_formats_xhquant_model_name_from_workflow_config(tmp_path):
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    hf_model_dir = tmp_path / "hf"
    nested_config_dir = hf_model_dir / "nested"
    nested_config_dir.mkdir(parents=True)
    (nested_config_dir / "config.json").write_text(
        json.dumps({"text_config": {"max_position_embeddings": 262144}}),
        encoding="utf-8",
    )
    config_path = tmp_path / "workflow.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "quant": {"algorithm": "gptqmodel", "bits": 4},
                "export": {
                    "model": {
                        "chip_arch": "XH2a",
                        "model_type": "DummyForCausalLM",
                        "hf_model": "weights/dummy",
                        "model_name": "qwen3_6_27b_full_dflash",
                        "spec_decode_mode": "dflash",
                        "context_max_length": 2048,
                        "prefill_chunk_length": 256,
                        "quant_scheme": {"quant_type": "w8a8h1_sefp"},
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    workflow_config = WorkflowConfig.from_file(str(config_path))
    workflow = BaseLLMWorkflow(str(hf_model_dir), str(config_path))

    assert (
        workflow._format_model_name(workflow_config, str(hf_model_dir))
        == "xh2_qwen3_6_27b_full_dflash_dflash_w4a8_256_2k_mpe256k"
    )


def test_base_workflow_model_name_missing_format_field_returns_original_name(tmp_path):
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    hf_model_dir = tmp_path / "hf"
    hf_model_dir.mkdir()
    config_path = tmp_path / "workflow.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "quant": None,
                "export": {
                    "model": {
                        "chip_arch": "XH2a",
                        "model_type": "DummyForCausalLM",
                        "hf_model": "weights/dummy",
                        "model_name": "dummy",
                        "context_max_length": 2048,
                        "prefill_chunk_length": 256,
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    workflow_config = WorkflowConfig.from_file(str(config_path))
    workflow = BaseLLMWorkflow(str(hf_model_dir), str(config_path))

    assert workflow._format_model_name(workflow_config, str(hf_model_dir)) == "dummy"


def test_auto_llm_workflow_falls_back_to_base_workflow(monkeypatch, tmp_path):
    class DummyModel:
        pass

    monkeypatch.setattr("xhmodel_merak.xh_llm.workflows.auto.get_model_class", lambda cfg: DummyModel)
    hf_model_dir = tmp_path / "hf"
    hf_model_dir.mkdir()
    config_path = _write_workflow_config(tmp_path / "workflow.yaml", "DummyForCausalLM")

    workflow = AutoLLMWorkflow.from_config(str(hf_model_dir), str(config_path))

    assert type(workflow) is BaseLLMWorkflow


@pytest.mark.parametrize(
    ("visual_buckets", "expected_calls"),
    [
        (False, 0),
        (True, 1),
    ],
)
def test_qwen2_vl_workflow_dispatches_visual_buckets(monkeypatch, tmp_path, visual_buckets, expected_calls):
    from xhmodel_merak.xh_llm.models.qwen2_vl.workflow import XHQwen2VLHMONNXWorkflow

    def fake_export(self, quant_result, output_dir, device, config_overrides=None):
        return ExportResult(work_dir=str(tmp_path), config_file=str(tmp_path / "used.yaml"), meta=object())

    calls = []

    def fake_write_visual_bucket_manifest(self, export_result, visual_buckets_cfg):
        calls.append(visual_buckets_cfg)
        return str(tmp_path / "mineru_visual_buckets.json")

    monkeypatch.setattr(BaseLLMWorkflow, "export", fake_export)
    monkeypatch.setattr(XHQwen2VLHMONNXWorkflow, "_write_visual_bucket_manifest", fake_write_visual_bucket_manifest)

    hf_model_dir = tmp_path / "hf"
    hf_model_dir.mkdir()
    config_path = _write_workflow_config(
        tmp_path / "workflow.yaml",
        "Qwen2VLForConditionalGeneration",
        visual_buckets=visual_buckets,
    )
    workflow = XHQwen2VLHMONNXWorkflow(str(hf_model_dir), str(config_path))

    workflow.export(
        quant_result=QuantResult(raw_model_dir=str(hf_model_dir), skipped=True),
        output_dir=str(tmp_path / "export"),
        device="cpu",
    )

    assert len(calls) == expected_calls


def _write_gemma4_workflow_config(
    path: Path,
    *,
    model_type: str = "Gemma4ForConditionalGeneration",
    quant: dict | None = None,
) -> Path:
    if quant is None:
        quant = {
            "algorithm": "gptqmodel",
            "method": "gptq",
            "preset": "full_multimodal",
            "rotation": None,
            "artifact_format": "gptqmodel_hf",
            "output_format": "gptqmodel_hf",
            "bits": 4,
            "group_size": 64,
            "sym": True,
            "iters": 200,
            "seed": 42,
            "quant_nontext_module": False,
            "calibration": {
                "dataset": "wikitext",
                "split": "train",
                "nsamples": 256,
                "seqlen": 2048,
            },
            "runtime": {
                "batch_size": 1,
                "device_map": "auto",
                "trust_remote_code": True,
                "offload_to_disk": False,
            },
            "validation": {
                "check_quant_text_demo": True,
                "check_quant_image_demo": True,
                "check_quant_video_demo": True,
                "check_quant_audio_demo": True,
            },
        }
    data = {
        "quant": quant,
        "export": {
            "model": {
                "chip_arch": "XH2a",
                "model_type": model_type,
                "hf_model": None,
                "model_name": "xh2_gemma4_test_full_256_2k",
                "context_max_length": 2048,
                "prefill_chunk_length": 256,
                "use_cache": True,
                "num_logits_to_keep": 1,
                "quant_scheme": {"quant_type": "w8a8h1_sefp", "ops": {}},
                "visual_config": {
                    "export_mode": "padded",
                    "image_seq_length": 280,
                    "max_patches": 2520,
                    "patch_size": 16,
                    "pooling_kernel_size": 3,
                    "input_modality": "image",
                    "quant_scheme": {"quant_type": "w8a8h1_sefp", "ops": {}},
                },
                "video_visual_config": {
                    "export_mode": "padded",
                    "image_seq_length": 70,
                    "max_patches": 630,
                    "patch_size": 16,
                    "pooling_kernel_size": 3,
                    "input_modality": "video",
                    "quant_scheme": {"quant_type": "w8a8h1_sefp", "ops": {}},
                },
                "only_first_block": False,
            }
        },
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_gemma4_workflow_requires_explicit_base_quant_override(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow

    hf_model_dir = tmp_path / "gemma4-hf"
    hf_model_dir.mkdir()
    config_path = _write_gemma4_workflow_config(tmp_path / "gemma4_full.yaml")
    workflow = Gemma4SeriesWorkflow.from_config(str(hf_model_dir), str(config_path))

    def fake_gptqmodel_recipe(*args, **kwargs):
        raise ImportError("GPTQModel Gemma4 recipe unavailable in test")

    monkeypatch.setattr(Gemma4SeriesWorkflow, "_quant_gptqmodel_recipe", fake_gptqmodel_recipe)
    with pytest.raises(ImportError, match="GPTQModel"):
        workflow.quant(output_dir=str(tmp_path / "quant"), device="cpu")

    quant_result = workflow.quant(
        output_dir=str(tmp_path / "base_quant"),
        device="cpu",
        config_overrides={"quant": None},
    )
    assert quant_result.skipped is True
    assert quant_result.raw_model_dir == str(hf_model_dir.resolve())


def test_gemma4_workflow_existing_hf_quant_result_is_normalized(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow

    hf_model_dir = tmp_path / "gemma4-hf"
    existing_quant = tmp_path / "gemma4-existing-quant"
    hf_model_dir.mkdir()
    existing_quant.mkdir()
    config_path = _write_gemma4_workflow_config(tmp_path / "gemma4_full.yaml")
    workflow = Gemma4SeriesWorkflow.from_config(str(hf_model_dir), str(config_path))

    quant_result = workflow.quant(
        output_dir=str(tmp_path / "quant"),
        device="cpu",
        config_overrides={
            "quant": {
                "algorithm": "existing_hf",
                "artifact_format": "gptqmodel_hf",
                "source_algorithm": "autoround",
                "existing_hf_model_dir": str(existing_quant),
            }
        },
    )

    assert quant_result.skipped is False
    assert quant_result.raw_model_dir == str(hf_model_dir.resolve())
    assert quant_result.quanted_model_dir == str(existing_quant.resolve())


def test_gemma4_workflow_existing_hf_reports_missing_path(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow

    hf_model_dir = tmp_path / "gemma4-hf"
    hf_model_dir.mkdir()
    config_path = _write_gemma4_workflow_config(tmp_path / "gemma4_full.yaml")
    workflow = Gemma4SeriesWorkflow.from_config(str(hf_model_dir), str(config_path))

    with pytest.raises(FileNotFoundError, match="quant.existing_hf_model_dir"):
        workflow.quant(
            output_dir=str(tmp_path / "quant"),
            device="cpu",
            config_overrides={
                "quant": {
                    "algorithm": "existing_hf",
                    "artifact_format": "gptqmodel_hf",
                    "existing_hf_model_dir": "${MISSING_GEMMA4_QUANT_HF}",
                }
            },
        )


def test_gemma4_workflow_dispatches_gptqmodel_recipe(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow

    hf_model_dir = tmp_path / "gemma4-hf"
    hf_model_dir.mkdir()
    (hf_model_dir / "config.json").write_text(
        json.dumps(
            {
                "text_config": {"num_kv_shared_layers": 1},
                "vision_config": {"hidden_size": 768},
                "audio_config": {"feature_size": 128},
            }
        ),
        encoding="utf-8",
    )
    config_path = _write_gemma4_workflow_config(tmp_path / "gemma4_full.yaml")
    workflow = Gemma4SeriesWorkflow.from_config(str(hf_model_dir), str(config_path), seed=7)
    captured = {}

    def fake_recipe(self, **kwargs):
        captured.update(kwargs)
        return QuantResult(
            raw_model_dir=str(hf_model_dir.resolve()),
            quanted_model_dir=str(tmp_path / "quantized-hf"),
        )

    monkeypatch.setattr(Gemma4SeriesWorkflow, "_quant_gptqmodel_recipe", fake_recipe)

    result = workflow.quant(output_dir=str(tmp_path / "quant"), device="cuda:0")

    assert result.quanted_model_dir == str(tmp_path / "quantized-hf")
    assert captured["device"] == "cuda:0"
    assert captured["quant_cfg"]["algorithm"] == "gptqmodel"
    assert captured["quant_cfg"]["method"] == "gptq"
    assert captured["export_model_cfg"]["context_max_length"] == 2048
    assert captured["effective_config_file"] == str(config_path)


def test_gemma4_workflow_rejects_mismatched_quant_formats(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow

    hf_model_dir = tmp_path / "gemma4-hf"
    hf_model_dir.mkdir()
    config_path = _write_gemma4_workflow_config(tmp_path / "gemma4_full.yaml")
    workflow = Gemma4SeriesWorkflow.from_config(str(hf_model_dir), str(config_path))

    with pytest.raises(ValueError, match="artifact_format and quant.output_format"):
        workflow.quant(
            output_dir=str(tmp_path / "quant"),
            device="cpu",
            config_overrides={
                "quant": {
                    "algorithm": "existing_hf",
                    "artifact_format": "gptqmodel_hf",
                    "output_format": "other_format",
                    "existing_hf_model_dir": str(tmp_path / "existing"),
                }
            },
        )


def test_gemma4_quant_adapter_builds_recipe_kwargs_from_series_plan(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.quant_adapter import build_gptqmodel_recipe_kwargs

    hf_model_dir = tmp_path / "gemma4-e4b-hf"
    hf_model_dir.mkdir()
    (hf_model_dir / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "num_kv_shared_layers": 1,
                    "hidden_size_per_layer_input": 256,
                    "max_position_embeddings": 2048,
                },
                "vision_config": {"hidden_size": 768},
                "audio_config": {"feature_size": 128},
            }
        ),
        encoding="utf-8",
    )
    config_path = _write_gemma4_workflow_config(tmp_path / "gemma4_full.yaml")
    workflow_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    kwargs = build_gptqmodel_recipe_kwargs(
        hf_model_dir=str(hf_model_dir),
        output_dir=str(tmp_path / "quant"),
        device="cuda:1",
        quant_cfg=workflow_config["quant"],
        export_model_cfg=workflow_config["export"]["model"],
        workflow_seed=1024,
    )

    assert kwargs["model_dir"] == str(hf_model_dir.resolve())
    assert kwargs["output_dir"] == str((tmp_path / "quant").resolve())
    assert kwargs["method"] == "gptq"
    assert kwargs["preset"] == "full_multimodal"
    assert "rotation" not in kwargs
    assert kwargs["artifact_format"] == "gptqmodel_hf"
    assert kwargs["variant"] == "e4b"
    assert kwargs["topology"] == "dense"
    assert kwargs["capabilities"] == {"text": True, "image": True, "video": True, "audio": True}
    assert kwargs["export_subgraphs"]["video_visual"] is True
    assert kwargs["export_subgraphs"]["audio"] is True
    assert kwargs["context_max_length"] == 2048
    assert kwargs["prefill_chunk_length"] == 256
    assert kwargs["group_size"] == 64


def test_gemma4_quant_preflight_reports_missing_package_resource(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series import quant_adapter

    hf_model_dir = tmp_path / "gemma4-e4b-hf"
    hf_model_dir.mkdir()
    (hf_model_dir / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "num_kv_shared_layers": 1,
                    "hidden_size_per_layer_input": 256,
                    "max_position_embeddings": 2048,
                },
                "vision_config": {"hidden_size": 768},
                "audio_config": {"feature_size": 128},
            }
        ),
        encoding="utf-8",
    )
    config_path = _write_gemma4_workflow_config(tmp_path / "gemma4_full.yaml")
    workflow_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    monkeypatch.setattr(quant_adapter.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(FileNotFoundError, match="quant.calibration.jsonl"):
        quant_adapter.quantize_with_gptqmodel_recipe(
            model_dir=str(hf_model_dir),
            output_dir=str(tmp_path / "quant"),
            device="cuda:0",
            quant_cfg=workflow_config["quant"],
            export_model_cfg=workflow_config["export"]["model"],
            effective_config_file=str(config_path),
            workflow_seed=1024,
        )


def test_gemma4_recommended_configs_do_not_embed_personal_paths():
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import list_recommended_configs

    files = [Path(path) for path in list_recommended_configs().values()]
    files.append(Path("configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml"))
    files.append(Path("examples_merak/llm/gemma4_series/README.md"))

    for path in files:
        text = path.read_text(encoding="utf-8")
        assert "/data01/home/yujy" not in text
        if path.suffix in {".yaml", ".yml"}:
            config = yaml.safe_load(text)
            assert str(config["quant"]["calibration"]["jsonl"]).startswith("gptqmodel://")


def test_gemma4_quant_adapter_rejects_rotation(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.quant_adapter import build_gptqmodel_recipe_kwargs

    hf_model_dir = tmp_path / "gemma4-hf"
    hf_model_dir.mkdir()
    config_path = _write_gemma4_workflow_config(tmp_path / "gemma4_full.yaml")
    workflow_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    workflow_config["quant"]["rotation"] = "hadamard"

    with pytest.raises(ValueError, match="does not support rotation"):
        build_gptqmodel_recipe_kwargs(
            hf_model_dir=str(hf_model_dir),
            output_dir=str(tmp_path / "quant"),
            device="cuda:0",
            quant_cfg=workflow_config["quant"],
            export_model_cfg=workflow_config["export"]["model"],
            workflow_seed=1024,
        )


def test_gemma4_build_input_message_text_and_vlm(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow

    hf_model_dir = tmp_path / "gemma4-hf"
    hf_model_dir.mkdir()
    config_path = _write_gemma4_workflow_config(tmp_path / "gemma4_full.yaml")
    workflow = Gemma4SeriesWorkflow.from_config(str(hf_model_dir), str(config_path))

    assert workflow.build_input_message("hello") == [{"role": "user", "content": "hello"}]
    assert workflow.build_input_message({"text": "hello"}) == [{"role": "user", "content": "hello"}]
    assert workflow.build_input_message({"image": "image.jpg", "text": "describe"}) == [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "image.jpg"},
                {"type": "text", "text": "describe"},
            ],
        }
    ]


def test_gemma4_family_workflow_yamls_share_one_registration():
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    family_configs = [
        Path("configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml"),
        Path("configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_full.yaml"),
        Path("configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml"),
    ]
    from xhmodel_merak.xh_llm.builder import get_model_class

    for config_path in family_configs:
        workflow_config = WorkflowConfig.from_file(str(config_path))
        model = workflow_config.export["model"]
        assert model["model_type"] == "Gemma4ForConditionalGeneration"
        assert "gemma4e" not in str(model).lower()
        assert "with_mask" not in str(model).lower()
        model_cls = get_model_class(model)
        assert model_cls.__name__ == "XHGemma4SeriesModel"
        assert model_cls.CONFIG_CLS.__name__ == "XHGemma4SeriesModelConfig"


def test_auto_llm_workflow_dispatches_gemma4_to_unified_workflow(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow

    hf_model_dir = tmp_path / "gemma4-hf"
    hf_model_dir.mkdir()
    workflow = AutoLLMWorkflow.from_config(
        str(hf_model_dir),
        "configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml",
    )

    assert type(workflow) is Gemma4SeriesWorkflow


def test_gemma4_package_exports_workflow_api():
    from xhmodel_merak.xh_llm.models import gemma4_series
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow

    assert gemma4_series.Gemma4SeriesWorkflow is Gemma4SeriesWorkflow
    assert gemma4_series.XHGemma4HMONNXWorkflow is Gemma4SeriesWorkflow
    assert set(gemma4_series.list_recommended_configs()) == {"12b-unified", "e2b", "e4b", "31b", "26b-a4b"}
    assert "GPTQModel" in gemma4_series.get_quant_config_help()
    assert "Gemma4ForConditionalGeneration" in gemma4_series.get_export_config_help()


def test_gemma4_workflow_template_helpers_and_thin_quant_api(tmp_path):
    from xhmodel_merak.xh_llm.models import gemma4_series

    quant_template_path = tmp_path / "quant_template.yaml"
    export_template_path = tmp_path / "export_template.yaml"

    assert gemma4_series.dump_quant_config_template(quant_template_path) == str(quant_template_path)
    assert gemma4_series.dump_export_config_template(export_template_path) == str(export_template_path)

    quant_template = yaml.safe_load(quant_template_path.read_text(encoding="utf-8"))
    export_template = yaml.safe_load(export_template_path.read_text(encoding="utf-8"))
    assert export_template["export"]["model"]["sliding_kv_cache_input_mode"] == "slice_window"
    assert quant_template["quant"]["algorithm"] == "gptqmodel"
    assert quant_template["quant"]["method"] == "gptq"
    assert quant_template["quant"]["preset"] == "full_multimodal"
    assert quant_template["quant"]["artifact_format"] == "gptqmodel_hf"
    assert quant_template["quant"]["group_size"] == 64
    assert export_template["export"]["model"]["model_type"] == "Gemma4ForConditionalGeneration"
    assert export_template["export"]["model"]["visual_config"]["export_mode"] == "padded"

    hf_model_dir = tmp_path / "gemma4-hf"
    hf_model_dir.mkdir()
    config_path = _write_gemma4_workflow_config(tmp_path / "gemma4_full.yaml")
    quant_result = gemma4_series.quant(
        model_dir=str(hf_model_dir),
        config_path=str(config_path),
        output_dir=str(tmp_path / "quant"),
        device="cpu",
        config_overrides={"quant": None},
    )

    assert quant_result.skipped is True
    assert quant_result.raw_model_dir == str(hf_model_dir.resolve())


def test_gemma4_unified_config_detects_moe_without_public_model_type_split(tmp_path):
    from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel

    hf_model_dir = tmp_path / "gemma4-26b-a4b"
    hf_model_dir.mkdir()
    (hf_model_dir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "architectures": ["Gemma4ForConditionalGeneration"],
                "image_token_id": 262144,
                "text_config": {
                    "enable_moe_block": True,
                    "sliding_window": 1024,
                    "layer_types": ["sliding_attention", "full_attention"],
                    "num_hidden_layers": 2,
                    "num_key_value_heads": 4,
                    "num_global_key_value_heads": 4,
                    "head_dim": 72,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    config = AutoLLMConfig.from_pretrained(
        {
            "chip_arch": "XH2a",
            "model_type": "Gemma4ForConditionalGeneration",
            "hf_model": str(hf_model_dir),
            "model_name": "xh2_gemma4_26b_a4b_unified_test",
            "visual_config": None,
        }
    )
    model = AutoLLMModel.from_pretrained(config)

    assert type(config).__name__ == "XHGemma4SeriesModelConfig"
    assert config.enable_moe_block is True
    assert config.fallback_hf_model == str(hf_model_dir)
    assert type(model).__name__ == "XHGemma4SeriesModel"
    assert model.CONFIG_CLS.__name__ == "XHGemma4SeriesModelConfig"
    assert model.model_type == "Gemma4ForConditionalGeneration"




def test_gemma4_unified_processor_emits_padded_vit_contract_for_all_variants():
    from PIL import Image

    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_processor import XHGemma4Processor

    model_dirs = [
        Path("/data01/datasets/gemma-4-E4B-it"),
        Path("/data01/datasets/gemma-4-31B-it"),
        Path("/data01/datasets/gemma-4-26B-A4B-it"),
    ]
    missing = [str(path) for path in model_dirs if not path.exists()]
    if missing:
        pytest.skip(f"Gemma4 local HF fixtures are unavailable: {missing}")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": Image.new("RGB", (320, 224), color="white")},
                {"type": "text", "text": "describe the image"},
            ],
        }
    ]

    for model_dir in model_dirs:
        processor = XHGemma4Processor.from_pretrained(str(model_dir))
        inputs = processor.apply_chat_template(messages)

        assert tuple(inputs["pixel_values"].shape) == (1, 2520, 768)
        assert tuple(inputs["image_position_ids"].shape) == (1, 2520, 2)
        assert tuple(inputs["pixel_position_ids"].shape) == (1, 2520, 2)
        assert str(inputs["pixel_position_ids"].dtype) == "torch.int32"
        assert not bool((inputs["pixel_position_ids"] < 0).any())
        assert tuple(inputs["pooling_matrix"].shape) == (1, 280, 2520)
        assert str(inputs["pooling_matrix"].dtype) == "torch.float16"
        assert tuple(inputs["visual_attention_mask"].shape) == (1, 1, 1, 2520)
        assert str(inputs["visual_attention_mask"].dtype) == "torch.float16"

        valid_patch_mask = ~((inputs["image_position_ids"] == -1).all(dim=-1))
        expected_soft_tokens = int(valid_patch_mask.sum().item()) // 9
        image_soft_token_count = int(inputs["image_soft_token_count"].item())
        input_image_token_count = int((inputs["input_ids"] == processor.tokenizer.image_token_id).sum().item())
        assert image_soft_token_count == expected_soft_tokens == 280
        assert input_image_token_count == image_soft_token_count


def test_gemma4_public_model_type_auto_loads_series_module():
    from xhmodel_merak.xh_llm.configuration_auto import MODEL_TYPE_MAPPING_MODULES

    assert MODEL_TYPE_MAPPING_MODULES["Gemma4ForConditionalGeneration"] == "gemma4_series"

    from xhmodel_merak.xh_llm.builder import get_model_class

    model_cls = get_model_class({"chip_arch": "XH2a", "model_type": "Gemma4ForConditionalGeneration", "model_name": "gemma4_series_test"})
    assert model_cls.__name__ == "XHGemma4SeriesModel"
    assert model_cls.__module__.startswith("xhmodel_merak.xh_llm.models.gemma4_series.")


def test_gemma4_workflow_yaml_uses_padded_vit_contract():
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    config_paths = [
        Path("configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml"),
        Path("configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_full.yaml"),
        Path("configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml"),
    ]
    for config_path in config_paths:
        workflow_config = WorkflowConfig.from_file(str(config_path))
        quant = workflow_config.quant
        assert quant is not None
        assert quant["algorithm"] == "gptqmodel"
        assert quant["method"] == "gptq"
        assert quant["preset"] == "full_multimodal"
        model = workflow_config.export["model"]
        visual_config = model["visual_config"]
        assert visual_config["export_mode"] == "padded"
        assert visual_config["image_seq_length"] == 280
        assert visual_config["max_patches"] == 2520
        assert visual_config["patch_size"] == 16
        assert visual_config["pooling_kernel_size"] == 3
