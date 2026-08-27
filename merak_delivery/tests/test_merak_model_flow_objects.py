from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "merak_model_flow.py"
SPEC = importlib.util.spec_from_file_location("merak_model_flow", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
flow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(flow)


def test_object_oriented_flow_api_is_exposed():
    assert flow.MerakModelFlow
    assert flow.MerakEvaluator
    assert flow.MerakDeliveryStore
    assert flow.MerakModelFlowCLI
    assert flow.MerakModelFlow.__module__.endswith("core.model_flow")
    assert flow.MerakEvaluator.__module__.endswith("core.evaluator")
    assert flow.MerakDeliveryStore.__module__.endswith("core.delivery_store")
    assert flow.MerakModelFlowCLI.__module__.endswith("core.cli")


def test_cli_builds_existing_commands():
    parser = flow.MerakModelFlowCLI().build_parser()

    args = parser.parse_args(["validate"])
    assert args.func is flow.validate_command

    args = parser.parse_args(["run-workflow", "--model-card", "card.yaml", "--dry-run"])
    assert args.func is flow.run_workflow_command

    args = parser.parse_args(["run-eval", "--model-card", "card.yaml"])
    assert args.func is flow.run_eval_command
    assert args.eval_dataset_hub == "modelscope"

    args = parser.parse_args(["query-catalog"])
    assert args.func is flow.query_catalog_command


def test_model_card_discovery_only_reads_model_cards_directories(tmp_path: Path):
    card = tmp_path / "examples_merak" / "llm" / "demo" / "model_cards" / "demo.yaml"
    unrelated = tmp_path / "examples_merak" / "llm" / "demo" / "workflow.yaml"
    nested = card.parent / "archive" / "old.yaml"
    for path in (card, unrelated, nested):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("model: {}\n", encoding="utf-8")

    assert flow.discover_model_cards(tmp_path / "examples_merak") == [card]


def test_catalog_precision_uses_workflow_weight_bits_for_delivery_label(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    config_path = Path("configs_merak/workflows/xh2a/llm_models/demo/demo.yaml")
    full_path = tmp_path / config_path
    full_path.parent.mkdir(parents=True)
    full_path.write_text(
        yaml.safe_dump(
            {
                "quant": {
                    "algorithm": "gptqmodel",
                    "method": "autoround",
                    "bits": 4,
                    "group_size": 64,
                },
                "export": {"model": {"quant_scheme": {"quant_type": "w8a8h1_sefp"}}},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(flow, "ROOT", tmp_path)
    card = {
        "workflow": {
            "config_path": str(config_path),
            "precision": {
                "overall": "w8a8h1_sefp",
                "components": {"model_quant_scheme": "w8a8h1_sefp"},
            },
        }
    }

    precision = flow._catalog_precision(card)

    assert precision["display"] == "W4A8"
    assert precision["weight_bits"] == 4
    assert precision["activation_bits"] == 8
    assert precision["weight_group_size"] == 64
    assert precision["weight_algorithm"] == "gptqmodel"
    assert precision["weight_method"] == "autoround"
    assert precision["export_overall"] == "w8a8h1_sefp"


def test_evaluator_writes_normalized_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    model_card = tmp_path / "model.yaml"
    card = {
        "model": {"id": "demo"},
        "release": {"version_id": "demo-v1"},
    }
    evaluator = flow.MerakEvaluator(
        card=card,
        model_card_path=model_card,
        work_dir=tmp_path,
        output=tmp_path / "eval_report.json",
        eval_work_dir=tmp_path / "hm_eval",
        model_id="demo-eval",
        backend="hmonnx",
        datasets=["ceval"],
        dataset_hub="modelscope",
        limit=0,
        max_tokens=0,
    )
    monkeypatch.setattr(
        evaluator,
        "execute",
        lambda: {
            "datasets": {
                "ceval": {
                    "status": "completed",
                    "metrics": {"accuracy": 0.5},
                }
            }
        },
    )

    status = evaluator.run()

    report = json.loads((tmp_path / "eval_report.json").read_text(encoding="utf-8"))
    assert status == 0
    assert report["status"] == "passed"
    assert report["tasks"][0]["dataset"] == "ceval"
    assert report["logs"]["model"] == "demo-eval"


def test_model_flow_dry_run_keeps_output_contract(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    card = {
        "model": {"id": "demo"},
        "release": {"version_id": "demo-v1"},
        "workflow": {
            "class": "auto",
            "config_path": "config.yaml",
            "model_dir": "weights/demo",
            "actions": ["quant", "export", "dump_golden", "eval"],
        },
        "runtime": {"work_dir": str(tmp_path), "device": "cpu", "seed": 0},
    }
    args = argparse.Namespace(
        model_card="model.yaml",
        work_dir=str(tmp_path / "run"),
        model_dir="",
        quant_output_dir="",
        export_output_dir="",
        golden_meta="",
        eval_report="",
        eval_output="",
        eval_work_dir="",
        release_root="",
        catalog_root="",
        dry_run=True,
    )
    model_flow = flow.MerakModelFlow(args, card=card, model_card_path=tmp_path / "model.yaml")

    assert model_flow.run() == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["planned_steps"] == ["quant", "export", "dump_golden", "eval"]
    assert plan["outputs"]["manifest"].endswith("delivery_manifest.json")
    assert plan["outputs"]["eval_report"].endswith("eval_report.json")


def test_explicit_meta_adds_missing_hmonnx_backend(tmp_path: Path):
    model_config = SimpleNamespace(backends={})
    meta = tmp_path / "golden_meta_info.json"
    meta.write_text("{}", encoding="utf-8")

    flow._configure_hmonnx_backend(model_config, meta, None)

    backend = model_config.backends["hmonnx"]
    assert backend.type == "hmonnx"
    assert backend.export_meta_info == str(meta)


def test_default_eval_dataset_is_ceval_only():
    assert flow._infer_eval_datasets({}) == ["ceval"]


def test_float_backend_uses_independent_default_report(tmp_path: Path):
    assert flow._default_eval_report(tmp_path, "float") == tmp_path / "eval_report_float.json"
    assert flow._default_eval_report(tmp_path, "hmonnx") == tmp_path / "eval_report.json"


def test_float_cli_does_not_require_model_dir():
    parser = flow.MerakModelFlowCLI().build_parser()
    args = parser.parse_args(
        ["run-eval", "--model-card", "card.yaml", "--eval-backend", "float"]
    )
    assert args.eval_backend == "float"
    assert not hasattr(args, "eval_model_dir")


@pytest.mark.parametrize(
    ("model_id", "family", "category", "config_path", "expected"),
    [
        (
            "pi05_droid",
            "pi05",
            "other_models",
            "configs_merak/workflows/xh2a/other_models/pi05/droid/pi05_droid.yaml",
            "VLA",
        ),
        (
            "qwen3_0_6b",
            "qwen3",
            "llm_models",
            "configs_merak/workflows/xh2a/llm_models/qwen3/0_6b/model.yaml",
            "LLM",
        ),
        (
            "zimage",
            "zimage",
            "other_models",
            "configs_merak/workflows/xh2a/other_models/zimage/zimage.yaml",
            "生图",
        ),
        (
            "wan2_2_i2v",
            "wan2_2",
            "other_models",
            "configs_merak/workflows/xh2a/other_models/wan2_2/i2v/model.yaml",
            "视频理解",
        ),
        (
            "qwen3_tts",
            "qwen3_tts",
            "other_models",
            "configs_merak/workflows/xh2a/other_models/qwen3_tts/model.yaml",
            "TTS",
        ),
    ],
)
def test_catalog_model_category(
    model_id: str, family: str, category: str, config_path: str, expected: str
):
    card = {
        "model": {"id": model_id, "family": family},
        "workflow": {"category": category, "config_path": config_path, "precision": {"components": {}}},
    }

    assert flow._catalog_model_category(card) == expected
