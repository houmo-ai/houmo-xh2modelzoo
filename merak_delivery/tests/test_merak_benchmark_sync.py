from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml


MODULE_PATH = Path(__file__).resolve().parents[1] / "tools" / "sync_merak_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("sync_merak_benchmarks", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
syncer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(syncer)


def _write_card(root: Path, model_id: str, model_dir: str, precision: str) -> Path:
    path = root / "examples_merak/llm/demo/model_cards" / f"{model_id}.yaml"
    path.parent.mkdir(parents=True)
    card = syncer.to_v2_model_card(
        {
            "model": {"id": model_id, "family": "demo", "display_name": model_id, "modality": ["text"], "task": ["inference"], "tags": []},
            "source": {"provider": "待补", "name": model_id, "url": "", "license": "", "raw_model_path": model_dir},
            "workflow": {
                "config_path": "configs_merak/workflows/xh2a/llm_models/demo/demo.yaml",
                "model_dir": model_dir,
                "class": "auto",
                "actions": ["quant", "export", "dump_golden", "eval"],
                "precision": {"overall": precision, "components": {}},
            },
            "runtime": {"device": "cpu", "seed": 0, "work_dir": "work_dirs/merak_delivery/demo"},
            "frontend": {"summary": "", "inputs": [{"name": "input", "type": "tensor", "description": ""}], "outputs": [{"name": "output", "type": "tensor", "description": ""}], "submodels": [], "demo": {"kind": "metadata_only"}, "limitations": [], "evidence": {}},
            "release": {"version_id": f"{model_id}_draft", "target_status": "draft", "owner": "", "reviewer": ""},
        }
    )
    path.write_text(
        yaml.safe_dump(card, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_sync_marks_related_precision_mismatch_without_claiming_accuracy(tmp_path: Path):
    card_path = _write_card(tmp_path, "demo_1b", "/models/Demo-1B", "w8a16h1_sefp")
    benchmark_root = tmp_path / "benchmark_test"
    benchmark_root.mkdir()
    (benchmark_root / "benchmark_test_demo_1b.py").write_text(
        """
import allure
import pytest

@allure.title("Demo-1B 测试")
@pytest.mark.parametrize("w_bit", [8])
@pytest.mark.parametrize("a_bit", [8])
def test_demo(w_bit, a_bit):
    command = ["python", "export.py", "--model", "/models/Demo-1B", "--quant-type", "w8a8h1_sefp"]
    eval_ppl_path = "work_dirs/Demo-1B/eval_ppl.txt"
    ppl_value = 1.0
    assert ppl_value < 15
""",
        encoding="utf-8",
    )

    result = syncer.sync(
        tmp_path,
        Path("examples_merak"),
        Path("benchmark_test"),
    )
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))

    assert result == {"cards_updated": 1, "matched_cases": 1, "scripts_scanned": 1}
    assert card["internal"]["benchmark"]["status"] == "mismatch"
    case = card["internal"]["benchmark"]["cases"][0]
    assert case["match_status"] == "mismatch"
    assert case["metric"] == "PPL"
    assert case["threshold"] == "PPL < 15"
    assert case["observed"] == ""
    assert "未找到" in case["note"]
    assert "accuracy" not in card


def test_sync_reads_existing_ppl_result_as_measured(tmp_path: Path):
    card_path = _write_card(tmp_path, "demo_1b", "/models/Demo-1B", "w8a8h1_sefp")
    card_before = yaml.safe_load(card_path.read_text(encoding="utf-8"))
    card_before["external"]["parameters"] = {
        "total": 1.2,
        "active": None,
        "unit": "B",
        "status": "verified",
    }
    card_before["external"]["compute"] = {
        "input_shape": [1, 2048],
        "prefill": 7.5,
        "decode": 0.0037,
        "unit": "TFLOPs",
        "status": "verified",
    }
    card_before["external"]["precision"]["float_format"] = "BF16"
    card_path.write_text(yaml.safe_dump(card_before, sort_keys=False), encoding="utf-8")
    benchmark_root = tmp_path / "benchmark_test"
    benchmark_root.mkdir()
    (benchmark_root / "benchmark_test_demo_1b.py").write_text(
        """
import allure

@allure.title("Demo-1B 测试")
def test_demo():
    command = ["python", "export.py", "--model", "/models/Demo-1B", "--quant-type", "w8a8h1_sefp"]
    eval_ppl_path = "work_dirs/Demo-1B/eval_ppl.txt"
""",
        encoding="utf-8",
    )
    result_path = tmp_path / "work_dirs/Demo-1B/eval_ppl.txt"
    result_path.parent.mkdir(parents=True)
    result_path.write_text("{'wikitext ppl': 8.25}", encoding="utf-8")

    syncer.sync(tmp_path)
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))

    assert card["internal"]["benchmark"]["status"] == "measured"
    assert card["internal"]["benchmark"]["cases"][0]["observed"] == 8.25
    assert card["external"]["parameters"] == card_before["external"]["parameters"]
    assert card["external"]["compute"] == card_before["external"]["compute"]
    assert card["external"]["precision"]["float_format"] == "BF16"


def test_sync_rejects_different_semantic_model_variant(tmp_path: Path):
    card_path = _write_card(tmp_path, "demo_embedding_1b", "/models/Demo-1B-Instruct", "w8a8h1_sefp")
    benchmark_root = tmp_path / "benchmark_test"
    benchmark_root.mkdir()
    (benchmark_root / "benchmark_test_demo_1b.py").write_text(
        """
import allure

@allure.title("Demo-1B 测试")
def test_demo():
    command = ["python", "export.py", "--model", "/models/Demo-1B-Instruct", "--quant-type", "w8a8h1_sefp"]
""",
        encoding="utf-8",
    )

    result = syncer.sync(tmp_path)
    card = yaml.safe_load(card_path.read_text(encoding="utf-8"))

    assert result["matched_cases"] == 0
    assert "benchmark" not in card["internal"]
