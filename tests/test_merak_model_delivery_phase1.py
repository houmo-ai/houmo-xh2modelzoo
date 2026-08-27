import subprocess
import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]


DELIVERY_FILES = [
    ROOT / "examples_merak/llm/gemma4_series/model_cards/gemma4_e4b.yaml",
    ROOT / "examples_merak/llm/qwen3/model_cards/qwen3_0_6b.yaml",
    ROOT / "examples_merak/llm/qwen3_5/model_cards/qwen3_5_9b.yaml",
]


def test_phase1_delivery_schema_and_samples_exist():
    assert (ROOT / "merak_delivery/schemas/merak_model_card.schema.json").is_file()
    assert (ROOT / "merak_delivery/tools/merak_model_flow.py").is_file()
    for delivery_file in DELIVERY_FILES:
        assert delivery_file.is_file()


def test_phase1_delivery_samples_have_expected_minimal_shape():
    for delivery_file in DELIVERY_FILES:
        data = yaml.safe_load(delivery_file.read_text(encoding="utf-8"))
        assert set(data) == {"schema_version", "external", "internal"}
        assert data["schema_version"] == 2
        assert data["external"]["model"]["id"]
        assert data["external"]["parameters"]["status"] in {"verified", "inferred", "missing"}
        compute = data["external"]["compute"]
        assert {"input_shape", "prefill", "decode", "unit", "status"}.issubset(compute)
        assert set(compute).issubset({"input_shape", "prefill", "decode", "unit", "status", "note"})
        if "note" in compute:
            assert isinstance(compute["note"], str)
        assert compute["unit"] == "TFLOPs"
        assert compute["status"] in {"verified", "inferred", "missing"}
        if compute["status"] != "missing":
            assert len(compute["input_shape"]) >= 2
            assert all(isinstance(size, int) and size > 0 for size in compute["input_shape"])
            assert isinstance(compute["prefill"], (int, float))
            assert compute["decode"] is None or isinstance(compute["decode"], (int, float))
        assert isinstance(data["external"]["owner"], str)
        workflow = data["internal"]["workflow"]
        assert workflow["config_path"].startswith("configs_merak/workflows/")
        assert (ROOT / workflow["config_path"]).is_file()
        assert workflow["class"] == "auto"
        assert workflow["actions"] == ["quant", "export", "dump_golden", "eval"]
        assert data["internal"]["runtime"]["work_dir"].startswith("work_dirs/merak_delivery/")
        assert data["internal"]["components"]
        assert data["internal"]["components"][0]["id"] == "model"
        assert data["internal"]["io"]["inputs"]
        assert data["internal"]["io"]["outputs"]
        assert data["internal"]["release"]["version_id"].startswith(data["external"]["model"]["id"])


def test_default_quantization_policy_uses_w4a8_except_for_small_models():
    special_formats = {"w16a16_sefp", "w8a16_sefp", "w8a16h0_ssfp", "w8a16h1_sefp", "w4a8h0_ssfp"}
    for card_path in (ROOT / "examples_merak").glob("**/model_cards/*.yaml"):
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        total = card["external"]["parameters"]["total"]
        quantized_format = card["external"]["precision"]["quantized_format"]
        if card["internal"]["model"]["type"] == "other_models" or quantized_format in special_formats:
            continue
        expected_prefix = "w8a8" if total is not None and total < 7_000_000_000 else "w4a8"
        assert quantized_format.startswith(expected_prefix), card_path


def test_catalog_cards_have_complete_public_capacity_metrics():
    for card_path in (ROOT / "examples_merak").glob("**/model_cards/*.yaml"):
        card = yaml.safe_load(card_path.read_text(encoding="utf-8"))
        parameters = card["external"]["parameters"]
        compute = card["external"]["compute"]
        assert parameters["total"] is not None, card_path
        assert parameters["status"] != "missing", card_path
        assert compute["input_shape"], card_path
        assert compute["prefill"] is not None, card_path
        assert compute["status"] != "missing", card_path


def test_phase1_validate_command_accepts_all_samples():
    model_card_count = len(list((ROOT / "examples_merak").glob("**/model_cards/*.yaml")))
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "merak_delivery/tools/merak_model_flow.py"),
            "validate",
            "--root",
            "examples_merak",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Validated {model_card_count} model card YAML file(s)" in result.stdout


@pytest.mark.parametrize("mutation", ["missing_required", "unexpected_nested", "missing_components"])
def test_phase1_validate_command_rejects_schema_violations(tmp_path: Path, mutation: str):
    data = yaml.safe_load(DELIVERY_FILES[1].read_text(encoding="utf-8"))
    if mutation == "missing_required":
        del data["external"]["precision"]["float_format"]
    elif mutation == "unexpected_nested":
        data["internal"]["unexpected_nested"] = True
    else:
        data["internal"]["components"] = []
    card_path = tmp_path / "model_cards" / "invalid.yaml"
    card_path.parent.mkdir()
    card_path.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "merak_delivery/tools/merak_model_flow.py"),
            "validate",
            "--root",
            str(card_path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    assert result.returncode != 0
    assert "schema violation" in result.stderr
