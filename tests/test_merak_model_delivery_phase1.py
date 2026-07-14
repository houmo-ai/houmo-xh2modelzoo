import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


DELIVERY_FILES = [
    ROOT / "merak_delivery/model_cards/merak/gemma4_series/gemma4_e4b.yaml",
    ROOT / "merak_delivery/model_cards/merak/qwen3/qwen3_0_6b.yaml",
    ROOT / "merak_delivery/model_cards/merak/qwen3_5/qwen3_5_9b.yaml",
]


def test_phase1_delivery_schema_and_samples_exist():
    assert (ROOT / "merak_delivery/schemas/merak_model_card.schema.json").is_file()
    assert (ROOT / "merak_delivery/tools/merak_model_flow.py").is_file()
    for delivery_file in DELIVERY_FILES:
        assert delivery_file.is_file()


def test_phase1_delivery_samples_have_expected_minimal_shape():
    for delivery_file in DELIVERY_FILES:
        data = yaml.safe_load(delivery_file.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1
        assert data["model"]["id"]
        assert data["workflow"]["config_path"].startswith("configs_merak/workflows/")
        assert (ROOT / data["workflow"]["config_path"]).is_file()
        assert data["workflow"]["class"] == "auto"
        assert data["workflow"]["actions"] == ["quant", "export", "dump_golden", "eval"]
        assert data["runtime"]["work_dir"].startswith("work_dirs/merak_delivery/")
        assert data["frontend"]["inputs"]
        assert data["frontend"]["outputs"]
        assert data["release"]["version_id"].startswith(data["model"]["id"])


def test_phase1_validate_command_accepts_all_samples():
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "merak_delivery/tools/merak_model_flow.py"),
            "validate",
            "--root",
            "merak_delivery/model_cards/merak",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Validated 3 model card YAML file(s)" in result.stdout
