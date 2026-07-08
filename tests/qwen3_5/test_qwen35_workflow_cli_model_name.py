from pathlib import Path
import runpy


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOW_SCRIPT = REPO_ROOT / "examples_merak/llm/qwen3_5/qwen3_5_workflow.py"


def _load_workflow_script():
    return runpy.run_path(str(WORKFLOW_SCRIPT), run_name="qwen3_5_workflow_cli_test")


def test_qwen35_workflow_parser_accepts_model_name(monkeypatch):
    namespace = _load_workflow_script()
    monkeypatch.setattr(
        "sys.argv",
        [
            "qwen3_5_workflow.py",
            "--model-dir",
            "/tmp/model",
            "--config-path",
            "configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_6_27b_full.yaml",
            "--model-name",
            "Qwen3.6-27B-mode1-llm-only",
        ],
    )

    args = namespace["parse_args"]()

    assert args.model_name == "Qwen3.6-27B-mode1-llm-only"


def test_qwen35_workflow_applies_model_name_to_overrides():
    namespace = _load_workflow_script()
    apply_model_name_override = namespace["_apply_model_name_override"]
    overrides = {"quant": None}

    apply_model_name_override(overrides, "Qwen3.6-27B-mode1-llm-only")

    assert overrides == {
        "quant": None,
        "export.model.model_name": "qwen3_6_27b_mode1_llm_only",
    }


def test_qwen35_workflow_ignores_empty_model_name_override():
    namespace = _load_workflow_script()
    apply_model_name_override = namespace["_apply_model_name_override"]
    overrides = {"quant.bits": 4}

    apply_model_name_override(overrides, None)
    apply_model_name_override(overrides, "")

    assert overrides == {"quant.bits": 4}
