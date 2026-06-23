"""Lightweight checks for the config-only Merak Qwen3.5 export entrypoint."""

from __future__ import annotations

import json
import runpy
import sys
import types
from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = REPO_ROOT / "examples_merak/llm/qwen3_5/debug_scripts/qwen3_5_xh_export_hmonnx.py"
QUANT_SCRIPT = REPO_ROOT / "examples_merak/llm/qwen3_5/qwen3_5_quant.py"
QUANT_EXPORT_SCRIPT = REPO_ROOT / "examples_merak/llm/qwen3_5/qwen3_5_quant_export.py"
VALIDATION_MATRIX_SCRIPT = REPO_ROOT / "examples_merak/llm/qwen3_5/debug_scripts/qwen3_5_validation_matrix.py"
REMOVED_MOE_EXAMPLE_DIR = REPO_ROOT / "examples_merak/llm/qwen3_5_moe"
README = REPO_ROOT / "examples_merak/llm/qwen3_5/README.md"
README_WORKFLOW = REPO_ROOT / "examples_merak/llm/qwen3_5/README_workflow.md"

WORKFLOW_9B_FULL = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml"
WORKFLOW_9B_MTP = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_mtp.yaml"
WORKFLOW_9B_DFLASH = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_dflash.yaml"
WORKFLOW_MOE_FULL = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml"
WORKFLOW_MOE_MTP = (
    REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_mtp.yaml"
)
WORKFLOW_MOE_DFLASH = (
    REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_dflash.yaml"
)
WORKFLOW_27B_VISUAL_448 = (
    REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_6_27b_visual_only_448.yaml"
)
WORKFLOW_27B_VISUAL_896 = (
    REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_6_27b_visual_only_896.yaml"
)
WORKFLOW_27B_MTP = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_6_27b_full_mtp.yaml"


def _install_stub_modules(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    fake_xh_llm = types.ModuleType("xhmodel_merak.xh_llm")
    fake_xh_llm.AutoLLMConfig = object
    fake_xh_llm.AutoLLMModel = object

    class DummyConfig:
        @classmethod
        def fromfile(cls, path):  # pragma: no cover - main() is not run in parser tests
            raise AssertionError(f"Config.fromfile should not be called in this test: {path}")

    fake_xhquant_api = types.ModuleType("xhquant.api")
    fake_xhquant_api.Config = DummyConfig
    fake_xhquant_api.get_xhquant_logger = lambda: None
    fake_xhquant_api.set_random_seed = lambda seed: None
    fake_xhquant_api.xhquant_init = lambda *args, **kwargs: None

    fake_xhquant_utils = types.ModuleType("xhquant.utils")
    fake_xhquant_utils.MemoryTracker = object
    fake_xhquant_utils.TimeProfiler = object

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "xhmodel_merak.xh_llm", fake_xh_llm)
    monkeypatch.setitem(sys.modules, "xhquant.api", fake_xhquant_api)
    monkeypatch.setitem(sys.modules, "xhquant.utils", fake_xhquant_utils)


def _load_export_script(monkeypatch):
    _install_stub_modules(monkeypatch)
    return runpy.run_path(str(EXPORT_SCRIPT), run_name="qwen3_5_unified_export_test")


def _load_validation_matrix_script():
    return runpy.run_path(str(VALIDATION_MATRIX_SCRIPT), run_name="qwen3_5_validation_matrix_test")


def _load_quant_script():
    return runpy.run_path(str(QUANT_SCRIPT), run_name="qwen3_5_quant_test")


def _load_quant_export_script():
    return runpy.run_path(str(QUANT_EXPORT_SCRIPT), run_name="qwen3_5_quant_export_test")


def test_unified_export_script_parses(monkeypatch):
    _load_export_script(monkeypatch)


def test_export_entrypoint_keeps_legacy_compat_flags(monkeypatch):
    namespace = _load_export_script(monkeypatch)
    parser = namespace["build_parser"]()
    option_strings = {opt for action in parser._actions for opt in action.option_strings}

    assert {"--config", "--debug", "--force", "--seed", "--work-dir", "--work_dir"} <= option_strings
    assert {"--fuse-gdr-ops", "--fuse_gdr_ops", "--no-fuse-gdr-ops", "--no_fuse_gdr_ops"} <= option_strings
    for forbidden in (
        "--model",
        "--model-type",
        "--quant-weight",
        "--quant-type",
        "--context-length",
        "--prefill-chunk-length",
        "--spec-decode-mode",
        "--dflash-model-dir",
        "--max-size-w",
        "--max-size-h",
    ):
        assert forbidden not in option_strings
    assert parser._option_string_actions["--config"].required is True


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fin:
        return yaml.safe_load(fin)


def test_quant_script_exposes_small_path_only_cli():
    namespace = _load_quant_script()
    parser = namespace["build_parser"]()
    option_strings = {opt for action in parser._actions for opt in action.option_strings}

    assert {
        "--hf-model-dir",
        "--config",
        "--output-dir",
        "--device",
        "--force",
        "--base",
        "--existing-hf-model-dir",
    } <= option_strings
    for forbidden in (
        "--bits",
        "--group-size",
        "--llm-bits",
        "--llm-group-size",
        "--attn-bits",
        "--shared-expert-bits",
        "--dataset",
        "--iters",
        "--nsamples",
    ):
        assert forbidden not in option_strings

    args = parser.parse_args(
        [
            "--hf-model-dir",
            "weights/Qwen3.5-9B",
            "--config",
            str(WORKFLOW_9B_FULL),
            "--output-dir",
            "work_dirs/qwen35_quant",
            "--device",
            "cuda:0",
        ]
    )
    assert args.hf_model_dir == "weights/Qwen3.5-9B"
    assert args.config == str(WORKFLOW_9B_FULL)
    assert args.output_dir == "work_dirs/qwen35_quant"
    assert args.device == "cuda:0"


def test_quant_script_builds_only_explicit_source_overrides():
    namespace = _load_quant_script()
    parser = namespace["build_parser"]()

    base_args = parser.parse_args(
        [
            "--hf-model-dir",
            "weights/Qwen3.5-9B",
            "--config",
            str(WORKFLOW_9B_FULL),
            "--output-dir",
            "work_dirs/qwen35_quant",
            "--base",
        ]
    )
    assert namespace["_build_quant_overrides"](base_args) == {"quant": None}

    existing_args = parser.parse_args(
        [
            "--hf-model-dir",
            "weights/Qwen3.5-9B",
            "--config",
            str(WORKFLOW_9B_FULL),
            "--output-dir",
            "work_dirs/qwen35_quant",
            "--existing-hf-model-dir",
            "weights/Qwen3.5-9B-mode1-llm-only",
        ]
    )
    assert namespace["_build_quant_overrides"](existing_args) == {
        "quant": {
            "algorithm": "existing_hf",
            "artifact_format": "gptqmodel_hf",
            "existing_hf_model_dir": "weights/Qwen3.5-9B-mode1-llm-only",
        }
    }


def test_quant_export_script_exposes_config_driven_cli():
    namespace = _load_quant_export_script()
    parser = namespace["build_parser"]()
    option_strings = {opt for action in parser._actions for opt in action.option_strings}

    assert {
        "--hf-model-dir",
        "--config",
        "--quant-output-dir",
        "--export-output-dir",
        "--device",
        "--quant-device",
        "--export-device",
        "--force",
        "--base",
        "--existing-hf-model-dir",
        "--override",
        "--dump-golden",
    } <= option_strings
    assert "--source-algorithm" not in option_strings
    for forbidden in ("--bits", "--group-size", "--dataset", "--iters", "--nsamples"):
        assert forbidden not in option_strings


def test_quant_export_script_builds_existing_hf_overrides_without_metadata():
    namespace = _load_quant_export_script()
    parser = namespace["build_parser"]()
    args = parser.parse_args(
        [
            "--hf-model-dir",
            "weights/Qwen3.5-9B",
            "--config",
            str(WORKFLOW_9B_FULL),
            "--export-output-dir",
            "work_dirs/qwen35_export",
            "--existing-hf-model-dir",
            "weights/Qwen3.5-9B-mode1-llm-only",
            "--override",
            "export.model.fuse_gdr_ops=true",
        ]
    )

    assert namespace["_resolve_quant_output_dir"](args).endswith("_workflow_existing_or_base_quant_placeholder")
    assert namespace["_build_quant_overrides"](args) == {
        "quant": {
            "algorithm": "existing_hf",
            "artifact_format": "gptqmodel_hf",
            "existing_hf_model_dir": "weights/Qwen3.5-9B-mode1-llm-only",
        }
    }
    assert namespace["_parse_dotted_overrides"](args.override) == {"export.model.fuse_gdr_ops": True}


def test_quant_export_script_main_runs_quant_then_export(monkeypatch, capsys):
    namespace = _load_quant_export_script()
    parser = namespace["build_parser"]()
    calls = {}

    class FakeQuantResult:
        def __init__(self):
            self.hf_model_dir = "weights/Qwen3.5-9B"
            self.skipped = False
            self.quanted_model_dir = "weights/Qwen3.5-9B-mode1-llm-only"
            self.is_quant_weight_format = False

    class FakeExportResult:
        def __init__(self):
            self.work_dir = "work_dirs/qwen35_export"
            self.config_file = "work_dirs/qwen35_export/qwen3_5_9b_full.yaml"
            self.meta = None

    class FakeWorkflow:
        def quant(self, *, output_dir, device, config_overrides):
            calls["quant"] = (output_dir, device, config_overrides)
            return FakeQuantResult()

        def export(self, *, quant_result, output_dir, device, config_overrides):
            calls["export"] = (quant_result, output_dir, device, config_overrides)
            return FakeExportResult()

    class FakeAutoLLMWorkflow:
        @classmethod
        def from_config(cls, *, hf_model_dir, config_path, seed, debug):
            calls["from_config"] = (hf_model_dir, config_path, seed, debug)
            return FakeWorkflow()

    fake_workflows = types.ModuleType("xhmodel_merak.xh_llm.workflows")
    fake_workflows.AutoLLMWorkflow = FakeAutoLLMWorkflow
    monkeypatch.setitem(sys.modules, "xhmodel_merak.xh_llm.workflows", fake_workflows)

    args = parser.parse_args(
        [
            "--hf-model-dir",
            "weights/Qwen3.5-9B",
            "--config",
            str(WORKFLOW_9B_FULL),
            "--export-output-dir",
            "work_dirs/qwen35_export",
            "--existing-hf-model-dir",
            "weights/Qwen3.5-9B-mode1-llm-only",
            "--device",
            "cuda:0",
        ]
    )

    namespace["main"](args)

    expected_overrides = {
        "quant": {
            "algorithm": "existing_hf",
            "artifact_format": "gptqmodel_hf",
            "existing_hf_model_dir": "weights/Qwen3.5-9B-mode1-llm-only",
        }
    }
    assert calls["from_config"] == ("weights/Qwen3.5-9B", str(WORKFLOW_9B_FULL), 1024, False)
    assert calls["quant"] == (
        "work_dirs/qwen35_export/_workflow_existing_or_base_quant_placeholder",
        "cuda:0",
        expected_overrides,
    )
    assert calls["export"][1:] == ("work_dirs/qwen35_export", "cuda:0", expected_overrides)
    output = json.loads(capsys.readouterr().out)
    assert output["quant_result"]["quanted_model_dir"] == "weights/Qwen3.5-9B-mode1-llm-only"
    assert output["export_result"]["work_dir"] == "work_dirs/qwen35_export"


def test_legacy_moe_example_directory_is_not_part_of_new_workflow_docs():
    assert REMOVED_MOE_EXAMPLE_DIR.exists()
    src = README.read_text(encoding="utf-8")
    assert "examples_merak/llm/qwen3_5_moe" not in src


def test_readme_does_not_document_model_override_cli():
    src = README.read_text(encoding="utf-8")
    assert "--model " not in src
    assert "--quant-weight" not in src
    assert "qwen3_5_moe/qwen3_5_moe" not in src


def test_workflow_config_paths_cover_requested_models_and_existing_hf_overrides():
    assert _load_yaml(WORKFLOW_9B_FULL)["export"]["model"]["hf_model"] == "weights/Qwen3.5-9B"
    assert _load_yaml(WORKFLOW_MOE_FULL)["export"]["model"]["hf_model"] == "weights/Qwen3.6-35B-A3B"
    assert _load_yaml(WORKFLOW_27B_VISUAL_448)["export"]["model"]["hf_model"] == "weights/Qwen3.6-27B"
    assert _load_yaml(WORKFLOW_27B_VISUAL_896)["export"]["model"]["hf_model"] == "weights/Qwen3.6-27B"

    src = README_WORKFLOW.read_text(encoding="utf-8")
    assert "weights/Qwen3.5-9B-mode1-llm-only" in src
    assert "weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400" in src


def test_validation_matrix_covers_requested_runtime_cases():
    namespace = _load_validation_matrix_script()
    scenarios = namespace["VALIDATION_SCENARIOS"]
    by_name = {scenario.name: scenario for scenario in scenarios}

    assert set(by_name) == {
        "qwen35_9b_base_fuse_false",
        "qwen35_9b_base_fuse_true",
        "qwen35_9b_existing_hf_fuse_false",
        "qwen35_9b_existing_hf_fuse_true",
        "qwen35_9b_mtp_existing_hf",
        "qwen35_9b_dflash_existing_hf",
        "qwen36_35b_a3b_base_fuse_false",
        "qwen36_35b_a3b_base_fuse_true",
        "qwen36_35b_a3b_existing_hf_fuse_false",
        "qwen36_35b_a3b_existing_hf_fuse_true",
        "qwen36_35b_a3b_mtp_existing_hf",
        "qwen36_35b_a3b_dflash_existing_hf",
    }
    assert by_name["qwen35_9b_base_fuse_false"].config_overrides == {
        "quant": None,
        "export.model.fuse_gdr_ops": False,
    }
    assert by_name["qwen35_9b_base_fuse_true"].config_overrides == {
        "quant": None,
        "export.model.fuse_gdr_ops": True,
    }
    assert (
        by_name["qwen35_9b_existing_hf_fuse_false"].config_overrides["quant"]["existing_hf_model_dir"]
        == "weights/Qwen3.5-9B-mode1-llm-only"
    )
    assert (
        by_name["qwen36_35b_a3b_existing_hf_fuse_true"].config_overrides["quant"]["existing_hf_model_dir"]
        == "weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400"
    )
    assert by_name["qwen36_35b_a3b_existing_hf_fuse_true"].config_overrides["export.model.fuse_gdr_ops"] is True
    assert by_name["qwen35_9b_mtp_existing_hf"].config_path.endswith("qwen3_5_9b_full_mtp.yaml")
    assert by_name["qwen35_9b_mtp_existing_hf"].config_overrides == {
        "quant": {
            "algorithm": "existing_hf",
            "artifact_format": "gptqmodel_hf",
            "existing_hf_model_dir": "weights/Qwen3.5-9B-mode1-llm-only",
        },
        "export.model.fuse_gdr_ops": False,
    }
    assert by_name["qwen35_9b_dflash_existing_hf"].config_path.endswith("qwen3_5_9b_full_dflash.yaml")
    assert by_name["qwen36_35b_a3b_mtp_existing_hf"].config_path.endswith("qwen3_6_35b_a3b_full_mtp.yaml")
    assert by_name["qwen36_35b_a3b_dflash_existing_hf"].config_path.endswith("qwen3_6_35b_a3b_full_dflash.yaml")


def test_validation_matrix_preflight_checks_paths_without_importing_runtime_modules():
    namespace = _load_validation_matrix_script()
    ValidationScenario = namespace["ValidationScenario"]
    preflight_scenarios = namespace["preflight_scenarios"]
    scenario = ValidationScenario(
        name="missing_paths",
        hf_model_dir="missing/base",
        config_path=str(WORKFLOW_9B_FULL),
        quant_overrides={
            "quant": {
                "algorithm": "existing_hf",
                "artifact_format": "gptqmodel_hf",
            "existing_hf_model_dir": "missing/quant",
            }
        },
        fuse_gdr_ops=False,
    )

    issues = preflight_scenarios([scenario], check_modules=False)

    assert "missing_paths: hf_model_dir does not exist: missing/base" in issues
    assert "missing_paths: quant.existing_hf_model_dir does not exist: missing/quant" in issues


def test_validation_matrix_parser_supports_preflight_only():
    namespace = _load_validation_matrix_script()
    parser = namespace["build_parser"]()
    args = parser.parse_args(
        [
            "--preflight-only",
            "--quick-test",
            "--quick-test-max-new-tokens",
            "16",
            "--skip-export",
            "--scenario",
            "qwen35_9b_base_fuse_false",
        ]
    )

    assert args.preflight_only is True
    assert args.quick_test is True
    assert args.quick_test_max_new_tokens == 16
    assert args.skip_export is True
    assert args.scenario == ["qwen35_9b_base_fuse_false"]


def test_validation_matrix_version_tuple_handles_suffixes():
    namespace = _load_validation_matrix_script()

    assert namespace["_version_tuple"]("5.5.0") == (5, 5, 0)
    assert namespace["_version_tuple"]("2.8.0+cu128") == (2, 8, 0)


def test_workflow_yamls_keep_autoround_llm_only_quant_contract():
    dense_cfg = _load_yaml(WORKFLOW_9B_FULL)
    moe_cfg = _load_yaml(WORKFLOW_MOE_FULL)

    for cfg in (dense_cfg, moe_cfg):
        quant = cfg["quant"]
        assert quant["algorithm"] == "autoround"
        assert quant["output_format"] == "gptqmodel_hf"
        assert quant["artifact_format"] == "gptqmodel_hf"
        assert quant["bits"] == 4
        assert quant["group_size"] == 64
        assert quant["sym"] is True
        assert quant["iters"] == 200
        assert quant["seed"] == 42
        assert quant["quant_nontext_module"] is False
        assert quant["calibration"] == {
            "dataset": "NeelNanda/pile-10k",
            "nsamples": 128,
            "seqlen": 2048,
        }
        assert quant["runtime"]["batch_size"] == 8
        assert quant["runtime"]["trust_remote_code"] is True
        assert cfg["export"]["model"]["quant_scheme"]["quant_type"] == "w8a8h1_sefp"

    assert dense_cfg["quant"]["autoround_format"] == "auto_gptq"
    assert "device_map" not in dense_cfg["quant"]["runtime"]
    assert moe_cfg["quant"]["autoround_format"] == "auto_round:gptqmodel"
    assert moe_cfg["quant"]["runtime"]["device_map"] == "balanced"
    assert moe_cfg["quant"]["runtime"]["low_gpu_mem_usage"] is True
    assert moe_cfg["quant"]["moe"] == {"attn_bits": 8, "shared_expert_bits": 8}


def test_spec_decode_workflow_yamls_are_file_based():
    mtp_9b = _load_yaml(WORKFLOW_9B_MTP)["export"]["model"]
    dflash_9b = _load_yaml(WORKFLOW_9B_DFLASH)["export"]["model"]
    mtp_moe = _load_yaml(WORKFLOW_MOE_MTP)["export"]["model"]
    dflash_moe = _load_yaml(WORKFLOW_MOE_DFLASH)["export"]["model"]

    assert mtp_9b["spec_decode_mode"] == "mtp"
    assert mtp_9b["hf_model"] == "weights/Qwen3.5-9B"
    assert dflash_9b["dflash_config"]["hf_model"] == "weights/Qwen3.5-9B-DFlash"
    assert dflash_9b["output_hidden_state_indices"] == [1, 8, 15, 22, 29]

    assert mtp_moe["spec_decode_mode"] == "mtp"
    assert mtp_moe["hf_model"] == "weights/Qwen3.6-35B-A3B"
    assert dflash_moe["dflash_config"]["hf_model"] == "weights/Qwen3.6-35B-A3B-DFlash"
    assert dflash_moe["output_hidden_state_indices"] == [1, 10, 19, 28, 37]


def test_mtp_workflow_yamls_match_qwen35_hf_attention_shapes():
    expected_shapes = {
        WORKFLOW_9B_MTP: {
            "hidden_size": 4096,
            "num_key_value_heads": 4,
            "head_dim": 256,
        },
        WORKFLOW_27B_MTP: {
            "hidden_size": 5120,
            "num_key_value_heads": 4,
            "head_dim": 256,
        },
        WORKFLOW_MOE_MTP: {
            "hidden_size": 2048,
            "num_key_value_heads": 2,
            "head_dim": 256,
        },
    }

    for workflow_path, expected in expected_shapes.items():
        mtp_config = _load_yaml(workflow_path)["export"]["model"]["mtp_config"]
        assert {key: mtp_config[key] for key in expected} == expected


def test_force_delete_refuses_project_root(monkeypatch):
    namespace = _load_export_script(monkeypatch)

    with pytest.raises(ValueError, match="Refusing to delete unsafe work_dir"):
        namespace["_remove_existing_work_dir"](REPO_ROOT, user_supplied=True)


def test_force_delete_refuses_unmarked_user_work_dir(monkeypatch, tmp_path):
    namespace = _load_export_script(monkeypatch)
    work_dir = tmp_path / "important"
    work_dir.mkdir()
    (work_dir / "notes.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="without export markers"):
        namespace["_remove_existing_work_dir"](work_dir, user_supplied=True)

    assert (work_dir / "notes.txt").is_file()


def test_force_delete_refuses_user_dir_with_only_python_files(monkeypatch, tmp_path):
    namespace = _load_export_script(monkeypatch)
    work_dir = tmp_path / "source_like"
    work_dir.mkdir()
    (work_dir / "module.py").write_text("print('do not delete')", encoding="utf-8")

    with pytest.raises(ValueError, match="without export markers"):
        namespace["_remove_existing_work_dir"](work_dir, user_supplied=True)

    assert (work_dir / "module.py").is_file()


def test_force_delete_allows_marked_user_work_dir(monkeypatch, tmp_path):
    namespace = _load_export_script(monkeypatch)
    work_dir = tmp_path / "old_export"
    work_dir.mkdir()
    (work_dir / "export_hmonnx.log").write_text("old export", encoding="utf-8")

    namespace["_remove_existing_work_dir"](work_dir, user_supplied=True)

    assert not work_dir.exists()


def test_force_delete_allows_default_work_dir_under_safe_root(monkeypatch, tmp_path):
    namespace = _load_export_script(monkeypatch)
    monkeypatch.chdir(tmp_path)
    work_dir = tmp_path / "work_dirs" / "qwen3_5_export"
    work_dir.mkdir(parents=True)
    (work_dir / "notes.txt").write_text("generated", encoding="utf-8")

    namespace["_remove_existing_work_dir"](work_dir, user_supplied=False)

    assert not work_dir.exists()
