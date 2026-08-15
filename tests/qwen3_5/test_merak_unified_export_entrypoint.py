"""Lightweight checks for the config-only Merak Qwen3.5 export entrypoint."""

from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import pytest

from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = REPO_ROOT / "examples_merak/llm/qwen3_5/debug_scripts/qwen3_5_xh_export_hmonnx.py"
BASE_LLM_MODEL = REPO_ROOT / "xhmodel_merak/xh_llm/base_llm_model.py"
REMOVED_MOE_EXAMPLE_DIR = REPO_ROOT / "examples_merak/llm/qwen3_5_moe"
README = REPO_ROOT / "examples_merak/llm/qwen3_5/README.md"

WORKFLOW_9B_FULL = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml"
WORKFLOW_9B_MTP = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_mtp.yaml"
WORKFLOW_9B_DFLASH = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_dflash.yaml"
WORKFLOW_MOE_FULL = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml"
WORKFLOW_9B_FULL_GPTQ = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_gptq.yaml"
WORKFLOW_MOE_FULL_GPTQ = (
    REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_gptq.yaml"
)
WORKFLOW_MOE_MTP = (
    REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_mtp.yaml"
)
WORKFLOW_MOE_DFLASH = (
    REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_dflash.yaml"
)
WORKFLOW_122B_DFLASH = (
    REPO_ROOT
    / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/122b_a10b"
    / "qwen3_5_122b_a10b_full_dflash.yaml"
)
WORKFLOW_27B_VISUAL_448 = (
    REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_6_27b_visual_only_448.yaml"
)
WORKFLOW_27B_VISUAL_896 = (
    REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_6_27b_visual_only_896.yaml"
)
WORKFLOW_27B_MTP = REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_6_27b_full_mtp.yaml"
WORKFLOW_38_27B_FULL = (
    REPO_ROOT
    / "configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_8_27b_full.yaml"
)
WORKFLOW_38_27B_MTP = (
    REPO_ROOT
    / "configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_8_27b_full_mtp.yaml"
)
WORKFLOW_38_27B_VISUAL_GEARS = (
    REPO_ROOT
    / "configs_merak/workflows/xh2a/llm_models/qwen3_5/27b"
    / "qwen3_8_27b_visual_only_token_gears_99p_candidate.yaml"
)


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
    return WorkflowConfig.from_file(str(path)).data


def test_hf_config_copy_excludes_weight_index():
    source = BASE_LLM_MODEL.read_text(encoding="utf-8")

    assert 'cfg_file.name.endswith(".safetensors.index.json")' in source


def test_legacy_moe_example_directory_is_not_part_of_qwen35_readme():
    assert REMOVED_MOE_EXAMPLE_DIR.exists()
    src = README.read_text(encoding="utf-8")
    assert "examples_merak/llm/qwen3_5_moe" not in src


def test_readme_does_not_document_model_override_cli():
    src = README.read_text(encoding="utf-8")
    assert "--model " not in src
    assert "--quant-weight" not in src
    assert "qwen3_5_moe/qwen3_5_moe" not in src


def test_workflow_config_paths_cover_requested_models():
    assert _load_yaml(WORKFLOW_9B_FULL)["export"]["model"]["hf_model"] is None
    assert _load_yaml(WORKFLOW_MOE_FULL)["export"]["model"]["hf_model"] is None
    assert _load_yaml(WORKFLOW_27B_VISUAL_448)["export"]["model"]["hf_model"] is None
    assert _load_yaml(WORKFLOW_27B_VISUAL_896)["export"]["model"]["hf_model"] is None
    assert _load_yaml(WORKFLOW_38_27B_FULL)["export"]["model"]["hf_model"] is None
    assert _load_yaml(WORKFLOW_38_27B_MTP)["export"]["model"]["hf_model"] is None
    assert (
        _load_yaml(WORKFLOW_38_27B_VISUAL_GEARS)["export"]["model"]["hf_model"]
        is None
    )


def test_qwen38_workflows_reuse_qwen35_family_contract():
    base = _load_yaml(WORKFLOW_38_27B_FULL)["export"]["model"]
    mtp = _load_yaml(WORKFLOW_38_27B_MTP)["export"]["model"]
    visual = _load_yaml(WORKFLOW_38_27B_VISUAL_GEARS)["export"]["model"]

    for model in (base, mtp):
        assert model["model_type"] == "Qwen3_5ForConditionalGeneration"
        assert model["model_name"] == "qwen3_8_27b"
        assert model["context_max_length"] == 2048
        assert model["fuse_gdr_ops"] is False
        assert model["fuse_gdr_block_recurrent_ops"] is False

    assert mtp["spec_decode_mode"] == "mtp"
    assert mtp["num_draft_tokens"] == 4
    assert mtp["mtp_config"]["hidden_size"] == 5120
    assert mtp["mtp_config"]["num_key_value_heads"] == 4
    assert mtp["mtp_config"]["head_dim"] == 256

    assert visual["model_type"] == "Qwen3_5ForConditionalGeneration_visual"
    assert visual["model_name"] == "qwen3_8_27b_visual_token_gears"
    assert visual["image_token_gears"] == [96, 196, 384, 704, 1536]
    assert visual["image_token_capacity"] == 1536


def test_workflow_yamls_keep_autoround_llm_only_quant_contract():
    dense_cfg = _load_yaml(WORKFLOW_9B_FULL)
    moe_cfg = _load_yaml(WORKFLOW_MOE_FULL)

    for cfg in (dense_cfg, moe_cfg):
        quant = cfg["quant"]
        assert quant["algorithm"] == "gptqmodel"
        assert quant["output_format"] == "gptqmodel_hf"
        assert quant["artifact_format"] == "gptqmodel_hf"
        assert quant["bits"] == 4
        assert quant["group_size"] == 64
        assert quant["method"] == "autoround"
        assert quant["rotation"] is False
        assert quant["sym"] is True
        assert quant["iters"] == 200
        assert quant["seed"] == 42
        assert quant["quant_nontext_module"] is False
        assert quant["calibration"]["nsamples"] == 128
        assert quant["calibration"]["seqlen"] == 2048
        assert quant["runtime"]["trust_remote_code"] is True

    assert dense_cfg["quant"]["calibration"]["dataset"] == "NeelNanda/pile-10k"
    assert moe_cfg["quant"]["calibration"]["jsonl"] == ("xh2modelzoo://data/calib_data/NeelNanda-pile-10k.jsonl")

    assert dense_cfg["export"]["model"]["quant_scheme"]["quant_type"] == "w8a8h1_sefp"
    assert moe_cfg["export"]["model"]["quant_scheme"]["quant_type"] == "w8a8h1_sefp"

    assert dense_cfg["quant"]["format"] == "auto_gptq"
    assert dense_cfg["quant"]["runtime"]["batch_size"] == 8
    assert dense_cfg["quant"]["runtime"]["low_gpu_mem_usage"] is True
    assert "device_map" not in dense_cfg["quant"]["runtime"]
    assert moe_cfg["quant"]["format"] == "auto_round:gptqmodel"
    assert moe_cfg["quant"]["runtime"]["batch_size"] == 8
    assert moe_cfg["quant"]["runtime"]["gradient_accumulate_steps"] == 1
    assert moe_cfg["quant"]["runtime"]["device_map"] == "0"
    assert moe_cfg["quant"]["runtime"]["low_gpu_mem_usage"] is True
    assert moe_cfg["quant"]["moe"] == {"attn_bits": 8, "shared_expert_bits": 8}


def test_workflow_yamls_include_gptq_companion_configs():
    dense_cfg = _load_yaml(WORKFLOW_9B_FULL_GPTQ)
    moe_cfg = _load_yaml(WORKFLOW_MOE_FULL_GPTQ)

    for cfg in (dense_cfg, moe_cfg):
        quant = cfg["quant"]
        assert quant["algorithm"] == "gptqmodel"
        assert quant["method"] == "gptq"
        assert quant["artifact_format"] == "gptqmodel_hf"
        assert quant["preset"] == "full_vlm"
        assert quant["bits"] == 4
        assert quant["group_size"] == 64
        assert quant["rotation"] is False
        assert quant["hessian_mse"] is True
        assert quant["runtime"] == {
            "batch_size": 1,
            "trust_remote_code": True,
            "device_map": "auto",
            "offload_to_disk": False,
        }
        assert quant["calibration"]["text_key"] == "text"
        assert quant["calibration"]["seqlen"] == 1024
        assert quant["validation"]["check_quant_vision_demo"] is True
        model_name = cfg["export"]["model"]["model_name"]
        assert not model_name.startswith("xh2_")
        assert model_name != "auto"
        assert "_mpe" not in model_name
        assert "_w4a8" not in model_name
        assert "naming" not in cfg["export"]

    assert dense_cfg["quant"]["calibration"]["jsonl"].endswith("Qwen3.5-27B.jsonl")
    assert dense_cfg["quant"]["calibration"]["nsamples"] == 256
    assert moe_cfg["quant"]["calibration"]["jsonl"].endswith("Qwen3-Next-80B-A3B-Instruct.jsonl")
    assert moe_cfg["quant"]["calibration"]["nsamples"] == 512
    assert moe_cfg["quant"]["moe"]["attn_bits"] == 8
    assert moe_cfg["quant"]["moe"]["shared_expert_bits"] == 8


def test_spec_decode_workflow_yamls_are_file_based():
    mtp_9b = _load_yaml(WORKFLOW_9B_MTP)["export"]["model"]
    dflash_9b = _load_yaml(WORKFLOW_9B_DFLASH)["export"]["model"]
    mtp_moe = _load_yaml(WORKFLOW_MOE_MTP)["export"]["model"]
    dflash_moe = _load_yaml(WORKFLOW_MOE_DFLASH)["export"]["model"]
    dflash_122b = _load_yaml(WORKFLOW_122B_DFLASH)["export"]["model"]

    assert mtp_9b["spec_decode_mode"] == "mtp"
    assert mtp_9b["hf_model"] is None
    assert dflash_9b["dflash_config"]["hf_model"] == "weights/Qwen3.5-9B-DFlash"
    assert "output_hidden_state_indices" not in dflash_9b

    assert mtp_moe["spec_decode_mode"] == "mtp"
    assert mtp_moe["hf_model"] is None
    assert dflash_moe["dflash_config"]["hf_model"] == "weights/Qwen3.6-35B-A3B-DFlash"
    assert "output_hidden_state_indices" not in dflash_moe

    assert dflash_122b["dflash_config"]["hf_model"] == (
        "weights/Qwen3.5-122B-A10B-DFlash"
    )
    assert "output_hidden_state_indices" not in dflash_122b
    assert "num_hidden_layers" not in dflash_122b["dflash_config"]
    assert "num_target_layers" not in dflash_122b["dflash_config"]


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
        WORKFLOW_38_27B_MTP: {
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
