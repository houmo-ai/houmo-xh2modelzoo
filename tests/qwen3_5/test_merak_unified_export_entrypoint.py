"""Lightweight checks for the config-only Merak Qwen3.5 export entrypoint."""
from __future__ import annotations

import runpy
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = REPO_ROOT / "examples_merak/llm/qwen3_5/qwen3_5_xh_export_hmonnx.py"
REMOVED_MOE_EXAMPLE_DIR = REPO_ROOT / "examples_merak/llm/qwen3_5_moe"
README = REPO_ROOT / "examples_merak/llm/qwen3_5/README.md"

CONFIG_9B_FLOAT = REPO_ROOT / "configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_instruct_xh2a_2k.py"
CONFIG_9B_QUANT = REPO_ROOT / "configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_instruct_hf_gptq_xh2a_2k.py"
CONFIG_9B_MTP = REPO_ROOT / "configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_spec_mtp_xh2a_2k.py"
CONFIG_9B_DFLASH = REPO_ROOT / "configs_merak/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_spec_dflash_xh2a_2k.py"
CONFIG_MOE_FLOAT = (
    REPO_ROOT / "configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_instruct_xh2a_2k.py"
)
CONFIG_MOE_QUANT = (
    REPO_ROOT
    / "configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_instruct_hf_autoround_xh2a_2k.py"
)
CONFIG_MOE_MTP = (
    REPO_ROOT / "configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_spec_mtp_xh2a_2k.py"
)
CONFIG_MOE_DFLASH = (
    REPO_ROOT / "configs_merak/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_5_moe_35b_a3b_spec_dflash_xh2a_2k.py"
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


def test_export_entrypoint_is_config_only(monkeypatch):
    namespace = _load_export_script(monkeypatch)
    parser = namespace["build_parser"]()
    option_strings = {opt for action in parser._actions for opt in action.option_strings}

    assert {"--config", "--debug", "--force", "--seed", "--work-dir", "--work_dir"} <= option_strings
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
    assert getattr(parser._option_string_actions["--config"], "required") is True


def test_removed_moe_example_directory_is_deleted():
    assert not REMOVED_MOE_EXAMPLE_DIR.exists()


def test_readme_does_not_document_model_override_cli():
    src = README.read_text(encoding="utf-8")
    assert "--model " not in src
    assert "--quant-weight" not in src
    assert "qwen3_5_moe/qwen3_5_moe" not in src


def test_config_paths_cover_requested_models():
    assert 'hf_model_dir = "weights/Qwen3.5-9B"' in CONFIG_9B_FLOAT.read_text()
    assert "/data01/home/yujy/work/gptqmodel/output/Qwen3.5-9B-mode1-llm-only" in CONFIG_9B_QUANT.read_text()
    assert 'hf_model_dir = "weights/Qwen3.6-35B-A3B"' in CONFIG_MOE_FLOAT.read_text()
    assert 'hf_model_dir = "weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400"' in CONFIG_MOE_QUANT.read_text()


def test_quantized_hf_configs_use_hf_model_not_quant_weight():
    for path in (CONFIG_9B_QUANT, CONFIG_MOE_QUANT):
        src = path.read_text()
        assert "quant_weight=None" in src
        assert "hf_model=hf_model_dir" in src
        assert "quant_type=\"w4a8h1_sefp\"" in src


def test_spec_decode_configs_are_file_based():
    mtp_9b = CONFIG_9B_MTP.read_text()
    dflash_9b = CONFIG_9B_DFLASH.read_text()
    mtp_moe = CONFIG_MOE_MTP.read_text()
    dflash_moe = CONFIG_MOE_DFLASH.read_text()

    assert 'spec_decode_mode="mtp"' in mtp_9b
    assert 'hf_model_dir = "weights/Qwen3.5-9B"' in mtp_9b
    assert 'dflash_model_dir = "weights/Qwen3.5-9B-DFlash"' in dflash_9b
    assert "output_hidden_state_indices=[1, 8, 15, 22, 29]" in dflash_9b

    assert 'spec_decode_mode="mtp"' in mtp_moe
    assert 'hf_model_dir = "weights/Qwen3.6-35B-A3B"' in mtp_moe
    assert 'dflash_model_dir = "weights/Qwen3.6-35B-A3B-DFlash"' in dflash_moe
    assert "output_hidden_state_indices=[1, 10, 19, 28, 37]" in dflash_moe


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
