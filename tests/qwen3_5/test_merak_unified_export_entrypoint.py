"""Lightweight checks for the unified Merak Qwen3.5 export entrypoint."""
from __future__ import annotations

import runpy
import sys
import types
from argparse import Namespace
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = REPO_ROOT / "examples_merak/llm/qwen3_5/qwen3_5_xh_export_hmonnx.py"
REMOVED_MOE_EXPORT_SCRIPT = (
    REPO_ROOT / "examples_merak/llm/qwen3_5_moe/qwen3_5_moe_xh_export_hmonnx.py"
)


def _install_stub_modules(monkeypatch):
    fake_torch = types.ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)

    fake_xh_llm = types.ModuleType("xhmodel_merak.xh_llm")
    fake_xh_llm.AutoLLMConfig = object
    fake_xh_llm.AutoLLMModel = object
    fake_xh_llm.format_model_name = lambda cfg: cfg
    fake_xh_llm.support_llm_model_types = (
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    )

    class DummyConfig:
        def __init__(self, cfg):
            self._cfg = cfg
            for key, value in cfg.items():
                if key == "model" and isinstance(value, dict):
                    value = types.SimpleNamespace(**value)
                setattr(self, key, value)

        @classmethod
        def fromfile(cls, path):
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
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "xhmodel_merak.xh_llm", fake_xh_llm)
    monkeypatch.setitem(sys.modules, "xhquant.api", fake_xhquant_api)
    monkeypatch.setitem(sys.modules, "xhquant.utils", fake_xhquant_utils)


def _load_export_script(monkeypatch):
    _install_stub_modules(monkeypatch)
    return runpy.run_path(str(EXPORT_SCRIPT), run_name="qwen3_5_unified_export_test")


def _args(model_dir: Path, **overrides):
    values = dict(
        model=str(model_dir),
        model_type="",
        chip_arch="XH2a",
        dtype="fp16",
        batch_size=1,
        context_length=2048,
        prefill_chunk_length=256,
        max_pe_length=None,
        num_logits_to_keep=1,
        linear_attention_mode="auto",
        linear_chunk_size=64,
        quant_type=None,
        quant_weight=None,
        max_size_w=448,
        max_size_h=448,
        spec_decode_mode=None,
        dflash_model_dir=None,
        num_draft_tokens=4,
        spec_draft_head_weight_bits=4,
        split_conv_cache=True,
        normalize_force_fp32=False,
        use_manual_depthwise_conv1d=False,
        release_xh_version=None,
        release_modelscope_name=None,
        release_wmix_amix=None,
        release_date=None,
        package_release=False,
    )
    values.update(overrides)
    return Namespace(**values)


def test_unified_export_script_parses(monkeypatch):
    _load_export_script(monkeypatch)


def test_removed_moe_export_script_is_deleted():
    assert not REMOVED_MOE_EXPORT_SCRIPT.exists()


def test_build_cfg_from_dense_model_infers_architecture(monkeypatch, tmp_path):
    namespace = _load_export_script(monkeypatch)
    monkeypatch.setattr(
        namespace["AutoConfig"],
        "from_pretrained",
        lambda *args, **kwargs: types.SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"]
        ),
    )

    model_dir = tmp_path / "Qwen3.5-9B"
    model_dir.mkdir()
    _, cfg = namespace["_build_cfg_from_model"](_args(model_dir))

    assert cfg.model.model_type == "Qwen3_5ForConditionalGeneration"
    assert cfg.dtype == "fp16"
    assert cfg.model.quant_scheme["quant_type"] == "w8a8h1_sefp"
    assert cfg.model.batch_size == 1
    assert cfg.model.max_pe_length == 32768
    assert cfg.model.num_logits_to_keep == 1
    assert cfg.model.linear_attention_mode == "auto"
    assert cfg.model.linear_chunk_size == 64
    assert cfg.model.split_conv_cache is True
    assert cfg.model.normalize_force_fp32 is False
    assert cfg.model.use_manual_depthwise_conv1d is False
    assert cfg.model.visual_config["max_size_w"] == 448
    assert cfg.model.visual_config["max_size_h"] == 448
    assert cfg.model.visual_config["quant_scheme"]["quant_type"] == "w8a8h1_sefp"


def test_build_cfg_from_model_applies_legacy_export_arg_overrides(monkeypatch, tmp_path):
    namespace = _load_export_script(monkeypatch)
    monkeypatch.setattr(
        namespace["AutoConfig"],
        "from_pretrained",
        lambda *args, **kwargs: types.SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"]
        ),
    )

    model_dir = tmp_path / "Qwen3.5-9B"
    model_dir.mkdir()
    _, cfg = namespace["_build_cfg_from_model"](
        _args(
            model_dir,
            dtype="bf16",
            batch_size=2,
            max_pe_length=65536,
            num_logits_to_keep=0,
            linear_attention_mode="chunk",
            linear_chunk_size=128,
            release_xh_version="xh2",
            release_modelscope_name="Qwen3.5-9B",
            release_wmix_amix="wmix_amix",
            release_date="20260603",
            package_release=True,
            spec_draft_head_weight_bits=8,
        )
    )

    assert cfg.dtype == "bf16"
    assert cfg.model.batch_size == 2
    assert cfg.model.max_pe_length == 65536
    assert cfg.model.num_logits_to_keep == 0
    assert cfg.model.linear_attention_mode == "chunk"
    assert cfg.model.linear_chunk_size == 128
    assert cfg.release["xh_version"] == "xh2"
    assert cfg.release["modelscope_name"] == "Qwen3.5-9B"
    assert cfg.release["wmix_amix"] == "wmix_amix"
    assert cfg.release["date"] == "20260603"
    assert cfg.release["package_release"] is True
    assert cfg.export_options["spec_draft_head_weight_bits"] == 8


def test_build_cfg_from_moe_model_infers_architecture_and_visual_config(monkeypatch, tmp_path):
    namespace = _load_export_script(monkeypatch)
    monkeypatch.setattr(
        namespace["AutoConfig"],
        "from_pretrained",
        lambda *args, **kwargs: types.SimpleNamespace(
            architectures=["Qwen3_5MoeForConditionalGeneration"]
        ),
    )

    model_dir = tmp_path / "Qwen3.5-35B-A3B"
    model_dir.mkdir()
    _, cfg = namespace["_build_cfg_from_model"](_args(model_dir, max_size_w=560, max_size_h=336))

    assert cfg.model.model_type == "Qwen3_5MoeForConditionalGeneration"
    assert cfg.model.quant_scheme["quant_type"] == "w8a8h0_ssfp"
    assert cfg.model.split_conv_cache is True
    assert cfg.model.normalize_force_fp32 is False
    assert cfg.model.use_manual_depthwise_conv1d is False
    assert cfg.model.visual_config["max_size_w"] == 560
    assert cfg.model.visual_config["max_size_h"] == 336
    assert cfg.model.visual_config["quant_scheme"]["quant_type"] == "w8a8h0_ssfp"


def test_build_cfg_from_model_applies_conv_and_normalize_overrides(monkeypatch, tmp_path):
    namespace = _load_export_script(monkeypatch)
    monkeypatch.setattr(
        namespace["AutoConfig"],
        "from_pretrained",
        lambda *args, **kwargs: types.SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"]
        ),
    )

    model_dir = tmp_path / "Qwen3.5-9B"
    model_dir.mkdir()
    _, cfg = namespace["_build_cfg_from_model"](
        _args(
            model_dir,
            split_conv_cache=False,
            normalize_force_fp32=True,
            use_manual_depthwise_conv1d=True,
        )
    )

    assert cfg.model.split_conv_cache is False
    assert cfg.model.normalize_force_fp32 is True
    assert cfg.model.use_manual_depthwise_conv1d is True


def test_build_cfg_from_quantized_hf_repo_uses_repo_as_hf_model(monkeypatch, tmp_path):
    namespace = _load_export_script(monkeypatch)
    source_model_dir = tmp_path / "Qwen3.5-9B"
    quant_model_dir = tmp_path / "Qwen3.5-9B-mode1-llm-only"
    source_model_dir.mkdir()
    quant_model_dir.mkdir()
    (quant_model_dir / "config.json").write_text("{}", encoding="utf-8")

    def fake_from_pretrained(path, **_kwargs):
        architectures = ["Qwen3_5ForConditionalGeneration"]
        if Path(path).name == quant_model_dir.name:
            return types.SimpleNamespace(
                architectures=architectures,
                quantization_config={"quant_method": "gptq"},
            )
        return types.SimpleNamespace(architectures=architectures, quantization_config=None)

    monkeypatch.setattr(namespace["AutoConfig"], "from_pretrained", fake_from_pretrained)

    _, cfg = namespace["_build_cfg_from_model"](
        _args(source_model_dir, quant_weight=str(quant_model_dir), quant_type="w4a8h1_sefp")
    )

    assert cfg.model.hf_model == str(quant_model_dir)
    assert cfg.model.model_name == source_model_dir.name
    assert cfg.model.quant_weight is None
    assert cfg.model.quant_scheme["quant_type"] == "w4a8h1_sefp"


def test_build_cfg_from_checkpoint_quant_weight_keeps_quant_weight(monkeypatch, tmp_path):
    namespace = _load_export_script(monkeypatch)
    monkeypatch.setattr(
        namespace["AutoConfig"],
        "from_pretrained",
        lambda *args, **kwargs: types.SimpleNamespace(
            architectures=["Qwen3_5ForConditionalGeneration"],
            quantization_config=None,
        ),
    )

    model_dir = tmp_path / "Qwen3.5-9B"
    quant_weight = tmp_path / "quant_weight.pt"
    model_dir.mkdir()
    quant_weight.write_bytes(b"checkpoint")
    _, cfg = namespace["_build_cfg_from_model"](_args(model_dir, quant_weight=str(quant_weight)))

    assert cfg.model.hf_model == str(model_dir)
    assert cfg.model.quant_weight == str(quant_weight)


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
