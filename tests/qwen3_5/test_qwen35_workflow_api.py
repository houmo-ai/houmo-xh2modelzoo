"""Lightweight tests for the Qwen3.5/Qwen3.6 workflow API and YAML contract.

These tests intentionally avoid GPU, model weights, network, transformers, and
xhquant imports.  They load only workflow/config modules needed for config and
quant dispatch behavior.
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN35_CONFIG_ROOTS = (
    REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5",
    REPO_ROOT / "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe",
)
FORBIDDEN_FILENAME_TOKENS = ("w4", "w8", "w4a8", "w8a8", "gptq", "autoround")


def _install_lightweight_xh_llm_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass xhmodel_merak.xh_llm.__init__, which imports optional deps."""
    packages = {
        "xhmodel_merak.xh_llm": REPO_ROOT / "xhmodel_merak/xh_llm",
        "xhmodel_merak.xh_llm.workflows": REPO_ROOT / "xhmodel_merak/xh_llm/workflows",
        "xhmodel_merak.xh_llm.models": REPO_ROOT / "xhmodel_merak/xh_llm/models",
        "xhmodel_merak.xh_llm.models.qwen3_5": REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_5",
    }
    for name, path in packages.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, module)


def _load_workflow_modules(monkeypatch: pytest.MonkeyPatch):
    _install_lightweight_xh_llm_packages(monkeypatch)
    config_mod = importlib.import_module("xhmodel_merak.xh_llm.workflows.config")
    workflow_mod = importlib.import_module("xhmodel_merak.xh_llm.models.qwen3_5.workflow")
    return config_mod.WorkflowConfig, workflow_mod.Qwen35Workflow


def _workflow_yaml_paths() -> list[Path]:
    paths: list[Path] = []
    for root in QWEN35_CONFIG_ROOTS:
        paths.extend(root.rglob("*.yaml"))
        paths.extend(root.rglob("*.yml"))
    return sorted(paths)


@pytest.fixture()
def qwen35_modules(monkeypatch: pytest.MonkeyPatch):
    return _load_workflow_modules(monkeypatch)


def test_all_qwen35_workflow_yamls_parse(qwen35_modules):
    WorkflowConfig, _ = qwen35_modules
    paths = _workflow_yaml_paths()

    assert paths, "expected Qwen3.5/Qwen3.6 workflow YAMLs"
    for path in paths:
        cfg = WorkflowConfig.from_file(str(path))
        assert cfg.source == str(path)
        assert cfg.export["model"]["chip_arch"] == "XH2a"
        assert cfg.export["model"]["model_type"]
        assert cfg.quant["algorithm"] == "autoround"
        assert cfg.quant["artifact_format"] == "gptqmodel_hf"
        assert cfg.quant["group_size"] == 64


def test_qwen35_workflow_config_accepts_base_and_existing_hf_quant_overrides(qwen35_modules):
    WorkflowConfig, _ = qwen35_modules
    base = WorkflowConfig.from_file(str(QWEN35_CONFIG_ROOTS[0] / "9b/qwen3_5_9b_full.yaml"))

    assert base.with_overrides({"quant": None}).quant is None

    existing_hf = {
        "algorithm": "existing_hf",
        "artifact_format": "gptqmodel_hf",
        "source_algorithm": "autoround",
        "existing_hf_model_dir": "weights/Qwen3.5-9B-mode1-llm-only",
    }
    overridden = base.with_overrides({"quant": existing_hf})

    assert overridden.quant == existing_hf


def test_qwen35_quant_null_requires_explicit_override(qwen35_modules, tmp_path: Path):
    _, Qwen35Workflow = qwen35_modules
    config_path = tmp_path / "bad_quant_null.yaml"
    config_path.write_text(
        """
quant: null
export:
  model:
    chip_arch: XH2a
    model_type: Qwen3_5ForConditionalGeneration
""".lstrip(),
        encoding="utf-8",
    )
    workflow = Qwen35Workflow.from_config(
        hf_model_dir="weights/Qwen3.5-9B",
        config_path=str(config_path),
    )

    with pytest.raises(ValueError, match="Use config_overrides=\\{'quant': None\\}"):
        workflow.quant(output_dir=str(tmp_path / "quant"), device="cpu")

    result = workflow.quant(
        output_dir=str(tmp_path / "quant"),
        device="cpu",
        config_overrides={"quant": None},
    )
    assert result.skipped is True
    assert result.hf_model_dir == os.path.abspath("weights/Qwen3.5-9B")
    assert result.algorithm is None
    assert result.effective_config_file == str(config_path.with_stem("bad_quant_null_override"))


def test_qwen35_quant_rejects_non_64_group_size(qwen35_modules, tmp_path: Path):
    _, Qwen35Workflow = qwen35_modules
    config_path = QWEN35_CONFIG_ROOTS[0] / "9b/qwen3_5_9b_full.yaml"
    workflow = Qwen35Workflow.from_config(
        hf_model_dir="weights/Qwen3.5-9B",
        config_path=str(config_path),
    )

    with pytest.raises(ValueError, match="group_size must be 64"):
        workflow.quant(
            output_dir=str(tmp_path / "quant"),
            device="cpu",
            config_overrides={"quant.group_size": 128},
        )


def test_qwen35_existing_hf_quant_returns_normalized_quant_result(qwen35_modules, tmp_path: Path):
    _, Qwen35Workflow = qwen35_modules
    config_path = QWEN35_CONFIG_ROOTS[0] / "9b/qwen3_5_9b_full.yaml"
    workflow = Qwen35Workflow.from_config(
        hf_model_dir="weights/Qwen3.5-9B",
        config_path=str(config_path),
    )
    existing_rel = "weights/../weights/Qwen3.5-9B-mode1-llm-only"

    result = workflow.quant(
        output_dir=str(tmp_path / "unused_quant"),
        device="cpu",
        config_overrides={
            "quant": {
                "algorithm": "existing_hf",
                "artifact_format": "gptqmodel_hf",
                "source_algorithm": "autoround",
                "existing_hf_model_dir": existing_rel,
            }
        },
    )

    assert result.skipped is False
    assert result.hf_model_dir == os.path.abspath("weights/Qwen3.5-9B")
    assert result.quanted_model_dir == os.path.abspath("weights/Qwen3.5-9B-mode1-llm-only")
    assert result.algorithm == "autoround"
    assert result.effective_config_file == str(config_path.with_stem("qwen3_5_9b_full_override"))


def test_qwen35_autoround_quant_uses_yaml_calibration_runtime_and_stable_export_format(
    qwen35_modules, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _, Qwen35Workflow = qwen35_modules
    calls: dict[str, object] = {}

    class FakeAutoRound:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def quantize_and_save(self, save_path: str, format: str):
            calls["save"] = {"save_path": save_path, "format": format}

    fake_auto_round = types.ModuleType("auto_round")
    fake_auto_round.AutoRound = FakeAutoRound
    monkeypatch.setitem(sys.modules, "auto_round", fake_auto_round)

    config_path = QWEN35_CONFIG_ROOTS[0] / "9b/qwen3_5_9b_full.yaml"
    workflow = Qwen35Workflow.from_config(
        hf_model_dir="weights/Qwen3.5-9B",
        config_path=str(config_path),
    )

    result = workflow.quant(output_dir=str(tmp_path / "quant"), device="cuda:0")

    assert calls["init"] == {
        "model": os.path.abspath("weights/Qwen3.5-9B"),
        "bits": 4,
        "group_size": 64,
        "sym": True,
        "batch_size": 8,
        "seqlen": 2048,
        "nsamples": 128,
        "iters": 200,
        "dataset": "NeelNanda/pile-10k",
        "device": "cuda:0",
        "trust_remote_code": True,
        "seed": 42,
        "quant_nontext_module": False,
    }
    expected_save_path = os.path.abspath(tmp_path / "quant" / "Qwen3.5-9B-autoround-gptqmodel")
    assert calls["save"] == {"save_path": expected_save_path, "format": "auto_gptq"}
    assert result.hf_model_dir == os.path.abspath("weights/Qwen3.5-9B")
    assert result.quanted_model_dir == expected_save_path
    assert result.skipped is False
    assert result.algorithm == "autoround"
    assert result.effective_config_file == str(config_path)


def test_qwen35_autoround_quant_matches_dense_llm_only_script_contract(
    qwen35_modules, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _, Qwen35Workflow = qwen35_modules
    calls: dict[str, object] = {}

    class FakeAutoRound:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def quantize(self):
            calls["quantize"] = True

        def save_quantized(self, save_path: str, format: str):
            calls["save"] = {"save_path": save_path, "format": format}

    fake_auto_round = types.ModuleType("auto_round")
    fake_auto_round.AutoRound = FakeAutoRound
    monkeypatch.setitem(sys.modules, "auto_round", fake_auto_round)

    config_path = QWEN35_CONFIG_ROOTS[0] / "9b/qwen3_5_9b_full.yaml"
    workflow = Qwen35Workflow.from_config(
        hf_model_dir="weights/Qwen3.5-9B",
        config_path=str(config_path),
    )

    workflow.quant(output_dir=str(tmp_path / "quant"), device="cuda:0")

    assert calls["init"]["dataset"] == "NeelNanda/pile-10k"
    assert calls["init"]["batch_size"] == 8
    assert calls["init"]["seed"] == 42
    assert calls["init"]["quant_nontext_module"] is False
    assert "device_map" not in calls["init"]
    assert calls["quantize"] is True
    assert calls["save"]["format"] == "auto_gptq"


def test_qwen35_autoround_quant_passes_moe_runtime_and_layer_overrides(
    qwen35_modules, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    _, Qwen35Workflow = qwen35_modules
    calls: dict[str, object] = {}

    class FakeAutoRound:
        def __init__(self, **kwargs):
            calls["init"] = kwargs

        def quantize(self):
            calls["quantize"] = True

        def save_quantized(self, save_path: str, format: str):
            calls["save"] = {"save_path": save_path, "format": format}

    fake_auto_round = types.ModuleType("auto_round")
    fake_auto_round.AutoRound = FakeAutoRound
    monkeypatch.setitem(sys.modules, "auto_round", fake_auto_round)

    config_path = QWEN35_CONFIG_ROOTS[1] / "35b_a3b/qwen3_6_35b_a3b_full.yaml"
    workflow = Qwen35Workflow.from_config(
        hf_model_dir="weights/Qwen3.6-35B-A3B",
        config_path=str(config_path),
    )

    workflow.quant(output_dir=str(tmp_path / "quant"), device="cuda:0")

    assert calls["init"]["dataset"] == "NeelNanda/pile-10k"
    assert calls["init"]["batch_size"] == 8
    assert calls["init"]["device_map"] == "balanced"
    assert calls["init"]["low_gpu_mem_usage"] is True
    assert calls["init"]["layer_config"] == {
        "model.language_model.layers.*.self_attn.(q_proj|k_proj|v_proj|o_proj)": {
            "bits": 8,
            "group_size": 64,
        },
        "model.language_model.layers.*.linear_attn.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)": {
            "bits": 8,
            "group_size": 64,
        },
        "model.language_model.layers.*.mlp.shared_expert.(gate_proj|up_proj|down_proj)": {
            "bits": 8,
            "group_size": 64,
        },
    }
    assert calls["save"]["format"] == "auto_round:gptqmodel"


def test_qwen35_autoround_quant_rejects_invalid_layer_config_value(qwen35_modules, tmp_path: Path):
    _, Qwen35Workflow = qwen35_modules
    workflow = Qwen35Workflow.from_config(
        hf_model_dir="weights/Qwen3.6-35B-A3B",
        config_path=str(QWEN35_CONFIG_ROOTS[1] / "35b_a3b/qwen3_6_35b_a3b_full.yaml"),
    )

    with pytest.raises(TypeError, match=r"quant\.layer_config\['bad'\] must be a mapping"):
        workflow.quant(
            output_dir=str(tmp_path / "quant"),
            device="cpu",
            config_overrides={
                "quant": {
                    "algorithm": "autoround",
                    "artifact_format": "gptqmodel_hf",
                    "bits": 4,
                    "group_size": 64,
                    "layer_config": {"bad": 8},
                }
            },
        )


def test_qwen35_quant_rejects_mismatched_format_aliases(qwen35_modules, tmp_path: Path):
    _, Qwen35Workflow = qwen35_modules
    workflow = Qwen35Workflow.from_config(
        hf_model_dir="weights/Qwen3.5-9B",
        config_path=str(QWEN35_CONFIG_ROOTS[0] / "9b/qwen3_5_9b_full.yaml"),
    )

    with pytest.raises(ValueError, match="artifact_format and quant.output_format must match"):
        workflow.quant(
            output_dir=str(tmp_path / "unused_quant"),
            device="cpu",
            config_overrides={
                "quant": {
                    "algorithm": "existing_hf",
                    "artifact_format": "gptqmodel_hf",
                    "output_format": "other",
                    "existing_hf_model_dir": "weights/Qwen3.5-9B-mode1-llm-only",
                }
            },
        )


def test_qwen35_workflow_yaml_filenames_are_topology_only():
    paths = _workflow_yaml_paths()

    assert paths, "expected Qwen3.5/Qwen3.6 workflow YAMLs"
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in paths
        if any(token in path.name.lower() for token in FORBIDDEN_FILENAME_TOKENS)
    ]

    assert offenders == []


def test_qwen35_public_api_imports_with_lightweight_stubs(monkeypatch: pytest.MonkeyPatch):
    _install_lightweight_xh_llm_packages(monkeypatch)
    package_name = "xhmodel_merak.xh_llm.models.qwen3_5"
    sys.modules.pop(package_name, None)

    stub_attrs = {
        "modeling_qwen3_5": {"Qwen3_5ForConditionalGeneration": object},
        "qwen3_5_hmonnx_inference": {"XHQwen3_5_HMONNXModel": object},
        "qwen3_5_llm_model": {"XHQwen3_5Model": object},
        "qwen3_5_vision_model": {"XHQwen3_5VisionModel": object},
        "xh_qwen3_5_config": {"XHQwen3_5_VisualConfig": object, "XHQwen3_5ModelConfig": object},
    }
    for module_basename, attrs in stub_attrs.items():
        module = types.ModuleType(f"{package_name}.{module_basename}")
        for attr_name, attr_value in attrs.items():
            setattr(module, attr_name, attr_value)
        monkeypatch.setitem(sys.modules, module.__name__, module)

    module = importlib.import_module(package_name)

    for name in (
        "list_recommended_configs",
        "get_default_quant_config",
        "get_default_export_config",
        "get_default_workflow_config",
        "get_model_docs",
        "get_recommended_config_path",
        "get_quant_config_help",
        "get_export_config_help",
        "dump_quant_config_template",
        "dump_export_config_template",
        "quant",
        "export",
    ):
        assert callable(getattr(module, name))
        assert name in module.__all__


def test_qwen35_default_config_helpers_return_recommended_yaml(
    monkeypatch: pytest.MonkeyPatch,
):
    _install_lightweight_xh_llm_packages(monkeypatch)
    api = importlib.import_module("xhmodel_merak.xh_llm.models.qwen3_5.workflow_api")

    quant_cfg = api.get_default_quant_config()
    assert quant_cfg["algorithm"] == "autoround"
    assert quant_cfg["artifact_format"] == "gptqmodel_hf"
    assert quant_cfg["group_size"] == 64
    assert quant_cfg["sym"] is True
    assert quant_cfg["iters"] == 200
    assert quant_cfg["seed"] == 42
    assert quant_cfg["quant_nontext_module"] is False
    assert quant_cfg["calibration"]["dataset"] == "NeelNanda/pile-10k"
    assert quant_cfg["runtime"]["batch_size"] == 8
    assert quant_cfg["autoround_format"] == "auto_gptq"
    assert "existing_hf" not in quant_cfg

    quant_cfg["group_size"] = 128
    assert api.get_default_quant_config()["group_size"] == 64

    mtp_cfg = api.get_default_workflow_config(name="qwen3_5_9b_full_mtp")
    assert mtp_cfg["export"]["model"]["hf_model"] == "weights/Qwen3.5-9B"
    assert mtp_cfg["export"]["model"]["spec_decode_mode"] == "mtp"

    dflash_export = api.get_default_export_config(
        family="moe",
        model_size="35b_a3b",
        variant="dflash",
    )
    assert dflash_export["model"]["model_type"] == "Qwen3_5MoeForConditionalGeneration"
    assert dflash_export["model"]["dflash_config"]["target_model_dir"] is None

    visual_export = api.get_default_export_config(
        family="dense",
        model_size="9b",
        variant="visual_only",
        visual_size=896,
    )
    assert visual_export["model"]["max_size_w"] == 896
    assert visual_export["model"]["max_size_h"] == 896

    with pytest.raises(ValueError, match="Expected exactly one"):
        api.get_default_export_config(family="dense", model_size="9b", variant="visual_only")


def test_qwen35_list_recommended_configs_is_structured_and_complete(monkeypatch: pytest.MonkeyPatch):
    _install_lightweight_xh_llm_packages(monkeypatch)
    api = importlib.import_module("xhmodel_merak.xh_llm.models.qwen3_5.workflow_api")

    configs = api.list_recommended_configs()

    assert len(configs) == 15
    assert configs == sorted(configs, key=lambda item: item["config_path"])
    assert {item["variant"] for item in configs} == {"full", "mtp", "dflash", "visual_only"}
    assert {item["visual_size"] for item in configs if item["variant"] == "visual_only"} == {448, 896}
    for item in configs:
        assert set(item) == {
            "name",
            "family",
            "model_size",
            "variant",
            "visual_size",
            "config_path",
            "model_type",
            "model_name",
        }
        assert not os.path.isabs(item["config_path"])
        assert (REPO_ROOT / item["config_path"]).is_file()


def test_qwen35_template_dumps_are_loadable_and_match_current_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _install_lightweight_xh_llm_packages(monkeypatch)
    api = importlib.import_module("xhmodel_merak.xh_llm.models.qwen3_5.workflow_api")
    yaml = pytest.importorskip("yaml")

    quant_path = Path(api.dump_quant_config_template(tmp_path / "quant.yaml"))
    export_path = Path(api.dump_export_config_template(tmp_path / "export.yaml"))

    quant_cfg = yaml.safe_load(quant_path.read_text(encoding="utf-8"))
    export_cfg = yaml.safe_load(export_path.read_text(encoding="utf-8"))

    assert quant_cfg["algorithm"] == "autoround"
    assert quant_cfg["artifact_format"] == "gptqmodel_hf"
    assert quant_cfg["output_format"] == "gptqmodel_hf"
    assert quant_cfg["group_size"] == 64
    assert quant_cfg["sym"] is True
    assert quant_cfg["iters"] == 200
    assert quant_cfg["seed"] == 42
    assert quant_cfg["quant_nontext_module"] is False
    assert quant_cfg["calibration"]["dataset"] == "NeelNanda/pile-10k"
    assert quant_cfg["runtime"]["batch_size"] == 8
    assert quant_cfg["autoround_format"] == "auto_gptq"
    assert quant_cfg["existing_hf"]["algorithm"] == "existing_hf"
    assert export_cfg["variants"] == ["full", "mtp", "dflash", "visual_only"]
    assert export_cfg["visual_sizes"] == [448, 896]
    assert export_cfg["full"]["model"]["visual_config"]["max_size_w"] == 448
    assert export_cfg["visual_only"]["model"]["max_size_h"] == 448


def test_qwen35_help_matches_supported_quant_and_export_modes(monkeypatch: pytest.MonkeyPatch):
    _install_lightweight_xh_llm_packages(monkeypatch)
    api = importlib.import_module("xhmodel_merak.xh_llm.models.qwen3_5.workflow_api")

    quant_help = api.get_quant_config_help()
    export_help = api.get_export_config_help()

    assert quant_help["default"]["algorithm"] == "autoround"
    assert quant_help["default"]["artifact_format"] == "gptqmodel_hf"
    assert quant_help["default"]["group_size"] == 64
    assert quant_help["fields"]["group_size"]["default"] == 64
    assert quant_help["fields"]["iters"]["default"] == 200
    assert quant_help["fields"]["seed"]["default"] == 42
    assert quant_help["fields"]["quant_nontext_module"]["default"] is False
    assert quant_help["fields"]["runtime.batch_size"]["default"] == 8
    assert quant_help["fields"]["autoround_format"]["default"] == "auto_gptq"
    assert "group_size must be 64" in quant_help["required_constraints"]
    assert "existing_hf" in quant_help["supported_algorithms"]
    assert export_help["variants"] == ["full", "mtp", "dflash", "visual_only"]
    assert export_help["visual_sizes"] == [448, 896]
    assert export_help["fields"]["export.model.fuse_gdr_ops"]["default"] is False
    assert "Qwen3_5ForConditionalGeneration_visual" in export_help["model_types"]["visual_only"]


def test_qwen35_model_docs_are_complete_enough(monkeypatch: pytest.MonkeyPatch):
    _install_lightweight_xh_llm_packages(monkeypatch)
    api = importlib.import_module("xhmodel_merak.xh_llm.models.qwen3_5.workflow_api")

    model_docs = api.get_model_docs()

    assert model_docs["hmonnx_io_doc"] == "docs/qwen3_5_hmonnx_io_spec.md"
    assert model_docs["hmonnx_io_doc_exists"] is True
    assert model_docs["default_export_summary"]["fuse_gdr_ops"] is False
    assert model_docs["default_export_summary"]["visual_sizes"] == [448, 896]

    supported = {(item["family"], item["model_size"]): item for item in model_docs["supported_models"]}
    assert supported[("qwen3_5", "9b")]["verified_quant_model"] == "weights/Qwen3.5-9B-mode1-llm-only"
    assert (
        supported[("qwen3_5_moe", "35b_a3b")]["verified_quant_model"]
        == "weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400"
    )
    assert model_docs["validation_scope"]["spec_decode"].startswith("9B and 35B-A3B")


def test_qwen35_module_level_wrappers_delegate_to_workflow(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    _install_lightweight_xh_llm_packages(monkeypatch)
    api = importlib.import_module("xhmodel_merak.xh_llm.models.qwen3_5.workflow_api")

    calls: list[tuple[str, dict[str, object]]] = []

    class FakeWorkflow:
        def __init__(self, hf_model_dir: str, config_path: str, seed: int = 1024, debug: bool = False):
            calls.append(
                (
                    "init",
                    {"hf_model_dir": hf_model_dir, "config_path": config_path, "seed": seed, "debug": debug},
                )
            )

        def quant(self, output_dir: str, device: str, config_overrides=None):
            calls.append(("quant", {"output_dir": output_dir, "device": device, "config_overrides": config_overrides}))
            return "quant-result"

        def export(self, quant_result, output_dir: str, device: str, config_overrides=None):
            calls.append(
                (
                    "export",
                    {
                        "quant_result": quant_result,
                        "output_dir": output_dir,
                        "device": device,
                        "config_overrides": config_overrides,
                    },
                )
            )
            return "export-result"

    monkeypatch.setattr(api, "Qwen35Workflow", FakeWorkflow)

    quant_result = api.quant(
        hf_model_dir="weights/model",
        config_path="configs/model.yaml",
        output_dir=tmp_path / "quant",
        device="cpu",
        config_overrides={"quant": None},
        seed=7,
        debug=True,
    )
    export_result = api.export(
        hf_model_dir="weights/model",
        config_path="configs/model.yaml",
        quant_result=quant_result,
        output_dir=tmp_path / "export",
        device="cuda:0",
        config_overrides={"export.model.context_max_length": 128},
    )

    assert quant_result == "quant-result"
    assert export_result == "export-result"
    assert calls == [
        ("init", {"hf_model_dir": "weights/model", "config_path": "configs/model.yaml", "seed": 7, "debug": True}),
        ("quant", {"output_dir": str(tmp_path / "quant"), "device": "cpu", "config_overrides": {"quant": None}}),
        ("init", {"hf_model_dir": "weights/model", "config_path": "configs/model.yaml", "seed": 1024, "debug": False}),
        (
            "export",
            {
                "quant_result": "quant-result",
                "output_dir": str(tmp_path / "export"),
                "device": "cuda:0",
                "config_overrides": {"export.model.context_max_length": 128},
            },
        ),
    ]
