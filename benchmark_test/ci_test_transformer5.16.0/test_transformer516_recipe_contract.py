"""Cheap GPTQ/AutoRound API contracts for the four maintained Transformer blocks.

Real AutoRound quantization is performed once when publishing the immutable
checkpoints.  Pull-request CI checks ModelZoo's workflow translation and
GPTQModel recipe signatures here, then exports the published blocks in the
model-specific tests.
"""
from __future__ import annotations

import ast
import copy
import functools
import importlib.util
import inspect
import json
import os
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ContractCase:
    model: str
    family: str
    backend: str
    config: str
    bits: int
    model_name: str

    @property
    def case_id(self) -> str:
        return f"{self.model}-{self.backend}"


QWEN_DENSE_GPTQ_9B = "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_gptq.yaml"
QWEN_MOE_GPTQ_35B = "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_gptq.yaml"
QWEN_DENSE_AR_9B = "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml"
QWEN_MOE_AR_35B = "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml"
GEMMA_E2B_GPTQ = "configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml"
GEMMA_E2B_AR = "configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_autoround.yaml"
GEMMA_26B_GPTQ = "configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml"
GEMMA_26B_AR = "configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_autoround.yaml"


MODEL_CONFIGS: tuple[tuple[str, str, int, str, str, str], ...] = (
    ("Qwen3.5-9B", "qwen3_5", 4, QWEN_DENSE_GPTQ_9B, QWEN_DENSE_AR_9B, "qwen3_5_9b"),
    ("Qwen3.5-35B-A3B", "qwen3_5", 4, QWEN_MOE_GPTQ_35B, QWEN_MOE_AR_35B, "qwen3_5_35b_a3b"),
    ("gemma-4-E2B-it", "gemma4", 4, GEMMA_E2B_GPTQ, GEMMA_E2B_AR, "gemma_4_e2b"),
    ("gemma-4-26B-A4B-it", "gemma4", 4, GEMMA_26B_GPTQ, GEMMA_26B_AR, "gemma_4_26b_a4b"),
)


CASES = tuple(
    ContractCase(model, family, backend, config, bits, model_name)
    for model, family, bits, gptq_config, autoround_config, model_name in MODEL_CONFIGS
    for backend, config in (("gptq", gptq_config), ("autoround", autoround_config))
)


def _set_dotted(data: dict[str, Any], dotted_key: str, value: Any) -> None:
    target = data
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        child = target.get(part)
        if not isinstance(child, dict):
            child = {}
            target[part] = child
        target = child
    target[parts[-1]] = value


def _resolved_config(case: ContractCase) -> dict[str, Any]:
    path = REPO_ROOT / case.config
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data = copy.deepcopy(data)
    overrides = {
        "quant.bits": case.bits,
        "quant.calibration.nsamples": 8,
        "quant.calibration.seqlen": 256,
        "export.model.model_name": case.model_name,
    }
    if case.family == "qwen3_5" and case.backend == "gptq":
        overrides.update(
            {
                "quant.validation.check_quant_vision_demo": False,
                "quant.validation.check_rotation_ppl": False,
            }
        )
    elif case.family == "qwen3_5":
        overrides.update({"quant.iters": 4, "quant.runtime.batch_size": 1})
    elif case.backend == "gptq":
        overrides.update(
            {
                "quant.validation.check_quant_text_demo": False,
                "quant.validation.check_quant_image_demo": False,
                "quant.validation.check_quant_video_demo": False,
                "quant.validation.check_quant_audio_demo": False,
            }
        )
    else:
        overrides.update({"quant.iters": 4, "quant.runtime.batch_size": 1})
    if case.family == "gemma4":
        overrides.update(
            {
                "export.model.context_max_length": 2048,
                "export.model.prefill_chunk_length": 320,
            }
        )
        if case.backend == "autoround":
            overrides["quant.autoround_repo"] = str(_gptqmodel_root() / "third_party" / "auto-round")
    for key, value in overrides.items():
        _set_dotted(data, key, value)
    return data


def _gemma_hf_config(case: ContractCase) -> dict[str, Any]:
    base: dict[str, Any] = {
        "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {"max_position_embeddings": 2048},
        "vision_config": {"hidden_size": 768},
    }
    text = base["text_config"]
    if "12B" in case.model:
        base["architectures"] = ["Gemma4UnifiedForConditionalGeneration"]
        base["model_type"] = "gemma4_unified"
        text["use_bidirectional_attention"] = "vision"
        base["vision_config"].update(
            {
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "num_soft_tokens": 280,
                "mm_posemb_size": 1120,
            }
        )
        base["audio_config"] = {"audio_embed_dim": 640}
    elif "26B-A4B" in case.model:
        text["enable_moe_block"] = True
    elif "E2B" in case.model:
        text.update({"hidden_size": 1536, "num_hidden_layers": 35, "num_kv_shared_layers": 1})
        base["audio_config"] = {"feature_size": 128}
    elif "E4B" in case.model:
        text.update({"hidden_size": 2048, "num_hidden_layers": 46, "num_kv_shared_layers": 1})
        base["audio_config"] = {"feature_size": 128}
    return base


def _gemma_processor_config(case: ContractCase) -> dict[str, Any] | None:
    if "12B" not in case.model:
        return None
    return {
        "image_processor": {"patch_size": 16, "pooling_kernel_size": 3, "max_soft_tokens": 280},
        "video_processor": {"max_soft_tokens": 70},
        "feature_extractor": {"feature_size": 640, "sampling_rate": 16000},
        "image_seq_length": 280,
        "audio_seq_length": 750,
    }


def _gptqmodel_root() -> Path:
    configured = os.environ.get("GPTQMODEL_SOURCE_DIR")
    if configured:
        root = Path(configured).expanduser().resolve()
        if not (root / "gptqmodel" / "recipes").is_dir():
            pytest.fail(f"invalid GPTQMODEL_SOURCE_DIR: {root}", pytrace=False)
        return root

    spec = importlib.util.find_spec("gptqmodel")
    package_locations = tuple(spec.submodule_search_locations or ()) if spec else ()
    if package_locations:
        root = Path(package_locations[0]).resolve().parent
        if (root / "gptqmodel" / "recipes").is_dir():
            return root

    pytest.fail(
        "GPTQModel recipe sources are unavailable; install GPTQModel in the CI environment",
        pytrace=False,
    )


def _load_source_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


@functools.lru_cache(maxsize=4)
def _recipe_module(recipe_name: str):
    path = _gptqmodel_root() / "gptqmodel" / "recipes" / f"{recipe_name}.py"
    if not path.is_file():
        pytest.fail(f"GPTQModel recipe source is missing: {path}", pytrace=False)
    return _load_source_module(f"_ci_gptqmodel_recipe_{recipe_name}", path)


def _require_callable(recipe_name: str, callable_name: str) -> Callable[..., Any]:
    try:
        recipe = getattr(_recipe_module(recipe_name), callable_name)
    except (ImportError, AttributeError) as exc:
        pytest.fail(f"GPTQModel recipe interface is unavailable: {recipe_name}.{callable_name}: {exc}", pytrace=False)
    assert callable(recipe)
    return recipe


@contextmanager
def _temporary_modelzoo_packages():
    package_paths = {
        "xhmodel_merak": REPO_ROOT / "xhmodel_merak",
        "xhmodel_merak.xh_llm": REPO_ROOT / "xhmodel_merak/xh_llm",
        "xhmodel_merak.xh_llm.workflows": REPO_ROOT / "xhmodel_merak/xh_llm/workflows",
        "xhmodel_merak.xh_llm.models": REPO_ROOT / "xhmodel_merak/xh_llm/models",
        "xhmodel_merak.xh_llm.models.qwen3_5": REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_5",
        "xhmodel_merak.xh_llm.models.gemma4_series": REPO_ROOT / "xhmodel_merak/xh_llm/models/gemma4_series",
    }
    names = tuple(package_paths) + (
        "xhmodel_merak.xh_llm.workflows.result",
        "xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter",
        "xhmodel_merak.xh_llm.models.gemma4_series.modality_contract",
        "xhmodel_merak.xh_llm.models.gemma4_series.variants",
        "xhmodel_merak.xh_llm.models.gemma4_series.export_plan",
        "xhmodel_merak.xh_llm.models.gemma4_series.quant_adapter",
    )
    missing = object()
    saved = {name: sys.modules.get(name, missing) for name in names}
    try:
        for name, path in package_paths.items():
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            package.__package__ = name
            sys.modules[name] = package
        yield
    finally:
        for name, module in saved.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@functools.lru_cache(maxsize=1)
def _modelzoo_adapters():
    with _temporary_modelzoo_packages():
        _load_source_module(
            "xhmodel_merak.xh_llm.workflows.result",
            REPO_ROOT / "xhmodel_merak/xh_llm/workflows/result.py",
        )
        qwen = _load_source_module(
            "xhmodel_merak.xh_llm.models.qwen3_5.quant_adapter",
            REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_5/quant_adapter.py",
        )
        for module_name in ("modality_contract", "variants", "export_plan"):
            _load_source_module(
                f"xhmodel_merak.xh_llm.models.gemma4_series.{module_name}",
                REPO_ROOT / f"xhmodel_merak/xh_llm/models/gemma4_series/{module_name}.py",
            )
        gemma = _load_source_module(
            "xhmodel_merak.xh_llm.models.gemma4_series.quant_adapter",
            REPO_ROOT / "xhmodel_merak/xh_llm/models/gemma4_series/quant_adapter.py",
        )
    return qwen, gemma


def _assert_signature_accepts(recipe: Callable[..., Any], kwargs: dict[str, Any]) -> None:
    try:
        inspect.signature(recipe).bind(**kwargs)
    except TypeError as exc:
        pytest.fail(
            f"ModelZoo/GPTQModel recipe contract drift for {recipe.__module__}.{recipe.__name__}: {exc}",
            pytrace=False,
        )


def _parser_flags(script_path: Path) -> set[str]:
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))
    flags: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        declared_flags = {
            arg.value
            for arg in node.args
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--")
        }
        flags.update(declared_flags)
        action = next((keyword.value for keyword in node.keywords if keyword.arg == "action"), None)
        if (
            isinstance(action, ast.Attribute)
            and action.attr == "BooleanOptionalAction"
        ) or (
            isinstance(action, ast.Name)
            and action.id == "BooleanOptionalAction"
        ):
            flags.update("--no-" + flag.removeprefix("--") for flag in declared_flags)
    return flags


def _assert_command_matches_parser(command: list[str]) -> None:
    script_path = Path(command[1])
    assert script_path.is_file(), f"recipe backend script does not exist: {script_path}"
    emitted = {item for item in command[2:] if item.startswith("--")}
    unknown = sorted(emitted - _parser_flags(script_path))
    assert unknown == [], f"recipe emits flags not accepted by {script_path}: {unknown}"


@pytest.mark.transformer516
@pytest.mark.parametrize("case", CASES, ids=lambda case: case.case_id)
def test_transformer516_recipe_interface_contract(case: ContractCase, tmp_path: Path) -> None:
    config = _resolved_config(case)
    quant_cfg = config["quant"]
    export_model_cfg = config["export"]["model"]
    model_dir = tmp_path / case.model
    model_dir.mkdir()
    if case.family == "gemma4":
        (model_dir / "config.json").write_text(
            json.dumps(_gemma_hf_config(case)),
            encoding="utf-8",
        )
        processor_config = _gemma_processor_config(case)
        if processor_config is not None:
            (model_dir / "processor_config.json").write_text(
                json.dumps(processor_config),
                encoding="utf-8",
            )

    if case.family == "qwen3_5":
        qwen_adapter, _ = _modelzoo_adapters()

        if case.backend == "gptq":
            kwargs = qwen_adapter.build_qwen35_gptqmodel_kwargs(
                hf_model_dir=str(model_dir),
                output_dir=str(tmp_path / "out"),
                device="cuda:0",
                quant_cfg=quant_cfg,
                export_model_cfg=export_model_cfg,
                workflow_seed=42,
            )
            recipe = _require_callable("qwen35", "quantize_qwen35")
        else:
            kwargs = qwen_adapter.build_qwen35_autoround_kwargs(
                hf_model_dir=str(model_dir),
                output_dir=str(tmp_path / "out"),
                device="cuda:0",
                quant_cfg=quant_cfg,
                export_model_cfg=export_model_cfg,
                workflow_seed=42,
            )
            recipe = _require_callable("qwen35_autoround", "quantize_qwen35_autoround")
        kwargs["dry_run"] = True
        _assert_signature_accepts(recipe, kwargs)
        result = recipe(**kwargs)
        _assert_command_matches_parser(result.provenance["command"])
        return

    _, gemma_adapter = _modelzoo_adapters()

    if case.backend == "gptq":
        kwargs = gemma_adapter.build_gptqmodel_recipe_kwargs(
            hf_model_dir=str(model_dir),
            output_dir=str(tmp_path / "out"),
            device="cuda:0",
            quant_cfg=quant_cfg,
            export_model_cfg=export_model_cfg,
            workflow_seed=42,
        )
        kwargs["dry_run"] = True
        recipe = _require_callable("gemma4", "quantize_gemma4")
        _assert_signature_accepts(recipe, kwargs)
        result = recipe(**kwargs)
        _assert_command_matches_parser(result.provenance["command"])
    else:
        command, quanted_model_dir, algorithm = gemma_adapter.build_autoround_mode1_command(
            hf_model_dir=str(model_dir),
            output_dir=str(tmp_path / "out"),
            device="cuda:0",
            quant_cfg=quant_cfg,
            export_model_cfg=export_model_cfg,
            workflow_seed=42,
        )
        assert Path(quanted_model_dir).is_absolute()
        assert algorithm.startswith("autoround:")
        _assert_command_matches_parser(command)


@pytest.mark.transformer516
def test_transformer516_public_workflow_surface_is_stable() -> None:
    workflow_files = (
        REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_5/workflow.py",
        REPO_ROOT / "xhmodel_merak/xh_llm/models/gemma4_series/workflow.py",
    )
    for path in workflow_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        methods = {
            node.name: {
                argument.arg
                for argument in (*node.args.args, *node.args.kwonlyargs)
            }
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in {"quant", "export"}
        }
        assert {"output_dir", "device", "config_overrides"} <= methods["quant"]
        assert {"quant_result", "output_dir", "device", "config_overrides"} <= methods["export"]


@pytest.mark.transformer516
def test_gemma_recipe_contract_drift_fails_instead_of_dropping_kwargs(monkeypatch) -> None:
    _, gemma_adapter = _modelzoo_adapters()

    def old_recipe(*, model_dir: str):
        return model_dir

    monkeypatch.setattr(gemma_adapter, "_load_recipe_callable", lambda entrypoint: old_recipe)
    with pytest.raises(TypeError, match="recipe contract drift"):
        gemma_adapter._call_gptqmodel_recipe(
            "gptqmodel.recipes.gemma4:quantize_gemma4",
            {"model_dir": "/model", "new_required_contract": True},
        )
