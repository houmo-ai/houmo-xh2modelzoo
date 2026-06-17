"""Runtime validation matrix for the Qwen3.5/Qwen3.6 Merak workflow.

This script intentionally keeps the public workflow API small.  Each scenario
selects a YAML plus explicit config overrides, then calls only:

    workflow.quant(output_dir, device, config_overrides=...)
    workflow.export(quant_result, output_dir, device, config_overrides=...)

Default matrix covers the requested first-pass runtime validation scope:
Qwen3.5-9B and Qwen3.6-35B-A3B, base and externally quantized HF artifacts,
with ``fuse_gdr_ops`` false and true.

Additional spec-decode scenarios cover MTP/DFlash with externally quantized
HF artifacts only.  These keep fuse_gdr_ops at the YAML default because the
runtime question is whether the quantized topology exports successfully.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


QWEN35_9B_HF_MODEL_DIR = "weights/Qwen3.5-9B"
QWEN35_9B_CONFIG = "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml"
QWEN35_9B_MTP_CONFIG = "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_mtp.yaml"
QWEN35_9B_DFLASH_CONFIG = "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_dflash.yaml"
QWEN35_9B_EXISTING_HF_QUANT = "weights/Qwen3.5-9B-mode1-llm-only"

QWEN36_35B_A3B_HF_MODEL_DIR = "weights/Qwen3.6-35B-A3B"
QWEN36_35B_A3B_CONFIG = "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml"
QWEN36_35B_A3B_MTP_CONFIG = "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_mtp.yaml"
QWEN36_35B_A3B_DFLASH_CONFIG = (
    "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_dflash.yaml"
)
QWEN36_35B_A3B_EXISTING_HF_QUANT = "weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400"

DEFAULT_PROMPT = "用中文一句话说明本次导出验证是否完成。"
REQUIRED_MODULES = ("transformers", "auto_round", "gptqmodel", "datasets", "torch", "xhquant")
REQUIRED_MIN_VERSIONS = {"transformers": (5, 5, 0)}


def existing_hf_quant_override(existing_hf_model_dir: str) -> dict[str, Any]:
    return {
        "quant": {
            "algorithm": "existing_hf",
            "artifact_format": "gptqmodel_hf",
            "source_algorithm": "autoround",
            "existing_hf_model_dir": existing_hf_model_dir,
        }
    }


@dataclass(frozen=True)
class ValidationScenario:
    name: str
    hf_model_dir: str
    config_path: str
    quant_overrides: dict[str, Any]
    fuse_gdr_ops: bool

    @property
    def config_overrides(self) -> dict[str, Any]:
        overrides = dict(self.quant_overrides)
        overrides["export.model.fuse_gdr_ops"] = self.fuse_gdr_ops
        return overrides


VALIDATION_SCENARIOS: tuple[ValidationScenario, ...] = (
    ValidationScenario(
        name="qwen35_9b_base_fuse_false",
        hf_model_dir=QWEN35_9B_HF_MODEL_DIR,
        config_path=QWEN35_9B_CONFIG,
        quant_overrides={"quant": None},
        fuse_gdr_ops=False,
    ),
    ValidationScenario(
        name="qwen35_9b_base_fuse_true",
        hf_model_dir=QWEN35_9B_HF_MODEL_DIR,
        config_path=QWEN35_9B_CONFIG,
        quant_overrides={"quant": None},
        fuse_gdr_ops=True,
    ),
    ValidationScenario(
        name="qwen35_9b_existing_hf_fuse_false",
        hf_model_dir=QWEN35_9B_HF_MODEL_DIR,
        config_path=QWEN35_9B_CONFIG,
        quant_overrides=existing_hf_quant_override(QWEN35_9B_EXISTING_HF_QUANT),
        fuse_gdr_ops=False,
    ),
    ValidationScenario(
        name="qwen35_9b_existing_hf_fuse_true",
        hf_model_dir=QWEN35_9B_HF_MODEL_DIR,
        config_path=QWEN35_9B_CONFIG,
        quant_overrides=existing_hf_quant_override(QWEN35_9B_EXISTING_HF_QUANT),
        fuse_gdr_ops=True,
    ),
    ValidationScenario(
        name="qwen35_9b_mtp_existing_hf",
        hf_model_dir=QWEN35_9B_HF_MODEL_DIR,
        config_path=QWEN35_9B_MTP_CONFIG,
        quant_overrides=existing_hf_quant_override(QWEN35_9B_EXISTING_HF_QUANT),
        fuse_gdr_ops=False,
    ),
    ValidationScenario(
        name="qwen35_9b_dflash_existing_hf",
        hf_model_dir=QWEN35_9B_HF_MODEL_DIR,
        config_path=QWEN35_9B_DFLASH_CONFIG,
        quant_overrides=existing_hf_quant_override(QWEN35_9B_EXISTING_HF_QUANT),
        fuse_gdr_ops=False,
    ),
    ValidationScenario(
        name="qwen36_35b_a3b_base_fuse_false",
        hf_model_dir=QWEN36_35B_A3B_HF_MODEL_DIR,
        config_path=QWEN36_35B_A3B_CONFIG,
        quant_overrides={"quant": None},
        fuse_gdr_ops=False,
    ),
    ValidationScenario(
        name="qwen36_35b_a3b_base_fuse_true",
        hf_model_dir=QWEN36_35B_A3B_HF_MODEL_DIR,
        config_path=QWEN36_35B_A3B_CONFIG,
        quant_overrides={"quant": None},
        fuse_gdr_ops=True,
    ),
    ValidationScenario(
        name="qwen36_35b_a3b_existing_hf_fuse_false",
        hf_model_dir=QWEN36_35B_A3B_HF_MODEL_DIR,
        config_path=QWEN36_35B_A3B_CONFIG,
        quant_overrides=existing_hf_quant_override(QWEN36_35B_A3B_EXISTING_HF_QUANT),
        fuse_gdr_ops=False,
    ),
    ValidationScenario(
        name="qwen36_35b_a3b_existing_hf_fuse_true",
        hf_model_dir=QWEN36_35B_A3B_HF_MODEL_DIR,
        config_path=QWEN36_35B_A3B_CONFIG,
        quant_overrides=existing_hf_quant_override(QWEN36_35B_A3B_EXISTING_HF_QUANT),
        fuse_gdr_ops=True,
    ),
    ValidationScenario(
        name="qwen36_35b_a3b_mtp_existing_hf",
        hf_model_dir=QWEN36_35B_A3B_HF_MODEL_DIR,
        config_path=QWEN36_35B_A3B_MTP_CONFIG,
        quant_overrides=existing_hf_quant_override(QWEN36_35B_A3B_EXISTING_HF_QUANT),
        fuse_gdr_ops=False,
    ),
    ValidationScenario(
        name="qwen36_35b_a3b_dflash_existing_hf",
        hf_model_dir=QWEN36_35B_A3B_HF_MODEL_DIR,
        config_path=QWEN36_35B_A3B_DFLASH_CONFIG,
        quant_overrides=existing_hf_quant_override(QWEN36_35B_A3B_EXISTING_HF_QUANT),
        fuse_gdr_ops=False,
    ),
)


def scenario_names() -> tuple[str, ...]:
    return tuple(scenario.name for scenario in VALIDATION_SCENARIOS)


def select_scenarios(selected: list[str] | None) -> list[ValidationScenario]:
    if not selected:
        return list(VALIDATION_SCENARIOS)
    by_name = {scenario.name: scenario for scenario in VALIDATION_SCENARIOS}
    return [by_name[name] for name in selected]


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in version.split("."):
        digits = ""
        for char in part:
            if not char.isdigit():
                break
            digits += char
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


def preflight_scenarios(scenarios: list[ValidationScenario], *, check_modules: bool = True) -> list[str]:
    """Return human-readable issues that would prevent runtime validation."""
    issues: list[str] = []
    if check_modules:
        for module_name in REQUIRED_MODULES:
            try:
                importlib.import_module(module_name)
            except Exception as exc:  # pragma: no cover - depends on local runtime env
                issues.append(f"missing module {module_name}: {type(exc).__name__}: {exc}")
                continue
            min_version = REQUIRED_MIN_VERSIONS.get(module_name)
            if min_version is None:
                continue
            try:
                installed_version = importlib.metadata.version(module_name)
            except importlib.metadata.PackageNotFoundError:  # pragma: no cover - import succeeded, metadata unusual
                issues.append(f"cannot determine {module_name} package version")
                continue
            if _version_tuple(installed_version) < min_version:
                required = ".".join(str(part) for part in min_version)
                issues.append(f"{module_name}>={required} required, found {installed_version}")

    for scenario in scenarios:
        for label, path in (
            ("hf_model_dir", scenario.hf_model_dir),
            ("config_path", scenario.config_path),
        ):
            if not Path(path).exists():
                issues.append(f"{scenario.name}: {label} does not exist: {path}")
        quant_cfg = scenario.config_overrides.get("quant")
        if isinstance(quant_cfg, dict):
            existing_hf_model_dir = quant_cfg.get("existing_hf_model_dir")
            if existing_hf_model_dir and not Path(existing_hf_model_dir).exists():
                issues.append(f"{scenario.name}: quant.existing_hf_model_dir does not exist: {existing_hf_model_dir}")
    return issues


def _remove_output_dir_if_needed(output_dir: Path, force: bool) -> None:
    if force and output_dir.exists():
        shutil.rmtree(output_dir)


def run_scenario(
    scenario: ValidationScenario,
    *,
    output_root: Path,
    device: str,
    seed: int,
    debug: bool,
    force: bool,
    dump_golden: bool,
    skip_export: bool,
    quick_test: bool,
    quick_test_max_new_tokens: int,
    prompt: str,
) -> None:
    from xhmodel_merak.xh_llm.models.qwen3_5 import (
        Qwen35Workflow,
        find_hmonnx_meta_file,
        print_quick_test_result,
        quick_test_hmonnx,
    )

    scenario_root = output_root / scenario.name
    quant_output_dir = scenario_root / "quant"
    export_output_dir = scenario_root / "export"
    if skip_export:
        if force:
            raise ValueError("--skip-export cannot be combined with --force")
        if not export_output_dir.is_dir():
            raise FileNotFoundError(f"{scenario.name}: --skip-export requires existing export dir: {export_output_dir}")
        scenario_root.mkdir(parents=True, exist_ok=True)
        print(f"[{scenario.name}] skip_export=True reuse_export_dir={export_output_dir}")
        meta_file = find_hmonnx_meta_file(export_output_dir)
    else:
        _remove_output_dir_if_needed(scenario_root, force)
        scenario_root.mkdir(parents=True, exist_ok=True)

        workflow = Qwen35Workflow.from_config(
            hf_model_dir=scenario.hf_model_dir,
            config_path=scenario.config_path,
            seed=seed,
            debug=debug,
        )
        quant_result = workflow.quant(
            output_dir=str(quant_output_dir),
            device=device,
            config_overrides=scenario.config_overrides,
        )
        export_result = workflow.export(
            quant_result=quant_result,
            output_dir=str(export_output_dir),
            device=device,
            config_overrides=scenario.config_overrides,
        )
        if dump_golden:
            workflow.dump_golden(export_result=export_result, device=device, input_messages={"text": prompt})

        print(f"[{scenario.name}] quant_result={quant_result}")
        print(f"[{scenario.name}] export_result={export_result}")
        meta_file = find_hmonnx_meta_file(export_result)
    print(f"[{scenario.name}] hmonnx_meta_file={meta_file}")
    if quick_test:
        quick_result = quick_test_hmonnx(
            meta_file,
            prompt=prompt,
            device=device,
            max_new_tokens=quick_test_max_new_tokens,
            do_sample=False,
        )
        print_quick_test_result(quick_result)
        result_file = scenario_root / "quick_test_result.json"
        result_file.write_text(
            json.dumps(quick_result.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[{scenario.name}] quick_test_result={result_file}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Qwen3.5/Qwen3.6 Merak workflow validation matrix")
    parser.add_argument("--scenario", action="append", choices=scenario_names(), help="Scenario to run; repeatable")
    parser.add_argument("--output-root", default="work_dirs/qwen3_5_validation_matrix")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--force", action="store_true", help="Remove previous scenario output before running")
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Reuse existing <output-root>/<scenario>/export and only run meta lookup or quick test",
    )
    parser.add_argument("--skip-golden", action="store_true", help="Skip dump_golden after export")
    parser.add_argument("--quick-test", action="store_true", help="Run a quick HMONNX generate/spec-decode test")
    parser.add_argument("--quick-test-max-new-tokens", type=int, default=64)
    parser.add_argument("--preflight-only", action="store_true", help="Check imports and paths, then exit")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    scenarios = select_scenarios(args.scenario)
    issues = preflight_scenarios(scenarios)
    if issues:
        formatted = "\n".join(f"- {issue}" for issue in issues)
        raise SystemExit(f"Qwen3.5 validation matrix preflight failed:\n{formatted}")
    if args.preflight_only:
        print("Qwen3.5 validation matrix preflight ok")
        return

    output_root = Path(args.output_root)
    for scenario in scenarios:
        run_scenario(
            scenario,
            output_root=output_root,
            device=args.device,
            seed=args.seed,
            debug=args.debug,
            force=args.force,
            dump_golden=not args.skip_golden,
            skip_export=args.skip_export,
            quick_test=args.quick_test,
            quick_test_max_new_tokens=args.quick_test_max_new_tokens,
            prompt=args.prompt,
        )


if __name__ == "__main__":
    main()
