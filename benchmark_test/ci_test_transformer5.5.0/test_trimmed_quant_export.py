"""Transformer 5.5.0 CI: wget trimmed model, run quant, export HMONNX."""
from __future__ import annotations

import os
import subprocess
import tarfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTORY = "http://10.10.1.53:8082/artifactory/model_zoo2"
MODEL_REMOTE_DIR = "transformer55_ci_trimmed_simple_v2"
AUTOROUND_JSONL = "NeelNanda-pile-10k.jsonl"

CACHE_ROOT = Path(os.environ.get("TRANSFORMER55_CI_ARTIFACT_CACHE", REPO_ROOT / "work_dirs/transformer55_ci_cache"))
OUTPUT_ROOT = Path(os.environ.get("TRANSFORMER55_CI_OUTPUT_ROOT", REPO_ROOT / "work_dirs/transformer55_ci_outputs"))
DEVICE = os.environ.get("TRANSFORMER55_CI_DEVICE", "cuda")


@dataclass(frozen=True)
class Case:
    model: str
    family: str
    layers: int
    bits: int
    config: str
    overrides: dict[str, Any]
    backend: str = "gptq"

    @property
    def case_id(self) -> str:
        return f"{self.model}-{self.backend}"

    @property
    def archive(self) -> str:
        return f"transformer55-ci-{self.model}-layers{self.layers}.tar.gz"

    @property
    def url(self) -> str:
        return f"{ARTIFACTORY}/{MODEL_REMOTE_DIR}/{self.archive}"


def _base_overrides(bits: int, model_name: str) -> dict[str, Any]:
    return {
        "quant.bits": bits,
        "quant.calibration.nsamples": 8,
        "export.model.model_name": model_name,
    }


def _qwen_gptq(bits: int, model_name: str) -> dict[str, Any]:
    overrides = _base_overrides(bits, model_name)
    overrides.update(
        {
            "quant.calibration.seqlen": 256,
            "quant.validation.check_quant_vision_demo": False,
            "quant.validation.check_rotation_ppl": False,
        }
    )
    return overrides


def _qwen_autoround(bits: int, model_name: str) -> dict[str, Any]:
    overrides = _base_overrides(bits, model_name)
    overrides.update(
        {
            "quant.calibration.seqlen": 256,
            "quant.iters": 4,
            "quant.runtime.batch_size": 1,
        }
    )
    return overrides


def _gemma_gptq(model_name: str) -> dict[str, Any]:
    overrides = _base_overrides(4, model_name)
    overrides.update(
        {
            "quant.calibration.seqlen": 256,
            "quant.validation.check_quant_text_demo": False,
            "quant.validation.check_quant_image_demo": False,
            "quant.validation.check_quant_video_demo": False,
            "quant.validation.check_quant_audio_demo": False,
            "export.model.context_max_length": 2048,
            "export.model.prefill_chunk_length": 320,
        }
    )
    return overrides


def _gemma_autoround(model_name: str) -> dict[str, Any]:
    overrides = _base_overrides(4, model_name)
    overrides.update(
        {
            "quant.calibration.seqlen": 256,
            "quant.iters": 4,
            "quant.runtime.batch_size": 1,
            "export.model.context_max_length": 2048,
            "export.model.prefill_chunk_length": 320,
        }
    )
    return overrides


QWEN_DENSE_GPTQ_9B = "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full_gptq.yaml"
QWEN_DENSE_GPTQ_27B = "configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_6_27b_full_gptq.yaml"
QWEN_MOE_GPTQ_35B = "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full_gptq.yaml"
QWEN_DENSE_AR_9B = "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml"
QWEN_DENSE_AR_27B = "configs_merak/workflows/xh2a/llm_models/qwen3_5/27b/qwen3_6_27b_full.yaml"
QWEN_MOE_AR_35B = "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/qwen3_6_35b_a3b_full.yaml"
GEMMA_E2B_GPTQ = "configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml"
GEMMA_E2B_AR = "configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_autoround.yaml"
GEMMA_E4B_GPTQ = "configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml"
GEMMA_E4B_AR = "configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_autoround.yaml"
GEMMA_31B_GPTQ = "configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_full.yaml"
GEMMA_31B_AR = "configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_autoround.yaml"
GEMMA_26B_GPTQ = "configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml"
GEMMA_26B_AR = "configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_autoround.yaml"


MODELS: tuple[tuple[str, str, int, int, str, str, str], ...] = (
    ("Qwen3.5-0.8B", "qwen3_5", 4, 8, QWEN_DENSE_GPTQ_9B, QWEN_DENSE_AR_9B, "qwen3_5_0_8b"),
    ("Qwen3.5-2B", "qwen3_5", 4, 8, QWEN_DENSE_GPTQ_9B, QWEN_DENSE_AR_9B, "qwen3_5_2b"),
    ("Qwen3.5-4B", "qwen3_5", 4, 4, QWEN_DENSE_GPTQ_9B, QWEN_DENSE_AR_9B, "qwen3_5_4b"),
    ("Qwen3.5-9B", "qwen3_5", 4, 4, QWEN_DENSE_GPTQ_9B, QWEN_DENSE_AR_9B, "qwen3_5_9b"),
    ("Qwen3.5-27B", "qwen3_5", 4, 4, QWEN_DENSE_GPTQ_27B, QWEN_DENSE_AR_27B, "qwen3_5_27b"),
    ("Qwen3.5-35B-A3B", "qwen3_5", 2, 4, QWEN_MOE_GPTQ_35B, QWEN_MOE_AR_35B, "qwen3_5_35b_a3b"),
    ("gemma-4-E2B-it", "gemma4", 6, 4, GEMMA_E2B_GPTQ, GEMMA_E2B_AR, "gemma_4_e2b"),
    ("gemma-4-E4B-it", "gemma4", 6, 4, GEMMA_E4B_GPTQ, GEMMA_E4B_AR, "gemma_4_e4b"),
    ("gemma-4-31B-it", "gemma4", 6, 4, GEMMA_31B_GPTQ, GEMMA_31B_AR, "gemma_4_31b"),
    ("gemma-4-26B-A4B-it", "gemma4", 2, 4, GEMMA_26B_GPTQ, GEMMA_26B_AR, "gemma_4_26b_a4b"),
)


def _cases() -> list[Case]:
    cases: list[Case] = []
    for model, family, layers, bits, gptq_config, ar_config, model_name in MODELS:
        if family == "qwen3_5":
            cases.append(Case(model, family, layers, bits, gptq_config, _qwen_gptq(bits, model_name)))
            cases.append(
                Case(model, family, layers, bits, ar_config, _qwen_autoround(bits, model_name), "autoround")
            )
        else:
            cases.append(Case(model, family, layers, bits, gptq_config, _gemma_gptq(model_name)))
            cases.append(Case(model, family, layers, bits, ar_config, _gemma_autoround(model_name), "autoround"))

    selected = {item.strip() for item in os.environ.get("TRANSFORMER55_CI_CASES", "").split(",") if item.strip()}
    if selected:
        cases = [case for case in cases if case.case_id in selected or case.model in selected]
    return cases


def _wget(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    subprocess.run(["wget", "-q", "-O", str(dest), url], check=True)


def _extract_archive(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    top_levels: set[str] = set()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            parts = Path(member.name).parts
            if parts:
                top_levels.add(parts[0])
        tar.extractall(dest, filter="data")
    if len(top_levels) == 1:
        return dest / next(iter(top_levels))
    return dest


def _model_dir(case: Case) -> Path:
    dest = CACHE_ROOT / case.model
    if (dest / "config.json").is_file():
        return dest
    archive = CACHE_ROOT / case.archive
    _wget(case.url, archive)
    extracted = _extract_archive(archive, CACHE_ROOT / f"extract-{case.model}")
    if not (extracted / "config.json").is_file():
        raise FileNotFoundError(f"Extracted model config not found under {extracted}")
    return extracted


def _ensure_autoround_dataset() -> None:
    dataset = REPO_ROOT / "data" / "calib_data" / AUTOROUND_JSONL
    if not dataset.is_file():
        raise FileNotFoundError(
            f"AutoRound calibration data is required at {dataset}. "
            "The CI lane keeps this JSONL in git and must not download it."
        )


def _qwen_quant(
    *,
    model_dir: str,
    config_path: str,
    output_dir: str,
    device: str,
    config_overrides: dict[str, Any],
):
    workflow = _qwen_workflow(model_dir=model_dir, config_path=config_path)
    return workflow.quant(
        output_dir=output_dir,
        device=device,
        config_overrides=config_overrides,
    )


def _qwen_export(
    *,
    model_dir: str,
    config_path: str,
    quant_result,
    output_dir: str,
    device: str,
    config_overrides: dict[str, Any],
):
    workflow = _qwen_workflow(model_dir=model_dir, config_path=config_path)
    return workflow.export(
        quant_result=quant_result,
        output_dir=output_dir,
        device=device,
        config_overrides=config_overrides,
    )


def _qwen_workflow(*, model_dir: str, config_path: str):
    """Use the same public workflow entrypoint as the Qwen3.5 example."""
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    return AutoLLMWorkflow.from_config(
        model_dir=model_dir,
        config_path=config_path,
    )


def _workflow(case: Case):
    if case.family == "gemma4":
        from xhmodel_merak.xh_llm.models.gemma4_series.workflow import export, quant

        return quant, export
    return _qwen_quant, _qwen_export


def _format_subprocess_error(exc: subprocess.CalledProcessError) -> str:
    parts = [
        f"returncode: {exc.returncode}",
        "command:",
        " ".join(str(item) for item in exc.cmd) if isinstance(exc.cmd, (list, tuple)) else str(exc.cmd),
    ]
    if exc.stdout:
        parts.extend(["stdout:", str(exc.stdout)])
    if exc.stderr:
        parts.extend(["stderr:", str(exc.stderr)])
    return "\n".join(parts)


def _fail_with_case_context(case: Case, stage: str, exc: Exception) -> None:
    details = [
        f"CI case failed: {case.case_id}",
        f"stage: {stage}",
        f"model: {case.model}",
        f"backend: {case.backend}",
        f"config: {REPO_ROOT / case.config}",
        f"output_dir: {OUTPUT_ROOT / case.case_id}",
    ]
    if isinstance(exc, subprocess.CalledProcessError):
        details.extend(["subprocess error:", _format_subprocess_error(exc)])
    details.extend(["traceback:", traceback.format_exc()])
    pytest.fail("\n".join(details), pytrace=False)


@pytest.mark.parametrize("case", _cases(), ids=lambda case: case.case_id)
def test_trimmed_model_quant_to_export(case: Case) -> None:
    stage = "prepare"
    try:
        print(f"[ci-case] start {case.case_id}")
        print(f"[ci-case] config={REPO_ROOT / case.config}")
        print(f"[ci-case] output={OUTPUT_ROOT / case.case_id}")
        if case.backend == "autoround":
            stage = "check AutoRound calibration dataset"
            _ensure_autoround_dataset()

        stage = "download/extract trimmed model"
        model_dir = _model_dir(case)
        print(f"[ci-case] model_dir={model_dir}")
        quant, export = _workflow(case)
        case_output = OUTPUT_ROOT / case.case_id

        stage = "quant"
        quant_result = quant(
            model_dir=str(model_dir),
            config_path=str(REPO_ROOT / case.config),
            output_dir=str(case_output / "quant"),
            device=DEVICE,
            config_overrides=case.overrides,
        )

        stage = "export"
        export_result = export(
            model_dir=str(model_dir),
            config_path=str(REPO_ROOT / case.config),
            quant_result=quant_result,
            output_dir=str(case_output / "export"),
            device=DEVICE,
            config_overrides=case.overrides,
        )
        assert export_result is not None
    except Exception as exc:
        _fail_with_case_context(case, stage, exc)
