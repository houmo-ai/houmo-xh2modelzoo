"""Export immutable AutoRound block checkpoints in Transformers 5.16 CI."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTORY = os.environ.get(
    "TRANSFORMER516_CI_ARTIFACTORY",
    "http://10.10.1.53:8082/artifactory/model_zoo2",
).rstrip("/")
# These immutable object keys were published before the CI runtime rename.
# Keep them stable until a separately checksummed artifact set is uploaded.
PUBLISHED_REMOTE_DIR = "transformer55_ci_autoround_blocks_v1"
CACHE_ROOT = Path(
    os.environ.get(
        "TRANSFORMER516_CI_ARTIFACT_CACHE",
        REPO_ROOT / "work_dirs/transformer516_ci_autoround_block_cache",
    )
)
OUTPUT_ROOT = Path(
    os.environ.get(
        "TRANSFORMER516_CI_OUTPUT_ROOT",
        REPO_ROOT / "work_dirs/transformer516_ci_autoround_block_exports",
    )
)
DEVICE = os.environ.get("TRANSFORMER516_CI_DEVICE", "cuda")


@dataclass(frozen=True)
class BlockCase:
    case_id: str
    model: str
    family: str
    layers: int
    config: str
    model_name: str
    archive: str
    sha256: str

    @property
    def artifact_root(self) -> str:
        return self.archive.removesuffix(".tar.gz")

    @property
    def url(self) -> str:
        return f"{ARTIFACTORY}/{PUBLISHED_REMOTE_DIR}/{self.archive}"


QWEN35_9B = BlockCase(
    case_id="qwen3_5_9b_autoround_block",
    model="Qwen3.5-9B",
    family="qwen3_5",
    layers=4,
    config="configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml",
    model_name="qwen3_5_9b",
    archive="transformer55-ci-Qwen3.5-9B-layers4-autoround-v1.tar.gz",
    sha256="2f9e98fc82500ff76f4b4e72f235cef623ccf6bbd0d63e4b38037c5c4cd96169",
)

QWEN35_35B_A3B = BlockCase(
    case_id="qwen3_5_35b_a3b_autoround_block",
    model="Qwen3.5-35B-A3B",
    family="qwen3_5",
    layers=2,
    config=(
        "configs_merak/workflows/xh2a/llm_models/qwen3_5_moe/35b_a3b/"
        "qwen3_6_35b_a3b_full.yaml"
    ),
    model_name="qwen3_5_35b_a3b",
    archive="transformer55-ci-Qwen3.5-35B-A3B-layers2-autoround-v1.tar.gz",
    sha256="cb88e4a79b18b86b31cf661bc068edb7b48666d1691134b85395cbe4c03e44fe",
)

GEMMA4_E2B = BlockCase(
    case_id="gemma4_e2b_autoround_block",
    model="gemma-4-E2B-it",
    family="gemma4",
    layers=6,
    config=(
        "configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/"
        "gemma4_e2b_autoround.yaml"
    ),
    model_name="gemma_4_e2b",
    archive="transformer55-ci-gemma-4-E2B-it-layers6-autoround-v1.tar.gz",
    sha256="c2e2477be57a9a066eeebfbca08e4882e5882b24861fcffb444b75257cf5e803",
)

GEMMA4_26B_A4B = BlockCase(
    case_id="gemma4_26b_a4b_autoround_block",
    model="gemma-4-26B-A4B-it",
    family="gemma4",
    layers=2,
    config=(
        "configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/"
        "gemma4_26b_a4b_autoround.yaml"
    ),
    model_name="gemma_4_26b_a4b",
    archive="transformer55-ci-gemma-4-26B-A4B-it-layers2-autoround-v1.tar.gz",
    sha256="ea203730b90ad3028bb6b059b388fc6a74312f95931735e3d3c903fcf106ecd7",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(case: BlockCase) -> Path:
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    archive = CACHE_ROOT / case.archive
    if archive.is_file() and _sha256(archive) == case.sha256:
        return archive

    if archive.exists():
        archive.unlink()
    partial = archive.with_name(f"{archive.name}.part")
    command = [
        "wget",
        "--no-proxy",
        "--continue",
        "--progress=dot:giga",
        "--output-document",
        str(partial),
        case.url,
    ]
    print(f"[transformer516-ci] downloading {case.url}")
    subprocess.run(command, check=True)
    actual = _sha256(partial)
    if actual != case.sha256:
        partial.unlink(missing_ok=True)
        raise ValueError(
            f"Artifact checksum mismatch for {case.model}: "
            f"expected {case.sha256}, got {actual}"
        )
    partial.replace(archive)
    return archive


def _remove_owned_path(path: Path, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    if path.parent.resolve() != root.resolve():
        raise ValueError(f"Refusing to remove path outside owned root: {path}")
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def _extract(case: BlockCase, archive: Path) -> Path:
    model_dir = CACHE_ROOT / case.artifact_root
    marker = model_dir / ".ci_archive_sha256"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == case.sha256:
        _validate_checkpoint(case, model_dir)
        return model_dir

    _remove_owned_path(model_dir, CACHE_ROOT)
    temporary = Path(tempfile.mkdtemp(prefix=f"extract-{case.case_id}-", dir=CACHE_ROOT))
    try:
        with tarfile.open(archive, "r:gz") as payload:
            top_levels = {
                Path(member.name).parts[0]
                for member in payload.getmembers()
                if Path(member.name).parts
            }
            if top_levels != {case.artifact_root}:
                raise ValueError(
                    f"Unexpected archive roots for {case.model}: {sorted(top_levels)}"
                )
            payload.extractall(temporary, filter="data")
        extracted = temporary / case.artifact_root
        _validate_checkpoint(case, extracted)
        extracted.replace(model_dir)
        marker.write_text(f"{case.sha256}\n", encoding="utf-8")
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return model_dir


def _validate_checkpoint(case: BlockCase, model_dir: Path) -> None:
    required = (
        "config.json",
        "quantization_config.json",
        "tokenizer.json",
        "processor_config.json",
    )
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{case.model} block checkpoint is missing: {', '.join(missing)}"
        )
    if not any(model_dir.glob("*.safetensors")):
        raise FileNotFoundError(f"{case.model} block checkpoint has no safetensors payload")

    model_config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    text_config = model_config.get("text_config", model_config)
    actual_layers = text_config.get("num_hidden_layers")
    if actual_layers != case.layers:
        raise ValueError(
            f"{case.model} block layer mismatch: expected {case.layers}, got {actual_layers}"
        )

    quant_config = json.loads(
        (model_dir / "quantization_config.json").read_text(encoding="utf-8")
    )
    expected_quant = {"provider": "auto-round", "bits": 4, "group_size": 64}
    actual_quant = {key: quant_config.get(key) for key in expected_quant}
    if actual_quant != expected_quant:
        raise ValueError(
            f"{case.model} quantization contract mismatch: "
            f"expected {expected_quant}, got {actual_quant}"
        )


def _config_overrides(case: BlockCase) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "quant.bits": 4,
        "export.model.model_name": case.model_name,
    }
    if case.family == "gemma4":
        overrides.update(
            {
                "export.model.context_max_length": 2048,
                "export.model.prefill_chunk_length": 320,
            }
        )
    return overrides


def _assert_export_only(model_output: Path, case: BlockCase) -> None:
    for graph_kind in ("prefill", "decode"):
        graph_dir = model_output / graph_kind
        if not graph_dir.is_dir() or not any(graph_dir.glob("*.onnx")):
            raise AssertionError(f"{case.model} export did not produce a {graph_kind} HMONNX graph")

    golden_payload_dirs = sorted(
        path
        for path in model_output.rglob("*")
        if path.is_dir()
        and (
            path.name == "golden"
            or (path.name.startswith("step_") and path.name.removeprefix("step_").isdigit())
        )
    )
    if golden_payload_dirs:
        raise AssertionError(
            f"{case.model} export-only CI unexpectedly produced golden payloads: "
            + ", ".join(str(path) for path in golden_payload_dirs)
        )


def run_block_export(case: BlockCase) -> None:
    """Download one immutable quantized block, export it, and validate outputs."""
    archive = _download(case)
    model_dir = _extract(case, archive)

    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow
    from xhmodel_merak.xh_llm.workflows.result import QuantResult

    workflow = AutoLLMWorkflow.from_config(
        model_dir=str(model_dir),
        config_path=str(REPO_ROOT / case.config),
    )
    quant_result = QuantResult(
        raw_model_dir=str(model_dir),
        quanted_model_dir=str(model_dir),
    )

    output_dir = OUTPUT_ROOT / case.case_id
    _remove_owned_path(output_dir, OUTPUT_ROOT)
    print(f"[transformer516-ci] exporting {case.model} block to {output_dir}")
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=str(output_dir),
        device=DEVICE,
        config_overrides=_config_overrides(case),
    )
    if export_result is None:
        raise AssertionError(f"{case.model} export returned None")
    model_outputs = [path for path in output_dir.glob("hmquant*") if path.is_dir()]
    if len(model_outputs) != 1:
        raise AssertionError(f"{case.model} export produced {len(model_outputs)} model directories")
    _assert_export_only(model_outputs[0], case)
