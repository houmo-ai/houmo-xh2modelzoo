"""Lightweight (no-GPU) tests for QTL-312 MTP head pruning pipeline.

Verifies the export / benchmark CLI surface, helper presence, and that the
artifactory dataset hand-off is wired into the user guide.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = REPO_ROOT / "examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py"
BENCH_SCRIPT = REPO_ROOT / "examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py"
BUILD_HOT_VOCAB = REPO_ROOT / "examples/llm/qwen3_5/build_mtp_hot_vocab.py"
MODEL_UTILS = REPO_ROOT / "examples/llm/qwen3_5/utils/model_utils.py"
MTP_MODEL = REPO_ROOT / "xh_model_zoo/xh_llm/models/qwen3_5/_mtp_model.py"
PRUNING_DOC = REPO_ROOT / "examples/llm/qwen3_5/MTP_HEAD_PRUNING.md"
BENCH_DOC = REPO_ROOT / "examples/llm/qwen3_5/MTP_BENCHMARK_GUIDE.md"
PACKAGE_SH = REPO_ROOT / "tools/package_mtp_dataset.sh"

ARTIFACTORY_URL = (
    "http://10.10.1.53:8081/artifactory/model_zoo2/"
    "qwen3_5_mtp_head_pruning/qwen3_5_mtp_head_pruning_dataset_v1.tar.gz"
)
ARTIFACTORY_CORPUS_URL = (
    "http://10.10.1.53:8081/artifactory/model_zoo2/"
    "qwen3_5_mtp_head_pruning/qwen3_5_mtp_head_pruning_corpus_v1.tar.gz"
)
CORPUS_SHA256 = "cd07ea6ccbbb125f86f63cbbf851133b03cb9a4cf381263501c03991d8bb5479"


def _argparse_options(path: Path) -> dict[str, dict]:
    """Return a {option_string: kwargs} map for argparse add_argument calls."""
    tree = ast.parse(path.read_text())
    found: dict[str, dict] = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
            continue
        if node.func.attr != "add_argument":
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        opt = node.args[0].value
        if not isinstance(opt, str) or not opt.startswith("--"):
            continue
        kwargs: dict = {}
        for kw in node.keywords:
            if kw.arg in ("default", "choices", "type", "required"):
                try:
                    kwargs[kw.arg] = ast.literal_eval(kw.value)
                except Exception:  # pragma: no cover - non-literal default
                    kwargs[kw.arg] = "<expr>"
        found[opt] = kwargs
    return found


def test_export_script_exposes_pruning_cli():
    opts = _argparse_options(EXPORT_SCRIPT)
    assert "--mtp-head-k" in opts, "missing --mtp-head-k flag"
    assert "--reranked-repo-dir" in opts, "missing --reranked-repo-dir flag"
    assert "--force-rerank" in opts, "missing --force-rerank flag"


def test_export_script_has_prepare_reranked_repo():
    tree = ast.parse(EXPORT_SCRIPT.read_text())
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "prepare_reranked_repo" in funcs


def test_model_utils_exposes_rerank_helper():
    tree = ast.parse(MODEL_UTILS.read_text())
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "rerank_model_for_mtp" in funcs


def test_benchmark_script_auto_detects_mtp_lm_head():
    text = BENCH_SCRIPT.read_text()
    assert "--mtp-head-pt" in text
    assert 'mtp_lm_head.pt' in text
    opts = _argparse_options(BENCH_SCRIPT)
    assert "--mtp-head-pt" in opts


def test_mtp_model_auto_detects_trimmed_lm_head():
    text = MTP_MODEL.read_text()
    assert "mtp_lm_head.pt" in text, "mtp model must auto-detect reranked head"


def test_build_hot_vocab_script_is_executable():
    # syntactic sanity + must declare its key CLI surface
    ast.parse(BUILD_HOT_VOCAB.read_text())
    opts = _argparse_options(BUILD_HOT_VOCAB)
    # Pipeline emits hot_ids/id_map under a selection dir per K
    assert "--output-dir" in opts or "--selection-dir" in opts


def test_pruning_doc_advertises_dataset_url():
    text = PRUNING_DOC.read_text()
    assert ARTIFACTORY_URL in text, "user guide must surface the dataset artifactory URL"
    assert ARTIFACTORY_CORPUS_URL in text, "user guide must surface the corpus artifactory URL"
    assert CORPUS_SHA256 in text, "user guide must surface the corpus sha256"
    assert "tools/package_mtp_dataset.sh" in text


def test_pruning_doc_describes_full_e2e_flow():
    text = PRUNING_DOC.read_text()
    # Each stage of corpus -> freqs -> selection -> reranked repo -> hmonnx
    # must be discoverable in the guide.
    for marker in (
        "build_mtp_hot_vocab.py",
        "rerank_model_for_mtp",
        "selection/hot_ids_",
        "mtp_lm_head.pt",
        "--mtp-head-k",
    ):
        assert marker in text, f"missing pipeline marker in MTP_HEAD_PRUNING.md: {marker}"


def test_benchmark_guide_present():
    assert BENCH_DOC.is_file()
    assert BENCH_DOC.stat().st_size > 0


def test_package_script_shell_syntax():
    assert PACKAGE_SH.is_file()
    res = subprocess.run(
        ["bash", "-n", str(PACKAGE_SH)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert res.returncode == 0, f"bash -n failed: {res.stderr}"
    text = PACKAGE_SH.read_text()
    assert "qwen3_5_mtp_head_pruning_dataset_v1.tar.gz" in text
    assert "qwen3_5_mtp_head_pruning_corpus_v1.tar.gz" in text
    assert "--corpus" in text, "package script must expose --corpus flag"
    assert "freqs" in text and "selection" in text
