"""Structural smoke tests for the qwen3.5 dflash export pipeline (TLC-4404).

No GPU / no weights / no network. These guard the surgical removal of
QTL-328 (head 4-bit quant) and QTL-312 (mtp_head_k reranked vocab) so
later refactors don't accidentally reintroduce the dropped surface.

The ``examples/llm/qwen3_5`` directory is a script tree without
``__init__.py`` files, so we inspect the scripts by AST rather than via
``importlib``. The MoE converter is a real package and is imported normally.
"""
from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORT_SCRIPT = REPO_ROOT / "examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py"
BENCH_SCRIPT = REPO_ROOT / "examples/llm/qwen3_5/qwen3_5_mtp_benchmark.py"

_REMOVED_EXPORT_FLAGS = (
    "--spec-draft-head-weight-bits",
    "--mtp-head-k",
    "--reranked-repo-dir",
    "--force-rerank",
)


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _find_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def test_qwen3_5_export_script_parses():
    assert EXPORT_SCRIPT.is_file()
    _parse(EXPORT_SCRIPT)


def test_qwen3_5_export_script_preserves_dflash_drops_removed_flags():
    src = EXPORT_SCRIPT.read_text()
    assert "--spec_decode_mode" in src
    assert "--dflash_model_dir" in src
    for flag in _REMOVED_EXPORT_FLAGS:
        assert flag not in src, f"removed flag resurfaced: {flag}"
    for symbol in ("_build_spec_draft_quant_config", "DRAFT_BASE_QUANT_TYPE",
                   "prepare_reranked_repo"):
        assert symbol not in src, f"removed symbol resurfaced: {symbol}"


def test_build_draft_only_default_work_dir_is_two_args():
    tree = _parse(EXPORT_SCRIPT)
    fn = _find_function(tree, "_build_draft_only_default_work_dir")
    arg_names = [a.arg for a in fn.args.args]
    assert len(arg_names) == 2, f"expected 2 params, got {arg_names}"


def test_qwen3_5_moe_converter_imports_cleanly():
    importlib.import_module(
        "xh_model_zoo.xh_llm.models.qwen3_5_moe.qwen3_5_moe_converter"
    )


def test_qwen3_5_moe_converter_drops_spec_draft_quant_helper():
    mod = importlib.import_module(
        "xh_model_zoo.xh_llm.models.qwen3_5_moe.qwen3_5_moe_converter"
    )
    assert not hasattr(mod, "_build_spec_draft_quant_config")
    assert not hasattr(mod, "DRAFT_BASE_QUANT_TYPE")
    src = inspect.getsource(mod)
    assert "spec_draft_head_weight_bits" not in src


def test_qwen3_5_mtp_benchmark_drops_mtp_head_pt_flag():
    assert BENCH_SCRIPT.is_file()
    src = BENCH_SCRIPT.read_text()
    assert "--mtp-head-pt" not in src
    assert "mtp_lm_head.pt" not in src
    tree = _parse(BENCH_SCRIPT)
    fn = _find_function(tree, "build_mtp_head")
    arg_names = [a.arg for a in fn.args.args]
    assert "mtp_head_pt" not in arg_names
