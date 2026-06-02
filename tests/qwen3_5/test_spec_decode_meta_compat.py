"""Smoke tests for spec decode meta format compatibility (Sprint 2.3).

Verifies that the merak export produces golden_meta_info.json fields
compatible with qwen3_5_xh2a_spec_decode_test.py and bench.py scripts.
No GPU / no weights / no network required.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MERAK_MODEL = REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_5/qwen3_5_llm_model.py"
SPEC_TEST = REPO_ROOT / "examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_test.py"
SPEC_BENCH = REPO_ROOT / "examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_bench.py"
CONFIG_FILE = REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_5/xh_qwen3_5_config.py"


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found")


def _class_methods(cls_node: ast.ClassDef) -> list[str]:
    return [n.name for n in cls_node.body if isinstance(n, ast.FunctionDef)]


def test_merak_model_parses():
    _parse(MERAK_MODEL)


def test_kvcache_mixin_exists():
    tree = _parse(MERAK_MODEL)
    cls = _find_class(tree, "_Qwen3_5KVCacheMixin")
    methods = _class_methods(cls)
    assert "prepare_other_cache" in methods


def test_xhqwen3_5model_has_spec_decode_methods():
    tree = _parse(MERAK_MODEL)
    cls = _find_class(tree, "XHQwen3_5Model")
    methods = _class_methods(cls)
    assert "forward" in methods
    assert "get_export_cfg" in methods
    assert "export_hmonnx" in methods


def test_export_hmonnx_writes_spec_decode_section():
    """Verify export_hmonnx sets spec_decode, prefill_onnx, token_embedding_file."""
    src = MERAK_MODEL.read_text()
    assert "meta_info.spec_decode = spec_decode_section" in src
    assert "meta_info.prefill_onnx = meta_info.prefill_hmonnx" in src
    assert "meta_info.decode_onnx = meta_info.decode_hmonnx" in src
    assert "meta_info.token_embedding_file = meta_info.quant_embedding" in src
    assert "meta_info.max_context_tokens" in src


def test_export_hmonnx_spec_decode_fields():
    """Verify spec_decode section has fields expected by test scripts."""
    src = MERAK_MODEL.read_text()
    for field in ("mode", "block_size", "hidden_output_name"):
        assert f'"{field}"' in src or f"'{field}'" in src, f"missing spec_decode field: {field}"
    assert "draft_prefill_onnx" in src
    assert "draft_decode_onnx" in src
    assert "draft_context_onnx" in src


def test_get_export_cfg_handles_split_conv_cache():
    """Verify get_export_cfg generates split conv cache names."""
    src = MERAK_MODEL.read_text()
    assert 'past_conv_cache_{branch}_{cache_idx}' in src
    assert 'conv_cache_out_{branch}_{cache_idx}' in src


def test_get_export_cfg_handles_verify_intermediates():
    """Verify get_export_cfg generates per-step output names."""
    src = MERAK_MODEL.read_text()
    assert "verify_output_intermediates" in src
    assert "verify_steps" in src
    assert 'conv_cache_out_{cache_idx}_{step_idx}' in src


def test_get_export_cfg_handles_spec_decode_outputs():
    """Verify get_export_cfg adds target_hidden/post_norm_hidden."""
    src = MERAK_MODEL.read_text()
    assert '"target_hidden"' in src
    assert '"post_norm_hidden"' in src
    assert "output_hidden_state_indices" in src
    assert "output_post_norm_hidden" in src


def test_config_has_spec_decode_fields():
    """Verify XHQwen3_5ModelConfig has spec_decode_mode and num_draft_tokens."""
    tree = _parse(CONFIG_FILE)
    cls = _find_class(tree, "XHQwen3_5ModelConfig")
    init_fn = None
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == "__init__":
            init_fn = node
            break
    assert init_fn is not None
    arg_names = [a.arg for a in init_fn.args.args + init_fn.args.kwonlyargs]
    assert "spec_decode_mode" in arg_names
    assert "num_draft_tokens" in arg_names
    assert "mtp_config" in arg_names
    assert "dflash_config" in arg_names


def test_spec_decode_test_script_parses():
    _parse(SPEC_TEST)


def test_spec_decode_bench_script_parses():
    _parse(SPEC_BENCH)
