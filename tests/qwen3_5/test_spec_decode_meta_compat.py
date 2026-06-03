"""Smoke tests for spec decode meta format compatibility (Sprint 2.3).

Verifies that the merak export produces golden_meta_info.json fields
compatible with qwen3_5_xh2a_spec_decode_test.py and bench.py scripts.
No GPU / no weights / no network required.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MERAK_MODEL = REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_5/qwen3_5_llm_model.py"
SPEC_TEST = REPO_ROOT / "examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_test.py"
SPEC_BENCH = REPO_ROOT / "examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_bench.py"
CONFIG_FILE = REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_5/xh_qwen3_5_config.py"

SPEC_DECODE_DIRS = {
    "mtp_draft_prefill",
    "mtp_draft_decode",
    "dflash_draft_context",
    "dflash_draft_context_decode",
    "dflash_draft_decode",
}
LEGACY_SPEC_DECODE_DIRS = {
    "mtp",
    "dflash",
    "decoder",
    "vision",
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _find_class(tree: ast.Module, name: str) -> ast.ClassDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found")


def _class_methods(cls_node: ast.ClassDef) -> list[str]:
    return [n.name for n in cls_node.body if isinstance(n, ast.FunctionDef)]


def _assert_golden_export_layout(export_dir: Path) -> None:
    release_prefix = export_dir.name
    assert release_prefix == release_prefix.lower()
    assert (export_dir / "prefill").is_dir()
    assert (export_dir / "decode").is_dir()
    assert not any((export_dir / name).exists() for name in LEGACY_SPEC_DECODE_DIRS)
    assert (export_dir / "quant_embedding.pt").is_file()
    assert (export_dir / "hf_config").is_dir()
    assert any((export_dir / "hf_config").iterdir())

    meta = json.loads((export_dir / "golden_meta_info.json").read_text())
    for key in ("prefill_hmonnx", "decode_hmonnx", "quant_embedding", "hf_config"):
        assert key in meta
    if (export_dir / "visual").exists():
        assert "visual_config" in meta
        visual_hmonnx = meta["visual_config"].get("hmonnx")
        assert isinstance(visual_hmonnx, str)
        assert visual_hmonnx.startswith("visual/")

    spec_decode = meta.get("spec_decode", {})
    spec_path_values = [
        value
        for key, value in spec_decode.items()
        if key.endswith("_onnx") and isinstance(value, str)
    ]
    for path_value in spec_path_values:
        top_dir = path_value.split("/", 1)[0]
        assert top_dir in SPEC_DECODE_DIRS or top_dir in {"prefill", "decode"}
        assert not path_value.startswith(("mtp/", "dflash/", "decoder/", "vision/"))

    for key in (
        "mtp_draft_prefill_onnx",
        "mtp_draft_decode_onnx",
        "dflash_draft_context_onnx",
        "dflash_draft_context_decode_onnx",
        "dflash_draft_decode_onnx",
    ):
        if key in spec_decode:
            assert spec_decode[key].split("/", 1)[0] == key.removesuffix("_onnx")

    for stage_dir in [
        export_dir / "prefill",
        export_dir / "decode",
        *([export_dir / "visual"] if (export_dir / "visual").exists() else []),
        *(export_dir / name for name in SPEC_DECODE_DIRS if (export_dir / name).exists()),
    ]:
        _assert_hmonnx_stage_layout(stage_dir, release_prefix)


def _assert_hmonnx_stage_layout(stage_dir: Path, release_prefix: str) -> None:
    assert stage_dir.name not in {"decoder", "vision"}
    onnx_files = sorted(stage_dir.rglob("*.onnx"))
    external_data_files = sorted(path for path in stage_dir.rglob("*external_data") if path.is_file())
    assert onnx_files, f"missing onnx under {stage_dir}"
    assert external_data_files, f"missing external_data under {stage_dir}"
    assert all(path.name.startswith(release_prefix) for path in onnx_files + external_data_files)

    step_dirs = sorted(path for path in stage_dir.rglob("step_*") if path.is_dir())
    for step_dir in step_dirs:
        step_onnx = list(step_dir.glob("*.onnx"))
        step_external_data = list(step_dir.glob("*external_data"))
        assert step_onnx, f"missing step onnx under {step_dir}"
        assert step_external_data, f"missing step external_data under {step_dir}"
        for path in step_onnx + step_external_data:
            assert path.is_symlink() or path.exists()


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
    assert "mtp_draft_prefill_onnx" in src
    assert "mtp_draft_decode_onnx" in src
    assert "dflash_draft_context_onnx" in src
    assert "dflash_draft_context_decode_onnx" in src
    assert "dflash_draft_decode_onnx" in src


def test_export_hmonnx_uses_release_spec_decode_dirs():
    """Verify spec decode draft exports use release-spec directory names."""
    src = MERAK_MODEL.read_text()
    for dirname in SPEC_DECODE_DIRS:
        assert f'"{dirname}"' in src
    assert 'Path(exported_info.exported_dir) / "mtp" / "prefill"' not in src
    assert 'Path(exported_info.exported_dir) / "mtp" / "decode"' not in src
    assert 'Path(dflash_output_dir) / "context"' not in src
    assert 'Path(dflash_output_dir) / "context_decode"' not in src
    assert 'Path(dflash_output_dir) / "decode"' not in src


def test_golden_export_layout_validator_accepts_release_spec(tmp_path):
    """Lightweight fixture for HM golden export naming/layout checks."""
    export_dir = tmp_path / "hmquant_xh2_qwen3_5_4w4a_256_32768_448x448_20260603"
    release_prefix = export_dir.name
    (export_dir / "hf_config").mkdir(parents=True)
    (export_dir / "hf_config" / "config.json").write_text("{}")
    (export_dir / "quant_embedding.pt").write_bytes(b"embedding")

    for dirname, suffix in (
        ("prefill", "prefill"),
        ("decode", "decode"),
        ("mtp_draft_prefill", "mtp_draft_prefill"),
        ("mtp_draft_decode", "mtp_draft_decode"),
        ("dflash_draft_context", "dflash_draft_context"),
        ("dflash_draft_context_decode", "dflash_draft_context_decode"),
        ("dflash_draft_decode", "dflash_draft_decode"),
    ):
        stage_dir = export_dir / dirname
        stage_dir.mkdir(parents=True)
        onnx = stage_dir / f"{release_prefix}_{suffix}.onnx"
        external_data = stage_dir / f"{release_prefix}_{suffix}_external_data"
        onnx.write_bytes(b"onnx")
        external_data.write_bytes(b"external")
        step_dir = stage_dir / "step_0"
        step_dir.mkdir()
        (step_dir / onnx.name).symlink_to(onnx)
        (step_dir / external_data.name).symlink_to(external_data)

    meta = {
        "hf_config": "hf_config",
        "quant_embedding": "quant_embedding.pt",
        "prefill_hmonnx": f"prefill/{release_prefix}_prefill.onnx",
        "decode_hmonnx": f"decode/{release_prefix}_decode.onnx",
        "spec_decode": {
            "mode": "dflash",
            "block_size": 4,
            "hidden_output_name": "target_hidden",
            "mtp_draft_prefill_onnx": f"mtp_draft_prefill/{release_prefix}_mtp_draft_prefill.onnx",
            "mtp_draft_decode_onnx": f"mtp_draft_decode/{release_prefix}_mtp_draft_decode.onnx",
            "dflash_draft_context_onnx": f"dflash_draft_context/{release_prefix}_dflash_draft_context.onnx",
            "dflash_draft_context_decode_onnx": (
                f"dflash_draft_context_decode/{release_prefix}_dflash_draft_context_decode.onnx"
            ),
            "dflash_draft_decode_onnx": f"dflash_draft_decode/{release_prefix}_dflash_draft_decode.onnx",
        },
    }
    (export_dir / "golden_meta_info.json").write_text(json.dumps(meta))

    _assert_golden_export_layout(export_dir)


def test_golden_export_layout_validator_allows_visual_branch(tmp_path):
    """VLM exports may include a top-level visual/ branch."""
    export_dir = tmp_path / "hmquant_xh2_qwen3_5_4w4a_256_32768_448x448_20260603"
    release_prefix = export_dir.name
    (export_dir / "hf_config").mkdir(parents=True)
    (export_dir / "hf_config" / "config.json").write_text("{}")
    (export_dir / "quant_embedding.pt").write_bytes(b"embedding")

    for dirname in ("prefill", "decode", "visual"):
        stage_dir = export_dir / dirname
        stage_dir.mkdir(parents=True)
        onnx = stage_dir / f"{release_prefix}_{dirname}.onnx"
        external_data = stage_dir / f"{release_prefix}_{dirname}_external_data"
        onnx.write_bytes(b"onnx")
        external_data.write_bytes(b"external")

    meta = {
        "hf_config": "hf_config",
        "quant_embedding": "quant_embedding.pt",
        "prefill_hmonnx": f"prefill/{release_prefix}_prefill.onnx",
        "decode_hmonnx": f"decode/{release_prefix}_decode.onnx",
        "visual_config": {"hmonnx": f"visual/{release_prefix}_visual.onnx"},
    }
    (export_dir / "golden_meta_info.json").write_text(json.dumps(meta))

    _assert_golden_export_layout(export_dir)


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
