"""Smoke tests for spec decode meta format compatibility (Sprint 2.3).

Verifies that the merak export produces golden_meta_info.json fields
compatible with qwen3_5_xh2a_spec_decode_test.py and bench.py scripts.
No GPU / no weights / no network required.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MERAK_MODEL = REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_5/qwen3_5_llm_model.py"
SPEC_TEST = REPO_ROOT / "examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_test.py"
SPEC_BENCH = REPO_ROOT / "examples/llm/qwen3_5/qwen3_5_xh2a_spec_decode_bench.py"
CONFIG_FILE = REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_5/xh_qwen3_5_config.py"
LAYOUT_VALIDATOR = REPO_ROOT / "examples_merak/llm/qwen3_5/debug_scripts/validate_hm_release_layout.py"

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
STAGE_SUFFIXES = {
    "prefill": "prefill",
    "decode": "decode",
    "visual": "visual",
    "mtp_draft_prefill": "mtp_draft_prefill",
    "mtp_draft_decode": "mtp_draft_decode",
    "dflash_draft_context": "dflash_draft_context",
    "dflash_draft_context_decode": "dflash_draft_context_decode",
    "dflash_draft_decode": "dflash_draft_decode",
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _load_layout_validator():
    spec = importlib.util.spec_from_file_location("validate_hm_release_layout", LAYOUT_VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_external_data_onnx(onnx_path: Path, external_data_name: str) -> None:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper

    tensor = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=[1],
        vals=np.array([1.0], dtype=np.float32).tobytes(),
        raw=True,
    )
    graph = helper.make_graph(
        nodes=[],
        name="external_data_test",
        inputs=[],
        outputs=[],
        initializer=[tensor],
    )
    model = helper.make_model(graph)
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=external_data_name,
        size_threshold=0,
    )


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
    assert stage_dir.name in STAGE_SUFFIXES
    stage_suffix = STAGE_SUFFIXES[stage_dir.name]
    expected_onnx_name = f"{release_prefix}_{stage_suffix}_with_act.onnx"
    expected_external_data_name = f"{release_prefix}_{stage_suffix}_external_data"
    onnx_files = sorted(stage_dir.rglob("*.onnx"))
    external_data_files = sorted(path for path in stage_dir.rglob("*external_data") if path.is_file())
    assert onnx_files, f"missing onnx under {stage_dir}"
    assert external_data_files, f"missing external_data under {stage_dir}"
    assert any(path.name == expected_onnx_name for path in onnx_files)
    assert any(path.name == expected_external_data_name for path in external_data_files)
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
    assert "meta_info.spec_decode_draft_head_weight_bits" in src


def test_export_hmonnx_spec_decode_fields():
    """Verify spec_decode section has fields expected by test scripts."""
    src = MERAK_MODEL.read_text()
    for field in ("mode", "block_size", "draft_head_weight_bits", "hidden_output_name"):
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
        onnx = stage_dir / f"{release_prefix}_{suffix}_with_act.onnx"
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
        "prefill_hmonnx": f"prefill/{release_prefix}_prefill_with_act.onnx",
        "decode_hmonnx": f"decode/{release_prefix}_decode_with_act.onnx",
        "spec_decode": {
            "mode": "dflash",
            "block_size": 4,
            "hidden_output_name": "target_hidden",
            "mtp_draft_prefill_onnx": f"mtp_draft_prefill/{release_prefix}_mtp_draft_prefill_with_act.onnx",
            "mtp_draft_decode_onnx": f"mtp_draft_decode/{release_prefix}_mtp_draft_decode_with_act.onnx",
            "dflash_draft_context_onnx": f"dflash_draft_context/{release_prefix}_dflash_draft_context_with_act.onnx",
            "dflash_draft_context_decode_onnx": (
                f"dflash_draft_context_decode/{release_prefix}_dflash_draft_context_decode_with_act.onnx"
            ),
            "dflash_draft_decode_onnx": f"dflash_draft_decode/{release_prefix}_dflash_draft_decode_with_act.onnx",
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
        onnx = stage_dir / f"{release_prefix}_{dirname}_with_act.onnx"
        external_data = stage_dir / f"{release_prefix}_{dirname}_external_data"
        onnx.write_bytes(b"onnx")
        external_data.write_bytes(b"external")

    meta = {
        "hf_config": "hf_config",
        "quant_embedding": "quant_embedding.pt",
        "prefill_hmonnx": f"prefill/{release_prefix}_prefill_with_act.onnx",
        "decode_hmonnx": f"decode/{release_prefix}_decode_with_act.onnx",
        "visual_config": {"hmonnx": f"visual/{release_prefix}_visual_with_act.onnx"},
    }
    (export_dir / "golden_meta_info.json").write_text(json.dumps(meta))

    _assert_golden_export_layout(export_dir)


def test_release_layout_validator_checks_onnx_external_data_location(tmp_path):
    """Validator catches stale ONNX external_data locations after artifact renames."""
    validator = _load_layout_validator()
    export_dir = tmp_path / "hmquant_xh2_qwen3_5_4w4a_256_32768_448x448_20260603"
    release_prefix = export_dir.name
    (export_dir / "hf_config").mkdir(parents=True)
    (export_dir / "hf_config" / "config.json").write_text("{}")
    (export_dir / "quant_embedding.pt").write_bytes(b"embedding")

    for stage in ("prefill", "decode"):
        stage_dir = export_dir / stage
        stage_dir.mkdir(parents=True)
        external_data_name = f"{release_prefix}_{stage}_external_data"
        _write_external_data_onnx(stage_dir / f"{release_prefix}_{stage}_with_act.onnx", external_data_name)

    (export_dir / "golden_meta_info.json").write_text(
        json.dumps(
            {
                "hf_config": "hf_config",
                "quant_embedding": "quant_embedding.pt",
                "prefill_hmonnx": f"prefill/{release_prefix}_prefill_with_act.onnx",
                "decode_hmonnx": f"decode/{release_prefix}_decode_with_act.onnx",
            }
        )
    )

    assert validator.validate_release_layout(export_dir) == []

    import onnx

    prefill_onnx = export_dir / "prefill" / f"{release_prefix}_prefill_with_act.onnx"
    model = onnx.load_model(str(prefill_onnx), load_external_data=False)
    for tensor in model.graph.initializer:
        for entry in tensor.external_data:
            if entry.key == "location":
                entry.value = "stale_external_data"
    onnx.save_model(model, str(prefill_onnx))

    failures = validator.validate_release_layout(export_dir)
    assert any("external_data location mismatch" in failure for failure in failures)


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


def test_visual_export_uses_release_spec_stage_suffix():
    """Visual branch names must follow <release_prefix>_visual_* under visual/."""
    src = MERAK_MODEL.read_text()
    assert 'self.visual.config.model_name = f"{exported_info.model_name}_visual"' in src
    assert "self.visual.config.max_size_w}x{self.visual.config.max_size_h}" not in src


def test_spec_decode_export_uses_release_spec_stage_suffixes():
    """Draft branches must use the final release prefix, not child config names."""
    src = MERAK_MODEL.read_text()
    for suffix in (
        "mtp_draft_prefill",
        "mtp_draft_decode",
        "dflash_draft_context",
        "dflash_draft_context_decode",
        "dflash_draft_decode",
    ):
        assert f'f"{{exported_info.model_name}}_{suffix}"' in src
        assert f'f"{{mtp_base_cfg.model_name}}_{suffix}"' not in src
        assert f'f"{{dflash_base_cfg.model_name}}_{suffix}"' not in src


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
    assert "spec_draft_head_weight_bits" in arg_names
    assert "mtp_config" in arg_names
    assert "dflash_config" in arg_names


def test_spec_decode_test_script_parses():
    _parse(SPEC_TEST)


def test_spec_decode_bench_script_parses():
    _parse(SPEC_BENCH)

ONNX_RUNTIME_MODEL = REPO_ROOT / "xh_model_zoo/xh_llm/models/qwen3_5/qwen3_5_onnx_model.py"
MODEL_ZOO_CONVERT_CONFIG = REPO_ROOT / "xh_model_zoo/xh_llm/models/qwen3_5/qwen3_5_convert_config.py"
MODEL_ZOO_CONVERTER = REPO_ROOT / "xh_model_zoo/xh_llm/models/qwen3_5/qwen3_5_converter.py"
MODEL_ZOO_WRAP_MODEL = REPO_ROOT / "xh_model_zoo/xh_llm/models/qwen3_5/_model.py"
MODEL_ZOO_LLM_MODEL = REPO_ROOT / "xh_model_zoo/xh_llm/models/qwen3_5/qwen3_5_llm_model.py"
EXPORT_SCRIPT = REPO_ROOT / "examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py"
DEMO_RUNTIME = REPO_ROOT / "examples/llm/qwen3_5/_runtime.py"


def _load_function_from_source(path: Path, name: str):
    tree = _parse(path)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            module = ast.Module(
                body=[
                    ast.ImportFrom(
                        module="typing",
                        names=[ast.alias(name="Optional"), ast.alias(name="Tuple")],
                        level=0,
                    ),
                    node,
                ],
                type_ignores=[],
            )
            ast.fix_missing_locations(module)
            namespace = {}
            exec(compile(module, filename=str(path), mode="exec"), namespace)
            return namespace[name]
    raise AssertionError(f"function {name!r} not found in {path}")


def test_parse_conv_cache_name_accepts_split_and_legacy_suffixes():
    parse_name = _load_function_from_source(ONNX_RUNTIME_MODEL, "_parse_conv_cache_name")

    assert parse_name("past_conv_cache_0") == (None, "0")
    assert parse_name("past_conv_cache_q_0") == ("q", "0")
    assert parse_name("past_conv_cache_k_12") == ("k", "12")
    assert parse_name("past_conv_cache_v_3") == ("v", "3")


def test_dense_qwen3_5_model_zoo_defaults_preserve_split_conv_contract():
    convert_config_src = MODEL_ZOO_CONVERT_CONFIG.read_text()
    converter_src = MODEL_ZOO_CONVERTER.read_text()
    wrap_model_src = MODEL_ZOO_WRAP_MODEL.read_text()

    assert "split_conv_cache: bool = True" in convert_config_src
    assert "normalize_force_fp32: bool = False" in convert_config_src
    assert "use_manual_depthwise_conv1d: bool = False" in convert_config_src
    assert "fuse_gdr_ops: bool = False" in convert_config_src
    assert "force_fp32=self.config.normalize_force_fp32" in converter_src
    assert "split_conv_cache=self.config.split_conv_cache" in converter_src
    assert "use_manual_depthwise_conv1d=self.config.use_manual_depthwise_conv1d" in converter_src
    assert "fuse_gdr_ops=self.config.fuse_gdr_ops" in converter_src
    assert 'self.split_conv_cache = cfg.get("split_conv_cache", True)' in wrap_model_src
    assert 'self.fuse_gdr_ops = cfg.get("fuse_gdr_ops", False)' in wrap_model_src
    assert '"use_manual_depthwise_conv1d", False' in wrap_model_src


def test_dense_qwen3_5_split_cache_dims_prefer_actual_split_modules():
    dim_resolver = _load_function_from_source(MODEL_ZOO_LLM_MODEL, "_linear_split_conv_dims")

    class Conv:
        def __init__(self, in_channels: int):
            self.in_channels = in_channels

    class LinearAttn:
        key_dim = 4096
        value_dim = 4096
        head_k_dim = 64
        head_v_dim = 128
        num_v_heads = 32
        conv1d_q = Conv(2048)
        conv1d_k = Conv(2048)
        conv1d_v = Conv(4096)

    assert dim_resolver(LinearAttn()) == (2048, 2048, 4096)

    class PreSplitLinearAttn:
        key_dim = 4096
        value_dim = 4096
        head_k_dim = 64
        head_v_dim = 128
        num_v_heads = 32

    assert dim_resolver(PreSplitLinearAttn()) == (2048, 2048, 4096)


def test_dense_qwen3_5_merak_and_demo_expose_split_and_merged_modes():
    merak_impl_src = MERAK_MODEL.with_name("_llm_model_impl.py").read_text()
    export_script_src = EXPORT_SCRIPT.read_text()

    assert 'self.split_conv_cache = cfg.get("split_conv_cache", True)' in merak_impl_src
    assert 'self.fuse_gdr_ops = cfg.get("fuse_gdr_ops", False)' in merak_impl_src
    assert 'self.use_manual_depthwise_conv1d = cfg.get("use_manual_depthwise_conv1d", False)' in merak_impl_src
    assert 'cfg.model.wrap_cfg.split_conv_cache = getattr(args, "split_conv_cache", True)' in export_script_src
    assert 'default=True' in export_script_src
    assert 'dest="split_conv_cache"' in export_script_src
    assert '"--split_conv_cache"' in export_script_src
    assert '"--no_split_conv_cache"' in export_script_src
    assert 'action="store_false"' in export_script_src
    assert 'cfg.model.wrap_cfg.fuse_gdr_ops = getattr(args, "fuse_gdr_ops", False)' in export_script_src
    assert 'default=False' in export_script_src
    assert 'normalize_force_fp32 = getattr(args, "normalize_force_fp32", False)' in export_script_src
    assert 'cfg.model.wrap_cfg.use_manual_depthwise_conv1d = getattr(' in export_script_src


def test_dense_qwen3_5_dflash_uses_checkpoint_target_ids_and_guards_num_blocks():
    export_script_src = EXPORT_SCRIPT.read_text()
    runtime_src = (REPO_ROOT / "xh_model_zoo/xh_llm/models/qwen3_5/qwen3_5_spec_decode_onnx_model.py").read_text()

    assert "def _load_dflash_target_layer_ids" in export_script_src
    assert 'cfg.get("dflash_config", {}).get("target_layer_ids")' in export_script_src
    assert "def _validate_dflash_target_layer_ids" in export_script_src
    assert "--num_blocks/max_layers does not cover DFlash target_layer_ids" in export_script_src
    assert "_validate_dflash_target_layer_ids(" in export_script_src
    assert "DFlash target_hidden shape mismatch before running draft context" in runtime_src
    assert "dflash_config.target_layer_ids" in runtime_src




def test_dense_qwen3_5_demo_runtime_accepts_current_golden_meta_fields():
    runtime_src = DEMO_RUNTIME.read_text()

    assert 'meta_info.get("hf_config") or meta_info.get("hf_config_dir") or "hf_config"' in runtime_src
    assert 'meta_info.get("token_embedding_file")' in runtime_src
    assert 'or meta_info.get("quant_embedding")' in runtime_src
    assert 'or "quant_embedding.pt"' in runtime_src

def test_spec_decode_export_cfg_keeps_split_conv_mtp_dflash_input_names():
    src = MERAK_MODEL.read_text()

    assert 'self._decode_input_sequence_length = self.config.num_draft_tokens + 1' in src
    assert '"verify_output_intermediates": True' in src
    assert 'for branch in ("q", "k", "v")' in src
    assert 'f"past_conv_cache_{branch}_{cache_idx}"' in src
    assert 'f"conv_cache_out_{branch}_{cache_idx}_{step_idx}"' in src
    assert 'hidden_output_name = (\n                "target_hidden" if spec_decode_mode == "dflash" else "post_norm_hidden"\n            )' in src
