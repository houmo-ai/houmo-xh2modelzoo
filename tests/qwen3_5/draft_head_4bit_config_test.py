"""Tests for QTL-328: MTP / DFlash draft lm_head 4bit quantization config.

These tests intentionally avoid loading torch weights or instantiating GPUs.
They validate (a) helper config shape, (b) CLI surface, (c) shell script and
example script wiring.
"""

from __future__ import annotations

import ast
import importlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


# ----------------------------- helpers ---------------------------------------

def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


# ----------------------------- converter helper ------------------------------

@pytest.fixture(scope="module")
def converter_module():
    """Import the MoE converter module dynamically to access the helper."""
    return importlib.import_module(
        "xh_model_zoo.xh_llm.models.qwen3_5_moe.qwen3_5_moe_converter"
    )


def test_draft_quant_config_w4_structure(converter_module):
    cfg = converter_module._build_spec_draft_quant_config(4)
    assert cfg["quant_type"] == "w8a8h1_sefp"
    lm_head_schema = cfg["nodes_cfg"]["lm_head"]["w_schema"]
    assert lm_head_schema["bits"] == 4
    assert lm_head_schema["fp_mode"] == "ssfp"
    assert lm_head_schema["hidden_bit"] is False


def test_draft_quant_config_w8_no_override(converter_module):
    cfg = converter_module._build_spec_draft_quant_config(8)
    assert cfg["quant_type"] == "w8a8h1_sefp"
    # bit-8 should not impose a lm_head w_schema override
    assert "nodes_cfg" not in cfg or "lm_head" not in cfg.get("nodes_cfg", {})


def test_draft_quant_config_invalid_bits(converter_module):
    with pytest.raises(ValueError, match=r"Expected 4 or 8"):
        converter_module._build_spec_draft_quant_config(16)


def test_draft_base_quant_type_constant(converter_module):
    assert converter_module.DRAFT_BASE_QUANT_TYPE == "w8a8h1_sefp"


# ----------------------------- convert_config field --------------------------

def test_convert_config_has_field():
    src = _read("xh_model_zoo/xh_llm/models/qwen3_5_moe/qwen3_5_moe_convert_config.py")
    tree = ast.parse(src)
    fields = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5MoeConvertConfig":
            for stmt in node.body:
                if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                    fields.append(stmt.target.id)
    assert "spec_draft_head_weight_bits" in fields, (
        "Qwen3_5MoeConvertConfig must declare spec_draft_head_weight_bits"
    )


# ----------------------------- CLI surface -----------------------------------

@pytest.mark.parametrize(
    "rel_path",
    [
        "examples/llm/qwen3_5/qwen3_5_xh2a_export_hmonnx.py",
        "examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py",
    ],
)
def test_export_script_exposes_head_weight_bits_cli(rel_path):
    src = _read(rel_path)
    ast.parse(src)  # syntax check
    assert "--spec-draft-head-weight-bits" in src
    assert "spec_draft_head_weight_bits" in src
    # default must remain 4 to preserve QTL-328 default
    assert re.search(r"default\s*=\s*4", src), "CLI default must be 4"
    assert re.search(r"choices\s*=\s*\[\s*4\s*,\s*8\s*\]", src), "choices must be [4, 8]"


# ----------------------------- batch shell script ----------------------------

def test_batch_export_shell_has_head_bits_envs():
    rel = "examples/llm/qwen3_5/batch_export_8k.sh"
    src = _read(rel)
    assert "MTP_HEAD_WEIGHT_BITS" in src
    assert "DFLASH_HEAD_WEIGHT_BITS" in src
    assert "--spec-draft-head-weight-bits" in src

    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash not available")
    res = subprocess.run([bash, "-n", str(REPO_ROOT / rel)], capture_output=True, text=True)
    assert res.returncode == 0, f"bash -n failed: {res.stderr}"
