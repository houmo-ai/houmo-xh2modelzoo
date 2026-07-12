import importlib.util
from pathlib import Path

import onnx
from onnx import TensorProto, helper


_MODULE_PATH = Path(__file__).parents[2] / "tools" / "audit_onnx_sefp16.py"
_SPEC = importlib.util.spec_from_file_location("audit_onnx_sefp16", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
audit_model = _MODULE.audit_model


def _write_model(
    path: Path,
    *,
    act_exp_bit: int = 5,
    act_hidden_bit: int = 1,
    act_nshare: int = 64,
    p_bits: int | None = 16,
) -> None:
    value = helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 1])
    output = helper.make_tensor_value_info("z", TensorProto.FLOAT16, [1, 1])
    linear = helper.make_node(
        "Linear",
        ["x", "x"],
        ["y"],
        name="linear",
        domain="ai.houmo.xh2a",
        mode="ssfp",
        hmfp_act_man_bit=16,
        hmfp_act_exp_bit=act_exp_bit,
        hmfp_act_hidden_bit=act_hidden_bit,
        hmfp_act_nshare=act_nshare,
    )
    flash_kwargs = {name: 16 for name in ("q_bits", "k_bits", "v_bits", "s_bits")}
    if p_bits is not None:
        flash_kwargs["p_bits"] = p_bits
    flash = helper.make_node(
        "FlashAttention",
        ["y"],
        ["z"],
        name="flash",
        domain="ai.houmo.xh2a",
        **flash_kwargs,
    )
    graph = helper.make_graph([linear, flash], "audit", [value], [output])
    onnx.save(helper.make_model(graph), path)


def test_audit_accepts_explicit_all_sefp16(tmp_path: Path) -> None:
    path = tmp_path / "valid.onnx"
    _write_model(path)

    report = audit_model(path)

    assert report["passed"] is True
    assert report["activation_node_count"] == 1
    assert report["flash_attention_node_count"] == 1
    assert report["precision_inventory"] == [
        {
            "op_type": "Linear",
            "mode": "ssfp",
            "hmfp_act_man_bit": 16,
            "hmfp_act_exp_bit": 5,
            "hmfp_act_hidden_bit": 1,
            "hmfp_act_nshare": 64,
            "count": 1,
        }
    ]


def test_audit_reports_each_invalid_or_missing_precision_field(tmp_path: Path) -> None:
    path = tmp_path / "invalid.onnx"
    _write_model(path, act_hidden_bit=0, p_bits=None)

    report = audit_model(path)

    assert report["passed"] is False
    assert any("activation is not SEFP16" in item for item in report["violations"])
    assert any("p_bits must be explicit 16" in item for item in report["violations"])


def test_audit_rejects_wrong_activation_exponent(tmp_path: Path) -> None:
    path = tmp_path / "wrong_exp.onnx"
    _write_model(path, act_exp_bit=4)

    report = audit_model(path)

    assert report["passed"] is False
    assert any("hmfp_act_exp_bit=4" in item for item in report["violations"])


def test_audit_rejects_wrong_activation_nshare(tmp_path: Path) -> None:
    path = tmp_path / "wrong_nshare.onnx"
    _write_model(path, act_nshare=32)

    report = audit_model(path)

    assert report["passed"] is False
    assert any("hmfp_act_nshare=32" in item for item in report["violations"])
