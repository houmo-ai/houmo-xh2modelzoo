import importlib.util
import pathlib

import numpy as np
import onnx
import pytest
from onnx import mapping
from onnx.reference import ReferenceEvaluator

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONVERT_LOOP_PATH = ROOT / "examples/cv/zte/convert_loop.py"
spec = importlib.util.spec_from_file_location("convert_loop", CONVERT_LOOP_PATH)
convert_loop = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(convert_loop)
unroll_loop = convert_loop.unroll_loop


def _make_inputs(model: onnx.ModelProto):
    initializer_names = {init.name for init in model.graph.initializer}
    feeds = {}
    for inp in model.graph.input:
        if inp.name in initializer_names:
            continue
        t = inp.type.tensor_type
        np_type = mapping.TENSOR_TYPE_TO_NP_TYPE.get(t.elem_type, np.float32)
        shape = []
        for d in t.shape.dim:
            if d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(1)
        if np_type == np.bool_:
            data = np.random.rand(*shape) > 0.5
        elif np.issubdtype(np_type, np.integer):
            data = np.random.randint(0, 3, size=shape, dtype=np_type)
        else:
            data = np.random.randn(*shape).astype(np_type)
        feeds[inp.name] = data
    return feeds


def test_unroll_loop_matches_reference():
    model_path = ROOT / "weights/zte/model_hexiaoxi_sim.onnx"
    if not model_path.exists():
        pytest.skip(f"missing model file: {model_path}")

    np.random.seed(0)
    model = onnx.load(str(model_path))
    unrolled = unroll_loop(onnx.load(str(model_path)), "while_loop", 9)

    assert all(n.op_type != "Loop" for n in unrolled.graph.node)

    feeds = _make_inputs(model)

    ref_orig = ReferenceEvaluator(model)
    ref_unrolled = ReferenceEvaluator(unrolled)

    outs_orig = ref_orig.run(None, feeds)
    outs_unrolled = ref_unrolled.run(None, feeds)

    assert len(outs_orig) == len(outs_unrolled)
    for a, b in zip(outs_orig, outs_unrolled):
        np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)
