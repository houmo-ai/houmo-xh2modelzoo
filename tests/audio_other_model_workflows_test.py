from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper

from xhmodel_merak.workflows import AutoWorkflow
from xhmodel_merak.xh_other_model.models.melotts.graph import (
    materialize_axes_constants,
)
from xhmodel_merak.xh_other_model.models.zipformer.graph import (
    CACHE_FAMILIES,
    STACK_LAYER_COUNTS,
    convert_cached_len_interface_to_int32,
    split_layer_cache_interface,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "SileroVADWorkflow": ("configs_merak/workflows/xh2a/other_models/silero_vad/silero_vad_xh2a_w16.yaml"),
    "MeloTTSWorkflow": ("configs_merak/workflows/xh2a/other_models/melotts/melotts_xh2a_w16_l32_t64.yaml"),
    "KokoroWorkflow": ("configs_merak/workflows/xh2a/other_models/kokoro/kokoro_xh2a_bucketed_w16.yaml"),
    "StreamingZipformerWorkflow": ("configs_merak/workflows/xh2a/other_models/zipformer/zipformer_xh2a_w16.yaml"),
}


def test_audio_configs_resolve_registered_merak_workflows(
    tmp_path: Path,
) -> None:
    for expected_name, relative_config in CONFIGS.items():
        workflow = AutoWorkflow.from_config(
            model_dir=str(tmp_path),
            config_path=str(ROOT / relative_config),
        )
        assert type(workflow).__name__ == expected_name


def _cache_shape(family: str, stack: int, layers: int) -> list[int]:
    left_context = (64, 32, 16, 8, 32)[stack]
    if family == "len":
        return [layers, 1]
    if family == "avg":
        return [layers, 1, 160]
    if family == "key":
        return [layers, left_context, 1, 96]
    if family in {"val", "val2"}:
        return [layers, left_context, 1, 48]
    return [layers, 1, 160, 30]


def _synthetic_zipformer() -> onnx.ModelProto:
    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 39, 80])]
    outputs = [helper.make_tensor_value_info("encoder_out", TensorProto.FLOAT, [1, 39, 80])]
    nodes = [helper.make_node("Identity", ["x"], ["encoder_out"])]
    for family in CACHE_FAMILIES:
        for stack, layers in enumerate(STACK_LAYER_COUNTS):
            name = f"cached_{family}_{stack}"
            shape = _cache_shape(family, stack, layers)
            dtype = TensorProto.INT64 if family == "len" else TensorProto.FLOAT
            inputs.append(helper.make_tensor_value_info(name, dtype, shape))
            outputs.append(helper.make_tensor_value_info(f"new_{name}", dtype, shape))
            nodes.append(helper.make_node("Identity", [name], [f"new_{name}"]))
    graph = helper.make_graph(nodes, "synthetic_zipformer", inputs, outputs)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def test_zipformer_layer_cache_contract_is_85_by_85() -> None:
    model = _synthetic_zipformer()
    assert convert_cached_len_interface_to_int32(model) == 10
    assert split_layer_cache_interface(model) == 84
    onnx.checker.check_model(model)
    assert len(model.graph.input) == 85
    assert len(model.graph.output) == 85
    assert model.graph.input[0].name == "x"
    assert model.graph.output[0].name == "encoder_out"
    assert all(value.type.tensor_type.shape.dim[0].dim_value == 1 for value in model.graph.input[1:])
    input_names = {value.name for value in model.graph.input}
    output_names = {value.name for value in model.graph.output}
    assert "cached_key_1_layer2" in input_names
    assert "new_cached_conv2_4_layer2" in output_names


def test_melotts_structural_constant_is_materialized() -> None:
    axes = helper.make_tensor("axes_value", TensorProto.INT64, [1], [1])
    graph = helper.make_graph(
        [
            helper.make_node("Constant", [], ["axes"], name="axes", value=axes),
            helper.make_node("Unsqueeze", ["x", "axes"], ["y"], name="unsqueeze"),
        ],
        "axes_graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 1, 3])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    assert materialize_axes_constants(model) == 1
    onnx.checker.check_model(model)
    assert {value.name for value in model.graph.initializer} == {"axes"}
    assert [node.op_type for node in model.graph.node] == ["Unsqueeze"]
