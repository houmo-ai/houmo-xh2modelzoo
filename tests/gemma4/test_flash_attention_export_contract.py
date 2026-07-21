from pathlib import Path
from types import SimpleNamespace

import onnx
import pytest
import torch
import yaml
from onnx import TensorProto, helper


def _write_flash_contract_graph(
    path: Path,
    *,
    include_dense_input: bool = False,
    compact_inputs: tuple[str, ...] = ("kv_window_start_abs", "kv_valid_length"),
    sliding_windows: tuple[int, ...] = (512, -1),
    cache_widths: tuple[int, ...] = (832, 2048),
    stale_attention: bool = False,
    xhquant_export_names: bool = False,
    full_has_sliding_metadata: bool = False,
    attention_kv_source_layers: tuple[int, ...] | None = None,
    include_mm_prefix_input: bool = True,
    flash_mm_input_name: str | None = None,
    attention_scale: float = 1.0,
) -> None:
    past_length_name = "valid_length" if xhquant_export_names else "past_seq_length"
    current_length_name = "current_length" if xhquant_export_names else "current_input_length"

    def cache_name(cache_kind: str, cache_idx: int) -> str:
        if xhquant_export_names:
            suffix = "kcache" if cache_kind == "key" else "vcache"
            return f"model_layers_{cache_idx}_self_attn_{suffix}_input"
        return f"past_{cache_kind}_cache_{cache_idx}"

    inputs = [
        helper.make_tensor_value_info("inputs_embeds", TensorProto.FLOAT16, [1, 2, 8]),
        helper.make_tensor_value_info(past_length_name, TensorProto.INT32, [1]),
        helper.make_tensor_value_info(current_length_name, TensorProto.INT32, [1]),
    ]
    if include_mm_prefix_input:
        inputs.append(helper.make_tensor_value_info("mm_prefix_ranges", TensorProto.INT32, [1, 1, 2]))
    if include_dense_input:
        inputs.append(helper.make_tensor_value_info("sliding_attention_mask", TensorProto.FLOAT16, [1, 1, 2, 16]))
    for compact_input in compact_inputs:
        inputs.append(helper.make_tensor_value_info(compact_input, TensorProto.INT64, [1]))
    for cache_idx, width in enumerate(cache_widths):
        inputs.append(
            helper.make_tensor_value_info(cache_name("key", cache_idx), TensorProto.FLOAT16, [1, 2, width, 4])
        )
    for cache_idx, width in enumerate(cache_widths):
        inputs.append(
            helper.make_tensor_value_info(cache_name("value", cache_idx), TensorProto.FLOAT16, [1, 2, width, 4])
        )
    inputs.extend(
        [
            helper.make_tensor_value_info("table", TensorProto.FLOAT16, [1]),
            helper.make_tensor_value_info("mlp_in", TensorProto.FLOAT16, [1, 4]),
            helper.make_tensor_value_info("mlp_weight", TensorProto.FLOAT16, [4, 4]),
            helper.make_tensor_value_info("router_hidden", TensorProto.FLOAT16, [2, 8]),
            helper.make_tensor_value_info("router_weight", TensorProto.FLOAT16, [8, 4]),
        ]
    )

    compact_tail = list(compact_inputs)
    nodes = []
    for layer_idx, sliding_window in enumerate(sliding_windows):
        for prefix in ("q", "k", "v"):
            inputs.append(helper.make_tensor_value_info(f"{prefix}_{layer_idx}", TensorProto.FLOAT16, [1, 1, 2, 4]))
        kv_source_layer = attention_kv_source_layers[layer_idx] if attention_kv_source_layers is not None else layer_idx
        flash_inputs = [
            f"q_{layer_idx}",
            f"k_{kv_source_layer}",
            f"v_{kv_source_layer}",
            "table",
            past_length_name,
            current_length_name,
        ]
        if sliding_window > 0:
            mm_input = flash_mm_input_name
            if mm_input is None:
                mm_input = "mm_prefix_ranges" if include_mm_prefix_input else ""
            flash_inputs.extend(["", mm_input, *compact_tail])
        elif include_mm_prefix_input:
            # Bidirectional visual ranges apply to both Gemma4 sliding and
            # full attention; only the sliding layer consumes window scalars.
            flash_inputs.extend(["", flash_mm_input_name or "mm_prefix_ranges"])
            if full_has_sliding_metadata:
                flash_inputs.extend(compact_tail)
        nodes.append(
            helper.make_node(
                "FlashAttention",
                flash_inputs,
                [f"attention_{layer_idx}"],
                name=f"attention_layer_{layer_idx}",
                domain="xh2a",
                sliding_window=sliding_window,
                scale=attention_scale,
            )
        )
    for cache_idx, _ in enumerate(cache_widths):
        nodes.extend(
            [
                helper.make_node(
                    "KVcache",
                    [cache_name("key", cache_idx)],
                    [f"present_key_{cache_idx}"],
                    name=f"key_cache_{cache_idx}",
                    domain="xh2a",
                ),
                helper.make_node(
                    "KVcache",
                    [cache_name("value", cache_idx)],
                    [f"present_value_{cache_idx}"],
                    name=f"value_cache_{cache_idx}",
                    domain="xh2a",
                ),
            ]
        )
    nodes.extend(
        [
            helper.make_node("MatMul", ["mlp_in", "mlp_weight"], ["mlp_out"], name="mlp_matmul"),
            helper.make_node(
                "MatMul",
                ["router_hidden", "router_weight"],
                ["router_scores"],
                name="moe_router_matmul",
            ),
            helper.make_node(
                "Softmax",
                ["router_scores"],
                ["router_probs"],
                name="moe_router_softmax",
            ),
        ]
    )
    if stale_attention:
        inputs.extend(
            [
                helper.make_tensor_value_info("batched_left", TensorProto.FLOAT16, [1, 2, 4]),
                helper.make_tensor_value_info("batched_right", TensorProto.FLOAT16, [1, 4, 2]),
            ]
        )
        nodes.extend(
            [
                helper.make_node(
                    "MatMul",
                    ["batched_left", "batched_right"],
                    ["batched_scores"],
                    name="batched_matmul",
                ),
                helper.make_node(
                    "Softmax",
                    ["batched_scores"],
                    ["stale_probs"],
                    name="probability_normalization",
                ),
            ]
        )
    graph = helper.make_graph(
        nodes,
        "gemma4_flash_contract",
        inputs,
        [helper.make_tensor_value_info(f"attention_{len(sliding_windows) - 1}", TensorProto.FLOAT16, [1, 2, 8])],
    )
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18), helper.make_opsetid("xh2a", 1)]),
        path,
    )


def _flash_contract_meta() -> dict:
    return {
        "attention_contract_version": 2,
        "uses_sliding_flash_attention_v2": True,
        "bidirectional_vision_attention": True,
        "layer_types": ["sliding_attention", "full_attention"],
        "layer_cache_indices": [0, 1],
        "layer_cache_owner_indices": [0, 1],
        "layer_cache_types": ["sliding_attention", "full_attention"],
        "layer_kv_shapes": [[1, 2, 832, 4], [1, 2, 2048, 4]],
    }


def test_onnx_validator_counts_attention_layers_separately_from_shared_kv_cache_owners(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        validate_gemma4_flash_attention_graph,
    )

    graph_path = tmp_path / "shared_kv.onnx"
    _write_flash_contract_graph(
        graph_path,
        sliding_windows=(512, 512, -1),
        cache_widths=(832, 2048),
        attention_kv_source_layers=(0, 0, 2),
    )
    meta = _flash_contract_meta()
    meta.update(
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        layer_cache_indices=[0, 0, 1],
        layer_cache_owner_indices=[0, 2],
    )

    assert validate_gemma4_flash_attention_graph(graph_path, meta) == {
        "flash_attention_nodes": 3,
        "sliding_nodes": 2,
        "full_nodes": 1,
    }


def test_e2b_layer_cache_layout_maps_every_shared_attention_layer_to_its_physical_owner():
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        _build_gemma4_layer_cache_layout,
    )

    layer_types = ["full_attention" if (idx + 1) % 5 == 0 else "sliding_attention" for idx in range(35)]
    layers = []
    for layer_idx, layer_type in enumerate(layer_types):
        is_shared = layer_idx >= 15
        owner_layer_idx = 14 if layer_type == "full_attention" else 13
        head_dim = 512 if layer_type == "full_attention" else 256
        layers.append(
            SimpleNamespace(
                self_attn=SimpleNamespace(
                    is_kv_shared_layer=is_shared,
                    kv_shared_layer_index=owner_layer_idx if is_shared else None,
                    head_dim=head_dim,
                    k_proj=SimpleNamespace(out_features=head_dim),
                )
            )
        )

    shapes, cache_types, owner_indices, layer_cache_indices = _build_gemma4_layer_cache_layout(
        layers=layers,
        layer_types=layer_types,
        context_max_length=131072,
        sliding_window=512,
        input_seq_len=320,
        sliding_kv_cache_input_mode="slice_window",
    )

    shared_mapping = [14 if layer_type == "full_attention" else 13 for layer_type in layer_types[15:]]
    expected_mapping = list(range(15)) + shared_mapping
    assert owner_indices == list(range(15))
    assert layer_cache_indices == expected_mapping
    assert len(layer_cache_indices) == len(layer_types) == 35
    assert set(layer_cache_indices) == set(range(15))
    assert len(layer_cache_indices) - len(set(layer_cache_indices)) == 20
    assert cache_types == layer_types[:15]
    assert [shape[2] for shape in shapes] == [131072 if kind == "full_attention" else 832 for kind in cache_types]


def test_onnx_validator_rejects_shared_layer_whose_kv_inputs_do_not_match_mapped_owner(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        validate_gemma4_flash_attention_graph,
    )

    graph_path = tmp_path / "wrong_shared_kv.onnx"
    _write_flash_contract_graph(
        graph_path,
        sliding_windows=(512, 512, -1),
        cache_widths=(832, 2048),
    )
    meta = _flash_contract_meta()
    meta.update(
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        layer_cache_indices=[0, 0, 1],
        layer_cache_owner_indices=[0, 2],
    )

    with pytest.raises(ValueError, match="layer 1 maps to cache 0.*K/V inputs"):
        validate_gemma4_flash_attention_graph(graph_path, meta)


def test_onnx_validator_accepts_v2_flash_contract_and_unrelated_router_softmax(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        validate_gemma4_flash_attention_graph,
    )

    graph_path = tmp_path / "valid.onnx"
    _write_flash_contract_graph(graph_path)
    onnx.checker.check_model(onnx.load(graph_path))

    assert validate_gemma4_flash_attention_graph(graph_path, _flash_contract_meta()) == {
        "flash_attention_nodes": 2,
        "sliding_nodes": 1,
        "full_nodes": 1,
    }


def test_onnx_validator_rejects_false_sliding_flash_v2_metadata_flag(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        validate_gemma4_flash_attention_graph,
    )

    graph_path = tmp_path / "false_v2_flag.onnx"
    _write_flash_contract_graph(graph_path)
    meta = _flash_contract_meta()
    meta["uses_sliding_flash_attention_v2"] = False

    with pytest.raises(ValueError, match="uses_sliding_flash_attention_v2=False.*expected True"):
        validate_gemma4_flash_attention_graph(graph_path, meta)


def test_onnx_validator_accepts_non_bidirectional_graph_without_mm_input(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        validate_gemma4_flash_attention_graph,
    )

    graph_path = tmp_path / "non_bidi.onnx"
    _write_flash_contract_graph(graph_path, include_mm_prefix_input=False)
    meta = _flash_contract_meta()
    meta["bidirectional_vision_attention"] = False

    assert validate_gemma4_flash_attention_graph(graph_path, meta)["flash_attention_nodes"] == 2


def test_onnx_validator_rejects_non_bidirectional_graph_level_mm_input(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        validate_gemma4_flash_attention_graph,
    )

    graph_path = tmp_path / "non_bidi_with_mm_graph_input.onnx"
    _write_flash_contract_graph(graph_path)
    meta = _flash_contract_meta()
    meta["bidirectional_vision_attention"] = False

    with pytest.raises(ValueError, match="non-bidirectional graph retains mm_prefix_ranges"):
        validate_gemma4_flash_attention_graph(graph_path, meta)


def test_onnx_validator_rejects_non_bidirectional_sliding_node_mm_slot(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        validate_gemma4_flash_attention_graph,
    )

    graph_path = tmp_path / "non_bidi_with_mm_node_slot.onnx"
    _write_flash_contract_graph(
        graph_path,
        include_mm_prefix_input=False,
        flash_mm_input_name="stale_mm_prefix_ranges",
    )
    meta = _flash_contract_meta()
    meta["bidirectional_vision_attention"] = False

    with pytest.raises(ValueError, match="non-bidirectional sliding FlashAttention layer 0 carries input\\[7\\]"):
        validate_gemma4_flash_attention_graph(graph_path, meta)


def test_onnx_validator_accepts_real_xhquant_hmonnx_input_names(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        validate_gemma4_flash_attention_graph,
    )

    graph_path = tmp_path / "xhquant_names.onnx"
    _write_flash_contract_graph(graph_path, xhquant_export_names=True)

    assert validate_gemma4_flash_attention_graph(graph_path, _flash_contract_meta()) == {
        "flash_attention_nodes": 2,
        "sliding_nodes": 1,
        "full_nodes": 1,
    }


@pytest.mark.parametrize(
    ("graph_kwargs", "meta_mutation", "message"),
    [
        ({"include_dense_input": True}, None, "retains sliding_attention_mask"),
        ({"compact_inputs": ("kv_valid_length",)}, None, "missing kv_window_start_abs"),
        ({"compact_inputs": ("kv_window_start_abs",)}, None, "missing kv_valid_length"),
        (
            {"compact_inputs": ("kv_valid_length", "kv_window_start_abs")},
            None,
            "external input prefix=.*kv_valid_length.*expected.*kv_window_start_abs",
        ),
        ({"sliding_windows": (512,)}, None, "FlashAttention count=1, expected 2"),
        ({"sliding_windows": (512, 512)}, None, "full_attention.*sliding_window"),
        ({"full_has_sliding_metadata": True}, None, "full_attention.*sliding metadata"),
        ({"attention_scale": 0.0625}, None, "FlashAttention.*scale=0.0625.*expected 1.0"),
        ({"cache_widths": (816, 2048)}, None, "past_key_cache_0.*shape"),
        ({"stale_attention": True}, None, "attention Softmax.*probability_normalization"),
    ],
)
def test_onnx_validator_rejects_stale_graph_contract(
    tmp_path,
    graph_kwargs,
    meta_mutation,
    message,
):
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        validate_gemma4_flash_attention_graph,
    )

    graph_path = tmp_path / "stale.onnx"
    _write_flash_contract_graph(graph_path, **graph_kwargs)
    meta = _flash_contract_meta()
    if meta_mutation is not None:
        meta_mutation(meta)

    with pytest.raises(ValueError, match=message):
        validate_gemma4_flash_attention_graph(graph_path, meta)


def test_quant_export_validation_fails_before_golden_generation(monkeypatch, tmp_path):
    from examples_merak.llm.gemma4_series import gemma4_series_quant_export as quant_export
    from xhmodel_merak.xh_llm import workflows

    events = []
    work_dir = tmp_path / "export"
    artifact_dir = work_dir / "hmquant_gemma4"

    class FakeWorkflow:
        def quant(self, **kwargs):
            events.append("quant")
            return SimpleNamespace(skipped=True)

        def export(self, **kwargs):
            events.append("export")
            artifact_dir.mkdir(parents=True)
            _write_flash_contract_graph(artifact_dir / "prefill.onnx", stale_attention=True)
            _write_flash_contract_graph(artifact_dir / "decode.onnx")
            meta = {
                **_flash_contract_meta(),
                "prefill_hmonnx": "prefill.onnx",
                "decode_hmonnx": "decode.onnx",
            }
            (artifact_dir / "golden_meta_info.json").write_text(__import__("json").dumps(meta), encoding="utf-8")
            return SimpleNamespace(work_dir=str(work_dir), config_file="effective.yaml")

        def dump_golden(self, **kwargs):
            events.append("golden")
            return "should-not-run.json"

    monkeypatch.setattr(workflows.AutoLLMWorkflow, "from_config", lambda **kwargs: FakeWorkflow())
    args = quant_export.build_parser().parse_args(
        [
            "--hf-model-dir",
            str(tmp_path / "hf"),
            "--config",
            str(tmp_path / "config.yaml"),
            "--export-output-dir",
            str(work_dir),
            "--base",
            "--dump-golden",
        ]
    )

    with pytest.raises(ValueError, match="attention Softmax"):
        quant_export.main(args)
    assert events == ["quant", "export"]


def test_quant_export_validation_reports_prefill_and_decode_facts(tmp_path):
    from examples_merak.llm.gemma4_series import gemma4_series_quant_export as quant_export

    artifact_dir = tmp_path / "work" / "hmquant_gemma4"
    artifact_dir.mkdir(parents=True)
    _write_flash_contract_graph(artifact_dir / "prefill.onnx")
    _write_flash_contract_graph(artifact_dir / "decode.onnx")
    meta = {
        **_flash_contract_meta(),
        "prefill_hmonnx": "prefill.onnx",
        "decode_hmonnx": "decode.onnx",
    }
    (artifact_dir / "golden_meta_info.json").write_text(__import__("json").dumps(meta), encoding="utf-8")

    facts = quant_export._validate_flash_attention_export_result(SimpleNamespace(work_dir=str(tmp_path / "work")))
    assert facts == {
        "prefill_hmonnx": {
            "flash_attention_nodes": 2,
            "sliding_nodes": 1,
            "full_nodes": 1,
        },
        "decode_hmonnx": {
            "flash_attention_nodes": 2,
            "sliding_nodes": 1,
            "full_nodes": 1,
        },
    }


def test_quant_export_validation_model_hook_rejects_stale_decode(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        XHGemma4SeriesModel,
    )
    from xhmodel_merak.xh_llm.vision_llm_model import VisionLLMModel

    _write_flash_contract_graph(tmp_path / "prefill.onnx")
    _write_flash_contract_graph(tmp_path / "decode.onnx", stale_attention=True)
    meta_dict = {
        **_flash_contract_meta(),
        "prefill_hmonnx": "prefill.onnx",
        "decode_hmonnx": "decode.onnx",
    }
    meta = SimpleNamespace(**meta_dict, to_dict=lambda: dict(meta_dict))
    exported_info = SimpleNamespace(meta=meta, exported_dir=str(tmp_path))
    monkeypatch.setattr(
        VisionLLMModel,
        "_export_hmonnx",
        lambda self, info: info,
    )
    model = object.__new__(XHGemma4SeriesModel)

    with pytest.raises(ValueError, match="decode.onnx.*attention Softmax"):
        model._export_hmonnx(exported_info)


def test_e2e_validation_uses_reusable_v2_graph_validator(tmp_path):
    from examples_merak.llm.gemma4_series.gemma4_e2e_validation import (
        _validate_flash_hmonnx_contract,
    )

    _write_flash_contract_graph(tmp_path / "prefill.onnx")
    _write_flash_contract_graph(tmp_path / "decode.onnx")
    meta = {
        **_flash_contract_meta(),
        "prefill_hmonnx": "prefill.onnx",
        "decode_hmonnx": "decode.onnx",
    }

    facts = _validate_flash_hmonnx_contract(tmp_path / "golden_meta_info.json", meta)
    assert facts["prefill_hmonnx"]["sliding_nodes"] == 1
    assert facts["decode_hmonnx"]["full_nodes"] == 1


def _patch_hf_config(monkeypatch, *, layer_types=None, sliding_window=512):
    from xhmodel_merak.xh_llm.models.gemma4_series.xh_gemma4_series_config import (
        XHGemma4SeriesModelConfig,
    )

    if layer_types is None:
        layer_types = ["sliding_attention", "full_attention"]
    hf_config = {
        "text_config": {
            "layer_types": layer_types,
            "sliding_window": sliding_window,
        }
    }
    monkeypatch.setattr(
        XHGemma4SeriesModelConfig,
        "_load_hf_config",
        staticmethod(lambda _: hf_config),
    )
    return XHGemma4SeriesModelConfig


def test_v2_mtp_is_rejected_by_public_config(monkeypatch):
    config_cls = _patch_hf_config(monkeypatch)

    with pytest.raises(ValueError, match="contract-v2.*MTP"):
        config_cls(
            model_name="e2b",
            attention_contract_version=2,
            spec_decode_mode="mtp",
        )


def test_contract_gate_is_typed_and_requires_explicit_sliding_layer(monkeypatch):
    config_cls = _patch_hf_config(monkeypatch)

    v2 = config_cls(
        model_name="e2b",
        attention_contract_version=2,
        flash_attention={
            "q_bits": 16,
            "k_bits": 8,
            "v_bits": 16,
            "s_bits": 8,
            "p_bits": 16,
        },
        max_mm_ranges_per_chunk=3,
    )
    assert v2.attention_contract_version == 2
    assert v2.max_mm_ranges_per_chunk == 3
    assert v2.flash_attention == {
        "q_bits": 16,
        "k_bits": 8,
        "v_bits": 16,
        "s_bits": 8,
        "p_bits": 16,
    }
    assert v2.uses_sliding_flash_attention_v2 is True

    config_cls = _patch_hf_config(monkeypatch, layer_types=["full_attention"])
    full_only = config_cls(model_name="full_only", attention_contract_version=2)
    assert full_only.uses_sliding_flash_attention_v2 is False

    legacy = config_cls(model_name="legacy")
    assert legacy.attention_contract_version == 1
    assert legacy.uses_sliding_flash_attention_v2 is False


def test_attention_visibility_spec_is_identical_across_lowering_versions(monkeypatch):
    config_cls = _patch_hf_config(
        monkeypatch,
        layer_types=["sliding_attention", "full_attention"],
        sliding_window=512,
    )

    legacy = config_cls(model_name="legacy", attention_contract_version=1, max_mm_ranges_per_chunk=3)
    flash = config_cls(model_name="flash", attention_contract_version=2, max_mm_ranges_per_chunk=3)

    assert legacy.attention_lowering == "legacy_attention"
    assert flash.attention_lowering == "flash_attention"
    assert legacy.attention_visibility_spec == flash.attention_visibility_spec
    assert flash.attention_visibility_spec == {
        "layer_types": ["sliding_attention", "full_attention"],
        "is_causal": True,
        "bidirectional_vision_attention": legacy.bidirectional_vision_attention,
        "sliding_window": 512,
        "max_mm_ranges_per_chunk": 3,
        "has_full_attention": True,
        "has_sliding_attention": True,
        "requires_kv_window_metadata": True,
        "requires_mm_prefix_ranges": legacy.bidirectional_vision_attention,
    }


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"attention_contract_version": 3}, "attention_contract_version"),
        ({"attention_contract_version": 2, "max_mm_ranges_per_chunk": 0}, "max_mm_ranges_per_chunk"),
        (
            {"attention_contract_version": 2, "sliding_kv_cache_input_mode": "legacy_full"},
            "contract-v2.*slice_window",
        ),
    ],
)
def test_contract_v2_rejects_invalid_configuration(monkeypatch, kwargs, message):
    config_cls = _patch_hf_config(monkeypatch)
    with pytest.raises(ValueError, match=message):
        config_cls(model_name="invalid", **kwargs)


def test_contract_metadata_fields_are_normalized():
    from xhmodel_merak.xh_llm.models.gemma4_series.xh_gemma4_series_config import (
        Gemma4SeriesModelMeta,
    )

    meta = Gemma4SeriesModelMeta(
        attention_contract_version="2",
        max_mm_ranges_per_chunk="4",
        uses_sliding_flash_attention_v2=1,
    )
    assert meta.attention_contract_version == 2
    assert meta.max_mm_ranges_per_chunk == 4
    assert meta.uses_sliding_flash_attention_v2 is True

    restored = Gemma4SeriesModelMeta.from_dict(
        {
            "meta": {"class_name": "Gemma4SeriesModelMeta"},
            "attention_contract_version": "2",
            "max_mm_ranges_per_chunk": "5",
            "uses_sliding_flash_attention_v2": 1,
        }
    )
    assert restored.attention_contract_version == 2
    assert restored.max_mm_ranges_per_chunk == 5
    assert restored.uses_sliding_flash_attention_v2 is True


def test_flash_attention_yamls_are_additive_and_preserve_existing_configs():
    root = Path("configs_merak/workflows/xh2a/llm_models/gemma4_series")
    paths = sorted(root.glob("*/*.yaml"))
    assert len(paths) == 24

    mtp_paths = [path for path in paths if path.stem.endswith("_mtp")]
    base_paths = [
        path for path in paths if not path.stem.endswith(("_mtp", "_flash_attention"))
    ]
    flash_paths = [path for path in paths if path.stem.endswith("_flash_attention")]
    assert len(mtp_paths) == len(base_paths) == len(flash_paths) == 8

    for path in [*mtp_paths, *base_paths]:
        model = yaml.safe_load(path.read_text(encoding="utf-8"))["export"]["model"]
        assert int(model.get("attention_contract_version", 1)) == 1, path
        assert "max_mm_ranges_per_chunk" not in model, path

    for base_path in base_paths:
        flash_path = base_path.with_name(f"{base_path.stem}_flash_attention.yaml")
        assert flash_path in flash_paths
        base_payload = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        flash_payload = yaml.safe_load(flash_path.read_text(encoding="utf-8"))
        flash_model = flash_payload["export"]["model"]
        assert flash_model.pop("attention_contract_version") == 2, flash_path
        assert flash_model.pop("max_mm_ranges_per_chunk") >= 1, flash_path
        assert flash_model.pop("flash_attention") == {
            "q_bits": 8,
            "k_bits": 8,
            "v_bits": 8,
            "s_bits": 8,
            "p_bits": 8,
        }, flash_path
        assert flash_payload == base_payload, flash_path


def _make_series_processor(
    *,
    window: int,
    query: int,
    bidirectional: bool,
    max_ranges: int = 1,
    contract_version: int = 2,
    attention_visibility_spec=None,
):
    import torch

    from xhmodel_merak.xh_llm.models.gemma4_series.data_preprocess import (
        Gemma4DataPreprocess,
    )

    processor = Gemma4DataPreprocess(
        token_embedding=torch.nn.Embedding(32, 8),
        input_sequence_length=query,
        context_length=512,
        past_key_caches=[],
        past_value_caches=[],
        pad_token_id=0,
        sliding_window=window,
        bidirectional_vision_attention=bidirectional,
        attention_contract_version=contract_version,
        max_mm_ranges_per_chunk=max_ranges,
        attention_visibility_spec=attention_visibility_spec,
    )
    return processor.to(device="cpu", dtype=torch.float32)


@pytest.mark.parametrize(
    ("window", "query", "past", "current", "expected"),
    [
        (512, 320, 0, 17, (0, 17)),
        (512, 320, 511, 320, (0, 831)),
        (512, 320, 512, 320, (1, 831)),
        (1024, 320, 4096, 7, (3073, 1030)),
    ],
)
def test_compact_metadata_formula(window, query, past, current, expected):
    import torch

    processor = _make_series_processor(
        window=window,
        query=query,
        bidirectional=True,
    )
    start, valid, ranges = processor._build_compact_attention_metadata(
        current_input_length=current,
        past_seq_length=past,
        mm_token_type_ids=torch.zeros(query, dtype=torch.long),
        device=torch.device("cpu"),
    )

    assert (start.item(), valid.item()) == expected
    assert start.shape == valid.shape == (1,)
    assert start.dtype == valid.dtype == torch.int64
    assert ranges.shape == (1, 1, 2)
    assert ranges.dtype == torch.int32
    assert ranges.tolist() == [[[0, 0]]]


def test_visibility_metadata_resolution_is_identical_across_lowering_versions():
    import torch

    processors = [
        _make_series_processor(
            window=16,
            query=8,
            bidirectional=True,
            max_ranges=2,
            contract_version=version,
        )
        for version in (1, 2)
    ]
    mm_token_type_ids = torch.tensor([0, 1, 1, 0, 1, 0, 0, 0], dtype=torch.long)

    assert processors[0].attention_lowering == "legacy_attention"
    assert processors[1].attention_lowering == "flash_attention"
    assert processors[0].attention_visibility_spec == processors[1].attention_visibility_spec
    resolved = [
        processor._build_compact_attention_metadata(
            current_input_length=6,
            past_seq_length=40,
            mm_token_type_ids=mm_token_type_ids,
            device=torch.device("cpu"),
        )
        for processor in processors
    ]
    for legacy_value, flash_value in zip(resolved[0], resolved[1], strict=True):
        torch.testing.assert_close(legacy_value, flash_value)
    assert resolved[0][0].tolist() == [25]
    assert resolved[0][1].tolist() == [21]
    assert resolved[0][2].tolist() == [[[41, 42], [44, 44]]]


def test_full_only_visibility_uses_absolute_kv_metadata_without_contract_gate():
    import torch

    from xhmodel_merak.xh_llm.models.gemma4_series.attention_visibility import (
        Gemma4AttentionVisibilitySpec,
    )

    spec = Gemma4AttentionVisibilitySpec.from_checkpoint_semantics(
        layer_types=["full_attention"],
        sliding_window=512,
        bidirectional_vision_attention=False,
        max_mm_ranges_per_chunk=1,
    )
    processors = [
        _make_series_processor(
            window=512,
            query=8,
            bidirectional=False,
            contract_version=version,
            attention_visibility_spec=spec,
        )
        for version in (1, 2)
    ]
    for processor in processors:
        start, valid, ranges = processor._build_compact_attention_metadata(
            current_input_length=3,
            past_seq_length=40,
            mm_token_type_ids=torch.zeros(8, dtype=torch.long),
            device=torch.device("cpu"),
        )
        assert start.tolist() == [0]
        assert valid.tolist() == [43]
        assert ranges.tolist() == [[[0, 0]]]


def test_visual_ranges_are_absolute_inclusive_and_overflow_fails():
    import torch

    processor = _make_series_processor(
        window=1024,
        query=320,
        bidirectional=True,
        max_ranges=1,
    )
    mm_token_type_ids = torch.tensor([0, 1, 1, 1, 0], dtype=torch.long)
    _, _, ranges = processor._build_compact_attention_metadata(
        5,
        2000,
        mm_token_type_ids,
        torch.device("cpu"),
    )
    assert ranges.tolist() == [[[2001, 2003]]]

    with pytest.raises(ValueError, match="max_mm_ranges_per_chunk"):
        processor._build_compact_attention_metadata(
            5,
            0,
            torch.tensor([1, 0, 1, 0, 0], dtype=torch.long),
            torch.device("cpu"),
        )


def test_visual_ranges_use_fixed_capacity_padding_and_nonvisual_mode_is_empty():
    import torch

    processor = _make_series_processor(
        window=8,
        query=8,
        bidirectional=True,
        max_ranges=3,
    )
    _, _, ranges = processor._build_compact_attention_metadata(
        8,
        100,
        torch.tensor([1, 1, 0, 0, 1, 0, 0, 0], dtype=torch.long),
        torch.device("cpu"),
    )
    assert ranges.tolist() == [[[100, 101], [104, 104], [0, 0]]]

    nonvisual = _make_series_processor(
        window=8,
        query=8,
        bidirectional=False,
        max_ranges=2,
    )
    _, _, nonvisual_ranges = nonvisual._build_compact_attention_metadata(
        8,
        100,
        torch.ones(8, dtype=torch.long),
        torch.device("cpu"),
    )
    assert nonvisual_ranges.tolist() == [[[0, 0], [0, 0]]]


@pytest.mark.parametrize("window", [512, 1024])
def test_compact_atomic_multimodal_chunks_stage_only_intersecting_inclusive_ranges(
    window,
):
    from xhmodel_merak.xh_llm.models.gemma4_series.llm_text import (
        plan_gemma4_atomic_prefill_chunks,
    )

    mm_token_type_ids = torch.zeros(900, dtype=torch.long)
    mm_token_type_ids[100:380] = 1
    mm_token_type_ids[500:780] = 2
    chunks = plan_gemma4_atomic_prefill_chunks(
        900,
        mm_token_type_ids,
        prefill_chunk_length=320,
    )
    assert [(chunk.start, chunk.end) for chunk in chunks] == [
        (0, 100),
        (100, 420),
        (420, 500),
        (500, 820),
        (820, 900),
    ]

    processor = _make_series_processor(
        window=window,
        query=320,
        bidirectional=True,
        max_ranges=2,
    )
    staged_ranges = []
    for chunk in chunks:
        _, _, ranges = processor._build_compact_attention_metadata(
            current_input_length=chunk.end - chunk.start,
            past_seq_length=chunk.start,
            mm_token_type_ids=mm_token_type_ids[chunk.start : chunk.end],
            device=torch.device("cpu"),
        )
        staged_ranges.append([(start, end) for start, end in ranges[0].tolist() if (start, end) != (0, 0)])

    assert staged_ranges == [
        [],
        [(100, 379)],
        [],
        [(500, 779)],
        [],
    ]


def _one_visual_span(query: int, current: int, selector: int):
    import torch

    mm_token_type_ids = torch.zeros(query, dtype=torch.long)
    span_length = min(3, current)
    placements = (0, max(0, (current - span_length) // 2), current - span_length)
    start = placements[selector % len(placements)]
    mm_token_type_ids[start : start + span_length] = 1
    return mm_token_type_ids


def _derive_compact_visibility(
    *,
    query_capacity: int,
    physical_capacity: int,
    sliding_window: int,
    current_input_length: int,
    query_start_abs: int,
    kv_window_start_abs,
    kv_valid_length,
    mm_prefix_ranges,
):
    """Independent CPU oracle for the compact absolute-position contract."""
    import torch

    query_local = torch.arange(query_capacity, dtype=torch.int64).view(-1, 1)
    key_local = torch.arange(physical_capacity, dtype=torch.int64).view(1, -1)
    query_abs = int(query_start_abs) + query_local
    key_abs = int(kv_window_start_abs.item()) + key_local

    visible = (key_abs <= query_abs) & (query_abs - key_abs < sliding_window)
    for range_start, range_end in mm_prefix_ranges[0].tolist():
        if (range_start, range_end) == (0, 0):
            continue
        query_in_range = (query_abs >= range_start) & (query_abs <= range_end)
        key_in_range = (key_abs >= range_start) & (key_abs <= range_end)
        visible |= query_in_range & key_in_range

    visible &= query_local < current_input_length
    visible &= key_local < int(kv_valid_length.item())
    visible[current_input_length:, 0] = True
    return visible.unsqueeze(0)


def test_compact_metadata_matches_dense_oracle_exhaustively():
    import torch

    compared_entries = 0
    for window in range(1, 9):
        for query in range(1, 9):
            processor = _make_series_processor(
                window=window,
                query=query,
                bidirectional=True,
                max_ranges=1,
            )
            physical_capacity = processor._aligned(window + query - 1, 16)
            for current in range(1, query + 1):
                for past in range(25):
                    mm_token_type_ids = _one_visual_span(
                        query,
                        current,
                        window + query + current + past,
                    )
                    start, valid, ranges = processor._build_compact_attention_metadata(
                        current,
                        past,
                        mm_token_type_ids,
                        torch.device("cpu"),
                    )
                    _, dense_sliding_mask = processor._build_attention_masks(
                        current_input_length=current,
                        past_seq_length=past,
                        mm_token_type_ids=mm_token_type_ids,
                        device=torch.device("cpu"),
                    )
                    compact_visibility = _derive_compact_visibility(
                        query_capacity=query,
                        physical_capacity=physical_capacity,
                        sliding_window=window,
                        current_input_length=current,
                        query_start_abs=past,
                        kv_window_start_abs=start,
                        kv_valid_length=valid,
                        mm_prefix_ranges=ranges,
                    )
                    dense_visibility = dense_sliding_mask[:, 0] == 0
                    assert torch.equal(compact_visibility, dense_visibility), (
                        window,
                        query,
                        current,
                        past,
                        ranges.tolist(),
                    )
                    compared_entries += dense_visibility.numel()

    assert compared_entries >= 200_800


def test_contract_v2_forward_emits_compact_metadata_without_dense_mask(monkeypatch):
    import torch

    processor = _make_series_processor(
        window=8,
        query=8,
        bidirectional=True,
        max_ranges=2,
        contract_version=2,
    )

    def fail_dense_oracle(*args, **kwargs):
        raise AssertionError("contract-v2 forward must not build a dense attention mask")

    monkeypatch.setattr(processor, "_build_attention_masks", fail_dense_oracle)
    output = processor(
        {
            "input_ids": torch.tensor([[2, 3, 4, 5, 0]], dtype=torch.long),
            "past_seq_length": 20,
            "mm_token_type_ids": torch.tensor([[0, 1, 1, 0, 0]], dtype=torch.long),
        }
    )

    assert len(output) == 8
    assert output[1].dtype == output[2].dtype == torch.int32
    assert output[3].dtype == torch.int32
    assert output[3].tolist() == [[[21, 22], [0, 0]]]
    assert output[4].dtype == output[5].dtype == torch.int64
    assert output[4].tolist() == [13]
    assert output[5].tolist() == [12]
    assert output[-2:] == ([], [])


def test_contract_v2_nonvisual_forward_omits_visual_ranges(monkeypatch):
    import torch

    processor = _make_series_processor(
        window=8,
        query=8,
        bidirectional=False,
        max_ranges=2,
        contract_version=2,
    )
    monkeypatch.setattr(
        processor,
        "_build_attention_masks",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("contract-v2 forward must not build a dense attention mask")
        ),
    )
    output = processor(
        {
            "input_ids": torch.tensor([[2, 3]], dtype=torch.long),
            "past_seq_length": 0,
        }
    )

    assert len(output) == 7
    assert output[3].dtype == output[4].dtype == torch.int64
    assert output[3].tolist() == [0]
    assert output[4].tolist() == [2]


def test_contract_v1_forward_retains_dense_tuple_semantics(monkeypatch):
    import torch

    processor = _make_series_processor(
        window=8,
        query=8,
        bidirectional=True,
        contract_version=1,
    )
    original = processor._build_attention_masks
    calls = []

    def record_dense_oracle(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(processor, "_build_attention_masks", record_dense_oracle)
    output = processor(
        {
            "input_ids": torch.tensor([[2, 3]], dtype=torch.long),
            "past_seq_length": 4,
            "mm_token_type_ids": torch.tensor([[1, 1]], dtype=torch.long),
        }
    )

    assert len(calls) == 1
    assert len(output) == 6
    assert output[3].dtype == torch.float16
    assert output[3].shape == (1, 1, 8, 16)
    assert output[-2:] == ([], [])


def test_compact_metadata_rejects_invalid_static_or_current_lengths():
    import torch

    processor = _make_series_processor(
        window=8,
        query=8,
        bidirectional=True,
    )
    with pytest.raises(ValueError, match="current_input_length"):
        processor._build_compact_attention_metadata(
            9,
            0,
            torch.zeros(9, dtype=torch.long),
            torch.device("cpu"),
        )

    processor.sliding_window = 0
    with pytest.raises(ValueError, match="sliding_window must be positive"):
        processor._build_compact_attention_metadata(
            1,
            0,
            torch.zeros(1, dtype=torch.long),
            torch.device("cpu"),
        )


def _series_model_stub(
    *,
    version: int,
    bidirectional: bool,
    ple: bool,
    window: int = 512,
):
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        XHGemma4SeriesModel,
    )

    model = object.__new__(XHGemma4SeriesModel)
    model.config = SimpleNamespace(
        attention_contract_version=version,
        bidirectional_vision_attention=bidirectional,
        hidden_size_per_layer_input=256 if ple else 0,
        spec_decode_mode=None,
        enable_mtp_outputs=False,
        variant="31b",
        capabilities={},
        prefill_chunk_length=320,
        sliding_kv_cache_input_mode="slice_window",
        max_mm_ranges_per_chunk=1,
        context_max_length=2048,
    )
    model.per_layer_input_embedding = object() if ple else None
    model._kvcache_config = SimpleNamespace(num_layers=1)
    model._llm_prefill = True
    model.layer_types = ["sliding_attention"]
    model.layer_cache_types = ["sliding_attention"]
    model.layer_cache_indices = [0]
    model.layer_cache_owner_indices = [0]
    model.sliding_window = window
    model._kvcache_mixin = SimpleNamespace(layer_kv_shapes=[[1, 2, ((window + 320 + 15) // 16) * 16, 128]])
    return model


def test_v2_export_input_order_without_dense_mask():
    model = _series_model_stub(version=2, bidirectional=True, ple=True)

    names = model.get_export_cfg()["input_names"]

    assert names[:7] == [
        "inputs_embeds",
        "past_seq_length",
        "current_input_length",
        "mm_prefix_ranges",
        "kv_window_start_abs",
        "kv_valid_length",
        "per_layer_inputs",
    ]
    assert "sliding_attention_mask" not in names
    assert names[7:9] == ["past_key_cache_0", "past_value_cache_0"]


def test_v2_nonvisual_input_order_omits_only_mm_ranges():
    model = _series_model_stub(version=2, bidirectional=False, ple=False)

    names = model.get_export_cfg()["input_names"]

    assert names[:5] == [
        "inputs_embeds",
        "past_seq_length",
        "current_input_length",
        "kv_window_start_abs",
        "kv_valid_length",
    ]
    assert "mm_prefix_ranges" not in names
    assert names[5:] == ["past_key_cache_0", "past_value_cache_0"]


def test_v1_input_order_retains_dense_mask():
    model = _series_model_stub(version=1, bidirectional=True, ple=True)

    names = model.get_export_cfg()["input_names"]

    assert names[:5] == [
        "inputs_embeds",
        "past_seq_length",
        "current_input_length",
        "sliding_attention_mask",
        "per_layer_inputs",
    ]
    assert "kv_window_start_abs" not in names
    assert "kv_valid_length" not in names


def test_v2_metadata_is_complete(tmp_path):
    model = _series_model_stub(
        version=2,
        bidirectional=True,
        ple=False,
        window=512,
    )
    meta = SimpleNamespace()

    model._extra_export_metadata(str(tmp_path), meta)

    assert meta.attention_contract_version == 2
    assert meta.uses_sliding_flash_attention_v2 is True
    assert meta.sliding_window == 512
    assert meta.prefill_chunk_length == 320
    assert meta.sliding_kv_cache_input_mode == "slice_window"
    assert meta.bidirectional_vision_attention is True
    assert meta.max_mm_ranges_per_chunk == 1
    assert meta.layer_types == ["sliding_attention"]
    assert meta.layer_cache_types == ["sliding_attention"]
    assert meta.layer_cache_indices == [0]
    assert meta.layer_cache_owner_indices == [0]
    assert meta.layer_kv_shapes == [[1, 2, 832, 128]]


def test_export_metadata_keeps_visibility_semantics_independent_of_lowering(tmp_path):
    metas = []
    for version in (1, 2):
        model = _series_model_stub(
            version=version,
            bidirectional=True,
            ple=False,
            window=512,
        )
        meta = SimpleNamespace()
        model._extra_export_metadata(str(tmp_path / str(version)), meta)
        metas.append(meta)

    assert metas[0].attention_lowering == "legacy_attention"
    assert metas[1].attention_lowering == "flash_attention"
    assert metas[0].attention_visibility_spec == metas[1].attention_visibility_spec
    assert metas[0].attention_visibility_spec["requires_kv_window_metadata"] is True
    assert metas[0].attention_visibility_spec["requires_mm_prefix_ranges"] is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda model: setattr(model, "layer_types", ["linear_attention"]), "layer_types"),
        (lambda model: setattr(model, "sliding_window", 0), "sliding_window"),
        (lambda model: setattr(model.config, "prefill_chunk_length", 279), "prefill_chunk_length"),
        (
            lambda model: setattr(
                model._kvcache_mixin,
                "layer_kv_shapes",
                [[1, 2, 512, 128]],
            ),
            "layer_kv_shapes",
        ),
    ],
)
def test_v2_metadata_rejects_inconsistent_contract(tmp_path, mutation, message):
    model = _series_model_stub(version=2, bidirectional=True, ple=False)
    mutation(model)

    with pytest.raises(ValueError, match=message):
        model._extra_export_metadata(str(tmp_path), SimpleNamespace())


class _BridgeLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []
        self.embedding = torch.nn.Embedding(8, 4)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, **kwargs):
        self.calls.append(kwargs)
        # Production wrapping replaces the HF text model with
        # _Gemma4TextModel, whose export contract is a tensor rather than a
        # transformers ModelOutput.
        return kwargs["inputs_embeds"]


def _bridge_hf_model(*, ple: bool):
    language_model = _BridgeLanguageModel()
    text_config = SimpleNamespace(
        hidden_size_per_layer_input=4 if ple else 0,
        final_logit_softcapping=None,
    )
    config = SimpleNamespace(
        text_config=text_config,
        get_text_config=lambda: text_config,
    )
    return (
        SimpleNamespace(
            config=config,
            model=SimpleNamespace(language_model=language_model),
            lm_head=torch.nn.Identity(),
        ),
        language_model,
    )


def test_v2_bridge_forwards_compact_metadata_without_dense_mask():
    from xhmodel_merak.xh_llm.models.gemma4_series.llm_text import (
        _Gemma4FlashAttentionBridge,
        _make_text_export_bridge_if_needed,
    )

    hf_model, language_model = _bridge_hf_model(ple=False)
    bridge = _make_text_export_bridge_if_needed(
        hf_model,
        attention_contract_version=2,
        bidirectional_vision_attention=True,
    )
    inputs_embeds = torch.randn(1, 2, 4)
    past = torch.tensor([9], dtype=torch.int32)
    current = torch.tensor([2], dtype=torch.int32)
    mm = torch.tensor([[[9, 10]]], dtype=torch.int32)
    base = torch.tensor([6], dtype=torch.int64)
    valid = torch.tensor([5], dtype=torch.int64)

    output = bridge(inputs_embeds, past, current, mm, base, valid, [], [])

    assert isinstance(bridge, _Gemma4FlashAttentionBridge)
    assert output is inputs_embeds
    call = language_model.calls[-1]
    assert call["sliding_attention_mask"] is None
    assert call["mm_prefix_ranges"] is mm
    assert call["kv_window_start_abs"] is base
    assert call["kv_valid_length"] is valid


def test_v2_bridge_treats_wrapped_language_model_output_as_tensor_during_fx_trace():
    from xhmodel_merak.xh_llm.models.gemma4_series.llm_text import (
        _make_text_export_bridge_if_needed,
    )

    class _LeafLanguageModelTracer(torch.fx.Tracer):
        def is_leaf_module(self, module, module_qualified_name):
            if isinstance(module, _BridgeLanguageModel):
                return True
            return super().is_leaf_module(module, module_qualified_name)

    hf_model, _ = _bridge_hf_model(ple=False)
    bridge = _make_text_export_bridge_if_needed(
        hf_model,
        attention_contract_version=2,
        bidirectional_vision_attention=True,
    )

    graph = _LeafLanguageModelTracer().trace(bridge)

    assert bridge.language_model_returns_tensor is True
    assert not any(
        node.op == "call_function"
        and node.target is getattr
        and len(node.args) > 1
        and node.args[1] == "last_hidden_state"
        for node in graph.nodes
    )


def test_v2_bridge_factory_has_static_nonvisual_and_ple_variants():
    from xhmodel_merak.xh_llm.models.gemma4_series.llm_text import (
        _Gemma4FlashAttentionBridgeNoMM,
        _Gemma4FlashAttentionBridgePLENoMM,
        _Gemma4TextExportBridgePLE,
        _make_text_export_bridge_if_needed,
    )

    hf_model, _ = _bridge_hf_model(ple=False)
    nonvisual = _make_text_export_bridge_if_needed(
        hf_model,
        attention_contract_version=2,
        bidirectional_vision_attention=False,
    )
    assert isinstance(nonvisual, _Gemma4FlashAttentionBridgeNoMM)

    ple_model, _ = _bridge_hf_model(ple=True)
    compact_ple = _make_text_export_bridge_if_needed(
        ple_model,
        attention_contract_version=2,
        bidirectional_vision_attention=False,
    )
    legacy_ple = _make_text_export_bridge_if_needed(
        ple_model,
        attention_contract_version=1,
        bidirectional_vision_attention=False,
    )
    assert isinstance(compact_ple, _Gemma4FlashAttentionBridgePLENoMM)
    assert isinstance(legacy_ple, _Gemma4TextExportBridgePLE)


class _CapturedFlash(torch.nn.Module):
    def __init__(self, *args, sliding_window=None, **kwargs):
        super().__init__()
        self.constructor_args = args
        self.constructor_kwargs = kwargs
        self.sliding_window_size = sliding_window
        self.scale = kwargs.get("scale")
        self.calls = []

    def forward(self, query, key, value, **kwargs):
        self.calls.append({"query": query, "key": key, "value": value, **kwargs})
        return query.transpose(1, 2).contiguous()


class _IdentityRope(torch.nn.Module):
    def forward(self, x, *position_embeddings):
        return x


def _setup_tiny_attention(
    monkeypatch,
    *,
    layer_type: str,
    window: int | None,
    version: int,
    use_cache: bool = False,
    head_dim: int = 4,
    flash_attention: dict | None = None,
):
    from xhmodel_merak.xh_llm.models.gemma4_series import _llm_model_impl as impl

    monkeypatch.setattr(impl.xhnn, "FlashAttention", _CapturedFlash)
    attention = object.__new__(impl._Gemma4TextAttention)
    torch.nn.Module.__init__(attention)
    attention.config = SimpleNamespace(num_key_value_heads=1)
    attention.layer_type = layer_type
    attention.sliding_window = window
    attention.head_dim = head_dim
    attention.q_proj = torch.nn.Linear(8, head_dim * 2, bias=False)
    attention.k_proj = torch.nn.Linear(8, head_dim, bias=False)
    attention.v_proj = torch.nn.Linear(8, head_dim, bias=False)
    attention.o_proj = torch.nn.Linear(head_dim * 2, 8, bias=False)
    attention.q_norm = torch.nn.Identity()
    attention.k_norm = torch.nn.Identity()
    attention.v_norm = torch.nn.Identity()
    attention.is_kv_shared_layer = False
    attention.store_full_length_kv = False
    attention._setup(
        SimpleNamespace(
            attention_contract_version=version,
            flash_attention=flash_attention,
            use_cache=use_cache,
            kv_cache=SimpleNamespace(cache_axis=2),
        )
    )
    attention.rope = _IdentityRope()
    attention.attn_compute_cast = torch.nn.Identity()
    attention.attn_output_cast = torch.nn.Identity()
    return attention


def test_v2_attention_builds_sliding_and_full_flash(monkeypatch):
    sliding = _setup_tiny_attention(
        monkeypatch,
        layer_type="sliding_attention",
        window=512,
        version=2,
    )
    full = _setup_tiny_attention(
        monkeypatch,
        layer_type="full_attention",
        window=None,
        version=2,
    )

    assert sliding.use_flash_attention_v2 is True
    assert full.use_flash_attention_v2 is True
    assert sliding.flash_attn.sliding_window_size == 512
    assert full.flash_attn.sliding_window_size is None
    assert not hasattr(sliding, "qk_matmul")
    assert not hasattr(full, "qk_matmul")


@pytest.mark.parametrize("head_dim", [256, 512])
@pytest.mark.parametrize(
    ("layer_type", "window"),
    [("sliding_attention", 512), ("full_attention", None)],
)
def test_v2_flash_attention_preserves_gemma4_unscaled_qk_contract(monkeypatch, head_dim, layer_type, window):
    attention = _setup_tiny_attention(
        monkeypatch,
        layer_type=layer_type,
        window=window,
        version=2,
        head_dim=head_dim,
    )

    # Transformers Gemma4 normalizes Q/K and then explicitly passes
    # self.scaling=1.0 to attention.  FlashAttention must not apply the usual
    # 1/sqrt(head_dim) scale a second time.
    assert attention.flash_attn.scale == 1.0


@pytest.mark.parametrize("value", [8, 16])
def test_v2_flash_attention_uses_keyword_api_and_passes_all_precision_bits(monkeypatch, value):
    attention = _setup_tiny_attention(
        monkeypatch,
        layer_type="full_attention",
        window=None,
        version=2,
        flash_attention={name: value for name in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")},
    )

    assert attention.flash_attn.constructor_args == ()
    assert attention.flash_attn.constructor_kwargs["num_heads"] == 2
    assert attention.flash_attn.constructor_kwargs["num_kv_heads"] == 1
    assert tuple(
        attention.flash_attn.constructor_kwargs[name]
        for name in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")
    ) == (value,) * 5


def test_v2_flash_attention_defaults_all_precision_bits_to_eight(monkeypatch):
    attention = _setup_tiny_attention(
        monkeypatch,
        layer_type="full_attention",
        window=None,
        version=2,
    )

    assert tuple(
        attention.flash_attn.constructor_kwargs[name]
        for name in ("q_bits", "k_bits", "v_bits", "s_bits", "p_bits")
    ) == (8,) * 5


@pytest.mark.parametrize("field", ["q_bits", "k_bits", "v_bits", "s_bits", "p_bits"])
def test_v2_flash_attention_rejects_invalid_precision_bit(monkeypatch, field):
    with pytest.raises(ValueError, match=rf"{field}=12"):
        _setup_tiny_attention(
            monkeypatch,
            layer_type="full_attention",
            window=None,
            version=2,
            flash_attention={field: 12},
        )


def test_all_flash_layers_receive_visual_ranges_but_only_sliding_receives_compact_window_metadata(monkeypatch):
    sliding = _setup_tiny_attention(
        monkeypatch,
        layer_type="sliding_attention",
        window=512,
        version=2,
    )
    full = _setup_tiny_attention(
        monkeypatch,
        layer_type="full_attention",
        window=None,
        version=2,
    )
    hidden_states = torch.randn(1, 2, 8, dtype=torch.float32)
    position_embeddings = (
        torch.ones(1, 1, 2, 4, dtype=torch.float32),
        torch.zeros(1, 1, 2, 4, dtype=torch.float32),
    )
    past = torch.tensor([9], dtype=torch.int32)
    current = torch.tensor([2], dtype=torch.int32)
    mm_ranges = torch.tensor([[[9, 10]]], dtype=torch.int32)
    start = torch.tensor([6], dtype=torch.int64)
    valid = torch.tensor([5], dtype=torch.int64)

    sliding_output, sliding_weights = sliding(
        hidden_states,
        position_embeddings,
        attention_mask=torch.ones(1, 1, 2, 16),
        past_seq_length=past,
        current_input_length=current,
        mm_prefix_ranges=mm_ranges,
        kv_window_start_abs=start,
        kv_valid_length=valid,
    )
    full_output, full_weights = full(
        hidden_states,
        position_embeddings,
        attention_mask=torch.ones(1, 1, 2, 16),
        past_seq_length=past,
        current_input_length=current,
        mm_prefix_ranges=mm_ranges,
        kv_window_start_abs=start,
        kv_valid_length=valid,
    )

    sliding_call = sliding.flash_attn.calls[-1]
    full_call = full.flash_attn.calls[-1]
    assert sliding_output.shape == full_output.shape == (1, 2, 8)
    assert sliding_weights is full_weights is None
    assert sliding_call["mm_prefix_range"] is mm_ranges
    assert sliding_call["kv_window_start_abs"] is start
    assert sliding_call["kv_valid_length"] is valid
    assert full_call["mm_prefix_range"] is mm_ranges
    assert full_call["kv_window_start_abs"] is None
    assert full_call["kv_valid_length"] is None
    assert sliding_call["past_seq_length"] is past
    assert sliding_call["current_input_length"] is current
    assert sliding_call["query"].shape[1] == 2
    assert sliding_call["key"].shape[1] == sliding_call["value"].shape[1] == 1


def test_v1_attention_retains_dense_matmul_softmax_chain(monkeypatch):
    legacy = _setup_tiny_attention(
        monkeypatch,
        layer_type="sliding_attention",
        window=512,
        version=1,
    )

    assert legacy.use_flash_attention_v2 is False
    assert not hasattr(legacy, "flash_attn")
    for name in (
        "k_repeat_interleave",
        "v_repeat_interleave",
        "qk_matmul",
        "masked_add",
        "softmax",
        "masked_softmax",
        "pv_matmul",
    ):
        assert hasattr(legacy, name)


@pytest.mark.parametrize(
    ("window", "expected_capacity"),
    [(512, 832), (1024, 1344)],
    ids=["window_512", "window_1024"],
)
def test_v2_flash_preserves_sliding_cache_capacity(monkeypatch, window, expected_capacity):
    from torch._subclasses.fake_tensor import FakeTensorMode

    attention = _setup_tiny_attention(
        monkeypatch,
        layer_type="sliding_attention",
        window=window,
        version=2,
        use_cache=True,
    )
    with FakeTensorMode():
        states = torch.empty(1, 1, 320, 4, dtype=torch.float16)
        past_cache = torch.empty(1, 1, expected_capacity, 4, dtype=torch.float16)
        output = attention.k_cache(
            states,
            torch.tensor([4096], dtype=torch.int64),
            torch.tensor([320], dtype=torch.int64),
            past_cache,
        )

    assert attention.k_cache.attention_max_length == window
    assert attention.v_cache.attention_max_length == window
    assert output.shape == (1, 1, expected_capacity, 4)


def test_v2_attention_rejects_unknown_layer_type(monkeypatch):
    with pytest.raises(ValueError, match="layer_type"):
        _setup_tiny_attention(
            monkeypatch,
            layer_type="unknown_attention",
            window=512,
            version=2,
        )


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize(
    ("layer_type", "window", "message"),
    [
        ("sliding_attention", None, "sliding_attention.*positive integer"),
        ("full_attention", 512, "full_attention.*sliding_window.*None"),
    ],
)
def test_attention_rejects_layer_type_window_mismatch_independent_of_lowering(
    monkeypatch, version, layer_type, window, message
):
    with pytest.raises(ValueError, match=message):
        _setup_tiny_attention(
            monkeypatch,
            layer_type=layer_type,
            window=window,
            version=version,
        )


class _CapturedSelfAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, hidden_states, **kwargs):
        self.calls.append(kwargs)
        return hidden_states, None


class _Zero(torch.nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


def test_decoder_flash_metadata_propagates_to_attention_by_identity():
    from xhmodel_merak.xh_llm.models.gemma4_series._llm_model_impl import _Gemma4TextDecoderLayer

    decoder = object.__new__(_Gemma4TextDecoderLayer)
    torch.nn.Module.__init__(decoder)
    decoder.input_layernorm = torch.nn.Identity()
    decoder.self_attn = _CapturedSelfAttention()
    decoder.post_attention_layernorm = torch.nn.Identity()
    decoder.pre_feedforward_layernorm = torch.nn.Identity()
    decoder.mlp = _Zero()
    decoder.post_feedforward_layernorm = torch.nn.Identity()
    decoder.enable_moe_block = False
    decoder.hidden_size_per_layer_input = 0
    hidden_states = torch.randn(1, 2, 8)
    mm_ranges = torch.tensor([[[9, 10]]], dtype=torch.int32)
    start = torch.tensor([6], dtype=torch.int64)
    valid = torch.tensor([5], dtype=torch.int64)

    output = decoder(
        hidden_states,
        position_embeddings=(torch.empty(0), torch.empty(0)),
        past_seq_length=torch.tensor([9], dtype=torch.int32),
        current_input_length=torch.tensor([2], dtype=torch.int32),
        mm_prefix_ranges=mm_ranges,
        kv_window_start_abs=start,
        kv_valid_length=valid,
    )

    call = decoder.self_attn.calls[-1]
    assert output.shape == hidden_states.shape
    assert call["mm_prefix_ranges"] is mm_ranges
    assert call["kv_window_start_abs"] is start
    assert call["kv_valid_length"] is valid


class _PassSlice(torch.nn.Module):
    def forward(self, cache, past_seq_length):
        return cache


class _CapturedDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = SimpleNamespace(is_kv_shared_layer=False)
        self.calls = []

    def forward(self, hidden_states, **kwargs):
        self.calls.append(kwargs)
        return hidden_states


def test_text_model_flash_metadata_reaches_every_decoder_layer_by_identity():
    from xhmodel_merak.xh_llm.models.gemma4_series._llm_model_impl import _Gemma4TextModel

    model = object.__new__(_Gemma4TextModel)
    torch.nn.Module.__init__(model)
    sliding_decoder = _CapturedDecoder()
    full_decoder = _CapturedDecoder()
    model.layers = torch.nn.ModuleList([sliding_decoder, full_decoder])
    model.config = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"],
        num_hidden_layers=2,
    )
    model.hidden_size_per_layer_input = 0
    model.only_first_block = False
    model.norm = torch.nn.Identity()
    model.num_logits_to_keep = 0
    model.enable_mtp_outputs = False
    for layer_type in model.config.layer_types:
        setattr(model, f"_{layer_type}_cos_cache", torch.zeros(1, 1, 2, 4))
        setattr(model, f"_{layer_type}_sin_cache", torch.zeros(1, 1, 2, 4))
        setattr(model, f"_{layer_type}_cos_slice", _PassSlice())
        setattr(model, f"_{layer_type}_sin_slice", _PassSlice())
    inputs_embeds = torch.randn(1, 2, 8)
    past = torch.tensor([9], dtype=torch.int32)
    current = torch.tensor([2], dtype=torch.int32)
    mm_ranges = torch.tensor([[[9, 10]]], dtype=torch.int32)
    start = torch.tensor([6], dtype=torch.int64)
    valid = torch.tensor([5], dtype=torch.int64)
    local_mask = torch.empty(1)
    full_mask = torch.empty(1)

    output = model(
        inputs_embeds=inputs_embeds,
        past_seq_length=past,
        current_input_length=current,
        local_attention_mask=local_mask,
        full_attention_mask=full_mask,
        mm_prefix_ranges=mm_ranges,
        kv_window_start_abs=start,
        kv_valid_length=valid,
    )

    assert output.shape == inputs_embeds.shape
    for decoder in (sliding_decoder, full_decoder):
        call = decoder.calls[-1]
        assert call["mm_prefix_ranges"] is mm_ranges
        assert call["kv_window_start_abs"] is start
        assert call["kv_valid_length"] is valid
    assert sliding_decoder.calls[-1]["attention_mask"] is local_mask
    assert full_decoder.calls[-1]["attention_mask"] is full_mask
