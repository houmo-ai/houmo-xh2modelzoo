from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest
import torch
from torch import nn

from xhmodel_merak.xh_llm.types import CacheList


def test_gemma4_series_ple_projection_is_fx_traceable():
    from torch.fx import symbolic_trace

    from xhmodel_merak.xh_llm.models.gemma4_series._llm_model_impl import _Gemma4TextModel

    class TinyPleProjection(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(num_hidden_layers=2)
            self.hidden_size_per_layer_input = 3
            self.per_layer_model_projection = nn.Linear(4, 6, bias=False)
            self.register_buffer("per_layer_model_projection_scale", torch.tensor(1.0))
            self.per_layer_projection_norm = nn.Identity()
            self.register_buffer("per_layer_embed_scale", torch.tensor(1.0))
            self.register_buffer("per_layer_input_scale", torch.tensor(1.0))
            self.project_per_layer_inputs = MethodType(_Gemma4TextModel._project_per_layer_inputs, self)

        def forward(self, inputs_embeds, per_layer_inputs):
            return self.project_per_layer_inputs(inputs_embeds, per_layer_inputs)

    traced = symbolic_trace(TinyPleProjection())
    out = traced(torch.ones(1, 5, 4), torch.ones(1, 5, 2, 3))
    assert out.shape == (1, 2, 5, 3)


def test_gemma4_series_preprocess_places_ple_before_kv_caches():
    from xhmodel_merak.xh_llm.models.gemma4_series.data_preprocess import (
        Gemma4DataPreprocess,
        Gemma4PerLayerInputEmbedding,
    )

    token_embedding = nn.Embedding(16, 4)
    per_layer_embedding = Gemma4PerLayerInputEmbedding(
        vocab_size_per_layer_input=16,
        num_hidden_layers=2,
        hidden_size_per_layer_input=3,
        pad_token_id=0,
    )
    preprocess = Gemma4DataPreprocess(
        token_embedding=token_embedding,
        input_sequence_length=4,
        context_length=8,
        past_key_caches=CacheList([torch.zeros((1, 1, 8, 4), dtype=torch.float16)]),
        past_value_caches=CacheList([torch.zeros((1, 1, 8, 4), dtype=torch.float16)]),
        pad_token_id=0,
        per_layer_input_embedding=per_layer_embedding,
        sliding_window=3,
    )

    outputs = preprocess({"input_ids": torch.tensor([[1, 2]], dtype=torch.long), "past_seq_length": 0})

    (
        inputs_embeds,
        past_seq_length,
        current_input_length,
        sliding_mask,
        per_layer_inputs,
        key_cache,
        value_cache,
    ) = outputs
    assert inputs_embeds.shape == (1, 4, 4)
    assert past_seq_length.item() == 0
    assert current_input_length.item() == 2
    assert sliding_mask.shape == (1, 1, 4, 16)
    assert per_layer_inputs.shape == (1, 4, 2, 3)
    assert key_cache is preprocess.past_key_caches
    assert value_cache is preprocess.past_value_caches


def test_gemma4_series_bidirectional_vision_mask_is_config_gated():
    from xhmodel_merak.xh_llm.models.gemma4_series.data_preprocess import Gemma4DataPreprocess

    token_embedding = nn.Embedding(16, 4)

    def build(enabled: bool):
        preprocess = Gemma4DataPreprocess(
            token_embedding=token_embedding,
            input_sequence_length=4,
            context_length=8,
            pad_token_id=0,
            image_token_id=9,
            sliding_window=3,
            bidirectional_vision_attention=enabled,
        )
        return preprocess._build_attention_masks(
            current_input_length=3,
            past_seq_length=0,
            mm_token_type_ids=torch.tensor([0, 1, 1, 0], dtype=torch.long),
            device=torch.device("cpu"),
        )[0]

    causal_full = build(False)
    bidir_full = build(True)
    neg = torch.finfo(torch.float16).min

    # Query at the first visual token must not see the next visual token for E4B.
    assert causal_full[0, 0, 1, 2].item() == neg
    # 31B/26B opt into HF's vision bidirectional attention and do see it.
    assert bidir_full[0, 0, 1, 2].item() == 0


def test_gemma4_series_e4b_bridge_uses_masked_softmax_for_full_layers():
    from pathlib import Path

    bridge_src = Path("xhmodel_merak/xh_llm/models/gemma4_series/llm_text.py").read_text(encoding="utf-8")
    llm_src = Path("xhmodel_merak/xh_llm/models/gemma4_series/_llm_model_impl.py").read_text(encoding="utf-8")
    assert "full_attention_mask=None" in bridge_src
    assert (
        'layer_attention_mask = local_attention_mask if layer_type == "sliding_attention" else full_attention_mask'
        in llm_src
    )
    assert "attn_weights = self.masked_softmax(attn_weights, past_seq_length)" in llm_src


def test_gemma4_series_export_cfg_places_ple_before_kv_cache_names():
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import XHGemma4SeriesModel

    model = object.__new__(XHGemma4SeriesModel)
    model.per_layer_input_embedding = object()
    model.config = SimpleNamespace(hidden_size_per_layer_input=256, bidirectional_vision_attention=False)
    model._kvcache_config = SimpleNamespace(num_layers=2)

    cfg = XHGemma4SeriesModel.get_export_cfg(model)
    assert cfg["input_names"][:5] == [
        "inputs_embeds",
        "past_seq_length",
        "current_input_length",
        "sliding_attention_mask",
        "per_layer_inputs",
    ]
    assert cfg["input_names"][5:] == [
        "past_key_cache_0",
        "past_key_cache_1",
        "past_value_cache_0",
        "past_value_cache_1",
    ]

    model.per_layer_input_embedding = None
    model.config = SimpleNamespace(hidden_size_per_layer_input=0, bidirectional_vision_attention=True)
    cfg = XHGemma4SeriesModel.get_export_cfg(model)
    assert cfg["input_names"][:5] == [
        "inputs_embeds",
        "past_seq_length",
        "current_input_length",
        "full_attention_mask",
        "sliding_attention_mask",
    ]


def test_gemma4_series_audio_attention_mask_is_external_additive_mask():
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_processor import XHGemma4Processor

    feature_mask = torch.zeros((1, 400), dtype=torch.float16)
    feature_mask[:, :99] = 1
    attention_mask = XHGemma4Processor.build_audio_attention_mask(
        feature_mask,
        chunk_size=12,
        context_left=13,
        context_right=0,
        dtype=torch.float16,
    )

    assert attention_mask.dtype == torch.float16
    assert attention_mask.shape == (1, 1, 9, 12, 24)
    assert attention_mask[0, 0, 0, 0, 12].item() == 0
    assert attention_mask[0, 0, 0, 0, 13].item() == torch.finfo(torch.float16).min
    assert attention_mask.min().item() == torch.finfo(torch.float16).min


def test_gemma4_series_audio_submodel_uses_three_input_contract():
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_audio_model import XHGemma4SeriesAudioModel
    from xhmodel_merak.xh_llm.models.gemma4_series.xh_gemma4_series_config import XHGemma4SeriesAudioConfig

    model_dir = Path("/data01/datasets/gemma-4-E4B-it")
    if not model_dir.exists():
        pytest.skip("Gemma4 E4B local HF config is not available")

    config = XHGemma4SeriesAudioConfig(
        model_name="gemma4_audio",
        hf_model=str(model_dir),
        input_feature_length=400,
    )
    model = XHGemma4SeriesAudioModel(config)

    dummy_inputs = model.get_dummy_inputs()
    assert set(dummy_inputs) == {"input_features", "input_features_mask", "audio_attention_mask"}
    assert dummy_inputs["input_features"].shape == (1, 400, 128)
    assert dummy_inputs["input_features_mask"].shape == (1, 400)
    assert dummy_inputs["input_features_mask"].dtype == torch.float16
    assert dummy_inputs["audio_attention_mask"].shape == (1, 1, 9, 12, 24)

    processed_inputs = model.get_data_preprocessor()(dummy_inputs)
    assert len(processed_inputs) == 3
    assert processed_inputs[2].shape == dummy_inputs["audio_attention_mask"].shape

    export_cfg = model.get_export_cfg()
    assert export_cfg["input_names"] == ["input_features", "input_features_mask", "audio_attention_mask"]

    meta = model.create_export_metadata("work_dirs/gemma4_audio_meta")
    assert meta.attention_chunk_size == 12
    assert meta.attention_context_left == 13
    assert meta.attention_context_right == 0


def test_gemma4_series_audio_graph_uses_single_masked_add_source_contract():
    from pathlib import Path

    src = Path("xhmodel_merak/xh_llm/models/gemma4_series/_audio_model_impl.py").read_text(encoding="utf-8")
    assert "masked_add_2" not in src
    assert "attention_mask.logical_not()" not in src
    assert "audio_attention_mask" in src


def test_gemma4_series_processor_builds_pooling_matrix_contract():
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_processor import XHGemma4Processor

    position_grid = torch.tensor(
        [
            [0, 0],
            [1, 0],
            [2, 0],
            [0, 1],
            [1, 1],
            [2, 1],
            [0, 2],
            [1, 2],
            [2, 2],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
            [0, 0],
        ],
        dtype=torch.long,
    )
    valid_mask = torch.zeros(18, dtype=torch.bool)
    valid_mask[:9] = True

    pooling_matrix = XHGemma4Processor._build_pooling_matrix(position_grid, valid_mask, kernel_size=3)

    assert pooling_matrix.shape == (1, 2, 18)
    assert pooling_matrix.dtype == torch.float16
    torch.testing.assert_close(pooling_matrix[0, 0, :9], torch.full((9,), 1 / 9, dtype=torch.float16))
    assert torch.count_nonzero(pooling_matrix[0, 1, :]).item() == 0


def test_gemma4_series_visual_pooling_matrix_matches_gather_mean_math():
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_vision_model import Gemma4VisualAdapter

    adapter = object.__new__(Gemma4VisualAdapter)
    hidden_states = torch.arange(18, dtype=torch.float32).reshape(1, 9, 2)
    pooling_matrix = torch.full((1, 1, 9), 1 / 9, dtype=torch.float16)

    pooled = Gemma4VisualAdapter._pool_by_matrix(adapter, hidden_states, pooling_matrix)

    torch.testing.assert_close(pooled, hidden_states.mean(dim=1, keepdim=True), rtol=0.0, atol=3e-3)


def test_gemma4_series_visual_export_uses_pooling_matrix_not_gather_indices():
    from pathlib import Path

    src = Path("xhmodel_merak/xh_llm/models/gemma4_series/gemma4_series_vision_model.py").read_text(
        encoding="utf-8"
    )
    assert "pooling_matrix" in src
    assert "position_embedding_x" in src
    assert "position_embedding_y" in src
    assert "pool_gather" not in src
    assert "GatherElements" not in src
    assert "pool_indices" not in src


def test_gemma4_series_audio_removes_only_noop_clip_nodes(tmp_path):
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_audio_model import (
        _remove_noop_clip_nodes,
    )

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])
    min_inf = numpy_helper.from_array(np.array([-np.inf], dtype=np.float32), "min_inf")
    max_inf = numpy_helper.from_array(np.array([np.inf], dtype=np.float32), "max_inf")
    min_finite = numpy_helper.from_array(np.array([-1.0], dtype=np.float32), "min_finite")
    max_finite = numpy_helper.from_array(np.array([1.0], dtype=np.float32), "max_finite")
    nodes = [
        helper.make_node("Clip", ["x", "min_inf", "max_inf"], ["noop"], name="noop_clip"),
        helper.make_node("Relu", ["noop"], ["relu"], name="relu_after_noop"),
        helper.make_node("Clip", ["relu", "min_finite", "max_finite"], ["y"], name="finite_clip"),
    ]
    graph = helper.make_graph(
        nodes,
        "noop_clip_graph",
        [x],
        [y],
        [min_inf, max_inf, min_finite, max_finite],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    onnx_path = tmp_path / "clip.onnx"
    onnx.save(model, onnx_path)

    assert _remove_noop_clip_nodes(onnx_path) == 1
    cleaned = onnx.load(onnx_path)

    assert [node.name for node in cleaned.graph.node] == ["relu_after_noop", "finite_clip"]
    assert cleaned.graph.node[0].input[0] == "x"
    assert cleaned.graph.node[1].op_type == "Clip"


def test_gemma4_series_fuses_static_scale_muls_in_source():
    audio_src = Path("xhmodel_merak/xh_llm/models/gemma4_series/_audio_model_impl.py").read_text()
    llm_src = Path("xhmodel_merak/xh_llm/models/gemma4_series/_llm_model_impl.py").read_text()

    assert "query_states = query_states * self.q_per_dim_scale.to(query_states)" in audio_src
    assert "F.softplus(self.per_dim_scale.detach()) * self.q_scale" in audio_src
    assert "Gemma4AudioLightConv1d" in audio_src
    assert "self.linear_start_value(hidden_states)" in audio_src
    assert "nn.functional.glu" not in audio_src
    assert "scaled = normed * moe_scale * self._moe_scalar_root_size" not in llm_src
    assert "self._moe_scalar_root_size =" not in llm_src
    assert "self.moe_router_norm.weight = nn.Parameter(moe_scale)" in llm_src
    assert "scaled = self.moe_router_norm(hidden_states_flat)" in llm_src
    assert "per_layer_projection = self.per_layer_model_projection(inputs_embeds) *" not in llm_src
    assert "return (per_layer_projection + per_layer_inputs) *" not in llm_src
    assert "if layer_scalar is not None and torch.all(layer_scalar == 1):" in llm_src


def test_gemma4_series_no_scale_rmsnorm_uses_fused_module():
    llm_src = Path("xhmodel_merak/xh_llm/models/gemma4_series/_llm_model_impl.py").read_text()

    assert "class _NoScaleRMSNorm" not in llm_src
    assert "torch.pow(mean_sq, -0.5)" not in llm_src
    assert "self.norm = RMSNorm(hidden_size, self.eps)" in llm_src
    assert "requires_grad=False" in llm_src


def test_gemma4_series_autoround_mode1_builds_dense_script_command():
    from xhmodel_merak.xh_llm.models.gemma4_series.quant_adapter import (
        build_autoround_mode1_command,
    )

    model_dir = Path("/data01/datasets/gemma-4-31B-it")
    if not model_dir.exists():
        pytest.skip("Gemma4 31B local HF config is not available")

    command, output_dir, algorithm = build_autoround_mode1_command(
        hf_model_dir=str(model_dir),
        output_dir="work_dirs/gemma4_mode1",
        device="cuda:1",
        workflow_seed=1024,
        export_model_cfg={
            "context_max_length": 2048,
            "prefill_chunk_length": 256,
        },
        quant_cfg={
            "algorithm": "autoround",
            "preset": "mode1",
            "rotation": None,
            "artifact_format": "gptqmodel_hf",
            "bits": 4,
            "group_size": 64,
            "sym": True,
            "iters": 200,
            "format": "auto_gptq",
            "calibration": {
                "dataset": "NeelNanda/pile-10k",
                "nsamples": 128,
                "seqlen": 2048,
            },
            "runtime": {
                "batch_size": 8,
                "seed": 42,
            },
        },
    )

    assert algorithm == "autoround:mode1"
    assert output_dir.endswith("work_dirs/gemma4_mode1")
    assert "scripts_gemma4/quantize.py" in command[1]
    assert command[command.index("--mode") + 1] == "llm-only"
    assert command[command.index("--device") + 1] == "cuda:1"
    assert command[command.index("--llm_bits") + 1] == "4"
    assert command[command.index("--llm_group_size") + 1] == "64"
    assert command[command.index("--nsamples") + 1] == "128"
    assert command[command.index("--seqlen") + 1] == "2048"
    assert command[command.index("--batch_size") + 1] == "8"
    assert command[command.index("--dataset") + 1] == "NeelNanda/pile-10k"
    assert "--sym" in command


def test_gemma4_series_autoround_mode1_has_separate_quant_template(tmp_path):
    import yaml

    from xhmodel_merak.xh_llm.models.gemma4_series.quant_adapter import (
        DEFAULT_DENSE_CALIBRATION_JSONL,
        DEFAULT_MOE_CALIBRATION_JSONL,
    )
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import (
        dump_autoround_moe_mode1_quant_config_template,
        dump_autoround_mode1_quant_config_template,
        dump_quant_config_template,
    )

    default_path = tmp_path / "default_quant.yaml"
    mode1_path = tmp_path / "mode1_quant.yaml"
    moe_mode1_path = tmp_path / "moe_mode1_quant.yaml"

    dump_quant_config_template(default_path)
    dump_autoround_mode1_quant_config_template(mode1_path)
    dump_autoround_moe_mode1_quant_config_template(moe_mode1_path)

    default_quant = yaml.safe_load(default_path.read_text())["quant"]
    mode1_quant = yaml.safe_load(mode1_path.read_text())["quant"]
    moe_mode1_quant = yaml.safe_load(moe_mode1_path.read_text())["quant"]

    assert default_quant["algorithm"] == "gptqmodel"
    assert default_quant["calibration"]["jsonl"] == DEFAULT_DENSE_CALIBRATION_JSONL
    assert "autoround_mode1" not in default_quant
    assert mode1_quant["algorithm"] == "autoround"
    assert mode1_quant["preset"] == "mode1"
    assert mode1_quant["calibration"]["seqlen"] == 512
    assert mode1_quant["runtime"]["batch_size"] == 8
    assert moe_mode1_quant["algorithm"] == "autoround"
    assert moe_mode1_quant["preset"] == "mode1"
    assert moe_mode1_quant["calibration"]["jsonl"] == DEFAULT_MOE_CALIBRATION_JSONL
    assert moe_mode1_quant["runtime"]["dtype"] == "bfloat16"


def test_gemma4_series_autoround_mode1_builds_moe_script_command():
    from xhmodel_merak.xh_llm.models.gemma4_series.quant_adapter import (
        build_autoround_mode1_command,
    )

    model_dir = Path("/data01/datasets/gemma-4-26B-A4B-it")
    if not model_dir.exists():
        pytest.skip("Gemma4 26B-A4B local HF config is not available")

    command, output_dir, algorithm = build_autoround_mode1_command(
        hf_model_dir=str(model_dir),
        output_dir="work_dirs/gemma4_mode1_moe",
        device="cuda:1",
        workflow_seed=1024,
        export_model_cfg={
            "context_max_length": 2048,
            "prefill_chunk_length": 256,
        },
        quant_cfg={
            "algorithm": "autoround",
            "preset": "mode1",
            "artifact_format": "gptqmodel_hf",
            "bits": 4,
            "group_size": 64,
            "sym": True,
            "calibration": {
                "nsamples": 128,
                "seqlen": 2048,
            },
            "runtime": {
                "batch_size": 8,
                "dtype": "bfloat16",
            },
        },
    )

    assert algorithm == "autoround:mode1_moe"
    assert output_dir.endswith("work_dirs/gemma4_mode1_moe")
    assert "scripts_gemma4_moe/quantize_moe.py" in command[1]
    assert command[command.index("--device") + 1] == "cuda:1"
    assert command[command.index("--llm_bits") + 1] == "4"
    assert command[command.index("--llm_group_size") + 1] == "64"
    assert command[command.index("--dataset") + 1].endswith(
        "quantization/calibration/moe_ebss/gen_data/Qwen3-Next-80B-A3B-Instruct.jsonl"
    )
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert "--sym" in command


def test_gemma4_series_gptq_defaults_use_dense_and_moe_calibration_jsonl():
    from xhmodel_merak.xh_llm.models.gemma4_series.quant_adapter import (
        DEFAULT_DENSE_CALIBRATION_JSONL,
        DEFAULT_MOE_CALIBRATION_JSONL,
        build_gptqmodel_recipe_kwargs,
    )

    dense_dir = Path("/data01/datasets/gemma-4-31B-it")
    moe_dir = Path("/data01/datasets/gemma-4-26B-A4B-it")
    if not dense_dir.exists() or not moe_dir.exists():
        pytest.skip("Gemma4 local HF configs are not available")

    common = {
        "algorithm": "gptqmodel",
        "method": "gptq",
        "preset": "full_multimodal",
        "artifact_format": "gptqmodel_hf",
        "group_size": 64,
        "calibration": {
            "dataset": "wikitext",
            "split": "train",
            "nsamples": 256,
            "seqlen": 2048,
        },
        "runtime": {
            "batch_size": 1,
            "trust_remote_code": True,
        },
        "validation": {"dry_run": True},
    }

    dense_kwargs = build_gptqmodel_recipe_kwargs(
        hf_model_dir=str(dense_dir),
        output_dir="work_dirs/gemma4_dense_gptq",
        device="cuda:0",
        workflow_seed=42,
        export_model_cfg={"context_max_length": 2048, "prefill_chunk_length": 256},
        quant_cfg=common,
    )
    moe_kwargs = build_gptqmodel_recipe_kwargs(
        hf_model_dir=str(moe_dir),
        output_dir="work_dirs/gemma4_moe_gptq",
        device="cuda:0",
        workflow_seed=42,
        export_model_cfg={"context_max_length": 2048, "prefill_chunk_length": 256},
        quant_cfg=common,
    )

    assert dense_kwargs["topology"] == "dense"
    assert dense_kwargs["calibration_jsonl"].endswith(
        "quantization/calibration/dense_ivsg/gen_data/Qwen3.5-27B.jsonl"
    )
    assert "calibration_dataset" not in dense_kwargs
    assert moe_kwargs["topology"] == "moe"
    assert moe_kwargs["calibration_jsonl"].endswith(
        "quantization/calibration/moe_ebss/gen_data/Qwen3-Next-80B-A3B-Instruct.jsonl"
    )
    assert moe_kwargs["moe"]["routing"] == "bypass"


def test_gemma4_series_detects_e2b_as_dense_e_series():
    from xhmodel_merak.xh_llm.models.gemma4_series.variants import resolve_gemma4_series_variant

    model_dir = Path("/data01/datasets/gemma-4-E2B-it")
    if not model_dir.exists():
        pytest.skip("Gemma4 E2B local HF config is not available")

    import json

    hf_config = json.loads((model_dir / "config.json").read_text())
    variant = resolve_gemma4_series_variant(hf_config)

    assert variant.name == "e2b"
    assert variant.topology == "dense"
    assert variant.has_audio is True
    assert variant.has_per_layer_input is True


def test_gemma4_series_decode_omits_full_attention_mask_contract():
    from types import SimpleNamespace

    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        XHGemma4SeriesModel,
    )

    model = object.__new__(XHGemma4SeriesModel)
    model.config = SimpleNamespace(
        bidirectional_vision_attention=True,
        hidden_size_per_layer_input=0,
    )
    model.per_layer_input_embedding = None
    model._kvcache_config = SimpleNamespace(num_layers=2)
    model._llm_prefill = True

    prefill_inputs = model.get_export_cfg()["input_names"]
    assert "full_attention_mask" in prefill_inputs
    assert "sliding_attention_mask" in prefill_inputs

    model._llm_prefill = False
    decode_inputs = model.get_export_cfg()["input_names"]
    assert "full_attention_mask" not in decode_inputs
    assert "sliding_attention_mask" in decode_inputs


def test_gemma4_series_decode_preprocess_skips_full_attention_mask():
    from xhmodel_merak.xh_llm.models.gemma4_series.data_preprocess import Gemma4DataPreprocess
    from xhmodel_merak.xh_llm.types import CacheList

    token_embedding = nn.Embedding(32, 4)
    common = dict(
        token_embedding=token_embedding,
        input_sequence_length=1,
        context_length=8,
        past_key_caches=CacheList([torch.zeros((1, 2, 8, 4), dtype=torch.float16)]),
        past_value_caches=CacheList([torch.zeros((1, 2, 8, 4), dtype=torch.float16)]),
        pad_token_id=0,
        image_token_id=9,
        audio_token_id=-1,
        video_token_id=-1,
        bidirectional_vision_attention=True,
        emit_full_attention_mask=False,
    )
    preprocess = Gemma4DataPreprocess(**common)
    outputs = preprocess(
        {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "past_seq_length": 4,
            "mm_token_type_ids": torch.tensor([[0]], dtype=torch.long),
        }
    )

    assert len(outputs) == 6
    assert outputs[3].shape[-1] != 8  # this is sliding_attention_mask, not full mask


def test_gemma4_series_sliding_kv_cache_input_mode_controls_shape():
    from xhmodel_merak.xh_llm.models.gemma4_series.llm_text import (
        _gemma4_cache_seq_len_for_layer,
    )
    from xhmodel_merak.xh_llm.models.gemma4_series.xh_gemma4_series_config import (
        XHGemma4SeriesModelConfig,
    )

    assert (
        _gemma4_cache_seq_len_for_layer(
            layer_type="sliding_attention",
            context_max_length=2048,
            sliding_window=1024,
            input_seq_len=256,
            sliding_kv_cache_input_mode="slice_window",
        )
        == 1280
    )
    legacy = _gemma4_cache_seq_len_for_layer(
        layer_type="sliding_attention",
        context_max_length=2048,
        sliding_window=1024,
        input_seq_len=256,
        sliding_kv_cache_input_mode="legacy_full",
    )
    assert legacy >= 2048
    assert legacy > 1280

    config = XHGemma4SeriesModelConfig(
        model_name="shape_mode",
        model_type="Gemma4ForConditionalGeneration",
        hf_model=None,
    )
    assert config.sliding_kv_cache_input_mode == "slice_window"

    legacy_config = XHGemma4SeriesModelConfig(
        model_name="legacy_shape_mode",
        model_type="Gemma4ForConditionalGeneration",
        hf_model=None,
        sliding_kv_cache_input_mode="legacy_full",
    )
    assert legacy_config.sliding_kv_cache_input_mode == "legacy_full"

    with pytest.raises(ValueError, match="sliding_kv_cache_input_mode"):
        XHGemma4SeriesModelConfig(
            model_name="bad_shape_mode",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=None,
            sliding_kv_cache_input_mode="bad",
        )
