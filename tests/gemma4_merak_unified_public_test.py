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
            "algorithm": "gptqmodel",
            "method": "autoround",
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
        DEFAULT_AUTOROUND_DATASET,
        DEFAULT_DENSE_CALIBRATION_JSONL,
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
    assert mode1_quant["algorithm"] == "gptqmodel"
    assert mode1_quant["method"] == "autoround"
    assert mode1_quant["preset"] == "mode1"
    assert mode1_quant["calibration"]["dataset"] == DEFAULT_AUTOROUND_DATASET
    assert "jsonl" not in mode1_quant["calibration"]
    assert mode1_quant["calibration"]["seqlen"] == 2048
    assert mode1_quant["runtime"]["batch_size"] == 8
    assert moe_mode1_quant["algorithm"] == "gptqmodel"
    assert moe_mode1_quant["method"] == "autoround"
    assert moe_mode1_quant["preset"] == "mode1"
    assert moe_mode1_quant["calibration"]["dataset"] == DEFAULT_AUTOROUND_DATASET
    assert "jsonl" not in moe_mode1_quant["calibration"]
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
            "algorithm": "gptqmodel",
            "method": "autoround",
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
    assert command[command.index("--dataset") + 1] == "NeelNanda/pile-10k"
    assert command[command.index("--dtype") + 1] == "bfloat16"
    assert "--sym" in command


def test_merak_auto_model_name_resolves_quant_shape_and_hf_pe_contract(tmp_path):
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig
    from xhmodel_merak.xh_llm.workflows.naming import resolve_auto_model_name

    base = {
        "quant": {
            "algorithm": "gptqmodel",
            "method": "gptq",
            "bits": 4,
        },
        "export": {
            "naming": {
                "family": "gemma4",
                "variant": "e4b",
                "profile": "full",
            },
            "model": {
                "chip_arch": "XH2a",
                "model_name": "auto",
                "context_max_length": 8192,
                "prefill_chunk_length": 256,
                "quant_scheme": {"quant_type": "w8a8h1_sefp"},
            },
        },
    }

    resolved = resolve_auto_model_name(WorkflowConfig(data=base, source="gemma4_e4b_full.yaml"))

    assert resolved.data["export"]["model"]["model_name"] == "xh2_gemma4_e4b_full_gptq_w4a8_256_8k_mpe32k"
    assert "h1_sefp" not in resolved.data["export"]["model"]["model_name"]

    base["quant"]["method"] = "autoround"
    resolved = resolve_auto_model_name(WorkflowConfig(data=base, source="gemma4_e4b_autoround.yaml"))
    assert resolved.data["export"]["model"]["model_name"] == "xh2_gemma4_e4b_full_autoround_w4a8_256_8k_mpe32k"

    hf_dir = tmp_path / "hf"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text(
        '{"text_config": {"max_position_embeddings": 262144}}',
        encoding="utf-8",
    )
    resolved = resolve_auto_model_name(
        WorkflowConfig(data=base, source="gemma4_e4b_autoround.yaml"),
        hf_model_dir=str(hf_dir),
    )
    assert resolved.data["export"]["model"]["model_name"] == "xh2_gemma4_e4b_full_autoround_w4a8_256_8k_mpe256k"


def test_merak_auto_model_name_requires_method_for_existing_hf_quant():
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig
    from xhmodel_merak.xh_llm.workflows.naming import resolve_auto_model_name

    cfg = WorkflowConfig(
        data={
            "quant": {
                "algorithm": "existing_hf",
                "existing_hf_model_dir": "/tmp/quant",
            },
            "export": {
                "naming": {
                    "family": "gemma4",
                    "variant": "e4b",
                    "profile": "full",
                },
                "model": {
                    "chip_arch": "XH2a",
                    "model_name": "auto",
                    "context_max_length": 2048,
                    "prefill_chunk_length": 256,
                    "quant_scheme": {"quant_type": "w8a8h1_sefp"},
                },
            },
        },
        source="existing_hf.yaml",
    )

    with pytest.raises(ValueError, match="existing_hf.*method"):
        resolve_auto_model_name(cfg)


def test_merak_auto_model_name_encodes_base_w8a8_contract():
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig
    from xhmodel_merak.xh_llm.workflows.naming import resolve_auto_model_name

    cfg = WorkflowConfig(
        data={
            "quant": None,
            "export": {
                "naming": {
                    "family": "gemma4",
                    "variant": "e4b",
                    "profile": "full",
                },
                "model": {
                    "chip_arch": "XH2a",
                    "model_name": "auto",
                    "context_max_length": 2048,
                    "prefill_chunk_length": 256,
                    "quant_scheme": {"quant_type": "w8a8h1_sefp"},
                },
            },
        },
        source="base.yaml",
    )

    resolved = resolve_auto_model_name(cfg)

    assert resolved.data["export"]["model"]["model_name"] == "xh2_gemma4_e4b_full_base_w8a8_256_2k_mpe32k"


def test_gemma4_series_workflow_model_names_encode_quant_contract():
    import re
    import yaml

    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig
    from xhmodel_merak.xh_llm.workflows.naming import resolve_auto_model_name

    config_paths = sorted(Path("configs_merak/workflows/xh2a/llm_models/gemma4_series").glob("*/*.yaml"))
    assert config_paths

    names: set[str] = set()
    for config_path in config_paths:
        raw_cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        assert raw_cfg["export"]["model"]["model_name"] == "auto"

        resolved = resolve_auto_model_name(WorkflowConfig.from_file(str(config_path)))
        cfg = resolved.data
        quant_cfg = cfg["quant"]
        naming_cfg = cfg["export"]["naming"]
        model_cfg = cfg["export"]["model"]
        model_name = model_cfg["model_name"]
        quant_type = model_cfg["quant_scheme"]["quant_type"]

        algorithm = str(quant_cfg["algorithm"]).lower()
        method = str(quant_cfg.get("method") or ("gptq" if algorithm == "gptqmodel" else algorithm)).lower()
        algorithm_token = "autoround" if method in {"autoround", "auto_round", "auto-round"} else "gptq"
        bits_token = f"w{int(quant_cfg['bits'])}"
        activation_match = re.search(r"a(\d+)", quant_type)
        assert activation_match, f"{config_path}: quant_type={quant_type!r} must encode activation bits"
        activation_token = f"a{activation_match.group(1)}"

        assert algorithm == "gptqmodel"
        assert method in {"gptq", "autoround"}
        assert model_name.startswith(f"xh2_gemma4_{naming_cfg['variant']}_{naming_cfg['profile']}_{algorithm_token}_")
        assert bits_token in model_name
        assert activation_token in model_name
        assert re.search(r"_256_2k_mpe(32|128|256)k$", model_name), model_name
        assert "h1_sefp" not in model_name
        assert model_name not in names
        names.add(model_name)


def test_gemma4_series_export_wrapper_stays_non_mtp_compatibility():
    from examples_merak.llm.gemma4_series import export_hmonnx

    args = export_hmonnx._build_parser().parse_args(
        [
            "--preset",
            "e2b",
            "--action",
            "existing-hf",
            "--model",
            "/tmp/base",
            "--existing-hf-model-dir",
            "/tmp/quant",
            "--config",
            "configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml",
            "--dry-run",
        ]
    )

    workflow_args = export_hmonnx._to_workflow_args(args)

    assert workflow_args.hf_model_dir == "/tmp/base"
    assert workflow_args.existing_hf_model_dir == "/tmp/quant"
    assert not hasattr(workflow_args, "assistant_model_dir")
    assert not hasattr(workflow_args, "mtp_config")


def test_gemma4_series_quant_export_supports_external_mtp_dir():
    from examples_merak.llm.gemma4_series import gemma4_series_quant_export
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig

    config_path = "configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full_mtp.yaml"
    args = gemma4_series_quant_export.build_parser().parse_args(
        [
            "--hf-model-dir",
            "/tmp/base",
            "--config",
            config_path,
            "--export-output-dir",
            "/tmp/export",
            "--existing-hf-model-dir",
            "/tmp/quant",
            "--mtp-assistant-model-dir",
            "/tmp/assistant",
        ]
    )

    mtp_overrides = gemma4_series_quant_export._build_mtp_overrides(args)
    assert mtp_overrides == {
        "export.model.spec_decode_mode": "mtp",
        "export.model.mtp_config.assistant_hf_model": "/tmp/assistant",
        "export.model.mtp_config.target_hf_model": "/tmp/quant",
        "export.model.mtp_config.body_quant_type": "w8a8h1_sefp",
        "export.model.mtp_config.lm_head_quant_type": "w4a8h0_ssfp",
    }
    WorkflowConfig.from_file(config_path).with_overrides(mtp_overrides)
    gemma4_series_quant_export._validate_mtp_config_complete(args, mtp_overrides)
    assert gemma4_series_quant_export._resolve_quant_output_dir(args).endswith(
        "_workflow_existing_or_base_quant_placeholder"
    )


def test_gemma4_series_quant_export_rejects_non_mtp_config_with_assistant():
    from examples_merak.llm.gemma4_series import gemma4_series_quant_export

    args = gemma4_series_quant_export.build_parser().parse_args(
        [
            "--hf-model-dir",
            "/tmp/base",
            "--config",
            "configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml",
            "--export-output-dir",
            "/tmp/export",
            "--existing-hf-model-dir",
            "/tmp/quant",
            "--mtp-assistant-model-dir",
            "/tmp/assistant",
        ]
    )
    config_overrides = gemma4_series_quant_export._merge_overrides(
        gemma4_series_quant_export._build_quant_overrides(args),
        gemma4_series_quant_export._build_mtp_overrides(args),
    )

    with pytest.raises(ValueError, match="full_mtp YAML.*mtp_config"):
        gemma4_series_quant_export._validate_mtp_config_complete(args, config_overrides)


def test_gemma4_series_quant_export_reports_incomplete_mtp_config(tmp_path):
    from examples_merak.llm.gemma4_series import gemma4_series_quant_export

    config_path = tmp_path / "gemma4_incomplete_mtp.yaml"
    config_path.write_text(
        """
quant: {}
export:
  model:
    chip_arch: XH2a
    model_type: Gemma4ForConditionalGeneration
    hf_model: /tmp/base
    model_name: test
    spec_decode_mode: mtp
    num_draft_tokens: 6
    mtp_config:
      assistant_hf_model: /tmp/assistant
      target_hf_model: /tmp/target
      body_quant_type: w8a8h1_sefp
      lm_head_quant_type: w4a8h0_ssfp
      num_draft_tokens: 6
""",
        encoding="utf-8",
    )
    args = gemma4_series_quant_export.build_parser().parse_args(
        [
            "--hf-model-dir",
            "/tmp/base",
            "--config",
            str(config_path),
            "--export-output-dir",
            "/tmp/export",
            "--existing-hf-model-dir",
            "/tmp/quant",
            "--mtp-assistant-model-dir",
            "/tmp/assistant",
        ]
    )

    with pytest.raises(ValueError) as exc_info:
        gemma4_series_quant_export._validate_mtp_config_complete(args)

    message = str(exc_info.value)
    assert "full_mtp YAML" in message
    assert str(config_path) in message
    assert "export.model.mtp_config.assistant_layer_pattern" in message
    assert "export.model.mtp_config.shared_kv_inputs" in message


def test_gemma4_series_quant_export_validates_effective_mtp_config(tmp_path):
    from examples_merak.llm.gemma4_series import gemma4_series_quant_export

    config_path = tmp_path / "gemma4_effective_mtp.yaml"
    config_path.write_text(
        """
quant: {}
export:
  model:
    chip_arch: XH2a
    model_type: Gemma4ForConditionalGeneration
    hf_model: /tmp/base
    model_name: test
    spec_decode_mode: mtp
    num_draft_tokens: 6
    mtp_config:
      assistant_hf_model: /tmp/default_assistant
      target_hf_model: /tmp/default_target
      body_quant_type: w8a8h1_sefp
      lm_head_quant_type: w4a8h0_ssfp
      batch_size: 1
      input_sequence_length: 1
      context_max_length: 4096
      use_cache: true
      num_draft_tokens: 6
      assistant_num_hidden_layers: 4
      assistant_layer_pattern:
      - sliding_attention
      - sliding_attention
      - sliding_attention
      - full_attention
      assistant_hidden_size: 256
      assistant_num_attention_heads: 4
      assistant_num_key_value_heads: 1
      head_dim: 256
      shared_kv_inputs:
      - shared_key_cache_sliding
      - shared_value_cache_sliding
      - shared_key_cache_full
      - shared_value_cache_full
""",
        encoding="utf-8",
    )
    args = gemma4_series_quant_export.build_parser().parse_args(
        [
            "--hf-model-dir",
            "/tmp/base",
            "--config",
            str(config_path),
            "--export-output-dir",
            "/tmp/export",
            "--existing-hf-model-dir",
            "/tmp/quant",
            "--mtp-assistant-model-dir",
            "/tmp/assistant",
        ]
    )
    config_overrides = gemma4_series_quant_export._merge_overrides(
        gemma4_series_quant_export._build_quant_overrides(args),
        gemma4_series_quant_export._build_mtp_overrides(args),
    )

    gemma4_series_quant_export._validate_mtp_config_complete(args, config_overrides)


def test_gemma4_series_mtp_manifest_prefers_nested_spec_decode(tmp_path):
    import json

    from examples_merak.llm.gemma4_series import gemma4_series_quant_export

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    meta_path = export_dir / "golden_meta_info.json"
    draft_path = export_dir / "mtp_draft_decode" / "draft.onnx"
    draft_path.parent.mkdir()
    draft_path.write_text("draft", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "spec_decode_mode": "mtp",
                "spec_decode_block_size": 4,
                "spec_decode_verify_length": 5,
                "spec_decode": {
                    "mode": "mtp",
                    "block_size": 6,
                    "verify_length": 7,
                    "shared_sliding_cache_length": 1536,
                    "shared_full_cache_length": 4096,
                },
                "sliding_window": 1024,
                "model_config": {
                    "num_draft_tokens": 6,
                    "sliding_window": 1024,
                    "context_max_length": 4096,
                },
            }
        ),
        encoding="utf-8",
    )

    gemma4_series_quant_export._update_manifest_with_draft(
        meta_path,
        draft_path,
        lm_head_quant_type="w3a8h0_ssfp",
        shared_sliding_len=768,
        shared_full_len=2048,
    )

    updated = json.loads(meta_path.read_text(encoding="utf-8"))
    spec_decode = updated["spec_decode"]
    assert spec_decode["block_size"] == 6
    assert spec_decode["verify_length"] == 7
    assert spec_decode["shared_sliding_cache_length"] == 1536
    assert spec_decode["shared_full_cache_length"] == 4096
    assert spec_decode["target_decode_sliding_output_length"] == 1040
    assert spec_decode["draft_head_weight_bits"] == 3
    assert spec_decode["draft_decode_onnx"] == "mtp_draft_decode/draft.onnx"
    assert updated["spec_decode_block_size"] == 6
    assert updated["spec_decode_verify_length"] == 7


def test_gemma4_series_mtp_manifest_uses_non_default_model_draft_tokens(tmp_path):
    import json

    from examples_merak.llm.gemma4_series import gemma4_series_quant_export

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    meta_path = export_dir / "golden_meta_info.json"
    draft_path = export_dir / "mtp_draft_decode" / "draft.onnx"
    draft_path.parent.mkdir()
    draft_path.write_text("draft", encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "spec_decode_mode": "mtp",
                # Legacy top-level fields may be stale after a non-default
                # MTP export.  model_config/spec_decode must win.
                "spec_decode_block_size": 4,
                "spec_decode_verify_length": 5,
                "spec_decode": {
                    "mode": "mtp",
                    "shared_sliding_cache_length": 2304,
                    "shared_full_cache_length": 8192,
                },
                "sliding_window": 2048,
                "model_config": {
                    "num_draft_tokens": 8,
                    "sliding_window": 2048,
                    "context_max_length": 8192,
                },
            }
        ),
        encoding="utf-8",
    )

    gemma4_series_quant_export._update_manifest_with_draft(
        meta_path,
        draft_path,
        lm_head_quant_type="w4a8h0_ssfp",
        shared_sliding_len=1152,
        shared_full_len=4096,
    )

    updated = json.loads(meta_path.read_text(encoding="utf-8"))
    spec_decode = updated["spec_decode"]
    assert spec_decode["block_size"] == 8
    assert spec_decode["verify_length"] == 9
    assert spec_decode["shared_sliding_cache_length"] == 2304
    assert spec_decode["shared_full_cache_length"] == 8192
    assert spec_decode["target_decode_sliding_output_length"] == 2064
    assert spec_decode["draft_head_weight_bits"] == 4
    assert updated["spec_decode_block_size"] == 8
    assert updated["spec_decode_verify_length"] == 9


def test_gemma4_series_export_mtp_draft_writes_single_decode_dir(monkeypatch, tmp_path):
    import json
    import sys
    import types

    from xhmodel_merak.xh_llm.models.gemma4_series import mtp_workflow

    hm_dir = tmp_path / "out" / "hmquant_fake"
    hm_dir.mkdir(parents=True)
    (hm_dir / "prefill.onnx").write_text("prefill", encoding="utf-8")
    meta_path = hm_dir / "golden_meta_info.json"
    meta_path.write_text(
        json.dumps(
            {
                "spec_decode_mode": "mtp",
                "spec_decode": {"mode": "mtp"},
                "model_config": {
                    "chip_arch": "XH2a",
                    "context_max_length": 2048,
                    "num_draft_tokens": 4,
                    "mtp_config": {
                        "assistant_hf_model": "/tmp/assistant",
                        "target_hf_model": "/tmp/base",
                        "input_sequence_length": 1,
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeConfigDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

    class FakePrecisionMode:
        ALIGNED = "aligned"

    fake_logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    fake_xhquant_api = types.ModuleType("xhquant.api")
    fake_xhquant_api.ConfigDict = FakeConfigDict
    fake_xhquant_api.PrecisionMode = FakePrecisionMode
    fake_xhquant_api.get_xhquant_logger = lambda: fake_logger
    fake_xhquant_api.ptq_quantize = lambda *args, **kwargs: None
    fake_xhquant = types.ModuleType("xhquant")
    fake_xhquant.api = fake_xhquant_api
    monkeypatch.setitem(sys.modules, "xhquant", fake_xhquant)
    monkeypatch.setitem(sys.modules, "xhquant.api", fake_xhquant_api)

    class FakeDraftModel:
        quanted_model = object()

        def __init__(self, *args, **kwargs):
            pass

        def init_wrap_model(self):
            pass

        def prepare_inputs(self, _data):
            return {"inputs_embeds": object()}

        def convert_to_fronted_graph(self, _dummy):
            pass

        def convert_to_quant_graph(self, _chip_arch):
            pass

        def convert_to_export_graph(self, _dummy):
            pass

        def to_export_onnx(self, _dummy, output_dir, prefix):
            onnx_file = Path(output_dir) / f"{prefix}.onnx"
            onnx_file.write_text("draft", encoding="utf-8")
            return [str(onnx_file)]

        def release_exported_model(self):
            pass

        def release_quanted_model(self):
            pass

        def release_frontend_model(self):
            pass

        def release_wraped_model(self):
            pass

    fake_mtp_model = types.ModuleType(
        "xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model"
    )
    fake_mtp_model.XHGemma4SeriesAssistantDraftModel = FakeDraftModel
    monkeypatch.setitem(
        sys.modules,
        "xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model",
        fake_mtp_model,
    )

    draft_onnx = mtp_workflow.export_mtp_draft(
        SimpleNamespace(work_dir=str(tmp_path / "out")),
        hf_model_dir="/tmp/base",
    )

    assert draft_onnx is not None
    assert draft_onnx.parent == hm_dir / "mtp_draft_decode"
    assert not (hm_dir / "draft_onnx").exists()
    updated = json.loads(meta_path.read_text(encoding="utf-8"))
    assert updated["spec_decode"]["draft_decode_onnx"].startswith("mtp_draft_decode/")
    assert updated["draft_decode_onnx_file"].startswith("mtp_draft_decode/")


def test_gemma4_series_workflow_export_owns_mtp_draft_export(monkeypatch, tmp_path):
    import json

    from xhmodel_merak.xh_llm.models.gemma4_series import mtp_workflow
    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow
    from xhmodel_merak.xh_llm.workflows.base import BaseHMONNXWorkflow
    from xhmodel_merak.xh_llm.workflows.result import ExportResult, QuantResult

    calls = []

    def fake_base_export(self, quant_result, output_dir, device, config_overrides=None):
        hm_dir = tmp_path / "out" / "hmquant_fake"
        hm_dir.mkdir(parents=True)
        (hm_dir / "prefill.onnx").write_text("prefill", encoding="utf-8")
        (hm_dir / "golden_meta_info.json").write_text(
            json.dumps(
                {
                    "spec_decode_mode": "mtp",
                    "spec_decode": {"mode": "mtp"},
                    "model_config": {
                        "mtp_config": {
                            "assistant_hf_model": "/tmp/assistant",
                            "target_hf_model": "/tmp/base",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        return ExportResult(work_dir=str(tmp_path / "out"), config_file=str(tmp_path / "out/config.yaml"))

    def fake_export_mtp_draft(export_result, *, hf_model_dir, chip_arch=None, draft_dtype="float16"):
        calls.append((export_result.work_dir, hf_model_dir, chip_arch, draft_dtype))
        return tmp_path / "out/hmquant_fake/mtp_draft_decode/draft.onnx"

    monkeypatch.setattr(BaseHMONNXWorkflow, "export", fake_base_export)
    monkeypatch.setattr(Gemma4SeriesWorkflow, "_validate_export_model", lambda self, config_overrides: None)
    monkeypatch.setattr(mtp_workflow, "export_mtp_draft", fake_export_mtp_draft)

    workflow = Gemma4SeriesWorkflow(
        hf_model_dir="/tmp/base",
        config_path="configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full_mtp.yaml",
    )
    result = workflow.export(
        quant_result=QuantResult(hf_model_dir="/tmp/base", skipped=True),
        output_dir=str(tmp_path / "out"),
        device="cpu",
    )

    assert result.work_dir == str(tmp_path / "out")
    assert calls == [(str(tmp_path / "out"), workflow.hf_model_dir, None, "float16")]


def test_gemma4_series_workflow_dump_golden_owns_mtp_draft_golden(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace

    from xhmodel_merak.xh_llm.models.gemma4_series.workflow import Gemma4SeriesWorkflow
    from xhmodel_merak.xh_llm.workflows.result import ExportResult

    hm_dir = tmp_path / "out" / "hmquant_fake"
    hm_dir.mkdir(parents=True)
    meta_file = hm_dir / "golden_meta_info.json"
    meta_file.write_text(
        json.dumps(
            {
                "spec_decode_mode": "mtp",
                "spec_decode": {
                    "mode": "mtp",
                    "draft_decode_onnx": "mtp_draft_decode/draft.onnx",
                },
                "model_config": {},
            }
        ),
        encoding="utf-8",
    )

    class DummyHMONNX:
        enable_golden = False

        def to(self, device):
            self.device = device

    import xhmodel_merak.xh_llm as xh_llm

    monkeypatch.setattr(
        xh_llm,
        "AutoLLMHONNXModel",
        SimpleNamespace(from_pretrained=lambda _meta_file: DummyHMONNX()),
    )
    monkeypatch.setattr(Gemma4SeriesWorkflow, "_build_golden_message_cases", lambda *args, **kwargs: [])
    called = []

    def fake_dump_mtp_draft_golden(self, meta_file_arg, device, *, logger=None):
        called.append((meta_file_arg, device, logger is not None))
        return hm_dir / "mtp_draft_decode"

    monkeypatch.setattr(Gemma4SeriesWorkflow, "_dump_mtp_draft_golden", fake_dump_mtp_draft_golden)

    workflow = Gemma4SeriesWorkflow(
        hf_model_dir="/tmp/base",
        config_path="configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full_mtp.yaml",
    )
    result = workflow.dump_golden(
        export_result=ExportResult(work_dir=str(tmp_path / "out"), config_file=str(tmp_path / "out/config.yaml")),
        device="cpu",
        input_messages={"text": "hello"},
    )

    assert result == str(meta_file)
    assert called == [(str(meta_file), "cpu", True)]


def test_gemma4_series_quant_export_non_mtp_leaves_spec_decode_unset():
    from examples_merak.llm.gemma4_series import gemma4_series_quant_export

    assert not hasattr(gemma4_series_quant_export, "_export_mtp_draft")

    args = gemma4_series_quant_export.build_parser().parse_args(
        [
            "--hf-model-dir",
            "/tmp/base",
            "--config",
            "configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml",
            "--export-output-dir",
            "/tmp/export",
            "--base",
        ]
    )

    assert gemma4_series_quant_export._build_mtp_overrides(args) is None
    assert gemma4_series_quant_export._build_quant_overrides(args) == {"quant": None}


def test_gemma4_series_workflow_demo_presets_use_relative_weight_paths():
    from examples_merak.llm.gemma4_series import gemma4_workflow_demo

    for preset in gemma4_workflow_demo.PRESETS.values():
        assert preset.hf_model_dir.startswith("weights/")
        assert preset.assistant_model_dir.startswith("weights/")
        assert not preset.hf_model_dir.startswith("/data01/")
        assert not preset.assistant_model_dir.startswith("/data01/")


def test_gemma4_series_auto_model_name_uses_hf_position_embedding_length():
    from xhmodel_merak.xh_llm.workflows.config import WorkflowConfig
    from xhmodel_merak.xh_llm.workflows.naming import resolve_auto_model_name

    cases = {
        "configs_merak/workflows/xh2a/llm_models/gemma4_series/e2b/gemma4_e2b_full.yaml": (
            "/data01/datasets/gemma-4-E2B-it",
            "mpe128k",
        ),
        "configs_merak/workflows/xh2a/llm_models/gemma4_series/e4b/gemma4_e4b_full.yaml": (
            "/data01/datasets/gemma-4-E4B-it",
            "mpe128k",
        ),
        "configs_merak/workflows/xh2a/llm_models/gemma4_series/31b/gemma4_31b_full.yaml": (
            "/data01/datasets/gemma-4-31B-it",
            "mpe256k",
        ),
        "configs_merak/workflows/xh2a/llm_models/gemma4_series/26b_a4b/gemma4_26b_a4b_full.yaml": (
            "/data01/datasets/gemma-4-26B-A4B-it",
            "mpe256k",
        ),
    }
    if not all(Path(hf_dir).exists() for hf_dir, _expected in cases.values()):
        pytest.skip("Gemma4 local HF configs are not available")

    for config_path, (hf_dir, expected_suffix) in cases.items():
        resolved = resolve_auto_model_name(
            WorkflowConfig.from_file(config_path),
            hf_model_dir=hf_dir,
        )
        assert resolved.data["export"]["model"]["model_name"].endswith(f"_256_2k_{expected_suffix}")


def test_gemma4_series_gptq_defaults_use_dense_and_moe_calibration_jsonl():
    from xhmodel_merak.xh_llm.models.gemma4_series.quant_adapter import (
        DEFAULT_DENSE_CALIBRATION_JSONL,
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



def test_gemma4_series_mtp_eos_reads_exported_generation_config(tmp_path):
    from examples_merak.llm.gemma4_series.mtp_hmonnx_inference import (
        _generation_config_candidates,
        _resolve_eos_token_ids,
    )

    export_dir = tmp_path / "hmquant_test"
    hf_config_dir = export_dir / "hf_config"
    hf_config_dir.mkdir(parents=True)
    meta_path = export_dir / "golden_meta_info.json"
    meta_path.write_text("{}", encoding="utf-8")
    (hf_config_dir / "generation_config.json").write_text(
        '{"eos_token_id": [1, 106, 50], "pad_token_id": 0}',
        encoding="utf-8",
    )

    class Tokenizer:
        eos_token_id = 1

    meta = {
        "_meta_path": str(meta_path),
        "hf_config": "hf_config",
        "model_config": {},
    }

    eos_token_ids = _resolve_eos_token_ids(Tokenizer(), *_generation_config_candidates(meta))

    assert eos_token_ids == {1, 106, 50}

def test_gemma4_series_mtp_sliding_mask_uses_compact_cache_tail():
    from xhmodel_merak.xh_llm.models.gemma4_series.data_preprocess import Gemma4DataPreprocess
    from xhmodel_merak.xh_llm.types import CacheList

    token_embedding = nn.Embedding(32, 4)
    preprocess = Gemma4DataPreprocess(
        token_embedding=token_embedding,
        input_sequence_length=2,
        context_length=16,
        past_key_caches=CacheList([torch.zeros((1, 2, 8, 4), dtype=torch.float16)]),
        past_value_caches=CacheList([torch.zeros((1, 2, 8, 4), dtype=torch.float16)]),
        pad_token_id=0,
        image_token_id=-1,
        audio_token_id=-1,
        video_token_id=-1,
        sliding_window=4,
        emit_full_attention_mask=False,
    )
    outputs = preprocess(
        {
            "input_ids": torch.tensor([[1]], dtype=torch.long),
            "past_seq_length": 8,
            "mm_token_type_ids": torch.tensor([[0]], dtype=torch.long),
        }
    )

    sliding_mask = outputs[3]
    # LLMCache compacts with attention_max_length=sliding_window.  For
    # sliding_window=4 and q=2 the natural compact width is aligned(5)=16, and
    # the first real decode token is appended at coordinate sw-1=3.
    assert sliding_mask.shape[-1] == 16
    assert torch.all(sliding_mask[0, 0, 0, 0:4] == 0)
    assert torch.all(sliding_mask[0, 0, 0, 4:] < 0)


def test_gemma4_series_draft_mask_uses_shared_sliding_tail():
    from types import SimpleNamespace

    from examples_merak.llm.gemma4_series.mtp_hmonnx_inference import _build_draft_masks

    target_model = SimpleNamespace(
        dtype=torch.float16,
        device=torch.device("cpu"),
        sliding_window=4,
        meta_info=SimpleNamespace(model_config=SimpleNamespace(sliding_window=4)),
    )
    assistant_session = SimpleNamespace(
        input_infos={
            "sliding_attention_mask": SimpleNamespace(shape=(1, 1, 1, 8)),
            "full_attention_mask": SimpleNamespace(shape=(1, 1, 1, 10)),
        }
    )

    full_mask, sliding_mask = _build_draft_masks(
        target_model,
        assistant_session,
        full_valid_length=8,
        sliding_valid_length=6,
    )

    assert torch.all(full_mask[0, 0, 0, :8] == 0)
    assert torch.all(full_mask[0, 0, 0, 8:] < 0)
    assert torch.all(sliding_mask[0, 0, 0, 2:6] == 0)
    assert torch.all(sliding_mask[0, 0, 0, :2] < 0)
    assert torch.all(sliding_mask[0, 0, 0, 6:] < 0)


def test_gemma4_series_hmonnx_shared_cache_indices_use_cache_list_metadata():
    from examples_merak.llm.gemma4_series.mtp_hmonnx_inference import _hmonnx_shared_cache_indices

    target_model = SimpleNamespace(
        layer_types=[
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
        layer_cache_types=["sliding_attention", "full_attention"],
    )

    assert _hmonnx_shared_cache_indices(target_model) == {
        "sliding_attention": 0,
        "full_attention": 1,
    }


def test_gemma4_series_hmonnx_shared_cache_indices_fall_back_to_kv_shapes():
    from examples_merak.llm.gemma4_series.mtp_hmonnx_inference import _hmonnx_shared_cache_indices

    target_model = SimpleNamespace(
        _kvcache_mixin=SimpleNamespace(
            layer_kv_shapes=[
                [1, 1, 1280, 1],
                [1, 1, 2048, 1],
                [1, 1, 1280, 1],
                [1, 1, 2048, 1],
            ]
        ),
        kvcache_config=SimpleNamespace(max_sequence_length=2048),
    )

    assert _hmonnx_shared_cache_indices(target_model) == {
        "sliding_attention": 2,
        "full_attention": 3,
    }


def test_gemma4_series_hmonnx_partial_commit_keeps_compact_cache_prefix_only():
    from xhquant.core import CacheTensor, HybridCacheTensor

    from examples_merak.llm.gemma4_series.mtp_hmonnx_inference import _commit_hmonnx_verified_cache

    before_sliding = HybridCacheTensor(torch.arange(8, dtype=torch.float16).view(1, 1, 8, 1))
    before_sliding.cache_valid_len = 8
    verified_sliding = HybridCacheTensor(torch.zeros((1, 1, 8, 1), dtype=torch.float16))
    # Full verify with true sliding_window compacts the valid suffix to the
    # front.  If only one of two verify tokens commits, rollback only reduces
    # HybridCacheTensor.cache_valid_len and masks the unaccepted compact tail.
    verified_sliding.data[0, 0, :6, 0] = torch.tensor([4, 5, 6, 7, 100, 101], dtype=torch.float16)
    verified_sliding.cache_valid_len = 6

    before_full = CacheTensor(torch.zeros((1, 1, 16, 1), dtype=torch.float16))
    verified_full = CacheTensor(before_full.data.clone())
    verified_full.data[:, :, 8, :] = 200
    verified_full.data[:, :, 9, :] = 201

    target_model = SimpleNamespace(
        layer_types=["sliding_attention", "full_attention"],
        _kvcache_mixin=SimpleNamespace(
            past_key_caches=[verified_sliding, verified_full],
            past_value_caches=[
                HybridCacheTensor(verified_sliding.data.clone()),
                CacheTensor(verified_full.data.clone()),
            ],
        ),
    )
    target_model._kvcache_mixin.past_value_caches[0].cache_valid_len = 8
    target_model.past_key_caches = target_model._kvcache_mixin.past_key_caches
    target_model.past_value_caches = target_model._kvcache_mixin.past_value_caches

    snapshot = ([before_sliding, before_full], [before_sliding, before_full])
    shared = _commit_hmonnx_verified_cache(
        target_model,
        snapshot,
        past_seq_length=8,
        commit_length=1,
        verify_length=2,
    )

    committed_sliding = target_model.past_key_caches[0]
    assert committed_sliding.cache_valid_len == 5
    assert committed_sliding.data[0, 0, :5, 0].tolist() == [4, 5, 6, 7, 100]
    assert torch.all(committed_sliding.data[0, 0, 5:, 0] == 0)
    assert target_model.past_key_caches[1].data[0, 0, 8, 0].item() == 200
    assert target_model.past_key_caches[1].data[0, 0, 9, 0].item() == 0
    assert torch.equal(shared["shared_key_cache_sliding"], committed_sliding.data)


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


def test_gemma4_series_mtp_updates_sliding_cache_attention_max_length():
    from types import SimpleNamespace

    from xhmodel_merak.xh_llm.models.gemma4_series._llm_model_impl import _Gemma4TextAttention

    attn = object.__new__(_Gemma4TextAttention)
    attn.use_cache = True
    attn.is_sliding_attention = True
    attn.sliding_window = 1024
    attn.k_cache = SimpleNamespace(attention_max_length=1024)
    attn.v_cache = SimpleNamespace(attention_max_length=1024)

    attn._update_cfg({"sliding_cache_output_length": 1280, "input_sequence_length": 5})

    assert attn.k_cache.attention_max_length == 1024
    assert attn.v_cache.attention_max_length == 1024


def test_gemma4_series_mtp_target_contract_records_separate_cache_widths():
    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import XHGemma4SeriesModel

    model = object.__new__(XHGemma4SeriesModel)
    model.config = SimpleNamespace(
        enable_mtp_outputs=True,
        spec_decode_mode="mtp",
        num_draft_tokens=4,
        mtp_config=SimpleNamespace(lm_head_quant_type="w4a8h0_ssfp"),
        context_max_length=2048,
        prefill_chunk_length=256,
        num_logits_to_keep=1,
        bidirectional_vision_attention=False,
        sliding_kv_cache_input_mode="slice_window",
    )
    model.sliding_window = 1024
    model.layer_types = ["sliding_attention", "full_attention"]
    model.layer_cache_types = ["sliding_attention", "full_attention"]
    model.layer_cache_indices = [0, 1]
    model.per_layer_input_embedding = None
    model._kvcache_config = SimpleNamespace(num_layers=0)
    model._kvcache_mixin = SimpleNamespace(layer_kv_shapes=[[1, 1, 1280, 1], [1, 1, 2048, 1]])
    model._llm_prefill = True

    export_cfg = XHGemma4SeriesModel.get_export_cfg(model)
    assert export_cfg["output_names"] == ["logits", "target_hidden_state"]

    meta = SimpleNamespace()
    XHGemma4SeriesModel._extra_export_metadata(model, str(Path(".")), meta)
    assert meta.spec_decode["shared_sliding_cache_length"] == 1280
    assert meta.spec_decode["shared_full_cache_length"] == 2048
    assert meta.spec_decode["target_decode_sliding_output_length"] == 1040
    assert "sliding_cache_output_length" not in meta.spec_decode
    assert meta.layer_cache_types == ["sliding_attention", "full_attention"]
    assert meta.layer_cache_indices == [0, 1]


def test_gemma4_series_e2e_validation_rejects_stale_mtp_decode(tmp_path):
    import json

    import onnx
    from onnx import TensorProto, helper

    from examples_merak.llm.gemma4_series.gemma4_e2e_validation import validate_meta

    def make_graph(path, *, q_len: int, mask_width: int, cache_width: int, amax: int):
        graph = helper.make_graph(
            [
                helper.make_node(
                    "KVcache",
                    ["x"],
                    ["cache_update"],
                    name="sliding_cache",
                    attention_max_length=amax,
                )
            ],
            "g",
            [
                helper.make_tensor_value_info("sliding_attention_mask", TensorProto.FLOAT16, [1, 1, q_len, mask_width]),
                helper.make_tensor_value_info("x", TensorProto.FLOAT16, [1, 1, q_len, 1]),
            ],
            [
                helper.make_tensor_value_info("logits", TensorProto.FLOAT16, [1, q_len, 32]),
                helper.make_tensor_value_info("target_hidden_state", TensorProto.FLOAT16, [1, q_len, 8]),
            ],
            value_info=[
                helper.make_tensor_value_info("cache_update", TensorProto.FLOAT16, [1, 1, cache_width, 1]),
            ],
        )
        onnx.save(helper.make_model(graph), path)

    make_graph(tmp_path / "prefill.onnx", q_len=256, mask_width=1280, cache_width=1280, amax=1024)
    # Stale decode: old single-token graph with slice_window-only cache width.
    make_graph(tmp_path / "decode.onnx", q_len=1, mask_width=1024, cache_width=1024, amax=1024)
    meta = {
        "spec_decode_mode": "mtp",
        "sliding_window": 1024,
        "layer_types": ["sliding_attention", "full_attention"],
        "layer_kv_shapes": [[1, 1, 1280, 1], [1, 1, 2048, 1]],
        "prefill_hmonnx": "prefill.onnx",
        "decode_hmonnx": "decode.onnx",
        "visual_config": {"hmonnx": "visual.onnx"},
        "video_visual_config": {"hmonnx": "video.onnx"},
        "spec_decode": {
            "mode": "mtp",
            "verify_length": 5,
            "shared_sliding_cache_length": 1280,
        },
        "model_config": {
            "model_type": "Gemma4ForConditionalGeneration",
            "context_max_length": 2048,
            "prefill_chunk_length": 256,
            "enable_mtp_outputs": True,
            "num_draft_tokens": 4,
        },
    }
    meta_path = tmp_path / "golden_meta_info.json"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    with pytest.raises(ValueError, match="stale decode exports"):
        validate_meta(meta_path, "31b")


def test_gemma4_series_export_strips_default_false_llmcache_attr(tmp_path):
    import onnx
    from onnx import TensorProto, helper

    from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_llm_model import (
        _strip_default_llmcache_only_handle_old_cache_attrs,
    )

    graph = helper.make_graph(
        [
            helper.make_node(
                "KVcache",
                ["x"],
                ["y"],
                name="default_false",
                attention_max_length=-1,
                only_handle_old_cache=0,
            ),
            helper.make_node(
                "KVcache",
                ["y"],
                ["z"],
                name="read_only",
                attention_max_length=-1,
                only_handle_old_cache=1,
            ),
        ],
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("z", TensorProto.FLOAT, [1])],
    )
    model_path = tmp_path / "cache.onnx"
    onnx.save(helper.make_model(graph), model_path)

    assert _strip_default_llmcache_only_handle_old_cache_attrs(model_path) == 1

    model = onnx.load(model_path)
    attrs = {
        node.name: {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}
        for node in model.graph.node
    }
    assert "only_handle_old_cache" not in attrs["default_false"]
    assert attrs["read_only"]["only_handle_old_cache"] == 1
