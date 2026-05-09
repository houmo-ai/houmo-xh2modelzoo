import copy
from pathlib import Path
from types import SimpleNamespace

import torch
import xhquant.nn as xhnn
from accelerate import init_empty_weights
from torch import nn
from transformers import AutoConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration, apply_rotary_pos_emb


MODEL_DIR = Path("/data01/datasets/gemma-4-E4B")


def _build_tiny_gemma4_model():
    config = copy.deepcopy(AutoConfig.from_pretrained(str(MODEL_DIR)))
    text = config.text_config
    vision = config.vision_config
    audio = config.audio_config

    text.hidden_size = 64
    text.intermediate_size = 128
    text.hidden_size_per_layer_input = 8
    text.num_hidden_layers = 4
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.head_dim = 16
    text.global_head_dim = 32
    text.num_kv_shared_layers = 1
    text.sliding_window = 8
    text.layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    text.vocab_size = 128
    text.max_position_embeddings = 64
    text.bos_token_id = 2
    text.eos_token_id = 1
    text.pad_token_id = 0

    vision.hidden_size = 32
    vision.intermediate_size = 64
    vision.num_hidden_layers = 1
    vision.num_attention_heads = 4
    vision.embedding_dim = 32

    audio.hidden_size = 32
    audio.intermediate_size = 64
    audio.num_hidden_layers = 1
    audio.num_attention_heads = 4
    audio.num_key_value_heads = 4
    audio.head_dim = 8
    audio.input_dim = 128

    config.image_token_id = 10
    config.audio_token_id = 11
    config.video_token_id = 12
    config.boi_token_id = 13
    config.eoi_token_id = 14
    config.boa_token_id = 15
    config.eoa_token_id = 16
    config.text_config = text
    config.vision_config = vision
    config.audio_config = audio
    return Gemma4ForConditionalGeneration(config).eval()


def test_gemma4_llm_wrap_and_preprocess_smoke():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    hf_model = _build_tiny_gemma4_model()
    model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="tiny_gemma4",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(MODEL_DIR),
            context_max_length=16,
            prefill_chunk_length=4,
        )
    )
    wrap_model = model.init_wrap_model(hf_model)

    with model.get_kvcache_mixin().kv_cache_scope(device="cpu"):
        preprocess = model._get_data_preprocessor()
        model_inputs = preprocess(
            {
                "input_ids": torch.tensor([[1, 10, 2, 11]], dtype=torch.long),
                "image_embeds": torch.randn(1, 64),
                "audio_embeds": torch.randn(1, 64),
                "past_seq_length": 0,
            }
        )
        logits = wrap_model(*model_inputs)

    assert logits.shape == (1, 1, 128)
    assert model.kvcache_config.num_layers == 3
    assert model.get_export_cfg()["input_names"][:5] == [
        "per_layer_inputs",
        "inputs_embeds",
        "position_ids",
        "past_seq_length",
        "current_input_length",
    ]
    assert model.get_export_cfg()["input_names"][5:7] == [
        "local_attention_mask",
        "global_attention_mask",
    ]


def test_gemma4_wrapped_rmsnorm_matches_hf_float32_norm():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    hf_model = _build_tiny_gemma4_model()
    hf_norm = hf_model.model.language_model.layers[0].input_layernorm
    hidden_states = torch.randn(2, 5, hf_model.config.text_config.hidden_size, dtype=torch.float16) * 8
    expected = hf_norm(hidden_states)

    model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="tiny_gemma4",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(MODEL_DIR),
            context_max_length=16,
            prefill_chunk_length=4,
        )
    )
    wrap_model = model.init_wrap_model(hf_model)

    actual = wrap_model.language_model.layers[0].input_layernorm(hidden_states)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-3)


def test_gemma4_kvcache_uses_per_layer_head_dims_without_padding():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    hf_model = _build_tiny_gemma4_model()
    model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="tiny_gemma4",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(MODEL_DIR),
            context_max_length=16,
            prefill_chunk_length=4,
        )
    )
    wrap_model = model.init_wrap_model(hf_model)

    expected_shapes = []
    for layer in wrap_model.language_model.layers:
        attn = layer.self_attn
        if getattr(attn, "is_kv_shared_layer", False):
            continue
        expected_shapes.append([1, attn.k_proj.out_features // attn.head_dim, 16, attn.head_dim])

    assert model.get_kvcache_mixin().layer_kv_shapes == expected_shapes
    assert model.kvcache_config.num_layers == len(expected_shapes)
    assert {shape[-1] for shape in expected_shapes} == {16, 32}


def test_gemma4_moe_with_mask_kvcache_uses_per_layer_head_dims_without_padding(tmp_path):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig, Gemma4Config, Gemma4VisionConfig

    from xhmodel_merak.xh_llm.models.gemma4_moe.gemma4_moe_with_mask_model import XHGemma4MoeWithMaskModel
    from xhmodel_merak.xh_llm.models.gemma4_moe.xh_gemma4_moe_config import XHGemma4MoeWithMaskConfig

    config = Gemma4Config()
    text = config.text_config
    if config.vision_config is None:
        config.vision_config = Gemma4VisionConfig()
    if config.audio_config is None:
        config.audio_config = Gemma4AudioConfig()
    vision = config.vision_config
    audio = config.audio_config

    text.hidden_size = 64
    text.intermediate_size = 128
    text.hidden_size_per_layer_input = 8
    text.num_hidden_layers = 4
    text.num_attention_heads = 4
    text.num_key_value_heads = 2
    text.head_dim = 16
    text.global_head_dim = 32
    text.num_global_key_value_heads = 1
    text.num_kv_shared_layers = 0
    text.sliding_window = 8
    text.layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
    text.vocab_size = 128
    text.max_position_embeddings = 64
    text.bos_token_id = 2
    text.eos_token_id = 1
    text.pad_token_id = 0

    vision.hidden_size = 32
    vision.intermediate_size = 64
    vision.num_hidden_layers = 1
    vision.num_attention_heads = 4
    vision.embedding_dim = 32

    audio.hidden_size = 32
    audio.intermediate_size = 64
    audio.num_hidden_layers = 1
    audio.num_attention_heads = 4
    audio.num_key_value_heads = 4
    audio.head_dim = 8
    audio.input_dim = 128

    config.image_token_id = 10
    config.audio_token_id = 11
    config.video_token_id = 12
    config.boi_token_id = 13
    config.eoi_token_id = 14
    config.boa_token_id = 15
    config.eoa_token_id = 16

    hf_model_dir = tmp_path / "gemma4_moe_config"
    config.save_pretrained(hf_model_dir)
    hf_model = Gemma4ForConditionalGeneration(config).eval()

    model = XHGemma4MoeWithMaskModel(
        XHGemma4MoeWithMaskConfig(
            model_name="tiny_gemma4_moe_with_mask",
            model_type="Gemma4ForConditionalGeneration_with_mask",
            hf_model=str(hf_model_dir),
            context_max_length=16,
            prefill_chunk_length=4,
        )
    )
    wrap_model = model.init_wrap_model(hf_model)

    expected_shapes = []
    for layer in wrap_model.model.layers:
        attn = layer.self_attn
        expected_shapes.append([1, attn.k_proj.out_features // attn.head_dim, 16, attn.head_dim])

    assert model.get_kvcache_mixin().layer_kv_shapes == expected_shapes
    assert model.kvcache_config.num_layers == len(expected_shapes)
    assert {shape[-1] for shape in expected_shapes} == {16, 32}

    with model.get_kvcache_mixin().kv_cache_scope(device="cpu"):
        actual_shapes = [list(cache.shape) for cache in model.past_key_caches]

    assert actual_shapes == expected_shapes


def test_gemma4_moe_with_mask_hmonnx_kvcache_mixin_uses_exported_layer_shapes():
    from xhmodel_merak.xh_llm.models.gemma4_moe.gemma4_moe_hmonnx_inference import Gemma4MoeKVCacheMixinHMONNX
    from xhmodel_merak.xh_llm.types import KVCacheConfig

    layer_kv_shapes = [[1, 4, 16, 16], [1, 2, 16, 32]]
    mixin = Gemma4MoeKVCacheMixinHMONNX(KVCacheConfig(num_layers=2, kv_cache_shape=layer_kv_shapes[0]), layer_kv_shapes)

    with mixin.kv_cache_scope(device="cpu"):
        key_shapes = [list(cache.shape) for cache in mixin.past_key_caches]
        value_shapes = [list(cache.shape) for cache in mixin.past_value_caches]

    assert key_shapes == layer_kv_shapes
    assert value_shapes == layer_kv_shapes


def test_gemma4_preprocess_externalizes_per_layer_inputs():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    hf_model = _build_tiny_gemma4_model()
    model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="tiny_gemma4",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(MODEL_DIR),
            context_max_length=16,
            prefill_chunk_length=5,
        )
    )
    wrap_model = model.init_wrap_model(hf_model)

    with model.get_kvcache_mixin().kv_cache_scope(device="cpu"):
        preprocess = model._get_data_preprocessor()
        model_inputs = preprocess(
            {
                "input_ids": torch.tensor([[1, 10, 2, 11, 12]], dtype=torch.long),
                "image_embeds": torch.randn(1, 64),
                "audio_embeds": torch.randn(1, 64),
                "video_embeds": torch.randn(1, 64),
                "past_seq_length": 0,
            }
        )

    llm_input_ids = preprocess._build_llm_input_ids(
        preprocess._pad_input_ids(torch.tensor([[1, 10, 2, 11, 12]], dtype=torch.long))
    )
    expected_inputs_embeds = model_inputs[1]
    expected_per_layer_inputs = wrap_model.language_model._project_per_layer_inputs(
        expected_inputs_embeds,
        wrap_model.language_model._get_per_layer_inputs(llm_input_ids),
    )

    assert model_inputs[0].shape == (1, 4, 5, 8)
    assert model_inputs[0].shape[1] == hf_model.config.text_config.num_hidden_layers
    assert model_inputs[0].shape[2] == model.wrap_cfg.input_sequence_length
    assert torch.equal(llm_input_ids, torch.tensor([[1, 0, 2, 0, 0]], dtype=torch.long))
    assert not torch.allclose(model_inputs[1][0, 1], wrap_model.language_model.embed_tokens(llm_input_ids)[0, 1])
    assert not torch.allclose(model_inputs[1][0, 3], wrap_model.language_model.embed_tokens(llm_input_ids)[0, 3])
    assert not torch.allclose(model_inputs[1][0, 4], wrap_model.language_model.embed_tokens(llm_input_ids)[0, 4])
    torch.testing.assert_close(model_inputs[0], expected_per_layer_inputs, rtol=0.0, atol=5e-3)
    assert model.get_export_cfg()["input_names"][:5] == [
        "per_layer_inputs",
        "inputs_embeds",
        "position_ids",
        "past_seq_length",
        "current_input_length",
    ]
    assert model.get_export_cfg()["input_names"][5:7] == [
        "local_attention_mask",
        "global_attention_mask",
    ]


def test_gemma4_wrap_uses_text_pad_token_for_multimodal_placeholders():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    hf_model = _build_tiny_gemma4_model()
    model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="tiny_gemma4",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(MODEL_DIR),
            context_max_length=16,
            prefill_chunk_length=5,
        )
    )
    model.init_wrap_model(hf_model)

    preprocess = model._get_data_preprocessor()

    assert model.pad_token_id == hf_model.config.text_config.pad_token_id
    assert torch.equal(
        preprocess._build_llm_input_ids(torch.tensor([[1, 10, 2, 11, 12]], dtype=torch.long)),
        torch.tensor(
            [
                [
                    1,
                    hf_model.config.text_config.pad_token_id,
                    2,
                    hf_model.config.text_config.pad_token_id,
                    hf_model.config.text_config.pad_token_id,
                ]
            ]
        ),
    )


def test_gemma4_frontend_conversion_accepts_externalized_per_layer_inputs():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="tiny_gemma4",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(MODEL_DIR),
            context_max_length=16,
            prefill_chunk_length=5,
        )
    )
    hf_model = _build_tiny_gemma4_model()
    model.init_wrap_model(hf_model)

    model.to_fronted()

    assert model.frontend_model is not None
    placeholders = [node for node in model.frontend_model.graph.nodes if node.op == "placeholder"]

    assert placeholders[0].target == "per_layer_inputs"
    per_layer_shape = placeholders[0].meta["tensor_meta"].shape
    assert len(per_layer_shape) == 4
    assert per_layer_shape[0] == 1
    assert per_layer_shape[2] == model.wrap_cfg.input_sequence_length
    assert placeholders[1].target == "inputs_embeds"
    inputs_embeds_shape = placeholders[1].meta["tensor_meta"].shape
    assert inputs_embeds_shape[0] == 1
    assert inputs_embeds_shape[1] == model.wrap_cfg.input_sequence_length
    assert per_layer_shape[2] == inputs_embeds_shape[1]
    assert per_layer_shape[1] != inputs_embeds_shape[1]
    assert placeholders[2].target == "position_ids"
    assert placeholders[2].meta["tensor_meta"].shape == torch.Size([1, model.wrap_cfg.input_sequence_length])


def test_gemma4_frontend_conversion_avoids_unconverted_cast_methods():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="tiny_gemma4",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(MODEL_DIR),
            context_max_length=16,
            prefill_chunk_length=4,
        )
    )
    model.init_wrap_model(_build_tiny_gemma4_model())

    model.to_fronted()

    unsupported_cast_nodes = [
        (node.name, node.target)
        for node in model.frontend_model.graph.nodes
        if node.op == "call_method" and node.target in {"type_as", "float", "double", "half", "to"}
    ]

    assert unsupported_cast_nodes == []


def test_gemma4_frontend_conversion_fuses_group_qk_matmul():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="tiny_gemma4",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(MODEL_DIR),
            context_max_length=16,
            prefill_chunk_length=4,
        )
    )
    model.init_wrap_model(_build_tiny_gemma4_model())

    model.to_fronted()

    group_matmul_nodes = []
    repeat_interleave_nodes = []
    for node in model.frontend_model.graph.nodes:
        if node.op != "call_module":
            continue
        module = model.frontend_model.get_submodule(node.target)
        if isinstance(module, xhnn.GroupMatMul):
            group_matmul_nodes.append(node.name)
        if isinstance(module, xhnn.RepeatInterleave):
            repeat_interleave_nodes.append(node.name)

    assert group_matmul_nodes
    assert repeat_interleave_nodes == []


def test_gemma4_frontend_conversion_has_no_cache_padding_nodes():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="tiny_gemma4",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(MODEL_DIR),
            context_max_length=16,
            prefill_chunk_length=4,
        )
    )
    model.init_wrap_model(_build_tiny_gemma4_model())
    model.to_fronted()

    pad_nodes = [
        (node.name, node.target)
        for node in model.frontend_model.graph.nodes
        if node.op == "call_function" and getattr(node.target, "__name__", None) == "pad"
    ]

    assert pad_nodes == []


def test_gemma4_rotary_cache_matches_model_dtype():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    hf_model = _build_tiny_gemma4_model().half()
    rotary_emb = hf_model.model.language_model.rotary_emb
    rotary_emb.full_attention_inv_freq = rotary_emb.full_attention_inv_freq.float()
    rotary_emb.sliding_attention_inv_freq = rotary_emb.sliding_attention_inv_freq.float()
    model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="tiny_gemma4",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(MODEL_DIR),
            context_max_length=16,
            prefill_chunk_length=4,
        )
    )
    wrap_model = model.init_wrap_model(hf_model)

    rotary_emb = wrap_model.language_model.rotary_emb
    model_dtype = wrap_model.language_model.embed_tokens.weight.dtype

    assert rotary_emb.full_attention_cos_cached.dtype == model_dtype
    assert rotary_emb.full_attention_sin_cached.dtype == model_dtype
    assert rotary_emb.sliding_attention_cos_cached.dtype == model_dtype
    assert rotary_emb.sliding_attention_sin_cached.dtype == model_dtype


def test_gemma4_attention_setup_tracks_sliding_window_cache_and_explicit_mask_ops():
    from xhquant.nn import LLMCacheV2, MaskedAdd, SoftmaxPlus

    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="tiny_gemma4",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(MODEL_DIR),
            context_max_length=16,
            prefill_chunk_length=4,
        )
    )
    wrap_model = model.init_wrap_model(_build_tiny_gemma4_model())

    sliding_attn = wrap_model.language_model.layers[0].self_attn
    full_attn = wrap_model.language_model.layers[1].self_attn

    assert isinstance(sliding_attn.masked_add, MaskedAdd)
    assert isinstance(full_attn.masked_add, MaskedAdd)
    assert isinstance(sliding_attn.softmax, SoftmaxPlus)
    assert isinstance(full_attn.softmax, SoftmaxPlus)
    assert isinstance(sliding_attn.k_cache, LLMCacheV2)
    assert isinstance(sliding_attn.v_cache, LLMCacheV2)
    assert isinstance(full_attn.k_cache, LLMCacheV2)
    assert isinstance(full_attn.v_cache, LLMCacheV2)
    assert sliding_attn.k_cache.attention_max_length == wrap_model.language_model.config.sliding_window
    assert sliding_attn.v_cache.attention_max_length == wrap_model.language_model.config.sliding_window
    assert full_attn.k_cache.attention_max_length == -1
    assert full_attn.v_cache.attention_max_length == -1


def test_gemma4_quant_cfg_maps_nodes_and_ops_for_xhquant():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="tiny_gemma4",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(MODEL_DIR),
            context_max_length=16,
            prefill_chunk_length=4,
            quant_scheme=dict(
                quant_type="w8a8h1_sefp",
                nodes={"lm_head": "w16a16h1_sefp"},
                ops={"Linear": "w16a16h1_sefp"},
            ),
        )
    )

    quant_cfg = model.get_quant_cfg()

    assert "nodes" not in quant_cfg
    assert "ops" not in quant_cfg
    assert quant_cfg["nodes_cfg"]["lm_head"]["quant_type"] == "w16a16h1_sefp"
    assert quant_cfg["ops_cfg"]["Linear"]["quant_type"] == "w16a16h1_sefp"


def test_gemma4_quant_cfg_keeps_attention_projections_in_global_quant_type():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="tiny_gemma4",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(MODEL_DIR),
            context_max_length=16,
            prefill_chunk_length=4,
            quant_scheme=dict(quant_type="w8a8h1_sefp"),
        )
    )
    model.init_wrap_model(_build_tiny_gemma4_model())
    model.to_fronted()

    quant_cfg = model.get_quant_cfg()
    attention_projection_nodes = {
        node.name
        for node in model.frontend_model.graph.nodes
        if node.op == "call_module"
        and str(node.target).endswith(("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"))
    }

    assert attention_projection_nodes
    assert quant_cfg["quant_type"] == "w8a8h1_sefp"
    nodes_cfg = quant_cfg.get("nodes_cfg", {})
    assert all(node_name not in nodes_cfg for node_name in attention_projection_nodes)


def test_gemma4_hmonnx_rebuilds_scaled_token_embedding(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_hmonnx_inference import XHGemma4_HMONNXModel

    hf_model = _build_tiny_gemma4_model()
    hf_model.config.save_pretrained(tmp_path / "hf_config")
    quant_embedding_path = tmp_path / "quant_embedding.pt"
    torch.save(hf_model.model.language_model.embed_tokens.state_dict(), quant_embedding_path)
    meta = SimpleNamespace(hf_config=str(tmp_path / "hf_config"), quant_embedding=str(quant_embedding_path))

    rebuilt_embedding = XHGemma4_HMONNXModel._build_embed_tokens_from_meta(meta)
    input_ids = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)

    expected = hf_model.model.language_model.embed_tokens(input_ids).float()
    actual = rebuilt_embedding(input_ids).float()

    assert torch.allclose(actual, expected)


def test_gemma4_constant_rotary_matches_hf_helper():
    from xhmodel_merak.xh_llm.models.gemma4e._llm_model_impl import _apply_rotary_pos_emb_with_constant_dim

    hidden_states = torch.randn(1, 4, 2, 16)
    cos = torch.randn(1, 4, 16)
    sin = torch.randn(1, 4, 16)

    expected = apply_rotary_pos_emb(hidden_states, cos, sin, unsqueeze_dim=2)
    actual = _apply_rotary_pos_emb_with_constant_dim(hidden_states, cos, sin, unsqueeze_dim=2, half_dim=8)

    assert torch.allclose(actual, expected)


def test_gemma4_text_rotary_export_avoids_trig_ops(tmp_path):
    import onnx

    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    class _RotaryExportBridge(nn.Module):
        def __init__(self, rotary_emb: nn.Module):
            super().__init__()
            self.rotary_emb = rotary_emb

        def forward(self, hidden_states, position_ids):
            cos, sin = self.rotary_emb(hidden_states, position_ids, "full_attention")
            return cos, sin

    hf_model = _build_tiny_gemma4_model()
    xh_model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="tiny_gemma4",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(MODEL_DIR),
            context_max_length=16,
            prefill_chunk_length=4,
        )
    )
    wrap_model = xh_model.init_wrap_model(hf_model)
    rotary_bridge = _RotaryExportBridge(wrap_model.language_model.rotary_emb).eval()

    export_path = tmp_path / "gemma4_rotary.onnx"
    torch.onnx.export(
        rotary_bridge,
        (torch.zeros((1, 4, 64), dtype=torch.float32), torch.arange(4, dtype=torch.int64).unsqueeze(0)),
        export_path,
        input_names=["hidden_states", "position_ids"],
        output_names=["cos", "sin"],
        opset_version=17,
    )

    exported_model = onnx.load(export_path)
    op_types = {node.op_type for node in exported_model.graph.node}

    assert "Cos" not in op_types
    assert "Sin" not in op_types


def test_gemma4_hf_compatible_drops_meta_backed_text_modules():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import build_gemma4_hf_compatible_model

    class _StubXHModel:
        def get_input_embeddings(self):
            return nn.Embedding(8, 8)

    config = copy.deepcopy(AutoConfig.from_pretrained(str(MODEL_DIR)))
    with init_empty_weights():
        hf_model = Gemma4ForConditionalGeneration(config).eval()

    compatible_model = build_gemma4_hf_compatible_model(hf_model, _StubXHModel())
    meta_params = [name for name, param in compatible_model.named_parameters() if getattr(param, "is_meta", False)]

    assert not hasattr(getattr(compatible_model, "model", None), "language_model")
    assert not hasattr(compatible_model, "lm_head")
    assert meta_params == []


def test_gemma4_hf_compatible_accepts_processor_generation_kwargs():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import build_gemma4_hf_compatible_model

    class _StubXHModel:
        def get_input_embeddings(self):
            return nn.Embedding(128, 64)

    compatible_model = build_gemma4_hf_compatible_model(_build_tiny_gemma4_model(), _StubXHModel())
    compatible_model._xh_orig_forward = compatible_model.forward
    compatible_model.forward = compatible_model._sample_forward

    try:
        compatible_model._validate_model_kwargs(
            {
                "image_position_ids": torch.zeros((1, 1, 2), dtype=torch.long),
                "mm_token_type_ids": torch.zeros((1, 1), dtype=torch.long),
            }
        )
    finally:
        compatible_model.forward = compatible_model._xh_orig_forward
        del compatible_model._xh_orig_forward


def test_gemma4_hf_compatible_trims_audio_embeds_by_mask():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import build_gemma4_hf_compatible_model

    recorded = {}
    kept_audio_embeds = torch.tensor([[1.0, 2.0], [5.0, 6.0]], dtype=torch.float32)

    class _FakeAudio:
        device = "cpu"
        dtype = torch.float32

        def forward(self, input_features, input_features_mask):
            del input_features, input_features_mask
            return (
                torch.tensor(
                    [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]],
                    dtype=torch.float32,
                ),
                torch.tensor([[True, False, True]]),
            )

    class _RecordingProcessor:
        input_sequence_length = 4

        def __call__(self, data):
            recorded["audio_embeds"] = data["audio_embeds"]
            return (
                torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                torch.zeros((1, 1, 64), dtype=torch.float32),
                torch.zeros((1, 1), dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([1], dtype=torch.int32),
                torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                torch.zeros((1, 1, 1, 1), dtype=torch.float32),
                [],
                [],
            )

    class _StubXHModel:
        def __init__(self):
            self.audio = _FakeAudio()
            self.visual = None
            self.data_processor = _RecordingProcessor()

        def get_input_embeddings(self):
            return nn.Embedding(128, 64)

        def get_data_preprocessor(self):
            return self.data_processor

        def get_input_sequence_length(self):
            return self.data_processor.input_sequence_length

        def forward(self, *args):
            del args
            return torch.zeros((1, 1, 128), dtype=torch.float32)

    compatible_model = build_gemma4_hf_compatible_model(_build_tiny_gemma4_model(), _StubXHModel())

    compatible_model.forward(
        input_ids=torch.tensor([[1]], dtype=torch.long),
        input_features=torch.randn(1, 8, 128),
        input_features_mask=torch.ones((1, 8), dtype=torch.long),
    )

    assert torch.equal(recorded["audio_embeds"], kept_audio_embeds)


def test_gemma4_hf_compatible_accepts_experts_implementation_hooks():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import build_gemma4_hf_compatible_model

    class _StubXHModel:
        def get_input_embeddings(self):
            return nn.Embedding(128, 64)

    compatible_model = build_gemma4_hf_compatible_model(_build_tiny_gemma4_model(), _StubXHModel())

    assert compatible_model.get_correct_experts_implementation("grouped_mm") == "grouped_mm"

    compatible_model.set_experts_implementation("grouped_mm")

    assert compatible_model.config.experts_implementation == "grouped_mm"
    assert compatible_model._grouped_mm_can_dispatch() is False


def test_gemma4_hf_compatible_prefill_chunks_long_prompt():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import build_gemma4_hf_compatible_model

    class _ChunkingStubDataProcessor:
        def __init__(self, input_sequence_length: int):
            self.input_sequence_length = input_sequence_length
            self.calls: list[tuple[int, int]] = []

        def __call__(self, data):
            input_ids = data["input_ids"]
            seq_length = input_ids.shape[1]
            past_seq_length = int(data["past_seq_length"])
            self.calls.append((seq_length, past_seq_length))
            if seq_length > self.input_sequence_length:
                raise ValueError(
                    f"Input sequence length is too long. max input sequence length is {self.input_sequence_length} but got {seq_length}"
                )

            padded_input_ids = torch.full((1, self.input_sequence_length), 0, dtype=torch.int32)
            padded_input_ids[:, :seq_length] = input_ids.to(torch.int32)
            inputs_embeds = torch.zeros((1, self.input_sequence_length, 8), dtype=torch.float32)
            position_ids = torch.arange(self.input_sequence_length, dtype=torch.int32).unsqueeze(0)
            return (
                padded_input_ids,
                inputs_embeds,
                position_ids,
                torch.tensor([past_seq_length], dtype=torch.int32),
                torch.tensor([seq_length], dtype=torch.int32),
                [],
                [],
            )

    class _ChunkingStubXHModel:
        def __init__(self):
            self.embed_tokens = nn.Embedding(128, 8)
            self.data_processor = _ChunkingStubDataProcessor(input_sequence_length=4)
            self.forward_calls: list[tuple[int, int]] = []
            self.visual = None
            self.audio = None

        def get_input_embeddings(self):
            return self.embed_tokens

        def get_data_preprocessor(self):
            return self.data_processor

        def get_input_sequence_length(self):
            return self.data_processor.input_sequence_length

        def get_num_logits_to_keep(self):
            return 0

        def forward(self, *args):
            inputs_embeds = args[1]
            past_seq_length = int(args[3].item())
            current_input_length = int(args[4].item())
            self.forward_calls.append((past_seq_length, current_input_length))
            vocab_size = 16
            logits = torch.full((1, inputs_embeds.shape[1], vocab_size), float(past_seq_length))
            return logits

    compatible_model = build_gemma4_hf_compatible_model(_build_tiny_gemma4_model(), _ChunkingStubXHModel())

    outputs = compatible_model.forward(input_ids=torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long))

    assert compatible_model._llm_model.data_processor.calls == [(4, 0), (1, 4)]
    assert compatible_model._llm_model.forward_calls == [(0, 4), (4, 1)]
    assert outputs.logits.shape == (1, 5, 16)


def test_gemma4_hf_compatible_trims_visual_features_with_exported_mask():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import build_gemma4_hf_compatible_model

    class _MaskedVisualStub:
        device = torch.device("cpu")
        dtype = torch.float32

        def forward(self, pixel_values, image_position_ids):
            del pixel_values, image_position_ids
            hidden_size = 8
            image_embeds = torch.arange(32, dtype=torch.float32).reshape(1, 4, hidden_size)
            image_mask = torch.tensor([[True, False, True, True]])
            return image_embeds, image_mask

    class _DataProcessorStub:
        def __call__(self, data):
            image_embeds = data["image_embeds"]
            assert image_embeds.shape == (3, 8)
            return (
                torch.zeros((1, 4), dtype=torch.int32),
                torch.zeros((1, 4, 8), dtype=torch.float32),
                torch.zeros((1, 4), dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([4], dtype=torch.int32),
                torch.zeros((1, 1, 4, 8), dtype=torch.float32),
                torch.zeros((1, 1, 4, 8), dtype=torch.float32),
                [],
                [],
            )

    class _StubXHModel:
        def __init__(self):
            self.visual = _MaskedVisualStub()
            self.audio = None
            self.data_processor = _DataProcessorStub()

        def get_input_embeddings(self):
            return nn.Embedding(128, 8)

        def get_data_preprocessor(self):
            return self.data_processor

        def get_input_sequence_length(self):
            return 4

        def forward(self, *args):
            return torch.zeros((1, 4, 16), dtype=torch.float32)

    compatible_model = build_gemma4_hf_compatible_model(_build_tiny_gemma4_model(), _StubXHModel())

    outputs = compatible_model.forward(
        input_ids=torch.tensor([[10, 10, 1, 10]], dtype=torch.long),
        pixel_values=torch.zeros((1, 4, 8), dtype=torch.float32),
        image_position_ids=torch.zeros((1, 4, 2), dtype=torch.long),
    )

    assert outputs.logits.shape == (1, 4, 16)


def test_gemma4_hf_compatible_compact_visual_omits_position_ids():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import build_gemma4_hf_compatible_model

    recorded = {}

    class _CompactVisualStub:
        device = torch.device("cpu")
        dtype = torch.float32
        export_mode = "compact"

        def forward(self, pixel_values):
            recorded["pixel_values_shape"] = tuple(pixel_values.shape)
            return torch.zeros((256, 8), dtype=torch.float32)

    class _DataProcessorStub:
        def __call__(self, data):
            recorded["image_embeds_shape"] = tuple(data["image_embeds"].shape)
            return (
                torch.zeros((1, 4), dtype=torch.int32),
                torch.zeros((1, 4, 8), dtype=torch.float32),
                torch.zeros((1, 4), dtype=torch.int32),
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([4], dtype=torch.int32),
                torch.zeros((1, 1, 4, 8), dtype=torch.float32),
                torch.zeros((1, 1, 4, 8), dtype=torch.float32),
                [],
                [],
            )

    class _StubXHModel:
        def __init__(self):
            self.visual = _CompactVisualStub()
            self.audio = None
            self.data_processor = _DataProcessorStub()
            self.config = SimpleNamespace(visual_config=SimpleNamespace(export_mode="compact"))

        def get_input_embeddings(self):
            return nn.Embedding(128, 8)

        def get_data_preprocessor(self):
            return self.data_processor

        def get_input_sequence_length(self):
            return 4

        def forward(self, *args):
            return torch.zeros((1, 4, 16), dtype=torch.float32)

    compatible_model = build_gemma4_hf_compatible_model(_build_tiny_gemma4_model(), _StubXHModel())

    outputs = compatible_model.forward(
        input_ids=torch.tensor([[10, 10, 1, 10]], dtype=torch.long),
        pixel_values=torch.zeros((1, 2304, 8), dtype=torch.float32),
        image_position_ids=torch.zeros((1, 2304, 2), dtype=torch.long),
    )

    assert outputs.logits.shape == (1, 4, 16)
    assert recorded["pixel_values_shape"] == (1, 2304, 8)
    assert recorded["image_embeds_shape"] == (256, 8)
