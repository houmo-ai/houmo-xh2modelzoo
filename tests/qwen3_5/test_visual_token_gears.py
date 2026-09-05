import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _visual_export_config(**overrides):
    from xhmodel_merak.xh_llm.models.qwen3_5.xh_qwen3_5_config import XHQwen3_5_VisualConfig

    values = {
        "model_name": "qwen3_5_visual",
        "visual_input_mode": "patches",
        "image_token_gears": [96, 196, 384, 704, 1536],
        "image_token_capacity": 1536,
        "spatial_merge_size": 2,
    }
    values.update(overrides)
    return XHQwen3_5_VisualConfig(**values)


def test_visual_config_only_accepts_patch_token_gears():
    config = _visual_export_config()

    assert config.visual_input_mode == "patches"
    assert config.image_token_gears == [96, 196, 384, 704, 1536]
    assert config.image_token_capacity == 1536
    assert config.visual_rope_cache_length == 3072
    assert not hasattr(config, "max_size_w")
    assert not hasattr(config, "max_size_h")

    with pytest.raises(ValueError, match="only supports 'patches'"):
        _visual_export_config(visual_input_mode="image")
    with pytest.raises(TypeError, match="no longer accepts fixed image sizes"):
        _visual_export_config(max_size_w=448, max_size_h=448)


def test_full_export_directory_name_contains_visual_token_gears(monkeypatch, tmp_path):
    from xhmodel_merak.xh_llm.models.qwen3_5 import qwen3_5_llm_model as model_module

    class FixedDatetime:
        @classmethod
        def now(cls):
            return SimpleNamespace(strftime=lambda _format: "20260825")

    monkeypatch.setattr(model_module, "datetime", FixedDatetime)
    model = model_module.XHQwen3_5Model.__new__(model_module.XHQwen3_5Model)
    model.config = SimpleNamespace(model_name="xh2_qwen3_8_27b_w4a8_256_256k_mpe256k")
    model.visual = SimpleNamespace(
        config=SimpleNamespace(image_token_gears=[96, 196, 384, 704, 1536])
    )
    model.create_export_metadata = lambda _output_dir: SimpleNamespace()

    export_info = model.get_export_info(tmp_path)

    expected_name = (
        "hmquant_xh2_qwen3_8_27b_w4a8_256_256k_mpe256k_"
        "visualm96_196_384_704_1536_20260825"
    )
    assert export_info.model_name == expected_name
    assert export_info.exported_dir == str(tmp_path / expected_name)


def test_visual_export_metadata_resolves_flat_gears_after_relocation(tmp_path):
    from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_vision_model import XHQwen3_5VisionModel
    from xhmodel_merak.xh_llm.types import VisualModelMeta

    class ExportFixture(XHQwen3_5VisionModel):
        def __init__(self, config):
            self.config = config

        def create_export_metadata(self, _output_dir):
            return VisualModelMeta()

        def _export_single_hmonnx(self, output_dir):
            graph = Path(output_dir) / f"{self.config.model_name}.onnx"
            graph.parent.mkdir(parents=True)
            graph.touch()
            meta = VisualModelMeta()
            meta.hmonnx = str(graph)
            return meta

    gears = [96, 196, 384, 704, 1536]
    model = ExportFixture(_visual_export_config())
    output_dir = tmp_path / "visual"
    model.export_hmonnx(str(output_dir))
    relocated = tmp_path / "relocated_visual"
    output_dir.rename(relocated)

    metadata = json.loads((relocated / "visual_meta_info.json").read_text())
    manifest = json.loads((relocated / metadata["gear_manifest"]).read_text())
    expected_paths = [f"m{gear}/qwen3_5_visual_m{gear}.onnx" for gear in gears]
    assert [entry["hmonnx"] for entry in metadata["gears"]] == expected_paths
    assert [entry["hmonnx"] for entry in manifest["gears"]] == expected_paths
    assert metadata["hmonnx"] == expected_paths[-1]
    assert sorted(path.name for path in relocated.iterdir() if path.is_dir()) == sorted(
        f"m{gear}" for gear in gears
    )
    assert all((relocated / path).is_file() for path in expected_paths)


def _tiny_vision_config():
    from xhmodel_merak.xh_llm.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig

    config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=128,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=256,
        num_heads=2,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=32,
        num_position_embeddings=16,
    )
    config._attn_implementation = "eager"
    return config


def test_visual_token_geometry_matches_native_position_and_rope():
    from xhmodel_merak.xh_llm.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    from xhmodel_merak.xh_llm.models.qwen3_5.visual_token_gears import build_visual_token_gear_inputs

    torch.manual_seed(17)
    config = _tiny_vision_config()
    model = Qwen3_5VisionModel(config).eval()
    grid_thw = torch.tensor([[1, 4, 6]], dtype=torch.int64)
    valid_tokens = int(grid_thw.prod())
    capacity = 32
    inputs = build_visual_token_gear_inputs(
        grid_thw,
        patch_capacity=capacity,
        num_position_embeddings=config.num_position_embeddings,
        spatial_merge_size=config.spatial_merge_size,
        dtype=model.pos_embed.weight.dtype,
        rotary_cache_length=16,
    )

    gathered = model.pos_embed(inputs["position_ids"])
    actual_position = (gathered * inputs["position_weights"].unsqueeze(-1)).sum(dim=0)
    expected_position = model.fast_pos_embed_interpolate(grid_thw)
    torch.testing.assert_close(actual_position[:valid_tokens], expected_position, rtol=1e-6, atol=1e-6)
    assert torch.count_nonzero(actual_position[valid_tokens:]) == 0

    rotary = model.rot_pos_emb(grid_thw).reshape(valid_tokens, -1)
    freq_table = model.rotary_pos_emb(max(int(grid_thw[0, 1]), int(grid_thw[0, 2])))
    rotary_position_ids = inputs["rotary_position_ids"]
    assert rotary_position_ids.shape == (2, capacity)
    gathered_rotary = freq_table[rotary_position_ids[:, :valid_tokens]].permute(1, 0, 2).reshape(
        valid_tokens, -1
    )
    torch.testing.assert_close(gathered_rotary, rotary, rtol=1e-6, atol=1e-6)
    assert torch.count_nonzero(rotary_position_ids[:, valid_tokens:]) == 0
    attention_mask = inputs["attention_mask"]
    assert attention_mask.shape == (1, 1, 1, capacity)
    assert torch.count_nonzero(attention_mask[..., :valid_tokens]) == 0
    assert torch.all(attention_mask[..., valid_tokens:] == -torch.finfo(attention_mask.dtype).max)


def test_visual_gear_manifest_is_sorted_and_uses_smallest_fit_contract():
    from xhmodel_merak.xh_llm.models.qwen3_5.visual_token_gears import build_visual_gear_manifest

    manifest = build_visual_gear_manifest(
        [
            {"image_token_capacity": 512, "patch_token_capacity": 2048, "hmonnx": "m512/v.onnx"},
            {"image_token_capacity": 96, "patch_token_capacity": 384, "hmonnx": "m96/v.onnx"},
            {"image_token_capacity": 320, "patch_token_capacity": 1280, "hmonnx": "m320/v.onnx"},
        ],
        visual_rope_cache_length=1024,
    )

    assert manifest["schema_version"] == 1
    assert manifest["routing_policy"] == "smallest_fit"
    assert manifest["attention_mask_format"] == "additive_key_padding_bias"
    assert manifest["attention_mask_operator"] == "xhquant.nn.MaskedAdd+Softmax"
    assert manifest["rotary_position_format"] == "height_width_2d_ids"
    assert manifest["visual_rope_cache_length"] == 1024
    assert manifest["shared_weight_loader"] == "MultiHMONNXLoader"
    assert [gear["image_token_capacity"] for gear in manifest["gears"]] == [96, 320, 512]


def test_visual_hmonnx_router_rejects_causal_mask_contract(monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen3_5 import qwen3_5_hmonnx_inference as runtime

    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    meta = SimpleNamespace(
        gears=[SimpleNamespace(image_token_capacity=6, patch_token_capacity=24, hmonnx="m6.onnx")],
        visual_input_mode="patches",
        attention_mask_format="additive_key_padding_bias",
        attention_mask_shape="[1,1,1,patch_token_capacity]",
        attention_mask_operator="xhquant.nn.MaskedSoftmax",
        spatial_merge_size=2,
    )

    with pytest.raises(ValueError, match=r"MaskedAdd\+Softmax"):
        runtime.VisualTokenGearHMONNXModel(meta, device="cpu")


def test_qwen35_moe_runtime_selects_visual_token_gear_router(monkeypatch):
    from xhmodel_merak.xh_llm.hmonnx.vision_llm_hmonnx_model import (
        VisonLLMHMONNXModel,
    )
    from xhmodel_merak.xh_llm.models.qwen3_5_moe import (
        qwen3_5_moe_hmonnx_inference as runtime,
    )

    visual_meta = SimpleNamespace(gears=[SimpleNamespace(patch_token_capacity=384)])
    meta_info = SimpleNamespace(
        visual_config=visual_meta,
        model_config=SimpleNamespace(split_conv_cache=False),
    )
    selected = object()

    def fake_base_init(self, _meta_info, **_kwargs):
        self.prefill_model = SimpleNamespace(device=torch.device("cpu"))
        self.kvcache_config = SimpleNamespace()

    class FakeCacheMixin:
        split_conv_cache = False

        def __init__(self, _config):
            pass

    def fake_router(actual_meta, *, device, enable_golden=False):
        assert actual_meta is visual_meta
        assert device == torch.device("cpu")
        assert enable_golden is True
        return selected

    monkeypatch.setattr(VisonLLMHMONNXModel, "__init__", fake_base_init)
    monkeypatch.setattr(runtime, "Qwen3_5HMONNXKVCacheMixin", FakeCacheMixin)
    monkeypatch.setattr(runtime, "VisualTokenGearHMONNXModel", fake_router)
    monkeypatch.setattr(runtime.XHQwen3_5MoeHMONNXModel, "_sync_page_attention_mode_to_kvcache", lambda self: None)

    model = runtime.XHQwen3_5MoeHMONNXModel(meta_info, enable_golden=True)

    assert model.visual is selected


def test_visual_token_graph_matches_native_and_masks_padding():
    from xhmodel_merak.xh_llm.models.qwen3_5 import _vision_model_impl  # noqa: F401
    from xhmodel_merak.xh_llm.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel
    from xhmodel_merak.xh_llm.models.qwen3_5.visual_token_gears import (
        build_visual_token_gear_inputs,
        pad_flattened_patches,
    )
    from xhmodel_merak.xh_llm.wrap_model import wrap_llm_model
    from xhquant.api import ConfigDict, FrontendType, to_frontend_graph
    from xhquant.nn import Clip, MaskedAdd, Rope, Softmax

    torch.manual_seed(23)
    config = _tiny_vision_config()
    native = Qwen3_5VisionModel(config).eval()
    wrapped = copy.deepcopy(native)
    image_token_capacity = 6
    patch_capacity = image_token_capacity * config.spatial_merge_size**2
    wrap_cfg = ConfigDict(
        dict(
            max_size_t=2,
            patch_size=config.patch_size,
            temporal_patch_size=config.temporal_patch_size,
            spatial_merge_size=config.spatial_merge_size,
            visual_input_mode="patches",
            image_token_capacity=image_token_capacity,
            visual_rope_cache_length=image_token_capacity * config.spatial_merge_size,
        )
    )
    wrapped = wrap_llm_model(wrapped, wrap_cfg).eval()

    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    valid_patch_tokens = int(grid_thw.prod())
    patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
    patches = torch.randn(valid_patch_tokens, patch_dim)
    expected = native(patches, grid_thw).pooler_output
    padded, _ = pad_flattened_patches(patches, patch_capacity)
    geometry = build_visual_token_gear_inputs(
        grid_thw,
        patch_capacity=patch_capacity,
        num_position_embeddings=config.num_position_embeddings,
        spatial_merge_size=config.spatial_merge_size,
        dtype=patches.dtype,
        rotary_cache_length=wrap_cfg.visual_rope_cache_length,
    )

    actual = wrapped(
        padded,
        geometry["position_ids"],
        geometry["position_weights"],
        geometry["rotary_position_ids"],
        geometry["attention_mask"],
    )
    valid_image_tokens = valid_patch_tokens // config.spatial_merge_size**2
    torch.testing.assert_close(actual[0, :valid_image_tokens], expected, rtol=2e-5, atol=2e-5)

    changed_padding = padded.clone()
    changed_padding[:, valid_patch_tokens:] = torch.randn_like(changed_padding[:, valid_patch_tokens:]) * 1000
    changed = wrapped(
        changed_padding,
        geometry["position_ids"],
        geometry["position_weights"],
        geometry["rotary_position_ids"],
        geometry["attention_mask"],
    )
    torch.testing.assert_close(
        changed[0, :valid_image_tokens],
        actual[0, :valid_image_tokens],
        rtol=0,
        atol=0,
    )

    assert wrapped.patch_embed.proj_linear.bias is None
    assert wrapped.patch_embed.proj_bias.shape == (1, 1, config.hidden_size)
    assert not hasattr(wrapped.patch_embed, "proj")
    assert isinstance(wrapped.blocks[0].attn.masked_add, MaskedAdd)
    assert isinstance(wrapped.blocks[0].attn.softmax, Softmax)
    assert isinstance(wrapped.blocks[0].attn.rope, Rope)

    frontend = to_frontend_graph(
        wrapped.float().cpu(),
        FrontendType.TorchFX,
        [
            padded,
            geometry["position_ids"],
            geometry["position_weights"],
            geometry["rotary_position_ids"],
            geometry["attention_mask"],
        ],
    )
    frontend_modules = list(frontend.modules())
    assert sum(isinstance(module, MaskedAdd) for module in frontend_modules) == 1
    assert sum(isinstance(module, Rope) for module in frontend_modules) == 1
    assert not any(isinstance(module, Clip) for module in frontend_modules)
    assert not any("floordiv" in str(node.target).lower() for node in frontend.graph.nodes)


def test_visual_patch_to_fronted_routes_wrap_model_directly_to_torchfx(monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen3_5 import qwen3_5_vision_model as vision_module
    from xhquant.api import FrontendType

    class StubVisionModel:
        config = SimpleNamespace(visual_input_mode="patches")

        @staticmethod
        def get_data_preprocessor():
            return lambda _: (torch.ones(1, 2, 3), torch.ones(1, 1, 1, 2))

        @staticmethod
        def get_dummy_inputs():
            return object()

    captured = {}
    frontend = object()

    def fake_to_frontend_graph(model, frontend_type, dummy_inputs, **kwargs):
        captured["model"] = model
        captured["frontend_type"] = frontend_type
        captured["dummy_inputs"] = dummy_inputs
        captured["kwargs"] = kwargs
        return frontend

    monkeypatch.setattr(vision_module, "to_frontend_graph", fake_to_frontend_graph)
    wrap_model = torch.nn.Identity()

    actual = vision_module.XHQwen3_5VisionModel._to_fronted(StubVisionModel(), wrap_model)

    assert actual is frontend
    assert captured["model"] is wrap_model
    assert captured["frontend_type"] is FrontendType.TorchFX
    assert len(captured["dummy_inputs"]) == 2
    assert captured["kwargs"] == {}


def test_plain_onnx_export_decomposes_masked_add_with_clip(tmp_path):
    import onnx

    from xhquant import nn as xhnn

    class MaskedAttentionSoftmax(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.masked_add = xhnn.MaskedAdd()
            self.softmax = xhnn.Softmax(dim=-1)

        def forward(self, logits, attention_mask):
            return self.softmax(self.masked_add(logits, attention_mask))

    logits = torch.randn(1, 2, 8, 8)
    attention_mask = torch.zeros(1, 1, 1, 8)
    attention_mask[..., 6:] = -torch.finfo(attention_mask.dtype).max
    onnx_path = tmp_path / "masked_attention_softmax.onnx"
    torch.onnx.export(
        MaskedAttentionSoftmax().eval(),
        (logits, attention_mask),
        onnx_path,
        opset_version=18,
        input_names=["logits", "attention_mask"],
        output_names=["probabilities"],
    )

    graph = onnx.load(onnx_path).graph
    mask_adds = [node for node in graph.node if node.op_type == "Add" and "attention_mask" in node.input]
    assert len(mask_adds) == 2
    assert any(node.op_type == "Clip" for node in graph.node)
    assert sum(node.op_type == "Softmax" for node in graph.node) == 1
    assert all(node.op_type != "MaskedSoftmax" for node in graph.node)


def test_visual_hmonnx_router_groups_by_gear_and_restores_image_order(monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen3_5 import qwen3_5_hmonnx_inference as runtime

    calls = []

    class FakeLoader:
        def __init__(self, graph_files):
            self.graphs = {name: name for name in graph_files}

    class FakeSession:
        DTYPES = {
            "pixel_values": torch.float16,
            "position_ids": torch.int32,
            "position_weights": torch.float16,
            "rotary_position_ids": torch.int32,
            "attention_mask": torch.float16,
        }

        def get_input(self, name):
            return SimpleNamespace(dtype=self.DTYPES[name])

    class FakeVisualModel:
        def __init__(self, path, onnx_graph=None, **kwargs):
            self.gear = int(str(onnx_graph).removeprefix("m"))
            self.device = torch.device("cpu")
            self.hmonnx_session = FakeSession()
            self.enable_golden = False
            self.steps_advanced = 0

        def forward(self, *args):
            calls.append(self.gear)
            return torch.full((1, self.gear, 3), float(self.gear), dtype=torch.float16)

        def to(self, device):
            self.device = torch.device(device)
            return self

        def _set_dtype(self, dtype):
            return self

        def update_step(self):
            self.steps_advanced += 1

    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    monkeypatch.setattr(runtime, "MultiHMONNXLoader", FakeLoader)
    monkeypatch.setattr(runtime, "VisualHMONNXModel", FakeVisualModel)
    meta = SimpleNamespace(
        gears=[
            SimpleNamespace(image_token_capacity=6, patch_token_capacity=24, hmonnx="m6.onnx"),
            SimpleNamespace(image_token_capacity=10, patch_token_capacity=40, hmonnx="m10.onnx"),
        ],
        visual_input_mode="patches",
        attention_mask_format="additive_key_padding_bias",
        attention_mask_shape="[1,1,1,patch_token_capacity]",
        attention_mask_operator="xhquant.nn.MaskedAdd+Softmax",
        rotary_position_format="height_width_2d_ids",
        spatial_merge_size=2,
        temporal_patch_size=2,
        patch_size=2,
        in_channels=3,
        visual_rope_cache_length=20,
        hidden_size=128,
        num_heads=2,
        num_position_embeddings=16,
    )
    router = runtime.VisualTokenGearHMONNXModel(meta, device="cpu", enable_golden=True)

    grids = torch.tensor([[1, 4, 8], [1, 4, 4]], dtype=torch.int64)
    first = torch.randn(32, 24)
    second = torch.randn(16, 24)
    outputs = router.encode_many(torch.cat((first, second)), grids)

    assert calls == [6, 10]
    assert [tuple(output.shape) for output in outputs] == [(1, 8, 3), (1, 4, 3)]
    assert torch.all(outputs[0] == 10)
    assert torch.all(outputs[1] == 6)

    direct_grid = torch.tensor([[1, 4, 4]], dtype=torch.int64)
    direct_pixels, _ = runtime.pad_flattened_patches(torch.randn(16, 24), 24)
    direct_geometry = runtime.build_visual_token_gear_inputs(
        direct_grid,
        patch_capacity=24,
        num_position_embeddings=16,
        spatial_merge_size=2,
        dtype=torch.float16,
        rotary_cache_length=20,
    )
    direct = router.forward(
        direct_pixels,
        direct_geometry["position_ids"],
        direct_geometry["position_weights"],
        direct_geometry["rotary_position_ids"],
        direct_geometry["attention_mask"],
    )
    assert direct.shape == (1, 4, 3)
    assert calls == [6, 10, 6]

    assert router.run_all_gears() == {
        6: (1, 6, 3),
        10: (1, 10, 3),
    }
    assert calls == [6, 10, 6, 6, 10]
    assert all(model.enable_golden for model in router.models.values())
    router.advance_golden_steps()
    assert all(model.steps_advanced == 1 for model in router.models.values())


def test_visual_hmonnx_router_rejects_late_golden_enable(monkeypatch):
    from xhmodel_merak.xh_llm.models.qwen3_5 import qwen3_5_hmonnx_inference as runtime

    class FakeLoader:
        def __init__(self, graph_files):
            self.graphs = {name: name for name in graph_files}

    class FakeVisualModel:
        def __init__(self, _path, onnx_graph=None, **_kwargs):
            self.device = torch.device("cpu")
            self.enable_golden = False

    monkeypatch.setenv("ENABLE_HMINFERENCE_V2", "1")
    monkeypatch.setattr(runtime, "MultiHMONNXLoader", FakeLoader)
    monkeypatch.setattr(runtime, "VisualHMONNXModel", FakeVisualModel)
    meta = SimpleNamespace(
        gears=[SimpleNamespace(image_token_capacity=6, patch_token_capacity=24, hmonnx="m6.onnx")],
        visual_input_mode="patches",
        attention_mask_format="additive_key_padding_bias",
        attention_mask_shape="[1,1,1,patch_token_capacity]",
        attention_mask_operator="xhquant.nn.MaskedAdd+Softmax",
        rotary_position_format="height_width_2d_ids",
        spatial_merge_size=2,
        visual_rope_cache_length=12,
    )
    router = runtime.VisualTokenGearHMONNXModel(meta, device="cpu")

    with pytest.raises(RuntimeError, match="enabled when the token-gear runtime is constructed"):
        router.enable_golden = True
