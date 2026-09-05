import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xhmodel_merak.xh_llm.models.qwen3_5 import _qwen3_5_big_export as big_export
from xhmodel_merak.xh_llm.models.qwen3_5._hybrid_gated_delta_net import (
    _unpack_split_conv_cache_outputs,
)
from xhmodel_merak.xh_llm.models.qwen3_5._llm_model_impl import (
    _Qwen3_5DecoderLayer,
    _Qwen3_5TextAttention,
)
from xhmodel_merak.xh_llm.models.qwen3_5._qwen3_5_big_export import (
    Qwen3_5BigHFModel,
    _Qwen3_5Attention_PlaceHolder,
    _Qwen3_5GatedDeltaNet_PlaceHolder,
    _remove_registered_members,
)
from xhmodel_merak.xh_llm.models.qwen3_5.qwen3_5_llm_model import (
    XHQwen3_5Model,
    _ensure_gptq_desc_act_default,
    _resolve_qwen_export_pad_token_id,
)
from xhmodel_merak.xh_llm.models.qwen3_5_moe import _qwen3_5_moe_big_export as moe_big_export
from xhmodel_merak.xh_llm.models.qwen3_5_moe._moe_model import (
    _Qwen3_5MoeAttention,
    _Qwen3_5MoeDecoderLayer,
    _Qwen3_5MoeSparseMoeBlock,
)
from xhmodel_merak.xh_llm.models.qwen3_5_moe._qwen3_5_moe_big_export import (
    Qwen3_5_MOE_BigHFModel,
    _Qwen3_5MoeAttention_PlaceHolder,
)
from xhmodel_merak.xh_llm.models.qwen3_5_moe.qwen3_5_moe_model import XHQwen3_5MoeModel
from xhmodel_merak.xh_llm.types import ExportData, VisualModelMeta, VLLMModelMeta
from xhmodel_merak.xh_llm.vision_llm_model import VisionLLMModel
from xhquant.utils.registry import _DMRegistryCls


def test_qwen_export_adds_missing_gptq_desc_act_default(tmp_path):
    hf_config_dir = tmp_path / "hf_config"
    hf_config_dir.mkdir()
    config_path = hf_config_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "quantization_config": {
                    "quant_method": "gptq",
                    "bits": 4,
                },
            }
        ),
        encoding="utf-8",
    )

    _ensure_gptq_desc_act_default(hf_config_dir)

    exported_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert exported_config["quantization_config"]["desc_act"] is False


@pytest.mark.parametrize("explicit_value", [False, True])
def test_qwen_export_preserves_explicit_gptq_desc_act(tmp_path, explicit_value):
    hf_config_dir = tmp_path / "hf_config"
    hf_config_dir.mkdir()
    config_path = hf_config_dir / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "gptq",
                    "desc_act": explicit_value,
                }
            }
        ),
        encoding="utf-8",
    )

    _ensure_gptq_desc_act_default(hf_config_dir)

    exported_config = json.loads(config_path.read_text(encoding="utf-8"))
    assert exported_config["quantization_config"]["desc_act"] is explicit_value


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"model_type": "qwen3_next", "eos_token_id": 151645}, 151645),
        (
            {
                "model_type": "qwen3_5_moe",
                "eos_token_id": 999,
                "text_config": {"eos_token_id": 248044},
            },
            248044,
        ),
        ({"model_type": "qwen", "eos_token_id": [11, 12]}, 11),
        ({"model_type": "qwen"}, None),
    ],
)
def test_qwen_export_resolves_pad_token_before_huge_model_wrap(
    tmp_path, config, expected
):
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")

    assert _resolve_qwen_export_pad_token_id(tmp_path) == expected


def test_qwen_export_metadata_sets_pad_token_before_parent_metadata(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"text_config": {"eos_token_id": 248044}}),
        encoding="utf-8",
    )
    model = object.__new__(XHQwen3_5Model)
    model.hf_model_dir = str(model_dir)
    model._default_pad_token_id = 123
    model.config = SimpleNamespace(context_max_length=2048, only_first_block=False)
    model._kvcache_config = SimpleNamespace()
    observed_pad_token_ids = []

    def fake_create_export_metadata(self, output_dir):
        observed_pad_token_ids.append(self._default_pad_token_id)
        return SimpleNamespace(hf_config="hf_config")

    monkeypatch.setattr(
        VisionLLMModel,
        "create_export_metadata",
        fake_create_export_metadata,
    )

    XHQwen3_5Model.create_export_metadata(model, str(tmp_path / "export"))

    assert observed_pad_token_ids == [248044]


def test_remove_registered_members_hides_modules_parameters_and_buffers_from_to():
    module = nn.Module()
    module.child = nn.Linear(2, 2)
    module.param = nn.Parameter(torch.ones(2))
    module.register_buffer("buffer", torch.ones(2))

    counts = _remove_registered_members(module)
    module.to(dtype=torch.float16)

    assert counts == {"modules": 0, "parameters": 0, "buffers": 0}
    assert list(module.children()) == []
    assert list(module.parameters()) == []
    assert list(module.buffers()) == []
    assert module._modules == {}
    assert module._parameters == {}
    assert module._buffers == {}


def test_gated_delta_net_placeholder_accepts_missing_conv_cache():
    placeholder = object.__new__(_Qwen3_5GatedDeltaNet_PlaceHolder)
    hidden_states = torch.ones(1, 2, 3)
    recurrent_state = torch.ones(1, 1)

    outputs = placeholder.forward(hidden_states, recurrent_state=recurrent_state)

    assert outputs == (hidden_states, recurrent_state)


@pytest.mark.parametrize(
    ("attention_cls", "placeholder_cls"),
    [
        (_Qwen3_5TextAttention, _Qwen3_5Attention_PlaceHolder),
        (_Qwen3_5MoeAttention, _Qwen3_5MoeAttention_PlaceHolder),
    ],
)
def test_qwen3_5_attention_placeholder_matches_reduced_input_contract(attention_cls, placeholder_cls):
    expected_parameters = [
        "hidden_states",
        "past_seq_length",
        "current_input_length",
        "attention_mask",
        "position_embeddings",
        "past_k_cache",
        "past_v_cache",
    ]

    attention_parameters = list(inspect.signature(attention_cls.forward).parameters)[1:]
    placeholder_parameters = list(inspect.signature(placeholder_cls.forward).parameters)[1:]

    assert attention_parameters == expected_parameters
    assert placeholder_parameters == expected_parameters

    hidden_states = torch.empty(1, 256, 2048)
    past_seq_length = torch.empty(1, dtype=torch.int32)
    current_input_length = torch.empty(1, dtype=torch.int32)
    attention_mask = torch.empty(1, 256)
    cos = torch.empty(1, 1, 256, 256)
    sin = torch.empty(1, 1, 256, 256)
    past_k_cache = torch.empty(1, 2, 2048, 256)
    past_v_cache = torch.empty(1, 2, 2048, 256)

    from xhmodel_merak.xh_llm.big_hf_model_helper import _placeholder_input_fake_tensors

    flat_inputs = _placeholder_input_fake_tensors(
        (
            hidden_states,
            past_seq_length,
            current_input_length,
            attention_mask,
            (cos, sin),
            past_k_cache,
            past_v_cache,
        )
    )

    assert flat_inputs == [
        hidden_states,
        past_seq_length,
        current_input_length,
        attention_mask,
        cos,
        sin,
        past_k_cache,
        past_v_cache,
    ]


def test_resolve_hf_qwen35_class_does_not_hide_unexpected_import_errors(monkeypatch):
    def fail_import(_):
        raise RuntimeError("transformers module initialization failed")

    monkeypatch.setattr(big_export.importlib, "import_module", fail_import)

    with pytest.raises(RuntimeError, match="initialization failed"):
        big_export._resolve_hf_qwen35_class("Qwen3_5Attention")


def test_resolve_hf_qwen35_class_does_not_hide_missing_transitive_dependency(monkeypatch):
    def fail_import(_):
        exc = ModuleNotFoundError("No module named 'missing_dependency'")
        exc.name = "missing_dependency"
        raise exc

    monkeypatch.setattr(big_export.importlib, "import_module", fail_import)

    with pytest.raises(ModuleNotFoundError, match="missing_dependency"):
        big_export._resolve_hf_qwen35_class("Qwen3_5Attention")


def test_resolve_hf_qwen35_class_allows_missing_parent_package(monkeypatch):
    def fail_import(_):
        exc = ModuleNotFoundError("No module named 'transformers.models.qwen3_5'")
        exc.name = "transformers.models.qwen3_5"
        raise exc

    monkeypatch.setattr(big_export.importlib, "import_module", fail_import)

    assert big_export._resolve_hf_qwen35_class("Qwen3_5Attention") is None


def test_export_visual_hmonnx_impl_updates_meta_without_releasing_visual(tmp_path):
    exported_dir = tmp_path / "release"
    exported_dir.mkdir()
    visual_hmonnx = exported_dir / "visual" / "m1536" / "vision.onnx"

    class FakeVisual:
        def __init__(self):
            self.config = SimpleNamespace(model_name="old")
            self.quanted = False

        def to_quanted_aligned(self):
            self.quanted = True

        def export_hmonnx(self, output_dir):
            assert self.quanted
            assert Path(output_dir) == exported_dir / "visual"
            visual_hmonnx.parent.mkdir(parents=True)
            visual_hmonnx.touch()
            meta = VisualModelMeta()
            meta.hmonnx = str(visual_hmonnx)
            meta.visual_input_mode = "patches"
            meta.image_token_gears = [96, 196, 384, 704, 1536]
            meta.gears = [
                {
                    "image_token_capacity": 1536,
                    "patch_token_capacity": 6144,
                    "hmonnx": "m1536/vision.onnx",
                }
            ]
            return meta

    model = XHQwen3_5Model.__new__(XHQwen3_5Model)
    model.visual = FakeVisual()
    meta_info = VLLMModelMeta()
    exported_info = ExportData()
    exported_info.exported_dir = str(exported_dir)
    exported_info.model_name = "hmquant_qwen3_5"
    exported_info.meta = meta_info

    model._export_visual_hmonnx_impl(exported_info)

    assert hasattr(model, "visual")
    assert meta_info.visual_config.hmonnx == "visual/m1536/vision.onnx"
    assert meta_info.visual_config.visual_input_mode == "patches"
    assert meta_info.visual_config.gears[0]["hmonnx"] == "visual/m1536/vision.onnx"
    assert (exported_dir / "golden_meta_info.json").exists()


def test_qwen3_5_moe_big_export_uses_moe_placeholder_components():
    model = XHQwen3_5MoeModel.__new__(XHQwen3_5MoeModel)

    big_hf_model_cls, placeholder_types = model._get_big_language_placeholder_export_components()

    assert big_hf_model_cls is Qwen3_5_MOE_BigHFModel
    assert placeholder_types is Qwen3_5_MOE_BigHFModel.PLACEHOLDER_TYPES
    assert placeholder_types == [
        "Qwen3_5MoeSparseMoeBlock",
        "LinearQwen3_5MoeSparseMoeBlock",
    ]
    assert big_hf_model_cls.register_placeholder is not Qwen3_5BigHFModel.register_placeholder
    assert callable(big_hf_model_cls.register_placeholder)


def test_qwen3_5_moe_big_export_resolves_hf_moe_module(monkeypatch):
    imported_modules = []

    class FakeMoeAttention:
        pass

    def import_module(module_name):
        imported_modules.append(module_name)
        return SimpleNamespace(Qwen3_5MoeAttention=FakeMoeAttention)

    monkeypatch.setattr(moe_big_export.importlib, "import_module", import_module)

    assert moe_big_export._resolve_hf_qwen35_class("Qwen3_5MoeAttention") is FakeMoeAttention
    assert imported_modules == ["transformers.models.qwen3_5_moe.modeling_qwen3_5_moe"]


def test_qwen3_5_moe_big_export_registers_runtime_fused_moe_alias():
    runtime_sparse_moe_type = type("Qwen3_5MoeSparseMoeBlock", (nn.Module,), {})
    hf_model = nn.Sequential(runtime_sparse_moe_type())
    registry = _DMRegistryCls("Qwen35MoeBigExportRuntimeAliasTest")

    Qwen3_5_MOE_BigHFModel.register_placeholder(registry, hf_model=hf_model)

    assert runtime_sparse_moe_type in registry
    registered_type = registry.get(runtime_sparse_moe_type)
    assert issubclass(registered_type, moe_big_export._Qwen3_5MoeSparseMoeBlock_PlaceHolder)
    assert issubclass(registered_type, runtime_sparse_moe_type)


def test_qwen3_5_moe_big_export_detects_autoround_fused_moe_placeholder(monkeypatch):
    runtime_sparse_moe_type = type("LinearQwen3_5MoeSparseMoeBlock", (nn.Module,), {})
    hf_model = nn.Sequential(runtime_sparse_moe_type())
    hf_model.config = SimpleNamespace()
    monkeypatch.setattr(Qwen3_5_MOE_BigHFModel, "_load_weight_mapping", lambda self, model: None)

    big_model = Qwen3_5_MOE_BigHFModel(
        "/tmp/qwen3_5_moe",
        hf_model,
        Qwen3_5_MOE_BigHFModel.PLACEHOLDER_TYPES,
    )

    assert big_model._placeholder_types == {runtime_sparse_moe_type}


@pytest.mark.parametrize(
    ("big_hf_model_cls", "module_types"),
    [
        (
            Qwen3_5BigHFModel,
            (big_export.Qwen3_5Attention, big_export.Qwen3_5GatedDeltaNet, big_export.Qwen3_5MLP),
        ),
        (
            Qwen3_5_MOE_BigHFModel,
            (
                moe_big_export.Qwen3_5MoeAttention,
                moe_big_export.Qwen3_5MoeGatedDeltaNet,
                moe_big_export.Qwen3_5MoeSparseMoeBlock,
            ),
        ),
    ],
)
def test_register_placeholder_only_registers_configured_module_types(big_hf_model_cls, module_types):
    registry = _DMRegistryCls(f"{big_hf_model_cls.__name__}ConfiguredPlaceholderTest")

    big_hf_model_cls.register_placeholder(registry)

    for module_type in module_types:
        assert (module_type in registry) is (module_type.__name__ in big_hf_model_cls.PLACEHOLDER_TYPES)


def test_qwen3_5_moe_big_export_resolves_main_gptq_skip_prefixes_before_constructor(monkeypatch):
    captured = []
    runtime_sparse_moe_type = type("Qwen3_5MoeSparseMoeBlock", (nn.Module,), {})
    hf_model = nn.Module()
    hf_model.config = SimpleNamespace()
    hf_model.model = nn.Module()
    hf_model.model.language_model = nn.Module()
    hf_model.model.language_model.layers = nn.ModuleList([nn.Module()])
    hf_model.model.language_model.layers[0].mlp = runtime_sparse_moe_type()
    hf_model.model.language_model.layers[0].mlp.experts = nn.ModuleList([nn.Linear(2, 2)])

    def capture_preprocess(model, hf_model_dir=None, skip_module_prefixes=None):
        captured.append((model, hf_model_dir, skip_module_prefixes))

    monkeypatch.setattr(
        Qwen3_5_MOE_BigHFModel,
        "_preprocess_quantized_hf_model",
        staticmethod(capture_preprocess),
    )
    monkeypatch.setattr(Qwen3_5_MOE_BigHFModel, "_load_weight_mapping", lambda self, model: None)

    placeholder_prefixes = Qwen3_5_MOE_BigHFModel.resolve_placeholder_prefixes(
        hf_model,
        Qwen3_5_MOE_BigHFModel.PLACEHOLDER_TYPES,
    )
    Qwen3_5_MOE_BigHFModel._preprocess_quantized_hf_model(
        hf_model,
        "/tmp/qwen3_5_moe",
        skip_module_prefixes=placeholder_prefixes,
    )
    Qwen3_5_MOE_BigHFModel(
        "/tmp/qwen3_5_moe",
        hf_model,
        Qwen3_5_MOE_BigHFModel.PLACEHOLDER_TYPES,
    )

    assert captured == [
        (hf_model, "/tmp/qwen3_5_moe", ["model.language_model.layers.0.mlp"]),
    ]


def test_qwen3_5_big_export_preprocesses_both_empty_models_before_constructor(monkeypatch):
    class StopAfterBigHFModelError(Exception):
        pass

    events = []

    class FakeBigHFModel:
        PLACEHOLDER_TYPES = ["FakePlaceholder"]

        @classmethod
        def initialize_process_worker(cls):
            events.append(("worker_init", cls))

        @classmethod
        def initialize_process_worker_after_model_load(cls, hf_model):
            events.append(("after_load", hf_model))

        @classmethod
        def initialize_process_worker_after_quantized_preprocess(cls, hf_model):
            events.append(("after_quant", hf_model))

        @staticmethod
        def resolve_placeholder_prefixes(hf_model, placeholder_types):
            events.append(("resolve", hf_model, placeholder_types))
            return ["model.language_model.layers"]

        @staticmethod
        def _preprocess_quantized_hf_model(hf_model, hf_model_dir, skip_module_prefixes=None):
            events.append(("preprocess", hf_model, hf_model_dir, skip_module_prefixes))

        def __init__(self, hf_model_dir, hf_model, placeholder_types):
            assert hf_model_dir == "/tmp/qwen3_5"
            assert placeholder_types == self.PLACEHOLDER_TYPES
            events.append(("init", hf_model))
            raise StopAfterBigHFModelError

    class FakeQwen35Model(XHQwen3_5Model):
        def _get_big_language_placeholder_export_components(self):
            return FakeBigHFModel, FakeBigHFModel.PLACEHOLDER_TYPES

        @classmethod
        def get_empty_hf_model(cls, hf_model_dir, **kwargs):
            events.append(("empty", kwargs.get("dtype")))
            model = nn.Module()
            model.config = SimpleNamespace(model_type="qwen3_5")
            model.model = nn.Module()
            model.model.visual = nn.Module()
            model.model.language_model = nn.Module()
            return model

        def _check_big_language_placeholder_export_supported(self, empty_hf_model):
            return None

    model = FakeQwen35Model.__new__(FakeQwen35Model)
    model.hf_model_dir = "/tmp/qwen3_5"
    model._dtype = torch.float16

    with pytest.raises(StopAfterBigHFModelError):
        model._export_big_language_hmonnx(SimpleNamespace(exported_dir="/tmp/out"))

    assert [event[0] for event in events] == [
        "worker_init",
        "empty",
        "after_load",
        "resolve",
        "preprocess",
        "after_quant",
        "preprocess",
        "after_quant",
        "init",
    ]
    assert events[1][1] is torch.float16
    assert events[2][1] is events[3][1]
    assert events[4][1] is not events[6][1]
    assert events[4][3] is None
    assert events[6][3] == ["model.language_model.layers"]
    assert events[4][1] is events[5][1]
    assert events[6][1] is events[7][1]
    assert events[6][1] is events[8][1]


def test_qwen3_5_big_export_support_check_accepts_dense_components():
    model = XHQwen3_5Model.__new__(XHQwen3_5Model)
    qwen3_5_attention = type("Qwen3_5Attention", (nn.Module,), {})
    qwen3_5_gated_delta_net = type("Qwen3_5GatedDeltaNet", (nn.Module,), {})
    qwen3_5_mlp = type("Qwen3_5MLP", (nn.Module,), {})
    empty_hf_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5"),
        model=SimpleNamespace(
            language_model=nn.Sequential(
                qwen3_5_attention(),
                qwen3_5_gated_delta_net(),
                qwen3_5_mlp(),
            )
        ),
    )

    model._check_big_language_placeholder_export_supported(empty_hf_model)


def test_qwen3_5_big_export_support_check_rejects_missing_dense_components():
    model = XHQwen3_5Model.__new__(XHQwen3_5Model)
    qwen3_5_attention = type("Qwen3_5Attention", (nn.Module,), {})
    empty_hf_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5"),
        model=SimpleNamespace(language_model=nn.Sequential(qwen3_5_attention())),
    )

    with pytest.raises(NotImplementedError, match="missing .*Qwen3_5GatedDeltaNet"):
        model._check_big_language_placeholder_export_supported(empty_hf_model)


def test_qwen3_5_moe_big_export_support_check_accepts_moe_components():
    model = XHQwen3_5MoeModel.__new__(XHQwen3_5MoeModel)
    qwen3_5_moe_attention = type("Qwen3_5MoeAttention", (nn.Module,), {})
    qwen3_5_moe_gated_delta_net = type("Qwen3_5MoeGatedDeltaNet", (nn.Module,), {})
    qwen3_5_moe_sparse_moe_block = type("Qwen3_5MoeSparseMoeBlock", (nn.Module,), {})
    empty_hf_model = SimpleNamespace(
        config=SimpleNamespace(model_type="qwen3_5_moe"),
        model=SimpleNamespace(
            language_model=nn.Sequential(
                qwen3_5_moe_attention(),
                qwen3_5_moe_gated_delta_net(),
                qwen3_5_moe_sparse_moe_block(),
            )
        ),
    )

    model._check_big_language_placeholder_export_supported(empty_hf_model)


def test_qwen3_5_moe_big_export_support_check_rejects_dense_model():
    model = XHQwen3_5MoeModel.__new__(XHQwen3_5MoeModel)
    empty_hf_model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5"))

    with pytest.raises(NotImplementedError, match="dense big-model placeholder export"):
        model._check_big_language_placeholder_export_supported(empty_hf_model)


def test_qwen3_5_moe_full_attention_uses_single_tensor_contract():
    class FakeAttention(nn.Module):
        def forward(self, hidden_states, **kwargs):
            return hidden_states * 2

    layer = object.__new__(_Qwen3_5MoeDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_type = "full_attention"
    layer.input_layernorm = nn.Identity()
    layer.self_attn = FakeAttention()
    layer.post_attention_layernorm = nn.Identity()
    layer.mlp = nn.Identity()
    hidden_states = torch.ones(1, 2, 3)

    output = layer(hidden_states)

    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output, torch.full_like(hidden_states, 6))


def test_qwen3_5_moe_linear_attention_flattens_split_conv_cache_outputs():
    class FakeLinearAttention(nn.Module):
        split_conv_cache = True

        def forward(self, hidden_states, conv_cache, recurrent_state, **kwargs):
            return hidden_states * 2, *conv_cache, recurrent_state

    layer = object.__new__(_Qwen3_5MoeDecoderLayer)
    nn.Module.__init__(layer)
    layer.layer_type = "linear_attention"
    layer.input_layernorm = nn.Identity()
    layer.linear_attn = FakeLinearAttention()
    layer.post_attention_layernorm = nn.Identity()
    layer.mlp = nn.Identity()
    hidden_states = torch.ones(1, 2, 3)
    conv_caches = [torch.full((1, 1), value) for value in (1.0, 2.0, 3.0)]
    recurrent_state = torch.full((1, 1), 4.0)

    output = layer(
        hidden_states,
        past_conv_cache=conv_caches,
        past_recurrent_state=recurrent_state,
    )

    assert len(output) == 5
    torch.testing.assert_close(output[0], torch.full_like(hidden_states, 6))
    for actual, expected in zip(output[1:4], conv_caches, strict=True):
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(output[4], recurrent_state)


def test_qwen3_5_split_conv_cache_accepts_fused_in_place_recurrent_state():
    hidden_states = torch.ones(1, 2, 3)
    conv_caches = tuple(torch.full((1, 1), value) for value in (1.0, 2.0, 3.0))

    actual_hidden, actual_conv_caches, recurrent_state = (
        _unpack_split_conv_cache_outputs(
            (hidden_states, *conv_caches),
            owner="Split Qwen3.5 linear attention",
        )
    )

    torch.testing.assert_close(actual_hidden, hidden_states)
    assert actual_conv_caches == conv_caches
    assert recurrent_state is None


def test_qwen3_5_split_conv_cache_accepts_speculative_verify_snapshots():
    hidden_states = torch.ones(1, 2, 3)
    conv_caches = tuple(torch.full((1, 1), float(value)) for value in range(1, 7))
    recurrent_states = tuple(torch.full((1, 1), float(value)) for value in (7, 8))

    actual_hidden, actual_conv_caches, actual_recurrent_states = (
        _unpack_split_conv_cache_outputs(
            (hidden_states, conv_caches, recurrent_states),
            owner="Split Qwen3.5 linear attention",
        )
    )

    torch.testing.assert_close(actual_hidden, hidden_states)
    assert actual_conv_caches == conv_caches
    assert actual_recurrent_states == recurrent_states


@pytest.mark.parametrize("decoder_cls", [_Qwen3_5DecoderLayer, _Qwen3_5MoeDecoderLayer])
def test_qwen3_5_linear_attention_preserves_fused_in_place_cache_abi(decoder_cls):
    class FakeLinearAttention(nn.Module):
        split_conv_cache = True

        def forward(self, hidden_states, conv_cache, recurrent_state, **kwargs):
            del recurrent_state, kwargs
            return hidden_states * 2, *conv_cache

    layer = object.__new__(decoder_cls)
    nn.Module.__init__(layer)
    layer.layer_type = "linear_attention"
    layer.input_layernorm = nn.Identity()
    layer.linear_attn = FakeLinearAttention()
    layer.post_attention_layernorm = nn.Identity()
    layer.mlp = nn.Identity()
    hidden_states = torch.ones(1, 2, 3)
    conv_caches = tuple(torch.full((1, 1), value) for value in (1.0, 2.0, 3.0))

    output = layer(
        hidden_states,
        past_conv_cache=conv_caches,
        past_recurrent_state=torch.full((1, 1), 4.0),
    )

    assert len(output) == 5
    torch.testing.assert_close(output[0], torch.full_like(hidden_states, 6))
    for actual, expected in zip(output[1:4], conv_caches, strict=True):
        torch.testing.assert_close(actual, expected)
    assert output[4] is None


@pytest.mark.parametrize("decoder_cls", [_Qwen3_5DecoderLayer, _Qwen3_5MoeDecoderLayer])
def test_qwen3_5_linear_attention_flattens_speculative_verify_cache_abi(decoder_cls):
    conv_caches = tuple(torch.full((1, 1), float(value)) for value in range(1, 7))
    recurrent_states = tuple(torch.full((1, 1), float(value)) for value in (7, 8))

    class FakeLinearAttention(nn.Module):
        split_conv_cache = True

        def forward(self, hidden_states, **kwargs):
            del kwargs
            return hidden_states * 2, conv_caches, recurrent_states

    layer = object.__new__(decoder_cls)
    nn.Module.__init__(layer)
    layer.layer_type = "linear_attention"
    layer.input_layernorm = nn.Identity()
    layer.linear_attn = FakeLinearAttention()
    layer.post_attention_layernorm = nn.Identity()
    layer.mlp = nn.Identity()
    hidden_states = torch.ones(1, 2, 3)

    output = layer(hidden_states)

    assert len(output) == 8
    torch.testing.assert_close(output[0], torch.full_like(hidden_states, 6))
    assert output[1:7] == conv_caches
    assert output[7] == recurrent_states


def test_qwen3_5_moe_sparse_block_uses_single_tensor_contract():
    class FakeMoeBlock(nn.Module):
        def forward(self, hidden_states, routing_weights):
            return hidden_states + routing_weights[..., :1]

    block = object.__new__(_Qwen3_5MoeSparseMoeBlock)
    nn.Module.__init__(block)
    block.gate = nn.Linear(3, 2, bias=False)
    nn.init.zeros_(block.gate.weight)
    block.moeblock = FakeMoeBlock()
    block.shared_expert = nn.Identity()
    block.shared_expert_gate = nn.Linear(3, 3, bias=False)
    nn.init.zeros_(block.shared_expert_gate.weight)
    hidden_states = torch.ones(1, 2, 3)

    output = block(hidden_states)

    assert isinstance(output, torch.Tensor)
    torch.testing.assert_close(output, torch.full_like(hidden_states, 2.0))
