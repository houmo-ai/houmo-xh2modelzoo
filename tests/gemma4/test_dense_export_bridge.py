"""HMSW-3948: tests for gemma4 dense/PLE export bridge split + data_preprocess tolerance."""
import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _make_hf_model_mock(hidden_size_per_layer_input: int):
    """Build a minimal MagicMock that satisfies bridge __init__ + factory selector."""
    hf_model = MagicMock()
    text_cfg = MagicMock()
    text_cfg.hidden_size_per_layer_input = hidden_size_per_layer_input
    hf_model.config.get_text_config.return_value = text_cfg
    # _Gemma4TextExportBridgeBase.__init__ touches these:
    hf_model.model.language_model = MagicMock()
    hf_model.lm_head = MagicMock()
    return hf_model


def test_make_text_export_bridge_returns_ple_when_per_layer_input_size_positive():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import (
        _Gemma4TextExportBridgePLE,
        _make_text_export_bridge,
    )

    bridge = _make_text_export_bridge(_make_hf_model_mock(hidden_size_per_layer_input=256))
    assert isinstance(bridge, _Gemma4TextExportBridgePLE)


def test_make_text_export_bridge_returns_dense_when_per_layer_input_size_zero():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import (
        _Gemma4TextExportBridgeDense,
        _make_text_export_bridge,
    )

    bridge = _make_text_export_bridge(_make_hf_model_mock(hidden_size_per_layer_input=0))
    assert isinstance(bridge, _Gemma4TextExportBridgeDense)


def test_dense_bridge_forward_signature_excludes_per_layer_inputs():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import _Gemma4TextExportBridgeDense

    params = inspect.signature(_Gemma4TextExportBridgeDense.forward).parameters
    assert "per_layer_inputs" not in params
    assert "position_ids" not in params, (
        "dense bridge should no longer accept position_ids; rope is computed via past_seq_length slicing"
    )
    for name in (
        "inputs_embeds",
        "past_seq_length",
        "current_input_length",
        "local_attention_mask",
    ):
        assert name in params, f"dense bridge missing required arg: {name}"
    assert "global_attention_mask" not in params, (
        "global attention reuses the explicit local mask path; exporting both masks "
        "reintroduces the duplicate masked-add fixed by QTL-304"
    )


def test_ple_bridge_forward_signature_includes_per_layer_inputs_first():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import _Gemma4TextExportBridgePLE

    params = list(inspect.signature(_Gemma4TextExportBridgePLE.forward).parameters)
    # self is at [0]; per_layer_inputs must immediately follow
    assert params[1] == "per_layer_inputs"


def test_data_preprocess_no_longer_raises_when_per_layer_inputs_is_none():
    """Source-level check: the ValueError on per_layer_inputs is None must be gone."""
    src = Path("xhmodel_merak/xh_llm/models/gemma4e/data_preprocess.py").read_text(encoding="utf-8")
    assert "Gemma4DataPreprocess requires per_layer_input_builder" not in src
    assert "if per_layer_inputs is not None:" in src


def test_llm_model_impl_handles_v_proj_none_for_attention_k_eq_v():
    """Source-level check: v_proj None path falls back to key_states (K=V)."""
    src = Path("xhmodel_merak/xh_llm/models/gemma4e/_llm_model_impl.py").read_text(encoding="utf-8")
    assert "if self.v_proj is not None" in src
    # value_states should default to key_states when v_proj is None
    assert "else key_states" in src


def test_prepare_submodel_for_hmonnx_export_quantizes_before_fixed():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model

    class DummyQuantedModel:
        def __init__(self):
            self.fixed_called = False

        def fixed(self):
            self.fixed_called = True

    class DummySubModel:
        def __init__(self):
            self.config = SimpleNamespace(enable=True)
            self.quanted_model = None
            self.to_quanted_aligned_called = False

        def to_quanted_aligned(self):
            self.to_quanted_aligned_called = True
            self.quanted_model = DummyQuantedModel()

    sub_model = DummySubModel()

    assert XHGemma4Model._prepare_submodel_for_hmonnx_export(sub_model)
    assert sub_model.to_quanted_aligned_called
    assert sub_model.quanted_model.fixed_called


def test_prepare_submodel_for_hmonnx_export_does_not_silently_skip_configured_submodel():
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model

    sub_model = SimpleNamespace(config=SimpleNamespace(enable=False), quanted_model=None)

    with pytest.raises(RuntimeError, match="visual export is configured but disabled"):
        XHGemma4Model._prepare_submodel_for_hmonnx_export(sub_model, "visual")


def test_gemma4_31b_config_keeps_visual_submodel_exportable():
    model_dir = Path("/data01/datasets/gemma-4-31b-it")
    if not model_dir.exists():
        pytest.skip(f"Gemma4 31B official model config is not available: {model_dir}")

    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_llm_model import XHGemma4Model
    from xhmodel_merak.xh_llm.models.gemma4e.xh_gemma4_config import XHGemma4ModelConfig

    model = XHGemma4Model(
        XHGemma4ModelConfig(
            model_name="xh2_gemma4_31b_it_w4a8_autoround_256_4k",
            model_type="Gemma4ForConditionalGeneration",
            hf_model=str(model_dir),
            context_max_length=4096,
            prefill_chunk_length=256,
            visual_config={
                "image_seq_length": 280,
                "patch_size": 16,
                "pooling_kernel_size": 3,
                "quant_scheme": {"quant_type": "w4a8h1_ssfp", "ops": {}},
            },
        )
    )

    assert model.visual is not None
    assert model.visual.config.hf_model == str(model_dir)
    assert model.visual.config.image_seq_length == 280
    assert model.visual.config.pooling_kernel_size == 3


def test_gemma4_visual_get_hf_model_bypasses_gptqmodel_for_dense_gptq_config(monkeypatch):
    from xhmodel_merak.xh_llm.models.gemma4e import gemma4_vision_model as vision_mod
    from xhmodel_merak.xh_llm.models.gemma4e.gemma4_vision_model import XHGemma4VisionModel

    loaded_model = SimpleNamespace(config=SimpleNamespace(quantization_config=object()))
    calls = []

    monkeypatch.setattr(
        vision_mod.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(quantization_config=SimpleNamespace(quant_method="gptq")),
    )

    def _fake_load(cls, hf_model_dir, **kwargs):
        calls.append(("load", hf_model_dir, kwargs))
        return loaded_model

    monkeypatch.setattr(XHGemma4VisionModel, "_load_hf_model", classmethod(_fake_load))

    out = XHGemma4VisionModel.get_hf_model("/fake/gemma4-dense-gptq")

    assert out is loaded_model
    assert loaded_model.config.quantization_config is None
    assert calls[0][0] == "load"
    assert calls[0][1] == "/fake/gemma4-dense-gptq"
    assert calls[0][2]["device_map"] == "cpu"
    assert calls[0][2]["trust_remote_code"] is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
