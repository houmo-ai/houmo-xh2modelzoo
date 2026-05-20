"""HMSW-3948: tests for gemma4 dense/PLE export bridge split + data_preprocess tolerance."""
import inspect
from pathlib import Path
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
    for name in (
        "inputs_embeds",
        "position_ids",
        "past_seq_length",
        "current_input_length",
        "local_attention_mask",
        "global_attention_mask",
    ):
        assert name in params, f"dense bridge missing required arg: {name}"


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
