from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from xhmodel_merak.xh_llm.models.gemma4_series import workflow
from xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_hmonnx_inference import (
    XHGemma4SeriesHMONNXModel,
    _gemma4_prefill_graph_hmonnx_from_meta,
    _gemma4_prefill_graph_lengths_from_meta,
    _select_gemma4_prefill_graph_name,
)
from xhmodel_merak.xh_llm.models.gemma4_series.llm_text import _gemma4_runtime_prefill_length
from xhmodel_merak.xh_llm.models.gemma4_series.quant_adapter import (
    DEFAULT_AUTOROUND_DATASET,
)


def test_gemma4_hmonnx_prefill_graph_name_selection_uses_single_declared_width():
    assert _select_gemma4_prefill_graph_name(320, {"prefill": 320}) == "prefill"


@pytest.mark.parametrize("requested_width", [256, 319, 321])
def test_gemma4_hmonnx_prefill_graph_name_rejects_missing_requested_width(requested_width: int):
    with pytest.raises(ValueError, match="No Gemma4 prefill graph"):
        _select_gemma4_prefill_graph_name(requested_width, {"prefill": 320})


def test_gemma4_hmonnx_prefill_graph_metadata_accepts_single_graph_contract():
    legacy_single = SimpleNamespace(
        model_config=SimpleNamespace(prefill_chunk_length=320),
    )
    assert _gemma4_prefill_graph_lengths_from_meta(legacy_single) == {"prefill": 320}

    modern = SimpleNamespace(
        model_config=SimpleNamespace(prefill_chunk_length=320),
        prefill_graphs={"prefill": {"input_sequence_length": 320, "hmonnx": "prefill/model.onnx"}},
    )
    assert _gemma4_prefill_graph_lengths_from_meta(modern) == {"prefill": 320}

    stale_dual = SimpleNamespace(
        model_config=SimpleNamespace(prefill_chunk_length=320),
        prefill_graphs={
            "prefill": {"input_sequence_length": 320},
            "prefill_mm": {"input_sequence_length": 320, "hmonnx": "prefill_mm/model.onnx"},
        },
    )
    with pytest.raises(ValueError, match="single 'prefill'"):
        _gemma4_prefill_graph_lengths_from_meta(stale_dual)


def test_gemma4_runtime_prefill_length_supports_hmonnx_meta_surface():
    runtime_model = SimpleNamespace(
        meta_info=SimpleNamespace(model_config=SimpleNamespace(prefill_chunk_length=320))
    )
    assert _gemma4_runtime_prefill_length(runtime_model) == 320


def test_gemma4_hmonnx_active_prefill_length_drives_new_preprocessors():
    model = XHGemma4SeriesHMONNXModel.__new__(XHGemma4SeriesHMONNXModel)
    model._llm_prefill = True
    model._active_prefill_input_sequence_length = 320
    assert model.get_input_sequence_length() == 320


def test_gemma4_prefill_path_resolves_from_absolute_prefill_path():
    meta = SimpleNamespace(
        prefill_hmonnx="/tmp/exported/prefill/model_prefill.onnx",
        prefill_graphs={"prefill": {"hmonnx": "prefill/model_prefill.onnx"}},
    )
    assert (
        _gemma4_prefill_graph_hmonnx_from_meta(meta, "prefill")
        == "/tmp/exported/prefill/model_prefill.onnx"
    )
    with pytest.raises(ValueError, match="single 'prefill'"):
        _gemma4_prefill_graph_hmonnx_from_meta(meta, "prefill_mm")


def test_gemma4_mtp_resolve_model_path_uses_cwd_for_relative_paths(tmp_path, monkeypatch):
    from xhmodel_merak.xh_llm.models.gemma4_series.mtp_workflow import resolve_model_path

    project_dir = tmp_path / "customer_project"
    project_dir.mkdir()
    monkeypatch.chdir(project_dir)

    assert resolve_model_path("weights/gemma-4-E2B-it-assistant") == (
        project_dir / "weights/gemma-4-E2B-it-assistant"
    ).resolve()


def test_gemma4_mtp_resolve_model_path_preserves_absolute_paths(tmp_path):
    from xhmodel_merak.xh_llm.models.gemma4_series.mtp_workflow import resolve_model_path

    absolute_model_dir = tmp_path / "weights" / "gemma-4-E2B-it-assistant"

    assert resolve_model_path(str(absolute_model_dir)) == absolute_model_dir.resolve()


def test_gemma4_series_workflow_yamls_use_single_prefill_and_stable_autoround_jsonl():
    config_root = Path("configs_merak/workflows/xh2a/llm_models/gemma4_series")
    yaml_files = sorted(config_root.glob("*/*.yaml"))
    assert yaml_files
    for yaml_file in yaml_files:
        data = yaml.safe_load(yaml_file.read_text(encoding="utf-8"))
        model_cfg = data["export"]["model"]
        assert model_cfg["prefill_chunk_length"] == 320, yaml_file.as_posix()
        assert "mm_prefill_chunk_length" not in model_cfg, yaml_file.as_posix()

        quant_cfg = data.get("quant") or {}
        if str(quant_cfg.get("method", "")).lower() == "autoround":
            calibration_cfg = quant_cfg["calibration"]
            assert "dataset" not in calibration_cfg, yaml_file.as_posix()
            assert calibration_cfg["jsonl"] == DEFAULT_AUTOROUND_DATASET, yaml_file.as_posix()


def test_gemma4_series_workflow_help_and_template_use_single_prefill(tmp_path):
    help_text = workflow.get_export_config_help()
    assert "mm_prefill_chunk_length" not in help_text
    assert "prefill_mm" not in help_text
    assert "320" in help_text

    out = tmp_path / "export_template.yaml"
    workflow.dump_export_config_template(out)
    data = yaml.safe_load(out.read_text(encoding="utf-8"))
    model_cfg = data["export"]["model"]
    assert model_cfg["context_max_length"] == 2048
    assert model_cfg["prefill_chunk_length"] == 320
    assert "mm_prefill_chunk_length" not in model_cfg
