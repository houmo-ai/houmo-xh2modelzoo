"""Lightweight tests for Qwen3.5 HMONNX runtime quick-test helpers."""
from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _install_lightweight_xh_llm_packages(monkeypatch) -> None:
    packages = {
        "xhmodel_merak.xh_llm": REPO_ROOT / "xhmodel_merak/xh_llm",
        "xhmodel_merak.xh_llm.workflows": REPO_ROOT / "xhmodel_merak/xh_llm/workflows",
        "xhmodel_merak.xh_llm.models": REPO_ROOT / "xhmodel_merak/xh_llm/models",
        "xhmodel_merak.xh_llm.models.qwen3_5": REPO_ROOT / "xhmodel_merak/xh_llm/models/qwen3_5",
    }
    for name, path in packages.items():
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, module)


def _load_runtime_module(monkeypatch):
    _install_lightweight_xh_llm_packages(monkeypatch)
    return importlib.import_module("xhmodel_merak.xh_llm.models.qwen3_5.hmonnx_validation")


def test_find_hmonnx_meta_file_accepts_export_result_dir_and_meta(monkeypatch, tmp_path: Path):
    runtime = _load_runtime_module(monkeypatch)
    result_mod = importlib.import_module("xhmodel_merak.xh_llm.workflows.result")
    meta_file = tmp_path / "export" / "hmquant_qwen35" / "golden_meta_info.json"
    meta_file.parent.mkdir(parents=True)
    meta_file.write_text("{}", encoding="utf-8")

    export_result = result_mod.ExportResult(work_dir=str(tmp_path / "export"), config_file="effective.yaml")

    assert runtime.find_hmonnx_meta_file(export_result) == str(meta_file)
    assert runtime.find_hmonnx_meta_file(tmp_path / "export") == str(meta_file)
    assert runtime.find_hmonnx_meta_file(meta_file) == str(meta_file)


def test_quick_test_hmonnx_routes_plain_and_spec_decode_meta(monkeypatch, tmp_path: Path):
    runtime = _load_runtime_module(monkeypatch)
    calls: list[tuple[str, str]] = []

    def fake_generate(meta_file, **kwargs):
        calls.append(("generate", str(meta_file)))
        return runtime.HMONNXQuickTestResult(
            meta_file=str(meta_file),
            output_text="plain",
            output_tokens=1,
            latency_s=0.5,
            tokens_per_second=2.0,
        )

    def fake_spec(meta_file, **kwargs):
        calls.append(("spec", str(meta_file)))
        return runtime.HMONNXQuickTestResult(
            meta_file=str(meta_file),
            output_text="spec",
            output_tokens=2,
            latency_s=1.0,
            tokens_per_second=2.0,
            spec_decode_mode="mtp",
            draft_tokens_total=8,
            accepted_drafts_total=6,
            accept_rate=0.75,
        )

    monkeypatch.setattr(runtime, "hmonnx_generate", fake_generate)
    monkeypatch.setattr(runtime, "spec_decode_generate", fake_spec)

    plain_meta = tmp_path / "plain.json"
    plain_meta.write_text("{}", encoding="utf-8")
    spec_meta = tmp_path / "spec.json"
    spec_meta.write_text(json.dumps({"spec_decode": {"mode": "mtp"}}), encoding="utf-8")

    assert runtime.quick_test_hmonnx(plain_meta).output_text == "plain"
    assert runtime.quick_test_hmonnx(spec_meta).accept_rate == 0.75
    assert calls == [("generate", str(plain_meta)), ("spec", str(spec_meta))]


def test_spec_decode_stats_are_summarized_with_accept_rate(monkeypatch, tmp_path: Path):
    runtime = _load_runtime_module(monkeypatch)

    result = runtime.build_spec_decode_result(
        meta_file=tmp_path / "golden_meta_info.json",
        output_text="hello",
        output_tokens=12,
        latency_s=3.0,
        spec_decode_mode="dflash",
        block_size=5,
        stats={
            "num_rounds": 4,
            "draft_tokens_total": 20,
            "accepted_drafts_total": 7,
            "avg_accepted_per_round": 1.75,
            "accepted_drafts_per_round": [2, 1, 3, 1],
        },
    )

    assert result.spec_decode_mode == "dflash"
    assert result.tokens_per_second == 4.0
    assert result.draft_tokens_total == 20
    assert result.accepted_drafts_total == 7
    assert result.accept_rate == 0.35
    assert result.avg_accepted_per_round == 1.75
    assert result.stats["accepted_drafts_per_round"] == [2, 1, 3, 1]
