from __future__ import annotations

import importlib.util
from pathlib import Path


_EXPORT = Path(__file__).resolve().parents[2] / "examples_merak/llm/deepseekv4/export_hmonnx.py"
_SPEC = importlib.util.spec_from_file_location("deepseek_v4_export_demo", _EXPORT)
assert _SPEC is not None and _SPEC.loader is not None
export_demo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(export_demo)


def test_export_cli_exposes_formal_golden_controls() -> None:
    args = export_demo.build_parser().parse_args(
        [
            "--model",
            "checkpoint",
            "--output-dir",
            "export",
            "--dump-golden",
            "--golden-device-map",
            "cuda:1",
            "2",
        ]
    )

    assert args.dump_golden is True
    assert args.golden_device_map == ["cuda:1", "2"]
    assert args.golden_prompt == "17乘以3等于多少？只回答结果。"
