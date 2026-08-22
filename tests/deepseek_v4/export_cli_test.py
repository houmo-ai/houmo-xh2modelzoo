from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from xhmodel_merak.xh_llm.models.deepseek_v4 import XHDeepSeekV4Model


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
    assert args.quant_type == "w8a8h1_sefp"

    config = export_demo.build_config(args)
    assert config.quant_scheme.ops == {}


def test_deepseek_v4_export_writes_merak_package_entrypoint(tmp_path: Path) -> None:
    config_path = XHDeepSeekV4Model._write_merak_config(tmp_path)

    assert config_path == tmp_path / "merak_config.json"
    assert json.loads(config_path.read_text(encoding="utf-8")) == {
        "architectures": ["MerakForCausalLM"],
        "config_format": "merak_llm",
        "load_format": "merak_llm",
        "xh_model": {
            "model_type": "hmonnx",
            "meta_info": "golden_meta_info.json",
        },
        "enable_page_attention": False,
        "model_type": "merak_llm",
    }
