from __future__ import annotations

from pathlib import Path

import yaml


def test_streaming_audio_uses_w8a16_as_production_default() -> None:
    config_path = Path("configs_merak/workflows/xh2a/llm_models/minicpm_o_4_5/minicpm_o_4_5_xh2a_w8a8_gptq.yaml")

    config = yaml.safe_load(config_path.read_text())

    assert config["export"]["components"]["audio"]["quant_type"] == "w8a16_sefp"
