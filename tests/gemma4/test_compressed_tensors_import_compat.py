from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_clean_import(source: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(REPO_ROOT),
            env.get("PYTHONPATH", ""),
        ]
    )
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_gemma4_runtime_import_does_not_load_optional_mtp_export_model() -> None:
    result = _run_clean_import(
        "\n".join(
            [
                "import sys",
                "from xhmodel_merak.xh_llm.builder import get_model_class",
                "import xhmodel_merak.xh_llm.models.gemma4_series as gemma4_series",
                "assert gemma4_series.XHGemma4SeriesHMONNXModel is not None",
                (
                    "gemma_cfg = {'chip_arch': 'XH2a', "
                    "'model_type': 'Gemma4ForConditionalGeneration', 'model_name': 'gemma4'}"
                ),
                "assert get_model_class(gemma_cfg) is not None",
                "assert 'xhmodel_merak.xh_llm.models.gemma4_series.gemma4_series_mtp_model' not in sys.modules",
            ]
        )
    )

    assert result.returncode == 0, result.stderr


def test_qwen35_runtime_packages_still_import() -> None:
    result = _run_clean_import(
        "\n".join(
            [
                "from xhmodel_merak.xh_llm.builder import get_model_class",
                "import xhmodel_merak.xh_llm.models.qwen3_5 as dense",
                "import xhmodel_merak.xh_llm.models.qwen3_5_moe as moe",
                "assert dense.XHQwen3_5_HMONNXModel is not None",
                "assert moe.XHQwen3_5MoeHMONNXModel is not None",
                (
                    "dense_cfg = {'chip_arch': 'XH2a', "
                    "'model_type': 'Qwen3_5ForConditionalGeneration', 'model_name': 'qwen35'}"
                ),
                (
                    "moe_cfg = {'chip_arch': 'XH2a', "
                    "'model_type': 'Qwen3_5MoeForConditionalGeneration', 'model_name': 'qwen35_moe'}"
                ),
                "assert get_model_class(dense_cfg) is not None",
                "assert get_model_class(moe_cfg) is not None",
            ]
        )
    )

    assert result.returncode == 0, result.stderr
