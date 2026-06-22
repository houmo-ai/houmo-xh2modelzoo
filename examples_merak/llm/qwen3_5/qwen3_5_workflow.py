"""Standard Qwen3.5/Qwen3.6 Merak workflow example.

Edit the constants below, then run this file from the repository root.  Model
shape, quantization, visual size, MTP/DFlash, and GDR options stay in YAML or
``CONFIG_OVERRIDES``; the workflow API only needs paths plus ``QuantResult``.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from xhmodel_merak.xh_llm.models.qwen3_5.workflow_runtime import (  # noqa: E402
    print_quick_test_result,
    quick_test_hmonnx,
)
from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow  # noqa: E402


HF_MODEL_DIR = "weights/Qwen3.5-9B"
CONFIG_PATH = "configs_merak/workflows/xh2a/llm_models/qwen3_5/9b/qwen3_5_9b_full.yaml"
QUANT_OUTPUT_DIR = "work_dirs/qwen3_5_9b_workflow_quant"
EXPORT_OUTPUT_DIR = "work_dirs/qwen3_5_9b_workflow_export"
DEVICE = "cuda"
SEED = 1024
DEBUG = False
FORCE_OVERWRITE = True
PROMPT = "用中文简单介绍 Qwen3.5。"
QUICK_TEST = True
QUICK_TEST_MAX_NEW_TOKENS = 64

# Base validation skips quantization and exports from HF_MODEL_DIR:
# CONFIG_OVERRIDES: dict[str, Any] | None = {"quant": None}
#
# Existing externally quantized HF/GPTQModel artifact (9B):
# CONFIG_OVERRIDES = {
#     "quant": {
#         "algorithm": "existing_hf",
#         "artifact_format": "gptqmodel_hf",
#         "existing_hf_model_dir": "weights/Qwen3.5-9B-mode1-llm-only",
#     }
# }
#
# Enable GDR fuse for export without changing the public API:
# CONFIG_OVERRIDES = {"export.model.fuse_gdr_ops": True}
CONFIG_OVERRIDES: dict[str, Any] | None = None


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def main() -> None:
    _remove_output_dir_if_needed(QUANT_OUTPUT_DIR, FORCE_OVERWRITE)
    _remove_output_dir_if_needed(EXPORT_OUTPUT_DIR, FORCE_OVERWRITE)

    workflow = AutoLLMWorkflow.from_config(
        hf_model_dir=HF_MODEL_DIR,
        config_path=CONFIG_PATH,
        seed=SEED,
        debug=DEBUG,
    )
    quant_result = workflow.quant(
        output_dir=QUANT_OUTPUT_DIR,
        device=DEVICE,
        config_overrides=CONFIG_OVERRIDES,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=EXPORT_OUTPUT_DIR,
        device=DEVICE,
        config_overrides=CONFIG_OVERRIDES,
    )

    workflow.dump_golden(export_result=export_result, device=DEVICE, input_messages={"text": PROMPT})

    print(f"quant_result: {quant_result}")
    print(f"export_result: {export_result}")
    if QUICK_TEST:
        quick_result = quick_test_hmonnx(
            export_result,
            prompt=PROMPT,
            device=DEVICE,
            max_new_tokens=QUICK_TEST_MAX_NEW_TOKENS,
            do_sample=False,
        )
        print_quick_test_result(quick_result)


if __name__ == "__main__":
    main()
