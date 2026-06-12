import shutil
from pathlib import Path
from typing import Any


HF_MODEL_DIR = "/data02/datasets/Qwen3-1.7B"
CONFIG_PATH = "./configs_merak/workflows/xh2a/llm_models/qwen3_legacy/1_7b/qwen3_1_7b_legacy_xh2a_w4a8_gptq_2k.yaml"
QUANT_OUTPUT_DIR = "./work_dirs/qwen3_legacy_workflow_quant"
EXPORT_OUTPUT_DIR = "./work_dirs/qwen3_legacy_workflow_export"
DEVICE = "cuda"
SEED = 1024
DEBUG = False
FORCE_OVERWRITE = True
PROMPT = "你多大了？用中文回答。"

# CONFIG_OVERRIDES: dict[str, Any] | None = None
CONFIG_OVERRIDES: dict[str, Any] | None = {
    "export.model.context_max_length": 8192,
    "export.model.prefill_chunk_length": 256,
    "export.model.chip_arch": "XH2a",
}


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    p = Path(output_dir)
    if not force or not p.exists():
        return
    shutil.rmtree(p)


def main() -> None:
    from xhmodel_merak.xh_llm.workflows.models.qwen3_legacy import XHQwen3LegacyHMONNXWorkflow

    _remove_output_dir_if_needed(QUANT_OUTPUT_DIR, FORCE_OVERWRITE)
    _remove_output_dir_if_needed(EXPORT_OUTPUT_DIR, FORCE_OVERWRITE)

    workflow = XHQwen3LegacyHMONNXWorkflow(
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

    workflow.dump_golden(
        export_result=export_result,
        device=DEVICE,
        input_messages={
            "text": PROMPT,
        },
    )

    print(f"quant_result: {quant_result}")
    print(f"export_result: {export_result}")


if __name__ == "__main__":
    main()
