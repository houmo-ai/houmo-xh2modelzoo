import shutil
from pathlib import Path
from typing import Any



HF_MODEL_DIR = "/data02/datasets/Qwen2-VL-2B-Instruct"
CONFIG_PATH = "./configs_merak/workflows/xh2a/llm_models/qwen2_vl/2b/qwen2_vl_2b_xh2a_4k.yaml"
QUANT_OUTPUT_DIR = "./work_dirs/qwen2_vl_workflow_quant_overrides"
EXPORT_OUTPUT_DIR = "./work_dirs/qwen2_vl_workflow_export_overrides"
DEVICE = "cuda"
SEED = 1024
DEBUG = False
FORCE_OVERWRITE = True
IMAGE_PATH = "./data/images/qwen2_vl_demo.jpeg"
PROMPT = "描述这张图片"

# CONFIG_OVERRIDES: dict[str, Any] | None = None
CONFIG_OVERRIDES = {
    "export.model.context_max_length": 8192,
    "export.model.prefill_chunk_length": 512,
    "export.model.chip_arch": "XH2a",
}


def _remove_output_dir_if_needed(output_dir: Path, force: bool) -> None:
    p = Path(output_dir)
    if not force or not p.exists():
        return
    shutil.rmtree(p)


def main() -> None:
    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

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

    workflow.dump_golden(
        export_result=export_result,
        device=DEVICE,
        input_messages={
            "image": IMAGE_PATH,
            "text": PROMPT,
        },
    )

    print(f"quant_result: {quant_result}")
    print(f"export_result: {export_result}")


if __name__ == "__main__":
    main()
