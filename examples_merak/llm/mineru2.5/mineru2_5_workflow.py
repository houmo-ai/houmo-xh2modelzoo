import shutil
from pathlib import Path
from typing import Any


HF_MODEL_DIR = "/data02/datasets/MinerU2.5-Pro-2604-1.2B"
CONFIG_PATH = (
    "./configs_merak/workflows/xh2a/llm_models/mineru2_5/"
    "mineru2_5_pro_xh2a_4k.yaml"
)
QUANT_OUTPUT_DIR = "./work_dirs/mineru2_5_workflow_quant"
EXPORT_OUTPUT_DIR = "./work_dirs/mineru2_5_workflow_export"
DEVICE = "cuda"
SEED = 1024
DEBUG = False
FORCE_OVERWRITE = True

CONFIG_OVERRIDES: dict[str, Any] | None = None


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
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

    print(f"quant_result: {quant_result}")
    print(f"export_result: {export_result}")


if __name__ == "__main__":
    main()
