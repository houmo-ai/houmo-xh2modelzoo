"""CosyVoice3 merak workflow runner.

Drives the migrated CosyVoice3 export workflow through the top-level
:class:`xhmodel_merak.workflows.AutoWorkflow` entrypoint.

Usage::

    CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$PWD python \
        examples_merak/tts/cosyvoice3/cosyvoice3_workflow.py \
        --model-dir <model_dir> --device cuda:0 --overwrite

    # with golden generation
    CUDA_VISIBLE_DEVICES=<gpu> PYTHONPATH=$PWD python \
        examples_merak/tts/cosyvoice3/cosyvoice3_workflow.py \
        --model-dir <model_dir> --device cuda:0 --dump-golden --overwrite
"""

import argparse
import shutil
from pathlib import Path


DEFAULT_CONFIG = "configs_merak/workflows/xh2a/other_models/cosyvoice3/cosyvoice3_0_5b.yaml"
DEFAULT_OUTPUT = "work_dirs/CosyVoice3-0.5B_XH2a"


def _remove_output_dir_if_needed(output_dir: str, force: bool) -> None:
    path = Path(output_dir)
    if force and path.exists():
        shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CosyVoice3 merak export workflow.")
    parser.add_argument("--model-dir", required=True, help="HF / CosyVoice3 pretrained model directory.")
    parser.add_argument("--config-path", default=DEFAULT_CONFIG, help="Workflow YAML path.")
    parser.add_argument("--export-output-dir", default=DEFAULT_OUTPUT, help="Export output directory.")
    parser.add_argument("--quant-output-dir", default="work_dirs/cosyvoice3_quant")
    parser.add_argument("--device", default="cuda", help="Device passed to workflow export.")
    parser.add_argument("--target-device", default=None, help="Override export.target_device.")
    parser.add_argument("--onnx-dir", default=None, help="Directory holding source onnx files for onnx submodels.")
    parser.add_argument("--llm-checkpoint", default=None, help="Override path to the CosyVoice3 llm.pt checkpoint.")
    parser.add_argument("--components", default=None, help="Comma-separated component list override.")
    parser.add_argument("--quant-type", default=None, help="Override all component quant types.")
    parser.add_argument("--dump-golden", action="store_true", help="Generate golden data after export.")
    parser.add_argument("--overwrite", action="store_true", help="Remove export output dir first.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


_ALL_COMPONENTS = (
    "llm",
    "llm_decoder",
    "speech_tokenizer_v3",
    "campplus",
    "flow_decoder",
    "hift",
    "spk_embed_affine_layer",
    "pre_lookahead_layer",
)


def _build_config_overrides(args: argparse.Namespace) -> dict[str, object]:
    overrides: dict[str, object] = {}
    if args.target_device is not None:
        overrides["export.target_device"] = args.target_device
    if args.onnx_dir is not None:
        overrides["export.onnx_dir"] = args.onnx_dir
    if args.llm_checkpoint is not None:
        overrides["export.llm_checkpoint"] = args.llm_checkpoint
    if args.components:
        overrides["export.components"] = [item.strip() for item in args.components.split(",") if item.strip()]
    if args.quant_type is not None:
        for name in _ALL_COMPONENTS:
            overrides[f"export.quant_types.{name}"] = args.quant_type
    return overrides


def main() -> None:
    args = parse_args()
    _remove_output_dir_if_needed(args.export_output_dir, args.overwrite)

    from xhmodel_merak.workflows import AutoWorkflow

    workflow = AutoWorkflow.from_config(
        model_dir=args.model_dir,
        config_path=args.config_path,
        debug=args.debug,
    )
    overrides = _build_config_overrides(args)
    quant_result = workflow.quant(
        output_dir=args.quant_output_dir,
        device=args.device,
        config_overrides=overrides,
    )
    export_result = workflow.export(
        quant_result=quant_result,
        output_dir=args.export_output_dir,
        device=args.device,
        config_overrides=overrides,
    )
    print(f"export_result: {export_result}")
    if args.dump_golden:
        golden_dir = workflow.dump_golden(
            export_result=export_result,
            device=args.device,
        )
        print(f"golden_dir: {golden_dir}")


if __name__ == "__main__":
    main()
