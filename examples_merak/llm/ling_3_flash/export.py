#!/usr/bin/env python3
"""Export an original or GPTQModel Ling-3-Flash checkpoint to HMONNX."""

from __future__ import annotations

import argparse
import json
import shutil
from collections.abc import Mapping
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = (
    _REPO_ROOT
    / "configs_merak"
    / "workflows"
    / "xh2a"
    / "llm_models"
    / "ling_3_flash"
    / "ling_3_flash_gptq.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        "--model-dir",
        dest="model",
        required=True,
        help="Original or quantized HF directory",
    )
    parser.add_argument(
        "--config",
        "--config-path",
        dest="config",
        default=str(_DEFAULT_CONFIG),
    )
    parser.add_argument("--output", "--export-output-dir", dest="output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype",
        choices=("float16", "fp16", "bfloat16", "bf16", "float32", "fp32"),
        help="Override export.model.dtype (the xh2modelzoo default is float16)",
    )
    parser.add_argument(
        "--export-from-quanted-model",
        action="store_true",
        help=(
            "Treat --model as an existing GPTQModel/AutoRound HF checkpoint and "
            "skip the workflow quantization stage"
        ),
    )
    parser.add_argument("--context-max-length", type=int)
    parser.add_argument("--prefill-chunk-length", type=int)
    parser.add_argument("--max-layers", type=int)
    parser.add_argument("--only-first-block", action="store_true")
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument(
        "--dump-golden",
        action="store_true",
        help="Generate aligned prefill/decode golden data after export",
    )
    parser.add_argument(
        "--golden-device-map",
        nargs="+",
        default=None,
        help=(
            "Explicit HMONNX device map used only by --dump-golden. More "
            "than one CUDA entry enables HMONNXInferenceV2 auto-offload."
        ),
    )
    parser.add_argument(
        "--golden-prompt",
        default="请用一句话介绍你自己。",
        help="Text prompt used to generate prefill/decode golden data",
    )
    low_memory_group = parser.add_mutually_exclusive_group()
    low_memory_group.add_argument(
        "--low-memory",
        dest="low_memory",
        action="store_true",
        help="Enable Qwen3.5-style streamed placeholder export",
    )
    low_memory_group.add_argument(
        "--no-low-memory",
        dest="low_memory",
        action="store_false",
        help="Force the regular full-model export path",
    )
    parser.set_defaults(low_memory=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _prepare_output(path: Path, *, overwrite: bool) -> None:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return
    if not overwrite:
        raise FileExistsError(f"{resolved} exists; pass --overwrite to replace it")
    if resolved == Path(resolved.anchor) or len(resolved.parts) < 3:
        raise ValueError(f"Refusing to recursively remove broad path: {resolved}")
    shutil.rmtree(resolved)


def _configure_low_memory_export(enabled: bool | None) -> None:
    from xhmodel_merak.xh_llm.utils import configure_huge_model_export

    configure_huge_model_export(enabled)


def _validate_quanted_checkpoint(path: str | Path) -> None:
    model_dir = Path(path).expanduser().resolve()
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Quantized checkpoint has no config.json: {config_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    quantization_config = config.get("quantization_config")
    if not isinstance(quantization_config, Mapping):
        raise ValueError(
            "--export-from-quanted-model requires config.json to contain a "
            f"GPTQModel/AutoRound quantization_config: {config_path}"
        )
    if not (model_dir / "model.safetensors.index.json").is_file() and not list(
        model_dir.glob("*.safetensors")
    ):
        raise FileNotFoundError(f"Quantized checkpoint has no safetensors weights: {model_dir}")


def main() -> None:
    args = parse_args()
    _configure_low_memory_export(args.low_memory)
    if args.export_from_quanted_model:
        _validate_quanted_checkpoint(args.model)

    from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

    output = Path(args.output)
    _prepare_output(output, overwrite=args.overwrite)
    workflow = AutoLLMWorkflow.from_config(
        model_dir=args.model,
        config_path=args.config,
    )
    # This entrypoint consumes an already selected HF checkpoint. In
    # --export-from-quanted-model mode this is the same quant=None contract as
    # the Qwen3.5 workflow CLI: packed GPTQModel/AutoRound weights are consumed
    # directly and quantization is not run again.
    quant_result = workflow.quant(
        output_dir=str(output.parent / f"{output.name}_unused_quant"),
        device=args.device,
        config_overrides={"quant": None},
    )
    overrides: dict[str, object] = {}
    if args.dtype is not None:
        overrides["export.model.dtype"] = args.dtype
    if args.context_max_length is not None:
        if args.context_max_length <= 0:
            raise ValueError("--context-max-length must be positive")
        overrides["export.model.context_max_length"] = args.context_max_length
    if args.prefill_chunk_length is not None:
        if args.prefill_chunk_length <= 0 or args.prefill_chunk_length % 64:
            raise ValueError("--prefill-chunk-length must be a positive multiple of 64")
        overrides["export.model.prefill_chunk_length"] = args.prefill_chunk_length
    if args.max_layers is not None:
        if args.max_layers <= 0:
            raise ValueError("--max-layers must be positive")
        overrides["export.model.max_layers"] = args.max_layers
    if args.only_first_block:
        overrides["export.model.only_first_block"] = True
    if args.flash_attention:
        overrides["export.model.flash_attention.enable"] = True

    result = workflow.export(
        quant_result=quant_result,
        output_dir=str(output),
        device=args.device,
        config_overrides=overrides,
    )
    if args.dump_golden:
        from xhmodel_merak.xh_llm.utils import configure_hmonnx_validation_runtime

        configure_hmonnx_validation_runtime(use_v2=True, pack_w4=True)
        workflow.dump_golden(
            export_result=result,
            device=args.device,
            input_messages=args.golden_prompt,
            device_map=args.golden_device_map,
            use_v2=True,
        )
    print(f"export_dir={result.work_dir}")
    print(f"workflow_config={result.config_file}")


if __name__ == "__main__":
    main()
