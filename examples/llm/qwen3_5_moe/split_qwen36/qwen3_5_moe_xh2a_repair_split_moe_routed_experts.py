from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from examples.llm.qwen3_5_moe.split_qwen36.qwen3_5_moe_xh2a_export_split_moe_hmonnx import (
    _apply_update_cfg,
    _build_convert_config,
    _export_expert_layers,
    _get_text_config,
    _get_wrapped_text_model,
    _load_models,
    _resolve_split_quant_types,
)
from examples.llm.qwen3_5_moe.split_qwen36.qwen3_5_moe_xh2a_split_moe_hmonnx_e2e_test import SplitMoEHMONNXRunner
from xhquant.api import get_root_logger, xhquant_init


DEFAULT_MODEL = "/data01/datasets/qwen36moe-no-rotate-attn8-shared8-n256-iter400"
DEFAULT_WORK_DIR = (
    "work_dirs/qwen36moe-no-rotate-attn8-shared8-n256-iter400-split-moe-premoe-w8a8h0_sefp-"
    "experts-w4a8h0_sefp"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair stale split-MoE W4 routed expert HMONNX files for the experts selected by one token. "
            "The model is loaded once, then each repaired layer is verified through the split chain."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="GPTQModel/HuggingFace model directory")
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR, help="Split-MoE export directory")
    parser.add_argument("--token-id", type=int, default=12675, help="Token used to discover routed experts")
    parser.add_argument("--start-layer", type=int, default=0, help="First layer to inspect/repair")
    parser.add_argument("--max-layers", type=int, default=None, help="Exclusive layer stop bound; default uses meta")
    parser.add_argument(
        "--restore-layer-spec",
        default=None,
        help=(
            "Layer spec to restore before wrapping. Defaults to the requested repair range. "
            "Keep this bounded; restoring many MoE layers at once is memory-heavy."
        ),
    )
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--decode-sequence-length", type=int, default=1)
    parser.add_argument("--expert-sequence-length", type=int, default=1)
    parser.add_argument("--premoe-quant-type", default="w8a8h0_sefp")
    parser.add_argument("--expert-quant-type", default="w4a8h0_sefp")
    parser.add_argument("--head-quant-type", default=None)
    parser.add_argument("--quant-type", default="w8a8h0_sefp")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--linear-attention-mode", default="recurrent", choices=["auto", "chunk", "recurrent"])
    parser.add_argument("--linear-chunk-size", type=int, default=64)
    parser.add_argument("--no-split-conv-cache", dest="split_conv_cache", action="store_false")
    parser.set_defaults(split_conv_cache=True)
    parser.add_argument("--normalize-force-fp32", action="store_true", default=False)
    parser.add_argument("--use-manual-depthwise-conv1d", action="store_true", default=False)
    parser.add_argument("--fuse-gdr-ops", action="store_true", default=False)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--execution-device", default="cuda")
    parser.add_argument("--zero-threshold", type=float, default=0.0, help="Treat max_abs <= this as stale/zero")
    parser.add_argument("--force", action="store_true", help="Re-export even when current routed outputs are nonzero")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    args.prefill_sequence_length = 0
    args.quant_weight = None
    args.work_dir = str(Path(args.work_dir).resolve())
    args.model = str(Path(args.model).resolve())
    args.export_parts = "experts"
    args.export_experts = "all"
    args.skip_premoe = True
    args.skip_premoe_prefill = True
    args.skip_premoe_decode = True
    args.skip_experts = False
    args.skip_postmoe = True
    args.skip_head = True
    args.head_sequence_length = 0
    args.overwrite = True
    args.premoe_quant_type, args.expert_quant_type, args.head_quant_type = _resolve_split_quant_types(args)
    args.quant_type = args.premoe_quant_type
    return args


def _route_stats(info: dict[str, torch.Tensor], layer_idx: int) -> tuple[list[int], list[float]]:
    expert_ids: list[int] = []
    max_abs: list[float] = []
    for route_idx in range(8):
        expert_ids.append(int(info[f"split_layer_{layer_idx}_expert_route_{route_idx}_id"].item()))
        tensor = info[f"split_layer_{layer_idx}_expert_route_{route_idx}_out"].float()
        max_abs.append(float(tensor.abs().max().item()))
    return expert_ids, max_abs


def _run_layer(
    work_dir: Path,
    token_id: int,
    layer_idx: int,
    device: str,
    execution_device: str,
) -> tuple[list[int], list[float]]:
    runner = SplitMoEHMONNXRunner(
        work_dir,
        device=torch.device(device),
        exec_device=torch.device(execution_device),
    )
    try:
        _, info = runner.forward_token_with_intermediates(
            token_id,
            max_layers=layer_idx + 1,
            include_layer_details=True,
        )
        return _route_stats(info, layer_idx)
    finally:
        runner.close()


def _qweight_counts(work_dir: Path, layer_idx: int, expert_ids: Iterable[int]) -> dict[int, list[int]]:
    counts: dict[int, list[int]] = {}
    for expert_id in expert_ids:
        path = work_dir / "hmonnx" / "experts" / f"layer_{layer_idx:03d}" / f"expert_{expert_id:03d}.onnx"
        model = onnx.load(str(path), load_external_data=True)
        expert_counts: list[int] = []
        for init in model.graph.initializer:
            if "qweight" not in init.name:
                continue
            expert_counts.append(int(np.count_nonzero(numpy_helper.to_array(init))))
        counts[int(expert_id)] = expert_counts
    return counts


def _all_nonzero(values: Sequence[float], threshold: float) -> bool:
    return all(value > threshold for value in values)


def main() -> None:
    start_time = time.time()
    args = _parse_args()
    work_dir = Path(args.work_dir)
    meta = json.loads((work_dir / "split_moe_meta.json").read_text(encoding="utf-8"))
    num_layers = int(meta["num_hidden_layers"])
    stop_layer = num_layers if args.max_layers is None else min(int(args.max_layers), num_layers)
    if args.start_layer < 0 or args.start_layer >= stop_layer:
        raise ValueError(f"Invalid layer range: start={args.start_layer}, stop={stop_layer}")

    if args.restore_layer_spec is None:
        args.export_layers = (
            str(args.start_layer) if stop_layer == args.start_layer + 1 else f"{args.start_layer}-{stop_layer - 1}"
        )
    else:
        args.export_layers = str(args.restore_layer_spec)
    xhquant_init(work_dir / "split_moe_repair_routed_experts.log", debug=args.debug)
    logger = get_root_logger()
    logger.info(f"Repair split routed experts in {work_dir}")
    logger.info(f"token_id={args.token_id}, layers={args.start_layer}..{stop_layer - 1}")
    logger.info(f"restore_layer_spec={args.export_layers}")

    config = _build_convert_config(args)
    converter, native_model, wrapped_model, wrap_cfg, _ = _load_models(args, config)
    text_config = _get_text_config(native_model)
    wrap_cfg_decode = deepcopy(wrap_cfg)
    wrap_cfg_decode.input_sequence_length = int(args.decode_sequence_length)
    wrapped_model.apply(lambda module: _apply_update_cfg(module, wrap_cfg_decode))
    wrapped_model.eval()
    wrapped_text_model = _get_wrapped_text_model(wrapped_model)

    repaired: list[tuple[int, list[int], dict[int, list[int]]]] = []
    for layer_idx in range(int(args.start_layer), stop_layer):
        expert_ids, before_max_abs = _run_layer(
            work_dir,
            args.token_id,
            layer_idx,
            args.device,
            args.execution_device,
        )
        print(f"layer {layer_idx}: experts={expert_ids} before_max_abs={before_max_abs}")
        if _all_nonzero(before_max_abs, args.zero_threshold) and not args.force:
            print(f"layer {layer_idx}: already nonzero, skipping export")
            continue

        _export_expert_layers(
            args,
            converter,
            wrapped_text_model,
            text_config,
            [layer_idx],
            expert_ids,
            work_dir,
            logger,
            quant_type=args.expert_quant_type,
        )
        counts = _qweight_counts(work_dir, layer_idx, expert_ids)
        print(f"layer {layer_idx}: qweight_counts={counts}")

        _, after_max_abs = _run_layer(
            work_dir,
            args.token_id,
            layer_idx,
            args.device,
            args.execution_device,
        )
        print(f"layer {layer_idx}: after_max_abs={after_max_abs}")
        if not _all_nonzero(after_max_abs, args.zero_threshold):
            raise RuntimeError(f"Layer {layer_idx} still has zero routed outputs after repair: {after_max_abs}")
        repaired.append((layer_idx, expert_ids, counts))

    print("\nRepair summary:")
    print(f"  elapsed_s: {time.time() - start_time:.0f}")
    print(f"  repaired_layers: {[layer for layer, _, _ in repaired]}")
    for layer_idx, expert_ids, counts in repaired:
        print(f"  layer {layer_idx}: experts={expert_ids}, qweight_counts={counts}")


if __name__ == "__main__":
    main()
