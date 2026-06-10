#!/usr/bin/env python3
# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

"""混合精度搜索 + HMONNX 导出一键脚本。

用法:
    python examples/cv/dinov3/2.mix_search_xh2_export_hmonnx.py

流程:
  1. 加载 COCO 采样图片
  2. ONNX → xhquant QuantGraph
  3. PTQ 校准
  4. DetectorMixPrecisionSearchV2 敏感度搜索
  5. 保存搜索结果 + sensitivity 报告
  6. ALIGNED 模式导出最终 HMONNX
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

# 将当前目录加入 sys.path，确保能 import 同目录下的模块
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from dinov3_common import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_ONNX,
    DEFAULT_QUANT_TYPE,
    DEFAULT_SEARCH_DIR,
    build_quant_config,
    detect_onnx_io,
    load_lightly_model,
    preprocess_image_batch,
    resolve_runtime_device,
    select_coco_image_ids,
)
from mix_search_tool import make_detector_searcher


DEFAULT_COCO_IMG = Path("/data01/datasets/coco2017/val2017")
DEFAULT_COCO_ANN = Path("/data01/datasets/coco2017/annotations/instances_val2017.json")
DEFAULT_MIX_SEARCH_OUT_DIR = DEFAULT_SEARCH_DIR / "auto_search_sample8_top40_v2"
DEFAULT_MIX_WEIGHTED_SEARCH_CANDIDATES = (
    "XH2aQuantQLinear",
    "XH2aQuantQConv2d",
    "XH2aQuantQConvTranspose2d",
    "XH2aQuantQMoeBlock",
    "XH2aQuantQMatMul",
    "XH2aQuantQGroupMatMul",
)
W16_WEIGHT_SCHEMA = {"bits": 16, "fp_mode": "sefp", "hidden_bit": True}
W16_ACT_SCHEMA = {"bits": 16}


class OptionalDependencyError(RuntimeError):
    """Raised when an optional parser dependency is missing."""


def parse_int_list(value: str | Sequence[int]) -> list[int]:
    """Parse a comma-separated integer list, for example '8,16' -> [8, 16]."""
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
        if not items:
            raise ValueError("Expected a comma-separated integer list")
        return [int(item) for item in items]
    return [int(item) for item in value]


def load_yaml_or_json_mapping(path: str | Path | None) -> dict[str, Any]:
    """Load optional JSON/YAML overrides for the mix-search configuration."""
    if path is None:
        return {}
    config_path = Path(path)
    if config_path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise OptionalDependencyError("PyYAML is required for YAML mix-search configs.") from exc
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return data or {}
    data = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {config_path}")
    return data


def build_mix_search_config(args: argparse.Namespace) -> dict[str, Any]:
    """Build the xhquant detector mixed-precision search config."""
    config = {
        "topk": args.mix_topk,
        "target_ratio": args.mix_target_ratio,
        "weighted_topk": args.mix_weighted_topk,
        "weighted_count": args.mix_weighted_count,
        "weighted_target_ratio": args.mix_weighted_target_ratio,
        "te_a_topk": args.mix_te_a_topk,
        "te_a_count": args.mix_te_a_count,
        "te_a_target_ratio": args.mix_te_a_target_ratio,
        "weight_bits": parse_int_list(args.mix_weight_bits),
        "act_bits": parse_int_list(args.mix_act_bits),
        "policy": args.mix_policy,
        "threshold": args.mix_threshold,
        "task": "cv_det",
        "metric": args.mix_metric,
        "key_name": "loss",
        "box_scale": float(args.image_size),
        "score_weight": args.mix_score_weight,
        "label_weight": args.mix_label_weight,
        "search_candidates": list(DEFAULT_MIX_WEIGHTED_SEARCH_CANDIDATES),
    }
    config.update(load_yaml_or_json_mapping(args.mix_search_config))
    return config


def qspec_to_summary(qspec: Any) -> dict[str, Any]:
    """Extract stable qspec fields for JSON summaries."""
    if qspec is None:
        return {}
    summary: dict[str, Any] = {}
    for name in ("man_bit", "exp_bit", "hidden_bit", "max_exp_boost"):
        if hasattr(qspec, name):
            value = getattr(qspec, name)
            if isinstance(value, np.integer):
                value = int(value)
            summary[name] = value
    return summary


def quant_cfg_man_bit(cfg: Any) -> int | None:
    """Return a quant config mantissa bit count, if available."""
    qspec = getattr(cfg, "qspec", None)
    if qspec is None:
        return None
    value = getattr(qspec, "man_bit", None)
    return int(value) if value is not None else None


def search_result_summary(search_result: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize mixed-precision search results for search_summary.json."""
    rows = []
    counts: dict[str, int] = {}
    for name, settings in sorted(search_result.items()):
        w_cfg = settings.get("w_cfg", settings.get("i_cfg_2"))
        a_cfg = settings.get("i_cfg", settings.get("i_cfg_1", settings.get("o_cfg")))
        o_cfg = settings.get("o_cfg")
        w_bit = quant_cfg_man_bit(w_cfg)
        a_bit = quant_cfg_man_bit(a_cfg)
        o_bit = quant_cfg_man_bit(o_cfg)
        key = f"w{w_bit}a{a_bit}" if w_bit is not None else f"a{a_bit}"
        counts[key] = counts.get(key, 0) + 1
        rows.append(
            {
                "name": name,
                "kind": settings.get("kind", "weighted" if w_bit is not None else "activation"),
                "precision": key,
                "weight": qspec_to_summary(getattr(w_cfg, "qspec", None)),
                "activation": qspec_to_summary(getattr(a_cfg, "qspec", None)),
                "output": qspec_to_summary(getattr(o_cfg, "qspec", None)),
                "output_bit": o_bit,
            }
        )
    return {
        "num_nodes": len(rows),
        "precision_counts": dict(sorted(counts.items())),
        "nodes": rows,
    }


def search_result_weighted_w16_config(search_result: Mapping[str, Any]) -> dict[str, Any]:
    """Extract weighted/TE W16 nodes into the canonical mixed-precision config."""
    nodes: dict[str, dict[str, Any]] = {}
    for name, settings in sorted(search_result.items()):
        weight_cfg = settings.get("w_cfg")
        matmul_rhs_cfg = settings.get("i_cfg_2")
        weighted_cfg = matmul_rhs_cfg if matmul_rhs_cfg is not None else weight_cfg
        if weighted_cfg is None or quant_cfg_man_bit(weighted_cfg) != 16:
            continue

        node_config: dict[str, Any] = {
            "act_schema": dict(W16_ACT_SCHEMA),
            "o_schema": dict(W16_ACT_SCHEMA),
        }
        if matmul_rhs_cfg is not None:
            node_config["act_schema_2"] = dict(W16_ACT_SCHEMA)
        else:
            node_config["w_schema"] = dict(W16_WEIGHT_SCHEMA)
        nodes[name] = node_config

    return {
        "input_enable_fp32": False,
        "output_enable_fp32": True,
        "ops": {},
        "nodes": nodes,
    }


def load_coco_batches_for_search(args: argparse.Namespace) -> tuple[list[Any], list[int]]:
    """Load sampled COCO images and preprocess them for sensitivity search."""
    from pycocotools.coco import COCO

    coco_gt = COCO(str(args.annotations))
    image_ids = coco_gt.getImgIds()
    if args.skip_missing_images:
        image_root = Path(args.images_dir)
        image_ids = [
            image_id
            for image_id in image_ids
            if (image_root / coco_gt.loadImgs([image_id])[0]["file_name"]).exists()
        ]
    image_ids = select_coco_image_ids(
        image_ids,
        limit=args.limit,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
    )
    if not image_ids:
        raise RuntimeError("No COCO images selected for mix precision search.")

    model = load_lightly_model(args)
    model.eval()
    model.deploy()

    batches = []
    for index, image_id in enumerate(image_ids, start=1):
        info = coco_gt.loadImgs([image_id])[0]
        image_path = Path(args.images_dir) / info["file_name"]
        if not image_path.exists():
            raise FileNotFoundError(f"COCO image not found: {image_path}")
        batch, _metadata = preprocess_image_batch(model, image_path)
        batches.append(batch.detach().cpu().float())
        if index % args.log_every == 0:
            print(f"Prepared {index}/{len(image_ids)} COCO search images")
    return batches, image_ids


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for mixed-precision search and HMONNX export."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ONNX / 数据
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--images-dir", default=str(DEFAULT_COCO_IMG), help="COCO val2017 图片目录")
    parser.add_argument("--annotations", default=str(DEFAULT_COCO_ANN), help="COCO instances_val2017.json")

    # 输出
    parser.add_argument("--out-dir", default=str(DEFAULT_MIX_SEARCH_OUT_DIR))
    parser.add_argument("--out", default=None, help="最终 HMONNX 输出路径")

    # 量化
    parser.add_argument("--quant-type", default=DEFAULT_QUANT_TYPE)
    parser.add_argument("--input-enable-fp32", action="store_true")
    parser.add_argument("--output-enable-fp32", action=argparse.BooleanOptionalAction, default=True)

    # 模型
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-name", default="dinov3/vitt16-ltdetr-coco")
    parser.add_argument("--model-cache-dir", default="examples/cv/dinov3/models")
    parser.add_argument("--data-cache-dir", default="examples/cv/dinov3/data_cache")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--output-names", default=None, help="逗号分隔的 ONNX 输出名（默认自动检测）")

    # 搜索
    parser.add_argument("--sample-size", type=int, default=8)
    parser.add_argument("--sample-seed", type=int, default=4200)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-missing-images", action="store_true")
    parser.add_argument("--mix-topk", type=float, default=0.4)
    parser.add_argument("--mix-target-ratio", type=float, default=1.0)
    parser.add_argument("--mix-weighted-topk", type=float, default=None)
    parser.add_argument("--mix-weighted-count", type=int, default=None)
    parser.add_argument("--mix-weighted-target-ratio", type=float, default=None)
    parser.add_argument("--mix-te-a-topk", dest="mix_te_a_topk", type=float, default=None)
    parser.add_argument("--mix-te-a-count", dest="mix_te_a_count", type=int, default=None)
    parser.add_argument("--mix-te-a-target-ratio", dest="mix_te_a_target_ratio", type=float, default=None)
    parser.add_argument("--mix-weight-bits", default="8,16")
    parser.add_argument("--mix-act-bits", default="8,16")
    parser.add_argument("--mix-policy", default="topk", choices=["coverage", "topk", "threshold"])
    parser.add_argument("--mix-threshold", type=float, default=1e-6)
    parser.add_argument("--mix-metric", default="l1", choices=["l1", "sqnr", "kl"])
    parser.add_argument("--mix-score-weight", type=float, default=1.0)
    parser.add_argument("--mix-label-weight", type=float, default=0.25)
    parser.add_argument("--mix-search-config", default=None, help="JSON/YAML 覆盖搜索配置")
    parser.add_argument("--log-every", type=int, default=10)
    return parser


def main(argv: list[str] | None = None) -> Path:
    parser = build_parser()
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 输出路径 ──
    onnx_path = Path(args.onnx).expanduser().resolve()
    search_result_path = out_dir / "search_result.pt"
    summary_path = out_dir / "search_summary.json"
    sensitivity_path = out_dir / "search_sensitivity.json"
    config_path = out_dir / "mix_search_config.json"
    weighted_w16_config_path = out_dir / "mixed_precision_weighted_w16.json"
    hmonnx_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else out_dir / "hmonnx" / f"{onnx_path.stem}_mix_search_{args.quant_type}_XH2a.onnx"
    )
    hmonnx_path.parent.mkdir(parents=True, exist_ok=True)

    # ── 加载 COCO 采样图片 ──
    batches, image_ids = load_coco_batches_for_search(args)
    print(f"Loaded {len(batches)} COCO images for search")

    # ── ONNX → QuantGraph ──
    from xhquant.api import DeviceType
    from xhquant.api.ptq_export_hmonnx import _convert_model_to_quanted_model, convert_quanted_model_to_hmonnx
    from xhquant.common.types import FrontendType

    input_name, _input_shape, output_names_list = detect_onnx_io(
        onnx_path, batch_size=args.batch_size, image_size=args.image_size,
    )
    if args.output_names:
        output_names_list = [n for n in args.output_names.split(",") if n]

    example_input = batches[0]
    quant_config = build_quant_config(args, mixed_precision=None)
    quanted_graph_module = _convert_model_to_quanted_model(
        str(onnx_path), FrontendType.ONNX, [example_input.cpu()],
        DeviceType.XH2a, quant_config=quant_config, use_ptq=False, input_names=[input_name],
    )

    # ── PTQ 校准 ──
    from xhquant.common.types import PrecisionMode
    from xhquant.quantization import ptq_quantize

    execution_device = torch.device(resolve_runtime_device(args.device))
    ptq_quantize(quanted_graph_module, [[example_input.cpu()]], PrecisionMode.FAST, execution_device)
    quanted_graph_module.half()
    quanted_graph_module = quanted_graph_module.to(execution_device)
    print("PTQ calibration done")

    # ── 混合精度搜索 ──
    mix_search_cfg = build_mix_search_config(args)
    config_path.write_text(json.dumps(mix_search_cfg, indent=2, sort_keys=True), encoding="utf-8")

    search_batches = [b.half().to(execution_device) for b in batches]
    searcher = make_detector_searcher(quanted_graph_module, mix_search_cfg)
    searcher.search(search_batches, PrecisionMode.FAST, labels=None)
    search_result = searcher.get_search_result()
    torch.save(search_result, search_result_path)
    print(f"Search done, {len(search_result)} nodes in result")

    # ── 保存报告 ──
    sensitivity_report = getattr(searcher, "sensitivity_report", {})
    sensitivity_report.update({
        "onnx": str(onnx_path), "hmonnx": str(hmonnx_path), "config": str(config_path),
        "search_result": str(search_result_path), "image_ids": image_ids,
        "sample_size": len(image_ids), "sample_seed": args.sample_seed,
    })
    sensitivity_path.write_text(json.dumps(sensitivity_report, indent=2, sort_keys=True), encoding="utf-8")

    weighted_w16_config = search_result_weighted_w16_config(search_result)
    weighted_w16_config_path.write_text(
        json.dumps(weighted_w16_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    summary = search_result_summary(search_result)
    summary.update({
        "onnx": str(onnx_path), "hmonnx": str(hmonnx_path), "config": str(config_path),
        "search_result": str(search_result_path), "sensitivity_summary": str(sensitivity_path),
        "weighted_w16_config": str(weighted_w16_config_path),
        "weighted_w16_num_nodes": len(weighted_w16_config["nodes"]),
        "image_ids": image_ids, "sample_size": len(image_ids), "sample_seed": args.sample_seed,
    })
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    # ── 导出 HMONNX ──
    from xhquant.quantization.quantize_wrap import fired_quant

    quanted_graph_module.enable_aligned_precision_mode()
    quanted_graph_module = quanted_graph_module.to("cpu")
    quanted_graph_module = fired_quant(quanted_graph_module)
    ptq_quantize(quanted_graph_module, [[example_input.cpu()]], PrecisionMode.ALIGNED, execution_device)
    convert_quanted_model_to_hmonnx(
        quanted_graph_module, [example_input.cpu()], str(hmonnx_path),
        [input_name], output_names_list,
    )

    print(f"\n✓ HMONNX:  {hmonnx_path}")
    print(f"✓ Config:  {config_path}")
    print(f"✓ W16 cfg: {weighted_w16_config_path} ({len(weighted_w16_config['nodes'])} nodes)")
    print(f"✓ Summary: {summary_path}")
    print(f"✓ Result:  {search_result_path}")
    return hmonnx_path


if __name__ == "__main__":
    main()
