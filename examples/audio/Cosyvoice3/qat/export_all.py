"""
CosyVoice3 QAT 权重导出（反量化）
==================================

将 QAT 训练后的量化权重做 QDQ 反量化，还原为与原始 llm.pt / flow.pt
key 结构一致的 FP32 state_dict，可在无 xhquant 环境中直接加载评测。

支持模块: llm, flow

流程（每个模块）:
    1. 加载原始 FP 模型 (hyperpyyaml)
    2. PTQ prepare (deploy mode) — 插入量化节点
    3. 加载 QAT checkpoint — 恢复训练后的量化参数
    4. 反量化 (dequantize_module_state_dict) — 从 QBaseModule 提取 FP 权重
    5. 保存 — 与原始 .pt key 结构一致

用法:
    python export_all.py \\
        --model_dir /data01/nfs_shared/ASR_TTS/CosyVoice3-0.5B-2512 \\
        --qat_dir ./output_cosyvoice3_qat_pred100 \\
        --qat_steps 5000
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

# 确保本目录在 import 路径中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qat_utils import (
    DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR,
    print_stage, _sanitize_numpy_attrs,
    load_checkpoint, save_model_weights,
    dequantize_module_state_dict,
    prepare_quanted_model_to_compile,
)


# ================================================================
#  量化配置
# ================================================================

def _quant_config(w_man_bit=8, deploy=True):
    return {
        "precision_mode": "aligned",
        "enable_qat_compiler_ops": not deploy,
        "w_schema": {
            "fp_mode": "sefp", "man_bit": w_man_bit,
            "hidden_bit": True, "nshare": 64, "rounding": "rne",
        },
        "act_schema": {
            "fp_mode": "sefp", "man_bit": 8,
            "hidden_bit": True, "nshare": 64, "rounding": "rne",
            "max_exp_boost": 0,
        },
    }


# ================================================================
#  通用 QDQ 导出 — 适用于通过 hyperpyyaml 加载的模块
# ================================================================

def export_dequant_for_module(
    mod_name: str,
    model_dir: str,
    qat_dir: Path,
    qat_steps: int,
    output_dir: Path,
    w_man_bit: int,
    mode: str = "qat",
):
    """对一个模块执行 PTQ prepare → [load QAT] → dequantize → save。

    Parameters
    ----------
    mod_name : "llm" 或 "flow"
    model_dir : 原始 CosyVoice3 模型目录
    qat_dir : QAT checkpoint 目录
    qat_steps : 训练步数 (用于定位 checkpoint 文件名)
    output_dir : 反量化 .pt 保存目录
    w_man_bit : 权重尾数位数
    mode : "qat" 或 "ptq"
    """
    from hyperpyyaml import load_hyperpyyaml

    tag = mod_name.upper()
    hf_model_dir = os.path.join(model_dir, "CosyVoice-BlankEN")
    yaml_path = os.path.join(model_dir, "cosyvoice3.yaml")
    ckpt_map = {
        "llm": os.path.join(model_dir, "llm.pt"),
        "flow": os.path.join(model_dir, "flow.pt"),
    }

    # ---- 1. 加载原始模型 ----
    print(f"\n[{tag}] 加载原始 FP 模型 ...")
    overrides = {"qwen_pretrain_path": hf_model_dir}
    for key in ("llm", "flow", "hift"):
        if key != mod_name:
            overrides[key] = None
    overrides["hifigan"] = None

    with open(yaml_path) as f:
        configs = load_hyperpyyaml(f, overrides=overrides)
    model = configs[mod_name]
    load_checkpoint(model, ckpt_map[mod_name], tag=tag)
    _sanitize_numpy_attrs(model)

    # ---- 2. PTQ prepare (deploy mode) ----
    print(f"[{tag}] PTQ prepare (deploy mode, w{w_man_bit}a8) ...")
    export_model = copy.deepcopy(model).cpu()
    quant_model = copy.deepcopy(model).cpu()
    del model

    qconfig = _quant_config(w_man_bit, deploy=True)
    qmodel, _ = prepare_quanted_model_to_compile(
        f"cosyvoice3_{mod_name}_dequant_export",
        quant_model, "xh2a", qconfig,
    )
    del quant_model

    # ---- 3. 加载 QAT checkpoint ----
    if mode == "qat":
        qat_path = qat_dir / mod_name / f"qat_steps{qat_steps}.pt"
        if not qat_path.exists():
            print(f"[{tag}] QAT checkpoint 不存在: {qat_path}")
            return None
        print(f"[{tag}] 加载 QAT checkpoint: {qat_path}")
        qat_sd = torch.load(qat_path, map_location="cpu", weights_only=True)
        missing, unexpected = qmodel.load_state_dict(qat_sd, strict=False)
        matched = len(qat_sd) - len(unexpected)

        # Pred100 wrapper 会产生双层前缀 (e.g. llm.llm.model.xxx)，
        # 独立模块只有单层 (llm.model.xxx)。直接加载 miss 时自动剥离。
        if matched == 0:
            prefix = mod_name + "."
            remapped = {k[len(prefix):]: v for k, v in qat_sd.items()
                        if k.startswith(prefix)}
            if remapped:
                missing, unexpected = qmodel.load_state_dict(
                    remapped, strict=False)
                matched = len(remapped) - len(unexpected)
                print(f"  [{tag}] key 前缀重映射: {matched}/{len(remapped)} matched")
        else:
            print(f"  [{tag}] 直接匹配: {matched}/{len(qat_sd)} keys")
        if missing:
            print(f"  missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"  unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    else:
        print(f"[{tag}] PTQ 模式，直接反量化")

    # ---- 4. 反量化 ----
    print(f"[{tag}] 反量化中 ...")
    dequant_sd = dequantize_module_state_dict(qmodel, export_model)
    export_model.load_state_dict(dequant_sd, strict=False)

    # ---- 5. 保存 ----
    save_name = f"dequant_steps{qat_steps}.pt" if mode == "qat" else "dequant_ptq.pt"
    save_path = output_dir / mod_name / save_name
    save_model_weights(export_model, save_path)

    del qmodel, export_model
    torch.cuda.empty_cache()
    return save_path


# ================================================================
#  主函数
# ================================================================

def main():
    p = argparse.ArgumentParser(description="CosyVoice3 QAT 权重导出")
    p.add_argument("--model_dir", default=DEFAULT_MODEL_DIR,
                   help="原始 CosyVoice3 模型目录")
    p.add_argument("--qat_dir", default=DEFAULT_OUTPUT_DIR,
                   help="QAT 训练输出目录")
    p.add_argument("--output_dir", default=None,
                   help="导出输出目录 (默认 = qat_dir)")
    p.add_argument("--qat_steps", type=int, default=200)
    p.add_argument("--w_man_bit", type=int, default=8)
    p.add_argument("--modules", nargs="+",
                   default=["llm", "flow"],
                   help="要导出的模块")
    p.add_argument("--mode", default="qat", choices=["qat", "ptq"])
    args = p.parse_args()

    qat_dir = Path(args.qat_dir)
    output_dir = Path(args.output_dir) if args.output_dir else qat_dir

    print_stage("CosyVoice3 QAT 权重导出")
    print(f"model_dir   = {args.model_dir}")
    print(f"qat_dir     = {qat_dir}")
    print(f"output_dir  = {output_dir}")
    print(f"modules     = {args.modules}")
    print(f"mode        = {args.mode}")
    print(f"qat_steps   = {args.qat_steps}")

    results = {}

    # ---- LLM / Flow ----
    for mod_name in ("llm", "flow"):
        if mod_name not in args.modules:
            continue
        path = export_dequant_for_module(
            mod_name=mod_name,
            model_dir=args.model_dir,
            qat_dir=qat_dir,
            qat_steps=args.qat_steps,
            output_dir=output_dir,
            w_man_bit=args.w_man_bit,
            mode=args.mode,
        )
        if path:
            results[mod_name] = path

    # ---- 汇总 ----
    print_stage("导出汇总")
    for name, path in results.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
