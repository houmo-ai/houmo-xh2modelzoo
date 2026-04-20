"""
CosyVoice3 评测模型目录构建
=============================

复制原始模型目录，用 QAT/PTQ 反量化权重替换对应文件，
生成可在无 xhquant 环境中直接加载的评测目录。

替换内容:
    - llm.pt  ← dequant_steps{N}.pt
    - flow.pt ← dequant_steps{N}.pt

用法:
    python setup_eval.py \\
        --src_model_dir /data01/nfs_shared/ASR_TTS/CosyVoice3-0.5B-2512 \\
        --qat_dir ./output_cosyvoice3_qat \\
        --output_dir ./eval_model_qat \\
        --qat_steps 5000
"""

import argparse
import os
import shutil
import torch


def parse_args():
    p = argparse.ArgumentParser(description="构建 QAT 评测模型目录")
    p.add_argument("--src_model_dir", required=True,
                   help="原始 CosyVoice3 模型目录")
    p.add_argument("--qat_dir", required=True,
                   help="QAT 导出目录 (export_all.py 的输出)")
    p.add_argument("--output_dir", required=True,
                   help="输出评测模型目录")
    p.add_argument("--mode", default="qat", choices=["qat", "ptq"])
    p.add_argument("--qat_steps", type=int, default=200)
    p.add_argument("--modules", nargs="+",
                   default=["llm", "flow"],
                   help="要替换的模块")
    return p.parse_args()


def verify_keys(orig_path, dequant_path, tag=""):
    """验证两个 state_dict 的 key 结构一致。"""
    orig = torch.load(orig_path, map_location="cpu", weights_only=True)
    deq = torch.load(dequant_path, map_location="cpu", weights_only=True)
    orig_keys, deq_keys = set(orig.keys()), set(deq.keys())

    if orig_keys != deq_keys:
        missing = orig_keys - deq_keys
        extra = deq_keys - orig_keys
        if missing:
            print(f"  [{tag}] WARNING: missing keys: {list(missing)[:5]}...")
        if extra:
            print(f"  [{tag}] WARNING: extra keys: {list(extra)[:5]}...")
        return False

    diff = sum(
        (orig[k].float() - deq[k].float()).abs().sum().item()
        for k in orig_keys
    )
    print(f"  [{tag}] keys OK ({len(orig_keys)}), total diff = {diff:.4f}")
    return True


def main():
    args = parse_args()

    # ---- 复制模型目录 ----
    if os.path.exists(args.output_dir):
        print(f"[WARN] {args.output_dir} 已存在，仅替换权重文件")
    else:
        print(f"[copy] {args.src_model_dir} -> {args.output_dir}")
        shutil.copytree(args.src_model_dir, args.output_dir)

    qat_dir = args.qat_dir.rstrip("/")
    save_name = (f"dequant_steps{args.qat_steps}.pt"
                 if args.mode == "qat" else "dequant_ptq.pt")

    # ---- 替换 .pt 权重 (llm, flow) ----
    pt_modules = [m for m in args.modules if m in ("llm", "flow")]
    for mod_name in pt_modules:
        dequant_path = os.path.join(qat_dir, mod_name, save_name)
        dst_path = os.path.join(args.output_dir, f"{mod_name}.pt")

        if not os.path.exists(dequant_path):
            print(f"  [{mod_name}] SKIP: {dequant_path} not found")
            continue

        print(f"\n[{mod_name}] verifying keys ...")
        if verify_keys(dst_path, dequant_path, tag=mod_name):
            shutil.copy2(dequant_path, dst_path)
            print(f"  [{mod_name}] replaced {dst_path}")
        else:
            print(f"  [{mod_name}] ABORT: key mismatch!")

    # ---- 替换 tokenizer ONNX ----
    if "tokenizer" in args.modules:
        tok_qat_onnx = os.path.join(
            qat_dir, "tokenizer", "speech_tokenizer_v3_qat.onnx"
        )
        dst_onnx = os.path.join(args.output_dir, "speech_tokenizer_v3.onnx")

        if os.path.exists(tok_qat_onnx):
            shutil.copy2(tok_qat_onnx, dst_onnx)
            print(f"\n[tokenizer] replaced {dst_onnx}")
        else:
            print(f"\n[tokenizer] SKIP: {tok_qat_onnx} not found")
            print(f"  请先运行: python tokenizer_export.py")

    print(f"\n{'=' * 60}")
    print(f"  评测目录已就绪: {args.output_dir}")
    print(f"  运行评测:")
    print(f"    python eval_cosyvoice3_fp.py --model_type qat \\")
    print(f"        --model_dir {args.output_dir} --max_samples 50")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
