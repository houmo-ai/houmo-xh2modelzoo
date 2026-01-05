import argparse
import os
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from gptqmodel import GPTQModel, QuantizeConfig


def build_calib_dataset(tokenizer, nsamples: int = 128, seqlen: int = 2048):
    texts = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train').select(range(nsamples))["text"]
    dataset = []
    for t in texts:
        enc = tokenizer(t, truncation=True, max_length=seqlen, return_tensors="pt")
        dataset.append({
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
        })
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/data01/datasets/Qwen2-7B", help="HF 模型 ID 或本地路径，例如 mistralai/GptOss-8x7B-v0.1")
    parser.add_argument("--out-dir", type=str, default=None)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--rotation", type=str, default=None, choices=[None, "hadamard"], help="可选旋转优化")
    parser.add_argument("--sym", default=True, help="对称量化")
    parser.add_argument("--mse", default=False, help="开启 MSE 辅助优化",action="store_true")
    parser.add_argument("--hessian-mse", default=False, help="开启 Hessian MSE 辅助优化",action="store_true")

    parser.add_argument("--demo", action="store_true", help="简单推理验证")
    parser.add_argument("--demo-prompt", type=str, default="中国是一个", help="简单推理验证的提示词")
    args = parser.parse_args()

    model_id = os.path.normpath(args.model)
    model_name = Path(model_id).name
    
    save_dir = f"output/{model_name}-{args.bits}bit-{args.group_size}g"

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, trust_remote_code=True)
    calib_dataset = build_calib_dataset(tokenizer, nsamples=args.nsamples, seqlen=args.seqlen)

    qcfg = QuantizeConfig(
        bits=args.bits,
        group_size=args.group_size,
        mse=bool(args.mse),
        rotation=args.rotation,
        hessian_mse = args.hessian_mse,
        device="cuda"
    )

    model = GPTQModel.load(
        model_id, 
        qcfg, 
        trust_remote_code=True,
        # device_map={"": "cuda:0"}  # 所有模块都在 GPU 0 上，关闭 tensor 并行
    )

    # 根据显存调整 batch_size 以加速量化
    model.quantize(calib_dataset, batch_size=args.batch_size)

    model.save(save_dir)

    # 简单推理验证
    if args.demo:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = GPTQModel.load(save_dir, device=device, trust_remote_code=True)
        prompt = args.demo_prompt
        print(tokenizer.decode(model.generate(**tokenizer(prompt, return_tensors="pt").to(device))[0]))


if __name__ == "__main__":
    main()