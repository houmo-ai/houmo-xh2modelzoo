import argparse

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from xhmodel_merak.xh_llm.models.spark_xh.spark.modeling_ipt import IPTForCausalLM


CLS = 131632
SEP = 131633
MASK = 131634


def main(args):
    # 可能用于处理特定分词器兼容性

    model_dir = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

    model: IPTForCausalLM = AutoModelForCausalLM.from_pretrained(  # type: ignore
        model_dir,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )

    assert isinstance(model, IPTForCausalLM), f"Expected model type IPTForCausalLM, but got {type(model).__name__}"

    # 模型裁剪逻辑
    layer_count = args.layers
    model.config.num_hidden_layers = layer_count
    model.config.num_layers = layer_count
    model.model.transformer.layers = model.model.transformer.layers[:layer_count]

    # 解耦操作
    model.unfuse_experts()
    model.unfuse_mlp()

    # 保存截断且解耦后的模型
    unfuse_expert_model_dir = f"{model_dir}_unfuse_{layer_count}"
    model.save_pretrained(unfuse_expert_model_dir)
    tokenizer.save_pretrained(unfuse_expert_model_dir)
    logger.info(f"Unfused and truncated model saved to {unfuse_expert_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Test Spark MoE (IPT) model")
    parser.add_argument("--model-dir", type=str, default="./data/models/xinghuo-30B-1108-wuxi-chunk")
    parser.add_argument(
        "--layers",
        type=int,
        default=-1,
        help="Number of layers to unfuse. Default is -1, which means unfuse all layers.",
    )
    args = parser.parse_args()
    main(args)
