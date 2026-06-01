import argparse
import shutil
from pathlib import Path
import torch

from transformers import AutoConfig, AutoModelForCausalLM


def extract_single_layer_model(
    model_path: str,
    output_path: str,
    layer_index: int = 0,
    keep_first_n_layers: int = 1
):
    """
    从大模型中提取指定层数，保存为小模型
    
    Args:
        model_path: 原始模型路径
        output_path: 输出模型路径
        layer_index: 保留的起始层索引（默认0）
        keep_first_n_layers: 保留的层数（默认1）
    """
    print(f"加载模型: {model_path}")
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    original_num_layers = config.num_hidden_layers
    
    print(f"原始模型层数: {original_num_layers}")
    print(f"保留层数: {keep_first_n_layers} (从第 {layer_index} 层开始)")
    
    config.num_hidden_layers = keep_first_n_layers
    
    print(f"加载模型权重...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"保存模型到: {output_path}")
    model.save_pretrained(str(output_dir))
    
    print(f"保存配置文件...")
    config.save_pretrained(str(output_dir))
    
    print(f"成功！小模型已保存到: {output_path}")
    print(f"原始层数: {original_num_layers} -> 保留层数: {keep_first_n_layers}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="从大模型提取单层用于CI测试")
    parser.add_argument(
        "--model",
        type=str,
        default="/data02/datasets/Qwen3-30B-A3B",
        help="原始大模型路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/data02/datasets/Qwen3-30B-A3B-1layer",
        help="输出小模型路径"
    )
    parser.add_argument(
        "--layer-index",
        type=int,
        default=0,
        help="保留的起始层索引（默认0）"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=1,
        help="保留的层数（默认1）"
    )
    
    args = parser.parse_args()
    
    extract_single_layer_model(
        args.model,
        args.output,
        args.layer_index,
        args.num_layers
    )
