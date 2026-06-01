"""
GigaBrain-0.1 其他模块导出脚本
导出 Action Heads、Time MLP 等模块
"""

import os
import torch
import numpy as np
import torch.nn as nn
import random
import onnx
import os.path as osp
from copy import deepcopy
from onnxsim import simplify
import argparse
import onnxruntime as ort

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
workdir = os.path.join(ROOT_DIR, "workdir")
hmonnx_file = os.path.join(ROOT_DIR, "hmonnx")
os.makedirs(workdir, exist_ok=True)
os.makedirs(hmonnx_file, exist_ok=True)

DEVICE = "cpu"


def set_seed(seed=42):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_gigabrain_model(model_path, device="cpu"):
    """加载 GigaBrain 模型"""
    print(f"Loading GigaBrain model from {model_path}...")

    # 添加 giga_models 路径
    import sys
    giga_models_path = "/data01/home/she.gao/vla/giga/giga-models"
    if giga_models_path not in sys.path:
        sys.path.insert(0, giga_models_path)

    # 使用 GigaBrain0Policy.from_pretrained 加载
    from giga_models.models import GigaBrain0Policy

    policy = GigaBrain0Policy.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
    )
    policy.to(device)
    policy.eval()
    print("Model loaded successfully using GigaBrain0Policy.from_pretrained!")
    return policy


def export_action_in_proj(policy, output_dir, device="cpu"):
    """导出 action_in_proj 模块
    
    Args:
        policy: GigaBrain 模型
        output_dir: 输出目录
        device: 推理设备
    """
    print("\n" + "="*60)
    print("Exporting action_in_proj...")
    print("="*60)
    
    action_in_proj = policy.action_in_proj
    action_in_proj.eval()
    action_in_proj = action_in_proj.float()
    
    # 输入形状: (batch, n_action_steps, max_action_dim)
    # max_action_dim = 32, n_action_steps = 50
    n_action_steps = policy.config.get('n_action_steps', 50)
    max_action_dim = policy.config.get('max_action_dim', 32)
    
    # 需要传入 emb_ids
    input_action = torch.randn(1, n_action_steps, max_action_dim, dtype=torch.float32)
    emb_ids = torch.tensor([0], dtype=torch.long)  # embodiment_id = 0 (AgileX)
    
    temp_onnx = osp.join(output_dir, "gigabrain_action_in_proj_temp.onnx")
    simplified_onnx = osp.join(output_dir, "gigabrain_action_in_proj.onnx")
    
    # 导出 ONNX (静态图，无 dynamic_axes)
    torch.onnx.export(
        action_in_proj,
        (input_action, emb_ids),
        temp_onnx,
        input_names=["action", "emb_ids"],
        output_names=["action_emb"],
        opset_version=17,
        verbose=False,
    )
    
    # 简化 ONNX
    onnx_model = onnx.load(temp_onnx)
    model_simplified, check = simplify(
        onnx_model,
        test_input_shapes={
            "action": [1, n_action_steps, max_action_dim],
            "emb_ids": [1],
        },
    )
    if check:
        onnx.save(model_simplified, simplified_onnx)
        print(f"Saved to {simplified_onnx}")
    else:
        onnx.save(onnx_model, simplified_onnx)
        print(f"Simplify failed, saved original to {simplified_onnx}")
    
    os.remove(temp_onnx)
    
    # 验证
    with torch.no_grad():
        torch_out = action_in_proj(input_action, emb_ids)
    torch_out_np = torch_out.cpu().numpy()
    
    sess = ort.InferenceSession(simplified_onnx, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {
        "action": input_action.cpu().numpy(),
        "emb_ids": emb_ids.cpu().numpy(),
    })[0]
    
    print(f"Max diff: {np.max(np.abs(torch_out_np - onnx_out))}")
    print(f"Mean diff: {np.mean(np.abs(torch_out_np - onnx_out))}")
    print(f"Allclose: {np.allclose(torch_out_np, onnx_out, rtol=1e-3, atol=1e-4)}")
    print(f"Output shape: {torch_out.shape}")

    # 转换为 HMONNX
    hmonnx_output = convert_to_hmonnx(
        simplified_onnx,
        "action_in_proj.onnx",
        {"action": (1, n_action_steps, max_action_dim), "emb_ids": (1,)}
    )

    return simplified_onnx, hmonnx_output

    return simplified_onnx


def convert_to_hmonnx(onnx_file, output_name, input_shapes_dict):
    """转换 ONNX 到 HMONNX 格式

    Args:
        onnx_file: ONNX 文件路径
        output_name: 输出文件名（不含路径）
        input_shapes_dict: 输入形状字典，例如 {"action": (1, 50, 32), "emb_ids": (1,)}
    """
    print("\n" + "=" * 60)
    print(f"Converting {output_name} to HMONNX...")
    print("=" * 60)

    try:
        from xhquant.api import convert_onnx_to_hmonnx, QuantScheme, create_quant_config, DeviceType

        quant_type = "w8a8h1_sefp"
        quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
        quant_config = create_quant_config(quant_scheme)

        # 创建输入张量
        input_tensors = []
        for name, shape in input_shapes_dict.items():
            if "emb_ids" in name:
                input_tensors.append(torch.zeros(*shape, dtype=torch.long))
            else:
                input_tensors.append(torch.randn(*shape, dtype=torch.float32))

        hmonnx_output = osp.join(hmonnx_file, output_name)

        convert_onnx_to_hmonnx(
            onnx_file,
            tuple(input_tensors),
            out_hmonnx_file=hmonnx_output,
            device_type="XH2A",
            quant_config=quant_config,
        )

        print(f"Saved HMONNX to {hmonnx_output}")
        return hmonnx_output

    except ImportError as e:
        print(f"Warning: xhquant not available, skipping HMONNX conversion: {e}")
        return None
    except Exception as e:
        print(f"Error converting to HMONNX: {e}")
        import traceback
        traceback.print_exc()
        return None


def export_action_out_proj(policy, output_dir, device="cpu"):
    """导出 action_out_proj 模块"""
    print("\n" + "="*60)
    print("Exporting action_out_proj...")
    print("="*60)
    
    action_out_proj = policy.action_out_proj
    action_out_proj.eval()
    action_out_proj = action_out_proj.float()
    
    # 输入形状: (batch, n_action_steps, proj_width)
    # proj_width = 1024
    n_action_steps = policy.config.get('n_action_steps', 50)
    proj_width = policy.config.get('proj_width', 1024)
    max_action_dim = policy.config.get('max_action_dim', 32)
    
    input_hidden = torch.randn(1, n_action_steps, proj_width, dtype=torch.float32)
    emb_ids = torch.tensor([0], dtype=torch.long)
    
    temp_onnx = osp.join(output_dir, "gigabrain_action_out_proj_temp.onnx")
    simplified_onnx = osp.join(output_dir, "gigabrain_action_out_proj.onnx")
    
    torch.onnx.export(
        action_out_proj,
        (input_hidden, emb_ids),
        temp_onnx,
        input_names=["hidden_states", "emb_ids"],
        output_names=["action"],
        opset_version=17,
        verbose=False,
    )
    
    onnx_model = onnx.load(temp_onnx)
    model_simplified, check = simplify(
        onnx_model,
        test_input_shapes={
            "hidden_states": [1, n_action_steps, proj_width],
            "emb_ids": [1],
        },
    )
    if check:
        onnx.save(model_simplified, simplified_onnx)
        print(f"Saved to {simplified_onnx}")
    else:
        onnx.save(onnx_model, simplified_onnx)
    
    os.remove(temp_onnx)
    
    # 验证
    with torch.no_grad():
        torch_out = action_out_proj(input_hidden, emb_ids)
    torch_out_np = torch_out.cpu().numpy()
    
    sess = ort.InferenceSession(simplified_onnx, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {
        "hidden_states": input_hidden.cpu().numpy(),
        "emb_ids": emb_ids.cpu().numpy(),
    })[0]

    print(f"Max diff: {np.max(np.abs(torch_out_np - onnx_out))}")
    print(f"Allclose: {np.allclose(torch_out_np, onnx_out, rtol=1e-3, atol=1e-4)}")
    print(f"Output shape: {torch_out.shape}")

    # 转换为 HMONNX
    hmonnx_output = convert_to_hmonnx(
        simplified_onnx,
        "action_out_proj.onnx",
        {"hidden_states": (1, n_action_steps, proj_width), "emb_ids": (1,)}
    )

    return simplified_onnx, hmonnx_output


def export_time_mlp(policy, output_dir, device="cpu"):
    """导出 time_mlp 模块 (时间步编码)"""
    print("\n" + "="*60)
    print("Exporting time_mlp...")
    print("="*60)
    
    time_mlp_in = policy.time_mlp_in
    time_mlp_out = policy.time_mlp_out
    proj_width = policy.config.get('proj_width', 1024)
    
    # 组合成单个模块
    class TimeMLP(nn.Module):
        def __init__(self, mlp_in, mlp_out):
            super().__init__()
            self.mlp_in = mlp_in
            self.mlp_out = mlp_out
        
        def forward(self, time_emb):
            # time_emb: (batch, proj_width)
            x = self.mlp_in(time_emb)
            x = torch.nn.functional.silu(x)
            x = self.mlp_out(x)
            x = torch.nn.functional.silu(x)
            return x
    
    time_mlp = TimeMLP(time_mlp_in, time_mlp_out)
    time_mlp.eval()
    time_mlp = time_mlp.float()
    
    # 输入: 时间步嵌入 (batch, proj_width)
    input_time_emb = torch.randn(1, proj_width, dtype=torch.float32)
    
    temp_onnx = osp.join(output_dir, "gigabrain_time_mlp_temp.onnx")
    simplified_onnx = osp.join(output_dir, "gigabrain_time_mlp.onnx")
    
    torch.onnx.export(
        time_mlp,
        input_time_emb,
        temp_onnx,
        input_names=["time_emb"],
        output_names=["adarms_cond"],
        opset_version=17,
        verbose=False,
    )
    
    onnx_model = onnx.load(temp_onnx)
    model_simplified, check = simplify(
        onnx_model,
        test_input_shapes={"time_emb": [1, proj_width]},
    )
    if check:
        onnx.save(model_simplified, simplified_onnx)
        print(f"Saved to {simplified_onnx}")
    else:
        onnx.save(onnx_model, simplified_onnx)
    
    os.remove(temp_onnx)
    
    # 验证
    with torch.no_grad():
        torch_out = time_mlp(input_time_emb)
    torch_out_np = torch_out.cpu().numpy()
    
    sess = ort.InferenceSession(simplified_onnx, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"time_emb": input_time_emb.cpu().numpy()})[0]

    print(f"Max diff: {np.max(np.abs(torch_out_np - onnx_out))}")
    print(f"Allclose: {np.allclose(torch_out_np, onnx_out, rtol=1e-3, atol=1e-4)}")
    print(f"Output shape: {torch_out.shape}")

    # 转换为 HMONNX
    hmonnx_output = convert_to_hmonnx(
        simplified_onnx,
        "time_mlp.onnx",
        {"time_emb": (1, proj_width)}
    )

    return simplified_onnx, hmonnx_output


def export_embodiment_embedding(policy, output_dir, device="cpu"):
    """导出机器人类型嵌入 (可选)"""
    print("\n" + "="*60)
    print("Exporting embodiment info...")
    print("="*60)
    
    # 打印 embodiment 相关信息
    num_embodiments = policy.config.get('num_embodiments', 3)
    print(f"Number of embodiments: {num_embodiments}")
    
    # 保存 embodiment 配置
    embodiment_config = {
        "num_embodiments": num_embodiments,
        "embodiments": {
            0: {"name": "AgileX Cobot Magic", "action_dim": 14},
            1: {"name": "Agibot G1", "action_dim": 20},
            2: {"name": "Other", "action_dim": 32},
        }
    }
    
    import json
    config_file = osp.join(output_dir, "embodiment_config.json")
    with open(config_file, "w") as f:
        json.dump(embodiment_config, f, indent=2)
    print(f"Saved embodiment config to {config_file}")
    
    return config_file


def export_all_other_modules(model_path, output_dir=None, device="cpu"):
    """导出所有其他模块"""
    set_seed(42)
    
    if output_dir is None:
        output_dir = workdir
    
    print(f"Loading GigaBrain model from {model_path}...")
    policy = load_gigabrain_model(model_path, device=device)
    
    # 导出各个模块
    exported_files = {}
    hmonnx_files = {}

    try:
        onnx_file, hmonnx_file = export_action_in_proj(policy, output_dir, device)
        exported_files['action_in_proj'] = onnx_file
        if hmonnx_file:
            hmonnx_files['action_in_proj'] = hmonnx_file
    except Exception as e:
        print(f"Error exporting action_in_proj: {e}")
        import traceback
        traceback.print_exc()

    try:
        onnx_file, hmonnx_file = export_action_out_proj(policy, output_dir, device)
        exported_files['action_out_proj'] = onnx_file
        if hmonnx_file:
            hmonnx_files['action_out_proj'] = hmonnx_file
    except Exception as e:
        print(f"Error exporting action_out_proj: {e}")
        import traceback
        traceback.print_exc()

    try:
        onnx_file, hmonnx_file = export_time_mlp(policy, output_dir, device)
        exported_files['time_mlp'] = onnx_file
        if hmonnx_file:
            hmonnx_files['time_mlp'] = hmonnx_file
    except Exception as e:
        print(f"Error exporting time_mlp: {e}")
        import traceback
        traceback.print_exc()

    try:
        exported_files['embodiment_config'] = export_embodiment_embedding(policy, output_dir, device)
    except Exception as e:
        print(f"Error exporting embodiment config: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*60)
    print("Export Summary")
    print("="*60)
    print("\nONNX Files:")
    for name, path in exported_files.items():
        print(f"  {name}: {path}")

    if hmonnx_files:
        print("\nHMONNX Files:")
        for name, path in hmonnx_files.items():
            print(f"  {name}: {path}")

    return exported_files, hmonnx_files


def main():
    parser = argparse.ArgumentParser(description="Export GigaBrain Other Modules")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data01/home/she.gao/.cache/huggingface/hub/models--open-gigaai--GigaBrain-0.1-3.5B-Base/snapshots/e705989fe052d53a8677db41d497a4f1ee519b66",
        help="GigaBrain 模型路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="推理设备",
    )
    
    args = parser.parse_args()
    export_all_other_modules(args.model_path, args.output_dir, args.device)


if __name__ == "__main__":
    main()
