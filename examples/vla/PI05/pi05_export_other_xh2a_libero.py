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
import numpy as np
from lerobot.policies.pi05 import PI05Config, PI05Policy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors

from xhquant.api import convert_onnx_to_hmonnx, QuantScheme, create_quant_config, DeviceType

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
workdir = os.path.join(ROOT_DIR, "workdir")
hmonnx_file = os.path.join(ROOT_DIR, "hmonnx")
os.makedirs(workdir, exist_ok=True)
os.makedirs(hmonnx_file, exist_ok=True)

DEVICE = "cpu"

def load_pi05(model_path):
    policy = PI05Policy.from_pretrained(model_path, strict=True)
    policy.to(DEVICE)
    policy.config.device = DEVICE

    return policy

# ---------------------------------------------------------------
# ③ PI0.5 推理（与官方 test 中 LeRobot 路径一致）
# ---------------------------------------------------------------
def run_pi05_inference(args):
    policy = load_pi05(args.model_path)
    policy.eval()
    
    quant_type = "w8a8h1_sefp"
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    # 导出action_in_proj模块
    # 导出action_in_proj
    action_in_proj = policy.model.action_in_proj  # 假设模块路径正确，需根据实际模型结构调整
    action_in_proj.eval()
    action_in_proj = action_in_proj.float()
    temp_onnx_action_in = "./workdir/pi0.5_action_in_proj_libero.onnx"
    simplified_onnx_action_in = "./workdir/pi0.5_action_in_proj_libero_simplified.onnx"
    input_action_in = torch.randn(1, 50, policy.config.max_action_dim, dtype=torch.float32)  # 输入形状匹配max_action_dim

    # 导出ONNX
    torch.onnx.export(
        action_in_proj,
        input_action_in,
        temp_onnx_action_in,
        input_names=["action_in"],
        output_names=["action_in_proj_out"],
        opset_version=17,
        verbose=True,
    )
    # 简化ONNX
    onnx_model_action_in = onnx.load(temp_onnx_action_in)
    model_simplified_action_in, check_action_in = simplify(
        onnx_model_action_in,
        test_input_shapes={"action_in": [1, 50, policy.config.max_action_dim]},
    )
    if check_action_in:
        onnx.save(model_simplified_action_in, simplified_onnx_action_in)
        print("Simplified action_in_proj:", simplified_onnx_action_in)
    else:
        print("Simplify action_in_proj failed")

    # 验证输出一致性
    with torch.no_grad():
        torch_out_action_in = action_in_proj(input_action_in)
    torch_out_np_action_in = torch_out_action_in.float().cpu().numpy()
    sess_action_in = ort.InferenceSession(simplified_onnx_action_in, providers=["CUDAExecutionProvider"])
    onnx_out_action_in = sess_action_in.run(None, {"action_in": input_action_in.cpu().numpy()})
    onnx_out_np_action_in = onnx_out_action_in[0]
    print("action_in_proj max abs diff:", np.max(np.abs(torch_out_np_action_in - onnx_out_np_action_in)))
    print("action_in_proj mean abs diff:", np.mean(np.abs(torch_out_np_action_in - onnx_out_np_action_in)))
    print("action_in_proj allclose:", np.allclose(torch_out_np_action_in, onnx_out_np_action_in, rtol=1e-3, atol=1e-4))

    # 转换为hmonnx
    convert_onnx_to_hmonnx(simplified_onnx_action_in, (input_action_in,), 
                          out_hmonnx_file=osp.join(hmonnx_file, "action_in_proj.onnx"), 
                          device_type="XH2A", quant_config=quant_config)

    #新增：导出action_out_proj模块
    # 导出action_out_proj
    action_out_proj = policy.model.action_out_proj  # 假设模块路径正确，需根据实际模型结构调整
    action_out_proj.eval()
    action_out_proj = action_out_proj.float()
    temp_onnx_action_out = "./workdir/pi0.5_action_out_proj_libero.onnx"
    simplified_onnx_action_out = "./workdir/pi0.5_action_out_proj_libero_simplified.onnx"
    # 输入形状匹配action_expert_config.width（假设从policy.config获取）
    input_action_out = torch.randn(1, 50, 1024, dtype=torch.float32)

    torch.onnx.export(
        action_out_proj,
        input_action_out,
        temp_onnx_action_out,
        input_names=["action_out"],
        output_names=["action_out_proj_out"],
        opset_version=17,
        verbose=True,
    )
    onnx_model_action_out = onnx.load(temp_onnx_action_out)
    model_simplified_action_out, check_action_out = simplify(
        onnx_model_action_out,
        test_input_shapes={"action_out": [1, 50, 1024]},
    )
    if check_action_out:
        onnx.save(model_simplified_action_out, simplified_onnx_action_out)
        print("Simplified action_out_proj:", simplified_onnx_action_out)
    else:
        print("Simplify action_out_proj failed")

    # 验证输出一致性
    with torch.no_grad():
        torch_out_action_out = action_out_proj(input_action_out)
    torch_out_np_action_out = torch_out_action_out.float().cpu().numpy()
    sess_action_out = ort.InferenceSession(simplified_onnx_action_out, providers=["CUDAExecutionProvider"])
    onnx_out_action_out = sess_action_out.run(None, {"action_out": input_action_out.cpu().numpy()})
    onnx_out_np_action_out = onnx_out_action_out[0]
    print("action_out_proj max abs diff:", np.max(np.abs(torch_out_np_action_out - onnx_out_np_action_out)))
    print("action_out_proj mean abs diff:", np.mean(np.abs(torch_out_np_action_out - onnx_out_np_action_out)))
    print("action_out_proj allclose:", np.allclose(torch_out_np_action_out, onnx_out_np_action_out, rtol=1e-3, atol=1e-4))

    # 转换为hmonnx
    convert_onnx_to_hmonnx(simplified_onnx_action_out, (input_action_out,), 
                          out_hmonnx_file=osp.join(hmonnx_file, "action_out_proj.onnx"), 
                          device_type="XH2A", quant_config=quant_config)

    # 新增：导出time_mlp模块
    # 导出time_mlp（包含time_mlp_in和time_mlp_out）
    from torch.nn import functional as F  # 导入F用于silu激活函数

    class TimeMLPWrapper(nn.Module):
        def __init__(self, time_mlp_in, time_mlp_out):
            super().__init__()
            self.time_mlp_in = time_mlp_in
            self.time_mlp_out = time_mlp_out

        def forward(self, time_emb):
            x = self.time_mlp_in(time_emb)
            x = F.silu(x)
            x = self.time_mlp_out(x)
            return F.silu(x)

    time_mlp = TimeMLPWrapper(policy.model.time_mlp_in, policy.model.time_mlp_out)  # 假设模块路径正确
    time_mlp.eval()
    time_mlp = time_mlp.float()
    temp_onnx_time_mlp = "./workdir/pi0.5_time_mlp_libero.onnx"
    simplified_onnx_time_mlp = "./workdir/pi0.5_time_mlp_libero_simplified.onnx"
    input_time_mlp = torch.randn(1, 1024, dtype=torch.float32)  # 输入形状匹配action_expert_config.width

    torch.onnx.export(
        time_mlp,
        input_time_mlp,
        temp_onnx_time_mlp,
        input_names=["time_emb"],
        output_names=["time_mlp_out"],
        opset_version=17,
        verbose=True,
    )
    onnx_model_time_mlp = onnx.load(temp_onnx_time_mlp)
    model_simplified_time_mlp, check_time_mlp = simplify(
        onnx_model_time_mlp,
        test_input_shapes={"time_emb": [1, 1024]},
    )
    if check_time_mlp:
        onnx.save(model_simplified_time_mlp, simplified_onnx_time_mlp)
        print("Simplified time_mlp:", simplified_onnx_time_mlp)
    else:
        print("Simplify time_mlp failed")

    # 验证输出一致性
    with torch.no_grad():
        torch_out_time_mlp = time_mlp(input_time_mlp)
    torch_out_np_time_mlp = torch_out_time_mlp.float().cpu().numpy()
    sess_time_mlp = ort.InferenceSession(simplified_onnx_time_mlp, providers=["CUDAExecutionProvider"])
    onnx_out_time_mlp = sess_time_mlp.run(None, {"time_emb": input_time_mlp.cpu().numpy()})
    onnx_out_np_time_mlp = onnx_out_time_mlp[0]
    print("time_mlp max abs diff:", np.max(np.abs(torch_out_np_time_mlp - onnx_out_np_time_mlp)))
    print("time_mlp mean abs diff:", np.mean(np.abs(torch_out_np_time_mlp - onnx_out_np_time_mlp)))
    print("time_mlp allclose:", np.allclose(torch_out_np_time_mlp, onnx_out_np_time_mlp, rtol=1e-3, atol=1e-4))

    # 转换为hmonnx
    convert_onnx_to_hmonnx(simplified_onnx_time_mlp, (input_time_mlp,), 
                          out_hmonnx_file=osp.join(hmonnx_file, "time_mlp.onnx"), 
                          device_type="XH2A", quant_config=quant_config)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/data01/home/she.gao/.cache/huggingface/hub/models--lerobot--pi05_libero_finetuned/snapshots/d8419fc249cbb1f29b0c528f05c0d2fe50f46855")
    args = parser.parse_args()
    run_pi05_inference(args)
