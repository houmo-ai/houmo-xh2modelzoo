"""
GigaBrain-0.1 Vision Encoder 导出脚本
导出 SigLIP 视觉编码器和 MultiModal Projector 到 ONNX 和 HMONNX
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

# 设置环境变量
os.environ["ENABLE_LAYERNORM2RMSNORM"] = "1"

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
workdir = os.path.join(ROOT_DIR, "workdir")
hmonnx_file = os.path.join(ROOT_DIR, "hmonnx")
os.makedirs(workdir, exist_ok=True)
os.makedirs(hmonnx_file, exist_ok=True)


def set_seed(seed=42):
    """固定随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------
# SigLIP Vision Encoder 包装类
# ---------------------------------------------------------------
class Siglip(nn.Module):
    """SigLIP Vision Encoder + Projector 包装类"""
    def __init__(self, model):
        super().__init__()
        self.vision_tower = model.vision_tower.eval()
        self.multi_modal_projector = model.multi_modal_projector.eval()

    def forward(self, pixel_values):
        image_outputs = self.vision_tower(pixel_values)
        selected_image_feature = image_outputs.last_hidden_state
        image_features = self.multi_modal_projector(selected_image_feature)
        return image_features


# ---------------------------------------------------------------
# 加载 GigaBrain 模型
# ---------------------------------------------------------------
def load_gigabrain_model(model_path, device="cpu"):
    """加载 GigaBrain 模型
    
    GigaBrain 使用自定义的 GigaBrain0Policy 类，继承自 diffusers 的 ModelMixin。
    参考: giga_models/pipelines/vla/giga_brain_0/pipeline_giga_brain_0.py
    """
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


# ---------------------------------------------------------------
# 导出 Vision Encoder
# ---------------------------------------------------------------
def export_vision_encoder(policy, output_dir, device="cpu"):
    """导出 Vision Encoder 到 ONNX"""
    print("\n" + "=" * 60)
    print("Exporting Vision Encoder...")
    print("=" * 60)
    
    # 获取 vision encoder 组件
    if hasattr(policy, 'paligemma_with_expert'):
        vision_tower = policy.paligemma_with_expert.vision_tower
        projector = policy.paligemma_with_expert.multi_modal_projector
    elif hasattr(policy, 'vision_tower'):
        vision_tower = policy.vision_tower
        projector = policy.multi_modal_projector
    else:
        vision_tower = policy.model.paligemma_with_expert.vision_tower
        projector = policy.model.paligemma_with_expert.multi_modal_projector
    
    # 创建自定义 SigLIP Vision Encoder 包装类（避免 autocast 和 bfloat16 转换）
    class SiglipVisionEncoderFloat32(nn.Module):
        """自定义 SigLIP Vision Encoder，强制使用 float32"""
        def __init__(self, vision_tower, projector):
            super().__init__()
            # 提取 encoder 组件
            self.embeddings = vision_tower.embeddings.float()
            self.encoder = vision_tower.encoder.float()
            self.post_layernorm = vision_tower.post_layernorm.float()
            self.multi_modal_projector = projector.float()
        
        def forward(self, pixel_values):
            # embeddings
            hidden_states = self.embeddings(pixel_values)
            
            # encoder (no autocast, no bfloat16 conversion)
            encoder_outputs = self.encoder(
                inputs_embeds=hidden_states,
                output_attentions=False,
                output_hidden_states=False,
            )
            last_hidden_state = encoder_outputs.last_hidden_state
            
            # post layernorm
            last_hidden_state = self.post_layernorm(last_hidden_state)
            
            # project to LLM hidden size
            image_features = self.multi_modal_projector(last_hidden_state)
            return image_features
    
    siglip_model = SiglipVisionEncoderFloat32(vision_tower, projector)
    siglip_model.eval()
    
    # 创建测试输入
    vision_in_channels = 3
    if hasattr(vision_tower, 'config') and hasattr(vision_tower.config, 'num_channels'):
        vision_in_channels = vision_tower.config.num_channels
    
    input_features = torch.randn(1, vision_in_channels, 224, 224, dtype=torch.float32)
    
    # 导出 ONNX
    temp_onnx_file = osp.join(output_dir, "gigabrain_vision_temp.onnx")
    simplified_onnx_file = osp.join(output_dir, "gigabrain_vision.onnx")
    
    print(f"Exporting to {temp_onnx_file}...")
    torch.onnx.export(
        siglip_model,
        input_features,
        temp_onnx_file,
        input_names=["pixel_values"],
        output_names=["image_features"],
        opset_version=17,
        verbose=False,
    )
    
    # 简化 ONNX
    print(f"Simplifying ONNX to {simplified_onnx_file}...")
    onnx_model = onnx.load(temp_onnx_file)
    model_simplified, check = simplify(
        onnx_model,
        test_input_shapes={"pixel_values": [1, vision_in_channels, 224, 224]},
    )
    
    if check:
        onnx.save(model_simplified, simplified_onnx_file)
        print(f"Saved simplified ONNX to {simplified_onnx_file}")
    else:
        onnx.save(onnx_model, simplified_onnx_file)
        print("Simplify check failed, saved original ONNX")
    
    if os.path.exists(temp_onnx_file):
        os.remove(temp_onnx_file)
    
    # 验证输出一致性
    print("\nValidating output consistency...")
    with torch.no_grad():
        torch_out = siglip_model(input_features)
        if isinstance(torch_out, (tuple, list)):
            torch_out = torch_out[0]
    torch_out_np = torch_out.float().cpu().numpy()
    
    sess = ort.InferenceSession(simplified_onnx_file, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"pixel_values": input_features.cpu().numpy()})[0]
    
    max_diff = np.max(np.abs(torch_out_np - onnx_out))
    mean_diff = np.mean(np.abs(torch_out_np - onnx_out))
    allclose = np.allclose(torch_out_np, onnx_out, rtol=1e-3, atol=1e-4)
    
    print(f"Max absolute difference: {max_diff}")
    print(f"Mean absolute difference: {mean_diff}")
    print(f"Allclose: {allclose}")
    print(f"Output shape: {torch_out.shape}")
    
    return simplified_onnx_file


# ---------------------------------------------------------------
# 转换为 HMONNX
# ---------------------------------------------------------------
def convert_to_hmonnx(onnx_file, output_dir, input_shape=(1, 3, 224, 224)):
    """转换 ONNX 到 HMONNX 格式"""
    print("\n" + "=" * 60)
    print("Converting to HMONNX...")
    print("=" * 60)
    
    try:
        from xhquant.api import convert_onnx_to_hmonnx, QuantScheme, create_quant_config, DeviceType
        
        quant_type = "w8a8h1_sefp"
        quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
        quant_config = create_quant_config(quant_scheme)
        
        input_tensor = torch.randn(*input_shape, dtype=torch.float32)
        hmonnx_output = osp.join(output_dir, "vision.onnx")
        
        convert_onnx_to_hmonnx(
            onnx_file,
            (input_tensor,),
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
        return None


# ---------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------
def run_gigabrain_vision_export(args):
    """执行完整的 Vision Encoder 导出流程"""
    set_seed(42)
    
    device = args.device
    if torch.cuda.is_available() and device == "cuda":
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # 加载模型
    try:
        policy = load_gigabrain_model(args.model_path, device=device)
        import ipdb;ipdb.set_trace()
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("\nPlease ensure the model is downloaded and diffusers is installed:")
        print("  pip install diffusers>=0.34.0")
        print(f"  Model path: {args.model_path}")
        return None
    
    # 导出 Vision Encoder
    onnx_file = export_vision_encoder(policy, workdir, device=device)
    
    # 转换为 HMONNX
    hmonnx_output = convert_to_hmonnx(onnx_file, hmonnx_file)
    
    print("\n" + "=" * 60)
    print("Export Summary")
    print("=" * 60)
    print(f"ONNX file: {onnx_file}")
    if hmonnx_output:
        print(f"HMONNX file: {hmonnx_output}")
    print("Done!")
    
    return onnx_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GigaBrain Vision Encoder")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data01/home/she.gao/.cache/huggingface/hub/models--open-gigaai--GigaBrain-0.1-3.5B-Base/snapshots/e705989fe052d53a8677db41d497a4f1ee519b66",
        help="GigaBrain 模型路径",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="推理设备",
    )
    
    args = parser.parse_args()
    run_gigabrain_vision_export(args)
