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

# --- LeRobot 核心模块导入 ---
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.xvla.modeling_xvla import XVLAPolicy

# --- 量化工具导入 ---
from xhquant.api import convert_onnx_to_hmonnx, QuantScheme, create_quant_config, DeviceType

# --- 路径设置 ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
workdir = os.path.join(ROOT_DIR, "xvla_vision_onnx")
hmonnx_dir = os.path.join(ROOT_DIR, "xvla_vision_hmonnx")
os.makedirs(workdir, exist_ok=True)
os.makedirs(hmonnx_dir, exist_ok=True)


class XVLAVisionPart(nn.Module):
    """
    针对该特定 XVLA 架构定制的 Vision 导出 Wrapper。
    完全复刻 _encode_image 的逻辑。
    """
    def __init__(self, policy):
        super().__init__()
        # 1. 绑定 Vision Tower
        # 注意：这里假设 policy.model.vision_tower 是存在的
        self.vision_tower = policy.model.vlm.vision_tower
        
        # 2. 绑定参数和配置 (从 policy.model 中提取)
        model = policy.model.vlm
        
        # 提取 Projector 权重 (就是那个矩阵)
        # 注意：self.image_projection 可能是一个 Parameter，不是 Layer
        self.register_parameter("image_projection", model.image_projection)
        
        # 提取 Norm 层
        self.image_proj_norm = model.image_proj_norm
        
        # 提取 Positional Embeddings (如果有)
        self.image_pos_embed = getattr(model, "image_pos_embed", None)
        self.visual_temporal_embed = getattr(model, "visual_temporal_embed", None)
        
        # 提取配置列表 (这一步很重要，决定了怎么拼接特征)
        self.image_feature_source = model.image_feature_source
        
        # 确保全部 eval
        self.vision_tower.eval()
        self.image_proj_norm.eval()

    def forward(self, pixel_values):
        # === 以下代码逻辑 99% 复刻自你提供的 _encode_image ===
        
        # 1. 强制类型转换 (ONNX 导出时通常不需要，但保留以防万一)
        # pixel_values = pixel_values.to(dtype=self.image_projection.dtype) 
        
        # 2. Vision Tower Forward
        if len(pixel_values.shape) == 4:
            batch_size, channels, height, width = pixel_values.shape
            num_frames = 1
            # 注意：这里调用的是 forward_features_unpool
            x = self.vision_tower.forward_features_unpool(pixel_values)
        else:
            # 导出时通常假设输入是合法的 (B, C, H, W)
            # 为了 ONNX 静态图，这里简化处理，假设就是 4 维
            batch_size, channels, height, width = pixel_values.shape
            num_frames = 1
            x = self.vision_tower.forward_features_unpool(pixel_values)

        # 3. Positional Embedding 处理
        if self.image_pos_embed is not None:
            # Reshape 逻辑
            x = x.view(batch_size * num_frames, -1, x.shape[-1])
            num_tokens = x.shape[-2]
            
            # 计算 h, w (导出时 num_tokens 固定，sqrt 安全)
            side = int(num_tokens**0.5)
            
            x = x.view(batch_size * num_frames, side, side, x.shape[-1])
            pos_embed = self.image_pos_embed(x)
            x = x + pos_embed
            x = x.view(batch_size, num_frames * side * side, x.shape[-1])

        # 4. Temporal Embedding (如果 num_frames=1，这部分逻辑可能被跳过或简化)
        if self.visual_temporal_embed is not None:
            # 假设 num_frames=1，简化逻辑以利于导出
            # pass 
            visual_temporal_embed = self.visual_temporal_embed(
                x.view(batch_size, num_frames, -1, x.shape[-1])[:, :, 0]
            )
            x = x.view(batch_size, num_frames, -1, x.shape[-1]) + visual_temporal_embed.view(
                1, num_frames, 1, x.shape[-1]
            )

        # 5. 特征提取与池化
        x_feat_dict = {}

        # Spatial Avg Pool
        # view: (B, T, L, C) -> mean(2) -> (B, T, C)
        spatial_avg_pool_x = x.view(batch_size, num_frames, -1, x.shape[-1]).mean(dim=2)
        x_feat_dict["spatial_avg_pool"] = spatial_avg_pool_x

        # Temporal Avg Pool
        # mean(1) -> (B, L, C)
        temporal_avg_pool_x = x.view(batch_size, num_frames, -1, x.shape[-1]).mean(dim=1)
        x_feat_dict["temporal_avg_pool"] = temporal_avg_pool_x

        # Last Frame
        # [:, -1]
        last_frame_x = x.view(batch_size, num_frames, -1, x.shape[-1])[:, -1]
        x_feat_dict["last_frame"] = last_frame_x

        # 6. 特征选择与拼接
        new_x = []
        for src in self.image_feature_source:
            # 这里必须用 if-else 显式展开，因为 ONNX 不支持动态字典 key 查询
            if src == "spatial_avg_pool":
                new_x.append(x_feat_dict["spatial_avg_pool"])
            elif src == "temporal_avg_pool":
                new_x.append(x_feat_dict["temporal_avg_pool"])
            elif src == "last_frame":
                new_x.append(x_feat_dict["last_frame"])
            
        x = torch.cat(new_x, dim=1)

        # 7. Projector (矩阵乘法)
        # 注意：原代码是 x @ self.image_projection
        x = x @ self.image_projection

        # 8. Norm
        x = self.image_proj_norm(x)

        return x

# ---------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

set_seed(42)
DEVICE = "cpu"

# ---------------------------------------------------------------
# 主导出逻辑
# ---------------------------------------------------------------
def run_xvla_export(args):
    print(f"Loading XVLA Policy from: {args.model_path}")

    # 1. 加载 Config (LeRobot 标准方式)
    config = PreTrainedConfig.from_pretrained(args.model_path)
    
    # 2. 加载 Policy
    # strict=False 允许加载时忽略一些非必要的 key (如优化器状态)
    policy = XVLAPolicy.from_pretrained(args.model_path, config=config, strict=False)
    policy.to(DEVICE)
    policy.eval()

    # 3. (可选) 验证处理器 - 仿照 pi0.5 验证流程
    # 这一步不是导出必须的，但可以用来确认你的 dataset_stats 配置是否正确
    # dummy_stats = { ... } # 如果需要构建 dummy stats 可以在这里构建
    # preprocessor, postprocessor = make_pre_post_processors(config=policy.config, dataset_stats=dummy_stats)

    # 4. 导出 Vision 部分
    if True:
        print(">>> 准备导出 XVLA Vision (DaVIT + Projector) ...")

        # 实例化 Wrapper
        vision_model = XVLAVisionPart(policy)
        vision_model.eval()
        vision_model = vision_model.half() # 转为 float16

        # 获取图像尺寸 (从 config 中读取，如果没有则默认 224)
        # 很多 DaVIT 使用 224，但 VLM 常用 336 或 384，请务必检查
        image_size = getattr(config, "image_size", 224) 
        if hasattr(config, "vision_config"):
             image_size = getattr(config.vision_config, "image_size", image_size)
        
        print(f"Using Image Size: {image_size}x{image_size}")

        # 定义文件名
        temp_onnx_file = os.path.join(workdir, "xvla_vision.onnx")
        simplified_onnx_file = os.path.join(workdir, "xvla_vision_simplified.onnx")
        
        # 构造 Dummy Input
        input_features = torch.randn(1, 3, image_size, image_size, dtype=torch.float16, device=DEVICE)

        # === ONNX Export ===
        print("Exporting to ONNX...")
        torch.onnx.export(
            vision_model,
            input_features,
            temp_onnx_file,
            input_names=["pixel_values"],
            output_names=["vision_embeddings"],
            opset_version=17, # DaVIT 结构复杂，建议使用高版本 opset
            verbose=False,
            do_constant_folding=True
        )

        # === ONNX Simplify ===
        print("Simplifying ONNX...")
        onnx_model = onnx.load(temp_onnx_file)
        model_simplified, check = simplify(
            onnx_model,
            test_input_shapes={"pixel_values": [1, 3, image_size, image_size]},
        )

        if check:
            onnx.save(model_simplified, simplified_onnx_file)
            print(f"Simplified model saved: {simplified_onnx_file}")
        else:
            print("Simplify check failed! Saving original simplified model anyway.")
            onnx.save(model_simplified, simplified_onnx_file)

        # # === 精度验证 (PyTorch vs ONNX Runtime) ===
        # print("Validating Accuracy...")
        # import onnxruntime as ort
        
        # # PyTorch Output
        # with torch.no_grad():
        #     torch_out = vision_model(input_features)
        # torch_out_np = torch_out.cpu().numpy()

        # # ONNX Runtime Output
        # sess = ort.InferenceSession(simplified_onnx_file, providers=["CPUExecutionProvider"])
        # onnx_out = sess.run(None, {"pixel_values": input_features.cpu().numpy()})
        # onnx_out_np = onnx_out[0]

        # # 比较
        # max_diff = np.max(np.abs(torch_out_np - onnx_out_np))
        # mean_diff = np.mean(np.abs(torch_out_np - onnx_out_np))
        # print(f"Max Diff: {max_diff}")
        # print(f"Mean Diff: {mean_diff}")
        
        # if max_diff > 1e-3:
        #     print("[Warning] Max diff is relatively high. Check if float16/bfloat16 mismatch occurred.")
        # else:
        #     print("[Pass] Accuracy verification passed.")

        # === 转换为 HMONNX XH2A ===
        print("Converting to HMONNX for Houmo XH2A...")
        
        # 配置量化参数
        quant_type = "w8a8h1_sefp" 
        quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
        quant_config = create_quant_config(quant_scheme)
        
        convert_onnx_to_hmonnx(
            simplified_onnx_file, 
            (input_features,), 
            out_hmonnx_file=osp.join(hmonnx_dir, "xvla_vision_davit_xh2.onnx"), 
            device_type="XH2A", 
            quant_config=quant_config
        )
        print(f"HMONNX Export Complete: {osp.join(hmonnx_dir, 'xvla_vision_davit_xh2.onnx')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 默认路径指向你的 huggingface cache 或本地 checkpoint 目录
    parser.add_argument("--model_path", type=str, default="/data02/datasets/xvla-libero")
    args = parser.parse_args()
    
    run_xvla_export(args)