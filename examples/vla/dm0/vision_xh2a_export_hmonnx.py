import onnx
import torch
import torch.nn as nn
from pathlib import Path
import os.path as osp
import os
from transformers import AutoTokenizer
from onnxsim import simplify
from dexbotic.model.dm0.dm0_arch import DM0ForCausalLM

from xhquant.api import (
    convert_onnx_to_hmonnx,
    QuantScheme,
    create_quant_config,
    DeviceType,
    HMONNXGoldenInference,
)


class VisionWithProjector(nn.Module):
    def __init__(self, vision_tower, projector):
        super().__init__()
        self.vision_tower = vision_tower
        self.projector = projector

    def forward(self, x):
        return self.projector(self.vision_tower(x))


def build_quant_config():
    quant_type = "w8a8_sefp"
    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a,
        quant_type=quant_type
    )
    return create_quant_config(quant_scheme)


def main():
    model_path = "/data01/home/she.gao/vla/dexbotic/checkpoints/DM0-libero"
    current_dir = Path(__file__).resolve().parent

    # 加载模型（保持 CPU，避免无用切换）
    model = DM0ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda")

    # 构建视觉模块
    vision_module = VisionWithProjector(
        model.model.mm_vision_tower,
        model.model.mm_projector,
    ).eval().cuda().float()

    # 导出 ONNX
    dummy = torch.randn(1, 3, 728, 728, dtype=torch.float32).cuda()
    onnx_path = current_dir / "vision_with_projector.onnx"

    torch.onnx.export(
        vision_module,
        dummy,
        str(onnx_path),
        input_names=["pixel_values"],
        output_names=["vision_embeds"],
        opset_version=17,
    )

    # 简化并保存
    model_onnx = onnx.load(str(onnx_path))
    model_simp, check = simplify(model_onnx)
    assert check, "Simplified ONNX model could not be validated"

    final_path = current_dir / "model_simplify.onnx"
    onnx.save_model(
        model_simp,
        str(final_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model_new.data",
    )

    print(f"Simplified ONNX saved to: {final_path}")
    
    output_file = osp.join(current_dir, "vision/vision_with_projector_hm.onnx")
    os.makedirs(osp.dirname(output_file), exist_ok=True)
    golden_dir = osp.join(osp.dirname(output_file), "step_0")
    os.makedirs(golden_dir, exist_ok=True)

    quant_config = build_quant_config()

    # convert（存在判断）
    if not osp.exists(output_file):
        convert_onnx_to_hmonnx(
            final_path,
            (dummy,),
            out_hmonnx_file=output_file,
            device_type="XH2A",
            quant_config=quant_config
        )

    # golden
    model = HMONNXGoldenInference(output_file)
    model.save_golden = True
    model.exec_device = torch.device("cuda:0")
    model.golden_dir = str(golden_dir)

    fp16_inputs = dummy.to(torch.float16)
    with torch.no_grad():
        model.forward(fp16_inputs)


if __name__ == "__main__":
    main()