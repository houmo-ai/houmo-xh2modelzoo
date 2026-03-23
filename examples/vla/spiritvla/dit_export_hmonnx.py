import os
import argparse
import torch
import os.path as osp
from pathlib import Path

from model.modeling_spirit_vla import SpiritVLAPolicy
from xhquant.api import (
    convert_onnx_to_hmonnx,
    QuantScheme,
    create_quant_config,
    DeviceType,
    HMONNXGoldenInference,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data01/home/she.gao/.cache/huggingface/hub/models--Spirit-AI-robotics--Spirit-v1.5/snapshots/9886fa7552de70b097c4214b7e40e66ff937a6fd",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="w8a8_sefp",
    )
    parser.add_argument("--output_dir", type=str, default="work_dirs/dit")
    parser.add_argument("--onnx_name", type=str, default="dit.onnx")
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()

    # 路径
    onnx_dir = osp.join("workdirs", "dit")
    os.makedirs(onnx_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    golden_dir = Path(args.output_dir) / "golden"
    golden_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = osp.join(onnx_dir, "basedit.onnx")
    hmonnx_path = osp.join(args.output_dir, args.onnx_name)

    # 加载模型
    model = SpiritVLAPolicy.from_pretrained(args.model_path).dit.eval().cpu()

    # dummy 输入
    hidden_states = torch.randn(1, 61, 1536)
    encoder_hidden_states = torch.randn(1, 300, 2560)
    timestep = torch.randint(0, 1000, (1,), dtype=torch.int32)
    encoder_attention_mask = torch.ones(1, 300, dtype=torch.float32)

    # 导出 ONNX
    torch.onnx.export(
        model,
        (hidden_states, encoder_hidden_states, timestep, encoder_attention_mask),
        onnx_path,
        opset_version=17,
        do_constant_folding=True,
        input_names=[
            "hidden_states",
            "encoder_hidden_states",
            "timestep",
            "encoder_attention_mask",
        ],
        output_names=["output"],
    )

    print(f"export done: {onnx_path}")

    # 量化输入
    hidden_states = hidden_states.to(torch.float16)
    encoder_hidden_states = encoder_hidden_states.to(torch.float16)
    encoder_attention_mask = encoder_attention_mask.to(torch.float16)

    # quant config
    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a,
        quant_type=args.quant_type,
    )
    quant_config = create_quant_config(quant_scheme)

    # 转换 hmonnx
    convert_onnx_to_hmonnx(
        onnx_path,
        (hidden_states, encoder_hidden_states, timestep, encoder_attention_mask),
        out_hmonnx_file=hmonnx_path,
        device_type="XH2A",
        quant_config=quant_config,
    )

    # golden
    vision_model = HMONNXGoldenInference(hmonnx_path)
    vision_model.save_golden = True
    vision_model.exec_device = torch.device(args.device)
    vision_model.golden_dir = str(golden_dir)

    with torch.no_grad():
        vision_model.forward(
            hidden_states,
            encoder_hidden_states,
            timestep,
            encoder_attention_mask,
        )


if __name__ == "__main__":
    main()