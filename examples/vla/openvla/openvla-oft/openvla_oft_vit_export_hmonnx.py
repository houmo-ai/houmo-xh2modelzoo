import argparse
from pathlib import Path
import os
import os.path as osp
import logging
from typing import Dict

import torch
import torch.nn as nn
import onnx
import onnxslim

from safetensors.torch import load_file as load_safetensors_file
from transformers import AutoModelForVision2Seq, AutoProcessor

from xhquant.api import convert_onnx_to_hmonnx, HMONNXGoldenInference, QuantScheme, DeviceType, create_quant_config


class Vision(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_tower = model.vision_backbone
        self.multi_modal_projector = model.projector

    def forward(self, pixel_values):
        images = torch.split(pixel_values, [6] * 2, dim=1)
        # Process each image and collect patches
        all_patches = []
        for img in images:
            # Split each image further into two stacks of channels (each with 3 channels)
            img_regular, img_fused = torch.split(img, [3, 3], dim=1)

            # Get patches from both SigLIP and DINOv2 vision transformers
            patches = self.vision_tower.featurizer(img_regular)
            patches_fused = self.vision_tower.fused_featurizer(img_fused)

            # Concatenate SigLIP and DINOv2 patches along the hidden dimension
            combined_patches = torch.cat([patches, patches_fused], dim=2)
            all_patches.append(combined_patches)

        # Concatenate all patches along the patch dimension
        image_outputs = torch.cat(all_patches, dim=1)
        image_embedding = self.multi_modal_projector(image_outputs)
        return image_embedding


def load_quant_weight(quant_weight_path: str, native_hf_model: nn.Module) -> bool:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(handler)

    logger.info(f"Load checkpoint: {quant_weight_path}")

    if quant_weight_path.endswith(".safetensors"):
        state_dict = load_safetensors_file(quant_weight_path, device="cpu")
    else:
        state_dict = torch.load(quant_weight_path, map_location="cpu")

    model_state_dict = native_hf_model.state_dict()
    unexpected_keys = []

    for k in list(state_dict.keys()):
        if k not in model_state_dict:
            unexpected_keys.append(k)

    for k in unexpected_keys:
        v = state_dict[k]
        paths = k.split(".")
        if paths[-1] == "quant_weight":
            submodule_name = ".".join(paths[:-1])
            submodule = native_hf_model.get_submodule(submodule_name)

            if v.min().item() >= -128 and v.max().item() <= 127:
                v = v.to(torch.int8)
            elif v.min().item() >= -32768 and v.max().item() <= 32767:
                v = v.to(torch.int16)
            else:
                v = v.to(torch.float32)

            submodule.register_buffer("quant_weight", v, persistent=False)
            logger.info(f"register quant_weight -> {submodule_name}")
        else:
            logger.warning(f"ignore key: {k}")

        state_dict.pop(k)

    native_hf_model.load_state_dict(state_dict, strict=False)
    del state_dict
    return True


def export_openvla_vit(args):
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(args.model_path, trust_remote_code=True)
    
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    if args.quant_weight:
        load_quant_weight(args.quant_path, model)

    vision_projector = Vision(model).eval()

    Path(args.output_path).mkdir(parents=True, exist_ok=True)
    (Path(args.output_path) / "vision/onnx").mkdir(parents=True, exist_ok=True)
    (Path(args.output_path) / "vision/hmonnx").mkdir(parents=True, exist_ok=True)
    (Path(args.output_path) / "vision/golden").mkdir(parents=True, exist_ok=True)

    x = torch.randn(1, 12, 224, 224)

    temp_onnx = Path(args.output_path) / "vision/onnx/vision_ori.onnx"
    slim_onnx = Path(args.output_path) / "vision/onnx/vision_ori_slim.onnx"
    golden_dir = Path(args.output_path) / "vision/golden"
    hmonnx_path = Path(args.output_path) / "vision/hmonnx/vision.onnx"

    torch.onnx.export(
        vision_projector,
        (x,),
        str(temp_onnx),
        opset_version=18,
        export_params=True,
        do_constant_folding=True,
        input_names=["pixel_values"],
        output_names=["image_embedding"],
    )

    slim_model = onnxslim.slim(str(temp_onnx))
    onnx.save(
        slim_model,
        str(slim_onnx),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{slim_onnx.stem}_external_data",
    )

    convert_onnx_to_hmonnx(
        str(slim_onnx),
        (x,),
        out_hmonnx_file=str(hmonnx_path),
        device_type="XH2A",
        quant_config=quant_config,
    )

    vision_model = HMONNXGoldenInference(str(hmonnx_path))
    vision_model.save_golden = True
    vision_model.exec_device = torch.device("cuda:0")
    vision_model.golden_dir = str(golden_dir)

    input_args = (x.to(torch.float16),)

    with torch.no_grad():
        vision_model.forward(*input_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_type", type=str, default="w8a8_sefp")
    parser.add_argument("--output_path", type=str, default="work_dirs")
    parser.add_argument("--model_path", type=str, default="/data01/home/she.gao/.cache/huggingface/hub/models--moojink--openvla-7b-oft-finetuned-libero-spatial/snapshots/6d0231af0e48c5985f1ff86908f4674b84bc049b")
    parser.add_argument("--quant_weight", action="store_true")
    parser.add_argument("--quant_path", type=str, default="/data01/home/she.gao/xhquant_llm/examples/work_dirs/fa5ae1e7509348889295bba8e08621d8b55e9baf_quarot_gptq_transformers-4.53.3/quarot_gptq-state-dict.safetensors")
    
    args = parser.parse_args()
    
    export_openvla_vit(args)