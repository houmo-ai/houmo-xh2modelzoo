import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoImageProcessor, AutoModel
from transformers.image_utils import load_image
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    xhquant_init,
)


class DinoForOnnx(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor):
        outputs = self.model(pixel_values=pixel_values, return_dict=False)
        if isinstance(outputs, tuple):
            if len(outputs) < 2 or outputs[1] is None:
                raise RuntimeError("Unexpected model outputs: pooler_output is missing.")
            return outputs[0], outputs[1]
        if outputs.pooler_output is None:
            raise RuntimeError("Unexpected model outputs: pooler_output is missing.")
        return outputs.last_hidden_state, outputs.pooler_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="weights/dinov3-vitb16-pretrain-lvd1689m",
        help="Local Hugging Face model dir for DINOv3.",
    )
    parser.add_argument(
        "--image-url",
        default="data/images/houmo_logo.jpg",
        help="Input image URL/path used to build export sample input.",
    )
    parser.add_argument(
        "--onnx-path",
        default="work_dirs/dinov3_lora/dinov3_lora.onnx",
        help="Output ONNX path.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--r",
        type=int,
        default=8,
        help="LoRA rank.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Export device.",
    )
    parser.add_argument(
        "--quant-type",
        default="w8a8h1_sefp",
        help="xhquant quant type used for HMONNX conversion.",
    )
    parser.add_argument(
        "--hmonnx-path",
        default="",
        help="Output HMONNX path. If empty, auto-generated from onnx path.",
    )
    parser.add_argument(
        "--golden-dir",
        default="",
        help="Golden output directory. If empty, auto-generated from hmonnx path.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable xhquant debug mode.")
    parser.add_argument("--skip-hmonnx", action="store_true", help="Skip HMONNX conversion and golden generation.")
    parser.add_argument("--skip-golden", action="store_true", help="Skip golden generation.")
    return parser.parse_args()


def pick_dinov3_qk_target_modules(model: torch.nn.Module) -> list[str]:
    names = [name for name, _ in model.named_modules()]
    has_q_proj = any(name.endswith("q_proj") for name in names)
    has_k_proj = any(name.endswith("k_proj") for name in names)
    if has_q_proj and has_k_proj:
        return ["q_proj", "k_proj"]

    raise RuntimeError(
        "Cannot find dinov3 q_proj/k_proj modules. "
        "Please check --model-id is a dinov3_vit checkpoint."
    )


def main():
    args = parse_args()
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")

    processor = AutoImageProcessor.from_pretrained(args.model_id, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model_id,
        local_files_only=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()

    target_modules = pick_dinov3_qk_target_modules(model)
    peft_config = LoraConfig(
        r=args.r,
        lora_alpha=args.r * 2,
        lora_dropout=0.0,
        bias="none",
        target_modules=target_modules,
        task_type=TaskType.FEATURE_EXTRACTION,
    )
    model = get_peft_model(model, peft_config)
    model.eval()
    model.print_trainable_parameters()

    image = load_image(args.image_url)
    pixel_values = processor(images=image, return_tensors="pt")["pixel_values"]
    if pixel_values.shape[0] != 1:
        pixel_values = pixel_values[:1]
    if pixel_values.shape[-2:] != (224, 224):
        pixel_values = F.interpolate(pixel_values, size=(224, 224), mode="bilinear", align_corners=False)
    pixel_values = pixel_values.to(device)

    export_model = DinoForOnnx(model).to(device)
    export_model.eval()

    out_path = Path(args.onnx_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        export_model,
        (pixel_values,),
        str(out_path),
        export_params=True,
        do_constant_folding=True,
        opset_version=args.opset,
        input_names=["pixel_values"],
        output_names=["last_hidden_state", "pooler_output"],
    )
    print(f"TaskType: {TaskType.FEATURE_EXTRACTION.value}")
    print(f"LoRA target modules: {target_modules}")
    print(f"LoRA rank (r): {args.r}")
    print(f"Export input shape fixed to: {tuple(pixel_values.shape)}")
    print(f"ONNX exported to: {out_path}")

    if args.skip_hmonnx:
        return

    xhquant_init(None, debug=args.debug)
    if args.hmonnx_path:
        out_hmonnx_path = Path(args.hmonnx_path)
    else:
        out_hmonnx_path = out_path.parent / "hmonnx" / f"{out_path.stem}_{DeviceType.XH2a}.onnx"
    out_hmonnx_path.parent.mkdir(parents=True, exist_ok=True)

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)
    convert_input = pixel_values.detach().to("cpu", dtype=torch.float32)
    convert_onnx_to_hmonnx(
        str(out_path),
        [convert_input],
        DeviceType.XH2a,
        str(out_hmonnx_path),
        quant_config=quant_config,
        input_names=["pixel_values"],
        output_names=["last_hidden_state", "pooler_output"],
    )
    print(f"HMONNX exported to: {out_hmonnx_path}")

    if args.skip_golden:
        return

    if args.golden_dir:
        golden_dir = Path(args.golden_dir)
    else:
        golden_dir = out_hmonnx_path.parent / f"golden_{args.quant_type}"
    golden_dir.mkdir(parents=True, exist_ok=True)

    session = HMONNXGoldenInference(str(out_hmonnx_path))
    session.to(str(device))
    session.save_golden = True
    session.golden_dir = golden_dir

    golden_input = pixel_values.detach().to(device=device, dtype=torch.float16)
    session(golden_input)
    print(f"Golden generated at: {golden_dir}")


if __name__ == "__main__":
    main()
