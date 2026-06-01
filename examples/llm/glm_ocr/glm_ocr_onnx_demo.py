import json

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.modeling_outputs import BaseModelOutputWithPooling

from common import build_inputs, build_messages, load_quarot_gptq_state_dict, resolve_torch_dtype


class ONNXVisionWrapper(nn.Module):
    def __init__(self, onnx_path: str, dtype: torch.dtype, spatial_merge_size: int, exec_device: torch.device):
        super().__init__()
        self.onnx_path = onnx_path
        self.dtype = dtype
        self.spatial_merge_size = int(spatial_merge_size)

        providers = ["CPUExecutionProvider"]
        if exec_device.type == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(self.onnx_path, providers=providers)

    @staticmethod
    def _np_dtype_from_onnx_type(type_str: str):
        if "float16" in type_str:
            return np.float16
        if "float" in type_str:
            return np.float32
        if "int64" in type_str:
            return np.int64
        if "int32" in type_str:
            return np.int32
        raise ValueError(f"Unsupported onnx input type: {type_str}")

    def _as_onnx_input(self, input_name: str, tensor: torch.Tensor):
        type_str = None
        for inp in self.session.get_inputs():
            if inp.name == input_name:
                type_str = inp.type
                break
        if type_str is None:
            raise KeyError(f"Input {input_name} not found in ONNX session")

        tensor_cpu = tensor.detach().cpu()
        if tensor_cpu.dtype == torch.bfloat16:
            tensor_cpu = tensor_cpu.to(torch.float32)
        arr = tensor_cpu.numpy()
        return arr.astype(self._np_dtype_from_onnx_type(type_str), copy=False)

    @torch.no_grad()
    def forward(self, pixel_values, grid_thw=None, image_grid_thw=None, return_dict=True, **kwargs):
        del kwargs
        if image_grid_thw is None:
            image_grid_thw = grid_thw
        if image_grid_thw is None:
            raise ValueError("image_grid_thw or grid_thw is required for ONNX vision inference.")

        ort_inputs = {
            "pixel_values": self._as_onnx_input("pixel_values", pixel_values),
            "image_grid_thw": self._as_onnx_input("image_grid_thw", image_grid_thw),
        }
        image_embeds = torch.from_numpy(self.session.run(None, ort_inputs)[0]).to(pixel_values.device, dtype=self.dtype)

        if return_dict:
            return BaseModelOutputWithPooling(last_hidden_state=image_embeds, pooler_output=image_embeds)
        return (image_embeds,)


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="/data02/datasets/GLM-OCR/")
    parser.add_argument("--visual_onnx_path", type=str, default="work_dirs/glm_ocr_onnx_export/glm_ocr_vision.onnx")
    parser.add_argument("--image", type=str, default="examples/llm/glm_ocr/data/img3.png")
    parser.add_argument("--prompt", type=str, default="Text Recognition:")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--attn_implementation", type=str, default="eager")
    parser.add_argument("--quarot_gptq_path", type=str, default=None)
    parser.add_argument("--images_json", type=str, default=None)
    parser.add_argument("--do_sample", action="store_true", help="do sample")
    parser.add_argument("--output_path", type=str, default="output.txt")
    return parser


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    model_path = args.model

    dtype = resolve_torch_dtype(args.dtype)
    device = torch.device(args.device)

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()

    if args.quarot_gptq_path is not None:
        load_quarot_gptq_state_dict(model, args.quarot_gptq_path, strict=False)

    model.to(device)

    visual_dtype = model.model.visual.dtype
    visual_spatial_merge_size = getattr(model.model.visual, "spatial_merge_size", 2)
    model.model.visual = ONNXVisionWrapper(
        args.visual_onnx_path,
        dtype=visual_dtype,
        spatial_merge_size=visual_spatial_merge_size,
        exec_device=device,
    )

    if args.images_json is not None:
        with open(args.output_path, "w", encoding="utf-8") as f_txt:
            with open(args.images_json, "r", encoding="utf-8") as f:
                images_json = json.load(f)
            for i, (image_name, image_info) in enumerate(images_json.items()):
                image_path = image_info["url"]
                prompt = image_info["prompt"]
                print(f"Processing {i}: {image_path}", flush=True)
                messages = build_messages(image_path, prompt)
                inputs = build_inputs(processor, messages, device=device)
                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=args.do_sample,
                    )
                out = processor.decode(generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
                f_txt.write(f"{image_name}\n{out}\n")
                f_txt.flush()
                print(f"[{i}] {image_name}", flush=True)
                print(out, flush=True)
    else:
        if args.image is None:
            raise ValueError("--image is required when --images_json is not provided")
        messages = build_messages(args.image, args.prompt)
        inputs = build_inputs(processor, messages, device=device)
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
            )
        out = processor.decode(generated_ids[0][inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        print(out, flush=True)


if __name__ == "__main__":
    main()
