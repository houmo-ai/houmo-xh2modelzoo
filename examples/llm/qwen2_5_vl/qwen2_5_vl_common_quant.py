import argparse
import os
import os.path as osp
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from loguru import logger
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from tqdm import tqdm
from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info



def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="weights/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--skip-quarot", action="store_true", help="skip_quarot")
    parser.add_argument("--skip-gptq", action="store_true", help="skip_quarot")
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--resume", action="store_true", help="resume from the cache")
    parser.add_argument("--out-dir", type=str, default="work_dirs/")
    parser.add_argument("--validate", action="store_true", help="validate")
    return parser


def msg_output_format(title):
    padding_str = "*" * 10
    title = f"{padding_str} {title} {padding_str}"
    return title


def demo(model, processor):
    from xh_model_zoo.xh_llm.quarot import utils
    from accelerate import dispatch_model, infer_auto_device_map
    from accelerate.utils import get_balanced_memory
    raw_device = next(model.parameters()).device
    # model.to(utils.DEV)

    no_split_module_classes = ['LlamaDecoderLayer','QuantDecoderLayer',"RotateModule","SmoothModule","Qwen2DecoderLayer", "Qwen2_5_VLDecoderLayer"]
    max_memory = get_balanced_memory(model, no_split_module_classes=no_split_module_classes)
    device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes)
    dispatch_model(model, device_map=device_map, offload_buffers=True, offload_dir="offload", state_dict=model.state_dict())

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # Inference: Generation of the output
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
    from accelerate.hooks import remove_hook_from_module
    remove_hook_from_module(model)
    model.to(raw_device)
    utils.cleanup_memory()


def main():
    parser = parse_arguments()
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    hf_model_dir = args.model

    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    cfg_name = model_name
    if not args.skip_quarot:
        cfg_name += "_quarot"
    if not args.skip_gptq:
        cfg_name += "_gptq"

    work_dir = Path(out_dir) / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    config = AutoConfig.from_pretrained(hf_model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    native_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_model_dir,
        torch_dtype=dtype,
        device_map="cpu",
        # config=config,
        trust_remote_code=True,
        attn_implementation="eager",
    )

    if native_model.config.tie_word_embeddings:
        old_torchscript = native_model.config.torchscript
        native_model.config.torchscript = True
        native_model.tie_weights()
        native_model.config.tie_word_embeddings = False
        native_model.config.torchscript = old_torchscript

    native_model.eval()
    native_model.to(dtype)

    processor = AutoProcessor.from_pretrained(hf_model_dir)

    if args.validate:
        demo(native_model, processor)

    quant_methods = []
    if not args.skip_quarot:
        torch.cuda.reset_peak_memory_stats()
        quant_methods.append("quarot")
        quant_name = "_".join(quant_methods)
        filename = work_dir / f"{quant_name}-state-dict.safetensors"
        if not os.path.exists(filename):
            from xh_model_zoo.xh_llm.quarot.quantizer_utils import quarot

            logger.info(msg_output_format("Start quarot quantization"))
            native_model = quarot(native_model, device=device)
            logger.info(msg_output_format("End quarot quantization"))
            native_model.to(torch.float16)
            state_dict = native_model.state_dict()
            logger.info(msg_output_format(f"Saving checkpoint to: {filename}"))
            # torch.save(state_dict, filename)
            save_safetensors_file(state_dict, filename)
            logger.info(f"Save checkpoint to: {filename}")
        else:
            state_dict = load_safetensors_file(filename)
            native_model.to(torch.float16)
            native_model.load_state_dict(state_dict)
            logger.info(msg_output_format(f"Load state_dict from {filename}"))
    
    if args.validate:
        demo(native_model.to(torch.float32), processor)
    consumption = torch.cuda.max_memory_allocated()
    unit = "B"
    if consumption > 1024:
        consumption = consumption / 1024
        unit = "k"
        if consumption > 1024:
            consumption = consumption / 1024
            unit = "M"
        consumption = round(consumption, 2)
    logger.info(f"GPU memory cost for export {consumption}{unit}")

    if not args.skip_gptq:
        from xh_model_zoo.xh_llm.quarot.quantizer_utils import gptq

        gptq_config = dict(
            calib_dataset="laion/220k-GPT4Vision-captions-from-LIVIS",
            calib_samples=1,
            seqlen=2048,
            w_clip=True,
            w_bits=args.w_bits,
            w_asym=False,
            w_groupsize=64,
            percdamp=0.01,
            act_order=False,
            int8_down_proj=False,
            heading_gptq=True,
        )

        torch.cuda.reset_peak_memory_stats()
        logger.info(msg_output_format("Start gptq quantization"))
        quant_methods.append("gptq")
        layers_cache_dir = work_dir / "layers_cache"
        layers_cache_dir.mkdir(exist_ok=True, parents=True)

        native_model = gptq(
            native_model,
            args=args,
            model_name=hf_model_dir,
            **gptq_config,
            device=device,
            layers_cache_dir=str(layers_cache_dir),
            is_qwen2_5_vl=True,
            processor=processor,
        )
        logger.info(msg_output_format("End gptq quantization"))

        consumption = torch.cuda.max_memory_allocated()
        unit = "B"
        if consumption > 1024:
            consumption = consumption / 1024
            unit = "k"
            if consumption > 1024:
                consumption = consumption / 1024
                unit = "M"
            consumption = round(consumption, 2)
        logger.info(f"GPU memory cost for export {consumption}{unit}")

    if len(quant_methods) != 0:
        quant_name = "_".join(quant_methods)
        filename = work_dir / f"{quant_name}-state-dict.safetensors"
        state_dict = native_model.state_dict()
        # del native_model
        for k in tqdm(state_dict):
            paths = k.split(".")
            v = state_dict[k]
            if paths[-1] == "quant_weight":
                if v.min().item() >= -pow(2, 7) and v.max().item() <= pow(2, 7) - 1:
                    v = v.to(torch.int8)
                elif v.min().item() >= -pow(2, 15) and v.max().item() <= pow(2, 15) - 1:
                    v = v.to(torch.int16)
                else:
                    v = v.to(torch.float32)
            else:
                v = v.to(torch.float16)

            state_dict[k] = v
        logger.info(msg_output_format(f"Saving checkpoint to: {filename}"))
        save_safetensors_file(state_dict, filename)
        logger.info(f"Save checkpoint to: {filename}")

if __name__ == "__main__":
    main()
