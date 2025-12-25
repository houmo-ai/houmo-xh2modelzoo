import argparse
import os
import os.path as osp
from pathlib import Path

import torch
import transformers
from loguru import logger
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

def parse_arguments():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model", type=str, default="weights/UI-TARS-1.5-7B")
    parser.add_argument("--skip-quarot", action="store_true", help="skip_quarot")
    parser.add_argument("--skip-gptq", action="store_true", help="skip_gptq")
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--w-head-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--resume", action="store_true", help="resume from the cache")
    parser.add_argument("--out-dir", type=str, default="work_dirs/")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--data_files", nargs="+", type=str, default=[], help="List of dataset files (json format)")
    return parser


def msg_output_format(title):
    padding_str = "*" * 10
    title = f"{padding_str} {title} {padding_str}"
    return title

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

    cfg_name += f"_transformers-{transformers.__version__}"

    work_dir = Path(out_dir) / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    # Load Model
    logger.info(f"Loading model from {hf_model_dir}...")
    native_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        hf_model_dir,
        torch_dtype=dtype,
        device_map="cpu", # Load to CPU first to save GPU memory
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
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

    quant_methods = []
    
    # 1. Quarot Quantization
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
            save_safetensors_file(state_dict, filename)
            logger.info(f"Save checkpoint to: {filename}")
        else:
            state_dict = load_safetensors_file(filename)
            native_model.to(torch.float16)
            native_model.load_state_dict(state_dict)
            logger.info(msg_output_format(f"Load state_dict from {filename}"))

    consumption = torch.cuda.max_memory_allocated()
    unit = "B"
    if consumption > 1024:
        consumption = consumption / 1024
        unit = "k"
        if consumption > 1024:
            consumption = consumption / 1024
            unit = "M"
        consumption = round(consumption, 2)
    logger.info(f"GPU memory cost after quarot: {consumption}{unit}")

    if not args.skip_gptq:
        original_dataloader = torch.utils.data.DataLoader
        
        class SafeDataLoader(original_dataloader):
            def __init__(self, *args, **kwargs):
                if 'collate_fn' not in kwargs and kwargs.get('batch_size') == 1:
                    kwargs['collate_fn'] = lambda x: x[0]
                super().__init__(*args, **kwargs)
        
        torch.utils.data.DataLoader = SafeDataLoader
        
        from xh_model_zoo.xh_llm.quarot.quantizer_utils import gptq

        gptq_config = dict(
            calib_dataset=args.data_files[0], # Use custom data list mode
            calib_samples=args.calib_samples,
            seqlen=2048,
            w_clip=True,
            w_bits=args.w_bits,
            w_asym=False,
            w_groupsize=64,
            percdamp=0.01,
            act_order=False,
            int8_down_proj=False,
            heading_gptq=True,
            w_head_bits=args.w_head_bits
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
            is_qwen2_5_vl=True,
            processor=processor,
            data_files=args.data_files, # Pass the generated json files
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
        logger.info(f"GPU memory cost after gptq: {consumption}{unit}")

    # 3. Save Final Model
    if len(quant_methods) != 0:
        quant_name = "_".join(quant_methods)
        filename = work_dir / f"{quant_name}-state-dict.safetensors"
        state_dict = native_model.state_dict()
        
        # Convert weights to target types (int8/int16/float16)
        for k in tqdm(state_dict, desc="Converting weights"):
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
            
        logger.info(msg_output_format(f"Saving final checkpoint to: {filename}"))
        save_safetensors_file(state_dict, filename)
        logger.info(f"Save checkpoint to: {filename}")


if __name__ == "__main__":
    main()
