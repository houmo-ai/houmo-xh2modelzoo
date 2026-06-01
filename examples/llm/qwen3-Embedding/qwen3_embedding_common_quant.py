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
from transformers import AutoConfig, AutoModel, AutoTokenizer


def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llm_models/Qwen/Qwen3-Embedding-4B",
        help="HF model directory or name",
    )
    parser.add_argument("--skip-quarot", action="store_true", help="skip_quarot")
    parser.add_argument("--skip-gptq", action="store_true", help="skip_gptq")
    parser.add_argument("--w-bits", type=int, default=4)
    parser.add_argument("--w-head-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--resume", action="store_true", help="resume from the cache")
    parser.add_argument("--out-dir", type=str, default="work_dirs/")
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument("--calib-dataset", type=str, default="wikitext2")
    parser.add_argument(
        "--data_files", nargs="+", type=str, default=[], help="List of dataset files"
    )

    parser.add_argument(
        "--gptq-config", type=str, default="", help="Optional GPTQ config yaml"
    )
    parser.add_argument("--seqlen", type=int, default=2048)
    return parser


def msg_output_format(title):
    padding_str = "*" * 10
    title = f"{padding_str} {title} {padding_str}"
    return title


class BaseModelWrapper(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.model = base_model
        self.config = base_model.config
        self.lm_head = None

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def main():
    parser = parse_arguments()
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    hf_model_dir = args.model

    hf_model_path = osp.normpath(osp.abspath(args.model))
    model_name = Path(hf_model_path).name
    is_0_6b = "0.6b" in model_name.lower()
    if is_0_6b:
        logger.info(
            msg_output_format(
                "Detected 0.6B model: skip Quarot/GPTQ. Use PTQ export flow (w8a8)."
            )
        )
        return
    cfg_name = model_name
    if not args.skip_quarot:
        cfg_name += "_quarot"
    if not args.skip_gptq:
        cfg_name += "_gptq"

    cfg_name += f"_transformers-{transformers.__version__}"

    work_dir = Path(out_dir) / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)
    _ = AutoConfig.from_pretrained(hf_model_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16

    native_model = AutoModel.from_pretrained(
        hf_model_dir,
        torch_dtype=dtype,
        device_map="cpu",
        trust_remote_code=True,
    )

    native_model.eval()
    native_model.to(dtype)

    tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, padding_side="left")

    quant_methods = []
    if not args.skip_quarot:
        torch.cuda.reset_peak_memory_stats()
        quant_methods.append("quarot")
        quant_name = "_".join(quant_methods)
        filename = work_dir / f"{quant_name}-state-dict.safetensors"
        if not os.path.exists(filename):
            from xh_model_zoo.xh_llm.models.qwen3_embedding.quarot_adapter.quantizer_utils import quarot

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
            if "post_norm_linear.weight" in state_dict:
                hidden_size = native_model.config.hidden_size
                native_model.post_norm_linear = torch.nn.Linear(
                    hidden_size, hidden_size, bias=False
                )
            native_model.load_state_dict(state_dict, strict=False)
            logger.info(msg_output_format(f"Load state_dict from {filename}"))

    consumption = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
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
        from xh_model_zoo.xh_llm.models.qwen3_embedding.quarot_adapter.quantizer_utils import gptq

        gptq_config = dict(
            calib_dataset=args.calib_dataset,
            calib_samples=args.calib_samples,
            seqlen=args.seqlen,
            w_clip=True,
            w_bits=args.w_bits,
            w_asym=False,
            w_groupsize=64,
            percdamp=0.01,
            act_order=False,
            int8_down_proj=False,
            heading_gptq=False,
            w_head_bits=args.w_head_bits,
        )

        torch.cuda.reset_peak_memory_stats()
        logger.info(msg_output_format("Start gptq quantization"))
        quant_methods.append("gptq")
        layers_cache_dir = work_dir / "layers_cache"
        layers_cache_dir.mkdir(exist_ok=True, parents=True)

        wrapper = BaseModelWrapper(native_model)
        wrapper = gptq(
            wrapper,
            args=args,
            model_name=hf_model_dir,
            **gptq_config,
            device=device,
            processor=None,
            data_files=args.data_files,
            is_qwen3_embedding=True,
        )
        logger.info(msg_output_format("End gptq quantization"))
        if isinstance(wrapper, BaseModelWrapper):
            native_model = wrapper.model

        consumption = (
            torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
        )
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
        if hasattr(native_model, "post_norm_linear"):
            if "post_norm_linear.weight" not in state_dict:
                state_dict["post_norm_linear.weight"] = (
                    native_model.post_norm_linear.weight.detach().clone()
                )
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
    # import debugpy

    # debugpy.listen(("0.0.0.0", 1160))
    # print("✅ debugpy listening on 0.0.0.0:5678, waiting for VSCode attach...")
    # debugpy.wait_for_client()
    # print("✅ VSCode attached, continue running.")
    main()
