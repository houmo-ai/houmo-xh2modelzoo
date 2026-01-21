import json, argparse, os, datetime, os.path as osp, torch, torch.nn as nn
from typing import Optional, Tuple
from tqdm import tqdm

from qwen_vl_utils import process_vision_info
from transformers import AutoConfig
from xh_model_zoo.xh_llm.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration

from xh_model_zoo_develop.xh_llm.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
from xh_model_zoo_develop.xh_llm.models.qwen2_5_vl.qwen2_5_vl_demo import (
    Qwen2_5_VLLLMInferHMONNXImpl,
    Qwen2_5_VLHFCompatible,
)

torch.set_grad_enabled(False)


def load(export_dir):
    with open(osp.join(export_dir, "meta.json"), "r") as f:
        meta_info = json.load(f)
    # Paths exported by converter
    vision_path = osp.join(export_dir, meta_info["vision_onnx"])
    prefill_path = osp.join(export_dir, meta_info["prefill_onnx"])
    decode_path = osp.join(export_dir, meta_info["decode_onnx"])
    hf_config_dir = osp.join(export_dir, meta_info["hf_config"])
    token_embedding_file = osp.join(export_dir, meta_info["token_embedding_file"])
    wrap_cfg = meta_info["wrap_cfg"]

    hf_config = AutoConfig.from_pretrained(hf_config_dir)
    # Load embedding weights exported together with ONNX to avoid touching native weights.
    token_embedding_state = torch.load(token_embedding_file, weights_only=True)
    vocab_size, hidden_size = token_embedding_state["weight"].shape
    token_embedding = nn.Embedding(vocab_size, hidden_size)
    token_embedding.load_state_dict(token_embedding_state)

    native_model = Qwen2_5_VLForConditionalGeneration._from_config(hf_config)
    processor = Qwen2_5_VLProcessor.from_pretrained(hf_config_dir)

    model = (
        Qwen2_5_VLHFCompatible.from_hmonnx(
            vision_path=vision_path,
            prefill_path=prefill_path,
            decoder_path=decode_path,
            wrap_cfg=wrap_cfg,
            hf_config=hf_config,
            token_embedding=token_embedding,
            native_model_or_path=native_model,
            processor=processor,
        )
        .cuda()
        .half()
        .eval()
    )
    return processor, model


def main(args):
    export_dir = args.export_dir
    processor, hf_compatible_model = load(export_dir)
    hf_compatible_model.demo("Describe this image.", "data/images/ILSVRC2012_val_00002031.JPEG")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--export-dir",
        type=str,
        default="work_dirs/Qwen2.5-VL-7B-Instruct-XH2a",
        help="work directory containing exported onnx files",
    )
    parser.add_argument("--dataset-name", type=str, default="CMMMU_VAL", help="dataset name")
    parser.add_argument("--offload", action="store_true", help="offload mode")
    parser.add_argument("--demo_prompt", type=str, default="你多大了？用中文回答。", help="demo prompt")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
