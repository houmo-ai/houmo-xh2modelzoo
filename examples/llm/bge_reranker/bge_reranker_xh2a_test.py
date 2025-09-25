import argparse
import json
from pathlib import Path

import torch
from xhquant.api import HMONNXGoldenInference as HMONNXInference
from xhquant.api import get_root_logger
from xhquant.api import xhquant_init
from xhquant.xhonnxruntime import config as xhonnxruntime_config


def main(args):
    xhquant_init(None, False)

    logger = get_root_logger()
    model_config_file = args.config
    model_dir_path = Path(model_config_file).parent
    meta_info = json.load(open(model_config_file, "r"))
    cfg_name = meta_info["model_name"]
    device = torch.device("cuda")
    hmonnx_file = str(model_dir_path / meta_info["hmonnx_file"])

    # golden
    # bz = 10
    # seq_length = 512
    # input_ids = torch.randint(0, 100, (bz, seq_length)).to(device)
    # token_type_ids = torch.zeros((bz, seq_length), dtype=torch.int32).to(device)
    # attention_mask = torch.ones((bz, seq_length), dtype=torch.int16).to(device)

    golden_dir_path = model_dir_path / "golden"
    golden_dir_path.mkdir(parents=True, exist_ok=True)
    golden_dir = str(golden_dir_path)
    hm_session = HMONNXInference(hmonnx_file)
    hm_session.initialize()

    input_ids_info = hm_session.get_input("input_ids").shape
    token_type_ids_shape = hm_session.get_input("token_type_ids").shape
    attention_mask_shape = hm_session.get_input("attention_mask").shape

    logger.info(f"input_ids_shape: {input_ids_shape}")
    logger.info(f"token_type_ids_shape: {token_type_ids_shape}")
    logger.info(f"attention_mask_shape: {attention_mask_shape}")
    input_ids = torch.randint(0, 100, input_ids_shape).to(device)
    token_type_ids = torch.zeros(token_type_ids_shape).to(device)
    attention_mask = torch.ones(attention_mask_shape).to(device)

    hm_session.to(device)
    hm_session.save_golden = True
    hm_session.golden_dir = golden_dir
    hm_session.run(
        {
            "input_ids": input_ids.to(device).to(torch.int32),
            "token_type_ids": token_type_ids.to(device).to(torch.int32),
            "attention_mask": attention_mask.to(device).to(torch.int16),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="work_dirs/bge-large-zh-v1.5-XH2a/meta.json",
    )
    args = parser.parse_args()
    main(args)
