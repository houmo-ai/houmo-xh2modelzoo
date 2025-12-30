import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))
import argparse
import random
import time
from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm
from xhquant.api import (
    Config,
    ConfigDict,
    DeviceType,
    FrontendType,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    export_onnx,
    get_root_logger,
    ptq_quantize,
    to_export_graph,
    to_frontend_graph,
    to_quant_graph,
    xhquant_init,
)
from xhquant.common.types import DeviceType, FrontendType, PrecisionMode
from xhquant.utils.logger import padding_message
from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

from xh2_model_zoo.xh_cv.models.vit import vit

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
random.seed(42)
np.random.seed(42)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--onnx_path", default="data/models/vit_new.onnx", type=str
    )
    parser.add_argument(
        "--input_shape", default=[1, 3, 224, 224], type=int, nargs="+", help="[h,w] use custom onnx should apply"
    )
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type, default is w8a8")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--image", type=str, default="data/images/000000001490.jpg")
    parser.add_argument("--save_golden", action="store_true", help="save golden model")
    parser.add_argument("--calib_num", default=32, type=int)
    parser.add_argument("--test_num", default=1000, type=int)
    parser.add_argument("--test_batch_size", default=1, type=int, help="resnest50_224x224: 1")
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--method", type=str, default="hquant")
    parser.add_argument("--mix_search", type=str, default=None, help="mix search settings")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    logger = get_root_logger()
    torch.manual_seed(1024)
    if args.onnx_path is None:
        args.onnx_path = "vit_small_patch16_224"

    if args.onnx_path == "vit_small_patch16_224":
        args.test_batch_size = 1

    hm_vit = vit.Vit(
        model_name=args.onnx_path,
        device=args.device,
        input_shape=args.input_shape[-2:],
    )
    onnx_name = args.onnx_path.split("/")[-1][:-5]
    if args.mix_search is not None:
        onnx_name += "_mix"
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a

    xhquant_init(None, debug=args.debug)
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}_w{quant_type[1]}a{quant_type[3]}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file: str = str(out_hmonnx_file)

    start = time.time()
    logger.info(f"Start convert onnx to hmonnx, time: {start}")

    start = time.time()
    logger.info(f"Start convert onnx to hmonnx, time: {start}")
    if not os.path.exists(out_hmonnx_file):
        input_names = ["images"]
        output_names = ["outs"]
        input_args: List[Tensor] = [torch.randn(args.input_shape, dtype=torch.float32)]

        convert_onnx_to_hmonnx(
            hm_vit.model_path,
            input_args,
            DeviceType.XH2a,
            out_hmonnx_file,
            quant_config=quant_config,
            input_names=["input"],
            output_names=output_names,
            mix_search=args.mix_search,
        )

    session = HMONNXInference(out_hmonnx_file)
    if args.save_golden:
        start = time.time()
        logger.info(f"Start save golden model, time: {start}")
        session.save_golden = True
        session.save_golden_dir = (
            work_dirs / "hmonnx" / f"{onnx_name}_{target_device}_w{quant_type[1]}a{quant_type[3]}_golden"
        )
        end = time.time()
        logger.info(f"Save golden model success, time: {end - start}")
        logger.info(
            f"Save golden model to: {work_dirs / 'hmonnx' / 'golden' / f'{onnx_name}_{target_device}_w{quant_type[1]}a{quant_type[3]}.onnx'}"
        )

        calib_dataset, test_dataset = hm_vit.dataset(
            calib_num=args.calib_num, test_batch_size=args.test_batch_size, subset=args.test_num
        )
        with torch.no_grad():
            q = tqdm(test_dataset)
            for i, (inp, target) in enumerate(q):
                inp = inp.to(args.device)
                pre_out = hm_vit.pre_process(inp)
                nn_out = session.cuda().forward(pre_out.half()).to(pre_out)
                break

    torch.set_grad_enabled(False)
    if not args.no_eval:
        session.save_golden = False
        calib_dataset, test_dataset = hm_vit.dataset(
            calib_num=args.calib_num, test_batch_size=args.test_batch_size, subset=args.test_num
        )

        from tqdm import tqdm

        pos = tot = 0
        with torch.no_grad():
            q = tqdm(test_dataset)
            for i, (inp, target) in enumerate(q):
                inp = inp.to(args.device)
                target = target.to(args.device)
                pre_out = hm_vit.pre_process(inp)
                nn_out = session.cuda().forward(pre_out.half()).to(pre_out)
                post_out = hm_vit.post_process(nn_out)
                pos_num = torch.sum(post_out.argmax(1) == target).item()
                pos += pos_num
                tot += inp.size(0)
                q.set_postfix({"acc": pos / tot})
        print(f"accuracy", pos / tot)
