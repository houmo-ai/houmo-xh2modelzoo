import os
import time
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple

import torch

from xhquant.api import Config, get_root_logger, ptq_quantize, HMONNXGoldenInference
from xhquant.common.types import PrecisionMode
from xhquant.utils.config import ConfigDict

from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.qwen3_forcealigner import XHQwen3ASRLLMModel

from qwen_asr.core.transformers_backend import (
    Qwen3ASRForConditionalGeneration
)

def xhmodel_export_onnx(
    xh_model,
    tokenizer,
    data_batch,
    onnx_output_dir: str,
    cfg_name,
    device,
    dtype,
    logger,
    valid: bool = True,
):
    logger.info("Start exporting...")
    xh_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exporting graph conversion...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xh_model.change_eval_type(EvalModelType.EXPORTED)
    xh_model.to("cpu")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("*************** Start exporting onnx ***************")
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


def _debug_print_outs(outs):
    print("\n==================== DEBUG: test_step outputs ====================")
    print("test_step output type:", type(outs))
    if hasattr(outs, "keys"):
        print("keys:", list(outs.keys()))
        for k in outs.keys():
            v = getattr(outs, k)
            if torch.is_tensor(v):
                print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    elif isinstance(outs, (list, tuple)):
        for i, v in enumerate(outs):
            if torch.is_tensor(v):
                print(f"  out[{i}]: shape={tuple(v.shape)} dtype={v.dtype}")
    elif torch.is_tensor(outs):
        print("  tensor:", tuple(outs.shape), outs.dtype)
    print("==================================================================\n")


def main(args):

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = os.path.normpath(args.model)

    hf_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map=DEVICE,
    )
    hf_model.eval()

    model_name = os.path.basename(MODEL_PATH)
    target_device = "XH2a"  # 量化目标设备

    cfg = Config.fromfile(args.config)
    cfg_name = f"{model_name}_{target_device}"

    cfg.work_dir = str(Path("./work_dirs") / cfg_name)
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.dtype = "float16"
    logger = get_root_logger()
    logger.info(f"\nConfig:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.dump(config_file)

    device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)

    xh_model = MODELS.build(cfg.model)
    model = xh_model.get_hf_model()
    assert isinstance(xh_model, XHQwen3ASRLLMModel)

    # =============== 关键：在 init_wrap_model 前覆盖 wrap_cfg ===============
    full_seq_len = 411
    xh_model.wrap_cfg.num_logits_to_keep = 0
    xh_model.wrap_cfg.only_first_block = False
    xh_model.wrap_cfg.input_sequence_length = full_seq_len

    # wrap text model
    xh_model.init_wrap_model(model.thinker.model)

    xh_model.wrap_model.lm_head = model.thinker.lm_head
    xh_model.wrap_model.lm_head.to(device)
    xh_model.wrap_model.lm_head.to(dtype)

    processor = xh_model.get_processor()
    tokenizer = processor.tokenizer

    # ===== 模型设置 =====
    xh_model.to(device)
    xh_model.to(dtype)
    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)

    model.to(device)
    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"

    text_config = model.config.thinker_config.text_config
    hidden_size = text_config.hidden_size
    num_decode_layers = text_config.num_hidden_layers

    logger.info(f"text_config.hidden_size={hidden_size}, layers={num_decode_layers}")

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(config_file.relative_to(cfg.work_dir)),
        )
    )
    meta_info["wrap_cfg"] = xh_model.wrap_cfg.to_dict()
    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir
    
    hf_model_config_dir = cfg.config_dir

    hf_config_dir = Path(cfg.work_dir) / "ConfigFiles"
    hf_config_dir.mkdir(exist_ok=True, parents=True)
    hf_config_files = [
        "chat_template.json",
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "configuration.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json"
    ]
    
    for cfg_file in hf_config_files:
        shutil.copyfile(
            Path(hf_model_config_dir) / cfg_file,
            Path(hf_config_dir) / cfg_file,
        )
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    token_embedding = xh_model.token_embedding
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    # ===== 构造 prefill 输入 =====
    final_inputs_embeds = torch.randn((1, full_seq_len, hidden_size), device=device, dtype=torch.float16)

    data_batch = {
        "input_embeds": final_inputs_embeds.half(),
        "past_seq_length": [0],
        "current_input_length": [full_seq_len],
    }

    # Debug: 这里必须看到 (1,411,1024)
    with torch.no_grad():
        outs = xh_model.test_step(data_batch)
    _debug_print_outs(outs)

    # ===== 量化 =====
    xh_model.interactive_mode = True
    logger.info("************* convert to frontend graph *************")
    xh_model.convert_to_fronted_graph(data_batch)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("************* convert to quanted graph *************")
    xh_model.convert_to_quant_graph(target_device)

    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()
    xh_model.to(dtype)
    xh_model.to(device)

    logger.info("*************** Start PTQ Quantize ***************")
    calib_data = xh_model.prepare_inputs(data_batch)
    new_args = []
    for arg in calib_data:
        if isinstance(arg, (List, Tuple)):
            new_args.extend(arg)
        else:
            new_args.append(arg)
    ptq_quantize(xh_model.quanted_model, [new_args], PrecisionMode.ALIGNED, [exec_device])
    logger.info("*************** Finished PTQ Quantize **************")

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

    # ===== prefill 导出 =====
    xh_model = xh_model.to("cpu")

    work_dir = Path("work_dirs") / cfg_name
    prefill_onnx_dir = work_dir / "Prefill"
    prefill_golden_path = prefill_onnx_dir / "hmonnx/golden"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)

    num_hidden_layers = num_decode_layers
    base_inputs = ["input_embeds", "past_seq_length", "current_input_length"]
    key_names = [f"past_key_cache_{i}" for i in range(num_hidden_layers)]
    value_names = [f"past_value_cache_{i}" for i in range(num_hidden_layers)]
    input_names = base_inputs + key_names + value_names

    xh_model.export_cfg = ConfigDict(dict(
        input_names=input_names,
        output_names=["hidden_states"],  # 名字不重要，关键是输出 shape
    ))

    xh_model.set_input_sequence_length(full_seq_len)

    prefill_onnx_file = xhmodel_export_onnx(
        xh_model, tokenizer, data_batch,
        str(prefill_onnx_dir),
        f"{cfg_name}_prefill_fullseq",
        "cpu", dtype, logger, False
    )

    if args.gen_golden and not Path(prefill_golden_path).exists():
        session = HMONNXGoldenInference(prefill_onnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = str(prefill_onnx_dir / "hmonnx/golden")
        session.step = 0
        session(*calib_data)

    xh_model.release_exported_model()
    logger.info(f"save prefill onnx model to {prefill_onnx_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/models/Qwen/Qwen3-ForcedAligner-0.6B/"))
    parser.add_argument(
        "--config",
        type=str,
        default="./config/llm/qwen3_asr_decode_xh2a.py",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type")
    parser.add_argument("--gen_golden", action="store_true", help="generate golden data")
    args = parser.parse_args()
    main(args)