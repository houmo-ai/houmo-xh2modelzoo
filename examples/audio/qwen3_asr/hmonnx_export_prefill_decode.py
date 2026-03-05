import os
import time
import json
import shutil
import librosa
import argparse

import torch
import torch.nn as nn

from pathlib import Path
from typing import List, Tuple
from xhquant.api import ptq_quantize

from xhquant.api import Config, ConfigDict, PrecisionMode, get_root_logger, ptq_quantize
from xhquant.common.types import PrecisionMode
from xhquant.utils.config import ConfigDict

from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.qwen3_asr import XHQwen3ASRLLMModel, XHQwen3ASRHMONNXModel

# from qwen_asr.core.transformers_backend import (
#     Qwen3ASRForConditionalGeneration
# )

from xh_model_zoo.xh_llm.models.qwen3_asr import (
    Qwen3ASRForConditionalGeneration
)

GB = int(2**30)
_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 1.8)

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
    xh_model.to("cpu")  # 切换到cpu上进行模型导出
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exporting...")

    logger.info(f"************* Start Exported Graph *************")
    # logger.info(str(xh_model.exported_model.graph))
    logger.info(f"************* End Exported Graph *************")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    xh_model.to("cpu")  # 切换到cpu上进行模型导出
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("*************** Start exporting onnx ***************")
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file

def main(args):

    # ============================================================ 配置与初始化 ============================================================ # 
    cfg = Config.fromfile(args.config)
    cfg_name = Path(args.config).stem
    cfg_name = f"{cfg_name}"
    cfg.work_dir = str(Path("./work_dirs") / cfg_name)
    log_file = Path(cfg.work_dir) / f"{cfg_name}.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    target_device = "XH2a"  # 量化目标设备
    
    cfg.dtype = "float16"
    # only_export = not args.valid  # 仅仅导出模型，不进行推理验证
    logger = get_root_logger()
    logger.info(f"\nConfig:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.dump(config_file)
    
    MODEL_PATH = os.path.expanduser("~/models/Qwen/Qwen3-ASR-0.6B/")
    hf_model = Qwen3ASRForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map=DEVICE,
    )
    hf_model.eval()
    
    device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)
    
    xh_model = MODELS.build(cfg.model)
    # breakpoint()
    
    model = xh_model.get_hf_model()
    assert isinstance(xh_model, XHQwen3ASRLLMModel), f"Model must be XHQwen3ASRLLMModel, but got {type(xh_model)}"

    xh_model.init_wrap_model(model.thinker.model)
        
    xh_model.wrap_model.lm_head = model.thinker.lm_head
    xh_model.wrap_model.lm_head.to(device)
    xh_model.wrap_model.lm_head.to(dtype)
        
    processor = xh_model.get_processor()
    
    prefill_onnx_dir = Path(cfg.work_dir) / "prefill_onnx"
    decode_onnx_dir = Path(cfg.work_dir) / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)
    
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

    hf_config_dir = Path(cfg.work_dir) / "hf_config"
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

    # xh_model.past_key_caches 在 init_wrap_model 时已经被包装成 list 了
    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape # torch.Size([1, 8, 2048, 128])
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)
    
    # ============================================================ 定义 Fx 包装类 ============================================================ # 

    # wrapped decoder
    xh_model.to(device)
    xh_model.to(dtype)
    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
    # entire model
    model.to(device)

    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"
    text_config = model.config.thinker_config.text_config

    cache_len = 1500 # kv cache 缓存长度
    head_dim = text_config.hidden_size // text_config.num_key_value_heads # 128
    num_key_value_heads = text_config.num_key_value_heads # 8
    num_decode_layers = text_config.num_hidden_layers # 28
    
    # ============================================================ 音频文本预处理与特征融合 ============================================================ # 

    tokenizer = processor.tokenizer
        
    final_inputs_embeds = torch.randn((1, 411, 1024), device=device, dtype=torch.float16)
    print(f"final_inputs_embeds.shape: {final_inputs_embeds.shape}") # torch.Size([1, 411, 2048])
    
    # ============================================================ 构造输入 ============================================================ # 

    seq_len = final_inputs_embeds.shape[1]
    # 补全第二维度到 411
    final_inputs_embeds = torch.cat([final_inputs_embeds, torch.zeros((1, 411 - seq_len, final_inputs_embeds.shape[2]), dtype=torch.float16, device=device)], dim=1)

    data_batch = {
        "input_embeds": final_inputs_embeds.half(),
        "past_seq_length": [0]
    }

    # xh_model.set_input_sequence_length(1)
    
    with torch.no_grad():
        outs = xh_model.test_step(data_batch)
        
    
    # ============================================================ 量化 ============================================================ # 
    
    xh_model.interactive_mode = True
    logger.info("************* convert to frontend graph *************")

    xh_model.convert_to_fronted_graph(data_batch) 
    logger.info(f"************* Start Frontend Graph *************")

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
    calib_data = new_args
    ptq_quantize(xh_model.quanted_model, [calib_data], PrecisionMode.ALIGNED, [exec_device])
    logger.info("*************** Finished PTQ Quantize **************")


    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

    # ============================================================ prefill 导出 ============================================================ # 

    xh_model = xh_model.to("cpu")
    # full_seq_len = final_inputs_embeds.shape[1]
    full_seq_len = 411

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    work_dir = Path("work_dirs") / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)

    prefill_onnx_dir = work_dir / "prefill_onnx"
    decode_onnx_dir = work_dir / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)
    
    # export_cfg 展开 past_key_cache 和 past_value_cache 的输入，变成多个输入，方便后续对齐
    num_hidden_layers = 28
    base_inputs = ["input_embeds", "past_seq_length", "current_input_length"]
    key_names = [f"past_key_cache_{i}" for i in range(num_hidden_layers)]
    value_names = [f"past_value_cache_{i}" for i in range(num_hidden_layers)]
    input_names = base_inputs + key_names + value_names
    xh_model.export_cfg = ConfigDict(dict(input_names=input_names, output_names=["last_hidden_state"]))
    

    xh_model.set_input_sequence_length(full_seq_len)

    prefill_onnx_file = xhmodel_export_onnx(
        xh_model,
        tokenizer,
        data_batch,
        str(prefill_onnx_dir),
        f"{cfg_name}_prefill",
        "cpu",               
        dtype,
        logger,
        False,
    )

    xh_model.release_exported_model()
    logger.info(f"save prefill onnx model to {prefill_onnx_file}")
    logger.info("*************** Finished exporting prefill model ***************")
    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))

    # ============================================================ decode 导出 ============================================================ # 
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)
    
    data_batch["input_embeds"] = data_batch["input_embeds"].to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 2. 设置输入长度为 1 (Decode模式)
    xh_model.set_input_sequence_length(1)

    # 3. 准备 Decode 阶段的 Embedding 和数据包
    past_seq_len = final_inputs_embeds.shape[1]
    prefill_next_token_embeds = final_inputs_embeds[:, -1:, :]
    final_inputs_embeds = prefill_next_token_embeds
    logger.info(f"past_seq_len: {past_seq_len}")

    data_batch = {
        "input_embeds": final_inputs_embeds.to(device),
        "past_seq_length": [past_seq_len],
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("*************** Start exporting decode model ***************")

    # 4. 切换到 CPU 进行导出准备
    xh_model = xh_model.to("cpu")

    decode_cpu_inputs_embeds = data_batch["input_embeds"].to("cpu")
    if decode_cpu_inputs_embeds.dim() == 2:
        decode_cpu_inputs_embeds = decode_cpu_inputs_embeds.unsqueeze(0)
        
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 执行导出
    decode_onnx_file = xhmodel_export_onnx(
        xh_model,
        tokenizer,
        data_batch,
        str(decode_onnx_dir),
        f"{cfg_name}_decode",
        "cpu",
        dtype,
        logger,
        False,
    )

    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
    json.dump(meta_info, open(Path(cfg.work_dir) / "export_meta_info.json", "w"), indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=os.path.expanduser("~/models/Qwen/Qwen3-ASR-0.6B/"))
    parser.add_argument(
        "--config",
        type=str,
        default="/data01/home/binghu.ji/xh2modelzoo/examples/llm/qwen3_asr/config/llm/qwen3_asr_decode_xh2a.py",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type")
    parser.add_argument("--gen_golden", action="store_true", help="generate golden data")
    args = parser.parse_args()
    main(args)