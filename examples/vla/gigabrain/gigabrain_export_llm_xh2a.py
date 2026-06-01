"""
GigaBrain-0.1 LLM 导出脚本
导出 PaliGemma2 语言模型主体部分
"""

import argparse
import importlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from xhquant.api import Config, ConfigDict, PrecisionMode, ptq_quantize, get_root_logger

# 确保导入模块以触发注册
import xh_model_zoo.xh_llm.models.gigabrain

from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.utils import decode_next_token


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
    """导出模型到 ONNX 格式"""
    logger.info("Start exporting...")
    xh_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exporting...")

    logger.info(f"************* Start Exported Graph *************")
    logger.info(f"************* End Exported Graph *************")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    xh_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("*************** Start exporting onnx ***************")
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


def main(args):
    cfg = Config.fromfile(args.config)
    cfg_name = Path(args.config).stem
    cfg_name = f"{cfg_name}"
    cfg.work_dir = str(Path("./work_dirs") / cfg_name)
    log_file = Path(cfg.work_dir) / f"{cfg_name}.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"
    only_export = not args.valid
    
    logger = get_root_logger()
    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.dump(config_file)

    device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)
    
    # 构建模型
    xh_model = MODELS.build(cfg.model)

    # 加载 GigaBrain 的 PaliGemma2 部分
    policy = xh_model.get_hf_model()

    # ==========================
    # NEW: 使用权重注入方案！
    # ==========================
    from xh_model_zoo.xh_llm.models.gigabrain.weight_transfer import create_and_inject_llm_skeleton

    standard_llm = create_and_inject_llm_skeleton(policy)

    # 初始化包装模型 - 访问的是被注入后的标准模型
    xh_model.init_wrap_model(standard_llm)

    tokenizer = xh_model.get_tokenizer(cfg.config_dir)

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
        "added_tokens.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
    ]
    for cfg_file in hf_config_files:
        src = Path(hf_model_config_dir) / cfg_file
        if src.exists():
            shutil.copyfile(src, Path(hf_config_dir) / cfg_file)
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    # 保存 token embedding
    token_embedding = xh_model.token_embedding
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)
    
    # 准备测试输入
    # GigaBrain 使用任务描述作为语言输入
    task_prompt = "Pick up the red block and place it in the bin"
    
    # 构造输入
    model_inputs = tokenizer([task_prompt], return_tensors="pt").to(device)
    input_ids = model_inputs.input_ids
    data_batch = {
        "input_ids": input_ids.to(device),
        "past_seq_length": [0],
    }
    xh_model.to(dtype)
    with torch.no_grad():
        outputs = xh_model.test_step(data_batch)
    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
    xh_model.to(device)
    xh_model.to(dtype)
    logger.info("Start wrap model prefill .......................")

    xh_model.interactive_mode = True
    logger.info("************* convert to frontend graph *************")
    xh_model.convert_to_fronted_graph(data_batch)
    logger.info(f"************* Start Frontend Graph *************")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("************* convert to quanted graph *************")
    xh_model.convert_to_quant_graph(cfg.target_device)

    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()
    xh_model.to(dtype)
    xh_model.to(device)

    # PTQ 量化
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
    logger.info("*************** Finished PTQ Quantize ***************")

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

    if not only_export:
        with torch.no_grad():
            outs = xh_model.test_step(data_batch)
            quanted_aligned_logits = outs.logits.detach()

        prefill_next_token_id, prefill_next_token_text = decode_next_token(
            tokenizer, quanted_aligned_logits
        )
        logger.info(f"Prefill Quanted Model next token: {prefill_next_token_id} {prefill_next_token_text}")
        xh_model.quanted_model.dump_quant_info_to_onnx(
            Path(cfg.work_dir) / f"{cfg_name}_quant_info.onnx"
        )
    else:
        prefill_next_token_id = None

    # 导出 Prefill 模型
    xh_model = xh_model.to("cpu")
    data_batch["input_ids"] = data_batch["input_ids"].to("cpu")
    logger.info("*************** Start exporting prefill model ***************")
    begin_time = time.time()

    prefill_onnx_file = xhmodel_export_onnx(
        xh_model,
        tokenizer,
        data_batch,
        str(prefill_onnx_dir),
        f"{cfg_name}_prefill",
        device,
        dtype,
        logger,
        not only_export,
    )

    end_time = time.time()
    logger.info(f"Time cost for exporting {end_time - begin_time}")

    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()
    logger.info(f"save prefill onnx model to {prefill_onnx_file}")
    logger.info("*************** Finished exporting prefill model ***************")

    # 导出 Decode 模型
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)
    data_batch["input_ids"] = data_batch["input_ids"].to(device)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xh_model.set_input_sequence_length(1)

    past_seq_len = input_ids.shape[-1]
    if prefill_next_token_id is None:
        prefill_next_token_id = input_ids[:, :1]
    input_ids = prefill_next_token_id
    logger.info(f"past_seq_len: {past_seq_len}")
    
    data_batch = {
        "input_ids": input_ids.to(device),
        "past_seq_length": [past_seq_len],
    }
    
    if not only_export:
        with torch.no_grad():
            outs = xh_model.test_step(data_batch)
            decode_logits = outs.logits.detach()
        decode_token_id, decode_token_text = decode_next_token(tokenizer, decode_logits)
        logger.info(f"Decode on Quanted Model next token: {decode_token_id} {decode_token_text}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("*************** Start exporting decode model ***************")
    xh_model = xh_model.to("cpu")
    data_batch["input_ids"] = data_batch["input_ids"].to("cpu")
    begin_time = time.time()

    decode_onnx_file = xhmodel_export_onnx(
        xh_model,
        tokenizer,
        data_batch,
        str(decode_onnx_dir),
        f"{cfg_name}_decode",
        device,
        dtype,
        logger,
        not only_export,
    )

    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()
    logger.info(f"save decode onnx model to {decode_onnx_file}")
    json.dump(meta_info, open(Path(cfg.work_dir) / "export_meta_info.json", "w"), indent=4)
    logger.info("*************** Finished exporting decode model ***************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/vla/gigabrain/config/gigabrain/llm/gigabrain_llm_xh2a.py",
    )
    parser.add_argument("--valid", action="store_true", help="validate the model")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()
    main(args)
