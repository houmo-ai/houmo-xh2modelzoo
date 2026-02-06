import argparse
import importlib
import importlib.metadata
import json
import shutil
import sys
import time
from pathlib import Path
from typing import List, Tuple
import torch

from transformers import AutoProcessor
from xhquant.api import Config, ConfigDict, PrecisionMode, get_root_logger, ptq_quantize
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.pi05 import XHGemmaLLMModel
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
    cfg = Config.fromfile(args.config)
    cfg_name = Path(args.config).stem
    cfg_name = f"{cfg_name}"
    cfg.work_dir = str(Path("./work_dirs") / cfg_name)
    log_file = Path(cfg.work_dir) / f"{cfg_name}.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"
    only_export = not args.valid  # 仅仅导出模型，不进行推理验证
    logger = get_root_logger()
    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.dump(config_file)

    device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)
    xh_model = MODELS.build(cfg.model)
    policy = xh_model.get_hf_model(model="pi0.5")
    assert isinstance(xh_model, XHGemmaLLMModel), f"Model must be XHGemmaLLMModel, but got {type(xh_model)}"

    xh_model.init_wrap_model(policy.model.paligemma_with_expert.paligemma.model.language_model)
    tokenizer = xh_model.get_tokenizer()

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
        shutil.copyfile(
            Path(hf_model_config_dir) / cfg_file,
            Path(hf_config_dir) / cfg_file,
        )
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    token_embedding = xh_model.token_embedding
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)
    
    system_prompt = "你是一个专业的AI助手，回答简洁、准确，使用中文。"
    user_query = "请解释Gemma模型的核心优势是什么？"

    text = f"""
    {system_prompt}
    <start_of_turn>user
    {user_query}<end_of_turn>
    <start_of_turn>assistant
    """.strip()
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    # input_ids = torch.randint(0, tokenizer.vocab_size, (1, 100), device=device)
    input_ids = model_inputs.input_ids
    data_batch = {
        "input_ids": input_ids.to(device),
        "past_seq_length": [0],
    }

    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
    xh_model.to(device)
    xh_model.to(dtype)
    logger.info("Start wrap model prefill .......................")
    # with torch.no_grad():
    #     outs = xh_model.test_step(data_batch)
    #     wraped_hidden_states = outs.hidden_states
        #logger.info(f"wraped_logits shape: {wraped_logits.shape}")

    xh_model.interactive_mode = True
    logger.info("************* convert to frontend graph *************")
    xh_model.convert_to_fronted_graph(data_batch)
    logger.info(f"************* Start Frontend Graph *************")
    # logger.info(str(xh_model.frontend_model.graph))
    logger.info(f"************* End Frontend Graph *************")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("************* convert to quanted graph *************")
    xh_model.convert_to_quant_graph(cfg.target_device)

    # logger.info(f"************* Start Quanted Graph *************")
    # logger.info(str(xh_model.quanted_model.graph))
    # logger.info(f"************* End Quanted Graph *************")

    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()
    xh_model.to(dtype)
    xh_model.to(device)

    ## 进行PTQ量化
    logger.info("*************** Start PTQ Quantize ***************")
    calib_data = xh_model.prepare_inputs(data_batch)
    ## 将输入的List展开
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

        prefill_next_token_id, prefill_next_token_text = decode_next_token(tokenizer, quanted_aligned_logits)
        logger.info(f"Prefill Quanted Model next token: {prefill_next_token_id} {prefill_next_token_text}")
        xh_model.quanted_model.dump_quant_info_to_onnx(Path(cfg.work_dir) / f"{cfg_name}_quant_info.onnx")
    else:
        prefill_next_token_id = None

    end_time = time.time()
    if torch.cuda.is_available():
        consumption = torch.cuda.max_memory_allocated()
        unit = "B"
        if consumption > 1024:
            consumption = consumption / 1024
            unit = "k"
            if consumption > 1024:
                consumption = consumption / 1024
                unit = "M"
            consumption = round(consumption, 2)
        logger.info(f"GPU memory cost for preparation {consumption}{unit}")

    # 导出Prefill 模型
    xh_model = xh_model.to("cpu")
    data_batch["input_ids"] = data_batch["input_ids"].to("cpu")
    logger.info("*************** Start exporting prefill model ***************")
    begin_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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
    if torch.cuda.is_available():
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

    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()  # 清空导出模型，避免影响后续的导出
    logger.info(f"save prefill onnx model to {prefill_onnx_file}")
    logger.info("*************** Finished exporting prefill model ***************")

    # 导出decode 模型
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
    # input_ids = torch.concat([input_ids, prefill_next_token_id], dim=-1)
    # past_seq_len = 0
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
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

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
    end_time = time.time()
    logger.info(f"Time cost for exporting {end_time - begin_time}")
    if torch.cuda.is_available():
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

    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()  # 清空导出模型，避免影响后续的导出
    logger.info(f"save decode onnx model to {decode_onnx_file}")
    json.dump(meta_info, open(Path(cfg.work_dir) / "export_meta_info.json", "w"), indent=4)
    logger.info("*************** Finished exporting decode model ***************")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/vla/PI05/config/pi0/llm/pi05_gemma_2b_xh2a_2k_libero_mask.py",
    )
    parser.add_argument("--valid", action="store_true", help="validate the model")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()
    main(args)