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
from xh_model_zoo.xh_llm.models.xvla.xvla_llm_model import XHFlorence2EncoderLLMModel
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType


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
    prefill_onnx_dir = Path(cfg.work_dir) / "prefill_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    logger = get_root_logger()
    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.dump(config_file)

    device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)
    xh_model = MODELS.build(cfg.model)
    policy = xh_model.get_hf_model()
    assert isinstance(xh_model, XHFlorence2EncoderLLMModel), f"Model must be XHFlorence2EncoderLLMModel, but got {type(xh_model)}"

    xh_model.init_wrap_model(policy)
    tokenizer = xh_model.get_tokenizer()

    onnx_dir = Path(cfg.work_dir) / "onnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(config_file.relative_to(cfg.work_dir)),
        )
    )
    meta_info["wrap_cfg"] = xh_model.wrap_cfg.to_dict()
    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir
    
    hf_model_config_dir = cfg.config_dir if hasattr(cfg, 'config_dir') else hf_model_dir

    hf_config_dir = Path(cfg.work_dir) / "hf_config"
    hf_config_dir.mkdir(exist_ok=True, parents=True)
    hf_config_files = [
        "config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "tokenizer.model",
    ]
    for cfg_file in hf_config_files:
        src_file = Path(hf_model_config_dir) / cfg_file
        if src_file.exists():
            shutil.copyfile(
                src_file,
                Path(hf_config_dir) / cfg_file,
            )
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    token_embedding = xh_model.token_embedding
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    input_seq_len = xh_model.wrap_cfg.get("input_sequence_length", 100)
    # Dummy input_ids to generate embeddings
    input_ids = torch.randint(0, 1000, (1, input_seq_len), device=device)
    inputs_embeds = token_embedding(input_ids).to(dtype)
    # Correct attention mask shape for Florence2Encoder if needed
    # attention_mask = torch.ones((1, input_seq_len), device=device, dtype=dtype)

    data_batch = {
        "inputs_embeds": inputs_embeds,
    }

    # 设置导出配置
    if xh_model.export_cfg is None:
        xh_model.export_cfg = ConfigDict()
    
    # 显式设置 input_names，匹配 prepare_inputs 返回的有效输入 (非 None)
    # 根据 prepare_inputs: (None, attention_mask, None, inputs_embeds)
    # 对应的输入应该是 attention_mask 和 inputs_embeds
    xh_model.export_cfg.input_names = ["inputs_embeds"]
    xh_model.export_cfg.output_names = ["last_hidden_state"]

    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
    xh_model.to(device)
    xh_model.to(dtype)
    temp = xh_model._wrap_model(inputs_embeds)
    logger.info("Start wrap model prefill .......................")

    logger.info("************* convert to frontend graph *************")
    xh_model.convert_to_fronted_graph([inputs_embeds])
    logger.info(f"************* Start Frontend Graph *************")
    # logger.info(str(xh_model.frontend_model.graph))
    logger.info(f"************* End Frontend Graph *************")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("************* convert to quanted graph *************")
    xh_model.convert_to_quant_graph(cfg.target_device)

    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()
    xh_model.to(dtype)
    xh_model.to(device)

    ## 进行PTQ量化
    logger.info("*************** Start PTQ Quantize ***************")
    calib_data = xh_model.prepare_inputs(data_batch)
    ptq_quantize(xh_model.quanted_model, [calib_data[3]], PrecisionMode.ALIGNED, [exec_device])
    logger.info("*************** Finished PTQ Quantize ***************")

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

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
    data_batch["inputs_embeds"] = data_batch["inputs_embeds"].to("cpu")
    logger.info("*************** Start exporting prefill model ***************")
    begin_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    prefill_onnx_file = xhmodel_export_onnx(
        xh_model,
        tokenizer,
        [data_batch["inputs_embeds"]],
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

    # 保存 meta info
    meta_file = Path(cfg.work_dir) / "meta_info.json"
    with open(meta_file, "w") as f:
        json.dump(meta_info.to_dict(), f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="examples/vla/xvla/xvla_florence_09b_llm_xh2a_2k_libero_onnx.py",
    )
    parser.add_argument("--valid", action="store_true", help="validate the model")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()
    main(args)
