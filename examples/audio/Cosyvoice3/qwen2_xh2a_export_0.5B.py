import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any, List, Tuple

import accelerate.hooks
import torch
import torch.fx as fx
import torch.nn as nn
import xhquant.utils.suppress_printing
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor
from xhquant.api import (
    Config, 
    ConfigDict, 
    FrontendType, 
    Hook, 
    PrecisionMode, 
    QTensor, 
    ptq_quantize, 
    set_random_seed, 
    get_root_logger, 
    convert_fx_model_to_quanted_model, 
    QuantScheme, 
    DeviceType,
    create_quant_config,
)

from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.base_llm_model import LLMBaseModel
from xh_model_zoo.xh_llm.models.cosyvoice3 import XHQwen2LegacyModel
from xh_model_zoo_develop.utils.cpu_gpu_utils import print_gpu_info
from xh_model_zoo.utils.time_profiler import time_profiler
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.utils import decode_next_token


def to_device(inputs, device):
    if isinstance(inputs, Tensor):
        return inputs.to(device)
    elif isinstance(inputs, (list, tuple)):
        return type(inputs)([to_device(x, device) for x in inputs])
    elif isinstance(inputs, dict):
        return {k: to_device(v, device) for k, v in inputs.items()}
    elif isinstance(inputs, QTensor):
        return inputs.to(device)
    else:
        return inputs


def xhmodel_export_onnx(
    xh_model: LLMBaseModel,
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
    print_gpu_info(logger)
    xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exporting...")

    logger.info(f"************* Start Exported Graph *************")
    # logger.info(str(xh_model.exported_model.graph))
    logger.info(f"************* End Exported Graph *************")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    if valid:
        xh_model.to(device)
        xh_model.to(dtype)
        data_batch["input_ids"] = data_batch["input_ids"].to(device)
        with torch.no_grad():
            outs = xh_model.test_step(data_batch)
            exported_logits = outs.logits.detach()

            exported_logits = exported_logits.squeeze(1)
            next_tokens = torch.argmax(exported_logits, dim=-1)
            next_tokens = next_tokens.unsqueeze(0)
            next_token_str = tokenizer.batch_decode(next_tokens, skip_special_tokens=True)[0]
        logger.info(f"Exported model next token: {next_tokens} {next_token_str}")

    xh_model.to("cpu")  # 切换到cpu上进行模型导出
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print_gpu_info(logger)
    logger.info("*************** Start exporting onnx ***************")
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/config/qwen2_05b/qwen2_05b_instruct_xh2a_2k.py",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--valid", action="store_true", help="validate the model")
    parser.add_argument("--prompt", type=str, default="你多大了？用中文回答。")
    # 新增量化类型命令行参数
    parser.add_argument(
        "--quant-type",
        type=str,
        default="w8a16_sefp",
        help="Quantization type (e.g., w8a16_sefp)"
    )
    return parser


def main(args):
    begin_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # 简化 cfg_name 赋值
    cfg = Config.fromfile(args.config)
    cfg_name = Path(args.config).stem
    cfg.work_dir = str(Path("./hmonnx") / cfg_name)

    # 简化工作目录创建逻辑
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(exist_ok=True, parents=True)
    log_file = work_dir / f"{cfg_name}_debug.log"

    # 设备配置简化
    is_big_model = cfg.get("is_big_model", False)
    only_export = not args.valid
    cfg.device = "cpu" if is_big_model else ("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg.exec_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"
    cfg.debug = args.debug
    # 补充默认配置
    for key in ["quarot", "gptq"]:
        if key not in cfg:
            cfg[key] = False

    # 调试目录创建
    (work_dir / "debug").mkdir(exist_ok=True, parents=True)

    # 初始化日志和随机种子
    set_random_seed(cfg.get("seed", 1024))
    logger = get_root_logger()
    logger.info(f"Config:\n{cfg.pretty_text}")
    cfg.dump(work_dir / Path(args.config).name)

    xhquant.utils.suppress_printing.disable_printing = True

    # 检查量化模型路径
    if cfg.quarot or cfg.gptq:
        assert cfg.resume_from and Path(cfg.resume_from).exists(), "resume_from must be valid path"

    # 设备和数据类型设置
    device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)

    # 创建ONNX输出目录
    prefill_onnx_dir = work_dir / "prefill_onnx"
    decode_onnx_dir = work_dir / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)

    # 元信息初始化
    meta_info = ConfigDict({
        "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "config": str(Path(args.config).name),
        "model_name": cfg_name,
        "wrap_cfg": cfg.model.wrap_cfg.to_dict()
    })

    # 模型和分词器加载
    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir
    xh_model: XHQwen2LegacyModel = MODELS.build(cfg.model)
    tokenizer = xh_model.get_tokenizer()
    native_model = xh_model.get_hf_model("cpu")

    # 复制HF配置文件
    hf_config_dir = work_dir / "hf_config"
    hf_config_dir.mkdir(exist_ok=True, parents=True)
    for cfg_file in ["config.json", "generation_config.json", "tokenizer_config.json", 
                    "vocab.json", "tokenizer.json", "chat_template.jinja", "added_tokens.json"]:
        src = Path(hf_model_dir) / cfg_file
        if src.exists():
            shutil.copyfile(src, hf_config_dir / cfg_file)
    meta_info.hf_config = str(hf_config_dir.relative_to(work_dir))

    # 输入数据准备
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer([text], return_tensors="pt").input_ids.to(device)

    # 模型权重加载
    archive_file = "/data01/home/she.gao/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/llm.pt"
    xh_model.load_wraped_model_state_dict_prefix(native_model, archive_file)
    resume_from = cfg.get("resume_from", None)
    if resume_from is not None:
        xh_model.load_wraped_model_state_dict(native_model, cfg.resume_from)

    # 模型包装和嵌入保存
    xh_model.init_wrap_model(native_model)
    native_model = None  # 释放内存
    token_embedding_file = work_dir / "token_embedding.pt"
    torch.save(xh_model.token_embedding.state_dict(), token_embedding_file)
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(work_dir))

    # KV缓存元信息
    if xh_model.past_key_caches and len(xh_model.past_key_caches) > 0:
        meta_info.update({
            "use_cache": True,
            "kv_cache_shape": xh_model.past_key_caches[0].shape,
            "num_hidden_layers": len(xh_model.past_key_caches)
        })

    # 设备钩子设置（保持原逻辑）
    xh_model.change_eval_type(EvalModelType.WRAPED)
    wraped_model: nn.Module = xh_model.wrap_model

    def pre_hook(module, inputs):
        if isinstance(module, nn.Linear):
            module.to(exec_device)
            return to_device(inputs, exec_device)
        return inputs

    def post_hook(module, inputs, outputs):
        if isinstance(module, nn.Linear):
            module.to("cpu")
            return to_device(outputs, "cpu")
        return outputs

    if device != exec_device:
        for module in wraped_model.modules():
            if isinstance(module, nn.Linear):
                module.register_forward_pre_hook(pre_hook)
                module.register_forward_hook(post_hook)
    else:
        wraped_model.to(device)

    # 量化配置（使用命令行传入的quant_type）
    xh_model.to(device).to(dtype)
    data_batch = {"input_ids": input_ids, "past_seq_length": 0}
    inputs = xh_model.prepare_inputs_for_graph(data_batch)

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = ConfigDict(create_quant_config(quant_scheme))
    xh_model._quanted_model = convert_fx_model_to_quanted_model(
        xh_model._wrap_model, inputs, cfg.target_device, quant_config=quant_config
    )

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

    print_gpu_info(logger)
    if not only_export:
        with torch.no_grad():
            with time_profiler() as t:
                outs = xh_model.test_step(data_batch)
            logger.info(f"QUANTED_ALIGNED: {t():.04f}")
            quanted_aligned_logits = outs.logits.detach()

        prefill_next_token_id, prefill_next_token_text = decode_next_token(tokenizer, quanted_aligned_logits)
        logger.info(f"Prefill Quanted Model next token: {prefill_next_token_id} {prefill_next_token_text}")
        xh_model.quanted_model.dump_quant_info_to_onnx(Path(cfg.work_dir) / f"{cfg_name}_quant_info.onnx")
    else:
        prefill_next_token_id = None

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

    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()  # 清空导出模型，避免影响后续的导出
    logger.info(f"save prefill onnx model to {prefill_onnx_file}")
    logger.info("*************** Finished exporting prefill model ***************")
    print_gpu_info(logger)

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
        "past_seq_length": past_seq_len,
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

    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()  # 清空导出模型，避免影响后续的导出
    logger.info(f"save decode onnx model to {decode_onnx_file}")
    json.dump(meta_info, open(Path(cfg.work_dir) / "meta_info.json", "w"), indent=4)
    logger.info("*************** Finished exporting decode model ***************")


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)
