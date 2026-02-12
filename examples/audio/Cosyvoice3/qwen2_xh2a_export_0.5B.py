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
from xhquant.api import Config, ConfigDict, FrontendType, Hook, PrecisionMode, QTensor, ptq_quantize, set_random_seed, get_root_logger

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


class SnapshothHook(Hook):
    def __init__(self, name) -> None:
        super().__init__()
        self.name = name

    def after_node_run(self, module, graph, node, output, args, kwargs) -> None:
        if isinstance(output, Tensor):
            output = output.detach()
            max_val = output.max()
            min_val = output.min()
            if max_val > 65504 or min_val < -65504:
                print(f"output {node.name} has max_val {max_val} and min_val {min_val} in {self.name}")
        return output

    def before_node_run(self, module, graph, node, args, kwargs) -> None:
        return args, kwargs


class PingPangGPUHook(Hook):
    def __init__(self, name, device) -> None:
        super().__init__()
        self.name = name
        self.device = device
        torch.cuda.set_per_process_memory_fraction(0.8, device)
        torch.cuda.max_split_size_mb = 128

    def after_node_run(self, graph_module: fx.GraphModule, graph: fx.Graph, node: fx.Node, output, args, kwargs) -> Any:
        module = None
        if node.op == "call_module":
            module = graph_module.get_submodule(node.target)
            module.to("cpu")
        # args = to_device(args, "cpu")
        # kwargs = to_device(kwargs, "cpu")
        # output = to_device(output, "cpu")

        if module is not None:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

        return output

    def before_node_run(
        self, graph_module: fx.GraphModule, graph: fx.Graph, node: fx.Node, args, kwargs
    ) -> Tuple[Any, Any]:
        if node.op == "call_module":
            module = graph_module.get_submodule(node.target)
            module.to(self.device)
        if node.op not in [
            "placeholder",
            "getattr",
        ]:
            args = to_device(args, self.device)
            kwargs = to_device(kwargs, self.device)
        return args, kwargs


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


def main(args):
    begin_time = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    cfg = Config.fromfile(args.config)
    cfg_name = Path(args.config).stem
    cfg_name = f"{cfg_name}"
    cfg.work_dir = str(Path("./work_dirs") / cfg_name)

    log_file = Path(cfg.work_dir) / f"{cfg_name}_debug.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    is_big_model = cfg.get("is_big_model", False)
    only_export = not args.valid  # 仅仅导出模型，不进行推理验证
    if is_big_model:
        cfg.device = "cpu"  # 放至模型权重的设备， 如果模型过大，单个GPU放不下，就设置为CPU

    cfg.dtype = "float16"
    cfg.debug = args.debug
    cfg.exec_device = (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # 执行设备，执行某个Module或者op时，再将数据搬到这个设备上
    if "quarot" not in cfg:
        cfg.quarot = False
    if "gptq" not in cfg:
        cfg.gptq = False

    debug_output_dir = Path(cfg.work_dir) / "debug"
    debug_output_dir.mkdir(exist_ok=True, parents=True)

    seed = cfg.get("seed", 1024)
    set_random_seed(seed)
    logger = get_root_logger()

    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.dump(config_file)

    xhquant.utils.suppress_printing.disable_printing = True  # 屏蔽不必要的打印信息

    if cfg.quarot or cfg.gptq:
        assert cfg.resume_from is not None, "resume_from must be set"
        assert Path(cfg.resume_from).exists(), f"resume_from {cfg.resume_from} not exists"

    device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)

    prefill_onnx_dir = Path(cfg.work_dir) / "prefill_onnx"
    decode_onnx_dir = Path(cfg.work_dir) / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)

    frontend_type = cfg.get("frontend_type", "TorchFX")
    frontend_type = FrontendType(frontend_type)

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(config_file.relative_to(cfg.work_dir)),
        )
    )
    meta_info["model_name"] = cfg_name
    meta_info["wrap_cfg"] = cfg.model.wrap_cfg.to_dict()

    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir
    xh_model: XHQwen2LegacyModel = MODELS.build(cfg.model)  # type: ignore

    tokenizer = xh_model.get_tokenizer()
    native_model = xh_model.get_hf_model("cpu")

    hf_config_dir = Path(cfg.work_dir) / "hf_config"
    hf_config_dir.mkdir(exist_ok=True, parents=True)
    hf_config_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "vocab.json",
        "tokenizer.json",
        "chat_template.jinja",
        "added_tokens.json",
    ]
    for cfg_file in hf_config_files:
        src_file = Path(hf_model_dir) / cfg_file
        dst_file = Path(hf_config_dir) / cfg_file
        if src_file.exists():
            shutil.copyfile(
                src_file,
                dst_file,
            )
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    prompt = args.prompt
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    input_ids = model_inputs.input_ids
    
    archive_file = "/data01/home/she.gao/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/llm.pt"
    xh_model.load_wraped_model_state_dict_prefix(native_model, archive_file)
    resume_from = cfg.get("resume_from", None)
    if resume_from is not None:
        # native_model.to("cpu")
        archive_file = cfg.resume_from
        xh_model.load_wraped_model_state_dict(native_model, archive_file)

    # 输出结果：最后一个token的logits
    if False:
        native_model.to(device)
        inputs_embeds = native_model.model.embed_tokens(input_ids)
        with torch.no_grad():
            outs = native_model(inputs_embeds=inputs_embeds, use_cache=False, num_logits_to_keep=1)
            logits_gt = outs.logits.detach()
            outs = None
        next_token_id, next_token_text = decode_next_token(tokenizer, logits_gt)
        logger.info(f"HF Model next token: {next_token_text}")
        native_model.to("cpu")
        torch.cuda.empty_cache()

    # 构建模型，并做Module替换，将原生Module替换为可以fx.trace的Module

    xh_model.init_wrap_model(native_model)
    native_model = None

    # xh_model.load_quant_

    token_embedding = xh_model.token_embedding
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)

    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
    wraped_model: nn.Module = xh_model.wrap_model

    def pre_hook(module, inputs):
        # logger.info(f"pre hook {module}")
        if not isinstance(module, nn.Linear):
            return inputs
        logger.info(f"move {type(module)} to {exec_device}")
        module.to(exec_device)
        inputs = to_device(inputs, exec_device)
        return inputs

    def post_hook(module, inputs, outputs):
        # logger.info(f"post hook {module}")
        if not isinstance(module, nn.Linear):
            return outputs
        module.to("cpu")
        outputs = to_device(outputs, "cpu")
        inputs = to_device(inputs, "cpu")
        # torch.cuda.empty_cache()
        return outputs

    if device != exec_device:
        for name, module in wraped_model.named_modules():
            if isinstance(module, nn.Linear):
                module.register_forward_pre_hook(pre_hook)
                module.register_forward_hook(post_hook)
    else:
        wraped_model.to(device)

    xh_model.to(device)
    xh_model.to(dtype)
    assert xh_model.use_cache, "xh_model must use cache"
    data_batch = {
        "input_ids": input_ids.to(device),
        "past_seq_length": 0,
    }

    # if device.type != "cuda":
    #     hook = PingPangGPUHook("PingPangGPU", exec_device)  ## CPU和GPU之间乒乓方式执行
    #     xh_model.register_hook(hook)

    if False:
        logger.info("Start wrap model prefill .......................")
        with torch.no_grad():
            outs = xh_model.test_step(data_batch)
            wraped_logits = outs.logits.detach()

        wraped_logits = wraped_logits.squeeze(1)
        next_tokens = torch.argmax(wraped_logits, dim=-1)
        next_tokens = next_tokens.unsqueeze(0)
        next_token_str = tokenizer.batch_decode(next_tokens, skip_special_tokens=True)[0]
        logger.info(f"wraped_logits next token: {next_token_str}")

        print_gpu_info(logger)

    if device != exec_device:
        for name, module in wraped_model.named_modules():
            if isinstance(module, nn.Linear):
                module._forward_pre_hooks = dict()
                module._forward_hooks = dict()

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

    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()
    xh_model.to(dtype)
    xh_model.to(device)
    print_gpu_info(logger)

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
    with time_profiler() as t:
        ptq_quantize(xh_model.quanted_model, [calib_data], PrecisionMode.ALIGNED, [exec_device])
        logger.info(f"PTQ Quantize time: {t():.04f} s")
    logger.info("*************** Finished PTQ Quantize ***************")

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

    end_time = time.time()
    logger.info(f"Time cost for preparation {end_time - begin_time}")
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
    json.dump(meta_info, open(Path(cfg.work_dir) / "meta_info.json", "w"), indent=4)
    logger.info("*************** Finished exporting decode model ***************")


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
    # parser.add_argument("--generate", action="store_true", help="generate response")
    return parser


if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)
