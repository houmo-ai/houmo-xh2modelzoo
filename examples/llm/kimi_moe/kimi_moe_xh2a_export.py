import argparse
import gc
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import lm_eval

# from examples._init_path import _init_path  # pylint: disable=unused-import # isort:skip
import torch
import torch.fx as fx
import torch.nn as nn
from accelerate import init_empty_weights

# from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import handle_non_serializable, make_table, simple_parse_args_string
from safetensors.torch import load_file as load_safetensors_file
from torch import Tensor

# from transformers import AutoConfig, AutoModelForCausalLM, Qwen3MoeForCausalLM
from transformers.modeling_utils import no_init_weights

import xhquant.utils.suppress_printing
from xh_model_zoo.api import Config, EvalModelType, decode_next_token, get_root_logger, xhquant_llm_init
from xh_model_zoo.utils import print_gpu_info
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.kimi_moe import XHKimiMoeModel
from xh_model_zoo.xh_llm.utils import auto_offload
from xhquant.api import ConfigDict, FrontendType, Hook, PrecisionMode, QTensor, ptq_quantize, set_random_seed
from xhquant.utils.time_profiler import time_profiler


# transformers==4.51.0

silu_lut_cut_points = torch.tensor(
    [
        -41760.0,
        -11.75,
        -8.7578125,
        -5.5859375,
        -1.970703125,
        -0.281982421875,
        0.073486328125,
        1.2568359375,
        7.68359375,
        41696.0,
        65504.0,
    ]
)
silu_lut_values = torch.tensor(
    [
        -0.0,
        -9.268522262573242e-05,
        -0.00010097026824951172,
        -0.0001099705696105957,
        -0.00011980533599853516,
        -0.0001304149627685547,
        -0.00014209747314453125,
        -0.00015473365783691406,
        -0.0001684427261352539,
        -0.00018334388732910156,
        -0.0001996755599975586,
        -0.00021731853485107422,
        -0.00023663043975830078,
        -0.0002574920654296875,
        -0.00028014183044433594,
        -0.00030493736267089844,
        -0.000331878662109375,
        -0.0003612041473388672,
        -0.000392913818359375,
        -0.00042748451232910156,
        -0.0004649162292480469,
        -0.0005059242248535156,
        -0.0005502700805664062,
        -0.0005984306335449219,
        -0.0006504058837890625,
        -0.0007071495056152344,
        -0.0007691383361816406,
        -0.0008358955383300781,
        -0.0009088516235351562,
        -0.000988006591796875,
        -0.0010728836059570312,
        -0.0011663436889648438,
        -0.0012674331665039062,
        -0.001377105712890625,
        -0.00150299072265625,
        -0.00164031982421875,
        -0.0017900466918945312,
        -0.001953125,
        -0.002132415771484375,
        -0.0023250579833984375,
        -0.002536773681640625,
        -0.002765655517578125,
        -0.003017425537109375,
        -0.00328826904296875,
        -0.0035858154296875,
        -0.00390625,
        -0.0042572021484375,
        -0.004638671875,
        -0.005054473876953125,
        -0.005504608154296875,
        -0.005992889404296875,
        -0.00652313232421875,
        -0.007099151611328125,
        -0.007724761962890625,
        -0.0084075927734375,
        -0.0091400146484375,
        -0.00994110107421875,
        -0.01080322265625,
        -0.01174163818359375,
        -0.01276397705078125,
        -0.01386260986328125,
        -0.01505279541015625,
        -0.0163421630859375,
        -0.017730712890625,
        -0.0192413330078125,
        -0.0208740234375,
        -0.02288818359375,
        -0.02508544921875,
        -0.0274658203125,
        -0.0300750732421875,
        -0.03289794921875,
        -0.035980224609375,
        -0.039337158203125,
        -0.04296875,
        -0.046875,
        -0.051116943359375,
        -0.05572509765625,
        -0.0606689453125,
        -0.06597900390625,
        -0.07171630859375,
        -0.077880859375,
        -0.08447265625,
        -0.09149169921875,
        -0.09893798828125,
        -0.10693359375,
        -0.1153564453125,
        -0.124267578125,
        -0.133544921875,
        -0.1434326171875,
        -0.153564453125,
        -0.1641845703125,
        -0.175048828125,
        -0.1861572265625,
        -0.1973876953125,
        -0.2086181640625,
        -0.2197265625,
        -0.2305908203125,
        -0.2410888671875,
        -0.2457275390625,
        -0.250244140625,
        -0.25439453125,
        -0.25830078125,
        -0.26220703125,
        -0.265625,
        -0.268798828125,
        -0.271484375,
        -0.27392578125,
        -0.275634765625,
        -0.277099609375,
        -0.278076171875,
        -0.278564453125,
        -0.2783203125,
        -0.27734375,
        -0.27587890625,
        -0.2734375,
        -0.270263671875,
        -0.266357421875,
        -0.26171875,
        -0.256103515625,
        -0.2493896484375,
        -0.24169921875,
        -0.2330322265625,
        -0.2232666015625,
        -0.2122802734375,
        -0.2001953125,
        -0.18701171875,
        -0.1724853515625,
        -0.15673828125,
        -0.1396484375,
        -0.1212158203125,
        -0.1171875,
        -0.11309814453125,
        -0.10894775390625,
        -0.104736328125,
        -0.1004638671875,
        -0.09613037109375,
        -0.09173583984375,
        -0.0872802734375,
        -0.082763671875,
        -0.07818603515625,
        -0.07354736328125,
        -0.06884765625,
        -0.0640869140625,
        -0.059234619140625,
        -0.054351806640625,
        -0.049407958984375,
        -0.044403076171875,
        -0.039337158203125,
        -0.034210205078125,
        -0.0290069580078125,
        -0.0237579345703125,
        -0.0184478759765625,
        -0.01306915283203125,
        -0.00762939453125,
        -0.002132415771484375,
        0.003429412841796875,
        0.00905609130859375,
        0.014739990234375,
        0.020477294921875,
        0.0262908935546875,
        0.03216552734375,
        0.0380859375,
        0.05828857421875,
        0.07916259765625,
        0.1007080078125,
        0.1229248046875,
        0.145751953125,
        0.1693115234375,
        0.1934814453125,
        0.2183837890625,
        0.243896484375,
        0.27001953125,
        0.296630859375,
        0.323974609375,
        0.35205078125,
        0.380615234375,
        0.40966796875,
        0.439208984375,
        0.469482421875,
        0.5,
        0.53173828125,
        0.5634765625,
        0.595703125,
        0.62841796875,
        0.66162109375,
        0.6953125,
        0.72900390625,
        0.763671875,
        0.79833984375,
        0.833984375,
        0.86962890625,
        0.9052734375,
        0.94189453125,
        0.978515625,
        1.1826171875,
        1.3935546875,
        1.6083984375,
        1.8271484375,
        2.046875,
        2.267578125,
        2.48828125,
        2.708984375,
        2.927734375,
        3.14453125,
        3.361328125,
        3.576171875,
        3.7890625,
        4.0,
        4.2109375,
        4.41796875,
        4.62890625,
        4.8359375,
        5.04296875,
        5.24609375,
        5.453125,
        5.65625,
        5.859375,
        6.0625,
        6.265625,
        6.46875,
        6.671875,
        6.875,
        7.07421875,
        7.27734375,
        7.48046875,
        7.6796875,
        1310.0,
        2614.0,
        3916.0,
        5220.0,
        6520.0,
        7824.0,
        9128.0,
        10432.0,
        11736.0,
        13032.0,
        14336.0,
        15640.0,
        16944.0,
        18240.0,
        19552.0,
        20848.0,
        22160.0,
        23456.0,
        24768.0,
        26064.0,
        27360.0,
        28672.0,
        29968.0,
        31280.0,
        32576.0,
        33888.0,
        35168.0,
        36480.0,
        37792.0,
        39104.0,
        40384.0,
        41696.0,
        65504.0,
    ]
)
silu_lut_scale = torch.tensor(
    [
        2.396106719970703e-05,
        10.6953125,
        10.0859375,
        8.8515625,
        18.953125,
        90.0,
        27.046875,
        4.98046875,
        0.0007677078247070312,
        4.202127456665039e-05,
    ]
)


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


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kimi/3b_30b/kimi_a3b_30b_instruct_legacy_xh2a_2k_batch.py",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--valid", action="store_true", help="validate the model")
    return parser


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
    torch.cuda.empty_cache()
    print_gpu_info(logger)
    xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exporting...")

    logger.info(f"************* Start Exported Graph *************")
    # logger.info(str(xh_model.exported_model.graph))
    logger.info(f"************* End Exported Graph *************")
    torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    if valid:
        xh_model.to(device)
        xh_model.to(dtype)
        data_batch["input_ids"] = data_batch["input_ids"].to(device)
        with torch.no_grad():
            outs = xh_model.test_step(data_batch)
            exported_logits = outs.logits.detach()
            next_tokens, next_token_str = decode_next_token(tokenizer, exported_logits)
        logger.info(f"Exported model next token: {next_tokens} {next_token_str}")

    xh_model.to("cpu")  # 切换到cpu上进行模型导出
    torch.cuda.empty_cache()
    print_gpu_info(logger)
    logger.info("*************** Start exporting onnx ***************")
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


def main():
    parser = parse_arguments()
    args = parser.parse_args()
    begin_time = time.time()
    torch.cuda.reset_peak_memory_stats()

    cfg = Config.fromfile(args.config)
    cfg_name = Path(args.config).stem
    cfg_name = f"{cfg_name}_eval"
    cfg.work_dir = str(Path("./work_dirs") / cfg_name)

    log_file = Path(cfg.work_dir) / f"{cfg_name}.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    only_export = not args.valid  # 仅仅导出模型，不进行推理验证
    # HUGE_MODEL_EXPORT_ENABLED=1/true/yes/on 时强制把权重放 CPU，避免单卡放不下。
    is_huge_model = os.environ.get("HUGE_MODEL_EXPORT_ENABLED", "").lower() in {"1", "true", "yes", "on"}
    cfg.device = "cpu" if is_huge_model else ("cuda:0" if torch.cuda.is_available() else "cpu")

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

    xhquant_llm_init(log_file, cfg.debug)
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
    xh_model: XHKimiMoeModel = MODELS.build(cfg.model)

    tokenizer = xh_model.get_tokenizer()
    native_model = xh_model.get_hf_model("cpu")  # 加载在CPU上 : Qwen3MoeForCausalLM
    # assert isinstance(native_model, Qwen3MoeForCausalLM)

    hf_config_dir = Path(cfg.work_dir) / "hf_config"
    hf_config_dir.mkdir(exist_ok=True, parents=True)
    hf_config_files = [
        "config.json",
        "configuration.json",
        "generation_config.json",
        "tokenizer_config.json",
        # "vocab.json",
        "tokenizer_config.json",
    ]
    for cfg_file in hf_config_files:
        shutil.copyfile(
            Path(hf_model_dir) / cfg_file,
            Path(hf_config_dir) / cfg_file,
        )
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    # 融合GPTQ权重
    prompt = "你多大了？用中文回答。"
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    input_ids = model_inputs.input_ids

    native_model.half()
    if False:
        generated_ids = native_model.generate(inputs=input_ids.cpu(), max_new_tokens=500)
        response = tokenizer.batch_decode(generated_ids)[0]

    resume_from = cfg.get("resume_from", None)
    if resume_from is not None:
        # native_model.to("cpu")
        archive_file = cfg.resume_from
        logger.info(f"Load previously saved checkpoint from: {archive_file}")
        is_safetensors = archive_file.endswith(".safetensors")
        state_dict = None
        if is_safetensors:
            state_dict = load_safetensors_file(archive_file, device="cpu")
        else:
            state_dict = torch.load(archive_file, weights_only=True, map_location="cpu")

        model_state_dict = native_model.state_dict()
        unexpect_state_dict = []
        for k, v in state_dict.items():
            if k not in model_state_dict:
                unexpect_state_dict.append(k)

        for k in unexpect_state_dict:
            paths = k.split(".")
            if paths[-1] == "quant_weight":
                submodule_name = ".".join(paths[:-1])
                submodule = native_model.get_submodule(submodule_name)
                # submodule = get_submodule(native_model, k)
                v = state_dict[k]
                if v.min().item() >= -pow(2, 7) and v.max().item() <= pow(2, 7) - 1:
                    v = v.to(torch.int8)
                elif v.min().item() >= -pow(2, 15) and v.max().item() <= pow(2, 15) - 1:
                    v = v.to(torch.int16)
                else:
                    v = v.to(torch.float32)
                submodule.register_buffer("quant_weight", v, persistent=False)
                logger.info(f"add quant_weight to {submodule_name}")
            else:
                logger.warning(f"ignore unexpect state dict: {k}")
            state_dict.pop(k)

        native_model.load_state_dict(state_dict)
        del state_dict

    xh_model.init_wrap_model(native_model)
    del native_model

    token_embedding = xh_model.token_embedding
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)

    assert xh_model.use_cache, "xh_model must use cache"
    data_batch = {
        "input_ids": input_ids.to(device),
        "past_seq_length": [0],
        "act_lut_cut_points": silu_lut_cut_points.to(device),
        "act_lut_values": silu_lut_values.to(device),
        "act_lut_scale": silu_lut_scale.to(device),
    }

    if device.type != "cuda":
        hook = PingPangGPUHook("PingPangGPU", exec_device)  ## CPU和GPU之间乒乓方式执行
        xh_model.register_hook(hook)

    if True:
        logger.info("Start wrap model prefill .......................")
        with torch.no_grad():
            xh_model.half()
            outs = xh_model.test_step(data_batch)
            wraped_logits = outs.logits.detach()

        wraped_logits = wraped_logits.squeeze(1)
        next_tokens = torch.argmax(wraped_logits, dim=-1)
        # next_tokens = next_tokens.unsqueeze(0)
        next_token_str = tokenizer.batch_decode(next_tokens, skip_special_tokens=True)[0]
        logger.info(f"wraped_logits next token: {next_token_str}")

        print_gpu_info(logger)

    xh_model.interactive_mode = True
    logger.info("************* convert to frontend graph *************")
    xh_model.convert_to_fronted_graph(data_batch)
    logger.info(f"************* Start Frontend Graph *************")
    # logger.info(str(xh_model.frontend_model.graph))
    logger.info(f"************* End Frontend Graph *************")
    torch.cuda.empty_cache()

    logger.info("************* convert to quanted graph *************")
    xh_model.convert_to_quant_graph(cfg.target_device)

    logger.info(f"************* Start Quanted Graph *************")
    # logger.info(str(xh_model.quanted_model.graph))
    logger.info(f"************* End Quanted Graph *************")

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
    if False:
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
    data_batch["act_lut_cut_points"] = data_batch["act_lut_cut_points"].to("cpu")
    data_batch["act_lut_values"] = data_batch["act_lut_values"].to("cpu")
    data_batch["act_lut_scale"] = data_batch["act_lut_scale"].to("cpu")

    logger.info("*************** Start exporting prefill model ***************")
    begin_time = time.time()
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
    # xh_model.to(device)
    xh_model.to(dtype)
    data_batch["input_ids"] = data_batch["input_ids"].to(device)
    data_batch["act_lut_cut_points"] = data_batch["act_lut_cut_points"].to(device)
    data_batch["act_lut_values"] = data_batch["act_lut_values"].to(device)
    data_batch["act_lut_scale"] = data_batch["act_lut_scale"].to(device)

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
        "act_lut_cut_points": silu_lut_cut_points.to(device),
        "act_lut_values": silu_lut_values.to(device),
        "act_lut_scale": silu_lut_scale.to(device),
    }
    if not only_export:
        with torch.no_grad():
            outs = xh_model.test_step(data_batch)
            decode_logits = outs.logits.detach()
        decode_token_id, decode_token_text = decode_next_token(tokenizer, decode_logits)
        logger.info(f"Decode on Quanted Model next token: {decode_token_id} {decode_token_text}")

    torch.cuda.empty_cache()

    logger.info("*************** Start exporting decode model ***************")
    xh_model = xh_model.to("cpu")
    data_batch["input_ids"] = data_batch["input_ids"].to("cpu")
    data_batch["act_lut_cut_points"] = data_batch["act_lut_cut_points"].to("cpu")
    data_batch["act_lut_values"] = data_batch["act_lut_values"].to("cpu")
    data_batch["act_lut_scale"] = data_batch["act_lut_scale"].to("cpu")

    begin_time = time.time()
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
    main()
