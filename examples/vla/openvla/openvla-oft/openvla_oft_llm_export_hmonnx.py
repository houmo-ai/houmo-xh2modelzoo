import argparse
import json
import time
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor

from xhquant.api import (
    Config,
    ConfigDict,
    get_root_logger,
    PrecisionMode,
    ptq_quantize,
    set_random_seed,
    QuantScheme,
    DeviceType,
    create_quant_config,
    convert_fx_model_to_quanted_model,
)
from xh_model_zoo.utils.time_profiler import  TimeProfiler
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.utils import decode_next_token
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType

from xh_model_zoo.xh_llm.models.openvla_oft import XHLlamaModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/data01/home/she.gao/xh2modelzoo/examples/vla/openvla/config/openvla_oft_llm.py")
    parser.add_argument("--quant-type", type=str, default="w8a8_sefp")
    parser.add_argument("--batch", type=int, default=1)
    return parser.parse_args()


def export_onnx(xh_model, tokenizer, data_batch, out_dir, name, device, dtype):
    logger = get_root_logger()

    xh_model.to("cpu")
    torch.cuda.empty_cache()

    xh_model.convert_to_export_graph(data_batch)
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    xh_model.to(device).to(dtype)
    xh_model.set_exec_device(device)

    xh_model.to("cpu")
    torch.cuda.empty_cache()

    return xh_model.to_export_onnx(data_batch, out_dir, name)[0]


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    cfg.batch_size = args.batch
    cfg.work_dir = str(Path("./work_dirs") / Path(args.config).stem)

    Path(cfg.work_dir).mkdir(parents=True, exist_ok=True)

    logger = get_root_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exec_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16

    set_random_seed(1024)

    cfg.config_file = str(Path(cfg.work_dir) / Path(args.config).name)
    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(Path(cfg.config_file).relative_to(cfg.work_dir)),
        )
    )
    meta_info.hf_model = cfg.hf_model_dir
    hf_config_dir = Path(cfg.work_dir) / "hf_config"
    hf_config_dir.mkdir(exist_ok=True, parents=True)
    hf_config_files = [
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ]
    for cfg_file in hf_config_files:
        shutil.copyfile(
            Path(cfg.hf_model_dir) / cfg_file,
            Path(hf_config_dir) / cfg_file,
        )
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))
    
    # build model
    xh_model : XHLlamaModel = MODELS.build(cfg.model)
    tokenizer = xh_model.get_tokenizer()

    processor = AutoProcessor.from_pretrained(cfg.hf_model_dir, trust_remote_code=True)

    native_model = AutoModelForVision2Seq.from_pretrained(
        cfg.hf_model_dir,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    xh_model.init_wrap_model(native_model.language_model)
    token_embedding = xh_model.token_embedding
    del native_model
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding, str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    xh_model.set_batch_size(args.batch)
    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)
    meta_info["wrap_cfg"] = xh_model.wrap_cfg.to_dict()
    xh_model.change_eval_type(EvalModelType.WRAPED)
    xh_model.to(device).to(dtype)

    prompt = "Give me a short introduction to large language model."
    text = tokenizer(prompt, return_tensors="pt")

    data_batch = {
        "input_ids": text.input_ids.to(device),
        "past_seq_length": [0] * args.batch,
    }
    inputs = xh_model.prepare_inputs_for_graph(data_batch)

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = ConfigDict(create_quant_config(quant_scheme))
    xh_model._quanted_model = convert_fx_model_to_quanted_model(
        xh_model._wrap_model, inputs, cfg.target_device, quant_config=quant_config
    )

    # # frontend
    # xh_model.convert_to_fronted_graph(data_batch)
    # xh_model.change_eval_type(EvalModelType.FRONTEND)

    # # quant
    # xh_model.convert_to_quant_graph(cfg.target_device)
    # xh_model.change_eval_type(EvalModelType.QUANTED_DISABLED)

    calib = xh_model.prepare_inputs(data_batch)
    flat = []
    for x in calib:
        flat.extend(x) if isinstance(x, (list, tuple)) else flat.append(x)

    # with TimeProfiler("PTQ Quantize", logger):
    #     ptq_quantize(xh_model.quanted_model, [flat], PrecisionMode.ALIGNED, [exec_device])

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)

    # prefill
    prefill_dir = Path(cfg.work_dir) / "prefill_onnx"
    prefill_dir.mkdir(exist_ok=True)

    with TimeProfiler("export prefill", logger):
        prefill = export_onnx(
            xh_model, tokenizer, data_batch, str(prefill_dir), "prefill", exec_device, dtype
        )

    prefill_golden_dir = str(Path(cfg.work_dir) / "golden" / "prefill")
    if not Path(prefill_golden_dir).exists():
        logger.info(f"start export prefill model golden............")
        from xhquant.api import HMONNXGoldenInference

        prefill_model = HMONNXGoldenInference(str(Path(cfg.work_dir) / "prefill_onnx"/ "prefill.onnx"))
        prefill_model.save_golden = True
        prefill_model.exec_device = torch.device("cuda:0")

        Path(prefill_golden_dir).mkdir(exist_ok=True, parents=True)
        prefill_model.golden_dir = str(prefill_golden_dir)

        with torch.no_grad():
            prefill_model.forward(*flat)

    xh_model.release_exported_model()
    meta_info.prefill_onnx_file = str(Path(prefill).relative_to(cfg.work_dir))
    # decode
    xh_model.set_input_sequence_length(1)

    data_batch = {
        "input_ids": [[1]],
        "past_seq_length": [len(text.input_ids[0])],
    }

    decode_dir = Path(cfg.work_dir) / "decode_onnx"
    decode_dir.mkdir(exist_ok=True)

    with TimeProfiler("export decode", logger):
        decode = export_onnx(
            xh_model, tokenizer, data_batch, str(decode_dir), "decode", exec_device, dtype
        )

    decode_input = xh_model.prepare_inputs(data_batch)
    decode_inputs = []
    for x in decode_input:
        decode_inputs.extend(x) if isinstance(x, (list, tuple)) else decode_inputs.append(x)
    decode_golden_dir = str(Path(cfg.work_dir) / "golden" / "decode")
    if not Path(decode_golden_dir).exists():
        logger.info(f"start export decode model golden............")
        from xhquant.api import HMONNXGoldenInference

        decode_model = HMONNXGoldenInference(str(Path(cfg.work_dir) / "decode_onnx"/ "decode.onnx"))
        decode_model.save_golden = True
        decode_model.exec_device = torch.device("cuda:0")

        Path(decode_golden_dir).mkdir(exist_ok=True, parents=True)
        decode_model.golden_dir = str(decode_golden_dir)

        with torch.no_grad():
            decode_model.forward(*decode_inputs)

    meta_info.decode_onnx_file = str(Path(decode).relative_to(cfg.work_dir))
    
    meta_file = str(Path(cfg.work_dir) / "meta_info.json")
    json.dump(meta_info, open(meta_file, "w"), indent=4)
    logger.info(f"Save meta info to {meta_file}")

    logger.info("export finished")


if __name__ == "__main__":
    main()