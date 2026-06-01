# 该示例展示了如何导出 Qwen3 TTS Talker 模型
import argparse
import json
import time
from pathlib import Path
from typing import cast

import soundfile as sf
import torch

from xhquant.api import Config, ConfigDict, PrecisionMode, ptq_quantize, set_random_seed
from xhquant.utils.time_profiler import time_profiler
from xh_model_zoo.api import get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.base_llm_model import LLMBaseModel
from xh_model_zoo.xh_llm.models.qwen3_tts import (
    XHQwen3TTSCodePredictor,
    XHQwen3TTSModel,
    build_qwen3_tts_code_predictor_hf_compatible,
)


def xhmodel_export_onnx(
    xh_model: LLMBaseModel,
    data_batch,
    onnx_output_dir: str,
    cfg_name,
    logger,
):
    logger.info("Start exporting...")
    xh_model.to("cpu")  # 切换到cpu上进行模型导出
    torch.cuda.empty_cache()
    # print_gpu_info(logger)
    xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exporting...")

    logger.info("************* Start Exported Graph *************")
    # logger.info(str(xh_model.exported_model.graph))
    logger.info("************* End Exported Graph *************")
    torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    xh_model.to("cpu")  # 切换到cpu上进行模型导出
    torch.cuda.empty_cache()
    logger.info("*************** Start exporting onnx ***************")
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


def _export_impl(cfg: Config, args: argparse.Namespace):
    exec_device = cfg.exec_device
    dtype = getattr(torch, cfg.dtype)
    logger = get_root_logger()
    work_dir = cfg.work_dir
    xh_model = MODELS.build(cfg.model)
    assert isinstance(xh_model, XHQwen3TTSCodePredictor)

    config_file = cfg.config_file
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

    # 获取原浮点模型
    hf_model = xh_model.get_hf_model(device_map="cpu", dtype=torch.float16)
    assert isinstance(hf_model, XHQwen3TTSModel)
    hf_model = cast(XHQwen3TTSModel, hf_model)
    feature_dim = None

    def _hook(self, args, kwargs):
        nonlocal feature_dim
        inputs_embeds = kwargs.get("inputs_embeds", None)
        feature_dim = inputs_embeds.shape[-1] if inputs_embeds is not None else None
        logger.info(f"feature_dim: {feature_dim}")
        logger.info(f"inputs_embeds: {inputs_embeds.shape}")
        raise RuntimeError("Stop forward after getting feature_dim for export")

    talker_hook = hf_model.model.talker.code_predictor.register_forward_pre_hook(_hook, with_kwargs=True)
    try:
        wavs, sr = hf_model.generate_voice_design(
            text="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地",
            language="Chinese",
            instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
        )
    except RuntimeError:
        pass
    talker_hook.remove()
    assert feature_dim is not None, "Failed to get feature_dim from the model, export cannot proceed without it."

    token_embedding = hf_model.model.talker.code_predictor.get_input_embeddings()
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    meta_file = Path(work_dir) / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)

    xh_model.init_wrap_model(hf_model)
    xh_model.to(dtype=dtype)
    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)

    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)

    del hf_model

    meta_file = Path(work_dir) / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)

    input_sequence_length = xh_model.wrap_cfg.input_sequence_length
    data_batch = {
        "inputs_embeds": torch.randn(1, input_sequence_length, feature_dim),
        "past_seq_length": 0,
        "generate_steps": 0,
    }

    logger.info("************* convert to frontend graph *************")
    xh_model.convert_to_fronted_graph(data_batch)
    # logger.info("************* Start Frontend Graph *************")
    # logger.info(str(xh_model.frontend_model.graph))
    # logger.info("************* End Frontend Graph *************")
    torch.cuda.empty_cache()
    xh_model.change_eval_type(eval_type=EvalModelType.FRONTEND)

    logger.info("************* convert to quanted graph *************")
    xh_model.convert_to_quant_graph(cfg.target_device)

    # logger.info("************* Start Quanted Graph *************")
    # # logger.info(str(xh_model.quanted_model.graph))
    # logger.info("************* End Quanted Graph *************")

    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()

    ## 进行PTQ量化
    logger.info("*************** Start PTQ Quantize ***************")
    calib_data = xh_model.prepare_inputs(data_batch)
    ## 将输入的List展开
    new_args = []
    for arg in calib_data:
        if isinstance(arg, (list, tuple)):
            new_args.extend(arg)
        else:
            new_args.append(arg)
    calib_data = new_args
    with time_profiler() as t:
        ptq_quantize(
            xh_model.quanted_model,
            [calib_data],
            PrecisionMode.ALIGNED,
            [exec_device],
            auto_release_unused_parameters=True,
        )
        logger.info(f"PTQ Quantize time: {t():.04f} s")
    logger.info("*************** Finished PTQ Quantize ***************")

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)

    if False:
        hf_model = xh_model.get_hf_model(device_map=exec_device, dtype=torch.float16)
        assert isinstance(hf_model, XHQwen3TTSModel)
        hf_model.model.to(dtype=dtype)
        hf_model.model.to(device=exec_device)

        xh_model.to(exec_device)
        xh_model.to(dtype)

        hf_compatible_model = build_qwen3_tts_code_predictor_hf_compatible(hf_model, xh_model)
        wavs, sr = hf_compatible_model.generate_voice_design(
            text="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地",
            language="Chinese",
            instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
        )
        out_file = Path(work_dir) / "output_voice_design.wav"
        sf.write(out_file, wavs[0], sr)
        logger.info(f"Audio saved to {out_file}")
    logger.info("*************** Start exporting prefill model ***************")

    prefill_onnx_dir = Path(cfg.work_dir) / "prefill_onnx"
    decode_onnx_dir = Path(cfg.work_dir) / "decode_onnx"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)
    prefill_onnx_file = xhmodel_export_onnx(
        xh_model,
        data_batch,
        str(prefill_onnx_dir),
        f"{cfg_name}_prefill",
        logger,
    )

    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()  # 清空导出模型，避免影响后续的导出
    logger.info(f"save prefill onnx model to {prefill_onnx_file}")
    logger.info("*************** Finished exporting prefill model ***************")

    # 导出decode 模型
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(dtype)
    data_batch = {
        "inputs_embeds": torch.randn(1, 1, feature_dim),
        "past_seq_length": 0,
        "generate_steps": 0,
    }

    torch.cuda.empty_cache()
    xh_model.set_input_sequence_length(1)

    logger.info("*************** Start exporting decode model ***************")
    xh_model.to("cpu")

    decode_onnx_file = xhmodel_export_onnx(
        xh_model,
        data_batch,
        str(decode_onnx_dir),
        f"{cfg_name}_decode",
        logger,
    )

    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
    xh_model.release_exported_model()  # 清空导出模型，避免影响后续的导出
    logger.info(f"save decode onnx model to {decode_onnx_file}")
    json.dump(meta_info, open(meta_file, "w"), indent=4)
    logger.info("*************** Finished exporting decode model ***************")


def main(args: argparse.Namespace) -> None:
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg_name = Path(args.config).stem
    log_file = Path(cfg.work_dir) / f"{cfg_name}_debug.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)

    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg.dtype = "float16"
    cfg.debug = args.debug
    cfg.exec_device = (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # 执行设备，执行某个Module或者op时，再将数据搬到这个设备上

    seed = cfg.get("seed", 1024)
    set_random_seed(seed)

    xhquant_llm_init(log_file, cfg.debug)
    logger = get_root_logger()

    logger.info(f"Config:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.dump(config_file)
    cfg.config_file = config_file

    _export_impl(cfg, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default="./config/llm/qwen3_tts_12hz_1_7B_voicedesign_code_predictor_2k_xh2a.py",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--valid", action="store_true", help="validate the model")

    args = parser.parse_args()
    cfg_name = Path(args.config).stem
    cfg_name = f"{cfg_name}"
    args.work_dir = str(Path("./work_dirs") / cfg_name)
    work_dir = args.work_dir
    if Path(work_dir).exists():
        import shutil

        from loguru import logger

        logger.info(f"Work dir {work_dir} already exists, removing it...")
        shutil.rmtree(work_dir)
    main(args)
