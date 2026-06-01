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
from xhquant.xhonnxruntime import config as xh_ort_config
from xh_model_zoo.api import get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.base_llm_model import LLMBaseModel
from xh_model_zoo.xh_llm.models.qwen3_tts import Qwen3TTSHMONNXInference, XHQwen3TTSModel


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


def _impl(cfg: Config, args: argparse.Namespace):
    logger = get_root_logger()
    work_dir = cfg.work_dir
    exec_device = cfg.exec_device
    xh_model = MODELS.build(cfg.model)
    assert isinstance(xh_model, Qwen3TTSHMONNXInference)
    xh_model.to(exec_device)

    xh_ort_config.disable_progress = False
    wavs, sr = xh_model.generate_custom_voice(
        text="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地",
        language="Chinese",
        speaker="vivian",
    )
    out_file = Path(work_dir) / "output_custom_voice.wav"
    sf.write(out_file, wavs[0], sr)
    logger.info(f"Audio saved to {out_file}")


def main(args: argparse.Namespace) -> None:
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg_name = Path(args.config).stem
    log_file = Path(cfg.work_dir) / f"{cfg_name}_debug.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)

    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg.dtype = "float16"
    cfg.debug = False
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

    _impl(cfg, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--config",
        type=str,
        default="./config/llm/qwen3_tts_12hz_0_6B_customvoice_xh2a_hmonnx.py",
    )
    parser.add_argument("--seed", type=int, default=1024)

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
