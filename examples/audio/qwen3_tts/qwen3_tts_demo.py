# Example: export a Qwen3-TTS sub-model to XH2a / HMONNX
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
    xh_model.to("cpu")  # move to CPU for export
    torch.cuda.empty_cache()
    # print_gpu_info(logger)
    xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exporting...")

    logger.info("************* Start Exported Graph *************")
    # logger.info(str(xh_model.exported_model.graph))
    logger.info("************* End Exported Graph *************")
    torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    xh_model.to("cpu")  # move to CPU for export
    torch.cuda.empty_cache()
    logger.info("*************** Start exporting onnx ***************")
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


# ---------------------------------------------------------------------------
# Generation helper: pick the generate_* method by cfg.tts_mode
# ---------------------------------------------------------------------------
_DEFAULT_TEXT = "基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地"


def _run_generate(xh_model, cfg):
    """Call the generate_* method matching cfg.tts_mode; returns (wavs, sr)."""
    mode = getattr(cfg, "tts_mode", "custom_voice")
    text = getattr(cfg, "tts_text", _DEFAULT_TEXT)
    if mode == "voice_design":
        return xh_model.generate_voice_design(
            text=text, language="Chinese",
            instruct=getattr(cfg, "tts_instruct", ""),
        )
    elif mode == "voice_clone":
        ref_audio = getattr(cfg, "ref_audio", "/tmp/clone_1.wav")
        assert Path(ref_audio).exists(), f"missing reference audio {ref_audio}"
        return xh_model.generate_voice_clone(
            text=text, language="Chinese",
            ref_audio=ref_audio, ref_text=getattr(cfg, "ref_text", ""),
        )
    else:  # custom_voice (default)
        return xh_model.generate_custom_voice(
            text=text, language="Chinese",
            speaker=getattr(cfg, "tts_speaker", "vivian"),
        )


def _impl(cfg: Config, args: argparse.Namespace):
    logger = get_root_logger()
    work_dir = cfg.work_dir
    exec_device = cfg.exec_device
    xh_model = MODELS.build(cfg.model)
    assert isinstance(xh_model, Qwen3TTSHMONNXInference)
    xh_model.to(exec_device)

    xh_ort_config.disable_progress = False
    wavs, sr = _run_generate(xh_model, cfg)
    out_file = Path(work_dir) / f"output_{getattr(cfg, 'tts_mode', 'custom_voice')}.wav"
    sf.write(out_file, wavs[0], sr)
    logger.info(f"Audio saved to {out_file}")


def main(args: argparse.Namespace) -> None:
    cfg = Config.fromfile(args.config)
    if getattr(args, "variant", None):
        from config.llm._components import apply_variant_hmonnx
        apply_variant_hmonnx(cfg, args.variant)
    cfg.work_dir = args.work_dir
    cfg_name = Path(args.config).stem
    log_file = Path(cfg.work_dir) / f"{cfg_name}_debug.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)

    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    cfg.dtype = "float16"
    cfg.debug = False
    cfg.exec_device = (
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )  # exec device: data is moved here when running a module/op

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
        default="./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py",
        help="HMONNX config (unified; pick variant with --variant)",
    )
    parser.add_argument("--variant", choices=["0_6B_base", "0_6B_customvoice", "1_7B_voicedesign"], default=None,
                        help="TTS variant; injects work_dirs paths + hf_model/tts_mode into the parsed config")
    parser.add_argument("--name", type=str, default=None,
                        help="scratch work_dir name; defaults to '<config stem>_<variant>' to avoid clashes")
    parser.add_argument("--seed", type=int, default=1024)

    args = parser.parse_args()
    _stem = Path(args.config).stem
    if args.name:
        cfg_name = args.name
    elif args.variant:
        cfg_name = f"{_stem}_{args.variant}"
    else:
        cfg_name = _stem
    args.work_dir = str(Path("./work_dirs") / cfg_name)
    work_dir = args.work_dir

    if Path(work_dir).exists():
        import shutil

        from loguru import logger

        logger.info(f"Work dir {work_dir} already exists, removing it...")
        shutil.rmtree(work_dir)
    main(args)
