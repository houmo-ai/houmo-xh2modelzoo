# 该示例展示了如何导出 Qwen3 TTS Talker 模型
import argparse
import json
import time
from pathlib import Path
from typing import cast

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from xhquant.api import (
    Config,
    ConfigDict,
    convert_onnx_to_hmonnx,
    set_random_seed,
)
from xhquant.api import Config
from xh_model_zoo.api import get_root_logger, xhquant_llm_init


def _export_impl(cfg: Config, args: argparse.Namespace):
    device = cfg.exec_device
    dtype = getattr(torch, cfg.dtype)
    logger = get_root_logger()
    work_dir = cfg.work_dir
    model_dir = cfg.hf_model_dir
    hf_model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map="cuda:0",
        dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )
    hf_model = cast(Qwen3TTSModel, hf_model)
    config_file = cfg.config_file
    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(config_file.relative_to(cfg.work_dir)),
        )
    )
    target_device = cfg.target_device
    meta_info["model_name"] = cfg_name
    meta_info["act"] = type(hf_model.model.talker.text_projection.act_fn).__name__
    meta_info["target_device"] = target_device
    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir

    meta_file = Path(work_dir) / "meta.json"
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)

    # 注册前向钩子，获取输入shape
    # instruct_ids,   # [1, 40, 2048]   #instruct不更新的话，多次请求可以共享
    # [[tts_bos_token_id, tts_eos_token_id, tts_pad_token_id]] #全局只需要一次推理
    # input_id[:, :3]: <|im_start|>assistant\n   #全局只需要一次推理
    # input_id[:, 3:4]
    # input_id[:, 3:-5]
    # 三个固定输入，两个动态输入
    feature_dim = 2048

    def _hook(module, inputs):
        logger.info(f"inputs[0].shape: {inputs[0].shape}")
        nonlocal feature_dim
        feature_dim = inputs[0].shape[-1]
        return inputs

    text_projection_hook = hf_model.model.talker.text_projection.register_forward_pre_hook(_hook)

    wavs, sr = hf_model.generate_voice_design(
        text="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地",
        language="Chinese",
        instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出黏人、做作又刻意卖萌的听觉效果。",
    )
    out_file = Path(work_dir) / "output_voice_design.wav"
    sf.write(out_file, wavs[0], sr)
    logger.info(f"Audio saved to {out_file}")

    text_projection_hook.remove()
    # 导出 text_projection
    example_input = torch.randn(1, 1, feature_dim, device=device, dtype=dtype)
    onnx_dir = Path(work_dir) / "onnx"
    onnx_dir.mkdir(exist_ok=True, parents=True)
    text_projection_onnx_file = Path(work_dir) / "onnx" / "text_projection.onnx"
    torch.onnx.export(
        hf_model.model.talker.text_projection.float().cpu(),
        (example_input.float().cpu(),),
        text_projection_onnx_file,
        input_names=["inputs_embeds"],
        output_names=["outputs"],
    )
    logger.info(f"text_projection exported to {text_projection_onnx_file}")

    hmonnx_dir = Path(work_dir) / "hmonnx"
    hmonnx_dir.mkdir(exist_ok=True, parents=True)
    text_projection_hmonnx_file = str(hmonnx_dir / f"text_projection_{target_device}.onnx")
    convert_onnx_to_hmonnx(
        text_projection_onnx_file,
        [example_input.float().cpu()],
        target_device,
        text_projection_hmonnx_file,
    )
    meta_info["hmonnx"] = str(Path(text_projection_hmonnx_file).relative_to(Path(meta_file).parent))
    with open(meta_file, "w") as f:
        json.dump(meta_info, f, indent=4)
    logger.info(f"text_projection converted to hmonnx and saved to {text_projection_hmonnx_file}")


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
        default="./config/llm/qwen3_tts_12hz_1_7B_text_projection_xh2a.py",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--valid", action="store_true", help="validate the model")

    args = parser.parse_args()
    cfg_name = Path(args.config).stem
    cfg_name = f"{cfg_name}"
    args.work_dir = str(Path("./work_dirs") / cfg_name)
    work_dir = args.work_dir
    # from loguru import logger

    # if Path(work_dir).exists():
    #     from loguru import logger

    #     logger.warning(f"{work_dir} already exists, please remove it first")
    #     exit(1)
    main(args)
