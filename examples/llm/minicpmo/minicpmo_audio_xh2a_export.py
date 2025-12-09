import argparse
import json
import math
import tempfile
import time
from collections import OrderedDict
from pathlib import Path

import librosa
import numpy as np
import torch

from moviepy import VideoFileClip
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from xhquant.api import ConfigDict, PrecisionMode, ptq_quantize, set_random_seed
from xh_model_zoo.api import Config, get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.minicpmo import MiniCPMO_HFCompatible, XHMiniCPMOAudioModel
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType

def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path)
    print("video_duration:", video.duration)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        # sr采样率（默认22050，但是有重采样的功能）
        # mono 设置为true是单通道，否则是双通道
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)

    # 1 frame + 1s audio chunk
    contents = []
    for i in range(num_units):
        frame = video.get_frame(i + 1)
        image = Image.fromarray((frame).astype(np.uint8))
        audio = audio_np[sr * i : sr * (i + 1)]
        if flatten:
            contents.extend(["<unit>", image, audio])
        else:
            contents.append(["<unit>", image, audio])

    return contents


def main(args):
    cfg = Config.fromfile(args.config)
    cfg.debug = args.debug
    cfg_name = Path(args.config).stem
    cfg_suffix = Path(args.config).suffix
    cfg_name = f"{cfg_name}"
    is_valid = args.valid

    if cfg.debug:
        cfg_name = f"{cfg_name}_debug"

    cfg_fname = f"{cfg_name}{cfg_suffix}"

    work_dir = Path("./work_dirs") / cfg_name
    cfg.work_dir = str(work_dir)
    work_dir.mkdir(exist_ok=True, parents=True)

    log_file = work_dir / f"{cfg_name}.log"

    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"

    seed = cfg.get("seed", 1024)
    set_random_seed(seed)

    xhquant_llm_init(log_file, cfg.debug)
    logger = get_root_logger()

    logger.info(f"Config:\n{cfg.pretty_text}")

    out_model_dir = Path(cfg.work_dir) / "hmonnx" / "audio"
    out_model_dir.mkdir(exist_ok=True, parents=True)

    config_file = str(out_model_dir / f"{cfg_fname}")
    cfg.dump(config_file)

    dtype = getattr(torch, cfg.dtype)
    device = torch.device(cfg.device)

    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_dir, trust_remote_code=True)

    xh_model: XHMiniCPMOAudioModel = MODELS.build(cfg.model)
    native_model = xh_model.get_hf_model()

    video_path = args.video
    # if use voice clone prompt, please set ref_audio
    ref_audio_path = args.audio
    ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
    sys_msg = native_model.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="en")
    # or use default prompt
    # sys_msg = model.get_sys_prompt(mode='omni', language='en')

    contents = get_video_chunk_content(video_path)
    msg = {"role": "user", "content": contents}
    msgs = [sys_msg, msg]

    # profile native model
    audio_hf_args = []
    audio_hf_kwargs = {}

    def _get_hf_audio_inputs(module, args, kwargs):
        for arg in args:
            if isinstance(arg, torch.Tensor):
                audio_hf_args.append(torch.empty_like(arg))
            else:
                audio_hf_args.append(arg)
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                audio_hf_kwargs[k] = torch.empty_like(v)
            else:
                audio_hf_kwargs[k] = v
        assert False

    native_model.to(device)
    native_model.to(dtype)

    try:
        handle = native_model.apm.register_forward_pre_hook(_get_hf_audio_inputs, with_kwargs=True)
        with torch.no_grad():
            res = native_model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.5,
                max_new_tokens=4096,
                omni_input=True,  # please set omni_input=True when omni inference
                use_tts_template=True,
                generate_audio=False,
                output_audio_path=None,
                max_slice_nums=1,
                use_image_id=False,
                return_dict=True,
            )
    except Exception as e:
        pass
    finally:
        del native_model.apm._forward_pre_hooks[handle.id]
        del native_model.apm._forward_pre_hooks_with_kwargs[handle.id]

    input_shapes = [list(x.shape) for x in audio_hf_args]
    kwarg_shapes = {k: list(v.shape) for k, v in audio_hf_kwargs.items() if isinstance(v, torch.Tensor)}
    logger.info("************************ native model profile ************************")
    logger.info(f"input_shapes: {input_shapes}")
    logger.info(f"kwarg_shapes: {kwarg_shapes}")


    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(Path(config_file).relative_to(out_model_dir)),
        )
    )
    meta_info["hf_model"] = xh_model.hf_model_dir
    meta_info["wrap_cfg"] = xh_model.wrap_cfg.to_dict()
    meta_info["model_profile"] = {
        "input_shapes": input_shapes,
        "kwarg_shapes": kwarg_shapes,
    }

    onnx_dir = out_model_dir
    json.dump(meta_info, open(onnx_dir / f"meta_info.json", "w"), indent=4)

    xh_model.init_wrap_model()
    xh_model.wrap_processor(native_model)

    # please set generate_audio=True and output_audio_path to save the tts result
    generate_audio = True
    output_audio_path = "output.wav"

    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
    xh_model.to(dtype)
    xh_model.to(device)

    native_model.to(dtype)
    native_model.to(device)

    native_model.init_tts()

    MiniCPMO_HFCompatible.to_hf_compatible(native_model, audio_model=xh_model)

    if is_valid:
        logger.info("************************ valid wraped model ************************")
        with torch.no_grad():
            res = native_model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.5,
                max_new_tokens=4096,
                omni_input=True,  # please set omni_input=True when omni inference
                use_tts_template=True,
                generate_audio=generate_audio,
                output_audio_path=output_audio_path,
                max_slice_nums=1,
                use_image_id=False,
                return_dict=True,
            )
        logger.info(res)

    net_inputs = [None] * 2  # [input_features, attention_mask]

    def get_audio_inputs(module, args, kwargs):
        for i, arg in enumerate(args):
            net_inputs[i] = arg
        if "input_features" in kwargs:
            net_inputs[0] = kwargs["input_features"]
        if "attention_mask" in kwargs:
            net_inputs[1] = kwargs["attention_mask"]
        assert False

    # xh_model

    try:
        handle = xh_model.register_forward_pre_hook(get_audio_inputs, with_kwargs=True)
        with torch.no_grad():
            res = native_model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.5,
                max_new_tokens=4096,
                omni_input=True,  # please set omni_input=True when omni inference
                use_tts_template=True,
                generate_audio=generate_audio,
                output_audio_path=output_audio_path,
                max_slice_nums=1,
                use_image_id=False,
                return_dict=True,
            )
    except Exception as e:
        pass
    finally:
        del xh_model._forward_pre_hooks[handle.id]
        del xh_model._forward_pre_hooks_with_kwargs[handle.id]

    net_inputs = [x.to(device).to(dtype) for x in net_inputs]
    logger.info(f"input_features shape: {net_inputs[0].shape} {net_inputs[0].dtype}")
    logger.info(f"audio_attention_mask shape: {net_inputs[1].shape} {net_inputs[1].dtype}")

    xh_model.to(dtype)

    onnx_dir = out_model_dir
    onnx_file = onnx_dir / f"{cfg_name}.onnx"

    from xhquant.api import convert_fx_model_to_hmonnx
    convert_fx_model_to_hmonnx(
        xh_model._wrap_model,
        net_inputs,
        cfg.target_device,
        onnx_file,
        quant_config=cfg.quant_config,
        input_names=["input_features", "audio_attention_mask"],
        output_names=["hidden_state"],
    )
    logger.info(f"Export onnx to {onnx_file}")

    meta_info.vision_hmonnx = str(Path(onnx_file).relative_to(onnx_dir))
    json.dump(meta_info, open(onnx_dir / f"meta_info.json", "w"), indent=4)

    from xhquant.api import HMONNXGoldenInference
    hm_model = HMONNXGoldenInference(onnx_file)
    hm_model.save_golden = True
    hm_model.exec_device = device

    golden_dir = Path(cfg.work_dir) / "golden" / f"{Path(onnx_file).stem}"
    golden_dir.mkdir(exist_ok=True, parents=True)
    hm_model.golden_dir = str(golden_dir)

    with torch.no_grad():
        # 将 net_inputs[1] 中的 -inf 替换为 float16 的最小值
        mask = torch.isneginf(net_inputs[1])
        if mask.any():
            min_fp16 = torch.finfo(torch.float16).min
            net_inputs[1] = net_inputs[1].clone()
            net_inputs[1][mask] = min_fp16
            logger.info(f"已将 net_inputs[1] 中的 -inf 替换为 float16 最小值: {min_fp16}")
        net_inputs[1] 
        hm_model.forward(*net_inputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", default="configs/minicpmo/audio/minicpmo_auido_7b_xh2a_2k.py", type=str)
    parser.add_argument("--video", type=str, default="weights/MiniCPM-o-2_6/assets/Skiing.mp4")
    parser.add_argument("--audio", type=str, default="weights/MiniCPM-o-2_6/assets/demo.wav")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--valid", action="store_true", help="check hmonnx mode")
    args = parser.parse_args()
    main(args)
