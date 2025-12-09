import argparse
import json
import math
import tempfile
import time
from pathlib import Path

import librosa
import numpy as np
import torch
from moviepy import VideoFileClip
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from xhquant.api import ConfigDict, PrecisionMode, ptq_quantize, set_random_seed

from xh_model_zoo.api import Config, get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.minicpmo import MiniCPMO_HFCompatible, XHMiniCPMOVisionModel
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

    out_model_dir = Path(cfg.work_dir) / "hmonnx" / "vision"
    out_model_dir.mkdir(exist_ok=True, parents=True)

    config_file = str(out_model_dir / f"{cfg_fname}")
    cfg.dump(config_file)

    dtype = getattr(torch, cfg.dtype)
    device = cfg.device

    xh_model: XHMiniCPMOVisionModel = MODELS.build(cfg.model)
    native_model = xh_model.get_hf_model()
    xh_model.init_wrap_model()
    xh_model.wrap_processor(native_model)
    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(Path(config_file).relative_to(out_model_dir)),
        )
    )
    meta_info["hf_model"] = xh_model.hf_model_dir
    meta_info["wrap_cfg"] = xh_model.wrap_cfg.to_dict()
    meta_info["patch_size"] = xh_model.patch_size
    meta_info["num_patches_per_side"] = xh_model.num_patches_per_side

    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_dir, trust_remote_code=True)

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

    # please set generate_audio=True and output_audio_path to save the tts result
    generate_audio = True
    output_audio_path = "output.wav"

    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
    xh_model.to(dtype)
    xh_model.to(device)

    native_model.to(dtype)
    native_model.to(device)

    native_model.init_tts()

    MiniCPMO_HFCompatible.to_hf_compatible(native_model, vision_model=xh_model)

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

    net_inputs = [None] * 5  # [all_pixel_values, position_ids]

    # def get_visual_inputs(module, args, kwargs):
    #     for i, arg in enumerate(args):
    #         net_inputs[i] = arg
    #     if "all_pixel_values" in kwargs:
    #         net_inputs[0] = kwargs["all_pixel_values"]
    #     if "position_ids" in kwargs:
    #         net_inputs[1] = kwargs["position_ids"]

    # xh_model

    # xh_model.register_forward_pre_hook(get_visual_inputs, with_kwargs=True)

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

    logger.info(f"all_pixel_values shape: {net_inputs[0].shape} {net_inputs[0].dtype}")
    logger.info(f"attention_mask shape: {net_inputs[1].shape} {net_inputs[1].dtype}")
    logger.info(f"position_ids shape: {net_inputs[2].shape} {net_inputs[2].dtype}")
    logger.info(f"resampler_pos_embed shape: {net_inputs[3].shape} {net_inputs[3].dtype}")
    logger.info(f"resampler_key_padding_mask shape: {net_inputs[4].shape} {net_inputs[4].dtype}")

    xh_model.to(dtype)

    # xh_model.interactive_mode = True
    # xh_model.convert_to_fronted_graph(net_inputs)
    # torch.cuda.empty_cache()

    # xh_model.convert_to_quant_graph(cfg.target_device)

    # calib_data = net_inputs
    # with TimeProfiler("PTQ Quantize", logger):
    #     ptq_quantize(xh_model.quanted_model, [calib_data], PrecisionMode.ALIGNED, [device])

    # xh_model.convert_to_export_graph(net_inputs)
    # onnx_dir = out_model_dir    # onnx_file = xh_model.to_export_onnx(net_inputs, onnx_dir, cfg_name)[0]

    onnx_dir = out_model_dir
    onnx_file = onnx_dir / f"{cfg_name}.onnx"

    from xhquant.api import convert_fx_model_to_hmonnx
    convert_fx_model_to_hmonnx(
        xh_model._wrap_model,
        net_inputs,
        cfg.target_device,
        onnx_file,
        quant_config=cfg.quant_config,
        input_names=["pixel_values", "position_ids", "attention_mask", "resampler_pos_embed", "resampler_key_padding_mask"],
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
        net_inputs[1] = net_inputs[1].to(torch.int32)
        hm_model.forward(*net_inputs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", default="configs/xh2a/minicpmo/vision/minicpmo_vision_7b_xh2a_2k.py", type=str)
    parser.add_argument("--video", type=str, default="weights/MiniCPM-o-2_6/assets/Skiing.mp4")
    parser.add_argument("--audio", type=str, default="weights/MiniCPM-o-2_6/assets/demo.wav")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--valid", action="store_true", help="check hmonnx mode")
    args = parser.parse_args()
    main(args)
