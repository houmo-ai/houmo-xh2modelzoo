import argparse
import json
import math
import tempfile
import time
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import torch
from copy import deepcopy
from moviepy import VideoFileClip
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer
from xhquant.api import ConfigDict, PrecisionMode, ptq_quantize, set_random_seed

from xh_model_zoo.api import Config, get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.minicpmo import MiniCPMO_HFCompatible, XHMiniCPMOTTSDVAEModel
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType

def cleanup_memory(verbos=True) -> None:
    """Run GC and clear GPU memory."""
    import gc
    import inspect
    caller_name = ''
    try:
        caller_name = f' (from {inspect.stack()[1].function})'
    except (ValueError, KeyError):
        pass

    def total_reserved_mem() -> int:
        return sum(torch.cuda.memory_reserved(device=i) for i in range(torch.cuda.device_count()))

    memory_before = total_reserved_mem()

    # gc.collect and empty cache are necessary to clean up GPU memory if the model was distributed
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        memory_after = total_reserved_mem()
        if verbos:
            print(
                f"GPU memory{caller_name}: {memory_before / (1024 ** 3):.2f} -> {memory_after / (1024 ** 3):.2f} GB"
                f" ({(memory_after - memory_before) / (1024 ** 3):.2f} GB)"
            )


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


def xhmodel_export_onnx(
    xh_model: LLMBaseModel,
    data_batch,
    onnx_output_dir: str,
    cfg_name,
    execution_device,
    dtype,
    logger,
    valid: bool = True,
):
    logger = get_root_logger()

    xh_model.to("cpu")  # 切换到cpu上进行模型导出
    torch.cuda.empty_cache()

    # memory_tracker = MemoryTracker(execution_device)
    # memory_tracker.log_memory("Before exporting graph", logger)
    xh_model.set_input_sequence_length(data_batch["current_input_length"].cpu().item())
    logger.info("Start exporting graph.............")
    with TimeProfiler("export graph"):
        xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish exported graph.")

    # memory_tracker.log_memory("after exporting graph", logger)

    torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    if valid:
        xh_model.to(execution_device)
        xh_model.to(dtype)
        xh_model.set_exec_device(execution_device)
        with torch.no_grad():
            outs = xh_model.test_step(data_batch)
            logits = outs.logits.detach()

        xh_model.to("cpu")  # 切换到cpu上进行模型导出
        torch.cuda.empty_cache()

    # memory_tracker.log_memory("before exporting onnx", logger)
    logger.info("*************** Start exporting onnx ***************")
    with TimeProfiler("export onnx"):
        onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    # memory_tracker.log_memory("after exporting onnx", logger)
    return onnx_file

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

    out_model_dir = Path(cfg.work_dir) / "hmonnx" / "tts"
    out_model_dir.mkdir(exist_ok=True, parents=True)

    config_file = str(out_model_dir / f"{cfg_fname}")
    cfg.dump(config_file)

    dtype = getattr(torch, cfg.dtype)
    device = cfg.device
    
    xh_model: XHMiniCPMOTTSDVAEModel = MODELS.build(cfg.model)
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

    original_forward = native_model.tts.dvae.forward

    input_data = list()
    def forward_with_hook(*args, **kwargs):
        input_data.append(args[0])
        return original_forward(*args, **kwargs)
    
    native_model.tts.dvae.forward = forward_with_hook

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
    
    native_model.tts.dvae.forward = original_forward

    dvae_onnx_dir = Path(cfg.work_dir) / "dvae"
    dvae_onnx_dir.mkdir(exist_ok=True, parents=True)
    dvae_part1_onnx_file = str(dvae_onnx_dir) +  "/dvae_part1.onnx"
    dvae_part2_onnx_file = str(dvae_onnx_dir) +  "/dvae_part2.onnx"
    import onnx
    import onnxsim
    with torch.no_grad():
        original_forward = xh_model._wrap_model.forward

        part1_input = input_data[0]

        aim_length = 1024
        current_length = part1_input.shape[2]
        if current_length < aim_length:
            padding_length = aim_length - current_length
            padding_tensor = torch.zeros(part1_input.shape[0], part1_input.shape[1], padding_length).to(part1_input)
            part1_input = torch.cat([part1_input, padding_tensor], dim=2)

        xh_model._wrap_model.forward = xh_model._wrap_model.forward_part1
        torch.onnx.export(
            xh_model._wrap_model,
            part1_input,
            dvae_part1_onnx_file,
            input_names = ["indice"],
            output_names = ["feature"],
            verbose=False,
            opset_version=18
        )
        mask = torch.ones(current_length*2).float().to(part1_input)

        if current_length < aim_length:
            new_mask = torch.zeros(aim_length*2).float().to(part1_input.device)
            new_mask[:current_length*2] = mask
            mask = new_mask
        mask = mask.unsqueeze(0).unsqueeze(0)

        part1_output = xh_model._wrap_model(part1_input)

        part1_output = part1_output.flatten(2)

        xh_model._wrap_model.forward = xh_model._wrap_model.forward_part2

        torch.onnx.export(
            xh_model._wrap_model,
            (part1_output, mask),
            dvae_part2_onnx_file,
            input_names = ["feature", "mask"],
            output_names = ["output"],
            verbose=False,
            opset_version=18
        )

        xh_model._wrap_model.forward = original_forward

        onnx_model = onnx.load(dvae_part1_onnx_file, load_external_data=True)
        onnx_model, check = onnxsim.simplify(onnx_model)
        onnx.save(onnx_model, dvae_part1_onnx_file)

        onnx_model = onnx.load(dvae_part2_onnx_file, load_external_data=True)
        onnx_model, check = onnxsim.simplify(onnx_model)
        onnx.save(onnx_model, dvae_part2_onnx_file)

        from xhquant.api import (
            DeviceType,
            convert_onnx_to_hmonnx,
        )

        out_hmonnx_file = Path(cfg.work_dir) / "hmonnx" / "dvae_part1.onnx"
        convert_onnx_to_hmonnx(
            dvae_part1_onnx_file,
            [part1_input.cpu()],  # a list of torch tensors
            DeviceType.XH2a,
            str(out_hmonnx_file),
            cfg.model.quant_config,
        )
        logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}")

        from xhquant.api import HMONNXGoldenInference
        hm_dvae = HMONNXGoldenInference(str(out_hmonnx_file))
        hm_dvae.to(device)
        hm_dvae.to(dtype)
        hm_dvae.save_golden = True

        golden_dir = Path(cfg.work_dir) / "part1/golden"
        golden_dir.mkdir(exist_ok=True, parents=True)
        hm_dvae.golden_dir = str(golden_dir)
        with torch.no_grad():
            hm_dvae.forward(part1_input.to(torch.int32))

    out_hmonnx_file = Path(cfg.work_dir) / "hmonnx" / "dvae_part2.onnx"
    convert_onnx_to_hmonnx(
        dvae_part2_onnx_file,
        [part1_output.cpu().half(), mask.cpu().half()],  # a list of torch tensors
        DeviceType.XH2a,
        str(out_hmonnx_file),
        cfg.model.quant_config,
    )
    logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}")

    from xhquant.api import HMONNXGoldenInference
    hm_dvae = HMONNXGoldenInference(str(out_hmonnx_file))
    hm_dvae.to(device)
    hm_dvae.to(dtype)
    hm_dvae.save_golden = True
    golden_dir = Path(cfg.work_dir) / "part2/golden"
    golden_dir.mkdir(exist_ok=True, parents=True)
    hm_dvae.golden_dir = str(golden_dir)
    with torch.no_grad():
        hm_dvae.forward(part1_output.to(torch.half), mask.to(torch.half))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config", default="configs/minicpmo/tts/minicpmo_tts_dvae_xh2a_2k.py", type=str)
    parser.add_argument("--video", type=str, default="weights/MiniCPM-o-2_6/assets/Skiing.mp4")
    parser.add_argument("--audio", type=str, default="weights/MiniCPM-o-2_6/assets/demo.wav")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--valid", action="store_true", help="check hmonnx mode")
    args = parser.parse_args()
    main(args)
