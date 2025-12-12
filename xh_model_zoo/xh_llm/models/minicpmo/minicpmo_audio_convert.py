import json
import shutil
import time
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM
from xhquant.api import CacheTensor
from moviepy import VideoFileClip
import tempfile
import librosa
from PIL import Image
import numpy as np
from transformers import AutoProcessor, AutoTokenizer

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model
from .minicpmo_audio_convert_config import MinicpmoAudioConvertConfig

from xhquant.api import (  # type: ignore # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    convert_fx_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    CacheTensor,
)
from xhquant.utils import set_random_seed
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_hf_compatible import MiniCPMO_HFCompatible
import os
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


class MinicpmoAudioConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: MinicpmoAudioConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None


    def _convert(self, hf_model_path: str, output_dir: str):
        cfg = self.config
        cfg.target_device = 'XH2a'
        cfg.hf_model_dir = hf_model_path
        cfg_name = Path(hf_model_path).name
        if cfg.debug:
            cfg_name = f"{cfg_name}_debug"
        work_dir = output_dir
        cfg.work_dir = str(work_dir)
        os.makedirs(work_dir, exist_ok=True)

        cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.dtype = "float16"

        seed = 1024
        set_random_seed(seed)

        logger = get_root_logger()

        out_model_dir = Path(cfg.work_dir) / "hmonnx" / "audio"
        out_model_dir.mkdir(exist_ok=True, parents=True)
        config_file = str(out_model_dir / f"audio_config.json")
        # json.dump(cfg, open(config_file, "w"), indent=4)

        dtype = getattr(torch, cfg.dtype)
        device = torch.device(cfg.device)

        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_dir, trust_remote_code=True)

        from .minicpmo_audio_model import XHMiniCPMOAudioModel
        xh_model = XHMiniCPMOAudioModel(
            hf_model=cfg.hf_model_dir,
            frontend_type="TorchFX",
            wrap_cfg=ConfigDict(
                image_slice_max_size=cfg.image_slice_max_size,
            ),
            quant_config=ConfigDict(),
            export_cfg=ConfigDict(
                input_names=[
                    "input_features",
                    "attention_mask",
                ],
                output_names=["audio_embeddings"],
                ),
        )
        native_model = xh_model.get_hf_model()

        video_path = cfg.video
        ref_audio_path = cfg.audio
        ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        sys_msg = native_model.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="en")

        contents = get_video_chunk_content(video_path)
        msg = {"role": "user", "content": contents}
        msgs = [sys_msg, msg]

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

        if cfg.valid:
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

        

    @classmethod
    def convert(cls, hf_model_path: str, config: MinicpmoAudioConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        config.quant_config = ConfigDict(quant_config)
        is_ssfp = is_ssfp_quant_config(quant_config)
        if is_ssfp:
            assert config.quant_weight is not None and Path(config.quant_weight).exists()
        MinicpmoAudioConverterXH2a(config)._convert(hf_model_path, output_dir)

