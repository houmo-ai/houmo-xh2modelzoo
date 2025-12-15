import json
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import librosa
import torch
from moviepy import VideoFileClip
from PIL import Image
from transformers import AutoTokenizer
import numpy as np

from ..base_converter import HFTransfromersConverter
from .minicpmo_tts_vocos_convert_config import MinicpmoTTSVocosConvertConfig

from xhquant.api import (  # type: ignore # isort:skip
    ConfigDict,
    DeviceType,
    create_quant_config,
    get_root_logger,
    is_ssfp_quant_config,
    convert_onnx_to_hmonnx,
)
from xhquant.utils import set_random_seed


def get_video_chunk_content(video_path, flatten=True):
    video = VideoFileClip(video_path)
    print("video_duration:", video.duration)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        video.audio.write_audiofile(temp_audio_file_path, codec="pcm_s16le", fps=16000)
        audio_np, sr = librosa.load(temp_audio_file_path, sr=16000, mono=True)
    num_units = math.ceil(video.duration)

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


class MinicpmoTTSVocosConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: MinicpmoTTSVocosConvertConfig):
        super().__init__()
        self.config = config

    def _convert(self, hf_model_path: str, output_dir: str):
        cfg = self.config
        cfg.target_device = "XH2a"
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

        out_model_dir = Path(cfg.work_dir) / "hmonnx" / "tts_vocos"
        out_model_dir.mkdir(exist_ok=True, parents=True)
        config_file = str(out_model_dir / "vocos_config.json")

        dtype = getattr(torch, cfg.dtype)
        device = torch.device(cfg.device)

        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_dir, trust_remote_code=True)

        from .minicpmo_tts_vocos_model import XHMiniCPMOTTSVOCOSModel
        from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
        from xh_model_zoo.xh_llm.models.minicpmo import MiniCPMO_HFCompatible

        xh_model = XHMiniCPMOTTSVOCOSModel(
            hf_model=cfg.hf_model_dir,
            frontend_type="TorchFX",
            wrap_cfg=ConfigDict(
                image_slice_max_size=cfg.image_slice_max_size,
            ),
            quant_config=ConfigDict(),
        )
        native_model = xh_model.get_hf_model()

        video_path = cfg.video
        ref_audio_path = cfg.audio
        ref_audio, _ = librosa.load(ref_audio_path, sr=16000, mono=True)
        sys_msg = native_model.get_sys_prompt(ref_audio=ref_audio, mode="omni", language="en")

        contents = get_video_chunk_content(video_path)
        msgs = [sys_msg, {"role": "user", "content": contents}]

        native_model.to(device)
        native_model.to(dtype)
        native_model.init_tts()

        xh_model.init_wrap_model()
        xh_model.wrap_processor(native_model)

        # profile vocos decode input
        input_data: List[torch.Tensor] = []
        original_decode = native_model.vocos.decode

        def forward_with_hook(*args, **kwargs):
            input_data.append(args[0])
            return original_decode(*args, **kwargs)

        native_model.vocos.decode = forward_with_hook
        with torch.no_grad():
            native_model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.5,
                max_new_tokens=128,
                omni_input=True,
                use_tts_template=True,
                generate_audio=True,
                output_audio_path="output.wav",
                max_slice_nums=1,
                use_image_id=False,
                return_dict=True,
            )
        native_model.vocos.decode = original_decode

        if not input_data:
            raise RuntimeError("Failed to collect VOCOS inputs.")

        features = input_data[0]
        aim_length = 2048
        current_length = features.shape[2]
        if current_length < aim_length:
            padding_length = aim_length - current_length
            padding_tensor = torch.zeros(features.shape[0], features.shape[1], padding_length).to(features)
            features = torch.cat([features, padding_tensor], dim=2)
        mask = torch.ones(current_length).to(features).unsqueeze(0).unsqueeze(0)
        if current_length < aim_length:
            new_mask = torch.zeros(aim_length).to(features.device)
            new_mask[:current_length] = mask
            mask = new_mask.unsqueeze(0).unsqueeze(0)

        vocos_dir = Path(cfg.work_dir) / "vocos_tmp"
        vocos_dir.mkdir(exist_ok=True, parents=True)
        vocos_onnx = vocos_dir / f"{cfg_name}_tts_vocos.onnx"

        original_forward = xh_model._wrap_model.forward
        xh_model._wrap_model.forward = xh_model._wrap_model.decode_mask
        torch.onnx.export(
            xh_model._wrap_model.cpu(),
            (features.cpu(), mask.cpu()),
            vocos_onnx,
            input_names=["features", "mask"],
            output_names=["x", "y", "mag"],
            verbose=False,
            opset_version=18,
        )
        xh_model._wrap_model.forward = original_forward

        out_hmonnx_dir = Path(cfg.work_dir) / "hmonnx" / "tts_vocos"
        out_hmonnx_dir.mkdir(exist_ok=True, parents=True)
        out_hmonnx = out_hmonnx_dir / f"{cfg_name}_tts_vocos.onnx"

        convert_onnx_to_hmonnx(
            str(vocos_onnx),
            [features.cpu(), mask.cpu()],
            DeviceType.XH2a,
            str(out_hmonnx),
            cfg.quant_config,
        )

        meta_info = ConfigDict(
            dict(
                create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                config=str(Path(config_file).relative_to(out_model_dir)),
                vocos=str(Path(out_hmonnx).relative_to(out_hmonnx_dir)),
            )
        )
        meta_info["hf_model"] = xh_model.hf_model_dir
        json.dump(meta_info, open(out_model_dir / "meta_info.json", "w"), indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config: MinicpmoTTSVocosConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        config.quant_config = ConfigDict(quant_config)
        if is_ssfp_quant_config(quant_config):
            assert config.quant_weight is not None and Path(config.quant_weight).exists()
        MinicpmoTTSVocosConverterXH2a(config)._convert(hf_model_path, output_dir)

