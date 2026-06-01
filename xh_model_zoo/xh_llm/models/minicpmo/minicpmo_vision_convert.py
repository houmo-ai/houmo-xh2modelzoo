# Copyright 2025 HOUMO AI
#
# File: minicpmo_vision_convert.py
# Description:
#   Minicpmo Vision Convert implementation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

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
from .minicpmo_vision_convert_config import MinicpmoVisionConvertConfig

from xhquant.api import (  # type: ignore # isort:skip
    ConfigDict,
    DeviceType,
    convert_fx_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
    is_ssfp_quant_config,
)
from xhquant.utils import set_random_seed
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.minicpmo.minicpmo_hf_compatible import MiniCPMO_HFCompatible


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


class MinicpmoVisionConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: MinicpmoVisionConvertConfig):
        super().__init__()
        self.config = config
        self.hf_model_path: Optional[str] = None
        self.output_dir: Optional[str] = None

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

        out_model_dir = Path(cfg.work_dir) / "hmonnx" / "vision"
        out_model_dir.mkdir(exist_ok=True, parents=True)
        config_file = str(out_model_dir / "vision_config.json")

        dtype = getattr(torch, cfg.dtype)
        device = torch.device(cfg.device)

        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_dir, trust_remote_code=True)

        from .minicpmo_vision_model import XHMiniCPMOVisionModel

        xh_model = XHMiniCPMOVisionModel(
            hf_model=cfg.hf_model_dir,
            frontend_type="TorchFX",
            wrap_cfg=ConfigDict(
                image_slice_max_size=cfg.image_slice_max_size,
            ),
            quant_config=ConfigDict(),
            export_cfg=ConfigDict(
                input_names=[
                    "pixel_values",
                    "position_ids",
                    "attention_mask",
                    "resampler_pos_embed",
                    "resampler_key_padding_mask",
                ],
                output_names=["hidden_state"],
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

        vision_hf_args: List[torch.Tensor] = []
        vision_hf_kwargs: Dict[str, Any] = {}

        def _get_hf_vision_inputs(module, args, kwargs):
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    vision_hf_args.append(torch.empty_like(arg))
                else:
                    vision_hf_args.append(arg)
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor):
                    vision_hf_kwargs[k] = torch.empty_like(v)
                else:
                    vision_hf_kwargs[k] = v
            assert False

        native_model.to(device)
        native_model.to(dtype)

        handle = None
        try:
            handle = native_model.vpm.register_forward_pre_hook(_get_hf_vision_inputs, with_kwargs=True)
            with torch.no_grad():
                native_model.chat(
                    msgs=msgs,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=0.5,
                    max_new_tokens=4096,
                    omni_input=True,
                    use_tts_template=True,
                    generate_audio=False,
                    output_audio_path=None,
                    max_slice_nums=1,
                    use_image_id=False,
                    return_dict=True,
                )
        except Exception:
            pass
        finally:
            if handle is not None:
                del native_model.vpm._forward_pre_hooks[handle.id]
                del native_model.vpm._forward_pre_hooks_with_kwargs[handle.id]

        input_shapes = [list(x.shape) for x in vision_hf_args if isinstance(x, torch.Tensor)]
        kwarg_shapes = {k: list(v.shape) for k, v in vision_hf_kwargs.items() if isinstance(v, torch.Tensor)}
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

        json.dump(meta_info, open(out_model_dir / "meta_info.json", "w"), indent=4)

        xh_model.init_wrap_model()
        xh_model.wrap_processor(native_model)
        meta_info["patch_size"] = xh_model.patch_size
        meta_info["num_patches_per_side"] = xh_model.num_patches_per_side

        generate_audio = True
        output_audio_path = "output.wav"

        xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
        xh_model.to(dtype)
        xh_model.to(device)

        native_model.to(dtype)
        native_model.to(device)

        native_model.init_tts()

        MiniCPMO_HFCompatible.to_hf_compatible(native_model, vision_model=xh_model)

        if cfg.valid:
            logger.info("************************ valid wraped model ************************")
            with torch.no_grad():
                res = native_model.chat(
                    msgs=msgs,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=0.5,
                    max_new_tokens=4096,
                    omni_input=True,
                    use_tts_template=True,
                    generate_audio=generate_audio,
                    output_audio_path=output_audio_path,
                    max_slice_nums=1,
                    use_image_id=False,
                    return_dict=True,
                )
            logger.info(res)

        net_inputs: List[Optional[torch.Tensor]] = [None] * 5

        def get_vision_inputs(module, args, kwargs):
            for i, arg in enumerate(args):
                if i < len(net_inputs):
                    net_inputs[i] = arg
            if "all_pixel_values" in kwargs:
                net_inputs[0] = kwargs["all_pixel_values"]
            if "pixel_values" in kwargs:
                net_inputs[0] = kwargs["pixel_values"]
            if "position_ids" in kwargs:
                net_inputs[1] = kwargs["position_ids"]
            if "attention_mask" in kwargs:
                net_inputs[2] = kwargs["attention_mask"]
            if "resampler_pos_embed" in kwargs:
                net_inputs[3] = kwargs["resampler_pos_embed"]
            if "resampler_key_padding_mask" in kwargs:
                net_inputs[4] = kwargs["resampler_key_padding_mask"]
            assert False

        handle = None
        try:
            handle = xh_model.register_forward_pre_hook(get_vision_inputs, with_kwargs=True)
            with torch.no_grad():
                native_model.chat(
                    msgs=msgs,
                    tokenizer=tokenizer,
                    sampling=True,
                    temperature=0.5,
                    max_new_tokens=4096,
                    omni_input=True,
                    use_tts_template=True,
                    generate_audio=generate_audio,
                    output_audio_path=output_audio_path,
                    max_slice_nums=1,
                    use_image_id=False,
                    return_dict=True,
                )
        except Exception:
            pass
        finally:
            if handle is not None:
                del xh_model._forward_pre_hooks[handle.id]
                del xh_model._forward_pre_hooks_with_kwargs[handle.id]

        for idx, value in enumerate(net_inputs):
            if isinstance(value, torch.Tensor):
                if value.dtype in (torch.int64, torch.int32):
                    net_inputs[idx] = value.to(device).to(torch.int32)
                else:
                    net_inputs[idx] = value.to(device).to(dtype)
        logger.info(f"pixel_values shape: {net_inputs[0].shape} {net_inputs[0].dtype}")
        logger.info(f"position_ids shape: {net_inputs[1].shape} {net_inputs[1].dtype}")
        logger.info(f"attention_mask shape: {net_inputs[2].shape} {net_inputs[2].dtype}")
        logger.info(f"resampler_pos_embed shape: {net_inputs[3].shape} {net_inputs[3].dtype}")
        logger.info(f"resampler_key_padding_mask shape: {net_inputs[4].shape} {net_inputs[4].dtype}")

        xh_model.to(dtype)

        onnx_dir = out_model_dir
        onnx_file = onnx_dir / f"{cfg_name}_vision.onnx"

        convert_fx_model_to_hmonnx(
            xh_model._wrap_model,
            net_inputs,
            cfg.target_device,
            onnx_file,
            quant_config=cfg.quant_config,
            input_names=[
                "pixel_values",
                "position_ids",
                "attention_mask",
                "resampler_pos_embed",
                "resampler_key_padding_mask",
            ],
            output_names=["hidden_state"],
        )
        logger.info(f"Export onnx to {onnx_file}")

        meta_info.vision_hmonnx = str(Path(onnx_file).relative_to(onnx_dir))
        json.dump(meta_info, open(onnx_dir / "meta_info.json", "w"), indent=4)

        # from xhquant.api import HMONNXGoldenInference

        # hm_model = HMONNXGoldenInference(onnx_file)
        # hm_model.save_golden = True
        # hm_model.exec_device = device

        # golden_dir = Path(cfg.work_dir) / "golden" / f"{Path(onnx_file).stem}"
        # golden_dir.mkdir(exist_ok=True, parents=True)
        # hm_model.golden_dir = str(golden_dir)

        # with torch.no_grad():
        #     hm_model.forward(*net_inputs)  # type: ignore[arg-type]

    @classmethod
    def convert(cls, hf_model_path: str, config: MinicpmoVisionConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        config.quant_config = ConfigDict(quant_config)
        is_ssfp = is_ssfp_quant_config(quant_config)
        if is_ssfp:
            assert config.quant_weight is not None and Path(config.quant_weight).exists()
        MinicpmoVisionConverterXH2a(config)._convert(hf_model_path, output_dir)

