# Copyright 2025 HOUMO AI
#
# File: minicpmo_tts_convert copy.py
# Description:
#   Minicpmo Tts Convert Copy implementation.
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
from typing import Any, Dict, List, Optional, Tuple

import librosa
import torch
from moviepy import VideoFileClip
from PIL import Image
from transformers import AutoTokenizer
import numpy as np


from ..base_converter import HFTransfromersConverter
from .minicpmo_tts_convert_config import MinicpmoTTSConvertConfig

from xhquant.api import (  # type: ignore # isort:skip
    ConfigDict,
    DeviceType,
    create_quant_config,
    get_root_logger,
    is_ssfp_quant_config,
)
from xhquant.utils import set_random_seed
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType
from xh_model_zoo.xh_llm.models.minicpmo import MiniCPMO_HFCompatible


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


class MinicpmoTTSConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config: MinicpmoTTSConvertConfig):
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

        out_model_dir = Path(cfg.work_dir) / "hmonnx" / "tts"
        out_model_dir.mkdir(exist_ok=True, parents=True)
        config_file = str(out_model_dir / "tts_config.json")

        dtype = getattr(torch, cfg.dtype)
        device = torch.device(cfg.device)

        tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model_dir, trust_remote_code=True)

        from .minicpmo_tts_model import XHMiniCPMOTTSModel

        xh_model = XHMiniCPMOTTSModel(
            hf_model=cfg.hf_model_dir,
            frontend_type="TorchFX",
            wrap_cfg=ConfigDict(
                batch_size=1,
                max_sequence_length=cfg.context_length,
                input_sequence_length=cfg.input_sequence_length,
                use_cache=True,
                num_logits_to_keep=1,
                kv_cache=dict(cache_axis=2),
                image_slice_max_size=cfg.image_slice_max_size,
            ),
            quant_config=ConfigDict(),
            export_cfg=ConfigDict(
                dict(
                    input_names=[
                        "inputs_embeds",
                        "past_seq_length",
                        "current_input_length",
                        "attention_mask",
                    ],
                    output_names=["logits", "hidden_state"],
                )
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

        native_model.to(device)
        native_model.to(dtype)

        native_model.init_tts()

        xh_model.init_wrap_model()
        xh_model.wrap_processor(native_model)
        MiniCPMO_HFCompatible.to_hf_compatible(native_model, tts_llama_model=xh_model)

        xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
        xh_model.to(dtype)
        xh_model.to(device)

        # capture prefill/decode inputs
        input_datas: List[Dict[str, torch.Tensor]] = []
        original_forward = native_model.tts._tts_llama_model._forward

        def forward_with_hook(*args, **kwargs):
            captured = {}
            for k, v in kwargs.items():
                captured[k] = v
            input_datas.append(captured)
            return original_forward(*args, **kwargs)

        native_model.tts._tts_llama_model._forward = forward_with_hook
        with torch.no_grad():
            native_model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                sampling=True,
                temperature=0.5,
                max_new_tokens=64,
                omni_input=True,
                use_tts_template=True,
                generate_audio=True,
                output_audio_path="output.wav",
                max_slice_nums=1,
                use_image_id=False,
                return_dict=True,
            )
        native_model.tts._tts_llama_model._forward = original_forward

        if len(input_datas) < 2:
            raise RuntimeError("Failed to capture TTS prefill/decode inputs.")

        data_prefill = input_datas[0]
        data_decode = input_datas[1]

        # convert to hmonnx using existing export helpers on BaseModel
        from xhquant.api import convert_fx_model_to_hmonnx
        from xh_model_zoo.xh_llm.models.base_llm_model import LLMBaseModel

        def _prepare_inputs(data: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
            tensors: List[torch.Tensor] = []
            ordered_keys = ["inputs_embeds", "past_seq_length", "current_input_length", "attention_mask"]
            for k in ordered_keys:
                t = data[k]
                if t.dtype in (torch.int64, torch.int32):
                    t = t.to(torch.int32)
                else:
                    t = t.to(dtype)
                t = t.to(device)
                tensors.append(t)
            # kv caches
            # for cache in xh_model.past_key_caches:
            #     cache_tensor = cache.data if hasattr(cache, "data") else cache
            #     cache_tensor = cache_tensor.to(device).to(dtype)
            #     # flatten CacheTensor into list if wrapped
            #     if isinstance(cache_tensor, (list, tuple)):
            #         tensors.extend(cache_tensor)
            #     else:
            #         tensors.append(cache_tensor)
            # for cache in xh_model.past_value_caches:
            #     cache_tensor = cache.data if hasattr(cache, "data") else cache
            #     cache_tensor = cache_tensor.to(device).to(dtype)
            #     if isinstance(cache_tensor, (list, tuple)):
            #         tensors.extend(cache_tensor)
            #     else:
            #         tensors.append(cache_tensor)
            tensors.append(xh_model.past_key_caches)
            tensors.append(xh_model.past_value_caches)
            return tensors

        # prefer model-provided export_cfg names (already populated with kv cache names)
        input_names = list(getattr(xh_model.export_cfg, "input_names", []))
        if not input_names:
            num_layers = len(xh_model.past_key_caches)
            input_names = [
                "inputs_embeds",
                "past_seq_length",
                "current_input_length",
                "attention_mask",
            ]
            input_names.extend([f"past_key_cache_{i}" for i in range(num_layers)])
            input_names.extend([f"past_value_cache_{i}" for i in range(num_layers)])

        output_names = list(getattr(xh_model.export_cfg, "output_names", ["logits", "hidden_state"]))
        if not output_names:
            output_names = ["logits", "hidden_state"]

        # prefill
        prefill_inputs = _prepare_inputs(data_prefill)
        # input_names = [f"input_{i}" for i in range(len(prefill_inputs))]
        # output_names = ["logits", "hidden_state"]
        prefill_dir = Path(cfg.work_dir) / "hmonnx" / "tts"
        prefill_dir.mkdir(exist_ok=True, parents=True)
        prefill_onnx = prefill_dir / f"{cfg_name}_prefill.onnx"
        convert_fx_model_to_hmonnx(
            xh_model._wrap_model,
            prefill_inputs,
            cfg.target_device,
            prefill_onnx,
            quant_config=cfg.quant_config,
            input_names=input_names,
            output_names=output_names,
        )

        # decode
        # adjust current_input_length to 1 if needed
        data_decode["current_input_length"] = torch.tensor([1], device=device, dtype=torch.int32)
        decode_inputs = _prepare_inputs(data_decode)
        decode_onnx = prefill_dir / f"{cfg_name}_decode.onnx"
        convert_fx_model_to_hmonnx(
            xh_model._wrap_model,
            decode_inputs,
            cfg.target_device,
            decode_onnx,
            quant_config=cfg.quant_config,
            input_names=input_names,
            output_names=output_names,
        )

        meta_info = ConfigDict(
            dict(
                create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                config=str(Path(config_file).relative_to(out_model_dir)),
                prefill=str(Path(prefill_onnx).relative_to(prefill_dir)),
                decode=str(Path(decode_onnx).relative_to(prefill_dir)),
            )
        )
        meta_info["hf_model"] = xh_model.hf_model_dir
        meta_info["wrap_cfg"] = xh_model.wrap_cfg.to_dict()
        json.dump(meta_info, open(out_model_dir / "meta_info.json", "w"), indent=4)

    @classmethod
    def convert(cls, hf_model_path: str, config: MinicpmoTTSConvertConfig, output_dir: str):
        quant_config = create_quant_config(config.quant_scheme)
        config.quant_config = ConfigDict(quant_config)
        if is_ssfp_quant_config(quant_config):
            assert config.quant_weight is not None and Path(config.quant_weight).exists()
        MinicpmoTTSConverterXH2a(config)._convert(hf_model_path, output_dir)

