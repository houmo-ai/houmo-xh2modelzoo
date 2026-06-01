# Copyright 2025 HOUMO AI
#
# File: hmonnx_export_encoder.py
# Description: Export GLM-ASR audio encoder to HMONNX format
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

import os
import sys
import json

import librosa
import argparse
import tempfile

import onnx
import onnxsim
import importlib.util

import torch
import torch.nn as nn

from pathlib import Path

from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    ptq_quantize,
    to_frontend_graph,
    to_quant_graph,
)
from xhquant.api.ptq_export_hmonnx import (
    convert_quanted_model_to_hmonnx,
)
from xhquant.common.types import PrecisionMode
from xhquant.core.datatype_mapping import TORCH_DTYPE_TO_FAKE_DTYPE
from xhquant.frontend.convert import to_frontend_graph
from xhquant.patch.core import RewriterContext
from xhquant.utils.config import ConfigDict
from xh_model_zoo.xh_llm.models.builder import wrap_llm_model

from xh_model_zoo.xh_llm.models.glm_asr import GlmAsrForConditionalGeneration

from transformers import AutoProcessor

GB = int(2**30)
_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 1.8)


class AudioEncoderWithProjector(nn.Module):
    """Wraps audio_tower + reshape + multi_modal_projector into a single exportable module.

    Input:  input_features (B, num_mel_bins, T)
    Output: audio_embeds   (B, T_out, text_hidden_size)
    """

    def __init__(self, audio_tower, multi_modal_projector, intermediate_size):
        super().__init__()
        self.audio_tower = audio_tower
        self.multi_modal_projector = multi_modal_projector
        self.intermediate_size = intermediate_size

    def forward(self, input_features):
        audio_outputs = self.audio_tower(input_features, return_dict=True)
        hidden_states = audio_outputs.last_hidden_state
        # Merge every 4 time-steps: (B, T, encoder_hidden) -> (B, T/4, intermediate_size)
        hidden_states = hidden_states.reshape(input_features.shape[0], -1, self.intermediate_size)
        audio_embeds = self.multi_modal_projector(hidden_states)
        return audio_embeds


def main(args):
    target_device = "XH2a"
    model_dir = os.path.normpath(args.model)
    model_name = os.path.basename(model_dir)

    model = GlmAsrForConditionalGeneration.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)

    model.eval()
    model.audio_tower.eval()
    # DEVICE = torch.device("cpu")
    # model.to(DEVICE)
    model.config.forced_decoder_ids = None
    model.config._attn_implementation = "eager"

    cfg_name = f"{model_name}_{target_device}"

    work_dir = Path("work_dirs") / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)

    # Get audio config
    head_dim = model.config.audio_config.head_dim
    num_heads = model.config.audio_config.num_attention_heads
    num_key_value_heads = model.config.audio_config.num_key_value_heads
    embed_dim = model.config.audio_config.hidden_size
    num_decode_layers = model.config.audio_config.num_hidden_layers
    max_source_positions = model.config.audio_config.max_position_embeddings

    meta_info = {}
    meta_info_file = work_dir / "meta_info.json"
    if meta_info_file.exists():
        with open(meta_info_file, "r", encoding="utf-8") as f:
            meta_info = json.load(f)
    meta_info["hf_model"] = model_dir
    meta_info["model_cfg"] = {
        "head_dim": head_dim,
        "num_heads": num_heads,
        "num_key_value_heads": num_key_value_heads,
        "embed_dim": embed_dim,
        "max_source_positions": max_source_positions,
        "num_decode_layers": num_decode_layers,
    }

    # encoder processing =======================================================================================
    name = "Encoder"
    encoder_work_dir = work_dir / name
    encoder_work_dir.mkdir(exist_ok=True, parents=True)
    onnx_file = encoder_work_dir / f"{model_name}_{name}.onnx"
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)
    hmonnx_file = encoder_work_dir / "hmonnx" / f"{model_name}_{name}_xh2a_{quant_type}.onnx"
    golden_path = encoder_work_dir / "hmonnx/golden"
    meta_info["encoder"] = str(hmonnx_file.relative_to(work_dir))
    num_mel_bins = model.config.audio_config.num_mel_bins

    # Fixed at T=3000 seq_lens
    model = model.to(torch.float32)
    input_features = torch.randn(1, num_mel_bins, 3000).to(model.device).to(model.dtype)

    # Build combined module: audio_tower + reshape + multi_modal_projector
    intermediate_size = model.config.audio_config.intermediate_size
    encoder_with_proj = AudioEncoderWithProjector(model.audio_tower, model.multi_modal_projector, intermediate_size)
    encoder_with_proj.eval()

    # 1. Export ONNX
    if not Path(onnx_file).exists():
        with tempfile.TemporaryDirectory() as tmp_dir:
            with RewriterContext(None, backend="onnxruntime"):
                temp_onnx_file = str(Path(tmp_dir) / Path(onnx_file).name)
                torch.onnx.export(
                    encoder_with_proj,
                    input_features,
                    temp_onnx_file,
                    input_names=["input_features"],
                    output_names=["audio_embeds"],
                )
                onnx_model = onnx.load(temp_onnx_file)
                model_byte_size = onnx_model.ByteSize()
                if model_byte_size <= _LARGE_MODEL_SIZE_THRESHOLD:
                    onnx_model_sim, checked = onnxsim.simplify(
                        onnx_model,
                        skipped_optimizers=[
                            "fuse_pad_into_conv",
                            "fuse_consecutive_slices",
                            "eliminate_common_subexpression",
                            "fuse_qkv",
                        ],
                    )
                else:
                    from xhquant.utils.onnxsim_large_model import simplify_large_onnx

                    onnx_model_sim, checked = simplify_large_onnx(
                        onnx_model,
                        skipped_optimizers=[
                            "fuse_pad_into_conv",
                            "fuse_consecutive_slices",
                            "eliminate_common_subexpression",
                            "fuse_qkv",
                        ],
                    )
                    if checked:
                        onnx_model = onnx_model_sim
    else:
        onnx_model = onnx.load(onnx_file)
    if not os.path.exists(onnx_file):
        onnx.save(
            onnx_model,
            onnx_file,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=f"{Path(onnx_file).stem}_external_data",
        )
    print(f"ONNX model saved: {onnx_file}, size: {onnx_model.ByteSize() / GB:.2f} GB")

    # 2. Construct output names
    output_names = []
    output_names.append("audio_embeds")

    # 3. Convert to HMONNX
    if not Path(hmonnx_file).exists():
        convert_onnx_to_hmonnx(
            str(onnx_file),
            [input_features],
            DeviceType.XH2a,
            hmonnx_file,
            quant_config=quant_config,
            input_names=["input_features"],
            output_names=output_names,
        )

    # Generate golden data
    if args.gen_golden and not Path(golden_path).exists():
        session = HMONNXGoldenInference(hmonnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = str(encoder_work_dir / "hmonnx/golden")
        session.step = 0
        session(input_features.half().to("cuda"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="glm-asr-nano-2512")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type, default is w8a8")
    parser.add_argument("--gen_golden", action="store_true", help="generate golden data")
    args = parser.parse_args()
    main(args)
