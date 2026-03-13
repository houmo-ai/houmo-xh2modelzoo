# Copyright 2025 HOUMO AI
#
# File: hmonnx_export_prefill_decode.py
# Description: Export GLM-ASR LLM prefill and decode phases to HMONNX
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
import time
import json
import shutil
import argparse

import torch
import torch.nn as nn

from pathlib import Path
from typing import List, Tuple
from xhquant.api import ptq_quantize

from xhquant.api import Config, ConfigDict, PrecisionMode, get_root_logger, ptq_quantize
from xhquant.common.types import PrecisionMode
from xhquant.utils.config import ConfigDict

from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.eval_model_type import EvalModelType

from xh_model_zoo.xh_llm.models.glm_asr import (
    GlmAsrForConditionalGeneration,
    XHGlmAsrLLMModel,
)

GB = int(2**30)
_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 1.8)
FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def xhmodel_export_onnx(
    xh_model,
    tokenizer,
    data_batch,
    onnx_output_dir: str,
    cfg_name,
    device,
    dtype,
    logger,
    valid: bool = True,
):
    logger.info("Start exporting...")
    xh_model.to("cpu")  # Switch to CPU for model export
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    xh_model.convert_to_export_graph(data_batch)
    logger.info("Finish converting to export graph...")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    xh_model.change_eval_type(EvalModelType.EXPORTED)

    xh_model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("*************** Start exporting onnx ***************")
    onnx_file = xh_model.to_export_onnx(data_batch, onnx_output_dir, cfg_name)[0]
    return onnx_file


def main(args):
    # ============================================================ Config and Initialization ============================================================
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = os.path.normpath(args.model)

    hf_model = GlmAsrForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        dtype=torch.float16,
        device_map=DEVICE,
    )
    hf_model.eval()

    model_name = os.path.basename(MODEL_PATH)
    target_device = "XH2a"  # Target quantization device

    cfg = Config.fromfile(args.config)
    cfg_name = f"{model_name}_{target_device}"

    cfg.work_dir = str(Path("work_dirs") / cfg_name)
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.exec_device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg.dtype = "float16"
    logger = get_root_logger()
    logger.info(f"\nConfig:\n{cfg.pretty_text}")
    config_file = Path(cfg.work_dir) / Path(args.config).name
    cfg.dump(config_file)

    device = torch.device(cfg.device)
    exec_device = torch.device(cfg.exec_device)
    dtype = getattr(torch, cfg.dtype)

    xh_model = MODELS.build(cfg.model)

    model = xh_model.get_hf_model()
    assert isinstance(xh_model, XHGlmAsrLLMModel), f"Model must be XHGlmAsrLLMModel, but got {type(xh_model)}"

    # For GLM-ASR, the LLM is accessed via language_model, not thinker.model
    xh_model.init_wrap_model(hf_model.language_model)

    xh_model.wrap_model.lm_head = hf_model.language_model.lm_head
    xh_model.wrap_model.lm_head.to(device)
    xh_model.wrap_model.lm_head.to(dtype)

    processor = xh_model.get_processor()

    prefill_onnx_dir = Path(cfg.work_dir) / "Prefill"
    prefill_golden_path = prefill_onnx_dir / "hmonnx/golden"
    decode_onnx_dir = Path(cfg.work_dir) / "Decoder"
    decode_golden_path = decode_onnx_dir / "hmonnx/golden"
    prefill_onnx_dir.mkdir(exist_ok=True, parents=True)
    decode_onnx_dir.mkdir(exist_ok=True, parents=True)

    meta_info = ConfigDict(
        dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            config=str(config_file.relative_to(cfg.work_dir)),
        )
    )
    meta_info["wrap_cfg"] = xh_model.wrap_cfg.to_dict()
    hf_model_dir = cfg.hf_model_dir
    meta_info.hf_model = hf_model_dir

    hf_model_config_dir = cfg.config_dir

    hf_config_dir = Path(cfg.work_dir) / "ConfigFiles"
    hf_config_dir.mkdir(exist_ok=True, parents=True)
    hf_config_files = [
        "chat_template.json",
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
        "configuration.json",
        "generation_config.json",
        "merges.txt",
        "preprocessor_config.json",
    ]

    for cfg_file in hf_config_files:
        src_file = Path(hf_model_config_dir) / cfg_file
        if src_file.exists():
            shutil.copyfile(src_file, Path(hf_config_dir) / cfg_file)
    meta_info.hf_config = str(hf_config_dir.relative_to(cfg.work_dir))

    token_embedding = xh_model.token_embedding
    token_embedding_file = Path(cfg.work_dir) / "token_embedding.pt"
    torch.save(token_embedding.state_dict(), str(token_embedding_file))
    meta_info.token_embedding_file = str(token_embedding_file.relative_to(cfg.work_dir))

    # xh_model.past_key_caches is already wrapped as list in init_wrap_model
    if xh_model.past_key_caches is not None and len(xh_model.past_key_caches) > 0:
        meta_info.use_cache = True
        meta_info.kv_cache_shape = xh_model.past_key_caches[0].shape
        meta_info.num_hidden_layers = len(xh_model.past_key_caches)

    # ============================================================ Move model to device ============================================================

    # wrapped decoder
    xh_model.to(device)
    xh_model.to(dtype)
    xh_model.change_eval_type(eval_type=EvalModelType.WRAPED)
    # entire model - since wrap_llm_model modifies in-place, moving the original
    # HF language_model also moves the _wrap_model
    hf_model.language_model.to(device)

    hf_model.config.forced_decoder_ids = None
    hf_model.config._attn_implementation = "eager"
    text_config = hf_model.config.text_config

    # ============================================================ Audio Text Preprocessing and Feature Fusion ============================================================

    tokenizer = processor.tokenizer

    hidden_size = text_config.hidden_size

    # Create dummy input embeds (simulating fused audio + text features)
    final_inputs_embeds = torch.randn((1, 411, hidden_size), device=device, dtype=torch.float16)
    print(f"final_inputs_embeds.shape: {final_inputs_embeds.shape}")

    # ============================================================ Construct Inputs ============================================================

    seq_len = final_inputs_embeds.shape[1]
    # Pad second dimension to 411
    if seq_len < 411:
        final_inputs_embeds = torch.cat(
            [
                final_inputs_embeds,
                torch.zeros((1, 411 - seq_len, final_inputs_embeds.shape[2]), dtype=torch.float16, device=device),
            ],
            dim=1,
        )

    data_batch = {"input_embeds": final_inputs_embeds.half(), "past_seq_length": [0]}

    with torch.no_grad():
        outs = xh_model.test_step(data_batch)

    # ============================================================ Quantization ============================================================

    xh_model.interactive_mode = True
    logger.info("************* convert to frontend graph *************")

    xh_model.convert_to_fronted_graph(data_batch)
    logger.info(f"************* Start Frontend Graph *************")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("************* convert to quanted graph *************")

    xh_model.convert_to_quant_graph(target_device)

    xh_model.change_eval_type(EvalModelType.CALIBRATION)
    xh_model.enable_calibration()
    xh_model.to(dtype)
    xh_model.to(device)

    logger.info("*************** Start PTQ Quantize ***************")

    calib_data = xh_model.prepare_inputs(data_batch)
    new_args = []
    for arg in calib_data:
        if isinstance(arg, (List, Tuple)):
            new_args.extend(arg)
        else:
            new_args.append(arg)
    calib_data = new_args
    ptq_quantize(xh_model.quanted_model, [calib_data], PrecisionMode.ALIGNED, [exec_device])
    logger.info("*************** Finished PTQ Quantize **************")

    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

    # ============================================================ Prefill Export ============================================================

    xh_model = xh_model.to("cpu")
    full_seq_len = 411

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    work_dir = Path("work_dirs") / cfg_name
    work_dir.mkdir(exist_ok=True, parents=True)

    # export_cfg: expand past_key_cache and past_value_cache inputs into multiple inputs for alignment
    num_hidden_layers = text_config.num_hidden_layers
    base_inputs = ["input_embeds", "past_seq_length", "current_input_length"]
    key_names = [f"past_key_cache_{i}" for i in range(num_hidden_layers)]
    value_names = [f"past_value_cache_{i}" for i in range(num_hidden_layers)]
    input_names = base_inputs + key_names + value_names
    xh_model.export_cfg = ConfigDict(dict(input_names=input_names, output_names=["last_hidden_state"]))

    xh_model.set_input_sequence_length(full_seq_len)

    prefill_onnx_file = xhmodel_export_onnx(
        xh_model,
        tokenizer,
        data_batch,
        str(prefill_onnx_dir),
        f"{cfg_name}_prefill",
        "cpu",
        dtype,
        logger,
        False,
    )

    if args.gen_golden and not Path(prefill_golden_path).exists():
        from xhquant.api import HMONNXGoldenInference

        session = HMONNXGoldenInference(prefill_onnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = str(prefill_onnx_dir / "hmonnx/golden")
        session.step = 0
        session(*calib_data)

    xh_model.release_exported_model()
    logger.info(f"save prefill onnx model to {prefill_onnx_file}")
    logger.info("*************** Finished exporting prefill model ***************")
    meta_info.prefill_onnx_file = str(Path(prefill_onnx_file).relative_to(cfg.work_dir))

    # ============================================================ Decode Export ============================================================
    xh_model.change_eval_type(EvalModelType.QUANTED_ALIGNED)
    xh_model.to(device)
    xh_model.to(dtype)

    data_batch["input_embeds"] = data_batch["input_embeds"].to(device)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Set input sequence length to 1 (Decode mode)
    xh_model.set_input_sequence_length(1)

    # Prepare Decode stage Embedding and data batch
    past_seq_len = final_inputs_embeds.shape[1]
    prefill_next_token_embeds = final_inputs_embeds[:, -1:, :]
    final_inputs_embeds = prefill_next_token_embeds
    logger.info(f"past_seq_len: {past_seq_len}")

    data_batch = {
        "input_embeds": final_inputs_embeds.to(device),
        "past_seq_length": [past_seq_len],
    }

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("*************** Start exporting decode model ***************")

    # Switch to CPU for export preparation
    xh_model = xh_model.to("cpu")

    decode_cpu_inputs_embeds = data_batch["input_embeds"].to("cpu")
    if decode_cpu_inputs_embeds.dim() == 2:
        decode_cpu_inputs_embeds = decode_cpu_inputs_embeds.unsqueeze(0)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Execute export
    decode_onnx_file = xhmodel_export_onnx(
        xh_model,
        tokenizer,
        data_batch,
        str(decode_onnx_dir),
        f"{cfg_name}_decode",
        "cpu",
        dtype,
        logger,
        False,
    )

    meta_info.decode_onnx_file = str(Path(decode_onnx_file).relative_to(cfg.work_dir))
    json.dump(meta_info, open(Path(cfg.work_dir) / "export_meta_info.json", "w"), indent=4)

    decode_inputs = xh_model.prepare_inputs(data_batch)
    decode_calib_data = []
    for arg in decode_inputs:
        if isinstance(arg, (List, Tuple)):
            decode_calib_data.extend(arg)
        else:
            decode_calib_data.append(arg)

    if args.gen_golden and not Path(decode_golden_path).exists():
        from xhquant.api import HMONNXGoldenInference

        session = HMONNXGoldenInference(decode_onnx_file)
        session.to("cuda")
        session.save_golden = True
        session.golden_dir = str(decode_onnx_dir / "hmonnx/golden")
        session.step = 0
        session(*decode_calib_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="glm-asr-nano-2512")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(FILE_DIR, "config/llm/glm_asr_decode_xh2a.py"),
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8_sefp", help="quant type")
    parser.add_argument("--gen_golden", action="store_true", help="generate golden data")
    args = parser.parse_args()
    main(args)
