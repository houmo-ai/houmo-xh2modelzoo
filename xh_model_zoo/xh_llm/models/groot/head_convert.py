# Copyright 2025 HOUMO AI
#
# File: dit_converter.py
# Description:
#   Dit Converter implementation.
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
import shutil
import time
from copy import deepcopy
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union
from numpy import dtype
import onnx
from regex import B
from sympy import false
import torch
import torch.nn as nn
from PIL import Image
from transformers.quantizers.quantizer_gptq import GptqHfQuantizer
from xhquant.api import convert_fx_model_to_quanted_model, convert_onnx_to_hmonnx, convert_quanted_model_to_hmonnx, convert_dynamo_model_to_quanted_model
from xhquant.utils.onnxsim_large_model.simplify_large_onnx import simplify_large_onnx

from ..base_converter import BaseConverter, HFTransfromersConverter
from ..builder import wrap_llm_model

from xhquant.api import (  # isort:skip
    Config,
    DeviceType,
    ConfigDict,
    get_root_logger,
    create_quant_config,
    is_ssfp_quant_config,
    CacheTensor,
)
from transformers.feature_extraction_utils import BatchFeature

class Network_pre(nn.Module):
    def __init__(self, encoder, action_encoder=None, position_embedding=None, vlln=None, device="cuda", state_encoder=None, action_decoder=None, dit=None, tag_id=20):
        super().__init__()
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.position_embedding = position_embedding    
        self.vlln = vlln
        self.state_encoder = state_encoder
        self.device = device
        pos_ids = torch.arange(50, dtype=torch.long, device=device)
        self.pos_embs = self.position_embedding(pos_ids).unsqueeze(0).half()
        self.action_encoder = action_encoder.to(device)
        self.action_decoder = action_decoder.to(device)
        self.model = dit.to(device)
        self.action_horizon = 50
        self.tag_id = tag_id

    def forward(self, backbone_features, state):
        # backbone_features  [1, 109, 2048]
        # state [1, 1, 128]
        # timesteps_tensor [0]
        # actions   [1, 50, 128]
        # [1,109]
        # [1,109]
        device = self.device
        embodiment_id=torch.tensor([self.tag_id], device=device)

        # features = self.encoder(backbone_output, action_input)
        backbone_features = self.vlln(backbone_features)
        state_features = self.state_encoder(state, embodiment_id)
        # vl_embeds = backbone_features

        return backbone_features, state_features


class Netwokr(nn.Module):
    def __init__(self, encoder, action_encoder=None, position_embedding=None, vlln=None, device="cuda", state_encoder=None, action_decoder=None, dit=None, tag_id=20):
        super().__init__()
        self.encoder = encoder
        self.action_encoder = action_encoder
        self.position_embedding = position_embedding    
        self.device = device
        pos_ids = torch.arange(50, dtype=torch.long, device=device)
        self.pos_embs = self.position_embedding(pos_ids).unsqueeze(0).half()
        self.action_encoder = action_encoder.to(device)
        self.action_decoder = action_decoder.to(device)
        self.model = dit.to(device)
        self.action_horizon = 50
        self.tag_id = tag_id

    def forward(self, backbone_features, state_features, timesteps_tensor=None, actions=None, image_mask=None, backbone_attention_mask=None):
        # backbone_features  [1, 109, 2048]
        # state [1, 1, 128]
        # timesteps_tensor [0]
        # actions   [1, 50, 128]
        # [1,109]
        # [1,109]


        device = self.device
        embodiment_id=torch.tensor([self.tag_id], device=device)


        # backbone_features = self.vlln(backbone_features)
        # state_features = self.state_encoder(state, embodiment_id)
        vl_embeds = backbone_features



        # Embed noised action trajectory.
        action_features = self.action_encoder(actions, timesteps_tensor, torch.tensor([self.tag_id], device=device, dtype=torch.float16))
        action_features = action_features + self.pos_embs

        # Join vision, language, state and action embedding along sequence dimension.
        sa_embs = torch.cat((state_features, action_features), dim=1)

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embeds,
            timestep=timesteps_tensor,
            image_attention_mask=image_mask,
            non_image_attention_mask=backbone_attention_mask,
        )

        pred = self.action_decoder(model_output, embodiment_id)

        pred_velocity = pred[:, -self.action_horizon :]

        # Update actions using euler integration.
        actions = actions + 0.25 * pred_velocity
        return actions

class Groot_HEAD_ConverterXH2a(HFTransfromersConverter):
    target_device = DeviceType.XH2a

    def __init__(self, config, device="cuda"):
        super().__init__()
        self.config = config
        self.work_dir = None
        self.wraped_llm_model = None
        self.device = device

    def _convert(self, hf_model, output_dir: str, postprocess=None, tag_id=20):
        logger = get_root_logger()
        config = self.config

        native_model = hf_model.half()
        with torch.no_grad():
            backbone_features = torch.rand(1, 256, 2048).to(self.device).half()
            state = torch.rand(1, 1, 128).to(self.device).half()
            timesteps_tensor = torch.tensor([0]).to(self.device)
            # actions = torch.load("/data01/home/xuchen/xh2/xh2_model_zoo/work_dirs/actions.pt").to(self.device)  # For testing with a fixed action trajectory
            actions = torch.rand(1, 50, 128).to(self.device).half()
            image_mask = torch.zeros(1, 256).to(self.device).to(torch.bool)
            image_mask[:, 26:109] = 1
            backbone_attention_mask = torch.ones(1, 256).to(self.device).to(torch.bool)
            backbone_outputs = {
                "backbone_features": backbone_features,
                "image_mask": image_mask,
                "backbone_attention_mask": backbone_attention_mask,
            }
            embodiment_id = torch.tensor([tag_id]).to(self.device)
            action_inputs = {
                "embodiment_id": embodiment_id,
                "state": state,
            }

            backbone_outputs = BatchFeature( data = backbone_outputs )
            action_inputs = BatchFeature( data = action_inputs )

            ori_output = native_model.get_action(backbone_outputs, action_inputs)


        from ._head import register_wrap_modules as vision_register_wrap_modules  # noqa: F401
        vision_register_wrap_modules()

        wrap_cfg = Config(
            dict(
                batch_size=1,
                token_len=256,
                num_logits_to_keep=0,
                input_sequence_length=324,
                use_cache=false,
                max_sequence_length=888888,
            )
        )

        wraped_llm_model = wrap_llm_model(native_model, wrap_cfg)
        wraped_llm_model = wraped_llm_model.to(self.device)
        wraped_llm_model.to(torch.float16)

        new_network = Netwokr(
            encoder=native_model._encode_features,
            action_encoder=native_model.action_encoder,
            position_embedding=native_model.position_embedding,
            device=self.device,
            vlln=native_model.vlln,
            state_encoder=native_model.state_encoder,
            action_decoder=native_model.action_decoder,
            dit=native_model.model,
            tag_id=tag_id,
        ).to(self.device).half()

        new_network_pre = Network_pre(
            encoder=native_model._encode_features,
            action_encoder=native_model.action_encoder,
            position_embedding=native_model.position_embedding,
            device=self.device,
            vlln=native_model.vlln,
            state_encoder=native_model.state_encoder,
            action_decoder=native_model.action_decoder,
            dit=native_model.model,
            tag_id=tag_id,
        ).to(self.device).half()

        with torch.no_grad():
            # backbone_features = torch.rand(1, 256, 2048).to(self.device).half()
            # state = torch.rand(1, 1, 128).to(self.device).half()
            # timesteps_tensor = torch.tensor([0]).to(self.device)
            # actions = torch.rand(1, 50, 128).to(self.device).half()
            # image_mask = torch.ones(1, 256).to(self.device).to(torch.bool)
            # backbone_attention_mask = torch.ones(1, 256).to(self.device).to(torch.bool)
            image_attention_mask= image_mask & backbone_attention_mask
            non_image_attention_mask= (~image_mask) & backbone_attention_mask

            image_attn_reshape = image_attention_mask.repeat(32,1).view(1,32,1,256)
            non_image_attn_reshape = non_image_attention_mask.repeat(32,1).view(1,32,1,256)

            image_attention_mask = torch.zeros_like(image_attn_reshape, dtype=torch.float16)
            image_attention_mask[image_attn_reshape == False] = -65504

            non_image_attention_mask = torch.zeros_like(non_image_attn_reshape, dtype=torch.float16)
            non_image_attention_mask[non_image_attn_reshape == False] = -65504

            backbone_features, state_features = new_network_pre(
                backbone_features=backbone_features,
                state=state,
            )

            outoput_warp = new_network(
                backbone_features=backbone_features,
                state_features=state_features,
                timesteps_tensor=timesteps_tensor,
                actions=actions,
                image_mask=image_attention_mask,
                backbone_attention_mask=non_image_attention_mask
            )
        # ========================================= head pre
        model_name = "groot_head_pre"
        target_device = config.quant_scheme.target_device

        quant_config = create_quant_config(config.quant_scheme)
        work_dir = Path(output_dir)
        self.work_dir = output_dir
        quant_config = ConfigDict(quant_config)

        Path(work_dir / "hmonnx").mkdir(exist_ok=True, parents=True)
        quant_type = config.quant_scheme.quant_type
        prefix = f"{model_name}-{target_device}-{quant_type}"

        target_device = self.config.quant_scheme.target_device
        input_names = ["backbone_features", "state"]
        inputs = [backbone_features, state]
 
        onnx_output_names = ["backbone_featureso", "state_features"]
        head_onnx_file = str(work_dir / "hmonnx" / f"{prefix}.onnx")
        quant_graph_model = None
        new_network_pre = new_network_pre.to(self.device)
        if not Path(head_onnx_file).exists():
            logger.info("********************* start export head model *********************")
            
            quant_graph_model = convert_fx_model_to_quanted_model(
                new_network_pre, inputs, target_device, quant_config
            )
            convert_quanted_model_to_hmonnx(
                quant_graph_model, inputs, head_onnx_file, input_names, onnx_output_names
            )
        else:
            logger.info(f"{head_onnx_file} exists, skip export head model.")


        head_golden_dir = "/data02/users/cc_work/golden/groot/head_pre" # str(work_dir / "golden" / f"{prefix}")
        if not Path(head_golden_dir).exists():
            logger.info(f"start export vision model golden............")
            from xhquant.api import HMONNXGoldenInference

            vae_model = HMONNXGoldenInference(head_onnx_file)
            vae_model.save_golden = True
            vae_model.exec_device = torch.device("cuda:0")

            Path(head_golden_dir).mkdir(exist_ok=True, parents=True)
            vae_model.golden_dir = str(head_golden_dir)

            inputs = [backbone_features, state]
            with torch.no_grad():
                vae_model.forward(*inputs)
            logger.info(f"Export head model golden to {head_golden_dir}")
        else:
            logger.info(f"{head_golden_dir} exists, skip export vision model golden.")        

        # ========================================= head
        model_name = "groot_head"
        target_device = config.quant_scheme.target_device
        # batch_size = config.batch_size
        assert target_device == DeviceType.XH2a, f"Only support convert to XH2a, but got {target_device}"

        quant_config = create_quant_config(config.quant_scheme)
        work_dir = Path(output_dir)
        self.work_dir = output_dir
        quant_config = ConfigDict(quant_config)


        meta_info = dict(
            create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        )
        meta_info["device"] = str(target_device)
        meta_info["model_name"] = model_name
        meta_info["quant_scheme"] = config.quant_scheme.to_dict()
        meta_info["wrap_cfg"] = wrap_cfg.to_dict()

        Path(work_dir / "hmonnx").mkdir(exist_ok=True, parents=True)
        quant_type = config.quant_scheme.quant_type
        prefix = f"{model_name}-{target_device}-{quant_type}"

        # latent = torch.rand(1, 3, 252, 252).to(self.device)


        target_device = self.config.quant_scheme.target_device
        input_names = ["backbone_features", "state_features", "timesteps_tensor", "actions", "image_mask", "backbone_attention_mask"]
        inputs = [backbone_features, state_features, timesteps_tensor, actions, image_attention_mask, non_image_attention_mask]
 
        onnx_output_names = ["act_pred"]
        head_onnx_file = str(work_dir / "hmonnx" / f"{prefix}.onnx")
        quant_graph_model = None
        new_network = new_network.to(self.device)
        if not Path(head_onnx_file).exists():
            logger.info("********************* start export head model *********************")
            
            quant_graph_model = convert_fx_model_to_quanted_model(
                new_network, inputs, target_device, quant_config
            )
            convert_quanted_model_to_hmonnx(
                quant_graph_model, inputs, head_onnx_file, input_names, onnx_output_names
            )
        else:
            logger.info(f"{head_onnx_file} exists, skip export head model.")

        meta_info["onnx"] = str(Path(head_onnx_file).relative_to(work_dir))
        head_golden_dir = str(work_dir / "golden" / f"{prefix}")

        if not Path(head_golden_dir).exists():
            logger.info(f"start export vision model golden............")
            from xhquant.api import HMONNXGoldenInference

            vae_model = HMONNXGoldenInference(head_onnx_file)
            vae_model.save_golden = True
            vae_model.exec_device = torch.device("cuda:0")

            Path(head_golden_dir).mkdir(exist_ok=True, parents=True)
            vae_model.golden_dir = str(head_golden_dir)

            inputs = [backbone_features, state_features, timesteps_tensor.to(torch.int32), actions, image_attention_mask, non_image_attention_mask]
            with torch.no_grad():
                vae_model.forward(*inputs)
            logger.info(f"Export head model golden to {head_golden_dir}")
        else:
            logger.info(f"{head_golden_dir} exists, skip export vision model golden.")


        # meta_info["decoder_golden_dir"] = str(Path(head_golden_dir))
        # json.dump(meta_info, open(work_dir / "meta_vision.json", "w"), indent=4)