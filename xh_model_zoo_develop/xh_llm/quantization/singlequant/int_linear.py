# -*- coding: utf-8 -*-
# Copyright 2024 The OmniQuant Authors. All rights reserved.
# Copyright 2025 HOUMO AI. All rights reserved.
#
# Modifications:
# - Portions of this file have been modified by HOUMO AI.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# File: int_linear.py
# Description:
#   single-shot quantization integer linear layer adapted for the xh2 model zoo (xh2modelzoo).

import torch
import torch.nn as nn
import torch.nn.functional as F
from xhquant_llm.singlequant.quantizer import UniformAffineQuantizer
import time


class QuantLinear(nn.Linear):
    def __init__(
        self,
        org_module: nn.Linear,
        weight_quant_params: dict = {},
        act_quant_params: dict = {},
        disable_input_quant=False,
        rotate=True,
        weight_dominated=False,
        fake_quant=False,
    ):
        super().__init__(org_module.in_features, org_module.out_features, org_module.bias is not None)
        self.fwd_kwargs = dict()
        self.fwd_func = F.linear
        self.weight = org_module.weight
        self.bias = org_module.bias
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(
            **weight_quant_params, shape=org_module.weight.shape, rotate=rotate
        )
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(
                **act_quant_params, rotate=rotate
            )
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False
        self.init_singlequant_params = (
            torch.tensor(0)
            if weight_quant_params["quant_method"] == "singlequant"
            else torch.tensor(1)
        )
        self.weight_dominated = weight_dominated
        self.fake_quant = fake_quant
        
    def forward(self, input: torch.Tensor):
        if self.weight_dominated:
            if self.use_temporary_parameter:
                weight = self.temp_weight
                bias = self.temp_bias
            elif self.use_weight_quant:
                weight = self.weight_quantizer(self.weight)
                bias = self.bias
            else:
                weight = self.weight
                bias = self.bias
            if self.use_act_quant and not self.disable_input_quant:
                if not self.init_singlequant_params:
                    self.act_quantizer.copy_singlequant_params(self.weight_quantizer)
                    self.init_singlequant_params = torch.tensor(1)
                input = self.act_quantizer(input)
            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
            return out
        else:
            if self.use_act_quant and not self.disable_input_quant:
                input = self.act_quantizer(input)
            if self.use_temporary_parameter:
                weight = self.temp_weight
                bias = self.temp_bias
            elif self.use_weight_quant:
                if not self.init_singlequant_params:
                    self.weight_quantizer.copy_singlequant_params(self.act_quantizer)
                    self.init_singlequant_params = torch.tensor(1)
                weight = self.weight_quantizer(self.weight)
                bias = self.bias
            else:
                weight = self.weight
                bias = self.bias
            out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)

            return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def copy_quantizers_singlequant_params(self, proj):
        assert proj.init_singlequant_params
        self.init_singlequant_params = torch.tensor(1)
        self.weight_quantizer.copy_singlequant_params(proj.weight_quantizer)
        self.act_quantizer.copy_singlequant_params(proj.act_quantizer)
