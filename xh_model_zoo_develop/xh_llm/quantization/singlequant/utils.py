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
# File: utils.py
# Description:
#   single-shot quantization utility helpers adapted for the xh2 model zoo (xh2modelzoo).

from collections import OrderedDict
from xhquant_llm.singlequant.int_linear import QuantLinear
import torch
import torch.nn as nn
from xhquant_llm.singlequant.int_matmul import QuantMatMul
from xhquant_llm.singlequant.quantizer import UniformAffineQuantizer
from xhquant_llm.singlequant.transformation import *
import pickle
from xhquant_llm.singlequant.const import CLIPMIN

def smooth_parameters(model, use_shift=True):
    params = []
    for n, m in model.named_parameters():
        if n.find('smooth') > -1:
            params.append(m)
    return iter(params)

def let_parameters(model, use_shift=True):
    params = []
    # template = "smooth" if use_shift else "smooth_scale"
    template = "post_scale"
    for n, m in model.named_parameters():
        if n.find(template) > -1:
            params.append(m)
    return iter(params)  

def lwc_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1:
            params.append(m)
    return iter(params)

def theta_parameters(model):
    params = []
    for n, m in model.named_parameters():
        if n.find('thetas') > -1:
            params.append(m)
    return iter(params)


def get_singlequant_parameters(model, use_shift=True):
    params = []
    template = "smooth" if use_shift else "smooth_scale"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template) > -1:
            params.append(m)
    return iter(params)  

def get_post_parameters(model):
    params = []
    template1 = "post_scale"
    template2 = "thetas"
    for n, m in model.named_parameters():
        if n.find('bound_factor') > -1 or n.find(template1) > -1 or n.find(template2) > -1:
            params.append(m)
    return iter(params)

def set_requires_grad(it, requires_grad):
    for param in it:
        param.requires_grad = requires_grad

def singlequant_state_dict(model, destination=None, prefix='', keep_vars=False):
    if destination is None:
        destination = OrderedDict()
    for name, param in model.named_parameters():
        if name.find('smooth') > -1 or name.find('bound_factor') > -1 or name.find('trans') > -1 or name.find('post')>-1:
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, param in model.named_buffers():
        if name.find('init_singlequant_params') > -1 or name.find('R') > -1 or name.find('permutation_list') > -1:
            destination[prefix + name] = param if keep_vars else param.detach()
    return destination

def register_scales_and_zeros(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_scales_and_zeros()

class TruncateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, threshold):
        truncated_tensor = input.clone()
        truncated_tensor[truncated_tensor.abs() < threshold] = truncated_tensor[truncated_tensor.abs() < threshold].sign() * threshold
        return truncated_tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

     
def truncate_number(number, threshold=1e-2):
    # avoid overflow with AMP training
    return TruncateFunction.apply(number, threshold)     




@torch.no_grad()   
def post_quant_inplace(model, args):
    if args.let:
        for name, module in model.named_parameters():
            if "post_scale" in name:
                module.data = truncate_number(module)
            if isinstance(module, QuantLinear):
                if module.act_quantizer.let_s is not None:
                    module.act_quantizer.let_s.requires_grad = False
                if module.weight_quantizer.let_s is not None:
                    module.weight_quantizer.let_s.requires_grad = False
            
def clear_temp_variable(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                del module.temp_weight
            if hasattr(module, "temp_bias"):
                del module.temp_bias

def set_registered_x_none(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.registered_x = None
            module.act_quantizer.registered_x = None

@torch.no_grad()
def register_singlequant_params(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight_quantizer.register_singlequant_params()
            module.act_quantizer.register_singlequant_params()

@torch.no_grad()
def set_init_singlequant_params_state(model, mode):
    if isinstance(mode, bool):
        mode = torch.tensor(mode)
    for name, module in model.named_modules():
        if hasattr(module, "init_singlequant_params"):
            module.init_singlequant_params = mode

def smooth_and_quant_temporary(model, args, isllama):
    if args.smooth:
        with torch.no_grad():
            for name, module in model.named_parameters():
                if "smooth_scale" in name:
                    module.data = truncate_number(module)
        if isllama:
            smooth_ln_fcs_temporary(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_fc_fc_temporary(model.mlp.up_proj,model.mlp.down_proj,
                            model.down_smooth_scale, model.down_smooth_shift)
            smooth_fc_fc_temporary(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.mlp.down_proj.temp_weight = model.mlp.down_proj.weight
        else:
            smooth_ln_fcs_temporary(model.self_attn_layer_norm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj],
                                    model.qkv_smooth_scale,model.qkv_smooth_shift)
            smooth_ln_fcs_temporary(model.final_layer_norm,[model.fc1],
                                    model.fc1_smooth_scale,model.fc1_smooth_shift)
            smooth_ln_fcs_temporary(model.self_attn.v_proj,model.self_attn.out_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_temporary(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
            model.fc2.temp_weight = model.fc2.weight
    else:
        for name, module in model.named_modules():
            if isinstance(module, QuantLinear):
                module.temp_weight = module.weight
    # quant
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            if hasattr(module, "temp_weight"):
                module.temp_weight = module.weight_quantizer(module.temp_weight)
            else:
                module.temp_weight = module.weight_quantizer(module.weight)
            if not hasattr(module, "temp_bias"):
                module.temp_bias = module.bias
            module.use_temporary_parameter=True
        

@torch.no_grad()   
def smooth_and_let_inplace(model, args):
    if args.smooth:
        for name, module in model.named_parameters():
            if "smooth_scale" in name:
                module.data = truncate_number(module)
        smooth_ln_fcs_inplace(model.input_layernorm,[model.self_attn.q_proj, model.self_attn.k_proj, model.self_attn.v_proj], model.qkv_smooth_scale,model.qkv_smooth_shift)
        smooth_ln_fcs_inplace(model.post_attention_layernorm,[model.mlp.up_proj,model.mlp.gate_proj],
                                model.fc1_smooth_scale,model.fc1_smooth_shift)
        smooth_fc_fc_inplace(model.mlp.up_proj,model.mlp.down_proj,
                            model.down_smooth_scale, model.down_smooth_shift)
        try:
            smooth_fc_fc_inplace(model.self_attn.v_proj,model.self_attn.o_proj,
                                model.out_smooth_scale, model.out_smooth_shift)
            smooth_q_k_inplace(model.self_attn.q_proj, model.self_attn.k_proj,
                                model.qkt_smooth_scale)
        except:
            smooth_fc_inplace(model.self_attn.o_proj, model.out_smooth_scale)
    
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.use_temporary_parameter=False
        
@torch.no_grad()
def quant_inplace(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight = module.weight_quantizer(module.weight, return_no_quant=False)

@torch.no_grad()
def quant_soft_inplace(model):
    for name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            module.weight.data = module.weight_quantizer(module.weight, return_no_quant=True)

def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
    # setting weight quantization here does not affect actual forward pass
    self.use_weight_quant = weight_quant
    self.use_act_quant = act_quant
    for m in self.modules():
        if isinstance(m, (QuantLinear, QuantMatMul)):
            m.set_quant_state(weight_quant, act_quant)
