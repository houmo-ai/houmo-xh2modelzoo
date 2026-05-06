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
# File: gptq_utils.py
# Description:
#   single-shot quantization post-training quantization adapted for the xh2 model zoo (xh2modelzoo).

import torch,torch.nn as nn,torch.nn.functional as F
import math,time,tqdm,gc
import logging

def get_minq_maxq(bits, sym):
    if sym:
        maxq = torch.tensor(2**(bits-1)-1)
        minq = -maxq -1
    else:
        maxq = torch.tensor(2**bits - 1)
        minq = 0

    return minq, maxq

def asym_quant(x, scale, zero, maxq):
    scale = scale.to(x.device)
    zero = zero.to(x.device)
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return q, scale, zero

def asym_dequant(q, scale, zero):
    return scale * (q - zero)

def asym_quant_dequant(x, scale, zero, maxq):
    return asym_dequant(*asym_quant(x, scale, zero, maxq))

def sym_quant(x, scale, maxq):
    scale = scale.to(x.device)
    q = torch.clamp(torch.round(x / scale), -(maxq+1), maxq)
    return q, scale
def sym_dequant(q, scale):
    return scale * q

def sym_quant_dequant(x, scale, maxq):
    return sym_dequant(*sym_quant(x, scale, maxq))

class WeightQuantizer(torch.nn.Module):
    '''From GPTQ Repo'''

    def __init__(self, shape=1):
        super(WeightQuantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(
        self,
        bits, perchannel=False, sym=True,
        mse=False, norm=2.4, grid=100, maxshrink=.8,pot=False
    ):
        self.bits = bits
        self.perchannel = perchannel
        self.sym = sym
        self.mse = mse
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        if sym:
            self.maxq = torch.tensor(2**(bits-1)-1)
        else:
            self.maxq = torch.tensor(2**bits - 1)
        self.pot = pot

    def find_params(self, x):
        if self.bits == 16:
            return
        dev = x.device
        self.maxq = self.maxq.to(dev)

        shape = x.shape
        if self.perchannel:
            x = x.flatten(1)
        else:
            x = x.flatten().unsqueeze(0)

        tmp = torch.zeros(x.shape[0], device=dev)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.sym:
            # xmax = torch.maximum(torch.abs(xmin), xmax).clamp(min=1e-5) # TODO 
            # self.scale = xmax / self.maxq
            self.scale = torch.maximum((xmin.clip(max=0))/(-self.maxq-1) ,xmax.clip(min=1e-6)/(self.maxq))
            self.zero = torch.zeros_like(self.scale)
        else:
            tmp = (xmin == 0) & (xmax == 0)
            self.scale = (xmax - xmin).clamp(min=1e-5) / self.maxq
            self.zero = torch.round(-xmin / self.scale)

        if self.mse:
            best = torch.full([x.shape[0]], float('inf'), device=dev)
            for i in range(int(self.maxshrink * self.grid)):
                p = 1 - i / self.grid
                xmin1 = p * xmin
                xmax1 = p * xmax

                if self.sym:
                    scale1 = torch.maximum((xmin1.clip(max=0))/(-self.maxq-1) ,xmax1.clip(min=1e-6)/(self.maxq))
                    # scale1 = xmax1 / self.maxq
                    if self.pot:
                        scale1 = 2**(scale1.log2().ceil())
                    zero1 = torch.zeros_like(scale1)
                    q = sym_quant_dequant(x, scale1.unsqueeze(1), self.maxq)
                else:

                    scale1 = (xmax1 - xmin1) / self.maxq
                    zero1 = torch.round(-xmin1 / scale1)
                    q = asym_quant_dequant(x, scale1.unsqueeze(1), zero1.unsqueeze(1), self.maxq)

                q -= x
                q.abs_()
                q.pow_(self.norm)
                err = torch.sum(q, 1)
                tmp = err < best
                if torch.any(tmp):
                    best[tmp] = err[tmp]
                    self.scale[tmp] = scale1[tmp]
                    self.zero[tmp] = zero1[tmp]
        if not self.perchannel:

            tmp = shape[0]
            self.scale = self.scale.repeat(tmp)
            self.zero = self.zero.repeat(tmp)

        shape = [-1] + [1] * (len(shape) - 1)
        self.scale = self.scale.reshape(shape)
        self.zero = self.zero.reshape(shape)
        return

    # TODO: This should be better refactored into `forward`, which applies quantize and dequantize. A new method `quantize` should be added (if needed) to return the quantized integers and scales, like in ActQuantizer.
    def quantize(self, x):
        x_dtype = x.dtype
        if self.ready() and self.bits < 16:
            if self.sym:
                return sym_quant_dequant(x, self.scale, self.maxq).to(x_dtype)
            return asym_quant_dequant(x, self.scale, self.zero, self.maxq).to(x_dtype)
        return x

    def enabled(self):
        return self.maxq > 0

    def ready(self):
        return torch.all(self.scale != 0)

def find_qlayers(module, layers=[torch.nn.Linear], name=''):
    if isinstance(module,tuple(layers)):
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_qlayers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res

def clear_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if hasattr(self.layer, "L") and self.layer.L is not None:
            init_shape = inp.shape
            inp=inp.reshape(-1, self.layer.dim_l, self.layer.dim_r)
            inp=self.layer.L @ inp @ self.layer.R
            inp=inp.reshape(init_shape)
        else:
            pass
        
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())
        try:
            torch.linalg.cholesky(self.H)
        except:
            self.H = (self.H + self.H.t()) * 0.5
            mean_diag = torch.mean(torch.diag(self.H)).clamp_min(1e-12)
            self.H = self.H / mean_diag

    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False
    ):
        W = self.layer.weight.data.clone()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)])
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        # H = (H + H.t()) * 0.5
        # mean_diag = torch.mean(torch.diag(H)).clamp_min(1e-12)
        # H = H / mean_diag
        # eye = torch.eye(self.columns, device=self.dev, dtype=H.dtype)
        # jitter = 0.0
        # for _ in range(6):
        #     try:
        #         L = torch.linalg.cholesky(H + jitter * eye)
        #         break
        #     except RuntimeError:
        #         jitter = 1e-6 if jitter == 0.0 else jitter * 10
        # else:
        #     e, V = torch.linalg.eigh(H)
        #     e = e.clamp_min(1e-6)
        #     H = (V * e) @ V.t()
        #     H = torch.linalg.cholesky(H)
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        W_int = torch.zeros_like(W)
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)])
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]
                w_int = torch.clamp(
                    torch.round(w.unsqueeze(1) / self.quantizer.scale.to(w.device)),
                    -(self.quantizer.maxq + 1),
                    self.quantizer.maxq,
                )
                W_int[:, i1 + i] = w_int.squeeze(1)
                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        self.W_int = W_int
        if actorder:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if torch.any(torch.isnan(self.layer.weight.data)):
            logging.warning('NaN in weights')
            import pprint
            pprint.pprint(self.quantizer.bits, self.quantizer.scale, self.quantizer.zero_point)
            raise ValueError('NaN in weights')

    def free(self):
        self.H = None
        self.Losses = None
        self.Trace = None
        gc.collect()
        torch.cuda.empty_cache()

@torch.no_grad()
def gptq_fwrd(model, tokenizer, dataloader, dev, args):
    '''
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    print('-----GPTQ Quantization-----')
    if args.group_size is None:
        args.group_size = -1
    
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = model.lm_head.weight.dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}
    other_caches = dict()

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            if kwargs.get("cache_position",None) is not None:
                other_caches['cache_position'] = kwargs['cache_position']
            if kwargs.get("position_embeddings",None) is not None:
                other_caches['position_embeddings'] = kwargs['position_embeddings']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            try:
                model(batch[0].to(dev))
            except:
                texts = batch["text"]
                queries = [query for query in texts]
                inputs = tokenizer(queries, return_tensors="pt", truncation=True, max_length=model.seqlen,padding=True).to('cuda')
                inputs['input_ids'] =  F.pad(inputs['input_ids'],(0,2048-inputs['input_ids'].shape[1]))
                model(inputs['input_ids'])
        except:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    if attention_mask is not None:
        attention_mask = attention_mask[:1]
    position_ids = cache['position_ids']

    quantizers = {}
    if hasattr(args, 'quant_method') and args.quant_method=="singlequant":
        sequential = [
                ['self_attn.k_proj.linear'], 
                ['self_attn.v_proj.linear'], 
                ['self_attn.q_proj.linear'],
                ['self_attn.o_proj.linear'],
                ['mlp.up_proj.linear'], 
                ['mlp.gate_proj.linear'],
                ['mlp.down_proj.linear']
            ]
    else:
        sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
    print ('layers length: ',len(layers))
    for i in range(len(layers)):
    # for i in range(1):
        print(f'\nLayer {i}:', flush=True, end=' ')
        layer = layers[i].to(dev)
        full = find_qlayers(layer, layers=[torch.nn.Linear])
        for names in sequential:
            subset = {n: full[n] for n in names}
            gptq = {}
            for name in subset:
                print(f'{name}', end='  ', flush=True)
                layer_weight_bits = args.wbits
                layer_weight_sym = args.symmetric
                if 'lm_head' in name:
                    print ('attention')
                    layer_weight_bits = 16
                    continue
                # if args.int8_down_proj and 'down_proj' in name:
                #     layer_weight_bits = 8
                gptq[name] = GPTQ(subset[name])
                gptq[name].quantizer = WeightQuantizer()
                gptq[name].quantizer.configure(
                    layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.mse,pot=getattr(args,"w_pot",False)
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    gptq[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name))) # 搜集数据
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,**other_caches)[0]
            for h in handles:
                h.remove()

            for name in subset:
                layer_w_groupsize = args.group_size
                gptq[name].fasterquant(
                    percdamp=0.01, groupsize=layer_w_groupsize, actorder=args.act_order, static_groups=False
                )
                # quantizers['model.layers.%d.%s' % (i, name)] = gptq[name].quantizer
                quant_value = gptq[name].W_int
                if quant_value.max().item() < pow(2, 7) and quant_value.min().item() >= -pow(2, 7):
                    quant_value = quant_value.to(torch.int8)
                elif quant_value.max().item() < pow(2, 15) and quant_value.min().item() >= -pow(2, 15):
                    quant_value = quant_value.to(torch.int16)
                subset[name].register_parameter("quant_weight", nn.Parameter(quant_value, requires_grad=False))
                gptq[name].free()
        print ('i: ',i)
        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids,**other_caches)[0]

        layers[i] = layer.cpu()
        
        del layer
        del gptq 
        torch.cuda.empty_cache()

        inps, outs = outs, inps
    
    if True: # heading quant
        print(f"\nhead:", flush=True, end=" ")
        # Convert to module and move to CUDA
        model.lm_head = model.lm_head.to(device='cuda', dtype=torch.float32)
        gptq = {}
        name = "lm_head"
        layer_weight_bits = args.wbits
        layer_weight_sym = args.symmetric
        gptq[name] = GPTQ(model.lm_head)
        gptq[name].quantizer = WeightQuantizer()
        gptq[name].quantizer.configure(layer_weight_bits, perchannel=True, sym=layer_weight_sym, mse=args.mse)

        def add_batch(name):
            def tmp(_, inp, out):
                gptq[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        handles.append(model.lm_head.register_forward_hook(add_batch("lm_head")))
        for j in range(args.nsamples):
            model.lm_head(inps[j].unsqueeze(0).to(device='cuda', dtype=torch.float32))
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        for h in handles:
            h.remove()

        layer_w_groupsize = args.group_size
        gptq[name].fasterquant(
            percdamp=0.01,
            groupsize=layer_w_groupsize,
            actorder=args.act_order,
            static_groups=False,
        )

        quant_value = gptq[name].W_int
        if quant_value.max().item() < pow(2, 7) and quant_value.min().item() >= -pow(2, 7):
            quant_value = quant_value.to(torch.int8)
        elif quant_value.max().item() < pow(2, 15) and quant_value.min().item() >= -pow(2, 15):
            quant_value = quant_value.to(torch.int16)
        model.lm_head.register_parameter("quant_weight", nn.Parameter(quant_value, requires_grad=False))
        gptq[name].free()
        model.lm_head = model.lm_head.cpu()

        del gptq
        torch.cuda.empty_cache()
        print ('head gptq finished')


    model.config.use_cache = use_cache
    clear_cache()
    print('-----GPTQ Quantization Done-----\n')
    return quantizers



       
@torch.no_grad()
def rtn_fwrd(model, dev, args):
    '''
    From GPTQ repo 
    TODO: Make this function general to support both OPT and LLaMA models
    '''
    assert args.w_groupsize ==-1, "Groupsize not supported in RTN!"
    layers = model.model.layers
    clear_cache()

    quantizers = {}

    for i in tqdm.tqdm(range(len(layers)), desc="(RtN Quant.) Layers"):
        layer = layers[i].to(dev)

        subset = find_qlayers(layer,layers=[torch.nn.Linear])

        for name in subset:
            layer_weight_bits = args.wbits
            if 'lm_head' in name:
                layer_weight_bits = 16
                continue
            if hasattr(args,"int8_down_proj") and args.int8_down_proj and 'down_proj' in name:
                layer_weight_bits = 8

            quantizer = WeightQuantizer()
            quantizer.configure(
                layer_weight_bits, perchannel=True, sym=args.symmetric, mse=args.mse,pot=getattr(args,"w_pot",False)
            )
            W = subset[name].weight.data
            quantizer.find_params(W)
            subset[name].weight.data = quantizer.quantize(W).to(
                layer.self_attn.q_proj.weight.dtype)
            quantizers['model.layers.%d.%s' % (i, name)] = quantizer.cpu()
        layers[i] = layer.cpu()
        torch.cuda.empty_cache()
        del layer
            
    clear_cache()
    return quantizers
