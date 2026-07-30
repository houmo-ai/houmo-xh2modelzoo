# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from safetensors import safe_open

from wan.modules.t5 import HuggingfaceTokenizer, umt5_xxl


def load_torch_state_dict(checkpoint_path: str):
    if checkpoint_path.endswith('.safetensors'):
        state_dict = {}
        scale_weights = {}
        with safe_open(checkpoint_path, framework='pt', device='cpu') as f:
            for key in f.keys():
                if key.endswith('.scale_weight'):
                    scale_weights[key[:-len('.scale_weight')]] = f.get_tensor(key)
                    continue
                state_dict[key] = f.get_tensor(key)
        for key, scale in scale_weights.items():
            weight_key = f'{key}.weight'
            if weight_key in state_dict and state_dict[weight_key].dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
                state_dict[weight_key] = state_dict[weight_key].float() * scale.float()
        return convert_hf_umt5_state_dict(state_dict)
    return torch.load(checkpoint_path, map_location='cpu')



def convert_hf_umt5_state_dict(state_dict):
    converted = {}
    for key, value in state_dict.items():
        if key in {'scaled_fp8', 'spiece_model'}:
            continue
        if key == 'shared.weight':
            converted['token_embedding.weight'] = value
            continue
        if key == 'encoder.final_layer_norm.weight':
            converted['norm.weight'] = value
            continue
        if key.startswith('encoder.block.'):
            parts = key.split('.')
            if len(parts) < 6:
                continue
            block = parts[2]
            layer = parts[4]
            name = '.'.join(parts[5:])
            prefix = f'blocks.{block}'
            if layer == '0':
                if name == 'layer_norm.weight':
                    converted[f'{prefix}.norm1.weight'] = value
                elif name.startswith('SelfAttention.'):
                    attn_name = name[len('SelfAttention.'):]
                    if attn_name == 'relative_attention_bias.weight':
                        converted[f'{prefix}.pos_embedding.embedding.weight'] = value
                    elif attn_name in {'q.weight', 'k.weight', 'v.weight', 'o.weight'}:
                        converted[f'{prefix}.attn.{attn_name}'] = value
            elif layer == '1':
                if name == 'layer_norm.weight':
                    converted[f'{prefix}.norm2.weight'] = value
                elif name == 'DenseReluDense.wi_0.weight':
                    converted[f'{prefix}.ffn.gate.0.weight'] = value
                elif name == 'DenseReluDense.wi_1.weight':
                    converted[f'{prefix}.ffn.fc1.weight'] = value
                elif name == 'DenseReluDense.wo.weight':
                    converted[f'{prefix}.ffn.fc2.weight'] = value
            continue
        converted[key] = value
    return converted


class LocalT5EncoderModel:
    def __init__(
        self,
        text_len,
        dtype=torch.bfloat16,
        device=torch.cuda.current_device(),
        checkpoint_path=None,
        tokenizer_path=None,
        shard_fn=None,
    ):
        self.text_len = text_len
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.tokenizer_path = tokenizer_path

        model = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=dtype,
            device=device,
        ).eval().requires_grad_(False)
        logging.info('loading %s', checkpoint_path)
        model.load_state_dict(load_torch_state_dict(checkpoint_path), strict=True)
        self.model = model
        if shard_fn is not None:
            self.model = shard_fn(self.model, sync_module_states=False)
        else:
            self.model.to(self.device)
        self.tokenizer = HuggingfaceTokenizer(name=tokenizer_path, seq_len=text_len, clean='whitespace')

    def __call__(self, texts, device):
        ids, mask = self.tokenizer(texts, return_mask=True, add_special_tokens=True)
        ids = ids.to(device)
        mask = mask.to(device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.model(ids, mask)
        return [u[:v] for u, v in zip(context, seq_lens)]
