# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import torch
from safetensors import safe_open

from wan.modules.vae2_1 import WanVAE_


def load_torch_state_dict(pretrained_path: str, device: str = 'cpu'):
    if pretrained_path.endswith('.safetensors'):
        state_dict = {}
        with safe_open(pretrained_path, framework='pt', device='cpu') as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
        return state_dict
    return torch.load(pretrained_path, map_location=device)



def _video_vae(pretrained_path=None, z_dim=None, device='cpu', **kwargs):
    cfg = dict(
        dim=96,
        z_dim=z_dim,
        dim_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attn_scales=[],
        temperal_downsample=[False, True, True],
        dropout=0.0,
    )
    cfg.update(**kwargs)

    with torch.device('meta'):
        model = WanVAE_(**cfg)

    logging.info('loading %s', pretrained_path)
    model.load_state_dict(load_torch_state_dict(pretrained_path, device=device), assign=True)
    return model


class LocalWan2_1_VAE:
    def __init__(
        self,
        z_dim=16,
        vae_pth='cache/vae_step_411000.pth',
        dtype=torch.float,
        device='cuda',
    ):
        self.dtype = dtype
        self.device = device

        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=dtype, device=device)
        self.std = torch.tensor(std, dtype=dtype, device=device)
        self.scale = [self.mean, 1.0 / self.std]
        self.model = _video_vae(pretrained_path=vae_pth, z_dim=z_dim).to(dtype).to(device)
        self.model.eval().requires_grad_(False)

    def encode(self, vids):
        device = self.device
        if isinstance(vids, torch.Tensor):
            vids = vids.unsqueeze(0).to(device, self.dtype)
            out = self.model.encode(vids, self.scale).float()
            return [u for u in out]
        out = [self.model.encode(u.unsqueeze(0).to(device, self.dtype), self.scale).float() for u in vids]
        return out

    def decode(self, zs):
        device = self.device
        if isinstance(zs, torch.Tensor):
            zs = zs.unsqueeze(0).to(device, self.dtype)
            out = self.model.decode(zs, self.scale)
            out = out.float().clamp_(-1, 1).cpu()
            return [u for u in out]
        out = [self.model.decode(u.unsqueeze(0).to(device, self.dtype), self.scale) for u in zs]
        out = [u.float().clamp_(-1, 1) for u in out]
        return out
