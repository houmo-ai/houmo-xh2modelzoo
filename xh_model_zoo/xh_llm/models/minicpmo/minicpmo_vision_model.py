# Copyright 2025 HOUMO AI
#
# File: minicpmo_vision_model.py
# Description:
#   Minicpmo Vision Model model implementation.
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

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

# import transformers_modules
from transformers import AutoModel

from ..base_model import BaseModel
from ..builder import MODELS, wrap_llm_model
from .minicpmo_base_model import XHMiniCPMOBaseModel


def _prepare_4d_attention_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask
    if dtype == torch.float16:
        return inverted_mask.masked_fill(inverted_mask.to(torch.bool), -65504)
    else:
        assert False


@MODELS.register_module()
class XHMiniCPMOVisionModel(XHMiniCPMOBaseModel):
    def __init__(
        self,
        hf_model: str,
        wrap_cfg,
        quant_config,
        frontend_type,
        allow_quant=True,
        export_cfg=None,
    ):
        super().__init__(
            hf_model,
            wrap_cfg=wrap_cfg,
            quant_config=quant_config,
            frontend_type=frontend_type,
            allow_quant=allow_quant,
            export_cfg=export_cfg,
        )

    def get_hf_model(self, device_map="cpu", **kwargs):
        hf_model = super().get_hf_model(device_map=device_map, **kwargs)
        self.resampler = hf_model.resampler
        return hf_model

    def get_position_ids(
        self,
        pixel_values: torch.Tensor,
        patch_attention_mask: torch.Tensor,
        tgt_sizes: Optional[torch.Tensor] = None,
    ):
        batch_size = pixel_values.size(0)
        max_im_h, max_im_w = pixel_values.size(2), pixel_values.size(3)
        # max_im_h = self.max_im_h
        # max_im_w = self.max_im_w
        max_nb_patches_h, max_nb_patches_w = max_im_h // self.patch_size, max_im_w // self.patch_size
        boundaries = torch.arange(1 / self.num_patches_per_side, 1.0, 1 / self.num_patches_per_side)
        position_ids = torch.full(
            size=(
                batch_size,
                max_nb_patches_h * max_nb_patches_w,
            ),
            fill_value=0,
        )

        for batch_idx, p_attn_mask in enumerate(patch_attention_mask):
            if tgt_sizes is not None:
                nb_patches_h = tgt_sizes[batch_idx][0]
                nb_patches_w = tgt_sizes[batch_idx][1]
            else:
                nb_patches_h = p_attn_mask[:, 0].sum()
                nb_patches_w = p_attn_mask[0].sum()

            fractional_coords_h = torch.arange(0, 1 - 1e-6, 1 / nb_patches_h)
            fractional_coords_w = torch.arange(0, 1 - 1e-6, 1 / nb_patches_w)

            bucket_coords_h = torch.bucketize(fractional_coords_h, boundaries, right=True)
            bucket_coords_w = torch.bucketize(fractional_coords_w, boundaries, right=True)

            pos_ids = (bucket_coords_h[:, None] * self.num_patches_per_side + bucket_coords_w).flatten()
            position_ids[batch_idx][p_attn_mask.view(-1).cpu()] = pos_ids

        return position_ids

    def init_wrap_model(self, hf_model=None):
        if hf_model is None:
            hf_model = self.get_hf_model()
        from ._vision_model_impl import register_wrap_cls as vision_register_wrap_cls  # noqa F401

        vision_register_wrap_cls(hf_model)
        vpm = hf_model.vpm
        self.patch_size = vpm.embeddings.patch_size
        self.num_patches_per_side = vpm.embeddings.num_patches_per_side
        vpm.resampler = hf_model.resampler
        self.resampler = hf_model.resampler
        return super().init_wrap_model(vpm)

    def _set_device(self, device: torch.device) -> None:
        super()._set_device(device)
        self.resampler.to(self.device)

    def prepare_inputs_for_graph(self, data: Dict[str, Union[torch.Tensor, Any]]) -> Any:
        inputs = self.prepare_inputs(data)
        all_pixel_values = inputs[0][:1]
        position_ids = inputs[1][:1]
        patch_attention_mask = inputs[2][:1]
        resampler_pos_embed = inputs[3][:, :1, :]
        resampler_key_padding_mask = inputs[4][:1]
        # tgt_sizes = inputs[5][:1]
        return (
            all_pixel_values,
            position_ids,
            patch_attention_mask,
            resampler_pos_embed,
            resampler_key_padding_mask,
        )

    def prepare_inputs(self, data: Dict[str, Union[torch.Tensor, Any]]) -> Any:
        pixel_values = data["pixel_values"]
        tgt_sizes = data["tgt_sizes"]

        pixel_values_list = pixel_values
        all_pixel_values: List[Tensor] = []
        imgs_cnt = []
        max_length = self.wrap_cfg.image_slice_max_size[0] * self.wrap_cfg.image_slice_max_size[1] * self.patch_size
        for pixel_values in pixel_values_list:
            imgs_cnt.append(len(pixel_values))
            # all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])
            for t in pixel_values:
                t = t.flatten(end_dim=1).permute(1, 0)
                s, f = t.shape
                padd_t = torch.zeros(max_length, f, dtype=t.dtype, device=t.device)
                padd_t[:s, :] = t
                all_pixel_values.append(padd_t)

        if all_pixel_values:
            tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
            tgt_sizes_t = torch.vstack(tgt_sizes).type(torch.int32)

            tgt_max_patches = torch.max(tgt_sizes_t[:, 0] * tgt_sizes_t[:, 1])
            max_patches = self.wrap_cfg.image_slice_max_size[0] * self.wrap_cfg.image_slice_max_size[1]
            assert tgt_max_patches <= max_patches
            assert torch.max(tgt_sizes_t[:, 0]) <= self.wrap_cfg.image_slice_max_size[0]
            assert torch.max(tgt_sizes_t[:, 1]) <= self.wrap_cfg.image_slice_max_size[1]
            #
            all_pixel_values_t = torch.nn.utils.rnn.pad_sequence(all_pixel_values, batch_first=True, padding_value=0.0)
            # all_pixel_values = torch.nn.utils.rnn.pad_packed_sequence(all_pixel_values, batch_first=True, )
            B, L, _ = all_pixel_values_t.shape
            all_pixel_values_t = all_pixel_values_t.permute(0, 2, 1).reshape(B, 3, -1, L)

            patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=self.device)
            for i in range(B):
                patch_attn_mask[i, 0, : tgt_sizes_t[i][0] * tgt_sizes_t[i][1]] = True
            patch_attn_mask = patch_attn_mask.view(B, -1)
            attention_mask = _prepare_4d_attention_mask(patch_attn_mask, torch.float16)

            position_ids = self.get_position_ids(all_pixel_values_t, patch_attn_mask, tgt_sizes=tgt_sizes_t)

            ## resampler
            key_padding_mask = torch.zeros((B, max_patches), dtype=torch.bool, device=self.device)
            pos_embed = []
            for i in range(B):
                tgt_h, tgt_w = tgt_sizes_t[i]
                patch_len = tgt_h * tgt_w
                b_pos_embed = self.resampler.pos_embed[:tgt_h, :tgt_w, :].reshape((tgt_h * tgt_w, -1)).to(self.dtype)
                s, f = b_pos_embed.shape
                # padd_pos_embed = torch.zeros((max_patch_len, f), dtype=dtype, device=device)
                # padd_pos_embed[:s, :] = b_pos_embed
                padd_pos_embed = torch.nn.functional.pad(b_pos_embed, (0, 0, 0, max_patches - s), value=0.0)
                pos_embed.append(padd_pos_embed)  # patches * D
                key_padding_mask[i, patch_len:] = True

            resampler_pos_embed = torch.nn.utils.rnn.pad_sequence(
                pos_embed, batch_first=True, padding_value=0.0
            ).permute(
                1, 0, 2
            )  # BLD => L * B * D
            resampler_key_padding_mask = torch.zeros_like(key_padding_mask, dtype=self.dtype).masked_fill_(
                key_padding_mask, -65504
            )

            # resampler_key_padding_mask = key_padding_mask

            return (
                all_pixel_values_t.to(self.device, self.dtype),
                position_ids.to(self.device),
                attention_mask.to(self.device, self.dtype),
                resampler_pos_embed,
                resampler_key_padding_mask,
                tgt_sizes_t,
                imgs_cnt,
            )
        else:
            assert False

    def _forward(
        self,
        all_pixel_values: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor,
        resampler_pos_embed: Tensor,
        resampler_key_padding_mask: Tensor,
        tgt_sizes: Tensor,
        imgs_cnt: List[int],
    ) -> List[Tensor]:
        vision_batch_size = 1
        all_pixel_values = all_pixel_values
        B = all_pixel_values.shape[0]
        hs = []
        for i in range(0, B):
            start_idx = i
            end_idx = i + vision_batch_size
            tmp_hs = self(
                all_pixel_values[start_idx:end_idx],
                position_ids[start_idx:end_idx],
                attention_mask[start_idx:end_idx],
                resampler_pos_embed[:, start_idx:end_idx],
                resampler_key_padding_mask[start_idx:end_idx],
                # tgt_sizes[start_idx:end_idx],
            )
            # tmp_hs = self.resampler(
            #     tmp_hs, resampler_pos_embed[:, start_idx:end_idx], resampler_key_padding_mask[start_idx:end_idx]
            # )
            hs.append(tmp_hs)
        vision_embedding = torch.cat(hs, dim=0)
        # vision_embedding = self.resampler(vision_embedding, resampler_pos_embed, resampler_key_padding_mask)
        start = 0
        vision_hidden_states = []
        for img_cnt in imgs_cnt:
            if img_cnt > 0:
                vision_hidden_states.append(vision_embedding[start : start + img_cnt])
                start += img_cnt
            else:
                vision_hidden_states.append(torch.tensor([]))
        return vision_hidden_states
