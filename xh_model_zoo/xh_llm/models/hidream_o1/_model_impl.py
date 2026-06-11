from __future__ import annotations

import math
from typing import Optional

import accelerate
import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutputWithPast
from xhquant import nn as xhnn
from xhquant.utils.registry import DynamicModule

from .pipeline import ensure_hidream_o1_imports


ensure_hidream_o1_imports()

from models.qwen3_vl_transformers import (  # pyright: ignore[reportMissingImports]  # noqa: E402
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    Qwen3VLTextRMSNorm,
    Qwen3VLTextRotaryEmbedding,
    TimestepEmbedder,
    repeat_kv,
)


class _HiDreamTextRMSNorm(DynamicModule):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.norm(hidden_states)

    def _setup(self, cfg: Optional[dict] = None):
        hidden_size = self.weight.shape[0]
        self.norm = xhnn.RMSNorm(hidden_size, self.variance_epsilon)
        self.norm.weight = nn.Parameter(self.weight.data.clone())
        return self


class _HiDreamTextRotaryEmbedding(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        return self

    def apply_interleaved_mrope(self, freqs: torch.Tensor, mrope_section):
        pieces = []
        last_dim = freqs.shape[-1]
        len_h = int(mrope_section[1]) * 3
        len_w = int(mrope_section[2]) * 3
        for offset in range(0, last_dim, 3):
            pieces.append(freqs[0, ..., offset : offset + 1])
            if offset + 1 < last_dim:
                src = freqs[1] if offset + 1 < len_h else freqs[0]
                pieces.append(src[..., offset + 1 : offset + 2])
            if offset + 2 < last_dim:
                src = freqs[2] if offset + 2 < len_w else freqs[0]
                pieces.append(src[..., offset + 2 : offset + 3])
        return torch.cat(pieces, dim=-1)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
        inv_freq = getattr(self, "original_inv_freq", self.inv_freq)
        inv_freq_expanded = (
            inv_freq[None, None, :, None].float().to(device=x.device).expand(3, position_ids.shape[1], -1, 1)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (inv_freq_expanded @ position_ids_expanded).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class _HiDreamTextAttention(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        if not hasattr(self, "num_key_value_heads"):
            self.num_key_value_heads = self.config.num_key_value_heads
        if not hasattr(self, "num_heads"):
            self.num_heads = self.config.num_attention_heads
        # self.register_buffer("kv_scale", torch.tensor(self.head_dim**-0.5, dtype=torch.float16), persistent=False)
        self.kv_scale = self.head_dim**-0.5
        self.masked_add = xhnn.MaskedAdd()
        self.rope = xhnn.Rope()
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        del kwargs
        input_shape = hidden_states.shape[:-1]  # 1, 166

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(*input_shape, self.num_heads, self.head_dim)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim)
        ).transpose(1, 2)
        value_states = (
            self.v_proj(hidden_states).view(*input_shape, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        )

        cos, sin = position_embeddings
        query_states = self.rope(query_states, cos, sin)
        key_states = self.rope(key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.kv_scale
        if attention_mask is not None:
            # attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = self.masked_add(attn_weights, attention_mask)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None


class _HiDreamTextDecoderLayer(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        return self

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del kwargs
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class _HiDreamTextModel(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        return self

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        position_embeddings=None,
        use_cache=False,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        return_mid_results_layers=None,
        **kwargs,
    ):
        del input_ids, use_cache, kwargs
        if position_embeddings is None:
            if position_ids.ndim == 2:
                position_ids = position_ids[None, ...].expand(3, position_ids.shape[0], -1)
            elif position_ids.ndim == 3 and position_ids.shape[0] == 4:
                position_ids = position_ids[1:]
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        hidden_states = inputs_embeds
        mid_results = [] if return_mid_results_layers else None
        for layer_idx, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
            if (
                deepstack_visual_embeds is not None
                and visual_pos_masks is not None
                and layer_idx < len(deepstack_visual_embeds)
            ):
                hidden_states = self._deepstack_process(
                    hidden_states, visual_pos_masks, deepstack_visual_embeds[layer_idx]
                )
            if return_mid_results_layers is not None and layer_idx in return_mid_results_layers:
                mid_results.append(hidden_states)

        hidden_states = self.norm(hidden_states)
        output = BaseModelOutputWithPast(last_hidden_state=hidden_states)
        output.mid_results = mid_results
        return output


class _HiDreamTimestepEmbedder(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half, dtype=torch.float32) / half)
        self.register_buffer("timestep_freqs", freqs, persistent=False)
        return self

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        args = (t * 1000)[:, None].float() * self.timestep_freqs[None].to(t.device)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return self.mlp(embedding.to(self.mlp[0].weight.dtype))


class _HiDreamQwen3VLModel(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        embed_param = next(self.t_embedder1.parameters())
        device = embed_param.device
        dtype = embed_param.dtype
        fixed_timesteps = torch.tensor(
            [
                0.001,
                0.007,
                0.014,
                0.021,
                0.029,
                0.036,
                0.044,
                0.052,
                0.060,
                0.069,
                0.078,
                0.087,
                0.096,
                0.105,
                0.115,
                0.126,
                0.136,
                0.147,
                0.159,
                0.170,
                0.182,
                0.195,
                0.208,
                0.222,
                0.236,
                0.251,
                0.266,
                0.282,
                0.298,
                0.316,
                0.334,
                0.353,
                0.373,
                0.393,
                0.415,
                0.438,
                0.462,
                0.487,
                0.514,
                0.542,
                0.572,
                0.604,
                0.637,
                0.672,
                0.710,
                0.751,
                0.794,
                0.840,
                0.889,
                0.943,
            ],
            dtype=torch.float32,
            device=device,
        )
        fixed_tembs = []
        with torch.no_grad():
            for timestep in fixed_timesteps:
                fixed_tembs.append(self.t_embedder1(timestep.reshape(1)).detach().to(dtype=dtype))
        self.register_buffer("hidream_fixed_timestep_values", fixed_timesteps, persistent=False)
        self.register_buffer("hidream_fixed_temb", torch.cat(fixed_tembs, dim=0), persistent=False)
        return self

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: Optional[torch.Tensor] = None):
        pixel_values = pixel_values.type(self.visual.dtype)
        image_embeds, deepstack_image_embeds = self.visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        image_embeds = torch.split(image_embeds, split_sizes)
        return image_embeds, deepstack_image_embeds

    def get_placeholder_mask(
        self,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        video_features: Optional[torch.Tensor] = None,
    ):
        special_image_mask = input_ids == self.config.image_token_id
        special_video_mask = input_ids == self.config.video_token_id

        special_image_mask = special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if image_features is not None and inputs_embeds[special_image_mask].numel() != image_features.numel():
            raise ValueError(
                "Image features and image tokens do not match: "
                f"tokens: {(input_ids == self.config.image_token_id).sum()}, features {image_features.shape[0]}"
            )

        special_video_mask = special_video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        if video_features is not None and inputs_embeds[special_video_mask].numel() != video_features.numel():
            raise ValueError(
                "Video features and video tokens do not match: "
                f"tokens: {(input_ids == self.config.video_token_id).sum()}, features {video_features.shape[0]}"
            )

        return special_image_mask, special_video_mask

    def forward(
        self,
        input_ids=None,
        position_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        vinputs=None,
        timestep=None,
        token_types=None,
        t_emb=None,
        position_embeddings=None,
        use_flash_attn: bool = False,
        return_mid_results_layers=None,
        **kwargs,
    ):
        if vinputs is None:
            return self._orig_forward(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                **kwargs,
            )
        del pixel_values_videos, video_grid_thw, use_flash_attn, kwargs

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        visual_pos_masks = None
        deepstack_visual_embeds = None
        if pixel_values is not None:
            image_embeds, deepstack_image_embeds = self.get_image_features(pixel_values, image_grid_thw)
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds

        if t_emb is None:
            # timestep = timestep.to(inputs_embeds.device, dtype=torch.long).reshape(-1)
            t_emb = self.hidream_fixed_temb[timestep]
        else:
            t_emb = t_emb.to(inputs_embeds.device, inputs_embeds.dtype)
        # HiDream text-to-image prompt built by build_t2i_text_sample always appends
        # one <|tms_token|> at the end of text tokens. Avoid torch.where because
        # xhquant quant graph does not support it.
        inputs_embeds = torch.cat([inputs_embeds[:, :-1, :], t_emb.unsqueeze(1)], dim=1)

        # vinputs = vinputs.to(inputs_embeds.device)
        vinputs_embedded = self.x_embedder(vinputs)
        inputs_embeds = torch.cat([inputs_embeds, vinputs_embedded], dim=1)

        batch_size, total_seq_len, _ = inputs_embeds.shape
        if visual_pos_masks is not None:
            vinputs_pad = torch.zeros(
                visual_pos_masks.shape[0],
                vinputs_embedded.shape[1],
                dtype=visual_pos_masks.dtype,
                device=visual_pos_masks.device,
            )
            visual_pos_masks = torch.cat([visual_pos_masks, vinputs_pad], dim=1)

        # token_types = token_types.to(inputs_embeds.device)
        # if token_types.dim() == 1:
        #     token_types = token_types.unsqueeze(0)
        # if token_types.shape[0] == 1 and batch_size > 1:
        #     token_types = token_types.expand(batch_size, -1)

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            position_embeddings=position_embeddings,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            return_mid_results_layers=return_mid_results_layers,
        )
        hidden_states = outputs.last_hidden_state
        x_pred = self.final_layer2(hidden_states)

        output = BaseModelOutputWithPast(last_hidden_state=hidden_states)
        output.x_pred = x_pred
        output.mid_results = getattr(outputs, "mid_results", None)
        output.cond_image_embeds = None
        output.cond_deepstack_image_embeds = None
        return output


class _HiDreamQwen3VLForConditionalGeneration(DynamicModule):
    def _setup(self, cfg: Optional[dict] = None):
        return self

    def forward(self, *args, **kwargs):
        outputs = self.model(*args, **kwargs)
        if kwargs.get("vinputs", None) is not None:
            return outputs
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        outputs.logits = logits
        return outputs


def register_hidream_o1_wrap_modules(model: nn.Module) -> nn.Module:
    model = accelerate.hooks.remove_hook_from_module(model, recurse=True)
    wrapped_classes = {
        Qwen3VLTextRMSNorm: _HiDreamTextRMSNorm,
        Qwen3VLTextRotaryEmbedding: _HiDreamTextRotaryEmbedding,
        Qwen3VLTextAttention: _HiDreamTextAttention,
        Qwen3VLTextDecoderLayer: _HiDreamTextDecoderLayer,
        Qwen3VLTextModel: _HiDreamTextModel,
        TimestepEmbedder: _HiDreamTimestepEmbedder,
        Qwen3VLModel: _HiDreamQwen3VLModel,
        Qwen3VLForConditionalGeneration: _HiDreamQwen3VLForConditionalGeneration,
    }
    for _, module in list(model.named_modules()):
        dynamic_cls = wrapped_classes.get(type(module))
        if dynamic_cls is not None and not isinstance(module, DynamicModule):
            if type(module) is Qwen3VLModel:
                module._orig_forward = module.forward
            dynamic_cls.convert(module)
    return model
