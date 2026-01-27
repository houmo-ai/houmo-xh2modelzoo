# Copyright 2025 HOUMO AI
#
# File: pipeline_cus.py
# Description:
#   Pipeline Cus implementation.
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

from diffusers.pipelines import QwenImagePipeline
import inspect
from typing import Any, Callable, Dict, List, Optional, Union
import torch
import numpy as np
from diffusers.pipelines.qwenimage.pipeline_output import QwenImagePipelineOutput
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps
import json
from xhquant.api import ConfigDict
from modelscope import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
import torch.nn as nn
from pathlib import Path
from .data_preprocess import Qwen2_5_VLDataPreprocess
from xhquant.core import CacheTensor


class cus_QwenImagePipeline(QwenImagePipeline):
    def __init__(self, *args, **kwargs):
        model_dir = Path(kwargs["meta_info"]).parent

        meta_info_path = kwargs["meta_info"]
        meta_info = json.load(open(meta_info_path, "r"))
        self.meta_info = ConfigDict(meta_info)       

        # hf_model_config_dir = str(model_dir / meta_info["hf_config"])
        # self.tokenizer = AutoTokenizer.from_pretrained(hf_model_config_dir) 

        token_embedding_state_dict = torch.load(
            model_dir / self.meta_info["token_embedding_file"], map_location="cpu", weights_only=False
        )
        self.token_embedding = token_embedding_state_dict
        # self.token_embedding.load_state_dict(token_embedding_state_dict)
    
    def text_encoder_infer(self, input_ids):
        data_prefill = {
            "input_ids": input_ids,
            "image_embeds": None,
            "past_seq_length": 0,
            "image_grid_thw": None,
        }

        input_sequence_length = 256

        input_seq_len = data_prefill["input_ids"].shape[-1]
        steps = (input_seq_len + input_sequence_length - 1) // input_sequence_length

        data_preprocess = Qwen2_5_VLDataPreprocess(
            self.token_embedding,
            input_sequence_length * steps,
        )
        data_input = data_preprocess(data_prefill)      

        kv_cache_shape = self.meta_info["kv_cache"]["shape"]
        num_decoder_layers = self.meta_info["kv_cache"]["num_decoder_layers"]
        past_key_caches = []
        past_value_caches = []
        for i in range(num_decoder_layers):
            past_k_cache = CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            past_v_cache = CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            past_key_caches.append(past_k_cache)
            past_value_caches.append(past_v_cache)  

        (
            inputs_embeds,
            time_position_ids,
            height_position_ids,
            width_position_ids,
            past_seq_length,
            current_seq_length,
        ) = data_input

        for i in range(steps):
            start = i * input_sequence_length
            end = (i + 1) * input_sequence_length
            current_input_length = min(end, input_seq_len) - start
            output_hidden = self._text_encoder(
                inputs_embeds[:, start:end, :],
                time_position_ids[:input_sequence_length],
                height_position_ids[:input_sequence_length],
                width_position_ids[:input_sequence_length],
                past_seq_length,
                torch.tensor([input_sequence_length], dtype=torch.int32).to(inputs_embeds.device),
                *past_key_caches,
                *past_value_caches,
            )
            past_seq_length += current_input_length
        return output_hidden

    def __setup__(self, text_encoder, transformer=None, vae=None):
        """"""
        """
        初始化模型
        """
        # self._prefill = True
        self._text_encoder = text_encoder
        # self.embed_tokens = llm_model.token_embedding
        # self._past_seq_length = 0
        self._transformer = transformer
        self._vae = vae
        return self

    def _get_qwen_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._text_encoder.exec_device
        dtype = dtype or self._text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.prompt_template_encode
        drop_idx = self.prompt_template_encode_start_idx
        txt = [template.format(e) for e in prompt]
        txt_tokens = self.tokenizer(
            txt, max_length=self.tokenizer_max_length + drop_idx, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        encoder_hidden_states = self.text_encoder_infer(txt_tokens.input_ids)
        # encoder_hidden_states = self._text_encoder(
        #     input_ids=txt_tokens.input_ids,
        #     attention_mask=txt_tokens.attention_mask,
        #     output_hidden_states=True,
        # )
        # hidden_states = encoder_hidden_states.hidden_states[-1]
        hidden_states = encoder_hidden_states[ :, :txt_tokens.input_ids.shape[1], : ]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
        attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
        max_seq_len = max([e.size(0) for e in split_hidden_states])
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0), u.size(1))]) for u in split_hidden_states]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([u, u.new_zeros(max_seq_len - u.size(0))]) for u in attn_mask_list]
        )

        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        return prompt_embeds, encoder_attention_mask
    
    @classmethod
    def to_hf_compatible(
        cls,
        hf_model,
        text_encoder = None,
        meta_info = None,
        vae = None,
        transformers = None,
    ):
        """
        将改写后的模型转换为兼容 Hugging Face 的模型
        """
        if text_encoder is not None:
            hf_model.__class__ = cls
            hf_model.__setup__(text_encoder, transformers,  vae)
            # hf_model.embed_tokens = hf_model.model.embed_tokens
            # del hf_model.text_encoder
            # del hf_model.lm_head
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return hf_model

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        meta_info = None,
    ):
        if not hasattr(self, "token_embedding"):
            self.__init__(meta_info=meta_info)

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._text_encoder.exec_device

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt( # [1, 126, 3584]
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt( # [1, 6, 3584]
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        prompt_embeds = prompt_embeds.to(torch.bfloat16)

        if do_true_cfg:
            negative_prompt_embeds = negative_prompt_embeds.to(torch.float16)

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4 # 16
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        ) # [1, 6032, 64]   (1, 58, 104)
        img_shapes = [[(1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2)]] * batch_size

        # 5. Prepare timesteps
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            mu=mu,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # handle guidance
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
            guidance = None
        elif not self.transformer.config.guidance_embeds and guidance_scale is None:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer( # self._transformer
                        hidden_states=latents,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        encoder_hidden_states_mask=prompt_embeds_mask,
                        encoder_hidden_states=prompt_embeds,
                        img_shapes=img_shapes,
                        txt_seq_lens=txt_seq_lens,
                        attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                if do_true_cfg:
                    with self.transformer.cache_context("uncond"):
                        neg_noise_pred = self.transformer( # self._transformer
                            hidden_states=latents,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=negative_prompt_embeds_mask,
                            encoder_hidden_states=negative_prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=negative_txt_seq_lens,
                            attention_kwargs=self.attention_kwargs,
                            return_dict=False,
                        )
                    comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                    cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                    noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                    noise_pred = comb_pred * (cond_norm / noise_norm)

                # compute the previous noisy sample x_t -> x_t-1
                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if latents.dtype != latents_dtype:
                    if torch.backends.mps.is_available():
                        # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                        latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()


        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(torch.float16)
            latents_mean = (
                torch.tensor(
                    [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508, 0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
                )
                .view(1, 16, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(
                [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743, 3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.916]
                ).view(1, 16, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean

            latents = latents.to(torch.float16).to("cuda:1")
            image = self._vae(latents)[:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        # self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)
