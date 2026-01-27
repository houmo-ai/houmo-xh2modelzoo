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
from diffusers.pipelines.z_image.pipeline_z_image import ZImagePipeline
from diffusers.pipelines.qwenimage.pipeline_qwenimage import calculate_shift, retrieve_timesteps
import json
from xhquant.api import ConfigDict
from modelscope import AutoProcessor, AutoTokenizer, Qwen2VLForConditionalGeneration
import torch.nn as nn
from pathlib import Path
from xhquant.core import CacheTensor


class cus_ZImagePipeline(ZImagePipeline):
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
        self.token_embedding = nn.Embedding(
            token_embedding_state_dict["weight"].shape[0],
            token_embedding_state_dict["weight"].shape[1],
        )
        self.token_embedding.weight.data = token_embedding_state_dict["weight"]
        # self.token_embedding.load_state_dict(token_embedding_state_dict)
        self.wraped_transformer = None
    
    def text_encoder_infer(self, input_ids, attention_mask):
        self._past_seq_length = 0
        self.token_embedding = self.token_embedding.to(input_ids.device)

        inputs_embeds = self.token_embedding(input_ids)

        kv_cache_shape = self.meta_info["kv_cache"]["shape"]
        num_decoder_layers = self.meta_info["kv_cache"]["num_decoder_layers"]
        past_key_caches = []
        past_value_caches = []
        for i in range(num_decoder_layers):
            past_k_cache = CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            past_v_cache = CacheTensor(torch.zeros(kv_cache_shape, dtype=torch.float16))
            past_key_caches.append(past_k_cache)
            past_value_caches.append(past_v_cache)  


        past_seq_length = torch.tensor([self._past_seq_length], dtype=torch.int32).to(inputs_embeds.device)

        # TODO: 需要根据attention mask计算seq_length
        if input_ids is not None:
            seq_length = input_ids.shape[-1]
        else:
            seq_length = inputs_embeds.shape[1] if inputs_embeds is not None else 0
        current_input_length = torch.tensor([seq_length], dtype=torch.int32).to(inputs_embeds.device)

        if self._text_encoder is None:
            raise ValueError("llm_model is not initialized")

        input_sequence_length = self._text_encoder.get_input_sequence_length()


        input_data = dict(
            input_ids=input_ids,
            past_seq_length=past_seq_length,
        )


        pad_input_seq_length = (
            (seq_length + input_sequence_length - 1) // input_sequence_length
        ) * input_sequence_length

        (
            inputs_embeds,
            past_seq_length,
            current_input_length,
            past_key_caches,
            past_value_caches,
        ) = self._text_encoder.prepare_inputs(input_data, pad_input_seq_length)

        pad_seq_lenght = inputs_embeds.shape[1]
        assert pad_seq_lenght % input_sequence_length == 0, "pad_seq_lenght must be divisible by input_sequence_length"
        steps = pad_seq_lenght // input_sequence_length
        for i in range(steps):
            start = i * input_sequence_length
            end = (i + 1) * input_sequence_length
            sub_inputs_embeds = inputs_embeds[:, start:end, :]
            sub_past_seq_length = past_seq_length + start
            sub_current_input_length = torch.tensor(
                [min(end, seq_length) - start], dtype=current_input_length.dtype
            ).to(current_input_length.device)

            output_hidden = self._text_encoder(
                sub_inputs_embeds, # [1, 256, 2560
                sub_past_seq_length, # 0
                sub_current_input_length, # 28
                past_key_caches,
                past_value_caches,
            )

        output_hidden = output_hidden[0,:seq_length, :]
        return output_hidden

    def transformer_infer(self, latent_model_input_list, timestep_model_input, prompt_embeds_model_input):
        # latent = torch.randn(1, 16, 128, 128).cuda()
        latent = latent_model_input_list[0].cuda()
        timestep_model_input = timestep_model_input.cuda() * 1000.0
        cap_feats = prompt_embeds_model_input[0].cuda() # [15, 2560]
        patch_size = 2
        f_patch_size = 1

        # latent = torch.load("examples/llm/zimage/dit/input/latent.pt").cuda() # [16, 1, 128, 128]
        # cap_feats = torch.load("examples/llm/zimage/dit/input/prompt.pt").cuda() # [15, 2560]

        device = "cuda"
        with torch.no_grad():
            adaln_input = self.transformer.t_embedder(timestep_model_input).to(torch.float16) 

            (
                x, # [4096, 64]
                cap_feats, # [128, 2560]   # [L, 2560]
                x_size,
                x_pos_ids,
                cap_pos_ids,
                x_pad_mask,
                cap_pad_mask,
            ) = self.transformer.patchify_and_embed([latent], [cap_feats], patch_size, f_patch_size) # [16, 1, 128, 128]   [101, 2560]   2 1 
            x_pos_offsets = x_noise_mask = cap_noise_mask = siglip_noise_mask = None     

            valid_len = int(cap_feats[0].shape[0])
            required_len = int(256) # 256

            # ==================
            x_freqs = self.transformer.rope_embedder( torch.cat(x_pos_ids, dim=0) ).unsqueeze(0)
            freqs_cis_expanded = x_freqs.unsqueeze(2)
            f_real = freqs_cis_expanded.real  # 频率的实部，形状匹配x_real
            f_imag = freqs_cis_expanded.imag  # 频率的虚部，形状匹配x_imag

            # Attention mask =========
            attn_mask = torch.zeros((1, 4096), dtype=torch.bool, device=device) # [1,4096]
            x_mask = attn_mask

            
            # ------------------------------------------------------------------------------
            pad_len = required_len - valid_len # 256 -32
            cap_mask = torch.zeros((1, valid_len), device=device)
            cap_feats = torch.concat( [ cap_feats[0], torch.zeros(pad_len, cap_feats[0].shape[1]).to(device)], dim=0)# .unsqueeze(0)  # [256, 2560]
            cap_mask = torch.concat([ cap_mask, torch.ones(pad_len).to(device).unsqueeze(0)*-65504], dim=1)
            cap_pos_ids = torch.concat(
                        [ torch.range(0, required_len-1).to(device).unsqueeze(-1), torch.zeros((required_len,2)).to(cap_feats.device) ], dim=1
                    ).to(torch.long) # [256, 3]
                    
            c_freqs_cis = self.transformer.rope_embedder( cap_pos_ids ).unsqueeze(0) # [256, 64]
            c_freqs_cis_expanded = c_freqs_cis.unsqueeze(2)
            c_f_real = c_freqs_cis_expanded.real  # 频率的实部，形状匹配x_real
            c_f_imag = c_freqs_cis_expanded.imag  # 频率的虚部，形状匹配x_imag

            cap_pad_mask = torch.concat( [cap_pad_mask[0], torch.zeros(pad_len, device=device)] ).half().unsqueeze(-1)
            # ------------------------------------------------------------------------------
            n_cap_pad_mask = 1 - cap_pad_mask

            # if self.wraped_transformer is None:
            #     from xh_model_zoo.xh_llm.models.zimage._dit_model import register_wrap_cls as llm_register_wrap_cls
            #     from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
            #     from xhquant.api import Config

            #     llm_register_wrap_cls(self.transformer)
            #     wrap_cfg = Config(
            #         dict(
            #             batch_size=1,
            #             token_len=256
            #         )
            #     )

            #     wrap_cfg["f_real"] = f_real.half().repeat([1,1,1,2])
            #     wrap_cfg["f_imag"] = f_imag.half().repeat([1,1,1,2])
            #     wrap_cfg["c_f_real"] = c_f_real.half().repeat([1,1,1,2])
            #     wrap_cfg["c_f_imag"] = c_f_imag.half().repeat([1,1,1,2])

            #     wraped_transformer = wrap_llm_model(self.transformer, wrap_cfg)
            #     wraped_transformer.cuda()
            #     wraped_transformer.to(torch.float16)
            #     self.wraped_transformer = wraped_transformer

            if True:
                # output = self.wraped_transformer(
                output = self._transformer(
                    x[0], x_mask.half(),  adaln_input, # f_real.half(), f_imag.half(),#
                    cap_feats.half(), cap_mask.half(),  cap_pad_mask, n_cap_pad_mask, # c_f_real.half(), c_f_imag.half(),
                )
                # Unpatchify
                output = output[:, :4096+valid_len]
                x = self.transformer.unpatchify(list((output).unbind(dim=0)), x_size, patch_size, f_patch_size, x_pos_offsets)
        return x

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

    def _encode_prompt(
        self,
        prompt: Union[str, List[str]],
        device: Optional[torch.device] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        max_sequence_length: int = 512,
    ) -> List[torch.FloatTensor]:
        # device = device or self._execution_device

        if prompt_embeds is not None:
            return prompt_embeds

        if isinstance(prompt, str):
            prompt = [prompt]

        for i, prompt_item in enumerate(prompt):
            messages = [
                {"role": "user", "content": prompt_item},
            ]
            prompt_item = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            prompt[i] = prompt_item

        text_inputs = self.tokenizer(
            prompt,
            # padding="max_length",
            # max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids.to(device)
        prompt_masks = text_inputs.attention_mask.to(device).bool()

        prompt_embeds = self.text_encoder_infer(
            input_ids=text_input_ids, # [1, 512] 
            attention_mask=prompt_masks, # [1, 512] 
            # output_hidden_states=True, # [1, 512, 2560]
        ) #.hidden_states[-2]

        embeddings_list = []

        embeddings_list.append(prompt_embeds)
        return embeddings_list
    
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
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 5.0,
        cfg_normalization: bool = False,
        cfg_truncation: float = 1.0,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        negative_prompt_embeds: Optional[List[torch.FloatTensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        meta_info = None,
    ):
        if not hasattr(self, "token_embedding"):
            self.__init__(meta_info=meta_info)

        height = height or 1024
        width = width or 1024

        vae_scale = 8 * 2
        if height % vae_scale != 0:
            raise ValueError(
                f"Height must be divisible by {vae_scale} (got {height}). "
                f"Please adjust the height to a multiple of {vae_scale}."
            )
        if width % vae_scale != 0:
            raise ValueError(
                f"Width must be divisible by {vae_scale} (got {width}). "
                f"Please adjust the width to a multiple of {vae_scale}."
            )

        device = "cuda"

        self._guidance_scale = guidance_scale
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False
        self._cfg_normalization = cfg_normalization
        self._cfg_truncation = cfg_truncation
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = len(prompt_embeds)

        # If prompt_embeds is provided and prompt is None, skip encoding
        if prompt_embeds is not None and prompt is None:
            if self.do_classifier_free_guidance and negative_prompt_embeds is None:
                raise ValueError(
                    "When `prompt_embeds` is provided without `prompt`, "
                    "`negative_prompt_embeds` must also be provided for classifier-free guidance."
                )
        else:
            (
                prompt_embeds,
                negative_prompt_embeds,
            ) = self.encode_prompt(
                prompt=prompt,
                negative_prompt=negative_prompt,
                do_classifier_free_guidance=self.do_classifier_free_guidance,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                device=device,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.in_channels

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            torch.float32,
            device,
            generator,
            latents, # [1, 16, 128, 128]
        )

        # Repeat prompt_embeds for num_images_per_prompt
        if num_images_per_prompt > 1:
            prompt_embeds = [pe for pe in prompt_embeds for _ in range(num_images_per_prompt)]
            if self.do_classifier_free_guidance and negative_prompt_embeds:
                negative_prompt_embeds = [npe for npe in negative_prompt_embeds for _ in range(num_images_per_prompt)]

        actual_batch_size = batch_size * num_images_per_prompt
        image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)

        # 5. Prepare timesteps
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        self.scheduler.sigma_min = 0.0
        scheduler_kwargs = {"mu": mu}
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler,
            num_inference_steps,
            device,
            sigmas=sigmas,
            **scheduler_kwargs,
        )
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        # 6. Denoising loop
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                timestep = t.expand(latents.shape[0])
                timestep = (1000 - timestep) / 1000
                # Normalized time for time-aware config (0 at start, 1 at end)
                t_norm = timestep[0].item()

                # Handle cfg truncation
                current_guidance_scale = self.guidance_scale # 0
                if (
                    self.do_classifier_free_guidance
                    and self._cfg_truncation is not None
                    and float(self._cfg_truncation) <= 1
                ):
                    if t_norm > self._cfg_truncation:
                        current_guidance_scale = 0.0

                # Run CFG only if configured AND scale is non-zero
                apply_cfg = self.do_classifier_free_guidance and current_guidance_scale > 0

                if apply_cfg:
                    latents_typed = latents.to(self.transformer.dtype)
                    latent_model_input = latents_typed.repeat(2, 1, 1, 1)
                    prompt_embeds_model_input = prompt_embeds + negative_prompt_embeds
                    timestep_model_input = timestep.repeat(2)
                else:
                    latent_model_input = latents.to(self.transformer.dtype)
                    prompt_embeds_model_input = prompt_embeds
                    timestep_model_input = timestep

                latent_model_input = latent_model_input.unsqueeze(2)
                latent_model_input_list = list(latent_model_input.unbind(dim=0))

                # model_out_list = self.transformer(
                #     latent_model_input_list, timestep_model_input, prompt_embeds_model_input, return_dict=False # [0]
                # )[0]
                model_out_list = self.transformer_infer(
                    latent_model_input_list, timestep_model_input, prompt_embeds_model_input #, return_dict=False
                )

                if apply_cfg:
                    # Perform CFG
                    pos_out = model_out_list[:actual_batch_size]
                    neg_out = model_out_list[actual_batch_size:]

                    noise_pred = []
                    for j in range(actual_batch_size):
                        pos = pos_out[j].float()
                        neg = neg_out[j].float()

                        pred = pos + current_guidance_scale * (pos - neg)

                        # Renormalization
                        if self._cfg_normalization and float(self._cfg_normalization) > 0.0:
                            ori_pos_norm = torch.linalg.vector_norm(pos)
                            new_pos_norm = torch.linalg.vector_norm(pred)
                            max_new_norm = ori_pos_norm * float(self._cfg_normalization)
                            if new_pos_norm > max_new_norm:
                                pred = pred * (max_new_norm / new_pos_norm)

                        noise_pred.append(pred)

                    noise_pred = torch.stack(noise_pred, dim=0)
                else:
                    noise_pred = torch.stack([t.float() for t in model_out_list], dim=0)

                noise_pred = noise_pred.squeeze(2)
                noise_pred = -noise_pred

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred.to(torch.float32), t, latents, return_dict=False)[0]
                assert latents.dtype == torch.float32

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if output_type == "latent":
            image = latents

        else:
            latents = latents.to(torch.float16)
            latents = (latents / 0.3611) + 0.1159

            # image = self.vae.decode(latents, return_dict=False)[0]
            image = self._vae(latents)

            image = self.image_processor.postprocess(image, output_type=output_type)

        return image
