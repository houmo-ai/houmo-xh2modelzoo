import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch


def _ensure_installed_flux2_diffusers() -> None:
    vendored_diffusers = Path(__file__).resolve().parents[4] / "data" / "difs" / "diffusers-main" / "src"
    vendored_str = str(vendored_diffusers)
    if vendored_str in sys.path:
        sys.path.remove(vendored_str)
    for module_name in list(sys.modules):
        if module_name == "diffusers" or module_name.startswith("diffusers."):
            del sys.modules[module_name]


_ensure_installed_flux2_diffusers()

from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import XLA_AVAILABLE, compute_empirical_mu, retrieve_timesteps
from diffusers.pipelines.flux2.pipeline_output import Flux2PipelineOutput

from .components_hmonnx import Flux2KleinTransformerInference, Flux2KleinVAEEncoderInference, Flux2KleinVAEInference
from .text_encoder_hmonnx import Flux2KleinTextEncoderInference


def _normalize_components(components: Union[str, Sequence[str]]) -> set[str]:
    if isinstance(components, str):
        components = [item.strip() for item in components.split(",") if item.strip()]
    aliases = {"te": "text_encoder", "text": "text_encoder", "trans": "transformer", "vae_decoder": "vae", "vae_encode": "vae_encoder"}
    normalized = {aliases.get(component.strip().lower(), component.strip().lower()) for component in components}
    supported = {"text_encoder", "transformer", "vae", "vae_encoder"}
    unknown = normalized - supported
    if unknown:
        raise ValueError(f"不支持的 HMONNX 组件: {sorted(unknown)}，支持: {sorted(supported)}")
    return normalized


def resolve_text_meta_path(meta_path: Union[str, Path], root_meta_path: Optional[Union[str, Path]] = None) -> Path:
    meta_file = Path(meta_path)
    meta_info = json.load(open(meta_file, "r"))
    if "text_encoder_meta" in meta_info:
        return meta_file.parent / meta_info["text_encoder_meta"]
    if root_meta_path is not None:
        root_meta_file = Path(root_meta_path)
        root_info = json.load(open(root_meta_file, "r"))
        if "text_encoder_meta" in root_info:
            return root_meta_file.parent / root_info["text_encoder_meta"]
    return meta_file


class Flux2KleinHMONNXPipeline(Flux2KleinPipeline):
    def __init__(
        self,
        scheduler,
        vae,
        text_encoder,
        tokenizer,
        transformer,
        is_distilled: bool = False,
    ):
        super().__init__(
            scheduler=scheduler,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            is_distilled=is_distilled,
        )
        self.hmonnx_text_encoder = None
        self.hmonnx_transformer = None
        self.hmonnx_vae = None
        self.hmonnx_vae_encoder = None
        self._bind_hmonnx_transformer_context()

    def set_hmonnx_components(
        self,
        text_encoder: Optional[Flux2KleinTextEncoderInference] = None,
        transformer: Optional[Flux2KleinTransformerInference] = None,
        vae: Optional[Flux2KleinVAEInference] = None,
        vae_encoder: Optional[Flux2KleinVAEEncoderInference] = None,
    ):
        if text_encoder is not None:
            self.hmonnx_text_encoder = text_encoder
        if transformer is not None:
            self.hmonnx_transformer = transformer
        if vae is not None:
            self.hmonnx_vae = vae
        if vae_encoder is not None:
            self.hmonnx_vae_encoder = vae_encoder
        self._bind_hmonnx_transformer_context()
        return self

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if self.hmonnx_vae_encoder is not None:
            return self.hmonnx_vae_encoder(image)
        return super()._encode_vae_image(image=image, generator=generator)

    def _bind_hmonnx_transformer_context(self):
        if self.hmonnx_transformer is None:
            return
        self.hmonnx_transformer.config = self.transformer.config
        if hasattr(self.transformer, "pos_embed"):
            self.hmonnx_transformer.pos_embed = self.transformer.pos_embed.to(self.hmonnx_transformer.device)

    @classmethod
    def from_pretrained_with_hmonnx(
        cls,
        pretrained_model_name_or_path: Union[str, Path],
        meta_path: Union[str, Path],
        root_meta_path: Optional[Union[str, Path]] = None,
        components: Union[str, Sequence[str]] = ("text_encoder", "transformer", "vae"),
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float16,
        **kwargs,
    ):
        pipe = cls.from_pretrained(pretrained_model_name_or_path, **kwargs)
        attach_flux2_klein_hmonnx_components(
            pipe,
            components=components,
            meta_path=meta_path,
            root_meta_path=root_meta_path,
            device=device,
            dtype=dtype,
        )
        return pipe

    @torch.no_grad()
    def __call__(
        self,
        image=None,
        prompt=None,
        height=None,
        width=None,
        num_inference_steps: int = 50,
        sigmas=None,
        guidance_scale: float = 4.0,
        num_images_per_prompt: int = 1,
        generator=None,
        latents=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs=None,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        text_encoder_out_layers: tuple[int] = (9, 18, 27),
        use_hmonnx_text_encoder: bool = False,
        use_hmonnx_transformer: bool = False,
        use_hmonnx_vae: bool = False,
    ):
        # No HMONNX: keep the official floating-point pipeline unchanged.
        if not (use_hmonnx_text_encoder or use_hmonnx_transformer or use_hmonnx_vae):
            return super().__call__(
                image=image,
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                sigmas=sigmas,
                guidance_scale=guidance_scale,
                num_images_per_prompt=num_images_per_prompt,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        self.check_inputs(
            prompt=prompt,
            height=height,
            width=width,
            prompt_embeds=prompt_embeds,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            guidance_scale=guidance_scale,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        if use_hmonnx_text_encoder and self.hmonnx_text_encoder is not None:
            prompt_embeds, text_ids = self.hmonnx_text_encoder.encode_prompt(
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )
        else:
            prompt_embeds, text_ids = self.encode_prompt(
                prompt=prompt,
                prompt_embeds=prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        if self.do_classifier_free_guidance:
            negative_prompt = ""
            if prompt is not None and isinstance(prompt, list):
                negative_prompt = [negative_prompt] * len(prompt)
            negative_prompt_embeds, negative_text_ids = self.encode_prompt(
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
                text_encoder_out_layers=text_encoder_out_layers,
            )

        if image is not None and not isinstance(image, list):
            image = [image]

        condition_images = None
        if image is not None:
            condition_images = []
            for img in image:
                self.image_processor.check_image_input(img)
                image_width, image_height = img.size
                if image_width * image_height > 1024 * 1024:
                    img = self.image_processor._resize_to_target_area(img, 1024 * 1024)
                    image_width, image_height = img.size
                multiple_of = self.vae_scale_factor * 2
                image_width = (image_width // multiple_of) * multiple_of
                image_height = (image_height // multiple_of) * multiple_of
                img = self.image_processor.preprocess(img, height=image_height, width=image_width, resize_mode="crop")
                condition_images.append(img)
                height = height or image_height
                width = width or image_width

        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor

        transformer_model = self.hmonnx_transformer if use_hmonnx_transformer and self.hmonnx_transformer is not None else self.transformer
        num_channels_latents = transformer_model.config.in_channels // 4
        latents, latent_ids = self.prepare_latents(
            batch_size=batch_size * num_images_per_prompt,
            num_latents_channels=num_channels_latents,
            height=height,
            width=width,
            dtype=prompt_embeds.dtype,
            device=device,
            generator=generator,
            latents=latents,
        )

        image_latents = None
        image_latent_ids = None
        if condition_images is not None:
            image_latents, image_latent_ids = self.prepare_image_latents(
                images=condition_images,
                batch_size=batch_size * num_images_per_prompt,
                generator=generator,
                device=device,
                dtype=self.vae.dtype,
            )

        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
        if hasattr(self.scheduler.config, "use_flow_sigmas") and self.scheduler.config.use_flow_sigmas:
            sigmas = None
        image_seq_len = latents.shape[1]
        mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
        timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, num_inference_steps, device, sigmas=sigmas, mu=mu)
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        self._num_timesteps = len(timesteps)

        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                latent_model_input = latents.to(transformer_model.dtype)
                latent_image_ids = latent_ids

                if image_latents is not None:
                    latent_model_input = torch.cat([latents, image_latents], dim=1).to(transformer_model.dtype)
                    latent_image_ids = torch.cat([latent_ids, image_latent_ids], dim=1)

                transformer_timestep = timestep / 1000
                if use_hmonnx_transformer and self.hmonnx_transformer is not None:
                    transformer_timestep = torch.tensor(
                        [i],
                        device=device,
                        dtype=torch.int32,
                    )

                with transformer_model.cache_context("cond"):
                    noise_pred = transformer_model(
                        hidden_states=latent_model_input,
                        timestep=transformer_timestep,
                        guidance=None,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_image_ids,
                        joint_attention_kwargs=self.attention_kwargs,
                        return_dict=False,
                    )[0]

                noise_pred = noise_pred.unsqueeze(0) 
                noise_pred = noise_pred[:, : latents.size(1) :]

                if self.do_classifier_free_guidance:
                    with transformer_model.cache_context("uncond"):
                        neg_noise_pred = transformer_model(
                            hidden_states=latent_model_input,
                            timestep=transformer_timestep,
                            guidance=None,
                            encoder_hidden_states=negative_prompt_embeds,
                            txt_ids=negative_text_ids,
                            img_ids=latent_image_ids,
                            joint_attention_kwargs=self._attention_kwargs,
                            return_dict=False,
                        )[0]
                    neg_noise_pred = neg_noise_pred[:, : latents.size(1) :]
                    noise_pred = neg_noise_pred + guidance_scale * (noise_pred - neg_noise_pred)

                latents_dtype = latents.dtype
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                if latents.dtype != latents_dtype and torch.backends.mps.is_available():
                    latents = latents.to(latents_dtype)

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    import torch_xla.core.xla_model as xm  # pyright: ignore[reportMissingImports]

                    xm.mark_step()

        self._current_timestep = None
        latent_height = 2 * (int(height) // (self.vae_scale_factor * 2))
        latent_width = 2 * (int(width) // (self.vae_scale_factor * 2))
        latents = self._unpack_latents_with_ids(latents, latent_ids, latent_height // 2, latent_width // 2)

        if output_type == "latent":
            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(latents.device, latents.dtype)
            image = self._unpatchify_latents(latents * latents_bn_std + latents_bn_mean)
        elif use_hmonnx_vae and self.hmonnx_vae is not None:
            image = self.hmonnx_vae(latents)
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
            latents_bn_std = torch.sqrt(self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).to(latents.device, latents.dtype)
            latents = latents * latents_bn_std + latents_bn_mean
            latents = self._unpatchify_latents(latents)
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        self.maybe_free_model_hooks()
        if not return_dict:
            return (image,)
        return Flux2PipelineOutput(images=image)


def attach_flux2_klein_hmonnx_components(
    pipe: Flux2KleinHMONNXPipeline,
    components: Union[str, Sequence[str]],
    meta_path: Union[str, Path],
    root_meta_path: Optional[Union[str, Path]] = None,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float16,
):
    if not isinstance(pipe, Flux2KleinHMONNXPipeline):
        raise TypeError("pipe 必须是 Flux2KleinHMONNXPipeline；请使用 Flux2KleinHMONNXPipeline.from_pretrained(...) 创建。")

    components = _normalize_components(components)
    meta_file = Path(meta_path)
    root_meta_file = Path(root_meta_path) if root_meta_path is not None else meta_file
    loaded = {}

    text_encoder = None
    transformer = None
    vae = None
    vae_encoder = None
    if "text_encoder" in components:
        text_meta_path = resolve_text_meta_path(meta_file, root_meta_file)
        text_encoder = Flux2KleinTextEncoderInference.from_meta(text_meta_path, tokenizer=pipe.tokenizer, device=device, dtype=dtype)
        loaded["text_encoder"] = text_encoder
    if "transformer" in components:
        transformer = Flux2KleinTransformerInference.from_root_meta(root_meta_file, device=device, dtype=dtype)
        loaded["transformer"] = transformer
    if "vae" in components:
        vae = Flux2KleinVAEInference.from_root_meta(root_meta_file, device=device, dtype=dtype)
        loaded["vae"] = vae
    if "vae_encoder" in components:
        vae_encoder = Flux2KleinVAEEncoderInference.from_root_meta(root_meta_file, device=device, dtype=dtype)
        loaded["vae_encoder"] = vae_encoder

    pipe.set_hmonnx_components(text_encoder=text_encoder, transformer=transformer, vae=vae, vae_encoder=vae_encoder)
    return pipe, loaded


def build_hmonnx_pipeline_cls(base_pipeline_cls=None):
    del base_pipeline_cls
    return Flux2KleinHMONNXPipeline


build_hmonnx_vae_pipeline_cls = build_hmonnx_pipeline_cls


def ensure_hmonnx_pipeline(pipe):
    if not isinstance(pipe, Flux2KleinHMONNXPipeline):
        raise TypeError("请直接使用 Flux2KleinHMONNXPipeline.from_pretrained(...) 创建 pipeline。")
    return pipe


def attach_hmonnx_vae(pipe: Flux2KleinHMONNXPipeline, vae: Flux2KleinVAEInference):
    ensure_hmonnx_pipeline(pipe)
    pipe.set_hmonnx_components(vae=vae)
    return pipe
