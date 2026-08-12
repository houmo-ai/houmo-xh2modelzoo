# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: I001

# pyright: reportMissingImports=false

from typing import Optional

import torch

from .common import ensure_wan2_2_repo
from .dit_hmonnx import Wan2_2DiTInference
from .dit_wrapper import build_wan_time_embeddings
from .t5_hmonnx import Wan2_2T5EncoderInference
from .vae_hmonnx import Wan2_2VAEDecoderInference, Wan2_2VAEEncoderInference

ensure_wan2_2_repo()
from wan.image2video import WanI2V  # noqa: E402
from wan.text2video import WanT2V  # noqa: E402


class _Wan22TextEncoderProxy:
    def __init__(self, owner, original_text_encoder):
        self._owner = owner
        self._original_text_encoder = original_text_encoder
        self.model = original_text_encoder.model
        self.tokenizer = original_text_encoder.tokenizer
        self.dtype = getattr(original_text_encoder, "dtype", None)
        self.text_len = getattr(original_text_encoder, "text_len", None)

    def __call__(self, texts, device):
        return self._owner._encode_text(texts, device)


class _Wan22RuntimeModelProxy:
    def __init__(self, model):
        self._model = model

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __call__(self, x, *args, **kwargs):
        model_type = getattr(self._model, "model_type", None)
        if model_type == "i2v" and kwargs.get("y") is None:
            if not isinstance(x, list) or not x:
                raise TypeError("Expected non-empty latent list for Wan2.2 i2v-compatible runtime model")
            cond_channels = int(self._model.config.in_dim - x[0].shape[0])
            if cond_channels <= 0:
                raise ValueError(f"Invalid conditional channel count inferred from model config: {cond_channels}")
            kwargs = dict(kwargs)
            kwargs["y"] = [latent.new_zeros((cond_channels, *latent.shape[1:])) for latent in x]
        return self._model(x, *args, **kwargs)


class _Wan22HmonnxDiTProxy:
    def __init__(self, runtime_model, float_model):
        self._runtime_model = runtime_model
        self._float_model = float_model

    def __getattr__(self, name):
        return getattr(self._runtime_model, name)

    @staticmethod
    def _as_single_tensor(value, name: str):
        if isinstance(value, (list, tuple)):
            if len(value) != 1:
                raise ValueError(f"{name} expects one tensor for current Wan2.2 HMONNX runtime, got {len(value)}")
            return value[0]
        return value

    def _prepare_latent(self, x, y):
        latent = self._as_single_tensor(x, "latent")
        if getattr(self._float_model, "model_type", None) != "i2v":
            return latent.half()

        cond = self._as_single_tensor(y, "y")
        return torch.cat([latent, cond], dim=0).half()

    def _prepare_context(self, context):
        context_tensor = self._as_single_tensor(context, "context")
        text_len = getattr(self._float_model, "text_len", None)
        if text_len is not None and context_tensor.size(0) < text_len:
            context_tensor = torch.cat(
                [
                    context_tensor,
                    context_tensor.new_zeros(text_len - context_tensor.size(0), context_tensor.size(1)),
                ],
                dim=0,
            )
        if context_tensor.dim() == 2:
            context_tensor = context_tensor.unsqueeze(0)
        return context_tensor.half()

    def __call__(self, x, *args, **kwargs):
        if "t" not in kwargs:
            raise KeyError("Expected timestep 't' for Wan2.2 DiT runtime call")
        kwargs = dict(kwargs)
        t = kwargs.pop("t")
        context = kwargs["context"]
        seq_len = kwargs["seq_len"]
        y = kwargs.get("y")
        if getattr(self._float_model, "model_type", None) == "i2v" and y is None:
            if not isinstance(x, list) or not x:
                raise TypeError("Expected non-empty latent list for Wan2.2 i2v-compatible runtime model")
            cond_channels = int(self._float_model.config.in_dim - x[0].shape[0])
            if cond_channels <= 0:
                raise ValueError(f"Invalid conditional channel count inferred from model config: {cond_channels}")
            kwargs["y"] = [latent.new_zeros((cond_channels, *latent.shape[1:])) for latent in x]
            y = kwargs["y"]
        e, e0 = build_wan_time_embeddings(self._float_model, t, seq_len, torch.float16)
        latent_tensor = self._prepare_latent(x, y)
        context_tensor = self._prepare_context(context)

        runtime_out = self._runtime_model(
            latent_tensor,
            context=context_tensor,
            e=e,
            e0=e0,
        )
        if not isinstance(runtime_out, (list, tuple)):
            runtime_out = (runtime_out,)
        return runtime_out


class _Wan22ModuleSwitchMixin:
    hmonnx_t5: Optional[Wan2_2T5EncoderInference] = None
    hmonnx_vae_encoder: Optional[Wan2_2VAEEncoderInference] = None
    hmonnx_vae_decoder: Optional[Wan2_2VAEDecoderInference] = None
    hmonnx_low_noise_model: Optional[Wan2_2DiTInference] = None
    hmonnx_high_noise_model: Optional[Wan2_2DiTInference] = None
    _float_vae_encode = None
    _float_vae_decode = None

    def set_hmonnx_components(
        self,
        t5: Optional[Wan2_2T5EncoderInference] = None,
        vae_encoder: Optional[Wan2_2VAEEncoderInference] = None,
        vae_decoder: Optional[Wan2_2VAEDecoderInference] = None,
        low_noise_model: Optional[Wan2_2DiTInference] = None,
        high_noise_model: Optional[Wan2_2DiTInference] = None,
    ):
        if t5 is not None:
            self.hmonnx_t5 = t5
        if vae_encoder is not None:
            self.hmonnx_vae_encoder = vae_encoder
        if vae_decoder is not None:
            self.hmonnx_vae_decoder = vae_decoder
        if low_noise_model is not None:
            self.hmonnx_low_noise_model = low_noise_model
        if high_noise_model is not None:
            self.hmonnx_high_noise_model = high_noise_model
        return self

    def _encode_text(self, texts, device):
        if self.hmonnx_t5 is not None:
            return self.hmonnx_t5(texts, device)
        return self.text_encoder(texts, device)

    def _encode_vae_video(self, video: torch.Tensor):
        if self.hmonnx_vae_encoder is not None:
            return self.hmonnx_vae_encoder(video)
        if self._float_vae_encode is None:
            return self.vae.encode([video])[0]
        return self._float_vae_encode([video])[0]

    def _decode_vae_latent(self, latent: torch.Tensor):
        if self.hmonnx_vae_decoder is not None:
            return self.hmonnx_vae_decoder(latent)
        if self._float_vae_decode is None:
            return self.vae.decode([latent])[0]
        return self._float_vae_decode([latent])[0]

    def _select_runtime_model(self, t, boundary, offload_model, float_prepare_fn=None):
        if float_prepare_fn is not None:
            model = float_prepare_fn(t, boundary, offload_model)
        else:
            model = self._prepare_model_for_timestep(t, boundary, offload_model)
        if t.item() >= boundary and self.hmonnx_high_noise_model is not None:
            return _Wan22HmonnxDiTProxy(self.hmonnx_high_noise_model, model)
        if t.item() < boundary and self.hmonnx_low_noise_model is not None:
            return _Wan22HmonnxDiTProxy(self.hmonnx_low_noise_model, model)
        return _Wan22RuntimeModelProxy(model)


class WanT2VHMONNXPipeline(_Wan22ModuleSwitchMixin, WanT2V):
    def generate(self, *args, **kwargs):
        n_prompt = kwargs.get("n_prompt", "")
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        kwargs["n_prompt"] = n_prompt

        original_text_encoder = self.text_encoder
        original_prepare = self._prepare_model_for_timestep
        original_decode = self.vae.decode
        self._float_vae_decode = original_decode
        if self.hmonnx_t5 is not None:
            self.text_encoder = _Wan22TextEncoderProxy(self, original_text_encoder)
        if self.hmonnx_low_noise_model is not None or self.hmonnx_high_noise_model is not None:
            self._prepare_model_for_timestep = lambda t, boundary, offload_model: self._select_runtime_model(
                t, boundary, offload_model, original_prepare
            )
        if self.hmonnx_vae_decoder is not None:
            self.vae.decode = lambda zs: [self._decode_vae_latent(zs[0])]
        try:
            return super().generate(*args, **kwargs)
        finally:
            self.text_encoder = original_text_encoder
            self._prepare_model_for_timestep = original_prepare
            self.vae.decode = original_decode
            self._float_vae_decode = None


class WanI2VHMONNXPipeline(_Wan22ModuleSwitchMixin, WanI2V):
    def generate(self, *args, **kwargs):
        n_prompt = kwargs.get("n_prompt", "")
        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        kwargs["n_prompt"] = n_prompt

        original_text_encoder = self.text_encoder
        original_prepare = self._prepare_model_for_timestep
        original_encode = self.vae.encode
        original_decode = self.vae.decode
        self._float_vae_encode = original_encode
        self._float_vae_decode = original_decode
        if self.hmonnx_t5 is not None:
            self.text_encoder = _Wan22TextEncoderProxy(self, original_text_encoder)
        if self.hmonnx_low_noise_model is not None or self.hmonnx_high_noise_model is not None:
            self._prepare_model_for_timestep = lambda t, boundary, offload_model: self._select_runtime_model(
                t, boundary, offload_model, original_prepare
            )
        if self.hmonnx_vae_encoder is not None:
            self.vae.encode = lambda videos: [self._encode_vae_video(videos[0])]
        if self.hmonnx_vae_decoder is not None:
            self.vae.decode = lambda zs: [self._decode_vae_latent(zs[0])]
        try:
            return super().generate(*args, **kwargs)
        finally:
            self.text_encoder = original_text_encoder
            self._prepare_model_for_timestep = original_prepare
            self.vae.encode = original_encode
            self.vae.decode = original_decode
            self._float_vae_encode = None
            self._float_vae_decode = None
