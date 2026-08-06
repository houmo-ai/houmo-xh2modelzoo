# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: I001

# pyright: reportMissingImports=false

import logging
import os
from typing import Optional

import torch

from .common import ensure_wan2_2_repo
from .dit_hmonnx import Wan2_2DiTInference
from .dit_wrapper import build_wan_time_embeddings
from .t5_hmonnx import Wan2_2T5EncoderInference
from .vae_hmonnx import Wan2_2VAEDecoderInference, Wan2_2VAEEncoderInference

ensure_wan2_2_repo()
from wan.distributed.util import get_world_size  # noqa: E402
from wan.configs import WAN_CONFIGS  # noqa: E402
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
    _debug_call_count = 0

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
        latent = x[0] if isinstance(x, list) else x
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
        if os.environ.get("WAN22_DEBUG_DIT_PROXY_COMPARE", ""):
            max_calls = int(os.environ.get("WAN22_DEBUG_DIT_PROXY_COMPARE_MAX", "8"))
            if _Wan22HmonnxDiTProxy._debug_call_count < max_calls:
                with torch.no_grad():
                    float_out = self._float_model(
                        x,
                        t=t,
                        context=context,
                        seq_len=seq_len,
                        y=y,
                    )[0]
                runtime_tensor = runtime_out[0] if isinstance(runtime_out, (list, tuple)) else runtime_out
                diff = (float_out.detach().float() - runtime_tensor.detach().float()).abs()
                print(
                    "WAN22_DEBUG_DIT_PROXY_COMPARE "
                    f"call={_Wan22HmonnxDiTProxy._debug_call_count} "
                    f"t={float(t.flatten()[0].item()):.1f} "
                    f"latent_shape={tuple(self._as_single_tensor(x, 'latent').shape)} "
                    f"y_shape={tuple(self._as_single_tensor(y, 'y').shape) if y is not None else None} "
                    f"context_len={self._as_single_tensor(context, 'context').size(0)} "
                    f"seq_len={seq_len} "
                    f"max_abs={diff.max().item():.6f} "
                    f"mean_abs={diff.mean().item():.6f}"
                )
                _Wan22HmonnxDiTProxy._debug_call_count += 1
        return runtime_out


class _Wan22ModuleSwitchMixin:
    hmonnx_t5: Optional[Wan2_2T5EncoderInference] = None
    hmonnx_vae_encoder: Optional[Wan2_2VAEEncoderInference] = None
    hmonnx_vae_decoder: Optional[Wan2_2VAEDecoderInference] = None
    hmonnx_low_noise_model: Optional[Wan2_2DiTInference] = None
    hmonnx_high_noise_model: Optional[Wan2_2DiTInference] = None
    _float_vae_encode = None
    _float_vae_decode = None

    def _init_with_resolved_float_loader(self, cfg, checkpoint_dir, *args, **kwargs):
        device_id = kwargs.get("device_id", args[0] if len(args) > 0 else 0)
        rank = kwargs.get("rank", args[1] if len(args) > 1 else 0)
        t5_fsdp = kwargs.get("t5_fsdp", False)
        dit_fsdp = kwargs.get("dit_fsdp", False)
        use_sp = kwargs.get("use_sp", False)
        t5_cpu = kwargs.get("t5_cpu", False)
        init_on_cpu = kwargs.get("init_on_cpu", True)

        self.device = torch.device(f"cuda:{device_id}")
        self.config = cfg
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = cfg.num_train_timesteps
        self.boundary = cfg.boundary
        self.param_dtype = cfg.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        self.vae_stride = cfg.vae_stride
        self.patch_size = cfg.patch_size
        self.sp_size = get_world_size() if use_sp else 1
        self.sample_neg_prompt = cfg.sample_neg_prompt

        logging.info("Creating Wan2.2 HMONNX pipeline from %s", checkpoint_dir)

        from .wan2_2_converter import Wan2_2ConvertConfig, Wan2_2Converter

        task = self._infer_task_name(cfg)
        converter = Wan2_2Converter(
            checkpoint_dir,
            Wan2_2ConvertConfig(task=task, use_resolved_float_loader=True),
        )
        converter._reload_pipeline_components_from_resolved_paths(self, cfg, device_id=device_id)

    @staticmethod
    def _infer_task_name(cfg):
        for task, task_cfg in WAN_CONFIGS.items():
            if task_cfg is cfg:
                return task
        for task, task_cfg in WAN_CONFIGS.items():
            if getattr(task_cfg, "dim", None) == getattr(cfg, "dim", None) and getattr(
                task_cfg,
                "t5_checkpoint",
                None,
            ) == getattr(cfg, "t5_checkpoint", None):
                return task
        raise ValueError("Unable to infer Wan2.2 task name from config")

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
            print("Using HMONNX high noise model for timestep", t.item())
            return _Wan22HmonnxDiTProxy(self.hmonnx_high_noise_model, model)
        if t.item() < boundary and self.hmonnx_low_noise_model is not None:
            print("Using HMONNX low noise model for timestep", t.item())
            return _Wan22HmonnxDiTProxy(self.hmonnx_low_noise_model, model)
        return _Wan22RuntimeModelProxy(model)


class WanT2VHMONNXPipeline(_Wan22ModuleSwitchMixin, WanT2V):
    def __init__(self, cfg, checkpoint_dir, *args, **kwargs):
        self._init_with_resolved_float_loader(cfg, checkpoint_dir, *args, **kwargs)

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
    def __init__(self, cfg, checkpoint_dir, *args, **kwargs):
        self._init_with_resolved_float_loader(cfg, checkpoint_dir, *args, **kwargs)

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
