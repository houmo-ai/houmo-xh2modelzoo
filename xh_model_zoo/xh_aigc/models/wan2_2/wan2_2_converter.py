# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: I001

# pyright: reportMissingImports=false

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence, Tuple

import torch
from safetensors import safe_open

from xhquant.api import (
    ConfigDict,
    HMONNXGoldenInference,
    QuantScheme,
    convert_dynamo_model_to_quanted_model,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
    get_root_logger,
)
from .common import ensure_wan2_2_repo

ensure_wan2_2_repo()

from .dit_wrapper import Wan2_2DiTExportWrapper, build_wan_time_embeddings
from .t5_local import LocalT5EncoderModel
from .t5_wrapper import Wan2_2T5EncoderExportWrapper
from .vae_local import LocalWan2_1_VAE
from .vae_wrapper import Wan2_2VAEDecoderExportWrapper, Wan2_2VAEEncoderExportWrapper

from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.image2video import WanI2V  # noqa: E402
from wan.modules.model import WanModel  # noqa: E402
from wan.text2video import WanT2V  # noqa: E402

WAN_EXPORT_COMPONENTS = ("t5", "vae_encode", "vae_decode", "low_noise_model", "high_noise_model")
# WAN_EXPORT_COMPONENTS = ()


def load_safetensors(path: str) -> dict[str, torch.Tensor]:
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


@dataclass
class Wan22ConvertConfig:
    quant_scheme: QuantScheme = field(default_factory=QuantScheme)
    task: str = "t2v-A14B"
    size: Tuple[int, int] = (832, 480)
    frame_num: int = 81
    sample_steps: int = 4
    sample_shift: float = 5.0
    sample_guide_scale: float = 5.0
    prompt: str = "A calm seaside scene with gentle waves."
    negative_prompt: str = ""
    export_components: Tuple[str, ...] = ("t5",)
    golden_components: Tuple[str, ...] = ()
    torch_dtype: torch.dtype = torch.bfloat16
    base_seed: int = 0
    use_resolved_float_loader: bool = False


class Wan22Converter:
    def __init__(self, pretrained_model_path: str, convert_config: Wan22ConvertConfig):
        self.pretrained_model_path = Path(pretrained_model_path)
        self.config = convert_config
        self.logger = get_root_logger()
        self.export_components = self._normalize_components(convert_config.export_components)
        self.golden_components = self._normalize_components(convert_config.golden_components)

    @staticmethod
    def _normalize_components(components: Sequence[str]):
        normalized = []
        aliases = {
            "vae_encoder": "vae_encode",
            "vae_decoder": "vae_decode",
            "low": "low_noise_model",
            "high": "high_noise_model",
        }
        for component in components:
            name = aliases.get(component.strip().lower(), component.strip().lower())
            if name not in WAN_EXPORT_COMPONENTS:
                raise ValueError(f"Unsupported component: {component}, expected one of {WAN_EXPORT_COMPONENTS}")
            if name not in normalized:
                normalized.append(name)
        return tuple(normalized)

    @staticmethod
    def _resolve_existing_path(*candidates: Path) -> Path:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def _resolve_text_encoder_path(self) -> Path:
        ckpt_dir = self.pretrained_model_path
        cfg = WAN_CONFIGS[self.config.task]
        return self._resolve_existing_path(
            ckpt_dir / cfg.t5_checkpoint,
            ckpt_dir / "split_files" / "text_encoders" / "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        )

    def _resolve_vae_path(self) -> Path:
        ckpt_dir = self.pretrained_model_path
        cfg = WAN_CONFIGS[self.config.task]
        return self._resolve_existing_path(
            ckpt_dir / cfg.vae_checkpoint,
            ckpt_dir / "split_files" / "vae" / "wan_2.1_vae.safetensors",
            ckpt_dir / "split_files" / "vae" / "wan2.2_vae.safetensors",
        )

    def _resolve_noise_model_path(self, noise_model_name: str) -> Path:
        ckpt_dir = self.pretrained_model_path
        cfg = WAN_CONFIGS[self.config.task]
        if noise_model_name == "low_noise_model":
            subfolder = cfg.low_noise_checkpoint
            role = "low_noise"
        else:
            subfolder = cfg.high_noise_checkpoint
            role = "high_noise"

        merged_dir = ckpt_dir / "merged"
        if merged_dir.is_dir():
            merged_candidates = sorted(
                path
                for path in merged_dir.iterdir()
                if path.suffix == ".safetensors" and role in path.name and "merged" in path.name
            )
            if merged_candidates:
                return merged_candidates[0]

        split_dir = ckpt_dir / "split_files" / "diffusion_models"
        if split_dir.is_dir():
            split_candidates = sorted(
                path for path in split_dir.iterdir() if path.suffix == ".safetensors" and role in path.name
            )
            if split_candidates:
                return split_candidates[0]

        return ckpt_dir / subfolder

    def _build_wan_model_from_config(self, cfg, *, is_i2v: bool) -> WanModel:
        model_kwargs = dict(
            model_type="i2v" if is_i2v else "t2v",
            patch_size=cfg.patch_size,
            text_len=cfg.text_len,
            dim=cfg.dim,
            ffn_dim=cfg.ffn_dim,
            freq_dim=cfg.freq_dim,
            num_heads=cfg.num_heads,
            num_layers=cfg.num_layers,
            window_size=cfg.window_size,
            qk_norm=cfg.qk_norm,
            cross_attn_norm=cfg.cross_attn_norm,
            eps=cfg.eps,
        )
        if is_i2v:
            model_kwargs["in_dim"] = 36
        return WanModel(**model_kwargs)

    def _configure_noise_model(self, model: WanModel, *, device: torch.device, param_dtype: torch.dtype) -> WanModel:
        model.eval().requires_grad_(False)
        model.to(param_dtype)
        model.to(device)
        return model

    def _load_noise_model_from_path(self, cfg, *, device: torch.device, noise_model_name: str):
        model_path = self._resolve_noise_model_path(noise_model_name)
        if model_path.is_dir():
            subfolder = cfg.low_noise_checkpoint if noise_model_name == "low_noise_model" else cfg.high_noise_checkpoint
            model = WanModel.from_pretrained(str(self.pretrained_model_path), subfolder=subfolder)
        else:
            model = self._build_wan_model_from_config(cfg, is_i2v=self.config.task.startswith("i2v"))
            model_state_dict = load_safetensors(str(model_path))
            missing_keys, unexpected_keys = model.load_state_dict(model_state_dict, strict=True)
            if missing_keys:
                self.logger.warning("Missing keys when loading %s: %s", noise_model_name, missing_keys[:20])
            if unexpected_keys:
                self.logger.warning("Unexpected keys when loading %s: %s", noise_model_name, unexpected_keys[:20])

        return self._configure_noise_model(model, device=device, param_dtype=cfg.param_dtype)

    def _build_resolved_float_pipeline(self, cfg, device_id: int = 0, rank: int = 0):
        device = torch.device(f"cuda:{device_id}")
        text_encoder_path = self._resolve_text_encoder_path()
        vae_path = self._resolve_vae_path()

        text_encoder = LocalT5EncoderModel(
            text_len=cfg.text_len,
            dtype=cfg.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=str(text_encoder_path),
            tokenizer_path=str(self.pretrained_model_path / cfg.t5_tokenizer),
            shard_fn=None,
        )
        vae = LocalWan2_1_VAE(
            vae_pth=str(vae_path),
            device=device,
            dtype=cfg.param_dtype,
        )
        vae.model = vae.model.to(cfg.param_dtype)

        need_low_noise = "low_noise_model" in self.export_components or "low_noise_model" in self.golden_components
        need_high_noise = "high_noise_model" in self.export_components or "high_noise_model" in self.golden_components
        low_noise_model = (
            self._load_noise_model_from_path(cfg, device=device, noise_model_name="low_noise_model")
            if need_low_noise
            else None
        )
        high_noise_model = (
            self._load_noise_model_from_path(cfg, device=device, noise_model_name="high_noise_model")
            if need_high_noise
            else None
        )

        return SimpleNamespace(
            device=device,
            config=cfg,
            rank=rank,
            t5_cpu=False,
            init_on_cpu=True,
            checkpoint_dir=str(self.pretrained_model_path),
            num_train_timesteps=cfg.num_train_timesteps,
            boundary=cfg.boundary,
            param_dtype=cfg.param_dtype,
            text_encoder=text_encoder,
            vae_stride=cfg.vae_stride,
            patch_size=cfg.patch_size,
            vae=vae,
            low_noise_model=low_noise_model,
            high_noise_model=high_noise_model,
            sp_size=1,
            sample_neg_prompt=getattr(cfg, "sample_neg_prompt", ""),
        )

    def _reload_pipeline_components_from_resolved_paths(self, pipe, cfg, device_id: int) -> None:
        text_encoder_path = self._resolve_text_encoder_path()
        vae_path = self._resolve_vae_path()

        pipe.text_encoder = LocalT5EncoderModel(
            text_len=cfg.text_len,
            dtype=cfg.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=str(text_encoder_path),
            tokenizer_path=str(self.pretrained_model_path / cfg.t5_tokenizer),
            shard_fn=None,
        )
        pipe.vae = LocalWan2_1_VAE(
            vae_pth=str(vae_path),
            device=torch.device(f"cuda:{device_id}"),
            dtype=cfg.param_dtype,
        )
        pipe.vae.model = pipe.vae.model.to(cfg.param_dtype)
        device = torch.device(f"cuda:{device_id}")
        pipe.low_noise_model = self._load_noise_model_from_path(cfg, device=device, noise_model_name="low_noise_model")
        pipe.high_noise_model = self._load_noise_model_from_path(
            cfg, device=device, noise_model_name="high_noise_model"
        )

    def build_float_pipeline(self, device_id: int = 0, rank: int = 0):
        cfg = WAN_CONFIGS[self.config.task]
        cfg.param_dtype = torch.float16
        cfg.t5_dtype = torch.float16
        if self.config.use_resolved_float_loader:
            return self._build_resolved_float_pipeline(cfg, device_id=device_id, rank=rank)
        pipeline_cls = WanI2V if self.config.task.startswith("i2v") else WanT2V
        pipe = pipeline_cls(
            cfg,
            str(self.pretrained_model_path),
            device_id=device_id,
            rank=rank,
            convert_model_dtype=True,
        )
        return pipe

    def _export_component(
        self,
        model: torch.nn.Module,
        inputs: Sequence[torch.Tensor],
        input_names: Sequence[str],
        output_names: Sequence[str],
        onnx_file: Path,
        golden_dir: Path,
        export_golden: bool,
    ) -> Dict[str, Any]:
        quant_config = ConfigDict(create_quant_config(self.config.quant_scheme))
        quanted_model = convert_dynamo_model_to_quanted_model(
            model,
            list(inputs),
            self.config.quant_scheme.target_device,
            quant_config,
        )
        onnx_file.parent.mkdir(parents=True, exist_ok=True)
        convert_quanted_model_to_hmonnx(
            quanted_model,
            list(inputs),
            str(onnx_file),
            list(input_names),
            list(output_names),
        )

        if export_golden:
            hm_session = HMONNXGoldenInference(str(onnx_file))
            hm_session.save_golden = True
            hm_session.exec_device = self.device
            golden_dir.mkdir(parents=True, exist_ok=True)
            hm_session.golden_dir = str(golden_dir)
            with torch.no_grad():
                hm_session.forward(*list(inputs))
        return {
            "onnx": str(onnx_file),
            "golden_dir": str(golden_dir) if export_golden else None,
            "input_names": list(input_names),
            "output_names": list(output_names),
        }

    def _build_latent_shape(self, pipe) -> Tuple[int, int, int, int]:
        width, height = self.config.size
        latent_t = (self.config.frame_num - 1) // pipe.vae_stride[0] + 1
        latent_h = height // pipe.vae_stride[1]
        latent_w = width // pipe.vae_stride[2]
        # latent_h = 60
        # latent_w = 104
        return pipe.vae.model.z_dim, latent_t, latent_h, latent_w

    def _build_seq_len(self, pipe, latent_shape: Tuple[int, int, int, int]) -> int:
        _, latent_t, latent_h, latent_w = latent_shape
        return int(
            math.ceil((latent_h * latent_w) / (pipe.patch_size[1] * pipe.patch_size[2]) * latent_t / pipe.sp_size)
            * pipe.sp_size
        )

    def _build_context(self, pipe, prompt: str):
        pipe.text_encoder.model.to(self.device)
        contexts = [u.to(self.device, dtype=self.config.torch_dtype) for u in pipe.text_encoder([prompt], self.device)]
        if len(contexts) != 1:
            raise ValueError(f"Expected a single context tensor for Wan2.2 DiT export, got {len(contexts)}")
        return contexts[0]

    def _export_t5(self, pipe, work_dir: Path) -> Dict[str, Any]:
        component_dir = work_dir / "t5"
        component_dir.mkdir(parents=True, exist_ok=True)
        t5_dtype = torch.float16
        wrapper = (
            Wan2_2T5EncoderExportWrapper(
                pipe.text_encoder.model,
                max_sequence_length=int(pipe.text_encoder.text_len),
            )
            .to(self.device)
            .eval()
        )
        ids, mask = pipe.text_encoder.tokenizer([self.config.prompt], return_mask=True, add_special_tokens=True)
        input_ids = ids.to(self.device, dtype=torch.int64)
        attention_mask = mask.to(self.device, dtype=torch.int32)
        inputs_embeds = pipe.text_encoder.model.token_embedding(input_ids.long()).to(
            self.device,
            dtype=t5_dtype,
        )
        mask_4d = attention_mask.view(attention_mask.shape[0], 1, 1, -1)
        mask_bias = torch.zeros(mask_4d.shape, device=self.device, dtype=t5_dtype).masked_fill(
            mask_4d == 0,
            -65504.0,
        )
        prefix = f"wan2_2_t5-{self.config.quant_scheme.target_device}-{self.config.quant_scheme.quant_type}"
        export_meta = self._export_component(
            model=wrapper,
            inputs=[inputs_embeds, mask_bias],
            input_names=["inputs_embeds", "mask_bias"],
            output_names=["context"],
            onnx_file=component_dir / "hmonnx" / f"{prefix}.onnx",
            golden_dir=component_dir / "golden" / prefix,
            export_golden="t5" in self.golden_components,
        )
        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task": self.config.task,
            "checkpoint_dir": str(self.pretrained_model_path),
            "tokenizer_path": str(self.pretrained_model_path / WAN_CONFIGS[self.config.task].t5_tokenizer),
            "checkpoint_path": str(self._resolve_text_encoder_path()),
            "max_sequence_length": int(pipe.text_encoder.text_len),
            "hmonnx_file": str(Path(export_meta["onnx"]).relative_to(work_dir)),
            "golden_dir": (
                str(Path(export_meta["golden_dir"]).relative_to(work_dir)) if export_meta["golden_dir"] else None
            ),
            "input_names": export_meta["input_names"],
            "output_names": export_meta["output_names"],
            "sample_input_shapes": {
                "inputs_embeds": list(inputs_embeds.shape),
                "mask_bias": list(mask_bias.shape),
            },
            "sample_token_input_shapes": {
                "input_ids": list(input_ids.shape),
                "attention_mask": list(attention_mask.shape),
            },
            "dtype": str(t5_dtype).replace("torch.", ""),
        }
        json.dump(meta, open(work_dir / "t5_meta.json", "w"), indent=2, ensure_ascii=False)
        return meta

    def _build_dit_inputs(self, pipe, noise_model_name: str):
        latent_shape = self._build_latent_shape(pipe)
        seq_len = self._build_seq_len(pipe, latent_shape)
        latent = torch.randn(latent_shape, device=self.device, dtype=self.config.torch_dtype)
        context = self._build_context(pipe, self.config.prompt)
        boundary = pipe.boundary * pipe.num_train_timesteps
        if noise_model_name == "low_noise_model":
            timestep = torch.tensor([max(0.0, float(boundary) - 1.0)], device=self.device, dtype=torch.float32)
            model = pipe.low_noise_model
        else:
            timestep = torch.tensor([float(boundary)], device=self.device, dtype=torch.float32)
            model = pipe.high_noise_model
        y = None
        if self.config.task.startswith("i2v"):
            cond_channels = int(model.config.in_dim - pipe.vae.model.z_dim)
            y = torch.randn(
                (cond_channels, latent_shape[1], latent_shape[2], latent_shape[3]),
                device=self.device,
                dtype=self.config.torch_dtype,
            )

        del pipe.text_encoder
        del pipe.vae
        torch.cuda.empty_cache()

        return model, latent, timestep, context, seq_len, y

    def _validate_dit_wrap(self, model, latent, timestep, context, seq_len, y, noise_model_name: str):

        validate_dtype = torch.float16
        model = model.to(device=self.device, dtype=validate_dtype).eval()
        latent_ref = latent.to(self.device, dtype=validate_dtype)
        context_ref = context.to(self.device, dtype=validate_dtype)
        y_ref = y.to(self.device, dtype=validate_dtype) if y is not None else None

        model.time_embedding = model.time_embedding.to(torch.float32)
        model.time_projection = model.time_projection.to(torch.float32)

        with torch.no_grad():
            out_ref = model(
                [latent_ref],
                t=timestep,
                context=[context_ref],
                seq_len=seq_len,
                y=[y_ref] if y_ref is not None else None,
            )[0]

        wrapper = Wan2_2DiTExportWrapper(model).to(self.device).eval()
        e_ref, e0_ref = build_wan_time_embeddings(model, timestep, seq_len, validate_dtype)
        latent_arg = wrapper._prepare_latent(latent_ref, y_ref)
        context_arg = wrapper._prepare_context(context_ref)

        torch.cuda.empty_cache()

        with torch.no_grad():
            out_wrap = model(
                latent_arg,
                context=context_arg,
                e=e_ref,
                e0=e0_ref,
            )[0]

        diff = (out_ref - out_wrap).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        is_close = torch.allclose(out_ref, out_wrap, atol=1e-2, rtol=1e-2)
        self.logger.info(
            "Wan2.2 %s warp validation: max_abs=%.6f mean_abs=%.6f allclose=%s",
            noise_model_name,
            max_abs,
            mean_abs,
            is_close,
        )
        if not is_close:
            raise AssertionError(
                f"{noise_model_name} warp validation failed: max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}"
            )
        return model

    def _export_dit(self, pipe, work_dir: Path, noise_model_name: str) -> Dict[str, Any]:
        component_dir = work_dir / noise_model_name
        component_dir.mkdir(parents=True, exist_ok=True)
        model, latent, timestep, context, seq_len, y = self._build_dit_inputs(pipe, noise_model_name)
        # model = self._validate_dit_wrap(model, latent, timestep, context, seq_len, y, noise_model_name)
        dit_dtype = self.config.torch_dtype
        wrapper = Wan2_2DiTExportWrapper(model).to(self.device).eval()
        model = wrapper.model.to(torch.float16).eval()
        latent_fp16 = latent.to(torch.float16)
        context_fp16 = context.to(torch.float16)
        y_fp16 = y.to(torch.float16) if y is not None else None
        latent_arg = wrapper._prepare_latent(latent_fp16, y_fp16)
        context_arg = wrapper._prepare_context(context_fp16)
        e, e0 = build_wan_time_embeddings(model, timestep, seq_len, torch.float16)
        prefix = (
            f"wan2_2_{noise_model_name}-{self.config.quant_scheme.target_device}-{self.config.quant_scheme.quant_type}"
        )
        inputs = [
            latent_arg,
            context_arg,
            e,
            e0,
        ]
        input_names = [
            "latent",
            "context",
            "e",
            "e0",
        ]
        export_meta = self._export_component(
            model=model,
            inputs=inputs,
            input_names=input_names,
            output_names=["sample"],
            onnx_file=component_dir / "hmonnx" / f"{prefix}.onnx",
            golden_dir=component_dir / "golden" / prefix,
            export_golden=noise_model_name in self.golden_components,
        )
        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task": self.config.task,
            "checkpoint_dir": str(self.pretrained_model_path),
            "checkpoint_path": str(self._resolve_noise_model_path(noise_model_name)),
            "hmonnx_file": str(Path(export_meta["onnx"]).relative_to(work_dir)),
            "golden_dir": (
                str(Path(export_meta["golden_dir"]).relative_to(work_dir)) if export_meta["golden_dir"] else None
            ),
            "input_names": export_meta["input_names"],
            "output_names": export_meta["output_names"],
            "sample_input_shapes": {
                name: list(t.shape) if isinstance(t, torch.Tensor) else list(t[0].shape)
                for name, t in zip(input_names, inputs, strict=True)
            },
            "seq_len": int(seq_len),
            "sample_timestep": timestep.tolist(),
            "dtype": str(dit_dtype).replace("torch.", ""),
        }
        json.dump(meta, open(work_dir / f"{noise_model_name}_meta.json", "w"), indent=2, ensure_ascii=False)
        return meta

    def _validate_vae_encode_wrap(self, pipe, video: torch.Tensor):
        validate_dtype = torch.float16
        vae_model = pipe.vae.model.to(device=self.device, dtype=validate_dtype).eval()
        scale = [
            item.to(self.device, dtype=validate_dtype) if isinstance(item, torch.Tensor) else item
            for item in pipe.vae.scale
        ]
        video_ref = video.to(self.device, dtype=validate_dtype)

        with torch.no_grad():
            out_ref = vae_model.encode(video_ref.unsqueeze(0), scale).squeeze(0)

        wrapper = Wan2_2VAEEncoderExportWrapper(vae_model, scale).to(self.device).eval()

        with torch.no_grad():
            out_wrap = wrapper(video_ref)

        diff = (out_ref - out_wrap).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        is_close = torch.allclose(out_ref, out_wrap, atol=1e-2, rtol=1e-2)
        self.logger.info(
            "Wan2.2 vae_encode warp validation: max_abs=%.6f mean_abs=%.6f allclose=%s",
            max_abs,
            mean_abs,
            is_close,
        )
        if not is_close:
            raise AssertionError(f"vae_encode warp validation failed: max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}")
        return vae_model, scale

    def _validate_vae_decode_wrap(self, pipe, latent: torch.Tensor):
        validate_dtype = torch.float16
        vae_model = pipe.vae.model.to(device=self.device, dtype=validate_dtype).eval()
        scale = [
            item.to(self.device, dtype=validate_dtype) if isinstance(item, torch.Tensor) else item
            for item in pipe.vae.scale
        ]
        latent_ref = latent.to(self.device, dtype=validate_dtype)

        with torch.no_grad():
            out_ref = vae_model.decode(latent_ref.unsqueeze(0), scale).squeeze(0)

        wrapper = Wan2_2VAEDecoderExportWrapper(vae_model, scale).to(self.device).eval()

        with torch.no_grad():
            out_wrap = wrapper(latent_ref)

        diff = (out_ref - out_wrap).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        is_close = torch.allclose(out_ref, out_wrap, atol=1e-2, rtol=1e-2)
        self.logger.info(
            "Wan2.2 vae_decode warp validation: max_abs=%.6f mean_abs=%.6f allclose=%s",
            max_abs,
            mean_abs,
            is_close,
        )
        if not is_close:
            raise AssertionError(f"vae_decode warp validation failed: max_abs={max_abs:.6f}, mean_abs={mean_abs:.6f}")
        return vae_model, scale

    def _export_vae_encode(self, pipe, work_dir: Path) -> Dict[str, Any]:
        component_dir = work_dir / "vae_encode"
        component_dir.mkdir(parents=True, exist_ok=True)
        width, height = self.config.size
        video = torch.randn((3, self.config.frame_num, height, width), device=self.device, dtype=pipe.param_dtype)
        # vae_model, scale = self._validate_vae_encode_wrap(pipe, video)

        validate_dtype = torch.float16
        vae_model = pipe.vae.model.to(device=self.device, dtype=validate_dtype).eval()
        scale = [
            item.to(self.device, dtype=validate_dtype) if isinstance(item, torch.Tensor) else item
            for item in pipe.vae.scale
        ]

        wrapper = Wan2_2VAEEncoderExportWrapper(vae_model, scale).to(self.device).eval()
        prefix = f"wan2_2_vae_encode-{self.config.quant_scheme.target_device}-{self.config.quant_scheme.quant_type}"
        export_meta = self._export_component(
            model=wrapper,
            inputs=[video],
            input_names=["video"],
            output_names=["latent"],
            onnx_file=component_dir / "hmonnx" / f"{prefix}.onnx",
            golden_dir=component_dir / "golden" / prefix,
            export_golden="vae_encode" in self.golden_components,
        )
        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task": self.config.task,
            "hmonnx_file": str(Path(export_meta["onnx"]).relative_to(component_dir)),
            "golden_dir": (
                str(Path(export_meta["golden_dir"]).relative_to(component_dir)) if export_meta["golden_dir"] else None
            ),
            "input_names": export_meta["input_names"],
            "output_names": export_meta["output_names"],
            "sample_input_shapes": {"video": list(video.shape)},
        }
        json.dump(meta, open(work_dir / "vae_encode_meta.json", "w"), indent=2, ensure_ascii=False)
        return meta

    def _export_vae_decode(self, pipe, work_dir: Path) -> Dict[str, Any]:
        component_dir = work_dir / "vae_decode"
        component_dir.mkdir(parents=True, exist_ok=True)
        latent_shape = self._build_latent_shape(pipe)
        latent = torch.randn(latent_shape, device=self.device, dtype=pipe.param_dtype)
        vae_model, scale = self._validate_vae_decode_wrap(pipe, latent)
        wrapper = Wan2_2VAEDecoderExportWrapper(vae_model, scale).to(self.device).eval()
        prefix = f"wan2_2_vae_decode-{self.config.quant_scheme.target_device}-{self.config.quant_scheme.quant_type}"
        export_meta = self._export_component(
            model=wrapper,
            inputs=[latent],
            input_names=["latent"],
            output_names=["video"],
            onnx_file=component_dir / "hmonnx" / f"{prefix}.onnx",
            golden_dir=component_dir / "golden" / prefix,
            export_golden="vae_decode" in self.golden_components,
        )
        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task": self.config.task,
            "hmonnx_file": str(Path(export_meta["onnx"]).relative_to(component_dir)),
            "golden_dir": (
                str(Path(export_meta["golden_dir"]).relative_to(component_dir)) if export_meta["golden_dir"] else None
            ),
            "input_names": export_meta["input_names"],
            "output_names": export_meta["output_names"],
            "sample_input_shapes": {"latent": list(latent.shape)},
        }
        json.dump(meta, open(work_dir / "vae_decode_meta.json", "w"), indent=2, ensure_ascii=False)
        return meta

    def export(self, work_dir: str):
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        component_meta: Dict[str, Any] = {}
        if "t5" in self.export_components:
            pipe = self.build_float_pipeline(device_id=0, rank=0)
            try:
                component_meta["t5"] = self._export_t5(pipe, work_dir)
            finally:
                del pipe
                torch.cuda.empty_cache()
        if "low_noise_model" in self.export_components:
            pipe = self.build_float_pipeline(device_id=0, rank=0)
            try:
                component_meta["low_noise_model"] = self._export_dit(pipe, work_dir, "low_noise_model")
            finally:
                del pipe
                torch.cuda.empty_cache()
        if "high_noise_model" in self.export_components:
            pipe = self.build_float_pipeline(device_id=0, rank=0)
            try:
                component_meta["high_noise_model"] = self._export_dit(pipe, work_dir, "high_noise_model")
            finally:
                del pipe
                torch.cuda.empty_cache()
        if "vae_encode" in self.export_components:
            pipe = self.build_float_pipeline(device_id=0, rank=0)
            try:
                component_meta["vae_encode"] = self._export_vae_encode(pipe, work_dir)
            finally:
                del pipe
                torch.cuda.empty_cache()
        if "vae_decode" in self.export_components:
            pipe = self.build_float_pipeline(device_id=0, rank=0)
            try:
                component_meta["vae_decode"] = self._export_vae_decode(pipe, work_dir)
            finally:
                del pipe
                torch.cuda.empty_cache()

        meta = {
            "create_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "task": self.config.task,
            "checkpoint_dir": str(self.pretrained_model_path),
            "export_components": list(self.export_components),
            "golden_components": list(self.golden_components),
            "quant_scheme": self.config.quant_scheme.to_dict(),
            "size": list(self.config.size),
            "frame_num": self.config.frame_num,
            "sample_steps": self.config.sample_steps,
            "sample_shift": self.config.sample_shift,
            "sample_guide_scale": self.config.sample_guide_scale,
            "prompt": self.config.prompt,
            "status": "exported",
        }
        if "t5" in component_meta:
            meta["t5_meta"] = "t5_meta.json"
        if "low_noise_model" in component_meta:
            meta["low_noise_model_meta"] = "low_noise_model_meta.json"
        if "high_noise_model" in component_meta:
            meta["high_noise_model_meta"] = "high_noise_model_meta.json"
        if "vae_encode" in component_meta:
            meta["vae_encode_meta"] = "vae_encode_meta.json"
        if "vae_decode" in component_meta:
            meta["vae_decode_meta"] = "vae_decode_meta.json"
        meta_path = work_dir / "wan2_2_export_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False))
        self.logger.info("Created Wan2.2 export meta at %s", meta_path)
        return meta_path

    @classmethod
    def from_pretrained(cls, pretrained_model_path: str, convert_config: Wan22ConvertConfig, work_dir: str):
        converter = cls(pretrained_model_path, convert_config)
        return converter.export(work_dir)


Wan2_2ConvertConfig = Wan22ConvertConfig
Wan2_2Converter = Wan22Converter
