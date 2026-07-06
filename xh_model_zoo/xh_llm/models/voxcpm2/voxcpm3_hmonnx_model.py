"""VoxCPM2 HMONNX 推理 pipeline."""

from __future__ import annotations

import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Generator, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaTokenizerFast

from voxcpm.model.voxcpm2 import LoRAConfig
from voxcpm.model.utils import get_dtype, mask_multichar_chinese_tokens
from voxcpm.modules.layers import ScalarQuantizationLayer

from .voxcpm2_hmonnx_sessions import (
    AudioVAEDecoderSession,
    AudioVAEEncoderSession,
    AudioVAEStatefulStreamingDecoderSession,
    BaseLMDecodeSession,
    BaseLMPrefillSession,
    LocDiTStepSession,
    LocEncStepSession,
    ResidualLMDecodeSession,
    ResidualLMPrefillSession,
    load_meta,
)


class _HostSideModules(nn.Module):
    """Host-side modules that stay out of HMONNX graphs."""

    def __init__(self, work_dir: Path, lm_meta: dict, hf_config: dict, device: torch.device, dtype: torch.dtype):
        super().__init__()

        host_dir = (work_dir / Path(lm_meta["host_modules"]["token_embedding"])).parent

        hidden_size = int(hf_config["lm_config"]["hidden_size"])
        vocab_size = int(hf_config["lm_config"]["vocab_size"])
        encoder_hidden = int(hf_config["encoder_config"]["hidden_dim"])
        dit_hidden = int(hf_config["dit_config"]["hidden_dim"])
        fsq_latent = int(hf_config["scalar_quantization_latent_dim"])
        fsq_scale = int(hf_config["scalar_quantization_scale"])

        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self._load_state(self.token_embedding, work_dir / lm_meta["host_modules"]["token_embedding"])

        self.enc_to_lm_proj = nn.Linear(encoder_hidden, hidden_size)
        self._load_state(self.enc_to_lm_proj, work_dir / lm_meta["host_modules"]["enc_to_lm_proj"])

        self.lm_to_dit_proj = nn.Linear(hidden_size, dit_hidden)
        self._load_state(self.lm_to_dit_proj, work_dir / lm_meta["host_modules"]["lm_to_dit_proj"])

        self.res_to_dit_proj = nn.Linear(hidden_size, dit_hidden)
        self._load_state(self.res_to_dit_proj, work_dir / lm_meta["host_modules"]["res_to_dit_proj"])

        self.fusion_concat_proj = nn.Linear(hidden_size * 2, hidden_size)
        self._load_state(self.fusion_concat_proj, work_dir / lm_meta["host_modules"]["fusion_concat_proj"])

        self.fsq_layer = ScalarQuantizationLayer(
            hidden_size, hidden_size, fsq_latent, fsq_scale,
        )
        self._load_state(self.fsq_layer, work_dir / lm_meta["host_modules"]["fsq_layer"])

        self.stop_proj = nn.Linear(hidden_size, hidden_size)
        self._load_state(self.stop_proj, work_dir / lm_meta["host_modules"]["stop_proj"])

        self.stop_actn = nn.SiLU()

        self.stop_head = nn.Linear(hidden_size, 2, bias=False)
        self._load_state(self.stop_head, work_dir / lm_meta["host_modules"]["stop_head"])

        self.to(device=device, dtype=dtype)
        self.eval()

    @staticmethod
    def _load_state(module: nn.Module, path: Path):
        sd = torch.load(str(path), map_location="cpu", weights_only=True)
        module.load_state_dict(sd)


class _LocalCFMSolver:
    """Host-side diffusion solver backed by LocDiTStepSession."""

    def __init__(
        self,
        locdit_session: LocDiTStepSession,
        in_channels: int,
        patch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        mean_mode: bool = False,
    ):
        self.locdit = locdit_session
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.device = device
        self.dtype = dtype
        self.mean_mode = mean_mode

    @torch.inference_mode()
    def __call__(
        self,
        mu: torch.Tensor,                  # [B=1, 2*H_dit]
        cond: torch.Tensor,                # [B=1, C, T_cond]
        n_timesteps: int = 10,
        cfg_value: float = 2.0,
        temperature: float = 1.0,
        sway_sampling_coef: float = 1.0,
        use_cfg_zero_star: bool = True,
    ) -> torch.Tensor:
        """返回 [B=1, C, T=patch_size] 的预测 latent。"""
        b, _ = mu.shape
        assert b == 1, "pipeline 目前仅支持 batch=1"
        t_dim = self.patch_size

        z = torch.randn(
            (b, self.in_channels, t_dim), device=self.device, dtype=self.dtype,
        ) * temperature

        t_span = torch.linspace(1, 0, n_timesteps + 1, device=self.device, dtype=self.dtype)
        t_span = t_span + sway_sampling_coef * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)

        x = z
        t = t_span[0]
        dt = t_span[0] - t_span[1]
        zero_init_steps = max(1, int(len(t_span) * 0.04))
        for step in range(1, len(t_span)):
            if use_cfg_zero_star and step <= zero_init_steps:
                dphi_dt = torch.zeros_like(x)
            else:
                x_in = torch.zeros(
                    (2 * b, self.in_channels, x.size(2)), device=self.device, dtype=self.dtype,
                )
                mu_in = torch.zeros(
                    (2 * b, mu.size(1)), device=self.device, dtype=self.dtype,
                )
                t_in = torch.zeros((2 * b,), device=self.device, dtype=self.dtype)
                dt_in = torch.zeros((2 * b,), device=self.device, dtype=self.dtype)
                cond_in = torch.zeros(
                    (2 * b, self.in_channels, cond.size(2)),
                    device=self.device, dtype=self.dtype,
                )
                x_in[:b], x_in[b:] = x, x
                mu_in[:b] = mu
                t_in[:b], t_in[b:] = t.unsqueeze(0), t.unsqueeze(0)
                dt_in[:b], dt_in[b:] = dt.unsqueeze(0), dt.unsqueeze(0)
                if not self.mean_mode:
                    dt_in = torch.zeros_like(dt_in)
                cond_in[:b], cond_in[b:] = cond, cond

                v_out = self.locdit(x_in, mu_in, t_in, cond_in, dt_in)
                dphi_dt, cfg_dphi_dt = torch.split(v_out, [b, b], dim=0)

                if use_cfg_zero_star:
                    positive_flat = dphi_dt.reshape(b, -1)
                    negative_flat = cfg_dphi_dt.reshape(b, -1)
                    dot = (positive_flat * negative_flat).sum(dim=1, keepdim=True)
                    sq = (negative_flat ** 2).sum(dim=1, keepdim=True) + 1e-8
                    st_star = (dot / sq).view(b, *([1] * (dphi_dt.dim() - 1)))
                else:
                    st_star = 1.0

                dphi_dt = cfg_dphi_dt * st_star + cfg_value * (dphi_dt - cfg_dphi_dt * st_star)

            x = x - dt * dphi_dt
            t = t - dt
            if step < len(t_span) - 1:
                dt = t - t_span[step + 1]

        return x


class VoxCPM2HMONNXTTSPipeline(nn.Module):
    """对齐 `voxcpm.VoxCPM.generate()` 接口的 HMONNX 推理 pipeline。

    - 所有神经网络 forward 换成 HMONNX session
    - `audio_vae.encode/decode` 换成 session
    - diffusion solver 在 host 侧用 Python 循环
    - 不支持 denoiser / text_normalizer(这些在原生 VoxCPM 外层做,
      pipeline 层保持纯推理)
    - 不支持 LoRA 切换(权重已经 bake 进量化产物)
    """

    def __init__(
        self,
        work_dir: Union[str, Path],
        device: str = "cpu",
        audio_encoder_backend: str = "auto",
        torch_audio_model_dir: Optional[Union[str, Path]] = None,
    ):
        super().__init__()
        work_dir = Path(work_dir).expanduser().resolve()
        self.work_dir = work_dir
        self.device = torch.device(device)
        self.audio_encoder_backend = str(audio_encoder_backend).lower().strip()
        if self.audio_encoder_backend not in {"auto", "hmonnx", "torch"}:
            raise ValueError(
                f"audio_encoder_backend must be one of ['auto','hmonnx','torch'], "
                f"but got {audio_encoder_backend!r}."
            )
        self._torch_audio_vae = None
        self._torch_audio_vae_load_error: Optional[Exception] = None

        lm_meta_path = work_dir / "lm_export_meta_info.json"
        if not lm_meta_path.exists():
            raise FileNotFoundError(f"lm_export_meta_info.json 不在 {work_dir}")
        with open(lm_meta_path, "r", encoding="utf-8") as f:
            lm_meta = json.load(f)
        self.lm_meta = lm_meta
        model_dir = torch_audio_model_dir or lm_meta.get("hf_model")
        self.torch_audio_model_dir = (
            Path(model_dir).expanduser().resolve() if model_dir else None
        )

        hf_config_dir = work_dir / lm_meta["hf_config"]
        with open(hf_config_dir / "config.json", "r", encoding="utf-8") as f:
            self.hf_config = json.load(f)

        self.input_dtype = getattr(torch, lm_meta["input_dtype"], torch.float16)
        self.prefill_length = lm_meta["prefill_length"]
        self.cache_length = int(lm_meta["cache_length"])

        tokenizer = LlamaTokenizerFast.from_pretrained(str(hf_config_dir))
        self.text_tokenizer = mask_multichar_chinese_tokens(tokenizer)

        self.audio_start_token = 101
        self.audio_end_token = 102
        self.ref_audio_start_token = 103
        self.ref_audio_end_token = 104
        self.pad_token_id = 0

        self.patch_size = int(self.hf_config["patch_size"])
        self.feat_dim = int(self.hf_config["feat_dim"])
        self.scale_emb = (
            float(self.hf_config["lm_config"]["scale_emb"])
            if self.hf_config["lm_config"].get("use_mup", False)
            else 1.0
        )
        self.dit_in_channels = self.feat_dim
        self.h_dit = int(self.hf_config["dit_config"]["hidden_dim"])
        self.inference_cfg_rate = float(
            self.hf_config["dit_config"]["cfm_config"]["inference_cfg_rate"]
        )

        self.host = _HostSideModules(
            work_dir, lm_meta, self.hf_config, self.device, self.input_dtype,
        )

        base = lm_meta["base_lm"]
        resid = lm_meta["residual_lm"]
        base_kv_len = int(base.get("kv_cache_shape", [0, 0, 0, 0])[2])
        resid_kv_len = int(resid.get("kv_cache_shape", [0, 0, 0, 0])[2])
        kv_cache_length = max(
            self.prefill_length + self.cache_length,
            base_kv_len,
            resid_kv_len,
        )
        self.base_prefill = BaseLMPrefillSession(
            onnx_path=str(work_dir / base["prefill_onnx"]),
            prefill_length=self.prefill_length,
            num_hidden_layers=base["num_hidden_layers"],
            num_key_value_heads=base["num_key_value_heads"],
            cache_length=kv_cache_length,
            head_dim=base["head_dim"],
            device=self.device,
            dtype=self.input_dtype,
        )
        self.base_decode = BaseLMDecodeSession(
            onnx_path=str(work_dir / base["decode_onnx"]),
            num_hidden_layers=base["num_hidden_layers"],
            num_key_value_heads=base["num_key_value_heads"],
            cache_length=kv_cache_length,
            head_dim=base["head_dim"],
            device=self.device,
            dtype=self.input_dtype,
        )
        self.base_decode.past_k_caches = self.base_prefill.past_k_caches
        self.base_decode.past_v_caches = self.base_prefill.past_v_caches

        self.residual_prefill = ResidualLMPrefillSession(
            onnx_path=str(work_dir / resid["prefill_onnx"]),
            prefill_length=self.prefill_length,
            num_hidden_layers=resid["num_hidden_layers"],
            num_key_value_heads=resid["num_key_value_heads"],
            cache_length=kv_cache_length,
            head_dim=resid["head_dim"],
            device=self.device,
            dtype=self.input_dtype,
        )
        self.residual_decode = ResidualLMDecodeSession(
            onnx_path=str(work_dir / resid["decode_onnx"]),
            num_hidden_layers=resid["num_hidden_layers"],
            num_key_value_heads=resid["num_key_value_heads"],
            cache_length=kv_cache_length,
            head_dim=resid["head_dim"],
            device=self.device,
            dtype=self.input_dtype,
        )
        self.residual_decode.past_k_caches = self.residual_prefill.past_k_caches
        self.residual_decode.past_v_caches = self.residual_prefill.past_v_caches

        locenc_dir = work_dir / "LocEnc"
        locenc_meta = load_meta(locenc_dir / "locenc_meta_info.json")
        self.locenc = LocEncStepSession(
            str(work_dir / locenc_meta["hmonnx_file"]),
            device=self.device,
            dtype=self.input_dtype,
        )

        locdit_dir = work_dir / "LocDiT"
        locdit_meta = load_meta(locdit_dir / "locdit_meta_info.json")
        self.locdit = LocDiTStepSession(
            str(work_dir / locdit_meta["hmonnx_file"]),
            device=self.device,
            dtype=self.input_dtype,
        )

        vae_cfg = self.hf_config.get("audio_vae_config", {})
        fallback_encode_sr = int(vae_cfg.get("sample_rate", 16000))
        fallback_chunk_size = int(math.prod(vae_cfg.get("encoder_rates", [1])))
        fallback_latent_dim = int(vae_cfg.get("latent_dim", 64))

        vae_enc_meta, _ = self._find_encoder(work_dir)
        if vae_enc_meta is not None:
            self.vae_encoder = AudioVAEEncoderSession(
                str(work_dir / vae_enc_meta["hmonnx_file"]),
                vae_enc_meta,
                device=self.device,
            )
            self.vae_encode_sr = int(vae_enc_meta["sample_rate"])
        else:
            warnings.warn(
                "AudioVAE_Encoder export not found. "
                "Zero-shot synthesis can still run, but prompt/reference wav modes are unavailable.",
                RuntimeWarning,
            )
            self.vae_encoder = None
            self.vae_encode_sr = fallback_encode_sr

        stream_meta, stream_path = self._find_decoder(work_dir, prefer="stream")
        full_meta, full_path = self._find_decoder(work_dir, prefer="full")
        self.vae_decoder = AudioVAEDecoderSession(
            stream_onnx_path=str(work_dir / stream_meta["hmonnx_file"]) if stream_meta else None,
            stream_meta=stream_meta,
            full_onnx_path=str(work_dir / full_meta["hmonnx_file"]) if full_meta else None,
            full_meta=full_meta,
            device=self.device,
            dtype=self.input_dtype,
        )
        stateful_stream_meta, _ = self._find_stateful_stream_decoder(work_dir)
        self.vae_stream_decoder = None
        if stateful_stream_meta is not None:
            self.vae_stream_decoder = AudioVAEStatefulStreamingDecoderSession(
                str(work_dir / stateful_stream_meta["hmonnx_file"]),
                stateful_stream_meta,
                device=self.device,
                dtype=self.input_dtype,
            )
        self.out_sample_rate = (full_meta or stream_meta)["out_sample_rate"]
        if vae_enc_meta is not None:
            self.chunk_size = int(vae_enc_meta["chunk_size"])
            self.latent_dim = int(vae_enc_meta["latent_dim"])
        else:
            self.chunk_size = fallback_chunk_size
            self.latent_dim = fallback_latent_dim

        self.solver = _LocalCFMSolver(
            self.locdit, self.dit_in_channels, self.patch_size,
            self.device, self.input_dtype,
            mean_mode=bool(locdit_meta.get("mean_mode", False)),
        )

        self.sample_rate = self.out_sample_rate
        self._encode_sample_rate = self.vae_encode_sr

    # ------------------------------------------------------------------ #
    # Decoder 目录搜索
    # ------------------------------------------------------------------ #

    def _find_decoder(self, work_dir: Path, prefer: str) -> Tuple[Optional[dict], Optional[str]]:
        """在 work_dir 下查找所有 AudioVAE_Decoder_np* 目录,按 prefer 返回。

        Args:
            prefer: "stream" → 选 num_patches 最小的那一份
                    "full"   → 选 num_patches 最大的那一份
        """
        dirs = sorted(work_dir.glob("AudioVAE_Decoder_np*"))
        if not dirs:
            return None, None
        metas = []
        for d in dirs:
            meta_files = list(d.glob("audiovae_decoder_np*_meta_info.json"))
            if not meta_files:
                continue
            meta = load_meta(meta_files[0])
            metas.append(meta)
        if not metas:
            return None, None
        if prefer == "stream":
            chosen = min(metas, key=lambda m: m["num_patches"])
        else:  # full
            chosen = max(metas, key=lambda m: m["num_patches"])
        return chosen, chosen["hmonnx_file"]

    def _find_stateful_stream_decoder(self, work_dir: Path) -> Tuple[Optional[dict], Optional[str]]:
        """查找真流式 AudioVAE decoder 元信息。"""
        dirs = sorted(work_dir.glob("AudioVAE_Decoder_StreamState_np*"))
        metas = []
        for d in dirs:
            meta_files = list(d.glob("audiovae_decoder_streaming_stateful_np*_meta_info.json"))
            if not meta_files:
                continue
            meta = load_meta(meta_files[0])
            metas.append(meta)
        if not metas:
            return None, None
        # 当前真流式默认 np1；若后续多份并存,优先选 num_patches 最小的 step 图。
        chosen = min(metas, key=lambda m: int(m.get("num_patches", 999999)))
        return chosen, chosen["hmonnx_file"]

    def _find_encoder(self, work_dir: Path) -> Tuple[Optional[dict], Optional[str]]:
        """在 work_dir 下查找 AudioVAE encoder 元信息。

        兼容两种导出目录命名:
        1) 旧版: AudioVAE_Encoder/audiovae_encoder_meta_info.json
        2) 新版: AudioVAE_Encoder_np*/audiovae_encoder_meta_info.json

        返回:
            (meta, hmonnx_file)；找不到则返回 (None, None)
        """
        legacy_meta = work_dir / "AudioVAE_Encoder" / "audiovae_encoder_meta_info.json"
        if legacy_meta.exists():
            meta = load_meta(legacy_meta)
            return meta, meta["hmonnx_file"]

        dirs = sorted(work_dir.glob("AudioVAE_Encoder_np*"))
        metas = []
        for d in dirs:
            meta_path = d / "audiovae_encoder_meta_info.json"
            if not meta_path.exists():
                continue
            meta = load_meta(meta_path)
            metas.append(meta)
        if not metas:
            return None, None

        # 多份 encoder 并存时,优先选择 num_patches 最大的一份。
        chosen = max(metas, key=lambda m: int(m.get("num_patches", 0)))
        return chosen, chosen["hmonnx_file"]

    def _load_torch_audio_vae(self):
        """按需加载 PyTorch AudioVAE,用于高保真 prompt/reference 编码。"""
        if self._torch_audio_vae is not None:
            return self._torch_audio_vae
        if self._torch_audio_vae_load_error is not None:
            raise RuntimeError(str(self._torch_audio_vae_load_error))

        try:
            from voxcpm.modules.audiovae import AudioVAEConfigV2, AudioVAEV2

            if self.torch_audio_model_dir is None:
                raise FileNotFoundError(
                    "Cannot resolve torch_audio_model_dir for AudioVAE loading."
                )
            model_dir = self.torch_audio_model_dir
            if not model_dir.exists():
                raise FileNotFoundError(f"torch_audio_model_dir not found: {model_dir}")

            cfg_obj = AudioVAEConfigV2.model_validate(self.hf_config["audio_vae_config"])
            audio_vae = AudioVAEV2(config=cfg_obj)

            vae_sd = None
            st_path = model_dir / "audiovae.safetensors"
            if st_path.exists():
                try:
                    from safetensors.torch import load_file

                    vae_sd = load_file(str(st_path), device="cpu")
                except Exception:
                    vae_sd = None

            pth_path = model_dir / "audiovae.pth"
            if vae_sd is None and pth_path.exists():
                ckpt = torch.load(str(pth_path), map_location="cpu", weights_only=True)
                vae_sd = ckpt.get("state_dict", ckpt)

            if vae_sd is None:
                raise FileNotFoundError(
                    f"AudioVAE checkpoint not found in {model_dir} "
                    "(expect audiovae.safetensors or audiovae.pth)."
                )

            audio_vae.load_state_dict(vae_sd, strict=True)
            self._torch_audio_vae = audio_vae.to(self.device, dtype=torch.float32).eval()
            return self._torch_audio_vae
        except Exception as exc:
            self._torch_audio_vae_load_error = exc
            raise

    @torch.inference_mode()
    def _encode_wav_torch(self, wav_path: str, padding_mode: str = "right") -> torch.Tensor:
        """使用 PyTorch AudioVAE 编码 wav,输出 [T, P, D]。"""
        audio_vae = self._load_torch_audio_vae()
        audio, _ = librosa.load(wav_path, sr=self.vae_encode_sr, mono=True)
        audio = torch.from_numpy(audio).unsqueeze(0)  # [1, L]
        patch_len = self.patch_size * self.chunk_size
        if audio.size(1) % patch_len != 0:
            padding_size = patch_len - audio.size(1) % patch_len
            pad = (padding_size, 0) if padding_mode == "left" else (0, padding_size)
            audio = F.pad(audio, pad)
        mu = audio_vae.encode(audio.to(self.device, dtype=torch.float32), self.vae_encode_sr)
        real_T = audio.shape[-1] // self.chunk_size
        mu = mu[..., :real_T]
        feat = mu.view(self.latent_dim, -1, self.patch_size).permute(1, 2, 0)
        return feat.cpu()

    # ------------------------------------------------------------------ #
    # VAE encode(host 侧包装:复刻 VoxCPMModel._encode_wav)
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def _encode_wav(self, wav_path: str, padding_mode: str = "right") -> torch.Tensor:
        """把 wav 编码成 [T, P, D] 的 latent patches(和 VoxCPM2Model 签名一致)。"""
        if self.audio_encoder_backend in {"auto", "torch"}:
            try:
                return self._encode_wav_torch(wav_path, padding_mode=padding_mode)
            except Exception as exc:
                if self.audio_encoder_backend == "torch":
                    raise RuntimeError(f"torch audio encoder failed: {exc}") from exc
                warnings.warn(
                    f"torch audio encoder unavailable, fallback to hmonnx encoder. reason: {exc}",
                    RuntimeWarning,
                )

        if self.vae_encoder is None:
            raise FileNotFoundError(
                "AudioVAE_Encoder export is missing in work_dir, "
                "prompt/reference wav modes require exporting AudioVAE encoder first."
            )
        audio, _ = librosa.load(wav_path, sr=self.vae_encode_sr, mono=True)
        audio = torch.from_numpy(audio).unsqueeze(0)  # [1, L]

        # 按 patch_size * chunk_size 对齐
        patch_len = self.patch_size * self.chunk_size
        if audio.size(1) % patch_len != 0:
            padding_size = patch_len - audio.size(1) % patch_len
            pad = (padding_size, 0) if padding_mode == "left" else (0, padding_size)
            audio = F.pad(audio, pad)

        mu, true_T = self.vae_encoder(audio)
        # mu shape [1, latent_dim, T_fixed];真实有效 T_fixed 区间 = true_T
        # 但由于我们已经把 audio pad 成了 patch_len 倍数,实际有效 latent 长度 = audio.shape[-1] / chunk_size
        real_T = audio.shape[-1] // self.chunk_size
        mu = mu[..., :real_T]  # [1, latent_dim, real_T]

        # reshape 成 [T=real_T/patch_size, P=patch_size, D=latent_dim]
        feat = mu.view(self.latent_dim, -1, self.patch_size).permute(1, 2, 0)
        return feat.cpu()

    # ------------------------------------------------------------------ #
    # Prompt 前缀构造(和 VoxCPM2Model._make_ref_prefix 对齐)
    # ------------------------------------------------------------------ #

    def _make_ref_prefix(self, ref_feat: torch.Tensor, device):
        ref_len = ref_feat.size(0)
        z1 = torch.zeros((1, self.patch_size, self.latent_dim), dtype=torch.float32, device=device)
        tokens = torch.cat([
            torch.tensor([self.ref_audio_start_token], dtype=torch.int32, device=device),
            torch.zeros(ref_len, dtype=torch.int32, device=device),
            torch.tensor([self.ref_audio_end_token], dtype=torch.int32, device=device),
        ])
        feats = torch.cat([z1, ref_feat.to(device), z1], dim=0)
        t_mask = torch.cat([
            torch.tensor([1], dtype=torch.int32),
            torch.zeros(ref_len, dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
        ]).to(device)
        a_mask = torch.cat([
            torch.tensor([0], dtype=torch.int32),
            torch.ones(ref_len, dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
        ]).to(device)
        return tokens, feats, t_mask, a_mask

    # ------------------------------------------------------------------ #
    # 组装 prefill 输入(和 VoxCPM2Model._generate 里四种模式对齐)
    # ------------------------------------------------------------------ #

    def _assemble_prefill_inputs(
        self,
        target_text: str,
        prompt_text: str = "",
        prompt_wav_path: str = "",
        reference_wav_path: str = "",
    ):
        """返回 (text_token, audio_feat, text_mask, audio_mask) 四个 tensor,
        shape 分别为 [1, L] / [1, L, P, D] / [1, L] / [1, L]。
        """
        device = self.device

        if reference_wav_path and prompt_wav_path:
            # combined
            text = prompt_text + target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat([
                text_token,
                torch.tensor([self.audio_start_token], dtype=torch.int64),
            ])
            text_length = text_token.shape[0]

            ref_feat = self._encode_wav(reference_wav_path, padding_mode="right")
            prompt_feat = self._encode_wav(prompt_wav_path, padding_mode="left")
            prompt_audio_length = prompt_feat.size(0)

            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_feat, device)

            prompt_pad_token = torch.zeros(prompt_audio_length, dtype=torch.int32, device=device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.latent_dim), dtype=torch.float32, device=device,
            )
            text_token = torch.cat([ref_tokens, text_token.to(device), prompt_pad_token])
            audio_feat = torch.cat([ref_feats, text_pad_feat, prompt_feat.to(device)], dim=0)
            text_mask = torch.cat([
                ref_t_mask,
                torch.ones(text_length, dtype=torch.int32, device=device),
                torch.zeros(prompt_audio_length, dtype=torch.int32, device=device),
            ])
            audio_mask = torch.cat([
                ref_a_mask,
                torch.zeros(text_length, dtype=torch.int32, device=device),
                torch.ones(prompt_audio_length, dtype=torch.int32, device=device),
            ])

        elif reference_wav_path:
            text = target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat([text_token, torch.tensor([self.audio_start_token], dtype=torch.int64)])
            text_length = text_token.shape[0]

            ref_feat = self._encode_wav(reference_wav_path, padding_mode="right")
            ref_tokens, ref_feats, ref_t_mask, ref_a_mask = self._make_ref_prefix(ref_feat, device)

            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.latent_dim), dtype=torch.float32, device=device,
            )
            text_token = torch.cat([ref_tokens, text_token.to(device)])
            audio_feat = torch.cat([ref_feats, text_pad_feat], dim=0)
            text_mask = torch.cat([ref_t_mask, torch.ones(text_length, dtype=torch.int32, device=device)])
            audio_mask = torch.cat([ref_a_mask, torch.zeros(text_length, dtype=torch.int32, device=device)])

        elif prompt_wav_path:
            # continuation-only
            text = prompt_text + target_text
            text_token = torch.LongTensor(self.text_tokenizer(text))
            text_token = torch.cat([text_token, torch.tensor([self.audio_start_token], dtype=torch.int64)])
            text_length = text_token.shape[0]

            prompt_feat = self._encode_wav(prompt_wav_path, padding_mode="left")
            prompt_audio_length = prompt_feat.size(0)
            prompt_pad_token = torch.zeros(prompt_audio_length, dtype=torch.int32, device=device)
            text_pad_feat = torch.zeros(
                (text_length, self.patch_size, self.latent_dim), dtype=torch.float32, device=device,
            )
            text_token = torch.cat([text_token.to(device), prompt_pad_token])
            audio_feat = torch.cat([text_pad_feat, prompt_feat.to(device)], dim=0)
            text_mask = torch.cat([
                torch.ones(text_length, dtype=torch.int32, device=device),
                torch.zeros(prompt_audio_length, dtype=torch.int32, device=device),
            ])
            audio_mask = torch.cat([
                torch.zeros(text_length, dtype=torch.int32, device=device),
                torch.ones(prompt_audio_length, dtype=torch.int32, device=device),
            ])
        else:
            # zero-shot
            text_token = torch.LongTensor(self.text_tokenizer(target_text))
            text_token = torch.cat([text_token, torch.tensor([self.audio_start_token], dtype=torch.int64)])
            text_length = text_token.shape[0]
            audio_feat = torch.zeros(
                (text_length, self.patch_size, self.latent_dim), dtype=torch.float32, device=device,
            )
            text_token = text_token.to(device)
            text_mask = torch.ones(text_length, dtype=torch.int32, device=device)
            audio_mask = torch.zeros(text_length, dtype=torch.int32, device=device)

        # 加 batch 维
        text_token = text_token.unsqueeze(0)
        audio_feat = audio_feat.unsqueeze(0).to(self.input_dtype)
        text_mask = text_mask.unsqueeze(0)
        audio_mask = audio_mask.unsqueeze(0)
        return text_token, audio_feat, text_mask, audio_mask

    # ------------------------------------------------------------------ #
    # combined_embed 构造
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def _build_combined_embed(
        self,
        text_token: torch.Tensor,
        audio_feat: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据 mask 混合 text 和 audio embeddings。

        Returns:
            combined_embed:  [1, L, H]  —— base_lm 输入
            feat_embed:      [1, L, H]  —— 供 residual_lm 输入 concat 用
        """
        # text embed
        text_embed = self.host.token_embedding(text_token) * self.scale_emb
        text_embed = text_embed.to(self.input_dtype)

        # audio embed: feat_encoder + enc_to_lm_proj,host 侧循环 LocEnc
        L = audio_feat.shape[1]
        # LocEnc-Step 的 batch=1, T=1,host 这里按 token 位置循环(即使某位置是全 0
        # 的 text 位,也过一次,简单起见;后面通过 audio_mask 屏蔽)
        feat_embed = self.locenc.encode_sequence(audio_feat)  # [1, L, H_enc] or [1, L, H_lm]
        feat_embed = feat_embed.to(self.input_dtype)
        enc_in = self.host.enc_to_lm_proj.in_features
        enc_out = self.host.enc_to_lm_proj.out_features
        if feat_embed.shape[-1] == enc_in:
            feat_embed = self.host.enc_to_lm_proj(feat_embed)
        elif feat_embed.shape[-1] == enc_out:
            # LocEnc 导出图已融合 enc_to_lm_proj（当前 voxcpm2 导出脚本默认行为）
            pass
        else:
            raise RuntimeError(
                f"Unexpected LocEnc hidden dim {feat_embed.shape[-1]} "
                f"(expect {enc_in} or {enc_out})."
            )

        # 混合
        combined = text_mask.unsqueeze(-1) * text_embed + audio_mask.unsqueeze(-1) * feat_embed
        return combined, feat_embed

    # ------------------------------------------------------------------ #
    # prefill(pad 到 prefill_length,调 base/residual prefill session)
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def _run_prefill(
        self,
        combined_embed: torch.Tensor,
        feat_embed: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """跑 prefill。返回 (lm_hidden_last, residual_hidden_last, valid_length)。"""
        L = combined_embed.shape[1]
        if L > self.prefill_length:
            raise RuntimeError(
                f"Prompt 长度 {L} 超过 prefill 图容量 {self.prefill_length},"
                f"请重新导出更大 --prefill_length 的 LM。"
            )
        valid = L
        # pad
        if L < self.prefill_length:
            pad = self.prefill_length - L
            combined_embed = F.pad(combined_embed, (0, 0, 0, pad))
            feat_embed = F.pad(feat_embed, (0, 0, 0, pad))
            text_mask = F.pad(text_mask, (0, pad))
            audio_mask = F.pad(audio_mask, (0, pad))

        # reset cache 并 prefill
        self.base_prefill.reset_kv()
        self.residual_prefill.reset_kv()

        enc_outputs = self.base_prefill(combined_embed, current_input_length=valid)  # [1, N, H]
        enc_outputs = enc_outputs.to(self.input_dtype)

        # fsq 仅应用在 audio 位,text 位保留原值
        enc_fsq = self.host.fsq_layer(enc_outputs)
        enc_outputs = (
            enc_fsq * audio_mask.unsqueeze(-1) + enc_outputs * text_mask.unsqueeze(-1)
        )

        residual_inputs = self.host.fusion_concat_proj(
            torch.cat((enc_outputs, audio_mask.unsqueeze(-1) * feat_embed), dim=-1)
        )
        residual_outputs = self.residual_prefill(residual_inputs, current_input_length=valid)  # [1, N, H]
        residual_outputs = residual_outputs.to(self.input_dtype)

        # 取有效区间的最后一个 token
        lm_hidden = enc_outputs[:, valid - 1, :]       # [1, H]
        residual_hidden = residual_outputs[:, valid - 1, :]  # [1, H]
        return lm_hidden, residual_hidden, valid

    # ------------------------------------------------------------------ #
    # decode step
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def _run_decode_step(
        self,
        curr_embed: torch.Tensor,   # [1, 1, H]  — 来自上一步 LocEnc
        past_seq_length_base: int,
        past_seq_length_residual: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """单步 decode,返回 (lm_hidden_fsqed [1,H], residual_hidden [1,H])。"""
        lm_hidden = self.base_decode(curr_embed, past_seq_length_base)  # [1, 1, H]
        lm_hidden = lm_hidden[:, 0, :].to(self.input_dtype)  # [1, H]
        lm_hidden_fsqed = self.host.fsq_layer(lm_hidden)

        curr_residual_input = self.host.fusion_concat_proj(
            torch.cat((lm_hidden_fsqed, curr_embed[:, 0, :]), dim=-1)
        ).unsqueeze(1)  # [1, 1, H]

        residual_hidden = self.residual_decode(curr_residual_input, past_seq_length_residual)
        residual_hidden = residual_hidden[:, 0, :].to(self.input_dtype)  # [1, H]
        return lm_hidden_fsqed, residual_hidden

    # ------------------------------------------------------------------ #
    # stop 判定
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def _stop_flag(self, lm_hidden: torch.Tensor) -> int:
        logits = self.host.stop_head(self.host.stop_actn(self.host.stop_proj(lm_hidden)))
        return int(logits.argmax(dim=-1)[0].cpu().item())

    # ------------------------------------------------------------------ #
    # 主推理循环
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def _inference_core(
        self,
        combined_embed: torch.Tensor,
        feat_embed: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
        original_audio_feat: torch.Tensor,
        min_len: int,
        max_len: int,
        inference_timesteps: int,
        cfg_value: float,
        streaming: bool,
        streaming_prefix_len: int,
    ) -> Generator[Tuple[torch.Tensor, List[torch.Tensor], int], None, None]:
        """核心推理循环(对应 VoxCPM2Model._inference)。"""
        # --- 1. prefill ---
        lm_hidden, residual_hidden, valid_len = self._run_prefill(
            combined_embed, feat_embed, text_mask, audio_mask,
        )
        # prefill 虽然是固定长度图,但 KV 只写入有效 token 区间 [0, valid_len)。
        # decode 必须从 valid_len 起步,与 PyTorch kv_cache.step() 对齐。
        past_seq_len_base = valid_len
        past_seq_len_residual = valid_len

        # prefix_feat_cond 用 original audio_feat 最后一位(对应 VoxCPM2._inference 里 feat[:,-1,...])
        prefix_feat_cond = original_audio_feat[:, -1, ...]  # [1, P, D]

        # 如果 audio_mask 末尾为 1(continuation 场景),pred_feat_seq 从已有音频
        # 拿 streaming_prefix_len-1 个作为起点,这样 streaming 拼接时有上下文
        pred_feat_seq: List[torch.Tensor] = []
        has_continuation = bool(audio_mask[0, valid_len - 1].item() == 1)
        context_len = 0
        if has_continuation:
            audio_indices = audio_mask[0, :valid_len].nonzero(as_tuple=True)[0]
            context_len = min(streaming_prefix_len - 1, int(len(audio_indices)))
            if context_len > 0:
                last_indices = audio_indices[-context_len:]
                ctx = original_audio_feat[:, last_indices, :, :]  # [1, ctx, P, D]
                pred_feat_seq = list(ctx.split(1, dim=1))

        # --- 2. decode loop ---
        from tqdm import tqdm
        for i in tqdm(range(max_len), desc="VoxCPM2 decode"):
            # diffusion 预测一个 patch
            dit_hidden_1 = self.host.lm_to_dit_proj(lm_hidden)          # [1, H_dit]
            dit_hidden_2 = self.host.res_to_dit_proj(residual_hidden)   # [1, H_dit]
            mu = torch.cat((dit_hidden_1, dit_hidden_2), dim=-1)         # [1, 2*H_dit]

            pred_feat = self.solver(
                mu=mu.to(self.input_dtype),
                cond=prefix_feat_cond.transpose(1, 2).contiguous().to(self.input_dtype),
                n_timesteps=inference_timesteps,
                cfg_value=cfg_value,
            ).transpose(1, 2)  # [1, P, D]

            # 过 LocEnc 得到 curr_embed
            curr_embed = self.locenc(pred_feat.unsqueeze(1))  # [1, H_lm]
            # 注意:LocEnc-Step 图里已经 fuse 了 enc_to_lm_proj,返回的是 H_lm 维
            curr_embed = curr_embed.unsqueeze(1)  # [1, 1, H_lm]

            pred_feat_seq.append(pred_feat.unsqueeze(1))  # [1, 1, P, D]
            prefix_feat_cond = pred_feat                   # [1, P, D]

            if streaming:
                chunk = torch.cat(pred_feat_seq[-streaming_prefix_len:], dim=1)  # [1, K, P, D]
                feat_pred = chunk.permute(0, 3, 1, 2).reshape(
                    1, self.latent_dim, -1,
                )  # 等价 rearrange("b t p d -> b d (t p)")
                yield feat_pred, pred_feat_seq, context_len

            # stop
            stop_flag = self._stop_flag(lm_hidden)
            if i > min_len and stop_flag == 1:
                break

            # 下一步的 LM hidden
            lm_hidden_fsqed, residual_hidden = self._run_decode_step(
                curr_embed, past_seq_len_base, past_seq_len_residual,
            )
            lm_hidden = lm_hidden_fsqed
            past_seq_len_base += 1
            past_seq_len_residual += 1

        if not streaming:
            full_seq = torch.cat(pred_feat_seq, dim=1)  # [1, T_all, P, D]
            feat_pred = full_seq.permute(0, 3, 1, 2).reshape(1, self.latent_dim, -1)
            yield feat_pred, pred_feat_seq, context_len

    # ------------------------------------------------------------------ #
    # 公共 API — 对齐 VoxCPM.generate
    # ------------------------------------------------------------------ #

    def generate(self, *args, **kwargs) -> np.ndarray:
        return next(self._generate(*args, streaming=False, **kwargs))

    def generate_streaming(self, *args, **kwargs):
        return self._generate(*args, streaming=True, streaming_backend="stateful", **kwargs)

    def generate_streaming_legacy(self, *args, **kwargs):
        """旧版 overlap/crop 流式接口,使用 AudioVAE_Decoder_np*。"""
        return self._generate(*args, streaming=True, streaming_backend="overlap", **kwargs)

    @torch.inference_mode()
    def _generate(
        self,
        text: str,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        reference_wav_path: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        min_len: int = 2,
        max_len: int = 2000,
        streaming: bool = False,
        streaming_prefix_len: int = 4,
        streaming_backend: str = "stateful",
    ) -> Generator[np.ndarray, None, None]:
        """对齐 VoxCPM.generate:返回 numpy.ndarray 波形。

        - normalize、denoise 留给 pipeline 外层做
        """
        if not text or not text.strip():
            raise ValueError("target text must be a non-empty string")
        if prompt_wav_path is not None and not os.path.exists(prompt_wav_path):
            raise FileNotFoundError(prompt_wav_path)
        if reference_wav_path is not None and not os.path.exists(reference_wav_path):
            raise FileNotFoundError(reference_wav_path)
        if (prompt_wav_path is None) != (prompt_text is None):
            raise ValueError("prompt_wav_path and prompt_text must both be provided or both be None")
        streaming_backend = str(streaming_backend).lower().strip()
        if streaming_backend not in {"stateful", "overlap"}:
            raise ValueError("streaming_backend must be one of ['stateful', 'overlap']")

        target_text = text.replace("\n", " ")
        # 对齐 PyTorch VoxCPM2Model.generate 的默认策略:
        # decode 上限按 text token 长度做比例约束，避免 stop 头偶发失效时
        # 在短文本上生成过长噪声尾巴。
        target_text_length = len(self.text_tokenizer(target_text))
        max_len = min(int(target_text_length * 6.0 + 10), int(max_len))
        max_len = max(1, max_len)

        # 1. 组装 prefill 输入
        text_token, audio_feat, text_mask, audio_mask = self._assemble_prefill_inputs(
            target_text=target_text,
            prompt_text=prompt_text or "",
            prompt_wav_path=prompt_wav_path or "",
            reference_wav_path=reference_wav_path or "",
        )

        # 2. 构造 combined embed
        combined_embed, feat_embed = self._build_combined_embed(
            text_token, audio_feat, text_mask, audio_mask,
        )

        # 一个 latent patch 对应的输出音频采样点数。encoder chunk_size 是 16k
        # 输入侧 hop，decoder upscale 是 48k 输出侧 hop，二者不能混用。
        decode_patch_len = self.patch_size * int(self.vae_decoder.upscale)

        # 3. 核心推理
        inference_gen = self._inference_core(
            combined_embed=combined_embed,
            feat_embed=feat_embed,
            text_mask=text_mask,
            audio_mask=audio_mask,
            original_audio_feat=audio_feat,
            min_len=min_len,
            max_len=max_len,
            inference_timesteps=inference_timesteps,
            cfg_value=cfg_value,
            streaming=streaming,
            streaming_prefix_len=streaming_prefix_len,
        )

        if streaming:
            if streaming_backend == "stateful" and self.vae_stream_decoder is None:
                raise FileNotFoundError(
                    "AudioVAE_Decoder_StreamState_np* not found. "
                    "Run export_audiovae_decoder_streaming_stateful.py or use "
                    "generate_streaming_legacy()/streaming_backend='overlap'."
                )
            if streaming_backend == "stateful":
                self.vae_stream_decoder.reset()
            for latent_pred, pred_feat_seq, _ in inference_gen:
                if streaming_backend == "stateful":
                    newest_patch = pred_feat_seq[-1]  # [1, 1, P, D]
                    newest_latent = newest_patch.permute(0, 3, 1, 2).reshape(
                        1, self.latent_dim, -1,
                    )
                    audio = self.vae_stream_decoder(newest_latent.to(self.input_dtype))
                else:
                    # 旧版无状态 overlap/crop 流式:解最近 streaming_prefix_len 个
                    # patch,再只吐最后一个 patch 对应的音频。
                    audio = self.vae_decoder(latent_pred.to(self.input_dtype), mode="stream")
                    audio = audio[..., -decode_patch_len:]
                audio = audio.squeeze(1).cpu().numpy()
                yield audio.squeeze(0) if audio.ndim >= 2 else audio
        else:
            latent_pred, _, context_len = next(inference_gen)
            # 非 streaming 一次解码完整
            audio = self.vae_decoder(latent_pred.to(self.input_dtype), mode="auto")
            if context_len > 0:
                # 跳过 prompt 对应的部分(参考 VoxCPM2Model._generate 末尾逻辑)
                audio = audio[..., decode_patch_len * context_len:]
            audio = audio.squeeze(1).cpu().numpy()
            yield audio.squeeze(0) if audio.ndim >= 2 else audio
