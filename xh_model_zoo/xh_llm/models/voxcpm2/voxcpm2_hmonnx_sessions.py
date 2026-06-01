"""VoxCPM2 各 HMONNX 子图的 session 封装。

每个 session 负责一张图,提供最小化的 __call__ 接口。之所以单独抽一层:
- 不同子图的输入名、shape 约定差别大,集中在这里管理便于测试
- pipeline orchestrator 只需要关心 "给张量→拿张量",不用管底层 HMONNX 细节
- 单测可以脱离 pipeline 直接测单张图

Session 清单:
    BaseLMPrefillSession       / BaseLMDecodeSession
    ResidualLMPrefillSession   / ResidualLMDecodeSession
    LocEncStepSession
    LocDiTStepSession
    AudioVAEEncoderSession     (单一定长 chunk)
    AudioVAEDecoderSession     (同时加载 stream 和 full 两张图,按调用时的
                                输入形状自动选择)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from xhquant.api import HMONNXGoldenInference
from xhquant.core import CacheTensor


# ---------------------------------------------------------------------------
# LM sessions
# ---------------------------------------------------------------------------

class _LMSessionBase:
    """LM 图的共同封装:加载 HMONNX、维护 KV cache buffer。"""

    def __init__(
        self,
        onnx_path: str,
        num_hidden_layers: int,
        num_key_value_heads: int,
        cache_length: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.session = HMONNXGoldenInference(onnx_path)
        self.session.to(device)
        self.device = device
        self.dtype = dtype
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.cache_length = cache_length
        self.head_dim = head_dim

        self.past_k_caches: List[torch.Tensor] = []
        self.past_v_caches: List[torch.Tensor] = []
        self._allocate_kv_buffers()

    def _allocate_kv_buffers(self):
        shape = (1, self.num_key_value_heads, self.cache_length, self.head_dim)
        self.past_k_caches = [
            CacheTensor(torch.zeros(shape, dtype=self.dtype)).to(self.device)
            for _ in range(self.num_hidden_layers)
        ]
        self.past_v_caches = [
            CacheTensor(torch.zeros(shape, dtype=self.dtype)).to(self.device)
            for _ in range(self.num_hidden_layers)
        ]

    def reset_kv(self):
        """清空 KV cache,用于切换 prompt。"""
        for k, v in zip(self.past_k_caches, self.past_v_caches):
            k.zero_() if hasattr(k, "zero_") else k.data.zero_()
            v.zero_() if hasattr(v, "zero_") else v.data.zero_()

    def _pack_inputs(
        self, inputs_embeds: torch.Tensor, past_seq_length: int, current_input_length: int,
    ) -> list:
        inputs_embeds = inputs_embeds.to(device=self.device, dtype=self.dtype)
        past = torch.tensor([past_seq_length], dtype=torch.int32, device=self.device)
        # 与导出图签名对齐: current_input_length 期望 shape=[1,1]
        cur = torch.tensor([[current_input_length]], dtype=torch.int32, device=self.device)
        return [inputs_embeds, past, cur, *self.past_k_caches, *self.past_v_caches]


class BaseLMPrefillSession(_LMSessionBase):
    """base_lm prefill 图:输入定长 N 的 inputs_embeds,输出 [1, N, H]。"""

    def __init__(self, onnx_path, prefill_length, **kwargs):
        super().__init__(onnx_path, **kwargs)
        self.prefill_length = prefill_length

    def __call__(self, inputs_embeds: torch.Tensor, current_input_length: Optional[int] = None) -> torch.Tensor:
        """
        Args:
            inputs_embeds: [1, N=prefill_length, H] (caller 已做 pad 到定长)
            current_input_length: 实际有效 token 数(<=prefill_length)。为空时默认全长。
        Returns:
            hidden: [1, N, H]
        """
        assert inputs_embeds.shape[1] == self.prefill_length
        cur = self.prefill_length if current_input_length is None else int(current_input_length)
        if cur <= 0 or cur > self.prefill_length:
            raise ValueError(
                f"current_input_length must be in [1, {self.prefill_length}], but got {cur}."
            )
        inputs = self._pack_inputs(inputs_embeds, 0, cur)
        out = self.session(*inputs)
        return _ensure_tensor(out)


class BaseLMDecodeSession(_LMSessionBase):
    """base_lm decode 图:单 token 输入,输出 [1, 1, H]。"""

    def __call__(self, curr_embed: torch.Tensor, past_seq_length: int) -> torch.Tensor:
        """
        Args:
            curr_embed:      [1, 1, H]
            past_seq_length: 已写入 cache 的 token 数
        Returns:
            hidden: [1, 1, H]
        """
        if curr_embed.dim() == 2:
            curr_embed = curr_embed.unsqueeze(1)
        assert curr_embed.shape[1] == 1
        inputs = self._pack_inputs(curr_embed, past_seq_length, 1)
        out = self.session(*inputs)
        return _ensure_tensor(out)


class ResidualLMPrefillSession(_LMSessionBase):
    """residual_lm prefill,签名和 BaseLMPrefillSession 一致。"""

    def __init__(self, onnx_path, prefill_length, **kwargs):
        super().__init__(onnx_path, **kwargs)
        self.prefill_length = prefill_length

    def __call__(self, inputs_embeds: torch.Tensor, current_input_length: Optional[int] = None) -> torch.Tensor:
        assert inputs_embeds.shape[1] == self.prefill_length
        cur = self.prefill_length if current_input_length is None else int(current_input_length)
        if cur <= 0 or cur > self.prefill_length:
            raise ValueError(
                f"current_input_length must be in [1, {self.prefill_length}], but got {cur}."
            )
        inputs = self._pack_inputs(inputs_embeds, 0, cur)
        out = self.session(*inputs)
        return _ensure_tensor(out)


class ResidualLMDecodeSession(_LMSessionBase):
    """residual_lm decode,签名和 BaseLMDecodeSession 一致。"""

    def __call__(self, curr_embed: torch.Tensor, past_seq_length: int) -> torch.Tensor:
        if curr_embed.dim() == 2:
            curr_embed = curr_embed.unsqueeze(1)
        inputs = self._pack_inputs(curr_embed, past_seq_length, 1)
        out = self.session(*inputs)
        return _ensure_tensor(out)


# ---------------------------------------------------------------------------
# LocEnc / LocDiT
# ---------------------------------------------------------------------------

class LocEncStepSession:
    """LocEnc-Step session:[1, 1, P, D] → [1, H_lm]。"""

    def __init__(self, onnx_path: str, device: torch.device, dtype: torch.dtype = torch.float16):
        self.session = HMONNXGoldenInference(onnx_path)
        self.session.to(device)
        self.device = device
        self.dtype = dtype

    def __call__(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [1, 1, P, D]
        Returns:
            hidden: [1, H_lm]
        """
        feat = feat.to(device=self.device, dtype=self.dtype)
        out = self.session(feat)
        return _ensure_tensor(out)

    def encode_sequence(self, feat_seq: torch.Tensor) -> torch.Tensor:
        """便利方法:把 [1, T, P, D] 按 T 循环编码成 [1, T, H_lm]。"""
        B, T, P, D = feat_seq.shape
        assert B == 1
        outs = []
        for t in range(T):
            patch = feat_seq[:, t : t + 1, :, :]
            h = self(patch)  # [1, H_lm]
            outs.append(h.unsqueeze(1))  # [1, 1, H_lm]
        return torch.cat(outs, dim=1) if outs else torch.empty(
            (1, 0, 0), device=self.device, dtype=self.dtype
        )


class LocDiTStepSession:
    """LocDiT-Step session:CFG batch=2 的单步 estimator。"""

    def __init__(self, onnx_path: str, device: torch.device, dtype: torch.dtype = torch.float16):
        self.session = HMONNXGoldenInference(onnx_path)
        self.session.to(device)
        self.device = device
        self.dtype = dtype

    def __call__(
        self,
        x: torch.Tensor,      # [2, C, T]
        mu: torch.Tensor,     # [2, 2*H_dit]
        t: torch.Tensor,      # [2]
        cond: torch.Tensor,   # [2, C, T]
        dt: torch.Tensor,     # [2]
    ) -> torch.Tensor:
        x = x.to(device=self.device, dtype=self.dtype)
        mu = mu.to(device=self.device, dtype=self.dtype)
        t = t.to(device=self.device, dtype=self.dtype)
        cond = cond.to(device=self.device, dtype=self.dtype)
        dt = dt.to(device=self.device, dtype=self.dtype)
        out = self.session(x, mu, t, cond, dt)
        return _ensure_tensor(out)


# ---------------------------------------------------------------------------
# AudioVAE
# ---------------------------------------------------------------------------

class AudioVAEEncoderSession:
    """AudioVAE Encoder session。

    图是定长的:输入 [1, 1, L_samp],L_samp = num_patches * patch_size * chunk_size。
    host 侧把任意长度音频 pad 到 L_samp(多余截断;不够右 pad 零)。
    """

    def __init__(self, onnx_path: str, meta: dict, device: torch.device):
        self.session = HMONNXGoldenInference(onnx_path)
        self.session.to(device)
        self.device = device
        self.dtype = torch.float32
        self.num_patches = meta["num_patches"]
        self.patch_size = meta["patch_size"]
        self.chunk_size = meta["chunk_size"]
        self.latent_dim = meta["latent_dim"]
        self.L_samp = self.num_patches * self.patch_size * self.chunk_size
        self.T_latent = self.num_patches * self.patch_size

    def __call__(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [1, L] 或 [1, 1, L] 任意长度
        Returns:
            mu: [1, latent_dim, T_valid]
               这里 T_valid = ceil(L / chunk_size),对应音频实际 patch 数 * patch_size
               但由于图是定长,实际输出 shape = [1, latent_dim, T_latent];
               caller 应根据真实音频长度裁掉尾部无效 latent。
        """
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)  # [1, 1, L]
        assert audio.shape[0] == 1 and audio.shape[1] == 1

        L = audio.shape[-1]
        if L > self.L_samp:
            audio = audio[..., : self.L_samp]
            true_T = self.T_latent
        else:
            pad = self.L_samp - L
            audio = F.pad(audio, (0, pad))
            # 真实有效 patch 数
            true_T = (L + self.chunk_size - 1) // self.chunk_size

        audio = audio.to(device=self.device, dtype=self.dtype)
        try:
            mu = _ensure_tensor(self.session(audio))   # [1, latent_dim, T_latent]
        except AssertionError as e:
            msg = str(e)
            # 某些导出图在运行时会要求 fp16 输入;首次命中后记住并复用
            if "expected torch.float16" in msg and audio.dtype != torch.float16:
                self.dtype = torch.float16
                audio = audio.to(dtype=self.dtype)
                mu = _ensure_tensor(self.session(audio))
            elif "expected torch.float32" in msg and audio.dtype != torch.float32:
                self.dtype = torch.float32
                audio = audio.to(dtype=self.dtype)
                mu = _ensure_tensor(self.session(audio))
            else:
                raise
        return mu, true_T


class AudioVAEDecoderSession:
    """AudioVAE Decoder session。支持 stream 和 full 两张图,按输入 T 自动选。

    - stream 版(小 num_patches):用于 streaming 每步解码少量 latent
    - full 版(大 num_patches):用于非 streaming 一次性解码
    - 输入 T 超过 full 版容量时,host 会按 full 版容量分块串行调用并 concat
    """

    def __init__(
        self,
        stream_onnx_path: Optional[str],
        stream_meta: Optional[dict],
        full_onnx_path: Optional[str],
        full_meta: Optional[dict],
        device: torch.device,
        dtype: torch.dtype = torch.float16,
    ):
        self.device = device
        self.dtype = dtype

        self.stream_session = None
        self.full_session = None
        self.stream_T = 0
        self.full_T = 0
        self.upscale = None
        self.latent_dim = None
        self.patch_size = None
        self.sr_idx = None

        if stream_onnx_path and stream_meta:
            self.stream_session = HMONNXGoldenInference(stream_onnx_path)
            self.stream_session.to(device)
            self.stream_T = stream_meta["num_patches"] * stream_meta["patch_size"]
            self._ingest_meta(stream_meta)

        if full_onnx_path and full_meta:
            self.full_session = HMONNXGoldenInference(full_onnx_path)
            self.full_session.to(device)
            self.full_T = full_meta["num_patches"] * full_meta["patch_size"]
            self._ingest_meta(full_meta)

        if self.stream_session is None and self.full_session is None:
            raise ValueError("AudioVAEDecoderSession 至少需要一张可用图(stream 或 full)")

    def _ingest_meta(self, meta: dict):
        self.upscale = meta["upscale"]
        self.latent_dim = meta["latent_dim"]
        self.patch_size = meta["patch_size"]
        self.sr_idx = meta.get("precomputed_sr_idx", 3)

    def _call_single(self, session, T: int, z: torch.Tensor, sr_idx_tensor: torch.Tensor) -> torch.Tensor:
        """调一次定长 decoder,输入 z 必须刚好 [1, D, T]。"""
        assert z.shape[-1] == T, f"z.shape[-1]={z.shape[-1]} != expected T={T}"
        audio = _ensure_tensor(session(z, sr_idx_tensor))  # [1, 1, T * upscale]
        return audio

    def __call__(self, z: torch.Tensor, mode: str = "auto") -> torch.Tensor:
        """
        Args:
            z: [1, latent_dim, T_req]
            mode: "stream" / "full" / "auto"
                  "auto" 根据 T_req 自动选:T_req <= stream_T 用 stream;
                         > stream_T 用 full(如需分块由 self.full_T 决定)
        Returns:
            audio: [1, 1, T_req * upscale]
        """
        assert z.dim() == 3 and z.shape[0] == 1
        T_req = z.shape[-1]
        sr_idx_tensor = torch.tensor([self.sr_idx], dtype=torch.int32, device=self.device)
        z = z.to(device=self.device, dtype=self.dtype)

        if mode == "auto":
            if self.stream_session is not None and T_req <= self.stream_T:
                mode = "stream"
            elif self.full_session is not None:
                mode = "full"
            elif self.stream_session is not None:
                mode = "stream"
            else:
                raise RuntimeError("No decoder available")

        if mode == "stream":
            assert self.stream_session is not None, "stream 图未加载"
            return self._decode_with_fixed(
                self.stream_session, self.stream_T, z, sr_idx_tensor, T_req,
            )
        elif mode == "full":
            assert self.full_session is not None, "full 图未加载"
            return self._decode_with_fixed(
                self.full_session, self.full_T, z, sr_idx_tensor, T_req,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")

    def _decode_with_fixed(
        self, session, T_fixed: int, z: torch.Tensor, sr_idx_tensor: torch.Tensor, T_req: int,
    ) -> torch.Tensor:
        """定长图执行:
        - T_req <= T_fixed:右 pad 到 T_fixed,调一次,输出裁到 T_req * upscale
        - T_req > T_fixed:按 T_fixed 分块,每块单独 decode,最后 concat
                          (注意 CausalDecoder 块间有轻微边界效应)
        """
        if T_req <= T_fixed:
            pad = T_fixed - T_req
            z_padded = F.pad(z, (0, pad)) if pad > 0 else z
            audio = self._call_single(session, T_fixed, z_padded, sr_idx_tensor)
            return audio[..., : T_req * self.upscale]
        else:
            parts = []
            pos = 0
            while pos < T_req:
                end = min(pos + T_fixed, T_req)
                cur = z[..., pos:end]
                cur_T = cur.shape[-1]
                if cur_T < T_fixed:
                    cur = F.pad(cur, (0, T_fixed - cur_T))
                piece = self._call_single(session, T_fixed, cur, sr_idx_tensor)
                piece = piece[..., : cur_T * self.upscale]
                parts.append(piece)
                pos = end
            return torch.cat(parts, dim=-1)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------

def _ensure_tensor(out) -> torch.Tensor:
    """HMONNXGoldenInference 有时返回 list/tuple,统一成 tensor。"""
    if isinstance(out, (list, tuple)):
        if len(out) == 1:
            return out[0]
        # 多输出场景本 pipeline 不用到,若遇到直接报错
        raise RuntimeError(f"Unexpected multi-output from session: {len(out)} outputs")
    return out


def load_meta(meta_path: Path) -> dict:
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)
