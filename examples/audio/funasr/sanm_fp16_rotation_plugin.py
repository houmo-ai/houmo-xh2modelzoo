import os
import sys
from typing import Tuple

import torch
import torch.nn as nn

from funasr.models.sanm.attention import MultiHeadedAttentionSANM
from funasr.models.transformer.layer_norm import LayerNorm


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.append(_THIS_DIR)

from hadamard.hadamard_utils import random_hadamard_matrix, matmul_hadU  # noqa: E402


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-12, affine: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(dim))
            self.bias = nn.Parameter(torch.zeros(dim))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        y = x / rms
        if self.weight is not None:
            y = y * self.weight
        if self.bias is not None:
            y = y + self.bias
        return y


def _get_encoder(model) -> nn.Module:
    if hasattr(model, "model") and hasattr(model.model, "encoder"):
        return model.model.encoder
    if hasattr(model, "encoder"):
        return model.encoder
    raise ValueError("Cannot locate encoder on the provided model.")


def _patch_attention_identity_forward() -> None:
    from funasr.models.sanm import attention as sanm_attention

    if getattr(sanm_attention.MultiHeadedAttentionSANM, "_identity_linear_patched", False):
        return

    orig_forward_fsmn = sanm_attention.MultiHeadedAttentionSANM.forward_fsmn

    def forward_fsmn(self, inputs, mask, mask_shfit_chunk=None):
        x = orig_forward_fsmn(self, inputs, mask, mask_shfit_chunk)
        if hasattr(self, "identity_linear"):
            x = self.identity_linear(x)
        return x

    sanm_attention.MultiHeadedAttentionSANM.forward_fsmn = forward_fsmn
    sanm_attention.MultiHeadedAttentionSANM._identity_linear_patched = True

    # Patch export attention to carry identity_linear and apply it.
    if hasattr(sanm_attention, "MultiHeadedAttentionSANMExport"):
        export_cls = sanm_attention.MultiHeadedAttentionSANMExport
        if not getattr(export_cls, "_identity_linear_patched", False):
            orig_init = export_cls.__init__
            orig_export_fsmn = export_cls.forward_fsmn

            def __init__(self, model):
                orig_init(self, model)
                if hasattr(model, "identity_linear"):
                    self.identity_linear = model.identity_linear

            def forward_fsmn(self, inputs, mask):
                x = orig_export_fsmn(self, inputs, mask)
                if hasattr(self, "identity_linear"):
                    x = self.identity_linear(x)
                return x

            export_cls.__init__ = __init__
            export_cls.forward_fsmn = forward_fsmn
            export_cls._identity_linear_patched = True


def _ensure_identity_linear(attn: MultiHeadedAttentionSANM) -> None:
    if hasattr(attn, "identity_linear"):
        return
    n_feat = attn.linear_out.out_features
    identity = nn.Linear(n_feat, n_feat, bias=False)
    with torch.no_grad():
        weight = torch.eye(
            n_feat,
            device=attn.linear_out.weight.device,
            dtype=attn.linear_out.weight.dtype,
        )
        identity.weight.copy_(weight)
    attn.identity_linear = identity


def _to_rmsnorm(norm: nn.Module) -> RMSNorm:
    if isinstance(norm, RMSNorm):
        return norm
    if isinstance(norm, (LayerNorm, nn.LayerNorm)):
        dim = norm.normalized_shape[0]
        rms = RMSNorm(dim=dim, eps=norm.eps, affine=norm.elementwise_affine)
        if norm.elementwise_affine:
            with torch.no_grad():
                rms.weight.copy_(norm.weight)
                rms.bias.copy_(norm.bias)
        device = norm.weight.device if norm.elementwise_affine else next(norm.parameters()).device
        dtype = norm.weight.dtype if norm.elementwise_affine else next(norm.parameters()).dtype
        return rms.to(device=device, dtype=dtype)
    raise TypeError(f"Unsupported norm type: {type(norm)}")


def _build_zero_mean_projection(dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    eye = torch.eye(dim, device=device, dtype=dtype)
    ones = torch.ones(dim, dim, device=device, dtype=dtype)
    return eye - ones / dim


def _apply_output_matrix(linear: nn.Module, mat: torch.Tensor) -> None:
    with torch.no_grad():
        w = linear.weight.data
        w_new = mat.to(w.device, torch.float32) @ w.to(torch.float32)
        linear.weight.copy_(w_new.to(dtype=w.dtype))
        if linear.bias is not None:
            b = linear.bias.data
            b_new = b.to(torch.float32) @ mat.to(b.device, torch.float32)
            linear.bias.copy_(b_new.to(dtype=b.dtype))


def _apply_output_rotation(linear: nn.Module, R: torch.Tensor) -> None:
    # Right-multiply output by R -> left-multiply weight by R^T.
    with torch.no_grad():
        w = linear.weight.data
        w_new = R.t().to(w.device, torch.float32) @ w.to(torch.float32)
        linear.weight.copy_(w_new.to(dtype=w.dtype))
        if linear.bias is not None:
            b = linear.bias.data
            b_new = b.to(torch.float32).unsqueeze(0) @ R.to(b.device, torch.float32)
            linear.bias.copy_(b_new.squeeze(0).to(dtype=b.dtype))


def _apply_input_rotation(linear: nn.Module, R: torch.Tensor) -> None:
    # Compensate rotated input by right-multiplying weight with R.
    with torch.no_grad():
        w = linear.weight.data
        w_new = w.to(torch.float32) @ R.to(w.device, torch.float32)
        linear.weight.copy_(w_new.to(dtype=w.dtype))


def _fuse_rmsnorm_weight(norm: RMSNorm, linear: nn.Module) -> None:
    if not isinstance(norm, RMSNorm) or norm.weight is None:
        return
    with torch.no_grad():
        w = linear.weight.data
        w_fp32 = w.to(torch.float32)
        gamma = norm.weight.data.to(torch.float32)
        beta = norm.bias.data.to(torch.float32) if norm.bias is not None else None

        w_scaled = w_fp32 * gamma

        if beta is not None:
            if linear.bias is None:
                linear.bias = nn.Parameter(torch.zeros(w.shape[0], device=w.device, dtype=w.dtype))
            b = linear.bias.data.to(torch.float32)
            b_new = b + w_fp32 @ beta
            linear.bias.copy_(b_new.to(dtype=w.dtype))

        linear.weight.copy_(w_scaled.to(dtype=w.dtype))
        norm.weight.fill_(1.0)
        if norm.bias is not None:
            norm.bias.zero_()


def _build_hadamard(
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
    random_sign: bool,
    seed: int,
) -> torch.Tensor:
    if random_sign:
        torch.manual_seed(seed)
        R = random_hadamard_matrix(dim, device="cpu")
    else:
        eye = torch.eye(dim, dtype=torch.float32)
        R = matmul_hadU(eye)
    return R.to(device=device, dtype=dtype)


def _attach_output_linear(
    encoder: nn.Module,
    R: torch.Tensor,
    use_rt: bool,
) -> None:
    n = R.shape[0]
    out_linear = nn.Linear(n, n, bias=False)
    with torch.no_grad():
        w = torch.eye(n, device=out_linear.weight.device, dtype=out_linear.weight.dtype)
        out_linear.weight.copy_(w)
        _fuse_rmsnorm_weight(encoder.after_norm, out_linear)
        _apply_input_rotation(out_linear, R)

    if hasattr(encoder, "after_norm") and encoder.after_norm is not None:
        encoder.after_norm = nn.Sequential(encoder.after_norm, out_linear)
    else:
        encoder.output_linear = out_linear


def _scale_linear(module: nn.Module, scale: float) -> None:
    if module is None:
        return
    if not isinstance(module, nn.Linear):
        return
    with torch.no_grad():
        module.weight.mul_(scale)
        if module.bias is not None:
            module.bias.mul_(scale)


def apply_sanm_fp16_rotation(
    model,
    *,
    step_identity: bool = True,
    step_rmsnorm: bool = True,
    step_fuse: bool = True,
    step_hadamard: bool = True,
    step_scale: bool = False,
    scale: float = 0.1,
    seed: int = 0,
    random_sign: bool = True,
    output_linear_use_rt: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply the SANM encoder conversion steps before export.
    Steps can be enabled/disabled to do a staged equivalence check.
    If output_linear_use_rt=True, the appended output linear uses R^T to invert
    non-symmetric Hadamard rotations.
    step_scale applies a global scale on attention/ffn outputs to reduce FP16 overflow
    while keeping RMSNorm outputs unchanged.

    Returns (P, R) matrices on CPU for reference/debug.
    """
    encoder = _get_encoder(model)
    if not hasattr(encoder, "_fp16_rotation_steps"):
        encoder._fp16_rotation_steps = set()

    if step_identity:
        _patch_attention_identity_forward()

    layers0 = list(encoder.encoders0) if hasattr(encoder, "encoders0") else []
    layers = list(encoder.encoders) if hasattr(encoder, "encoders") else []
    all_layers = layers0 + layers
    if not all_layers:
        raise ValueError("Encoder layers not found.")

    if step_identity:
        for layer in all_layers:
            if isinstance(layer.self_attn, MultiHeadedAttentionSANM):
                _ensure_identity_linear(layer.self_attn)

    dim = all_layers[0].self_attn.linear_out.out_features
    device = all_layers[0].self_attn.linear_out.weight.device
    dtype = all_layers[0].self_attn.linear_out.weight.dtype

    P = _build_zero_mean_projection(dim, device, dtype)

    # 1) LayerNorm -> RMSNorm and apply zero-mean projection for equivalence.
    if step_rmsnorm and "rmsnorm" not in encoder._fp16_rotation_steps:
        for layer in layers0:
            layer.norm2 = _to_rmsnorm(layer.norm2)
            if hasattr(layer.self_attn, "identity_linear"):
                _apply_output_matrix(layer.self_attn.identity_linear, P)
            if hasattr(layer.self_attn, "linear_out"):
                _apply_output_matrix(layer.self_attn.linear_out, P)
            if hasattr(layer.feed_forward, "w_2"):
                _apply_output_matrix(layer.feed_forward.w_2, P)

        for layer in layers:
            layer.norm1 = _to_rmsnorm(layer.norm1)
            layer.norm2 = _to_rmsnorm(layer.norm2)
            if hasattr(layer.self_attn, "identity_linear"):
                _apply_output_matrix(layer.self_attn.identity_linear, P)
            if hasattr(layer.self_attn, "linear_out"):
                _apply_output_matrix(layer.self_attn.linear_out, P)
            if hasattr(layer.feed_forward, "w_2"):
                _apply_output_matrix(layer.feed_forward.w_2, P)

        if hasattr(encoder, "after_norm") and encoder.after_norm is not None:
            encoder.after_norm = _to_rmsnorm(encoder.after_norm)
        encoder._fp16_rotation_steps.add("rmsnorm")

    # 2) Fuse RMSNorm weights into linears.
    if step_fuse and "fuse" not in encoder._fp16_rotation_steps:
        for layer in layers:
            _fuse_rmsnorm_weight(layer.norm1, layer.self_attn.linear_q_k_v)
        for layer in all_layers:
            _fuse_rmsnorm_weight(layer.norm2, layer.feed_forward.w_1)
        encoder._fp16_rotation_steps.add("fuse")

    # 3) Hadamard rotation.
    R = _build_hadamard(dim, device, dtype, random_sign=random_sign, seed=seed)

    if step_hadamard and "hadamard" not in encoder._fp16_rotation_steps:
        for layer in layers0:
            if hasattr(layer.self_attn, "identity_linear"):
                _apply_output_rotation(layer.self_attn.identity_linear, R)
            if hasattr(layer.self_attn, "linear_out"):
                _apply_output_rotation(layer.self_attn.linear_out, R)
            if hasattr(layer.feed_forward, "w_1"):
                _apply_input_rotation(layer.feed_forward.w_1, R)
            if hasattr(layer.feed_forward, "w_2"):
                _apply_output_rotation(layer.feed_forward.w_2, R)

        for layer in layers:
            if hasattr(layer.self_attn, "linear_q_k_v"):
                _apply_input_rotation(layer.self_attn.linear_q_k_v, R)
            if hasattr(layer.self_attn, "identity_linear"):
                _apply_output_rotation(layer.self_attn.identity_linear, R)
            if hasattr(layer.self_attn, "linear_out"):
                _apply_output_rotation(layer.self_attn.linear_out, R)
            if hasattr(layer.feed_forward, "w_1"):
                _apply_input_rotation(layer.feed_forward.w_1, R)
            if hasattr(layer.feed_forward, "w_2"):
                _apply_output_rotation(layer.feed_forward.w_2, R)

        _attach_output_linear(encoder, R, use_rt=output_linear_use_rt)
        encoder._fp16_rotation_steps.add("hadamard")

    # 4) Global scale to reduce FP16 overflow while keeping RMSNorm outputs stable.
    if step_scale and "scale" not in encoder._fp16_rotation_steps:
        for layer in layers0:
            if hasattr(layer.self_attn, "identity_linear"):
                _scale_linear(layer.self_attn.identity_linear, scale)
            if hasattr(layer.self_attn, "linear_out"):
                _scale_linear(layer.self_attn.linear_out, scale)
            if hasattr(layer.feed_forward, "w_2"):
                _scale_linear(layer.feed_forward.w_2, scale)

        for layer in layers:
            if hasattr(layer.self_attn, "identity_linear"):
                _scale_linear(layer.self_attn.identity_linear, scale)
            if hasattr(layer.self_attn, "linear_out"):
                _scale_linear(layer.self_attn.linear_out, scale)
            if hasattr(layer.feed_forward, "w_2"):
                _scale_linear(layer.feed_forward.w_2, scale)

        encoder._fp16_rotation_steps.add("scale")

    return P.detach().cpu(), R.detach().cpu()


def encoder_output_diff(
    model_before,
    model_after,
    feats: torch.Tensor,
    feats_len: torch.Tensor,
) -> torch.Tensor:
    """
    Utility to measure encoder output max-abs diff for sanity checks.
    """
    enc_before = _get_encoder(model_before)
    enc_after = _get_encoder(model_after)
    with torch.no_grad():
        out_before, _, _ = enc_before(feats, feats_len)
        out_after, _, _ = enc_after(feats, feats_len)
    return (out_before - out_after).abs().max()


def get_encoder_output(
    model,
    feats: torch.Tensor,
    feats_len: torch.Tensor,
) -> torch.Tensor:
    """
    Convenience helper to fetch encoder output for stepwise checks.
    """
    enc = _get_encoder(model)
    enc.eval()
    with torch.no_grad():
        out, _, _ = enc(feats, feats_len)
    return out
