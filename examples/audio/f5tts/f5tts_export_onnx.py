"""
F5-TTS DiT Backbone ONNX 导出
================================

将 DiT backbone 导出为静态形状 ONNX 模型，应用 LayerNorm 缩放
防止 FP16 溢出，并用 onnxsim 简化图（折叠 RotaryEmbedding 常量）。

用法:
    python f5tts_export_onnx.py \
        --ckpt /data01/nfs_shared/ASR_TTS/F5TTS_base/F5TTS_Base/model_1200000.safetensors \
        --vocab /data01/nfs_shared/ASR_TTS/F5TTS_base/F5TTS_Base/vocab.txt \
        --n-frames 2048 --n-text 256 \
        --out-dir work_dirs/f5tts/export_fp32
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from f5tts_common import (
    MODEL_CKPT,
    STATIC_N,
    STATIC_NT,
    VOCAB_PATH,
    load_f5tts_dit,
    replace_layernorm_with_scaled,
)


# ============================================================
# ONNX 导出包装
# ============================================================

class DiTForONNX(nn.Module):
    """暴露 DiT 干净的单步 forward（无 CFG / 无缓存）。"""

    def __init__(self, dit: nn.Module, static_n: int):
        super().__init__()
        self.dit = dit
        self.static_n = static_n

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        text: torch.Tensor,
        time: torch.Tensor,
        input_lengths: torch.Tensor,
    ) -> torch.Tensor:
        frame_ids = torch.arange(self.static_n, device=x.device, dtype=input_lengths.dtype).unsqueeze(0)
        audio_mask = frame_ids < input_lengths.unsqueeze(1)
        return self.dit(
            x=x,
            cond=cond,
            text=text,
            time=time,
            mask=audio_mask,
            drop_audio_cond=False,
            drop_text=False,
            cfg_infer=False,
            cache=False,
        )


def _patch_text_embed_for_onnx(dit_model: nn.Module, static_n: int):
    """
    修复 ONNX trace 兼容性问题。

    trace 时 x.shape[1] 返回 0-dim tensor，导致 TextEmbedding 内
    seq_len.unsqueeze(1) 维度越界。将 seq_len 强制转为 Python int。
    """
    text_embed = dit_model.text_embed
    original_forward = text_embed.forward

    def patched_forward(text, seq_len, drop_text=False):
        if torch.is_tensor(seq_len) and seq_len.ndim == 0:
            seq_len = int(seq_len.item())
        return original_forward(text, seq_len, drop_text)

    text_embed.forward = patched_forward


def _patch_ada_layernorm_for_quant(model: nn.Module):
    """
    用 reshape + gather 替代 chunk + [:, None] 模式。

    xhquant 的 FakeTensorProp 无法正确传播 chunk 产生的
    多个 Slice 操作的形状，导致 PTQ 阶段广播维度不匹配。
    reshape( B, 6, dim) + select 产生 Gather 算子，
    形状推断更稳定。
    """
    import types
    import sys as _sys
    from f5tts_common import F5TTS_SRC
    _sys.path.insert(0, str(Path(F5TTS_SRC).parent))
    from f5_tts.model.modules import AdaLayerNorm, AdaLayerNorm_Final

    def ada_forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))       # (B, 6*dim)
        emb = emb.reshape(emb.shape[0], 6, -1)  # (B, 6, dim)
        shift_msa = emb[:, 0].unsqueeze(1)      # (B, 1, dim)
        scale_msa = emb[:, 1].unsqueeze(1)
        gate_msa  = emb[:, 2]
        shift_mlp = emb[:, 3]
        scale_mlp = emb[:, 4]
        gate_mlp  = emb[:, 5]
        x = self.norm(x) * (1 + scale_msa) + shift_msa
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp

    def ada_final_forward(self, x, emb):
        emb = self.linear(self.silu(emb))       # (B, 2*dim)
        emb = emb.reshape(emb.shape[0], 2, -1)  # (B, 2, dim)
        scale = emb[:, 0].unsqueeze(1)           # (B, 1, dim)
        shift = emb[:, 1].unsqueeze(1)
        x = self.norm(x) * (1 + scale) + shift
        return x

    for module in model.modules():
        if isinstance(module, AdaLayerNorm):
            module.forward = types.MethodType(ada_forward, module)
        elif isinstance(module, AdaLayerNorm_Final):
            module.forward = types.MethodType(ada_final_forward, module)


def _expand_layernorm_affine(model: nn.Module):
    """
    将 elementwise_affine=False 的 LayerNorm 展开为 affine=True + ones/zeros。

    数学上完全等价：
      LayerNorm(x, affine=False) == LayerNorm(x, weight=1, bias=0)

    目的：让 xhquant 的 LayerNorm 融合能正确推断 normalized_shape。
    """
    for name, module in model.named_children():
        if isinstance(module, nn.LayerNorm) and not module.elementwise_affine:
            dim = module.normalized_shape[0]
            new_ln = nn.LayerNorm(dim, eps=module.eps, elementwise_affine=True)
            nn.init.ones_(new_ln.weight)
            nn.init.zeros_(new_ln.bias)
            setattr(model, name, new_ln)
        else:
            _expand_layernorm_affine(module)


# ============================================================
# ONNX IO 工具
# ============================================================

def _onnx_io_names(onnx_path: str):
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    g = model.graph
    init_names = {i.name for i in g.initializer}
    inputs = [vi.name for vi in g.input if vi.name not in init_names]
    outputs = [vi.name for vi in g.output]
    return inputs, outputs


# ============================================================
# 主流程
# ============================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", type=str, default=MODEL_CKPT)
    p.add_argument("--vocab", type=str, default=VOCAB_PATH)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--n-frames", type=int, default=STATIC_N)
    p.add_argument("--n-text", type=int, default=STATIC_NT)
    p.add_argument("--out-dir", type=str, default="work_dirs/f5tts/export_fp32")
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--ln-scale", type=float, default=32.0)
    p.add_argument("--skip-simplify", action="store_true")
    p.add_argument("--skip-verify", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    out_dir = Path(args.out_dir)
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    N, NT = args.n_frames, args.n_text

    # -- 1. 加载 DiT --
    print("[1/7] 加载 DiT 模型...")
    model = load_f5tts_dit(args.ckpt, args.vocab, device=args.device)

    # -- 2. 包装 + LayerNorm 缩放 + ONNX trace 补丁 --
    wrapper = DiTForONNX(model, static_n=N)
    wrapper.eval()
    print(f"[2/7] 修复 ONNX 兼容性（展开 LayerNorm + AdaLN patch + ScaledLN + seq_len patch）...")
    _patch_ada_layernorm_for_quant(wrapper)
    _expand_layernorm_affine(wrapper)
    replace_layernorm_with_scaled(wrapper, scale=args.ln_scale)
    _patch_text_embed_for_onnx(wrapper.dit, static_n=N)

    # -- 3. Dummy inputs --
    print(f"[3/7] 构造 dummy inputs (N={N}, NT={NT})...")
    dummy_x = torch.randn(1, N, 100, device=args.device)
    dummy_cond = torch.randn(1, N, 100, device=args.device)
    dummy_text = torch.randint(0, 2544, (1, NT), device=args.device)
    dummy_time = torch.tensor([0.5], device=args.device)
    dummy_input_lengths = torch.tensor([N], device=args.device, dtype=torch.int32)

    # -- 4. 导出 ONNX --
    onnx_path = onnx_dir / "f5tts_dit.onnx"
    print(f"[4/7] 导出 ONNX → {onnx_path} (opset={args.opset})...")
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_cond, dummy_text, dummy_time, dummy_input_lengths),
        str(onnx_path),
        opset_version=args.opset,
        input_names=["x", "cond", "text", "time", "input_lengths"],
        output_names=["velocity"],
    )
    print(f"  ONNX 大小: {onnx_path.stat().st_size / 1024**2:.1f} MiB")

    # -- 5. onnxsim 简化 --
    if not args.skip_simplify:
        print("[5/7] onnxsim 简化...")
        import onnx
        from onnxsim import simplify
        model_proto = onnx.load(str(onnx_path))
        model_sim, check = simplify(model_proto)
        if check:
            onnx.save(model_sim, str(onnx_path))
            print(f"  简化后大小: {onnx_path.stat().st_size / 1024**2:.1f} MiB")
        else:
            print("  [warn] onnxsim 验证失败，保留原始模型")
    else:
        print("[5/7] 跳过 onnxsim")

    # -- 6. 保存元信息 --
    print("[6/7] 保存 export_meta.json...")
    meta = {
        "ckpt": args.ckpt,
        "vocab": args.vocab,
        "n_frames": N,
        "n_text": NT,
        "opset": args.opset,
        "ln_scale": args.ln_scale,
        "onnx": str(onnx_path),
    }
    (onnx_dir / "export_meta.json").write_text(json.dumps(meta, indent=2))

    # -- 7. 数值验证 --
    if not args.skip_verify:
        print("[7/7] 验证 ONNX vs PyTorch...")
        import onnxruntime as ort

        with torch.no_grad():
            pt_out = wrapper(
                dummy_x.cpu(),
                dummy_cond.cpu(),
                dummy_text.cpu(),
                dummy_time.cpu(),
                dummy_input_lengths.cpu(),
            ).numpy()

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_out = sess.run(None, {
            "x": dummy_x.cpu().numpy(),
            "cond": dummy_cond.cpu().numpy(),
            "text": dummy_text.cpu().numpy().astype(np.int64),
            "time": dummy_time.cpu().numpy(),
            "input_lengths": dummy_input_lengths.cpu().numpy().astype(np.int32),
        })[0]

        pt_flat = pt_out.flatten()
        ort_flat = ort_out.flatten()
        pt_norm = np.linalg.norm(pt_flat)
        ort_norm = np.linalg.norm(ort_flat)
        max_diff = float(np.max(np.abs(pt_flat - ort_flat)))
        if pt_norm > 1e-6 and ort_norm > 1e-6:
            cos_sim = float(np.dot(pt_flat, ort_flat) / (pt_norm * ort_norm))
        else:
            cos_sim = 1.0 if max_diff < 1e-6 else 0.0
        print(f"  cos_sim  = {cos_sim:.6f}")
        print(f"  max_diff = {max_diff:.6e}")
        print(f"  pt_norm  = {pt_norm:.6f}, ort_norm = {ort_norm:.6f}")
        assert cos_sim > 0.999, f"ONNX 数值验证失败! cos_sim={cos_sim:.6f}"
        print("  ✓ ONNX 导出验证通过!")
    else:
        print("[7/7] 跳过验证")

    print(f"\n导出完成: {onnx_path}")


if __name__ == "__main__":
    main()
