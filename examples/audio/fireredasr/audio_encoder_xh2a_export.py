"""
FireRedASR Audio Encoder ONNX/HMONNX 导出脚本 (xh2a 适配)

将 Fbank 特征提取之后到 LLM 之前的所有网络（ConformerEncoder + Adapter）导出为 ONNX。

Mask 策略 (避免 masked_fill / -inf / bool 运算，适配 xh2a 硬件):
  - attn_mask:  [B, 1, 1, T_conv]   加法 mask，有效位置 0，无效位置 -65504。
               在 softmax 前加两次 (attn = attn + attn_mask + attn_mask)。
  - conv_mask:  [B, 1, T_conv]      乘法 mask，有效位置 1，无效位置 0。
               直接与 ConvModule 的输入/输出相乘，替代 masked_fill_ 操作。

输入:
  - fbank_features: [B, T_pad, 80]      Pad 到 30s 长度的 Fbank 特征 (T_pad=3000)
  - attn_mask:      [B, 1, 1, T_conv]   Self-Attention 的加法 mask
  - conv_mask:      [B, 1, T_conv]       ConvModule 的乘法 mask

输出:
  - speech_features: [B, T_out, 3584]    Adapter 输出的语音特征

参考:
  xhquant_llm/models/minicpmo/_tts_vocos_model_impl.py  (乘法 mask 模式)
  xhquant_llm/models/minicpmo/_tts_model_impl.py        (attention_mask 加两次模式)
"""

import argparse
import glob
import importlib.util
import os
import sys
import tempfile
import time
import types
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file

# Ensure local repo package import works when running this script directly.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# 添加 FireRedASR 路径
FIREREDASR_ROOT = REPO_ROOT / ".." / "FireRedASR"
sys.path.insert(0, str(FIREREDASR_ROOT))


# ======================== 常量定义 ========================
MAX_AUDIO_SECONDS = 30
FBANK_FRAME_SHIFT_MS = 10
FBANK_DIM = 80
CONTEXT_PAD = 6  # Conv2dSubsampling context padding: context=7, pad=context-1=6
T_FBANK_MAX = MAX_AUDIO_SECONDS * 1000 // FBANK_FRAME_SHIFT_MS  # 3000
ATTN_MASK_PAD_VALUE = -65504.0  # fp16 最大值的负数，替代 -inf


# ======================== 长度计算 ========================

def compute_conv_valid_length(input_length: int) -> int:
    """计算 Conv2dSubsampling 后的有效输出帧数。

    与原始 Conv2dSubsampling 的长度公式一致：
        output_lengths = (input_lengths - 3) // 2 + 1  (两次)
    input_length: 原始 fbank 帧数（不含 context padding）。
    """
    L = (input_length - 3) // 2 + 1
    L = (L - 3) // 2 + 1
    return L


def compute_conv_total_length(T_fbank_padded: int) -> int:
    """计算 Conv2dSubsampling 输出的总 tensor 长度。

    T_fbank_padded: pad 后的 fbank 总帧数。
    forward 中会再加 context_pad=6 帧后才过 Conv2d。
    """
    L = T_fbank_padded + CONTEXT_PAD
    L = (L - 3) // 2 + 1
    L = (L - 3) // 2 + 1
    return L


def compute_adapter_output_length(conv_len: int, downsample_rate: int = 2) -> int:
    """Adapter 将相邻 downsample_rate 帧拼接，丢弃末尾不足的帧。"""
    return conv_len // downsample_rate


# ======================== Mask 构建 ========================

def create_masks_from_lengths(fbank_lengths, T_fbank_pad: int):
    """根据 fbank 有效帧数创建 attn_mask 和 conv_mask。

    Args:
        fbank_lengths: [B] 每条音频的有效 fbank 帧数
        T_fbank_pad:   pad 后的 fbank 总帧数

    Returns:
        attn_mask: [B, 1, 1, T_conv]  加法 mask (0=valid, -65504=padding)
        conv_mask: [B, 1, T_conv]     乘法 mask (1=valid, 0=padding)
    """
    if isinstance(fbank_lengths, torch.Tensor):
        lengths_list = fbank_lengths.tolist()
    else:
        lengths_list = list(fbank_lengths)
    batch_size = len(lengths_list)

    T_conv = compute_conv_total_length(T_fbank_pad)

    conv_mask = torch.zeros(batch_size, 1, T_conv, dtype=torch.float32)
    attn_mask = torch.full(
        (batch_size, 1, 1, T_conv), ATTN_MASK_PAD_VALUE, dtype=torch.float32
    )

    for i in range(batch_size):
        valid_len = compute_conv_valid_length(int(lengths_list[i]))
        conv_mask[i, :, :valid_len] = 1.0
        attn_mask[i, :, :, :valid_len] = 0.0

    return attn_mask, conv_mask


def create_dummy_inputs(batch_size: int = 1, audio_seconds: float = 10.0, device="cpu"):
    """创建用于导出和验证的 dummy 输入。"""
    actual_frames = min(
        int(audio_seconds * 1000 / FBANK_FRAME_SHIFT_MS), T_FBANK_MAX
    )

    fbank_features = torch.randn(
        batch_size, T_FBANK_MAX, FBANK_DIM, device=device
    )
    fbank_features[:, actual_frames:, :] = 0.0

    fbank_lengths = torch.full(
        (batch_size,), actual_frames, dtype=torch.long, device=device
    )
    attn_mask, conv_mask = create_masks_from_lengths(fbank_lengths, T_FBANK_MAX)
    attn_mask = attn_mask.to(device)
    conv_mask = conv_mask.to(device)

    return fbank_features, attn_mask, conv_mask


def pad_fbank_to_fixed(fbank: torch.Tensor, T_max: int) -> torch.Tensor:
    """将 fbank 特征 pad 到固定长度 T_max。"""
    B, T, D = fbank.shape
    if T >= T_max:
        return fbank[:, :T_max, :]
    padded = torch.zeros(B, T_max, D, dtype=fbank.dtype, device=fbank.device)
    padded[:, :T, :] = fbank
    return padded


# ======================== Impl 替换 ========================
# 参考: xhquant_llm/models/minicpmo/_tts_vocos_model_impl.py  (ConvNeXtBlock: x *= mask)
#       xhquant_llm/models/minicpmo/_tts_model_impl.py        (attn + mask, attn + mask)
# 消除 masked_fill / -inf / bool 运算 (.ne, .eq)，适配 xh2a 硬件


def _impl_scaled_dot_product_attention_forward(self, attn, v, mask=None):
    """替代 ScaledDotProductAttention.forward_attention

    原始实现:
        mask = mask.unsqueeze(1).eq(0)
        attn = attn.masked_fill(mask, -inf)
        attn = softmax(attn).masked_fill(mask, 0.0)

    新实现 (xh2a 友好):
        attn = attn + attn_mask + attn_mask   # -65504 * 2 = -131008 → softmax ≈ 0
        attn = softmax(attn)

    mask: [B, 1, 1, T_conv], 有效位置 0, padding 位置 -65504
    attn: [B, n_head, T_q, T_k]
    """
    if mask is not None:
        attn = attn + mask
        attn = attn + mask
    attn = torch.softmax(attn, dim=-1)
    d_attn = self.dropout(attn)
    output = torch.matmul(d_attn, v)
    return output, attn


def _impl_conformer_convolution_forward(self, x, mask=None):
    """替代 ConformerConvolution.forward

    原始实现:
        out.masked_fill_(mask.ne(1), 0.0)   # pointwise_conv1 前
        out.masked_fill_(mask.ne(1), 0.0)   # pointwise_conv2 后

    新实现 (xh2a 友好):
        out = out * mask                    # 乘法替代 masked_fill

    x:    [B, T, D]
    mask: [B, 1, T], 有效位置 1, padding 位置 0
    """
    residual = x
    out = self.pre_layer_norm(x)
    out = out.transpose(1, 2)  # [B, D, T]
    if mask is not None:
        out = out * mask  # [B, D, T] * [B, 1, T]
    out = self.pointwise_conv1(out)
    out = F.glu(out, dim=1)
    out = self.depthwise_conv(out)
    out = out.transpose(1, 2)  # [B, T, D*2]
    out = self.swish(self.batch_norm(out))
    out = out.transpose(1, 2)  # [B, D*2, T]
    out = self.dropout(self.pointwise_conv2(out))
    if mask is not None:
        out = out * mask  # [B, D, T] * [B, 1, T]
    out = out.transpose(1, 2)  # [B, T, D]
    return out + residual


def patch_encoder_for_export(encoder):
    """对 Conformer Encoder 进行 impl 替换。

    替换内容:
    1. ScaledDotProductAttention.forward_attention
       → 加法 mask (-65504) 加两次, 替代 masked_fill(-inf)
    2. ConformerConvolution.forward
       → 乘法 mask (0/1) 相乘, 替代 masked_fill_(mask.ne(1), 0.0)
    """
    for block in encoder.layer_stack:
        # Patch attention
        block.mhsa.attention.forward_attention = types.MethodType(
            _impl_scaled_dot_product_attention_forward,
            block.mhsa.attention,
        )
        # Patch conv module
        block.conv.forward = types.MethodType(
            _impl_conformer_convolution_forward,
            block.conv,
        )
    print(f"  Patched {len(encoder.layer_stack)} Conformer blocks (impl 替换)")


# ======================== 导出模型 ========================

class AudioEncoderWithAdapter(nn.Module):
    """封装 ConformerEncoder + Adapter，使用分离的 attn_mask 和 conv_mask。

    注意:
    - Conv2dSubsampling 不需要 mask。Conv2d 处理 zero-padding 区域时自然
      产生近零输出，后续 Conformer 的 attn_mask/conv_mask 会处理无效位置。
      这与原始 FireRedASR 的设计一致（Conv2dSubsampling 内部不对卷积结果
      做 mask，只在卷积后重新计算 output_lengths 并生成 mask）。
    - encoder 必须已经过 patch_encoder_for_export() 处理。
    """

    def __init__(self, encoder, adapter):
        super().__init__()
        self.encoder = encoder
        self.adapter = adapter

    def forward(self, fbank_features, attn_mask, conv_mask):
        """
        Args:
            fbank_features: [B, T_pad, 80]     pad 到固定长度的 fbank 特征
            attn_mask:      [B, 1, 1, T_conv]  注意力加法 mask (0=valid, -65504=padding)
            conv_mask:      [B, 1, T_conv]      ConvModule 乘法 mask (1=valid, 0=padding)

        Returns:
            speech_features: [B, T_out, llm_dim]
        """
        # 1. Context padding (与原始 ConformerEncoder.forward pad=True 一致)
        padded_input = F.pad(
            fbank_features,
            (0, 0, 0, self.encoder.input_preprocessor.context - 1),
            "constant",
            0.0,
        )

        # 2. Conv2dSubsampling (不需要 mask，零填充区域由后续 mask 处理)
        x = padded_input.unsqueeze(1)  # [B, 1, T+6, 80]
        for layer in self.encoder.input_preprocessor.conv:
            x = layer(x)
        N, C, T, D = x.size()
        embed_output = self.encoder.input_preprocessor.out(
            x.transpose(1, 2).contiguous().view(N, T, C * D)
        )

        # 3. Dropout + 相对位置编码
        enc_output = self.encoder.dropout(embed_output)
        pos_emb = self.encoder.dropout(
            self.encoder.positional_encoding(embed_output)
        )

        # 4. Conformer blocks
        #    attn_mask → RelPosMultiHeadAttention → ScaledDotProductAttention (加法)
        #    conv_mask → ConformerConvolution (乘法)
        for enc_layer in self.encoder.layer_stack:
            enc_output = enc_layer(
                enc_output,
                pos_emb,
                slf_attn_mask=attn_mask,
                pad_mask=conv_mask,
            )

        # 5. Adapter (拼接相邻 2 帧 + MLP, 内联实现避免 x_lens 计算)
        batch_size, seq_len, feat_dim = enc_output.size()
        num_discard = seq_len % self.adapter.ds
        if num_discard > 0:
            enc_output = enc_output[:, :-num_discard, :]
            seq_len = enc_output.size(1)
        enc_output = enc_output.contiguous().view(
            batch_size, seq_len // self.adapter.ds, feat_dim * self.adapter.ds
        )
        speech_features = self.adapter.linear1(enc_output)
        speech_features = self.adapter.relu(speech_features)
        speech_features = self.adapter.linear2(speech_features)

        return speech_features


# ======================== 加载模型 ========================

def load_fireredasr_encoder_and_adapter(model_dir: str, device: str = "cpu"):
    """加载 FireRedASR 的 encoder 和 adapter。"""
    from fireredasr.models.fireredasr import load_firered_llm_model_and_tokenizer

    model_path = str(Path(model_dir) / "model.pth.tar")
    encoder_path = str(Path(model_dir) / "asr_encoder.pth.tar")
    llm_dir = str(Path(model_dir) / "Qwen2-7B-Instruct")

    model, tokenizer = load_firered_llm_model_and_tokenizer(
        model_path, encoder_path, llm_dir
    )
    model.eval()

    encoder = model.encoder
    adapter = model.encoder_projector

    return encoder, adapter, model, tokenizer


def _infer_rotated_adapter_path(resume_from: str | None, rotated_adapter_path: str | None):
    if rotated_adapter_path is not None:
        return rotated_adapter_path
    if resume_from is None:
        return None
    inferred = Path(resume_from).parent / "audio_projector_rotated.safetensors"
    if inferred.exists():
        return str(inferred)
    return None


def _load_rotated_adapter_if_needed(adapter: nn.Module, rotated_adapter_path: str | None):
    if rotated_adapter_path is None:
        return False
    adapter_file = Path(rotated_adapter_path)
    if not adapter_file.exists():
        print(f"  WARNING: rotated_adapter_path not found: {adapter_file}")
        return False
    rotated_sd = load_safetensors_file(str(adapter_file))
    missing, unexpected = adapter.load_state_dict(rotated_sd, strict=False)
    print(
        f"  Loaded rotated audio projector: {adapter_file} "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )
    return True


def _load_llm_quant_helpers():
    try:
        from examples.audio.fireredasr.audio_llm_xh2a_common_quant import (
            _load_quantized_hf_model_from_checkpoint,
            _sync_fireredasr_llm_special_tokens,
        )
        from examples.audio.fireredasr.audio_llm_xh2a_export import (
            _resolve_hf_model_dir,
            load_fireredasr_lora_weights,
        )
        return (
            _load_quantized_hf_model_from_checkpoint,
            _sync_fireredasr_llm_special_tokens,
            _resolve_hf_model_dir,
            load_fireredasr_lora_weights,
        )
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parents[3]
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))

        common_quant_file = Path(__file__).resolve().parent / "audio_llm_xh2a_common_quant.py"
        common_spec = importlib.util.spec_from_file_location("firered_common_quant_local", common_quant_file)
        if common_spec is None or common_spec.loader is None:
            raise ImportError(f"Cannot load common quant helpers from {common_quant_file}")
        common_mod = importlib.util.module_from_spec(common_spec)
        common_spec.loader.exec_module(common_mod)

        export_file = Path(__file__).resolve().parent / "audio_llm_xh2a_export.py"
        export_spec = importlib.util.spec_from_file_location("firered_export_local", export_file)
        if export_spec is None or export_spec.loader is None:
            raise ImportError(f"Cannot load export helpers from {export_file}")
        export_mod = importlib.util.module_from_spec(export_spec)
        export_spec.loader.exec_module(export_mod)
        return (
            common_mod._load_quantized_hf_model_from_checkpoint,
            common_mod._sync_fireredasr_llm_special_tokens,
            export_mod._resolve_hf_model_dir,
            export_mod.load_fireredasr_lora_weights,
        )


def _load_quantized_llm_if_needed(full_model, args):
    if args.resume_from is None:
        return False
    resume_file = Path(args.resume_from)
    if not resume_file.exists():
        raise FileNotFoundError(f"resume_from not found: {resume_file}")

    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from xh_model_zoo.api import Config
    (
        _load_quantized_hf_model_from_checkpoint,
        _sync_fireredasr_llm_special_tokens,
        _resolve_hf_model_dir,
        load_fireredasr_lora_weights,
    ) = _load_llm_quant_helpers()

    llm_cfg = Config.fromfile(args.llm_quant_config)
    resolved_hf_model_dir = _resolve_hf_model_dir(
        cfg_hf_model_dir=str(llm_cfg.hf_model_dir),
        fireredasr_model_dir=args.model_dir,
        cli_hf_model_dir=args.hf_model_dir,
    )
    llm_cfg.hf_model_dir = resolved_hf_model_dir
    llm_cfg.model.hf_model = resolved_hf_model_dir

    lora_state_dict, lora_config, _, _, _ = load_fireredasr_lora_weights(args.model_dir)
    load_args = types.SimpleNamespace(lora_mode=args.lora_mode)
    quantized_llm = _load_quantized_hf_model_from_checkpoint(
        checkpoint_file=resume_file,
        cfg=llm_cfg,
        args=load_args,
        lora_state_dict=lora_state_dict,
        lora_config=lora_config,
    )
    quantized_llm.eval().to(torch.float16)
    class _StdoutLogger:
        @staticmethod
        def info(msg):
            print(msg)

        @staticmethod
        def warning(msg):
            print(msg)

    _sync_fireredasr_llm_special_tokens(quantized_llm, full_model.llm, logger=_StdoutLogger())
    full_model.llm = quantized_llm
    print(f"  Loaded quantized LLM checkpoint: {resume_file}")
    return True


# ======================== ONNX 导出 ========================

def export_audio_encoder_onnx(
    model: AudioEncoderWithAdapter,
    fbank_features: torch.Tensor,
    attn_mask: torch.Tensor,
    conv_mask: torch.Tensor,
    output_path: str,
):
    """导出 Audio Encoder 到 ONNX。"""
    from xhquant.utils.onnxsim_large_model.simplify_large_onnx import (
        simplify_large_onnx,
    )

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)

    model.float().eval().cpu()
    fbank_features = fbank_features.float().cpu()
    attn_mask = attn_mask.float().cpu()
    conv_mask = conv_mask.float().cpu()

    if Path(output_path).exists():
        print(f"  ONNX already exists: {output_path}, skip export")
        return onnx.load(output_path, load_external_data=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = str(Path(tmp_dir) / "audio_encoder.onnx")
        print(f"  Exporting to {onnx_file} ...")
        torch.onnx.export(
            model,
            (fbank_features, attn_mask, conv_mask),
            onnx_file,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=["fbank_features", "attn_mask", "conv_mask"],
            output_names=["speech_features"],
            verbose=False,
        )
        onnx_model = onnx.load(onnx_file, load_external_data=True)

    print("  Simplifying ONNX model ...")
    onnx_model, check = simplify_large_onnx(onnx_model)
    assert check, "ONNX simplification failed!"

    onnx.save(
        onnx_model,
        output_path,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="audio_encoder_external_data",
        convert_attribute=True,
    )
    print(f"  Saved to {output_path}")
    return onnx_model


def convert_to_hmonnx(
    onnx_path: str,
    hmonnx_path: str,
    fbank_features: torch.Tensor,
    attn_mask: torch.Tensor,
    conv_mask: torch.Tensor,
    quant_config: dict = None,
):
    """将 ONNX 转换为 HMONNX。"""
    from xhquant.api import DeviceType, convert_onnx_to_hmonnx

    Path(hmonnx_path).parent.mkdir(exist_ok=True, parents=True)
    if Path(hmonnx_path).exists():
        print(f"  HMONNX already exists: {hmonnx_path}")
        return
    if quant_config is None:
        quant_config = dict(inputs=dict())

    convert_onnx_to_hmonnx(
        onnx_path,
        [fbank_features.float().cpu(), attn_mask.float().cpu(), conv_mask.float().cpu()],
        DeviceType.XH2a,
        hmonnx_path,
        quant_config,
    )
    print(f"  Saved HMONNX to {hmonnx_path}")


def generate_hmonnx_golden(
    hmonnx_path: str,
    fbank_features: torch.Tensor,
    attn_mask: torch.Tensor,
    conv_mask: torch.Tensor,
    golden_dir: str,
    exec_device: str = "cuda",
):
    """Generate HMONNX golden for Audio Encoder by xhquant native API."""
    from xhquant.api import HMONNXGoldenInference

    golden_root = Path(golden_dir)
    golden_root.mkdir(exist_ok=True, parents=True)

    run_device = exec_device
    if run_device.startswith("cuda") and not torch.cuda.is_available():
        run_device = "cpu"

    session = HMONNXGoldenInference(hmonnx_path)
    session.save_golden = True
    session.golden_dir = str(golden_root)
    # Compatibility with some xhquant versions.
    try:
        session.save_golden_dir = str(golden_root)
    except Exception:
        pass
    try:
        session.exec_device = run_device
    except Exception:
        pass
    try:
        session.to(run_device)
    except Exception:
        pass

    with torch.no_grad():
        session.forward(
            fbank_features.half().to(run_device),
            attn_mask.half().to(run_device),
            conv_mask.half().to(run_device),
        )
    print(f"  Saved HMONNX golden to: {golden_root}")


# ======================== 推理包装器 ========================

class ONNXAudioEncoder:
    """ONNX 推理包装器。"""

    def __init__(self, onnx_path: str):
        self.session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )

    def __call__(self, fbank_features, attn_mask, conv_mask):
        out = self.session.run(
            None,
            {
                "fbank_features": fbank_features.float().cpu().numpy(),
                "attn_mask": attn_mask.float().cpu().numpy(),
                "conv_mask": conv_mask.float().cpu().numpy(),
            },
        )[0]
        return torch.from_numpy(out)


class HMONNXAudioEncoder:
    """HMONNX 推理包装器 (使用 xhquant HMONNXInference)。"""

    def __init__(self, hmonnx_path: str, device: str = "cuda"):
        from xhquant.api import HMONNXInference

        self.session = HMONNXInference(hmonnx_path)
        self.device = device
        self.session.to(device)

    def __call__(self, fbank_features, attn_mask, conv_mask):
        out = self.session(
            fbank_features.half().to(self.device),
            attn_mask.half().to(self.device),
            conv_mask.half().to(self.device),
        )
        return out.cpu()


# ======================== 验证 ========================

def validate_onnx_dummy(
    export_model: AudioEncoderWithAdapter,
    onnx_path: str,
    fbank_features: torch.Tensor,
    attn_mask: torch.Tensor,
    conv_mask: torch.Tensor,
):
    """使用 dummy 输入验证 ONNX 精度（对比 patched PyTorch 模型与 ONNX）。"""
    print("\n  [Dummy Validation] patched model vs ONNX")
    export_model.eval().float().cpu()

    with torch.no_grad():
        pt_output = export_model(
            fbank_features.float().cpu(),
            attn_mask.float().cpu(),
            conv_mask.float().cpu(),
        )

    onnx_model = ONNXAudioEncoder(onnx_path)
    onnx_output = onnx_model(fbank_features, attn_mask, conv_mask)

    diff = torch.abs(pt_output - onnx_output)
    print(f"    Output shape: {pt_output.shape}")
    print(f"    Max diff:  {diff.max().item():.6e}")
    print(f"    Mean diff: {diff.mean().item():.6e}")
    passed = diff.max().item() < 1e-3
    print(f"    Result: {'PASSED' if passed else 'FAILED'}")
    return passed


def validate_hmonnx_dummy(
    onnx_path: str,
    hmonnx_path: str,
    fbank_features: torch.Tensor,
    attn_mask: torch.Tensor,
    conv_mask: torch.Tensor,
    max_diff_threshold: float = 0.5,
):
    """使用 dummy 输入验证 HMONNX 精度（对比 ONNX 与 HMONNX）。"""
    print("\n  [Dummy Validation] ONNX vs HMONNX")

    onnx_model = ONNXAudioEncoder(onnx_path)
    onnx_output = onnx_model(fbank_features, attn_mask, conv_mask)

    hmonnx_model = HMONNXAudioEncoder(hmonnx_path)
    hmonnx_output = hmonnx_model(fbank_features.half(), attn_mask.half(), conv_mask.half())

    diff = torch.abs(onnx_output.float() - hmonnx_output.float())
    cos_sim = F.cosine_similarity(
        onnx_output.float().reshape(1, -1),
        hmonnx_output.float().reshape(1, -1),
    ).item()
    print(f"    ONNX output shape:   {onnx_output.shape}")
    print(f"    HMONNX output shape: {hmonnx_output.shape}")
    print(f"    Max diff:    {diff.max().item():.6e}")
    print(f"    Mean diff:   {diff.mean().item():.6e}")
    print(f"    Cosine sim:  {cos_sim:.6f}")
    passed = diff.max().item() < max_diff_threshold
    print(f"    Result: {'PASSED' if passed else 'FAILED'}")
    return passed


def validate_real_audio(
    model_path: str,
    orig_speech_features: torch.Tensor,
    orig_speech_lens: torch.Tensor,
    fbank_padded: torch.Tensor,
    fbank_lengths: torch.Tensor,
    uttids: list,
    backend: str = "onnx",
    max_diff_threshold: float = 0.05,
):
    """使用真实音频验证：对比原始模型（未 patch）和 ONNX/HMONNX 的 encoder 输出。

    逐条送入（batch_size=1），避免静态 batch 限制。
    """
    label = backend.upper()
    print(f"\n  [Real Audio Validation] original model vs {label}")
    if backend == "hmonnx":
        inference_model = HMONNXAudioEncoder(model_path)
    else:
        inference_model = ONNXAudioEncoder(model_path)

    all_passed = True
    for i in range(fbank_padded.shape[0]):
        valid_len = int(orig_speech_lens[i].item())
        if valid_len <= 0:
            continue

        # 逐条构建输入（batch_size=1）
        fbank_i = fbank_padded[i : i + 1]  # [1, T_FBANK_MAX, 80]
        attn_mask_i, conv_mask_i = create_masks_from_lengths(
            fbank_lengths[i : i + 1], T_FBANK_MAX
        )
        output_i = inference_model(fbank_i, attn_mask_i, conv_mask_i)

        orig = orig_speech_features[i, :valid_len].float()
        pred = output_i[0, :valid_len].float()
        diff = torch.abs(orig - pred)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        cos_sim = F.cosine_similarity(
            orig.reshape(1, -1), pred.reshape(1, -1)
        ).item()
        passed = max_diff < max_diff_threshold
        all_passed = all_passed and passed
        status = "PASS" if passed else "FAIL"
        print(
            f"    [{status}] {uttids[i]}: valid_frames={valid_len}, "
            f"max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, "
            f"cosine_sim={cos_sim:.6f}"
        )
    print(f"    Overall: {'PASSED' if all_passed else 'FAILED'}")
    return all_passed


def _make_encoder_wrapper(inference_fn, device=None):
    """创建一个 nn.Module wrapper，将 encoder+adapter 替换为外部推理函数。

    inference_fn: callable(fbank_features, attn_mask, conv_mask) -> torch.Tensor
    """

    class _ExternalEncoder(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, padded_feat, feat_lengths):
            fbank_pad = pad_fbank_to_fixed(padded_feat, T_FBANK_MAX)
            attn_m, conv_m = create_masks_from_lengths(feat_lengths, T_FBANK_MAX)
            speech_features = inference_fn(fbank_pad, attn_m, conv_m)
            if isinstance(speech_features, np.ndarray):
                speech_features = torch.from_numpy(speech_features)
            speech_features = speech_features.to(
                device=padded_feat.device, dtype=padded_feat.dtype
            )
            # 根据实际 fbank 长度计算有效的 adapter 输出帧数，
            # 截断 padding 区域，避免垃圾特征被当作有效 speech token 喂给 LLM
            enc_lengths = torch.zeros(
                padded_feat.size(0),
                dtype=torch.long,
                device=padded_feat.device,
            )
            for i in range(padded_feat.size(0)):
                conv_valid = compute_conv_valid_length(int(feat_lengths[i].item()))
                enc_lengths[i] = compute_adapter_output_length(conv_valid)
            max_valid_len = int(enc_lengths.max().item())
            if max_valid_len > 0:
                speech_features = speech_features[:, :max_valid_len, :]
            return speech_features, enc_lengths, None

    return _ExternalEncoder()


def _make_onnx_inference_fn(onnx_path: str):
    """创建 ONNX 推理函数。"""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    def _infer(fbank_features, attn_mask, conv_mask):
        result = session.run(
            None,
            {
                "fbank_features": fbank_features.float().cpu().numpy(),
                "attn_mask": attn_mask.float().cpu().numpy(),
                "conv_mask": conv_mask.float().cpu().numpy(),
            },
        )
        return torch.from_numpy(result[0])

    return _infer


def _make_hmonnx_inference_fn(hmonnx_path: str, device: str = "cuda"):
    """创建 HMONNX 推理函数。"""
    from xhquant.api import HMONNXInference

    session = HMONNXInference(hmonnx_path)
    session.to(device)

    def _infer(fbank_features, attn_mask, conv_mask):
        out = session(
            fbank_features.half().to(device),
            attn_mask.half().to(device),
            conv_mask.half().to(device),
        )
        return out.cpu()

    return _infer


def validate_asr_transcription(
    full_model,
    tokenizer,
    model_path: str,
    wav_dir: str,
    model_dir: str,
    ref_text_file: str = None,
    use_gpu: bool = False,
    backend: str = "onnx",
):
    """完整 ASR 转写对比：原始模型 vs ONNX/HMONNX 替换后的模型。

    Args:
        backend: "onnx" 或 "hmonnx"
    """
    from fireredasr.models.fireredasr import FireRedAsr
    from fireredasr.data.asr_feat import ASRFeatExtractor

    label = backend.upper()
    print(f"\n  [ASR Transcription Validation] original vs {label}")

    # 收集 wav 文件
    wav_paths = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    if not wav_paths:
        print(f"    No wav files found in {wav_dir}, skip ASR validation")
        return True
    uttids = [Path(p).stem for p in wav_paths]
    print(f"    Found {len(wav_paths)} wav files")

    # 加载参考文本
    ref_texts = {}
    if ref_text_file and Path(ref_text_file).exists():
        with open(ref_text_file) as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    ref_texts[parts[0]] = parts[1].replace(" ", "")

    # 构建 ASR pipeline (复用已加载的模型)
    cmvn_path = os.path.join(model_dir, "cmvn.ark")
    feat_extractor = ASRFeatExtractor(cmvn_path)

    asr_model = FireRedAsr.__new__(FireRedAsr)
    asr_model.asr_type = "llm"
    asr_model.feat_extractor = feat_extractor
    asr_model.model = full_model
    asr_model.tokenizer = tokenizer

    # 逐条转写（batch=1）
    print("    Running original transcription ...")
    orig_results = []
    for uid, wp in zip(uttids, wav_paths):
        res = asr_model.transcribe([uid], [wp], {"use_gpu": use_gpu})
        orig_results.extend(res)

    # 替换 encoder + adapter
    orig_encoder = full_model.encoder
    orig_projector = full_model.encoder_projector

    if backend == "hmonnx":
        inference_fn = _make_hmonnx_inference_fn(model_path)
    else:
        inference_fn = _make_onnx_inference_fn(model_path)

    ext_encoder = _make_encoder_wrapper(inference_fn)

    class _IdentityProjector(nn.Module):
        def forward(self, x, x_lens):
            return x, x_lens

    # 用 object.__setattr__ 避免 nn.Module 类型检查
    object.__setattr__(full_model, 'encoder', ext_encoder)
    object.__setattr__(full_model, 'encoder_projector', _IdentityProjector())

    print(f"    Running {label} transcription ...")
    ext_results = []
    for uid, wp in zip(uttids, wav_paths):
        res = asr_model.transcribe([uid], [wp], {"use_gpu": use_gpu})
        ext_results.extend(res)

    # 恢复原始 encoder
    object.__setattr__(full_model, 'encoder', orig_encoder)
    object.__setattr__(full_model, 'encoder_projector', orig_projector)

    # 对比结果
    all_match = True
    for orig, ext_res in zip(orig_results, ext_results):
        uid = orig["uttid"]
        match = orig["text"] == ext_res["text"]
        all_match = all_match and match
        status = "MATCH" if match else "DIFF"
        print(f"    [{status}] {uid}")
        print(f"        {'Original':<10}: {orig['text']}")
        print(f"        {label:<10}: {ext_res['text']}")
        if uid in ref_texts:
            print(f"        {'Ref':<10}: {ref_texts[uid]}")

    print(f"    Overall: {'ALL MATCH' if all_match else 'MISMATCH'}")
    return all_match


# ======================== 主流程 ========================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="FireRedASR Audio Encoder ONNX/HMONNX 导出 (xh2a)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="weights/FireRedASR-LLM-L",
        help="FireRedASR 模型目录",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./work_dirs/fireredasr_audio_encoder",
        help="输出目录",
    )
    parser.add_argument(
        "--audio_seconds",
        type=float,
        default=10.0,
        help="dummy 音频时长（秒），最大 30s",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch 大小"
    )
    parser.add_argument(
        "--valid",
        action="store_true",
        help="导出后验证 ONNX 精度（dummy + 真实音频）",
    )
    parser.add_argument(
        "--valid_asr",
        action="store_true",
        help="完整 ASR 转写结果对比（需 GPU 加载 LLM）",
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default=None,
        help="真实音频目录（默认自动检测 data/wav/）",
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        default=None,
        help="参考文本文件",
    )
    parser.add_argument(
        "--export_hmonnx",
        action="store_true",
        help="同时导出 HMONNX",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        help="使用 GPU 进行 ASR 验证",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="可选：加载 quarot/gptq 的 LLM checkpoint（用于联调 encoder+llm）",
    )
    parser.add_argument(
        "--rotated_adapter_path",
        type=str,
        default=None,
        help="可选：加载旋转后的 audio projector（audio_projector_rotated.safetensors）",
    )
    parser.add_argument(
        "--llm_quant_config",
        type=str,
        default="configs/fireredasr/fireredasr_llm_xh2a_2k_gptq_quarot_4bit_ssfp.py",
        help="加载 quantized LLM 时使用的配置文件",
    )
    parser.add_argument(
        "--lora_mode",
        type=str,
        choices=["merge_lora", "keep_lora"],
        default="merge_lora",
        help="加载 quantized LLM checkpoint 的 LoRA 模式",
    )
    parser.add_argument(
        "--hf_model_dir",
        type=str,
        default=None,
        help="可选：Qwen2 HF 基座路径（默认自动推断）",
    )
    parser.add_argument(
        "--generate_hmonnx_golden",
        action="store_true",
        help="导出 HMONNX 后生成 Audio Encoder golden",
    )
    parser.add_argument(
        "--golden_dir",
        type=str,
        default=None,
        help="golden 输出目录（默认: <output_dir>/golden/audio_encoder）",
    )
    return parser


def _find_wav_dir():
    """自动检测 wav 目录。"""
    for candidate in ["data/wav", "data/wavdata/wav"]:
        p = Path(candidate)
        if p.exists() and list(p.glob("*.wav")):
            return str(p)
    return None


def _find_ref_text(wav_dir: str):
    """自动检测参考文本文件。"""
    text_file = Path(wav_dir) / "text"
    if text_file.exists():
        return str(text_file)
    return None


def main():
    parser = parse_arguments()
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    print("=" * 60)
    print("FireRedASR Audio Encoder Export (xh2a)")
    print("=" * 60)

    # ---- 1. 加载模型 ----
    print("\n[1/5] Loading FireRedASR model ...")
    encoder, adapter, full_model, tokenizer = load_fireredasr_encoder_and_adapter(
        args.model_dir
    )
    _load_quantized_llm_if_needed(full_model, args)
    rotated_adapter_path = _infer_rotated_adapter_path(args.resume_from, args.rotated_adapter_path)
    if rotated_adapter_path is not None:
        _load_rotated_adapter_if_needed(adapter, rotated_adapter_path)

    T_conv_total = compute_conv_total_length(T_FBANK_MAX)
    T_adapter_total = compute_adapter_output_length(T_conv_total)
    print(f"  Max audio: {MAX_AUDIO_SECONDS}s, fbank frames: {T_FBANK_MAX}")
    print(f"  Conv output (total): {T_conv_total}")
    print(f"  Adapter output (total): {T_adapter_total}")
    print(f"  Encoder dim: {encoder.odim}, LLM dim: {adapter.linear2.out_features}")

    # ---- 2. 真实音频: 在 patch 前先用原始 encoder 跑一次 ----
    wav_dir = args.wav_dir or _find_wav_dir()
    ref_text = args.ref_text

    orig_speech_features = None
    orig_speech_lens = None
    fbank_padded = None
    fbank_lengths = None
    wav_uttids = None

    if args.valid and wav_dir:
        print(f"\n[2/5] Running original encoder on real audio ({wav_dir}) ...")
        from fireredasr.data.asr_feat import ASRFeatExtractor

        cmvn_path = os.path.join(args.model_dir, "cmvn.ark")
        feat_extractor = ASRFeatExtractor(cmvn_path)

        wav_paths = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
        wav_uttids = [Path(p).stem for p in wav_paths]
        print(f"  Found {len(wav_paths)} wav files: {wav_uttids}")

        feats, lengths, durs = feat_extractor(wav_paths)
        fbank_lengths = lengths
        fbank_padded = pad_fbank_to_fixed(feats, T_FBANK_MAX)

        encoder.eval().float()
        adapter.eval().float()
        with torch.no_grad():
            enc_out, enc_lens, _ = encoder(fbank_padded, fbank_lengths)
            orig_speech_features, orig_speech_lens = adapter(enc_out, enc_lens)
        print(
            f"  Original output shape: {orig_speech_features.shape}, "
            f"speech_lens: {orig_speech_lens.tolist()}"
        )

        if ref_text is None:
            ref_text = _find_ref_text(wav_dir)
    else:
        print("\n[2/5] Skipped real audio (no wav files or --valid not set)")

    # ---- 3. Patch encoder & build export model ----
    print("\n[3/5] Patching encoder for xh2a export ...")
    patch_encoder_for_export(encoder)
    export_model = AudioEncoderWithAdapter(encoder, adapter)
    export_model.eval()

    # ---- 4. 创建 dummy 输入 & 导出 ONNX ----
    print("\n[4/5] Creating dummy inputs & exporting ONNX ...")
    fbank_dummy, attn_mask_dummy, conv_mask_dummy = create_dummy_inputs(
        batch_size=args.batch_size, audio_seconds=args.audio_seconds
    )
    print(f"  fbank_features: {fbank_dummy.shape}")
    print(f"  attn_mask:      {attn_mask_dummy.shape}")
    print(f"  conv_mask:      {conv_mask_dummy.shape}")

    valid_conv_len = compute_conv_valid_length(
        min(int(args.audio_seconds * 1000 / FBANK_FRAME_SHIFT_MS), T_FBANK_MAX)
    )
    print(f"  Valid conv frames: {valid_conv_len}/{T_conv_total}")

    onnx_path = str(output_dir / "audio_encoder.onnx")
    start_time = time.time()
    export_audio_encoder_onnx(
        export_model, fbank_dummy, attn_mask_dummy, conv_mask_dummy, onnx_path
    )
    print(f"  Export time: {time.time() - start_time:.1f}s")

    # ---- 5. 验证 ----
    if args.valid:
        print("\n[5/5] Validation ...")
        # 5a. Dummy 验证
        validate_onnx_dummy(
            export_model, onnx_path, fbank_dummy, attn_mask_dummy, conv_mask_dummy
        )

        # 5b. 真实音频数值验证
        if orig_speech_features is not None:
            validate_real_audio(
                onnx_path,
                orig_speech_features,
                orig_speech_lens,
                fbank_padded,
                fbank_lengths,
                wav_uttids,
            )

        # 5c. ASR 转写对比
        if args.valid_asr and wav_dir:
            validate_asr_transcription(
                full_model,
                tokenizer,
                onnx_path,
                wav_dir,
                args.model_dir,
                ref_text_file=ref_text,
                use_gpu=args.use_gpu,
            )
    else:
        print("\n[5/5] Validation skipped (no --valid)")

    # ---- 6. 导出 HMONNX & 验证 ----
    if args.export_hmonnx:
        hmonnx_path = str(output_dir / "audio_encoder_hmonnx.onnx")
        print(f"\n[6] Converting to HMONNX: {hmonnx_path} ...")
        convert_to_hmonnx(
            onnx_path,
            hmonnx_path,
            fbank_dummy,
            attn_mask_dummy,
            conv_mask_dummy,
        )
        if args.generate_hmonnx_golden:
            golden_dir = args.golden_dir
            if golden_dir is None:
                golden_dir = str(output_dir / "golden" / "audio_encoder")
            print(f"\n[6g] Generating HMONNX golden: {golden_dir} ...")
            generate_hmonnx_golden(
                hmonnx_path=hmonnx_path,
                fbank_features=fbank_dummy,
                attn_mask=attn_mask_dummy,
                conv_mask=conv_mask_dummy,
                golden_dir=golden_dir,
                exec_device="cuda" if torch.cuda.is_available() else "cpu",
            )

        # 6a. Dummy 验证：ONNX vs HMONNX
        if args.valid:
            print("\n[6a] HMONNX Dummy Validation ...")
            validate_hmonnx_dummy(
                onnx_path, hmonnx_path,
                fbank_dummy, attn_mask_dummy, conv_mask_dummy,
            )

        # 6b. 真实音频数值验证：Original vs HMONNX
        if args.valid and orig_speech_features is not None:
            print("\n[6b] HMONNX Real Audio Validation ...")
            validate_real_audio(
                hmonnx_path,
                orig_speech_features,
                orig_speech_lens,
                fbank_padded,
                fbank_lengths,
                wav_uttids,
                backend="hmonnx",
                max_diff_threshold=0.5,
            )

        # 6c. ASR 转写对比：Original vs HMONNX
        if args.valid_asr and wav_dir:
            print("\n[6c] HMONNX ASR Transcription Validation ...")
            validate_asr_transcription(
                full_model,
                tokenizer,
                hmonnx_path,
                wav_dir,
                args.model_dir,
                ref_text_file=ref_text,
                use_gpu=args.use_gpu,
                backend="hmonnx",
            )

    print("\n" + "=" * 60)
    print(f"Export completed! Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
