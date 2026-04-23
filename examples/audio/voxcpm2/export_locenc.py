"""VoxCPM2 LocEnc-Step 的 HMONNX 导出脚本。

用法:
    python export_locenc.py --model ~/models/VoxCPM2 \
        --cal_wav /path/to/some_audio.wav
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
import warnings
from pathlib import Path
from unittest.mock import patch

import onnx
import onnxsim
import torch
import torch.nn as nn
from onnx import TensorProto


_ONNX_DTYPE_TO_NAME = {
    TensorProto.FLOAT: "float32",
    TensorProto.FLOAT16: "float16",
    TensorProto.DOUBLE: "float64",
    TensorProto.BFLOAT16: "bfloat16",
    TensorProto.INT32: "int32",
    TensorProto.INT64: "int64",
}

_NAME_TO_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}


def _read_onnx_input_dtype(onnx_path: str, input_name: str) -> str | None:
    """读取 HMONNX/ONNX 模型指定 input 的真实 dtype 名字。"""
    try:
        model = onnx.load(onnx_path)
    except Exception:
        return None
    for graph_input in model.graph.input:
        if graph_input.name != input_name:
            continue
        elem_type = graph_input.type.tensor_type.elem_type
        return _ONNX_DTYPE_TO_NAME.get(elem_type)
    return None


def _cast_to_graph_dtype(tensor: torch.Tensor, dtype_name: str | None) -> torch.Tensor:
    """按图真实 dtype 把张量 cast 过去;读不到 dtype 就保持原样。"""
    if dtype_name is None:
        return tensor
    target = _NAME_TO_TORCH_DTYPE.get(dtype_name)
    if target is None or tensor.dtype == target:
        return tensor
    return tensor.to(dtype=target)

from xhquant.api import (
    DeviceType,
    HMONNXInference,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
)
from xhquant.patch.core import RewriterContext

from voxcpm import VoxCPM2Model

try:
    from .utils import write_json_file
except ImportError:
    from utils import write_json_file


GB = int(2**30)
_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 0.1)


def _first_output(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _onnx_safe_scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
    enable_gqa: bool = False,
):
    if enable_gqa:
        q_heads = int(query.shape[-3])
        kv_heads = int(key.shape[-3])
        if q_heads % kv_heads != 0:
            raise ValueError(f"Invalid GQA head mapping: q_heads={q_heads}, kv_heads={kv_heads}")
        groups = q_heads // kv_heads
        key = key.repeat_interleave(groups, dim=-3)
        value = value.repeat_interleave(groups, dim=-3)
        enable_gqa = False

    return torch.ops.aten.scaled_dot_product_attention.default(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale=scale,
        enable_gqa=enable_gqa,
    )


# ---------------------------------------------------------------------------
# 真实 latent 生成:跑 AudioVAE encoder,按 patch 切出若干 [1, 1, P, D] 样本
# ---------------------------------------------------------------------------
def _load_wav_mono(wav_path: Path, target_sr: int | None, logger) -> tuple[torch.Tensor, int]:
    """尽量鲁棒地读 wav,返回 [1, L] 的 mono float32 tensor 和采样率。

    优先用 soundfile,退到 torchaudio,再退到 scipy.io.wavfile。
    """
    wav_path = Path(wav_path).expanduser().resolve()
    if not wav_path.exists():
        raise FileNotFoundError(f"calibration wav not found: {wav_path}")

    wav = None
    sr = None
    last_err = None

    try:
        import soundfile as sf
        data, sr = sf.read(str(wav_path), always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)  # 多通道取平均转 mono
        wav = torch.from_numpy(data).float().unsqueeze(0)  # [1, L]
    except Exception as e:
        last_err = e

    if wav is None:
        try:
            import torchaudio
            wav_ta, sr = torchaudio.load(str(wav_path))  # [C, L]
            if wav_ta.shape[0] > 1:
                wav_ta = wav_ta.mean(dim=0, keepdim=True)
            wav = wav_ta.float()
        except Exception as e:
            last_err = e

    if wav is None:
        try:
            from scipy.io import wavfile
            sr, data = wavfile.read(str(wav_path))
            if data.dtype.kind == "i":
                data = data.astype("float32") / float(2 ** (8 * data.dtype.itemsize - 1))
            elif data.dtype.kind == "u":
                data = data.astype("float32") / float(2 ** (8 * data.dtype.itemsize)) - 0.5
            else:
                data = data.astype("float32")
            if data.ndim == 2:
                data = data.mean(axis=1)
            wav = torch.from_numpy(data).unsqueeze(0)
        except Exception as e:
            last_err = e

    if wav is None:
        raise RuntimeError(
            f"Failed to load {wav_path} with soundfile/torchaudio/scipy: {last_err}"
        )

    # 简单重采样
    if target_sr is not None and sr != target_sr:
        try:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, target_sr)
        except Exception:
            # 没 torchaudio 就用 scipy.signal.resample_poly 做一次
            from scipy.signal import resample_poly
            import math
            g = math.gcd(sr, target_sr)
            up = target_sr // g
            down = sr // g
            wav_np = wav.squeeze(0).numpy()
            wav_np = resample_poly(wav_np, up, down).astype("float32")
            wav = torch.from_numpy(wav_np).unsqueeze(0)
        logger.info("Resampled wav from %d Hz to %d Hz", sr, target_sr)
        sr = target_sr

    logger.info("Loaded calibration wav: %s, sr=%d, length=%d samples (%.2fs)",
                wav_path, sr, wav.shape[-1], wav.shape[-1] / float(sr))
    return wav, sr


def _build_real_calibration_samples(
    voxcpm2: VoxCPM2Model,
    wav_path: Path | None,
    num_samples: int,
    device: torch.device,
    dtype: torch.dtype,
    logger,
) -> tuple[torch.Tensor, list[torch.Tensor], bool]:
    """返回 (feat_cal, cal_list, used_real)。

    - feat_cal:   [1, 1, P, D],用于 sanity forward 和 parity 对比
    - cal_list:   list of [1, 1, P, D],送给 convert_onnx_to_hmonnx 做 calibration
    - used_real:  是否真的用了真实 latent(失败/未提供时为 False)
    """
    patch_size = voxcpm2.patch_size
    latent_dim = voxcpm2.audio_vae.latent_dim
    in_sr = getattr(voxcpm2.audio_vae, "in_sample_rate", None)

    torch.manual_seed(0)

    # ---- fallback: 随机 latent ----
    def _fallback_random():
        logger.warning(
            "Using RANDOM calibration samples. This is known to cause w8a8 "
            "precision issues for LocEnc. Pass --cal_wav to use real audio."
        )
        samples = [
            torch.randn((1, 1, patch_size, latent_dim), device=device, dtype=dtype) * 0.5
            for _ in range(num_samples)
        ]
        return samples[0], samples, False

    if wav_path is None:
        return _fallback_random()

    # ---- 真实音频 → AudioVAE encoder → latent → 切 patch ----
    try:
        wav, sr = _load_wav_mono(wav_path, target_sr=in_sr, logger=logger)
        wav = wav.to(device=device, dtype=dtype)  # [1, L]

        encoder = voxcpm2.audio_vae.encoder
        encoder.eval()

        with torch.no_grad():
            latent = None
            last_err = None
            # 1) 2D [B, L]
            try:
                latent = encoder(wav)
            except Exception as e:
                last_err = e
            # 2) 3D [B, 1, L]
            if latent is None:
                try:
                    latent = encoder(wav.unsqueeze(1))
                except Exception as e:
                    last_err = e
            # 3) 通过 audio_vae 顶层 encode 方法
            if latent is None and hasattr(voxcpm2.audio_vae, "encode"):
                try:
                    latent = voxcpm2.audio_vae.encode(wav)
                    if isinstance(latent, (tuple, list)):
                        latent = latent[0]
                except Exception as e:
                    last_err = e

            if latent is None:
                raise RuntimeError(f"Failed to run AudioVAE encoder: {last_err}")

        # 规整成 [1, latent_dim, T_lat]
        if latent.dim() != 3:
            raise RuntimeError(f"Unexpected latent shape: {tuple(latent.shape)}")
        # 兼容 [B, T, D] / [B, D, T]
        if latent.shape[1] == latent_dim and latent.shape[2] != latent_dim:
            pass  # 已经是 [B, D, T]
        elif latent.shape[2] == latent_dim:
            latent = latent.transpose(1, 2).contiguous()
        else:
            raise RuntimeError(
                f"Cannot find latent_dim={latent_dim} axis in latent shape {tuple(latent.shape)}"
            )

        B, D, T_lat = latent.shape
        assert B == 1 and D == latent_dim
        num_patches_total = T_lat // patch_size
        if num_patches_total < 1:
            raise RuntimeError(
                f"Latent too short ({T_lat} frames) to form a single patch of size {patch_size}"
            )
        logger.info(
            "Real latent ready: shape=%s, %d full patches available",
            tuple(latent.shape), num_patches_total,
        )

        # 切成 patch: 把 [1, D, T_lat] -> [1, T_lat, D] -> 按 P 切
        latent_bt = latent.transpose(1, 2).contiguous()  # [1, T_lat, D]
        latent_bt = latent_bt[:, : num_patches_total * patch_size, :]
        # [1, num_patches_total, P, D]
        patched = latent_bt.reshape(1, num_patches_total, patch_size, latent_dim)

        # 随机抽 num_samples 个 patch 作为 calibration
        if num_patches_total >= num_samples:
            perm = torch.randperm(num_patches_total)[:num_samples]
        else:
            # 样本不够就重复
            logger.warning(
                "Only %d patches available, need %d — will sample with replacement.",
                num_patches_total, num_samples,
            )
            perm = torch.randint(0, num_patches_total, (num_samples,))

        samples = []
        for idx in perm.tolist():
            one = patched[:, idx : idx + 1, :, :].contiguous()  # [1, 1, P, D]
            samples.append(one.to(device=device, dtype=dtype))

        # feat_cal(用于 parity)统一用第一个样本
        feat_cal = samples[0]

        # 打一下真实 latent 的统计,方便确认
        with torch.no_grad():
            lat_flat = latent_bt.reshape(-1).float()
            logger.info(
                "Real latent stats: mean=%.4f std=%.4f |x|.mean=%.4f |x|.max=%.4f",
                lat_flat.mean().item(), lat_flat.std().item(),
                lat_flat.abs().mean().item(), lat_flat.abs().max().item(),
            )

        logger.info("Built %d real calibration samples of shape %s",
                    len(samples), tuple(feat_cal.shape))
        return feat_cal, samples, True

    except Exception as e:
        logger.warning("Real-latent calibration failed (%s). Falling back to random.", e)
        return _fallback_random()


def _verify_hmonnx_vs_pytorch(
    hmonnx_file: Path,
    device: torch.device,
    feat_cal: torch.Tensor,
    torch_out: torch.Tensor,
    wrapper: nn.Module,
    logger,
    *,
    graph_feat_dtype: str | None,
    max_abs_tol: float,
    mean_abs_tol: float,
    cosine_tol: float,
):
    session = HMONNXInference(str(hmonnx_file))
    session.to(str(device))

    # 按图真实 dtype 送入(XH2a 量化后常把 fp32 降成 fp16,直接送 fp32
    # 会在 session 内部 assert dtype mismatch)。
    feat_in = _cast_to_graph_dtype(feat_cal, graph_feat_dtype)
    hmonnx_out = _first_output(session.run({"feat": feat_in}))

    # 如果送入 dtype 和 torch_out 用的 dtype 不一致,重新用相同 dtype 跑
    # 一次 PyTorch 侧作为参考基准,避免 fp32→fp16 本身的 cast 误差被当作
    # 量化误差算进 parity。
    if feat_in.dtype != feat_cal.dtype:
        wrapper_dtype = next(wrapper.parameters()).dtype
        with torch.no_grad():
            ref_out = wrapper(feat_in.to(device=device, dtype=wrapper_dtype))
    else:
        ref_out = torch_out

    ref = ref_out.detach().float().reshape(-1).cpu()
    got = hmonnx_out.detach().float().reshape(-1).cpu()

    if ref.shape != got.shape:
        raise RuntimeError(f"Parity shape mismatch: pytorch={tuple(ref.shape)} hmonnx={tuple(got.shape)}")

    abs_diff = (got - ref).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    cosine = float(torch.nn.functional.cosine_similarity(got.unsqueeze(0), ref.unsqueeze(0), dim=-1).item())
    ref_abs_mean = float(ref.abs().mean().item())
    ref_abs_max = float(ref.abs().max().item())

    logger.info(
        "Parity(LocEnc): max_abs=%.6f mean_abs=%.6f cosine=%.6f "
        "| ref |x|.mean=%.4f |x|.max=%.4f",
        max_abs, mean_abs, cosine, ref_abs_mean, ref_abs_max,
    )

    passed = bool(max_abs <= max_abs_tol and mean_abs <= mean_abs_tol and cosine >= cosine_tol)

    # ---- 失败时的 per-channel 诊断 ----
    if not passed:
        try:
            # 用 ref_out 的 shape 作为基准(在 dtype 重跑分支下它可能
            # 和传入的 torch_out 不是同一个 tensor)。
            out_shape = ref_out.shape
            diff_2d = (got.reshape(out_shape) - ref.reshape(out_shape)).abs()
            per_ch = diff_2d.squeeze(0)  # [H_lm]
            topk = torch.topk(per_ch, k=min(10, per_ch.numel()))
            logger.warning(
                "Per-channel err — top10 idx=%s", topk.indices.tolist(),
            )
            logger.warning(
                "Per-channel err — top10 val=%s",
                [round(v, 4) for v in topk.values.tolist()],
            )
            logger.warning(
                "Per-channel err stats: mean=%.6f std=%.6f "
                "(std/mean=%.2f → 高=少数 channel 坏, 低=全局偏)",
                per_ch.mean().item(), per_ch.std().item(),
                per_ch.std().item() / max(per_ch.mean().item(), 1e-9),
            )
        except Exception as e:
            logger.warning("per-channel diagnostic failed: %s", e)

    return {
        "passed": passed,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cosine": cosine,
        "ref_abs_mean": ref_abs_mean,
        "ref_abs_max": ref_abs_max,
        "graph_feat_dtype": graph_feat_dtype,
    }


# 组合 LocEnc + enc_to_lm_proj 的导出包装
class LocEncStepExportWrapper(nn.Module):
    """把 LocEnc 和 enc_to_lm_proj 合并成一张图,固定 batch=1, T=1。"""

    def __init__(self, feat_encoder: nn.Module, enc_to_lm_proj: nn.Module):
        super().__init__()
        self.feat_encoder = feat_encoder
        self.enc_to_lm_proj = enc_to_lm_proj

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feat: [1, 1, P, D]
        Returns:
            hidden: [1, H_lm]
        """
        h = self.feat_encoder(feat)           # [1, 1, H_enc]
        h = self.enc_to_lm_proj(h)            # [1, 1, H_lm]
        h = h.squeeze(1)                      # [1, H_lm]
        return h


def main(args):
    model_path = str(Path(args.model).expanduser().resolve())
    model_name = Path(model_path).name
    target_device = "XH2a"

    script_dir = Path(__file__).resolve().parent
    work_root = script_dir / "work_dirs"
    work_dir = work_root / f"{model_name}_{target_device}" / "LocEnc"
    work_dir.mkdir(exist_ok=True, parents=True)
    logger = get_root_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    logger.info("Loading VoxCPM2 from %s", model_path)
    voxcpm2 = VoxCPM2Model.from_local(model_path, optimize=False, training=False)
    voxcpm2.to(device=device)
    voxcpm2.eval()
    # wrapper 和 audio_vae 都统一到 fp32
    voxcpm2.audio_vae.to(dtype=dtype)

    # --- 构造导出 wrapper ---
    wrapper = LocEncStepExportWrapper(voxcpm2.feat_encoder, voxcpm2.enc_to_lm_proj)
    wrapper = wrapper.to(device=device, dtype=dtype)
    wrapper.eval()

    # --- 构造 calibration ---
    patch_size = voxcpm2.patch_size
    latent_dim = voxcpm2.audio_vae.latent_dim

    cal_wav_path = Path(args.cal_wav).expanduser().resolve() if args.cal_wav else None
    feat_cal, cal_samples, used_real = _build_real_calibration_samples(
        voxcpm2=voxcpm2,
        wav_path=cal_wav_path,
        num_samples=max(1, args.num_cal_samples),
        device=device,
        dtype=dtype,
        logger=logger,
    )
    logger.info(
        "Calibration feat.shape=%s dtype=%s num_samples=%d used_real=%s",
        tuple(feat_cal.shape), feat_cal.dtype, len(cal_samples), used_real,
    )

    # --- 前向 sanity check ---
    with torch.no_grad():
        y = wrapper(feat_cal)
    logger.info("Wrapper forward OK, output.shape=%s dtype=%s", tuple(y.shape), y.dtype)

    # --- 导出 ONNX ---
    onnx_file = work_dir / "voxcpm2_locenc_step.onnx"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_onnx = str(Path(tmp_dir) / onnx_file.name)

        with RewriterContext(None, backend="onnxruntime"):
            with patch(
                "torch.nn.functional.scaled_dot_product_attention",
                _onnx_safe_scaled_dot_product_attention,
            ):
                torch.onnx.export(
                    wrapper,
                    (feat_cal,),
                    tmp_onnx,
                    input_names=["feat"],
                    output_names=["hidden"],
                    opset_version=17,
                )
            logger.info("ONNX exported to %s", tmp_onnx)

            onnx_model = onnx.load(tmp_onnx)
            size = onnx_model.ByteSize()
            logger.info("ONNX size: %.3f GB", size / GB)

            if size <= _LARGE_MODEL_SIZE_THRESHOLD:
                simplified, ok = onnxsim.simplify(
                    onnx_model,
                    skipped_optimizers=[
                        "fuse_pad_into_conv",
                        "fuse_consecutive_slices",
                        "eliminate_common_subexpression",
                        "fuse_qkv",
                    ],
                )
            else:
                from xhquant.utils.onnxsim_large_model import simplify_large_onnx
                simplified, ok = simplify_large_onnx(
                    onnx_model,
                    skipped_optimizers=[
                        "fuse_pad_into_conv",
                        "fuse_consecutive_slices",
                        "eliminate_common_subexpression",
                        "fuse_qkv",
                    ],
                )
            if ok:
                onnx_model = simplified

    onnx.save(
        onnx_model,
        onnx_file,
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{onnx_file.stem}_external_data",
    )
    logger.info("ONNX saved: %s", onnx_file)

    # --- 转 HMONNX ---
    hmonnx_file = work_dir / "hmonnx" / f"voxcpm2_locenc_step_xh2a_{args.quant_type}.onnx"
    hmonnx_file.parent.mkdir(exist_ok=True, parents=True)

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)

    example_input = [feat_cal]

    injected_field = None
    if len(cal_samples) > 1:
        candidate_fields = (
            "calib_data",
            "calibration_data",
            "calib_dataset",
            "calibration_dataset",
            "calib_dataloader",
            "calibration_dataloader",
            "calib_inputs",
            "calibration_inputs",
        )
        for field in candidate_fields:
            try:
                if hasattr(quant_config, field):
                    setattr(quant_config, field, cal_samples)
                    injected_field = field
                    break
            except Exception:
                continue
        if injected_field is None and isinstance(quant_config, dict):
            for field in candidate_fields:
                if field in quant_config:
                    quant_config[field] = cal_samples
                    injected_field = field
                    break

    if injected_field is not None:
        logger.info(
            "Injected %d calibration samples via quant_config.%s",
            len(cal_samples), injected_field,
        )
    elif len(cal_samples) > 1:
        logger.warning(
            "quant_config does not expose a known calibration field; "
            "falling back to single-sample calibration. "
            "Please check xhquant docs for the correct field name and set it manually."
        )

    logger.info("Converting to HMONNX: %s", hmonnx_file)
    convert_onnx_to_hmonnx(
        str(onnx_file),
        example_input,
        DeviceType.XH2a,
        hmonnx_file,
        quant_config=quant_config,
        input_names=["feat"],
        output_names=["hidden"],
    )
    logger.info("HMONNX saved: %s", hmonnx_file)

    graph_feat_dtype = _read_onnx_input_dtype(str(hmonnx_file), "feat")
    logger.info("HMONNX graph input dtype: feat=%s", graph_feat_dtype)

    if not args.skip_verify:
        parity = _verify_hmonnx_vs_pytorch(
            hmonnx_file=hmonnx_file,
            device=device,
            feat_cal=feat_cal,
            torch_out=y,
            wrapper=wrapper,
            logger=logger,
            graph_feat_dtype=graph_feat_dtype,
            max_abs_tol=args.verify_max_abs_tol,
            mean_abs_tol=args.verify_mean_abs_tol,
            cosine_tol=args.verify_cosine_tol,
        )
        if parity["passed"]:
            logger.info("LocEnc parity check passed.")
        else:
            msg = (
                "LocEnc parity check failed: "
                f"max_abs={parity['max_abs']:.6f}(tol={args.verify_max_abs_tol}), "
                f"mean_abs={parity['mean_abs']:.6f}(tol={args.verify_mean_abs_tol}), "
                f"cosine={parity['cosine']:.6f}(tol>={args.verify_cosine_tol})"
            )
            if args.verify_fail_on_mismatch:
                raise AssertionError(msg)
            logger.warning(msg)

    # --- golden ---
    golden_dir = work_dir / "hmonnx" / "golden"
    if args.gen_golden:
        if golden_dir.exists():
            shutil.rmtree(golden_dir)
        session = HMONNXGoldenInference(hmonnx_file)
        session.to(device)
        session.save_golden = True
        session.golden_dir = str(golden_dir)
        session.step = 0
        # 同样按图 dtype 送入
        feat_for_golden = _cast_to_graph_dtype(feat_cal, graph_feat_dtype)
        session(feat_for_golden)
        logger.info("golden saved: %s", golden_dir)

    # --- meta ---
    meta = dict(
        create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        model_name=model_name,
        target_device=target_device,
        module="LocEnc-Step",
        onnx_file=str(onnx_file.relative_to(work_dir.parent)),
        hmonnx_file=str(hmonnx_file.relative_to(work_dir.parent)),
        golden_dir=str(golden_dir.relative_to(work_dir.parent)) if args.gen_golden else None,
        quant_type=args.quant_type,
        input_dtype=str(dtype).replace("torch.", ""),
        # HMONNX 图真实接受的 dtype,host 侧构造输入时必须按这个 cast
        graph_input_dtype=dict(feat=graph_feat_dtype),
        input_names=["feat"],
        output_names=["hidden"],
        input_shapes=dict(feat=[1, 1, patch_size, latent_dim]),
        output_shapes=dict(hidden=[1, int(y.shape[-1])]),
        patch_size=patch_size,
        latent_dim=latent_dim,
        hidden_size_lm=int(y.shape[-1]),
        calibration=dict(
            used_real_latent=used_real,
            wav_path=str(cal_wav_path) if cal_wav_path else None,
            num_samples=len(cal_samples),
        ),
    )
    meta_file = work_dir / "locenc_meta_info.json"
    write_json_file(meta_file, meta)
    logger.info("Meta saved: %s", meta_file)
    logger.info("=" * 60)
    logger.info("LocEnc export done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--cal_wav", type=str, default=None,
        help="真实音频路径(wav)。强烈建议传入:用它跑 AudioVAE encoder "
             "得到真实 latent 分布做 calibration。不传则 fallback 到 randn。",
    )
    parser.add_argument(
        "--num_cal_samples", type=int, default=32,
        help="从真实 latent 里抽多少个 patch 做 calibration(默认 32)。",
    )
    parser.add_argument("--quant_type", type=str, default="w8a8_sefp")
    parser.add_argument("--gen_golden", action="store_true")
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--verify_max_abs_tol", type=float, default=1.0)
    parser.add_argument("--verify_mean_abs_tol", type=float, default=0.1)
    parser.add_argument("--verify_cosine_tol", type=float, default=0.95)
    parser.add_argument("--verify_fail_on_mismatch", action="store_true")
    args = parser.parse_args()
    main(args)