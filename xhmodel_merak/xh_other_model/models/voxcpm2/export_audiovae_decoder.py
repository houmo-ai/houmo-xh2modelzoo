"""VoxCPM2 AudioVAE Decoder 的 HMONNX 导出脚本。
用法:
    python export_audiovae_decoder.py --model ~/models/VoxCPM2 \
        --num_patches 24 --gen_golden
"""

from __future__ import annotations

import argparse
import math
import shutil
import tempfile
import time
from pathlib import Path

import onnx
import onnxsim
import torch
import torch.nn as nn
from onnx import TensorProto

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
try:
    from voxcpm import VoxCPM2Model
except ImportError:
    from voxcpm.model.voxcpm2 import VoxCPM2Model

from .utils import (
    activate_export_device,
    compute_audiovae_decoder_output_samples,
    remove_weight_norm_recursively,
    write_json_file,
)


GB = int(2**30)
_LARGE_MODEL_SIZE_THRESHOLD = int(2**30 * 0.1)

# 和 encoder 脚本保持一致的 ONNX dtype 映射,用于从图里读取 input 真实 dtype。
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


def _first_output(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def _read_onnx_input_dtype(onnx_path: str, input_name: str) -> str | None:
    """读取 HMONNX/ONNX 模型指定 input 的真实 dtype 名字。

    和 encoder 侧保持完全一致,用来替代"捕获 AssertionError 再字符串匹配"
    的脆弱写法——xhquant 升级改报错文案时那种写法会直接崩。
    """
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


class AudioVAEDecoderExportWrapper(nn.Module):
    """把 decoder 的 bucketize 逻辑从图里拿掉,直接吃预先算好的 sr_idx。

    原生 CausalDecoder.forward 里:
        sr_cond = self.get_sr_idx(sr)   # torch.bucketize
        for layer, sr_cond_layer in zip(...):
            if sr_cond_layer is not None:
                x = sr_cond_layer(x, sr_cond)
            x = layer(x)

    导出时把 bucketize 放到 host(host 就知道是 48000 → bucket 3),
    图内直接循环调用即可。
    """

    def __init__(self, vae_decoder):
        super().__init__()
        # vae_decoder 是 CausalDecoder
        self.decoder = vae_decoder
        self.has_sr_cond = vae_decoder.sr_bin_boundaries is not None

    def forward(self, z: torch.Tensor, sr_idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z:      [1, latent_dim, T]
            sr_idx: [1]  int32  —— 已经 bucketize 过的索引
        Returns:
            audio: [1, 1, L_out]
        """
        if self.has_sr_cond:
            x = z
            for layer, sr_cond_layer in zip(self.decoder.model, self.decoder.sr_cond_model):
                if sr_cond_layer is not None:
                    x = sr_cond_layer(x, sr_idx)
                x = layer(x)
            return x
        else:
            return self.decoder.model(z)


def _compute_sr_idx(voxcpm2: VoxCPM2Model, sr_value: int) -> int:
    """host 侧预先算好 bucketize 索引(与 CausalDecoder.get_sr_idx 等价)。"""
    boundaries = voxcpm2.audio_vae.decoder.sr_bin_boundaries
    # 和原生 torch.bucketize 行为一致
    sr_tensor = torch.tensor([sr_value], dtype=torch.int32)
    idx = int(torch.bucketize(sr_tensor, boundaries.to(sr_tensor.device))[0].item())
    return idx


def _verify_hmonnx_vs_pytorch(
    hmonnx_file: Path,
    device: torch.device,
    z_cal: torch.Tensor,
    sr_cal: torch.Tensor,
    torch_out: torch.Tensor,
    wrapper: nn.Module,
    logger,
    *,
    graph_z_dtype: str | None,
    graph_sr_dtype: str | None,
    max_abs_tol: float,
    mean_abs_tol: float,
    cosine_tol: float,
):
    """对比 HMONNX 和 PyTorch 的数值差异。

    输入 dtype 以 HMONNX 图里的真实 dtype 为准(由调用方从图里读出来传入),
    避免 session 内部对 dtype 错配 assert 炸掉。
    """
    session = HMONNXInference(str(hmonnx_file))
    session.to(str(device))

    # 按图实际 dtype 构造 session 输入。读不到 dtype 则保持原样。
    z_in = _cast_to_graph_dtype(z_cal, graph_z_dtype)
    sr_in = _cast_to_graph_dtype(sr_cal, graph_sr_dtype)
    hmonnx_out = _first_output(session.run({"z": z_in, "sr_idx": sr_in}))

    # 如果 z 的送入 dtype 和 torch_out 用的 dtype 不一致(例如图只接受 fp16),
    # 重新用相同 dtype 跑一次 PyTorch 侧,保证参考基准对齐。
    if z_in.dtype != z_cal.dtype:
        wrapper_dtype = next(wrapper.parameters()).dtype
        with torch.no_grad():
            ref_out = wrapper(
                z_in.to(device=device, dtype=wrapper_dtype),
                sr_cal,  # sr_idx 是 int,wrapper 内部不参与浮点计算,用原始即可
            )
    else:
        ref_out = torch_out

    ref = ref_out.detach().float().reshape(-1).cpu()
    got = hmonnx_out.detach().float().reshape(-1).cpu()

    if ref.shape != got.shape:
        raise RuntimeError(
            f"Parity shape mismatch: pytorch={tuple(ref.shape)} hmonnx={tuple(got.shape)}"
        )

    abs_diff = (got - ref).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    cosine = float(
        torch.nn.functional.cosine_similarity(
            got.unsqueeze(0), ref.unsqueeze(0), dim=-1
        ).item()
    )
    # 顺便打印基准幅度,便于判断相对误差量级(回应前面讨论的"0.5 大不大")
    ref_abs_mean = float(ref.abs().mean().item())
    ref_abs_max = float(ref.abs().max().item())
    logger.info(
        "Parity(AudioVAE-Decoder): max_abs=%.6f mean_abs=%.6f cosine=%.6f "
        "| ref |x|.mean=%.4f |x|.max=%.4f",
        max_abs,
        mean_abs,
        cosine,
        ref_abs_mean,
        ref_abs_max,
    )
    passed = bool(max_abs <= max_abs_tol and mean_abs <= mean_abs_tol and cosine >= cosine_tol)
    return {
        "passed": passed,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cosine": cosine,
        "ref_abs_mean": ref_abs_mean,
        "ref_abs_max": ref_abs_max,
        "graph_z_dtype": graph_z_dtype,
        "graph_sr_dtype": graph_sr_dtype,
    }


def main(args):
    model_path = str(Path(args.model).expanduser().resolve())
    model_name = Path(model_path).name
    target_device = "XH2a"

    # num_patches 决定每次 decode 的 chunk 大小,stream 版(小)和 full 版(大)
    # 产物放到不同目录,避免互相覆盖。
    if getattr(args, "output_dir", None):
        work_dir = Path(args.output_dir).expanduser().resolve() / f"AudioVAE_Decoder_np{args.num_patches}"
    else:
        script_dir = Path(__file__).resolve().parent
        work_root = script_dir / "work_dirs"
        work_dir = work_root / f"{model_name}_{target_device}" / f"AudioVAE_Decoder_np{args.num_patches}"
    work_dir.mkdir(exist_ok=True, parents=True)
    logger = get_root_logger()

    device = activate_export_device(getattr(args, "device", None))
    dtype = torch.float32  # AudioVAE decoder 推荐 FP32
    logger.info("Using export device: %s", device)

    logger.info("从 %s 中加载 VoxCPM2 模型", model_path)
    voxcpm2 = VoxCPM2Model.from_local(
        model_path,
        optimize=False,
        training=False,
        device=str(device),
    )
    voxcpm2.audio_vae.to(device=device, dtype=dtype)
    voxcpm2.audio_vae.eval()

    # --- weight_norm 展开 ---
    # 推理时等价于一个固定权重,导出前展开,避免 ONNX trace 把
    # weight_g / weight_v 这对参数引入图。
    logger.info("Removing weight_norm from AudioVAE decoder")
    remove_weight_norm_recursively(voxcpm2.audio_vae.decoder)

    # --- 构造 wrapper ---
    wrapper = AudioVAEDecoderExportWrapper(voxcpm2.audio_vae.decoder)
    wrapper = wrapper.to(device=device, dtype=dtype)
    wrapper.eval()

    # --- shape 参数 ---
    patch_size = voxcpm2.patch_size                                  # 4
    latent_dim = voxcpm2.audio_vae.latent_dim                         # 64
    decoder_rates = voxcpm2.audio_vae.decoder_rates                   # [8,6,5,2,2,2]
    upscale = int(math.prod(decoder_rates))                           # 960
    num_patches = args.num_patches
    T = num_patches * patch_size
    L_out = compute_audiovae_decoder_output_samples(num_patches, patch_size, decoder_rates)
    out_sample_rate = voxcpm2.audio_vae.out_sample_rate

    logger.info(
        "固定维度: num_patches=%d patch_size=%d T=%d upscale=%d L_out=%d",
        num_patches, patch_size, T, upscale, L_out,
    )

    sr_idx_value = _compute_sr_idx(voxcpm2, out_sample_rate)
    logger.info("sr_idx(%d Hz) = %d", out_sample_rate, sr_idx_value)

    # --- calibration 输入 ---
    torch.manual_seed(0)
    z_cal = torch.randn((1, latent_dim, T), device=device, dtype=dtype) * 0.5
    sr_cal = torch.tensor([sr_idx_value], device=device, dtype=torch.int32)

    logger.info("Calibration z.shape=%s sr.shape=%s",
                tuple(z_cal.shape), tuple(sr_cal.shape))

    # --- sanity check ---
    with torch.no_grad():
        audio_ref = wrapper(z_cal, sr_cal)
    logger.info("Decoder forward OK, audio.shape=%s", tuple(audio_ref.shape))
    assert audio_ref.shape == (1, 1, L_out), (
        f"Expected audio shape (1, 1, {L_out}), got {audio_ref.shape}"
    )

    # --- 导出 ONNX ---
    onnx_file = work_dir / "voxcpm2_audiovae_decoder.onnx"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_onnx = str(Path(tmp_dir) / onnx_file.name)

        with RewriterContext(None, backend="onnxruntime"):
            torch.onnx.export(
                wrapper,
                (z_cal, sr_cal),
                tmp_onnx,
                input_names=["z", "sr_idx"],
                output_names=["audio"],
                opset_version=17,
            )
            logger.info("ONNX exported")

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
    hmonnx_file = work_dir / "hmonnx" / f"voxcpm2_audiovae_decoder_np{args.num_patches}_xh2a_{args.quant_type}.onnx"
    hmonnx_file.parent.mkdir(exist_ok=True, parents=True)

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)

    logger.info("Converting to HMONNX")
    convert_onnx_to_hmonnx(
        str(onnx_file),
        [z_cal, sr_cal],
        DeviceType.XH2a,
        hmonnx_file,
        quant_config=quant_config,
        input_names=["z", "sr_idx"],
        output_names=["audio"],
    )
    logger.info("HMONNX saved: %s", hmonnx_file)

    # 读取 HMONNX 里两路 input 的真实 dtype,parity / golden 都按这个 dtype 送入
    graph_z_dtype = _read_onnx_input_dtype(str(hmonnx_file), "z")
    graph_sr_dtype = _read_onnx_input_dtype(str(hmonnx_file), "sr_idx")
    logger.info(
        "HMONNX graph input dtypes: z=%s sr_idx=%s",
        graph_z_dtype,
        graph_sr_dtype,
    )

    # --- parity 检查 ---
    if not args.skip_verify:
        parity = _verify_hmonnx_vs_pytorch(
            hmonnx_file=hmonnx_file,
            device=device,
            z_cal=z_cal,
            sr_cal=sr_cal,
            torch_out=audio_ref,
            wrapper=wrapper,
            logger=logger,
            graph_z_dtype=graph_z_dtype,
            graph_sr_dtype=graph_sr_dtype,
            max_abs_tol=args.verify_max_abs_tol,
            mean_abs_tol=args.verify_mean_abs_tol,
            cosine_tol=args.verify_cosine_tol,
        )
        if parity["passed"]:
            logger.info("AudioVAE Decoder parity check passed.")
        else:
            msg = (
                "AudioVAE Decoder parity check failed: "
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
        # golden 用 HMONNX 真实接受的 dtype 送入,否则 session 内部 assert 会报错
        z_for_golden = _cast_to_graph_dtype(z_cal, graph_z_dtype)
        sr_for_golden = _cast_to_graph_dtype(sr_cal, graph_sr_dtype)
        session(z_for_golden, sr_for_golden)
        logger.info("golden saved: %s", golden_dir)

    # --- meta ---
    meta = dict(
        create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        model_name=model_name,
        target_device=target_device,
        export_device=str(device),
        module="AudioVAE-Decoder",
        onnx_file=str(onnx_file.relative_to(work_dir.parent)),
        hmonnx_file=str(hmonnx_file.relative_to(work_dir.parent)),
        golden_dir=str(golden_dir.relative_to(work_dir.parent)) if args.gen_golden else None,
        quant_type=args.quant_type,
        input_dtype="float32",  # wrapper / calibration 侧 dtype
        # HMONNX 图真实接受的 dtype,host 侧构造输入时必须按这个 cast
        graph_input_dtype=dict(z=graph_z_dtype, sr_idx=graph_sr_dtype),
        input_names=["z", "sr_idx"],
        output_names=["audio"],
        input_shapes=dict(z=[1, latent_dim, T], sr_idx=[1]),
        output_shapes=dict(audio=[1, 1, L_out]),
        num_patches=num_patches,
        patch_size=patch_size,
        latent_dim=latent_dim,
        decoder_rates=list(decoder_rates),
        upscale=upscale,
        out_sample_rate=out_sample_rate,
        # host 侧 bucketize 结果,推理时直接用这个值填 sr_idx
        precomputed_sr_idx=sr_idx_value,
        sr_bin_boundaries=[int(x) for x in voxcpm2.audio_vae.decoder.sr_bin_boundaries.tolist()],
    )
    meta_file = work_dir / f"audiovae_decoder_np{args.num_patches}_meta_info.json"
    write_json_file(meta_file, meta)
    logger.info("Meta saved: %s", meta_file)
    logger.info("=" * 60)
    logger.info("AudioVAE Decoder export done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--num_patches", type=int, default=3,
        help="每次 decode 的 patch 数(default: 3,对应 streaming 每步输出)",
    )
    parser.add_argument("--quant_type", type=str, default="w8a8_sefp")
    parser.add_argument("--device", default=None, help="Export device: cpu, cuda, or cuda:N")
    parser.add_argument("--gen_golden", action="store_true")
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--verify_max_abs_tol", type=float, default=0.5)
    parser.add_argument("--verify_mean_abs_tol", type=float, default=0.05)
    parser.add_argument("--verify_cosine_tol", type=float, default=0.99)
    parser.add_argument("--verify_fail_on_mismatch", action="store_true")
    args = parser.parse_args()
    main(args)
