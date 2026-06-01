"""VoxCPM2 LocDiT-Step 的 HMONNX 导出脚本。

用法:
    python export_locdit.py --model ~/models/VoxCPM2
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import onnx
import onnxsim
import torch
import torch.nn as nn

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
    """ONNX legacy exporter 不支持 enable_gqa=True，这里在导出时展开 GQA。"""
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


def _verify_hmonnx_vs_pytorch(
    hmonnx_file: Path,
    device: torch.device,
    x_cal: torch.Tensor,
    mu_cal: torch.Tensor,
    t_cal: torch.Tensor,
    cond_cal: torch.Tensor,
    dt_cal: torch.Tensor,
    torch_out: torch.Tensor,
    logger,
    *,
    max_abs_tol: float,
    mean_abs_tol: float,
    cosine_tol: float,
):
    session = HMONNXInference(str(hmonnx_file))
    session.to(str(device))
    hmonnx_out = _first_output(
        session.run(
            {
                "x": x_cal,
                "mu": mu_cal,
                "t": t_cal,
                "cond": cond_cal,
                "dt": dt_cal,
            }
        )
    )

    ref = torch_out.detach().float().reshape(-1).cpu()
    got = hmonnx_out.detach().float().reshape(-1).cpu()

    if ref.shape != got.shape:
        raise RuntimeError(f"Parity shape mismatch: pytorch={tuple(ref.shape)} hmonnx={tuple(got.shape)}")

    abs_diff = (got - ref).abs()
    max_abs = float(abs_diff.max().item())
    mean_abs = float(abs_diff.mean().item())
    cosine = float(torch.nn.functional.cosine_similarity(got.unsqueeze(0), ref.unsqueeze(0), dim=-1).item())

    logger.info(
        "Parity(LocDiT): max_abs=%.6f mean_abs=%.6f cosine=%.6f",
        max_abs,
        mean_abs,
        cosine,
    )
    passed = bool(max_abs <= max_abs_tol and mean_abs <= mean_abs_tol and cosine >= cosine_tol)
    return {
        "passed": passed,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "cosine": cosine,
    }


def main(args):
    model_path = str(Path(args.model).expanduser().resolve())
    model_name = Path(model_path).name
    target_device = "XH2a"

    script_dir = Path(__file__).resolve().parent
    work_root = script_dir / "work_dirs"
    work_dir = work_root / f"{model_name}_{target_device}" / "LocDiT"
    work_dir.mkdir(exist_ok=True, parents=True)
    logger = get_root_logger()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    logger.info("Loading VoxCPM2 from %s", model_path)
    voxcpm2 = VoxCPM2Model.from_local(model_path, optimize=False, training=False)
    voxcpm2.to(device)
    voxcpm2.eval()

    estimator = voxcpm2.feat_decoder.estimator
    estimator = estimator.to(device=device, dtype=dtype)
    estimator.eval()

    # --- shape 参数 ---
    patch_size = voxcpm2.patch_size                 # 4
    in_channels = estimator.in_channels             # 64 = feat_dim
    h_dit = voxcpm2.config.dit_config.hidden_dim     # 1024

    # CFG batch = 2B,B=1 → N=2
    N = 2

    # --- 构造校准输入 ---
    torch.manual_seed(0)
    x_cal = torch.randn((N, in_channels, patch_size), device=device, dtype=dtype) * 0.5
    mu_cal = torch.randn((N, 2 * h_dit), device=device, dtype=dtype) * 0.5
    t_cal = torch.tensor([0.5, 0.5], device=device, dtype=dtype)
    cond_cal = torch.randn((N, in_channels, patch_size), device=device, dtype=dtype) * 0.5
    dt_cal = torch.zeros((N,), device=device, dtype=dtype)

    logger.info("Calibration shapes: x=%s mu=%s t=%s cond=%s dt=%s",
                tuple(x_cal.shape), tuple(mu_cal.shape), tuple(t_cal.shape),
                tuple(cond_cal.shape), tuple(dt_cal.shape))

    # --- sanity check ---
    with torch.no_grad():
        v = estimator(x_cal, mu_cal, t_cal, cond_cal, dt_cal)
    logger.info("Estimator forward OK, output.shape=%s", tuple(v.shape))
    assert v.shape == x_cal.shape, f"LocDiT output shape mismatch: {v.shape} vs {x_cal.shape}"

    # --- 导出 ONNX ---
    onnx_file = work_dir / "voxcpm2_locdit_step.onnx"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_onnx = str(Path(tmp_dir) / onnx_file.name)

        with RewriterContext(None, backend="onnxruntime"):
            with patch("torch.nn.functional.scaled_dot_product_attention", _onnx_safe_scaled_dot_product_attention):
                torch.onnx.export(
                    estimator,
                    (x_cal, mu_cal, t_cal, cond_cal, dt_cal),
                    tmp_onnx,
                    input_names=["x", "mu", "t", "cond", "dt"],
                    output_names=["v"],
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
    hmonnx_file = work_dir / "hmonnx" / f"voxcpm2_locdit_step_xh2a_{args.quant_type}.onnx"
    hmonnx_file.parent.mkdir(exist_ok=True, parents=True)

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)

    logger.info("Converting to HMONNX: %s", hmonnx_file)
    convert_onnx_to_hmonnx(
        str(onnx_file),
        [x_cal, mu_cal, t_cal, cond_cal, dt_cal],
        DeviceType.XH2a,
        hmonnx_file,
        quant_config=quant_config,
        input_names=["x", "mu", "t", "cond", "dt"],
        output_names=["v"],
    )
    logger.info("HMONNX saved: %s", hmonnx_file)

    if not args.skip_verify:
        parity = _verify_hmonnx_vs_pytorch(
            hmonnx_file=hmonnx_file,
            device=device,
            x_cal=x_cal,
            mu_cal=mu_cal,
            t_cal=t_cal,
            cond_cal=cond_cal,
            dt_cal=dt_cal,
            torch_out=v,
            logger=logger,
            max_abs_tol=args.verify_max_abs_tol,
            mean_abs_tol=args.verify_mean_abs_tol,
            cosine_tol=args.verify_cosine_tol,
        )
        if parity["passed"]:
            logger.info("LocDiT parity check passed.")
        else:
            msg = (
                "LocDiT parity check failed: "
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
        session(x_cal, mu_cal, t_cal, cond_cal, dt_cal)
        logger.info("golden saved: %s", golden_dir)

    # --- meta ---
    meta = dict(
        create_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        model_name=model_name,
        target_device=target_device,
        module="LocDiT-Step",
        onnx_file=str(onnx_file.relative_to(work_dir.parent)),
        hmonnx_file=str(hmonnx_file.relative_to(work_dir.parent)),
        golden_dir=str(golden_dir.relative_to(work_dir.parent)) if args.gen_golden else None,
        quant_type=args.quant_type,
        input_dtype=str(dtype).replace("torch.", ""),
        input_names=["x", "mu", "t", "cond", "dt"],
        output_names=["v"],
        input_shapes=dict(
            x=[N, in_channels, patch_size],
            mu=[N, 2 * h_dit],
            t=[N],
            cond=[N, in_channels, patch_size],
            dt=[N],
        ),
        output_shapes=dict(v=[N, in_channels, patch_size]),
        # ODE 循环参数(给 host orchestrator 参考)
        inference_cfg_rate=voxcpm2.config.dit_config.cfm_config.inference_cfg_rate,
        mean_mode=voxcpm2.feat_decoder.mean_mode,
        cfg_batch_size=N,
        in_channels=in_channels,
        patch_size=patch_size,
        h_dit=h_dit,
    )
    meta_file = work_dir / "locdit_meta_info.json"
    write_json_file(meta_file, meta)
    logger.info("Meta saved: %s", meta_file)
    logger.info("=" * 60)
    logger.info("LocDiT export done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--quant_type", type=str, default="w8a8_sefp")
    parser.add_argument("--gen_golden", action="store_true")
    parser.add_argument("--skip_verify", action="store_true")
    parser.add_argument("--verify_max_abs_tol", type=float, default=1.0)
    parser.add_argument("--verify_mean_abs_tol", type=float, default=0.1)
    parser.add_argument("--verify_cosine_tol", type=float, default=0.95)
    parser.add_argument("--verify_fail_on_mismatch", action="store_true")
    args = parser.parse_args()
    main(args)
