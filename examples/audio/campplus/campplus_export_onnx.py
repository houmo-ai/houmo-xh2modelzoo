# -*- coding: utf-8 -*-
# Export CAM++ speaker embedding model to dynamic ONNX and fixed-shape ONNX.

import argparse
import os
import os.path as osp

import numpy as np
import onnx
import onnxruntime as ort
import torch

import campplus_components
from campplus_model import CAMPPlus

HERE = osp.dirname(osp.abspath(__file__))

DEFAULT_BIN_PATH = "/data01/nfs_shared/ASR_TTS/CAM++/campplus_cn_common.bin"
DEFAULT_FIXED_T = 1000
FEAT_DIM = 80
EMB_SIZE = 192
OPSET = 14


def default_dynamic_onnx_path():
    return osp.join(HERE, "onnx", "campplus.onnx")


def default_fixed_onnx_path(fixed_t):
    return osp.join(HERE, "onnx", f"campplus_{fixed_t}.onnx")


def build_model(bin_path):
    model = CAMPPlus(
        feat_dim=FEAT_DIM, embedding_size=EMB_SIZE, growth_rate=32, bn_size=4,
        init_channels=128, config_str="batchnorm-relu",
        memory_efficient=False, output_level="segment",
    )
    sd = torch.load(bin_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert not missing and not unexpected, (missing, unexpected)
    model.eval()
    print(f"[load] strict OK | params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    return model


def _save_and_check(path):
    m = onnx.load(path)
    onnx.checker.check_model(m)
    ishape = [[d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim] for i in m.graph.input]
    print(f"[export] {osp.basename(path)} ({osp.getsize(path)/1e6:.2f} MB) | checker OK | in={ishape}")


def export_dynamic(model, output_path):
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    campplus_components.SEG_POOLING_ONNX_MODE = "interpolate"
    dummy = torch.randn(1, 200, FEAT_DIM)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["feats"], output_names=["embedding"],
        dynamic_axes={"feats": {0: "batch", 1: "time"}, "embedding": {0: "batch"}},
        opset_version=OPSET, do_constant_folding=True,
    )
    _save_and_check(output_path)


def export_fixed(model, fixed_t, output_path):
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    campplus_components.SEG_POOLING_ONNX_MODE = "expand"
    dummy = torch.randn(1, fixed_t, FEAT_DIM)
    torch.onnx.export(
        model, dummy, output_path,
        input_names=["feats"], output_names=["embedding"],
        dynamic_axes=None,
        opset_version=OPSET, do_constant_folding=True,
    )
    _save_and_check(output_path)


def verify(model, fixed_t, dynamic_path, fixed_path):
    print("== parity: dynamic onnx (interpolate) vs torch ==")
    sess = ort.InferenceSession(dynamic_path, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name
    campplus_components.SEG_POOLING_ONNX_MODE = "interpolate"
    worst = 0.0
    for t in (137, 200, 311):
        x = np.random.randn(1, t, FEAT_DIM).astype(np.float32)
        with torch.no_grad():
            yt = model(torch.from_numpy(x)).numpy()
        yo = sess.run(None, {iname: x})[0]
        d = float(np.max(np.abs(yt - yo)))
        worst = max(worst, d)
        print(f"  T={t:4d}  max|Δ|={d:.2e}")
    print(f"  -> {'PASS' if worst < 1e-4 else 'FAIL'} (worst {worst:.2e})")

    print(f"== parity: fixed-{fixed_t} onnx (expand) vs torch ==")
    sessf = ort.InferenceSession(fixed_path, providers=["CPUExecutionProvider"])
    inf = sessf.get_inputs()[0].name
    campplus_components.SEG_POOLING_ONNX_MODE = "expand"
    x = np.random.randn(1, fixed_t, FEAT_DIM).astype(np.float32)
    with torch.no_grad():
        yt = model(torch.from_numpy(x)).numpy()
    yo = sessf.run(None, {inf: x})[0]
    d = float(np.max(np.abs(yt - yo)))
    print(f"  T={fixed_t}  max|Δ|={d:.2e} -> {'PASS' if d < 1e-4 else 'FAIL'}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export CAMPPlus dynamic ONNX and fixed-shape ONNX")
    parser.add_argument("--bin-path", default=DEFAULT_BIN_PATH, help="PyTorch .bin checkpoint path")
    parser.add_argument("--fixed-t", type=int, default=DEFAULT_FIXED_T, help="Fixed fbank frame length")
    parser.add_argument("--dynamic-onnx", default=default_dynamic_onnx_path(), help="Output dynamic ONNX path")
    parser.add_argument("--fixed-onnx", default=None, help="Output fixed ONNX path; defaults to onnx/campplus_${fixed_t}.onnx")
    args = parser.parse_args(argv)
    if args.fixed_t <= 0:
        raise ValueError("--fixed-t must be positive")

    fixed_onnx = args.fixed_onnx or default_fixed_onnx_path(args.fixed_t)
    model = build_model(args.bin_path)
    export_dynamic(model, args.dynamic_onnx)
    export_fixed(model, args.fixed_t, fixed_onnx)
    verify(model, args.fixed_t, args.dynamic_onnx, fixed_onnx)


if __name__ == "__main__":
    main()
