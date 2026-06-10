# -*- coding: utf-8 -*-
# Export fixed-shape masked CAMPPlus ONNX and/or HMONNX.

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
DEFAULT_QUANT_TYPE = "w8a8_sefp"
DEFAULT_TEST_WAVS = [
    osp.join(HERE, "test_wavs", "zero_shot_prompt.wav"),
    osp.join(HERE, "test_wavs", "xiaotian_chunk_000.wav"),
]
FEAT_DIM = 80
EMB_SIZE = 192
OPSET = 14


def default_dynamic_onnx_path():
    return osp.join(HERE, "onnx", "campplus.onnx")


def default_masked_onnx_path(fixed_t):
    return osp.join(HERE, "onnx", f"campplus_masked_{fixed_t}.onnx")


def default_simplified_path(fixed_t):
    if fixed_t == DEFAULT_FIXED_T:
        return osp.join(HERE, "onnx", "campplus_masked_simplify.onnx")
    return osp.join(HERE, "onnx", f"campplus_masked_{fixed_t}_simplify.onnx")


def default_output_root():
    return osp.join(HERE, "campplus")


def default_output_file(output_root, fixed_t, quant_type):
    if fixed_t == DEFAULT_FIXED_T:
        output_dir = osp.join(output_root, "campplus_masked", "prefill")
    else:
        output_dir = osp.join(output_root, f"campplus_masked_{fixed_t}", "prefill")
    return osp.join(output_dir, f"hmquant_xh2_campplus_masked_{quant_type}_{fixed_t}.onnx")


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


def make_feat_mask(valid_feat_t, fixed_t):
    valid_t = min(int(valid_feat_t), int(fixed_t))
    mask = torch.zeros(1, 1, fixed_t, dtype=torch.float32)
    mask[:, :, :valid_t] = 1.0
    return mask


def make_mask(valid_feat_t, fixed_t):
    mask_t = (fixed_t + 1) // 2
    valid_mask_t = (min(int(valid_feat_t), int(fixed_t)) + 1) // 2
    mask = torch.zeros(1, 1, mask_t, dtype=torch.float32)
    mask[:, :, :valid_mask_t] = 1.0
    return mask


def pad_or_crop_feats(feats, fixed_t):
    if feats.shape[1] >= fixed_t:
        return feats[:, :fixed_t, :]
    return torch.nn.functional.pad(feats, (0, 0, 0, fixed_t - feats.shape[1]))


def _test_lengths(fixed_t):
    return [t for t in [137, 200, 311, fixed_t - 1, fixed_t] if t > 0]


def _save_and_check(path):
    model = onnx.load(path)
    onnx.checker.check_model(model)
    ishape = [[d.dim_param or d.dim_value for d in i.type.tensor_type.shape.dim] for i in model.graph.input]
    print(f"[export] {osp.basename(path)} ({osp.getsize(path)/1e6:.2f} MB) | checker OK | in={ishape}")


def verify_padding_invariance(model, fixed_t):
    print(f"== parity: masked native length vs padded-{fixed_t} ==")
    campplus_components.SEG_POOLING_ONNX_MODE = "expand"
    worst = 0.0
    for t in _test_lengths(fixed_t):
        valid_t = min(t, fixed_t)
        x = torch.randn(1, valid_t, FEAT_DIM)
        x_pad = pad_or_crop_feats(x, fixed_t)
        with torch.no_grad():
            y_native = model(x, make_feat_mask(valid_t, valid_t), make_mask(valid_t, valid_t)).numpy()
            y_pad = model(x_pad, make_feat_mask(valid_t, fixed_t), make_mask(valid_t, fixed_t)).numpy()
        d = float(np.max(np.abs(y_native - y_pad)))
        worst = max(worst, d)
        print(f"  T={valid_t:4d}  max|Δ|={d:.2e}")
    print(f"  -> {'PASS' if worst < 1e-4 else 'FAIL'} (worst {worst:.2e})")


def export_masked_onnx(args):
    masked_onnx = args.masked_onnx or default_masked_onnx_path(args.fixed_t)
    model = build_model(args.bin_path)
    verify_padding_invariance(model, args.fixed_t)

    os.makedirs(osp.dirname(masked_onnx), exist_ok=True)
    campplus_components.SEG_POOLING_ONNX_MODE = "expand"
    mask_t = (args.fixed_t + 1) // 2
    feats = torch.randn(1, args.fixed_t, FEAT_DIM)
    feat_mask = torch.ones(1, 1, args.fixed_t)
    mask = torch.ones(1, 1, mask_t)
    torch.onnx.export(
        model, (feats, feat_mask, mask), masked_onnx,
        input_names=["feats", "feat_mask", "mask"], output_names=["embedding"],
        dynamic_axes=None, opset_version=OPSET, do_constant_folding=True,
    )
    _save_and_check(masked_onnx)
    verify_masked_onnx(model, args.fixed_t, masked_onnx)
    return masked_onnx


def verify_masked_onnx(model, fixed_t, onnx_path):
    print("== parity: torch masked vs onnxruntime masked ==")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    worst = 0.0
    for t in _test_lengths(fixed_t):
        valid_t = min(t, fixed_t)
        x = torch.randn(1, valid_t, FEAT_DIM)
        x_pad = pad_or_crop_feats(x, fixed_t)
        feat_mask = make_feat_mask(valid_t, fixed_t)
        mask = make_mask(valid_t, fixed_t)
        with torch.no_grad():
            yt = model(x_pad, feat_mask, mask).numpy()
        yo = sess.run(None, {
            names[0]: x_pad.numpy().astype(np.float32),
            names[1]: feat_mask.numpy().astype(np.float32),
            names[2]: mask.numpy().astype(np.float32),
        })[0]
        d = float(np.max(np.abs(yt - yo)))
        worst = max(worst, d)
        print(f"  T={valid_t:4d}  max|Δ|={d:.2e}")
    print(f"  -> {'PASS' if worst < 1e-4 else 'FAIL'} (worst {worst:.2e})")


def cosine(a, b):
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def load_fbank(path):
    import torchaudio
    import torchaudio.compliance.kaldi as Kaldi

    wav, sr = torchaudio.load(path)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
    if wav.shape[0] > 1:
        wav = wav[:1]
    feats = Kaldi.fbank(wav, num_mel_bins=FEAT_DIM, dither=0.0, sample_frequency=16000)
    feats = feats - feats.mean(dim=0, keepdim=True)
    return feats


def pack_masked_inputs(feats, fixed_t):
    valid_t = min(int(feats.shape[0]), fixed_t)
    x_pad = torch.zeros(1, fixed_t, FEAT_DIM, dtype=torch.float32)
    x_pad[0, :valid_t] = feats[:valid_t]
    feat_mask = make_feat_mask(valid_t, fixed_t)
    mask = make_mask(valid_t, fixed_t)
    return x_pad.numpy(), feat_mask.numpy(), mask.numpy(), valid_t


def to_numpy(output):
    if isinstance(output, (list, tuple)):
        output = output[0]
    if torch.is_tensor(output):
        return output.detach().cpu().float().numpy()
    return np.asarray(output, dtype=np.float32)


def run_real_wav_tests(args):
    dynamic_onnx = args.dynamic_onnx
    masked_onnx = args.masked_onnx or default_masked_onnx_path(args.fixed_t)
    output_root = args.output_root or default_output_root()
    hmonnx_file = args.output_file or default_output_file(output_root, args.fixed_t, args.quant_type)
    test_wavs = args.test_wav or DEFAULT_TEST_WAVS

    if not osp.exists(dynamic_onnx):
        raise FileNotFoundError(f"Missing dynamic ONNX: {dynamic_onnx}. Run campplus_export_onnx.py first.")
    if not osp.exists(masked_onnx):
        raise FileNotFoundError(f"Missing masked ONNX: {masked_onnx}. Run this script with --stage onnx first.")

    dyn = ort.InferenceSession(dynamic_onnx, providers=["CPUExecutionProvider"])
    masked = ort.InferenceSession(masked_onnx, providers=["CPUExecutionProvider"])
    dyn_in = dyn.get_inputs()[0].name
    masked_names = [i.name for i in masked.get_inputs()]

    hm = None
    if osp.exists(hmonnx_file):
        from xhquant.api import HMONNXGoldenInference

        hm = HMONNXGoldenInference(hmonnx_file)
        hm.save_golden = False
        hm.exec_device = torch.device("cuda:0")
        print(f"[hmonnx] enabled: {hmonnx_file}")
    else:
        print(f"[hmonnx] skip: missing {hmonnx_file}")

    print(f"== real wav test: dynamic ONNX vs masked ONNX/HMONNX padded-{args.fixed_t} ==")
    for wav_path in test_wavs:
        if not osp.exists(wav_path):
            print(f"[skip] missing wav: {wav_path}")
            continue

        feats = load_fbank(wav_path)
        x_pad, feat_mask, mask, valid_t = pack_masked_inputs(feats, args.fixed_t)
        e_masked = masked.run(None, {
            masked_names[0]: x_pad,
            masked_names[1]: feat_mask,
            masked_names[2]: mask,
        })[0]

        x_native = feats.unsqueeze(0).numpy().astype(np.float32)
        e_native = dyn.run(None, {dyn_in: x_native})[0]
        print(f"\nwav: {wav_path}")
        print(f"  fbank_t={feats.shape[0]} fixed_t={args.fixed_t} valid_t={valid_t} valid_mask_t={int(mask.sum())}")
        print(f"  cos(dynamic_full, masked_onnx)={cosine(e_native, e_masked):.8f}")
        print(f"  onnx_max_abs_diff={float(np.max(np.abs(e_native - e_masked))):.8e}")
        print(f"  onnx_mean_abs_diff={float(np.mean(np.abs(e_native - e_masked))):.8e}")

        e_ref = e_native
        if feats.shape[0] > args.fixed_t:
            x_crop = feats[:args.fixed_t].unsqueeze(0).numpy().astype(np.float32)
            e_ref = dyn.run(None, {dyn_in: x_crop})[0]
            print("  note: full utterance is longer than fixed_t, so masked path crops it.")
            print(f"  cos(dynamic_crop, masked_onnx)={cosine(e_ref, e_masked):.8f}")
            print(f"  crop_max_abs_diff={float(np.max(np.abs(e_ref - e_masked))):.8e}")

        if hm is not None:
            with torch.no_grad():
                e_hm = to_numpy(hm.forward(
                    torch.from_numpy(x_pad).to(torch.float16),
                    torch.from_numpy(feat_mask).to(torch.float16),
                    torch.from_numpy(mask).to(torch.float16),
                ))
            print(f"  cos(masked_onnx, hmonnx)={cosine(e_masked, e_hm):.8f}")
            print(f"  hmonnx_vs_onnx_max_abs_diff={float(np.max(np.abs(e_masked - e_hm))):.8e}")
            print(f"  hmonnx_vs_onnx_mean_abs_diff={float(np.mean(np.abs(e_masked - e_hm))):.8e}")
            print(f"  cos(dynamic_ref, hmonnx)={cosine(e_ref, e_hm):.8f}")


def inspect_onnx(model_path):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("模型输入节点:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type}, shape={inp.shape}")
    print("模型输出节点:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type}, shape={out.shape}")


def export_masked_hmonnx(args):
    from xhquant.api import (
        DeviceType,
        HMONNXGoldenInference,
        QuantScheme,
        convert_onnx_to_hmonnx,
        create_quant_config,
    )
    import onnxsim

    fixed_t = args.fixed_t
    mask_t = (fixed_t + 1) // 2
    model_path = args.masked_onnx or args.model_path or default_masked_onnx_path(fixed_t)
    simplified_path = args.simplified_path or default_simplified_path(fixed_t)
    output_root = args.output_root or default_output_root()
    output_file = args.output_file or default_output_file(output_root, fixed_t, args.quant_type)
    golden_dir = args.golden_dir or osp.join(osp.dirname(output_file), "step_0")

    if not osp.exists(model_path):
        raise FileNotFoundError(f"Missing masked ONNX: {model_path}. Run this script with --stage onnx first.")

    os.makedirs(osp.dirname(output_file), exist_ok=True)
    os.makedirs(golden_dir, exist_ok=True)
    inspect_onnx(model_path)

    model = onnx.load(model_path)
    model_simplified, _ = onnxsim.simplify(
        model,
        input_shapes={
            "feats": [1, fixed_t, 80],
            "feat_mask": [1, 1, fixed_t],
            "mask": [1, 1, mask_t],
        },
    )
    onnx.save(model_simplified, simplified_path)

    dummy_feats = torch.randn(1, fixed_t, 80)
    dummy_feat_mask = torch.ones(1, 1, fixed_t)
    dummy_mask = torch.ones(1, 1, mask_t)
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)

    if not osp.exists(output_file):
        convert_onnx_to_hmonnx(
            simplified_path,
            (dummy_feats, dummy_feat_mask, dummy_mask),
            out_hmonnx_file=output_file,
            device_type="XH2A",
            quant_config=quant_config,
        )

    model = HMONNXGoldenInference(output_file)
    model.save_golden = True
    model.exec_device = torch.device("cuda:0")
    model.golden_dir = str(golden_dir)
    with torch.no_grad():
        model.forward(dummy_feats.to(torch.float16), dummy_feat_mask.to(torch.float16), dummy_mask.to(torch.float16))


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export masked CAMPPlus ONNX/HMONNX")
    parser.add_argument("--stage", choices=("onnx", "hmonnx", "test", "all"), default="all")
    parser.add_argument("--fixed-t", type=int, default=DEFAULT_FIXED_T, help="Fixed fbank frame length")
    parser.add_argument("--bin-path", default=DEFAULT_BIN_PATH, help="PyTorch .bin checkpoint path")
    parser.add_argument("--dynamic-onnx", default=default_dynamic_onnx_path(), help="Input dynamic ONNX path for real wav tests")
    parser.add_argument("--masked-onnx", default=None, help="Output/input masked ONNX path")
    parser.add_argument("--model-path", default=None, help="Input masked ONNX path; alias for --masked-onnx in hmonnx stage")
    parser.add_argument("--simplified-path", default=None, help="Output simplified ONNX path")
    parser.add_argument("--output-root", default=None, help="HMONNX output root directory")
    parser.add_argument("--output-file", default=None, help="Output HMONNX file path")
    parser.add_argument("--golden-dir", default=None, help="Golden output directory")
    parser.add_argument("--quant-type", default=DEFAULT_QUANT_TYPE, help="Quant type")
    parser.add_argument("--test-wav", action="append", default=None, help="Real wav path for --stage test. Can be repeated.")
    args = parser.parse_args(argv)
    if args.fixed_t <= 0:
        raise ValueError("--fixed-t must be positive")

    if args.stage in ("onnx", "all"):
        export_masked_onnx(args)
    if args.stage in ("hmonnx", "all"):
        export_masked_hmonnx(args)
    if args.stage in ("test", "all"):
        run_real_wav_tests(args)


if __name__ == "__main__":
    main()
