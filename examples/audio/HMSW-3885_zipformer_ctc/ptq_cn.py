#!/usr/bin/env python3
"""
中文流式 ASR 模型导出 hmonnx 脚本
使用真实 wav 文件数据进行校准
"""
import os
import argparse
from pathlib import Path
import numpy as np
import torch
import onnx
import onnxruntime as ort
import soundfile as sf
import kaldi_native_fbank as knf
from scipy.signal import resample_poly
from xhquant.api import (
    DeviceType,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)


def load_audio_mono_16k(wav_path, target_sr=16000):
    """加载音频并重采样到 16k"""
    waveform, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=1)
    if sample_rate != target_sr:
        gcd = np.gcd(sample_rate, target_sr)
        waveform = resample_poly(
            waveform,
            up=target_sr // gcd,
            down=sample_rate // gcd,
        ).astype(np.float32)
        sample_rate = target_sr
    return waveform.astype(np.float32), sample_rate


def extract_fbank(waveform, sample_rate=16000, num_mel_bins=80):
    """提取 fbank 特征"""
    opts = knf.FbankOptions()
    opts.frame_opts.dither = 0
    opts.frame_opts.snip_edges = False
    opts.frame_opts.samp_freq = sample_rate
    opts.mel_opts.num_bins = num_mel_bins

    fbank = knf.OnlineFbank(opts)
    fbank.accept_waveform(sample_rate, waveform.tolist())
    fbank.input_finished()
    return np.array(
        [fbank.get_frame(i) for i in range(fbank.num_frames_ready)],
        dtype=np.float32,
    )


def create_zero_states(input_desc):
    """根据 ONNX 模型初始化空状态"""
    states = {}
    for item in input_desc:
        if item.name == "x":
            continue
        shape = []
        for dim in item.shape:
            if isinstance(dim, str):
                shape.append(1)
            else:
                shape.append(dim)
        dtype = np.float32 if item.type == "tensor(float)" else np.int64
        states[item.name] = np.zeros(shape, dtype=dtype)
    return states


def get_inputs_from_onnx(onnx_path):
    """从 ONNX 模型获取输入信息"""
    model = onnx.load(onnx_path)
    inputs = []
    for inp in model.graph.input:
        name = inp.name
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            if dim.dim_value > 0:
                shape.append(dim.dim_value)
            else:
                shape.append(1)
        dtype = np.float32
        if inp.type.tensor_type.elem_type == onnx.TensorProto.INT64:
            dtype = np.int64
        elif inp.type.tensor_type.elem_type == onnx.TensorProto.INT32:
            dtype = np.int32
        inputs.append({"name": name, "shape": shape, "dtype": dtype})
    return inputs


def get_outputs_from_onnx(onnx_path):
    """从 ONNX 模型获取输出名称"""
    model = onnx.load(onnx_path)
    return [out.name for out in model.graph.output]


def prepare_real_data_cn(wav_path, encoder_path, chunk_frames=45):
    """
    从真实 wav 文件准备校准数据
    返回 encoder_inputs 和输入输出名称 (PyTorch tensors)
    """
    print(f"Loading calibration audio: {wav_path}")

    # 加载音频
    waveform, sample_rate = load_audio_mono_16k(wav_path, target_sr=16000)

    # 提取 fbank 特征
    fbank = extract_fbank(waveform, sample_rate=sample_rate, num_mel_bins=80)
    print(f"Fbank shape: {fbank.shape}")

    # 创建 ONNX session
    sess_opt = ort.SessionOptions()
    providers = ["CPUExecutionProvider"]
    enc_sess = ort.InferenceSession(str(encoder_path), sess_opt, providers=providers)

    # 获取 encoder 输入信息
    enc_inputs = enc_sess.get_inputs()

    # 取第一个 chunk 作为校准数据
    chunk = fbank[:chunk_frames]
    if chunk.shape[0] < chunk_frames:
        padding = np.repeat(chunk[-1:], chunk_frames - chunk.shape[0], axis=0)
        chunk = np.concatenate([chunk, padding], axis=0)

    x = np.expand_dims(chunk, axis=0)

    # 初始化状态
    states = create_zero_states(enc_inputs)
    output_names = [o.name for o in enc_sess.get_outputs()]

    # 准备输入列表 (转换为 PyTorch tensors)
    input_names = [inp.name for inp in enc_inputs]
    inputs_list = []
    for inp in enc_inputs:
        if inp.name == "x":
            tensor = torch.from_numpy(x.copy())
        else:
            tensor = torch.from_numpy(states[inp.name].copy())
        if tensor.dtype == torch.int64:
            tensor = tensor.to(torch.int32)
        inputs_list.append(tensor)

    return {
        "encoder_inputs": inputs_list,
        "encoder_input_names": input_names,
        "encoder_output_names": output_names,
    }


def export_ctc_encoder(encoder_path, work_dir, quant_type, real_inputs, debug=False):
    """导出 CTC encoder 模型"""
    print(f"\n{'='*60}")
    print(f"Exporting CTC Encoder: {encoder_path}")
    print(f"{'='*60}")

    onnx_name = Path(encoder_path).stem
    out_dir = work_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    out_hmonnx = out_dir / f"{onnx_name}_XH2a.onnx"

    input_names = real_inputs["encoder_input_names"]
    output_names = real_inputs["encoder_output_names"]
    dummy_inputs = real_inputs["encoder_inputs"]

    print(f"Inputs ({len(input_names)}): {input_names[:5]}...")
    print(f"Outputs ({len(output_names)}): {output_names[:3]}...")
    print(f"Input x shape: {dummy_inputs[0].shape}")

    # 配置量化
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    print(f"Quantization config: {quant_config}")
    # 转换
    convert_onnx_to_hmonnx(
        encoder_path,
        dummy_inputs,
        DeviceType.XH2a,
        str(out_hmonnx),
        quant_config=quant_config,
        input_names=input_names,
        output_names=output_names,
    )

    logger = get_root_logger()
    logger.info(f"CTC Encoder saved to: {out_hmonnx}")
    print(f"CTC Encoder saved to: {out_hmonnx}")
    return out_hmonnx


def main():
    parser = argparse.ArgumentParser(
        description="中文流式 ASR 模型导出 hmonnx (使用真实数据校准)"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13",
        help="模型目录",
    )
    parser.add_argument(
        "--wav_path",
        type=str,
        default=None,
        help="校准用 wav 文件路径，默认使用 data/cn 下第一个 wav",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="cndata/cn",
        help="音频数据目录 (当 --wav_path 未指定时使用)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="输出目录，默认为 work_dirs/<model_name>",
    )
    parser.add_argument(
        "--chunk_frames",
        type=int,
        default=45,
        help="每次送入 encoder 的特征帧数",
    )
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--quant-type", default="w8a8h1_sefp", help="量化类型，默认 w8a8h1_sefp"
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if args.output_dir:
        work_dir = Path(args.output_dir)
    else:
        work_dir = Path("work_dirs") / model_dir.name

    # 初始化 xhquant
    xhquant_init(None, debug=args.debug)

    # 查找 _sim.onnx 文件
    encoder_path = model_dir / "ctc-epoch-20-avg-1-chunk-16-left-128_sim.onnx"

    if not encoder_path.exists():
        raise FileNotFoundError(f"Model not found: {encoder_path}")

    # 查找校准用 wav 文件
    if args.wav_path:
        wav_path = Path(args.wav_path)
    else:
        data_dir = Path(args.data_dir)
        wav_list = sorted(data_dir.glob("*.wav"))
        if not wav_list:
            raise FileNotFoundError(f"No wav files found in {data_dir}")
        wav_path = wav_list[0]

    if not wav_path.exists():
        raise FileNotFoundError(f"Wav file not found: {wav_path}")

    # 准备真实数据
    real_inputs = prepare_real_data_cn(wav_path, encoder_path, args.chunk_frames)

    # 导出模型
    encoder_hmonnx = export_ctc_encoder(
        encoder_path, work_dir, args.quant_type, real_inputs, args.debug
    )

    print(f"\n{'='*60}")
    print("Export completed!")
    print(f"{'='*60}")
    print(f"CTC Encoder: {encoder_hmonnx}")


if __name__ == "__main__":
    main()
