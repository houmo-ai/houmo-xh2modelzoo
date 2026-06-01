import os
import sys
import argparse
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from xhquant.api import (
    Config,
    ConfigDict,
    DeviceType,
    FrontendType,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    export_onnx,
    get_root_logger,
    ptq_quantize,
    to_export_graph,
    to_frontend_graph,
    to_quant_graph,
    xhquant_init,
)
from xhquant.common.types import DeviceType, FrontendType, PrecisionMode
from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

torch.manual_seed(42)

MODEL_DIR = "/data02/datasets/speech_eres2net_large_sv_zh-cn_3dspeaker_16k"
ONNX_PATH = "/data01/home/xuchen/xh2/xh2_model_zoo/examples/audio/ERes2Net/eres2net_embedding.onnx"
SIMPLIFIED_ONNX_PATH = "/data01/home/xuchen/xh2/xh2_model_zoo/examples/audio/ERes2Net/eres2net_embedding_simplified.onnx"
WORK_DIRS = Path("/data01/home/xuchen/xh2/xh2_model_zoo/work_dirs/eres2net")
WORK_DIRS.mkdir(exist_ok=True, parents=True)


def inspect_onnx(model_path):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type}, shape={inp.shape}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type}, shape={out.shape}")


def load_audio(path, target_sr=16000, fixed_length=16000):
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        if sr != target_sr:
            waveform = torchaudio.functional.resample(waveform, sr, target_sr)
        audio = waveform.squeeze(0).numpy()
    except ImportError:
        from scipy.io import wavfile
        sr, waveform = wavfile.read(path)
        if sr != target_sr:
            print(f"Warning: audio sample rate {sr} != {target_sr}")
        audio = waveform.flatten() / 32768.0

    if len(audio) < fixed_length:
        audio = np.pad(audio, (0, fixed_length - len(audio)))
    else:
        audio = audio[:fixed_length]
    return audio


def extract_feature(audio, num_mel_bins=80):
    from modelscope.models.audio.sv.ERes2Net import Kaldi
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    feature = Kaldi.fbank(audio, num_mel_bins=num_mel_bins)
    feature = feature - feature.mean(dim=0, keepdim=True)
    return feature


def postprocess_embedding(embedding):
    embedding = torch.from_numpy(embedding) if isinstance(embedding, np.ndarray) else embedding
    embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
    return embedding


def calculate_similarity(emb1, emb2):
    cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    return cos_sim


def main():
    import onnxsim
    from modelscope.models.audio.sv.ERes2Net import Kaldi

    args = argparse.Namespace(
        quant_type="w8a8_sefp",
        device="cuda:0",
        save_golden=True,
        no_eval=False,
    )

    xhquant_init(None, debug=False)
    logger = get_root_logger()

    target_device = DeviceType.XH2a
    onnx_name = Path(ONNX_PATH).stem
    quant_type = args.quant_type

    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    hmonnx_path = WORK_DIRS / "hmonnx" / f"{onnx_name}_{target_device}_{quant_type}.onnx"
    hmonnx_path.parent.mkdir(exist_ok=True, parents=True)
    hmonnx_path = str(hmonnx_path)

    if not os.path.exists(SIMPLIFIED_ONNX_PATH):
        logger.info("Simplifying ONNX model...")
        model = onnx.load(ONNX_PATH)
        model_simplified, check = onnxsim.simplify(model)
        onnx.save(model_simplified, SIMPLIFIED_ONNX_PATH)
        logger.info(f"Simplified ONNX saved to {SIMPLIFIED_ONNX_PATH}, check={check}")

    if not os.path.exists(hmonnx_path):
        logger.info(f"Converting ONNX to hmonnx: {hmonnx_path}")
        dummy_input = torch.randn(1, 98, 80)
        convert_onnx_to_hmonnx(
            SIMPLIFIED_ONNX_PATH,
            (dummy_input,),
            target_device,
            hmonnx_path,
            quant_config=quant_config,
            input_names=["feature"],
            output_names=["embedding"],
        )

    session = HMONNXInference(hmonnx_path)

    if args.save_golden:
        logger.info("Saving golden data...")
        session.save_golden = True
        session.save_golden_dir = str(WORK_DIRS / "hmonnx" / f"{onnx_name}_golden")

        speaker1_a_wav = f"{MODEL_DIR}/examples/speaker1_a_cn_16k.wav"
        speaker1_b_wav = f"{MODEL_DIR}/examples/speaker1_b_cn_16k.wav"

        audio1 = load_audio(speaker1_a_wav)
        audio2 = load_audio(speaker1_b_wav)

        feat1 = extract_feature(audio1)
        feat2 = extract_feature(audio2)

        feat_tensor1 = feat1.unsqueeze(0) if isinstance(feat1, torch.Tensor) else torch.from_numpy(feat1).unsqueeze(0)
        feat_tensor2 = feat2.unsqueeze(0) if isinstance(feat2, torch.Tensor) else torch.from_numpy(feat2).unsqueeze(0)

        session.exec_device = args.device
        session.to(args.device)

        with torch.no_grad():
            emb1 = session.forward(feat_tensor1.half().to(args.device))
            emb2 = session.forward(feat_tensor2.half().to(args.device))

            if isinstance(emb1, (tuple, list)):
                emb1 = emb1[0]
                emb2 = emb2[0]

            emb1 = postprocess_embedding(emb1.float().cpu())
            emb2 = postprocess_embedding(emb2.float().cpu())

            emb1_np = emb1.cpu().numpy()
            emb2_np = emb2.cpu().numpy()

        logger.info(f"Golden saved to {session.save_golden_dir}")

    if not args.no_eval:
        logger.info("Running evaluation...")

        speaker1_a_wav = f"{MODEL_DIR}/examples/speaker1_a_cn_16k.wav"
        speaker1_b_wav = f"{MODEL_DIR}/examples/speaker1_b_cn_16k.wav"
        speaker2_a_wav = f"{MODEL_DIR}/examples/speaker2_a_cn_16k.wav"

        audio1 = load_audio(speaker1_a_wav)
        audio2 = load_audio(speaker1_b_wav)
        audio3 = load_audio(speaker2_a_wav)

        feat1 = extract_feature(audio1)
        feat2 = extract_feature(audio2)
        feat3 = extract_feature(audio3)

        feat_tensors = []
        for feat in [feat1, feat2, feat3]:
            if isinstance(feat, torch.Tensor):
                feat_tensors.append(feat.unsqueeze(0))
            else:
                feat_tensors.append(torch.from_numpy(feat).unsqueeze(0))

        session.exec_device = args.device
        session.to(args.device)

        embeddings = []
        with torch.no_grad():
            for feat_tensor in feat_tensors:
                emb = session.forward(feat_tensor.half().to(args.device))
                if isinstance(emb, (tuple, list)):
                    emb = emb[0]
                emb = postprocess_embedding(emb.float().cpu())
                embeddings.append(emb.cpu().numpy())

        sim_same = calculate_similarity(embeddings[0][0], embeddings[1][0])
        sim_diff = calculate_similarity(embeddings[0][0], embeddings[2][0])

        logger.info(f"Similarity (same speaker): {sim_same}")
        logger.info(f"Similarity (different speaker): {sim_diff}")
        logger.info(f"Threshold at 0.262: same={sim_same > 0.262}, diff={sim_diff > 0.262}")


if __name__ == "__main__":
    main()