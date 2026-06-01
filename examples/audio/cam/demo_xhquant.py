import argparse
import os
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchaudio

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from xhquant.api import DeviceType, QuantScheme, convert_onnx_to_hmonnx, create_quant_config, get_root_logger, xhquant_init
from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[2]
MODEL_DIR = "/data02/datasets/funasr/CAM"
DEFAULT_ONNX_PATH = THIS_DIR / "cam_embedding.onnx"
DEFAULT_SIMPLIFIED_ONNX_PATH = THIS_DIR / "cam_embedding_simplified.onnx"
DEFAULT_WORK_DIR = REPO_ROOT / "work_dirs" / "cam"
FEATURE_DIM = 80
DEFAULT_FIXED_FRAMES = 529
DEFAULT_THRESHOLD = 0.31
EXAMPLE_WAVS = [
    f"{MODEL_DIR}/examples/speaker1_a_cn_16k.wav",
    f"{MODEL_DIR}/examples/speaker1_b_cn_16k.wav",
    f"{MODEL_DIR}/examples/speaker2_a_cn_16k.wav",
]

torch.manual_seed(42)


def inspect_onnx(model_path):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type}, shape={inp.shape}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type}, shape={out.shape}")


def simplify_onnx_if_needed(logger, onnx_path: Path, simplified_onnx_path: Path):
    if simplified_onnx_path.exists():
        return

    import onnxsim

    logger.info("Simplifying ONNX model...")
    model = onnx.load(onnx_path)
    model_simplified, check = onnxsim.simplify(model)
    onnx.save(model_simplified, simplified_onnx_path)
    logger.info(f"Simplified ONNX saved to {simplified_onnx_path}, check={check}")


def load_feature_extractor():
    from modelscope.pipelines import pipeline

    sv_pipeline = pipeline(
        "speaker-verification",
        model=MODEL_DIR,
        model_revision="v1.0.0",
    )
    return sv_pipeline.model


def extract_feature(audio_path, feature_extractor, fixed_frames):
    waveform, sample_rate = torchaudio.load(audio_path)
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

    feature = feature_extractor._SpeakerVerificationCAMPPlus__extract_feature(
        waveform.squeeze(0).unsqueeze(0)
    ).float()
    frames = feature.shape[1]
    if frames < fixed_frames:
        feature = F.pad(feature, (0, 0, 0, fixed_frames - frames))
    else:
        feature = feature[:, :fixed_frames, :]
    return feature


def normalize_embedding(embedding):
    tensor = torch.from_numpy(embedding) if isinstance(embedding, np.ndarray) else embedding
    return F.normalize(tensor, p=2, dim=1)


def cosine_similarity(emb1, emb2):
    emb1 = np.asarray(emb1)
    emb2 = np.asarray(emb2)
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))


def run_fp(features, feature_extractor):
    device = next(feature_extractor.embedding_model.parameters()).device
    embeddings = []
    with torch.no_grad():
        for feature in features:
            embedding = feature_extractor.embedding_model(feature.to(device))
            embedding = normalize_embedding(embedding).cpu().numpy()
            embeddings.append(embedding)
    return embeddings


def run_onnx(features, simplified_onnx_path: Path):
    session = ort.InferenceSession(str(simplified_onnx_path), providers=["CPUExecutionProvider"])
    embeddings = []
    for feature in features:
        embedding = session.run(None, {"feature": feature.numpy()})[0]
        embeddings.append(normalize_embedding(embedding).cpu().numpy())
    return embeddings


def run_hmonnx(features, hmonnx_path, exec_device, save_golden_dir=None):
    session = HMONNXInference(hmonnx_path)
    session.exec_device = exec_device
    session.to(exec_device)
    if save_golden_dir is not None:
        session.save_golden = True
        session.save_golden_dir = str(save_golden_dir)

    embeddings = []
    with torch.no_grad():
        for feature in features:
            embedding = session.forward(feature.half().to(exec_device))
            if isinstance(embedding, (list, tuple)):
                embedding = embedding[0]
            embeddings.append(normalize_embedding(embedding.float().cpu()).cpu().numpy())
    return embeddings


def report_scores(logger, name, embeddings, threshold):
    sim_same = cosine_similarity(embeddings[0][0], embeddings[1][0])
    sim_diff = cosine_similarity(embeddings[0][0], embeddings[2][0])
    logger.info(f"{name} similarity (same speaker): {sim_same}")
    logger.info(f"{name} similarity (different speaker): {sim_diff}")
    logger.info(f"{name} threshold at {threshold}: same={sim_same > threshold}, diff={sim_diff > threshold}")
    return sim_same, sim_diff


def max_abs_diff(lhs, rhs):
    return float(np.max(np.abs(lhs - rhs)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-path", type=Path, default=DEFAULT_ONNX_PATH)
    parser.add_argument("--simplified-onnx-path", type=Path, default=DEFAULT_SIMPLIFIED_ONNX_PATH)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--quant-type", type=str, default="w8a8_sefp")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--fixed-frames", type=int, default=DEFAULT_FIXED_FRAMES)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--save-golden", action="store_true")
    parser.add_argument("--no-eval", action="store_true")
    parser.add_argument("--inspect-onnx", action="store_true")
    args = parser.parse_args()

    xhquant_init(None, debug=False)
    logger = get_root_logger()
    args.work_dir.mkdir(exist_ok=True, parents=True)

    if not args.onnx_path.exists():
        raise FileNotFoundError(f"Missing ONNX model: {args.onnx_path}")

    if args.inspect_onnx:
        inspect_onnx(str(args.onnx_path))

    simplify_onnx_if_needed(logger, args.onnx_path, args.simplified_onnx_path)

    target_device = DeviceType.XH2a
    quant_scheme = QuantScheme(target_device=target_device, quant_type=args.quant_type)
    quant_config = create_quant_config(quant_scheme)
    onnx_name = args.onnx_path.stem
    hmonnx_path = args.work_dir / "hmonnx" / f"{onnx_name}_{target_device}_{args.quant_type}.onnx"
    hmonnx_path.parent.mkdir(exist_ok=True, parents=True)

    if not hmonnx_path.exists():
        logger.info(f"Converting ONNX to hmonnx: {hmonnx_path}")
        dummy_input = torch.randn(1, args.fixed_frames, FEATURE_DIM)
        convert_onnx_to_hmonnx(
            str(args.simplified_onnx_path),
            (dummy_input,),
            target_device,
            str(hmonnx_path),
            quant_config=quant_config,
            input_names=["feature"],
            output_names=["embedding"],
        )

    if args.no_eval and not args.save_golden:
        return

    feature_extractor = load_feature_extractor()
    features = [extract_feature(path, feature_extractor, args.fixed_frames) for path in EXAMPLE_WAVS]

    if not args.no_eval:
        fp_embeddings = run_fp(features, feature_extractor)
        onnx_embeddings = run_onnx(features, args.simplified_onnx_path)
        hmonnx_embeddings = run_hmonnx(features, str(hmonnx_path), args.device)

        logger.info("Evaluation on fixed-length CAM features")
        report_scores(logger, "FP", fp_embeddings, args.threshold)
        report_scores(logger, "ONNX", onnx_embeddings, args.threshold)
        report_scores(logger, "HMONNX", hmonnx_embeddings, args.threshold)

        logger.info(
            "Embedding max abs diff: fp-vs-onnx=%s, fp-vs-hmonnx=%s",
            max_abs_diff(fp_embeddings[0], onnx_embeddings[0]),
            max_abs_diff(fp_embeddings[0], hmonnx_embeddings[0]),
        )

    if args.save_golden:
        golden_dir = args.work_dir / "hmonnx" / f"{onnx_name}_golden"
        run_hmonnx(features, str(hmonnx_path), args.device, save_golden_dir=golden_dir)
        logger.info(f"Golden saved to {golden_dir}")


if __name__ == "__main__":
    main()