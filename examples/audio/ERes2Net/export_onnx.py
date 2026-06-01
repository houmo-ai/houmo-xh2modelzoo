import os
import argparse
import torch
import onnx
import onnxruntime as ort

MODEL_DIR = "/data02/datasets/speech_eres2net_large_sv_zh-cn_3dspeaker_16k"
OUTPUT_ONNX_PATH = "/data01/home/xuchen/xh2/xh2_model_zoo/examples/audio/ERes2Net/eres2net_embedding.onnx"
OUTPUT_SIMPLIFIED_PATH = "/data01/home/xuchen/xh2/xh2_model_zoo/examples/audio/ERes2Net/eres2net_embedding_simplified.onnx"
os.makedirs(os.path.dirname(OUTPUT_ONNX_PATH), exist_ok=True)


def inspect_onnx(model_path):
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    print("Model inputs:")
    for inp in session.get_inputs():
        print(f"  {inp.name}: {inp.type}, shape={inp.shape}")
    print("Model outputs:")
    for out in session.get_outputs():
        print(f"  {out.name}: {out.type}, shape={out.shape}")


def export_onnx():
    from modelscope.pipelines import pipeline
    sv_pipeline = pipeline(
        'speaker-verification',
        model=MODEL_DIR,
    )
    embedding_model = sv_pipeline.model.embedding_model

    embedding_model = embedding_model.cpu().float()
    embedding_model.eval()

    dummy_input = torch.randn(1, 98, 80)

    torch.onnx.export(
        embedding_model,
        dummy_input,
        OUTPUT_ONNX_PATH,
        input_names=["feature"],
        output_names=["embedding"],
        opset_version=14,
    )

    print(f"Exported ONNX model to {OUTPUT_ONNX_PATH}")

    onnx_model = onnx.load(OUTPUT_ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model check passed")

    inspect_onnx(OUTPUT_ONNX_PATH)


def simplify_onnx():
    import onnxsim

    model = onnx.load(OUTPUT_ONNX_PATH)
    model_simplified, check = onnxsim.simplify(model)
    onnx.save(model_simplified, OUTPUT_SIMPLIFIED_PATH)
    print(f"Simplified ONNX model saved to {OUTPUT_SIMPLIFIED_PATH}, check={check}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=str, default="all", choices=["export", "simplify", "all"])
    args = parser.parse_args()

    if args.step in ["export", "all"]:
        export_onnx()

    if args.step in ["simplify", "all"]:
        simplify_onnx()