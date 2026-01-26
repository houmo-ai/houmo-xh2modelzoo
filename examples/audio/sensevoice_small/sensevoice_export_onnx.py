import argparse
import os
from pathlib import Path
import torch

class ScaledLayerNorm(torch.nn.Module):
    def __init__(self, original_ln, scale=32.0):
        super().__init__()
        self.ln = original_ln
        self.register_buffer('inv_scale', torch.tensor(1.0 / scale))

    def forward(self, x):
        return self.ln(x * self.inv_scale)

def replace_layernorm_with_scaled(model, scale=32.0):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.LayerNorm):
            setattr(model, name, ScaledLayerNorm(module, scale))
        else:
            replace_layernorm_with_scaled(module, scale)

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-dir", type=str, default="/data01/nfs_shared/ASR_TTS/SenseVoiceSmall")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--out-dir", type=str, default="work_dirs/sensevoice_small/export_fp32")
    p.add_argument("--opset", type=int, default=14)
    p.add_argument("--static", action="store_true", help="Export with static shape")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    from funasr import AutoModel
    import torch

    from sensevoice_export_meta import rebuild_for_onnx

    model_dir = str(Path(args.model_dir).expanduser().resolve())
    out_dir = Path(args.out_dir).expanduser().resolve()
    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    model, kwargs = AutoModel.build_model(model=model_dir, device=args.device, trust_remote_code=False)
    model.eval()

    print("Applying Input Scaling to LayerNorms for FP16 stability...")
    replace_layernorm_with_scaled(model, scale=32.0)

    model_file = onnx_dir / "model.onnx"
    rebuilt_model = rebuild_for_onnx(
        model,
        max_seq_len=int(args.max_seq_len),
        device=str(args.device),
        dynamic=not args.static,
    )
    dummy_inputs = rebuilt_model.export_dummy_inputs()
    torch.onnx.export(
        rebuilt_model,
        dummy_inputs,
        str(model_file),
        verbose=bool(args.verbose),
        opset_version=int(args.opset),
        input_names=rebuilt_model.export_input_names(),
        output_names=rebuilt_model.export_output_names(),
        dynamic_axes=rebuilt_model.export_dynamic_axes(),
    )

    if args.static:
        try:
            import onnx
            from onnxsim import simplify
            print("Simplifying ONNX model for static shape...")
            # Use onnxsim to fold constants and remove dynamic ops
            # Load model first
            model_proto = onnx.load(str(model_file))
            model_sim, check = simplify(model_proto)
            if check:
                onnx.save(model_sim, str(model_file))
                print(f"Simplified model saved to {model_file}")
            else:
                print("Simplification check failed! Keeping original model.")
        except ImportError:
            print("onnxsim not installed, skipping simplification. Install with: pip install onnxsim")
        except Exception as e:
            print(f"Simplification failed with error: {e}")

    meta = {
        "model_dir": model_dir,
        "device": args.device,
        "max_seq_len": args.max_seq_len,
        "opset": args.opset,
        "onnx": str(model_file),
    }
    (onnx_dir / "export_meta.json").write_text(
        __import__("json").dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Exported ONNX: {model_file}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
