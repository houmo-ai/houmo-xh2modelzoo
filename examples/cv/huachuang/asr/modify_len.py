import onnx


def keep_only_outputs(in_onnx: str, out_onnx: str, keep_outputs):
    model = onnx.load(in_onnx)
    g = model.graph

    keep_set = set(keep_outputs)

    kept = [o for o in g.output if o.name in keep_set]
    missing = [name for name in keep_outputs if name not in {o.name for o in g.output}]

    del g.output[:]
    g.output.extend(kept)

    onnx.checker.check_model(model)
    onnx.save(model, out_onnx)

    print("[OK] saved:", out_onnx)
    if missing:
        print("Warning: requested outputs not found in original graph.output:", missing)
    print("Kept outputs:", [o.name for o in kept])


if __name__ == "__main__":
    keep_only_outputs("weights/huachuang/encoder_fix_len_sim.onnx", "weights/huachuang/encoder_fix_len_sim.onnx", keep_outputs=["enc"])