import argparse
import os
from pathlib import Path

import onnx
import torch
from xhquant.api import (
    DeviceType,
    HMONNXGoldenInference,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)


def _torch_dtype_from_onnx(elem_type: int):
    import onnx

    mapping = {
        onnx.TensorProto.FLOAT: torch.float32,
        onnx.TensorProto.FLOAT16: torch.float16,
        onnx.TensorProto.DOUBLE: torch.float64,
        onnx.TensorProto.INT64: torch.int64,
        onnx.TensorProto.INT32: torch.int32,
        onnx.TensorProto.INT16: torch.int16,
        onnx.TensorProto.INT8: torch.int8,
        onnx.TensorProto.UINT8: torch.uint8,
        onnx.TensorProto.BOOL: torch.bool,
    }
    return mapping.get(elem_type, torch.float32)


def _parse_shape_str(s: str):
    """Parse shape like "1x3x224x224" into a tuple of ints."""
    return tuple(int(x) for x in s.replace(",", "x").split("x") if x)


def _parse_overrides(kvs):
    """Parse ["name=1x3x224x224", ...] into dict{name: tuple}"""
    out = {}
    if not kvs:
        return out
    for item in kvs:
        if "=" not in item:
            continue
        name, shp = item.split("=", 1)
        out[name.strip()] = _parse_shape_str(shp.strip())
    return out


def _parse_mask_inputs(s: str | None) -> set[str]:
    """Parse comma-separated names into a set; empty string -> empty set."""
    if not s:
        return set()
    return {x.strip() for x in s.split(",") if x.strip()}


def _is_mask_input(name: str, mask_names: set[str]) -> bool:
    """Heuristic to decide whether an input is a boolean mask.
    True if name is explicitly listed or contains 'mask' (case-insensitive)."""
    n = name.lower()
    return (name in mask_names) or (n.find("mask") != -1)


def _get_model_inputs(onnx_model):
    """Return a list of (name, type_proto, shape_proto) for true graph inputs (excludes initializers)."""
    g = onnx_model.graph
    init_names = {init.name for init in g.initializer}
    inputs = []
    for vi in g.input:
        if vi.name in init_names:
            continue  # parameters, not real inputs
        tt = vi.type.tensor_type
        inputs.append((vi.name, tt.elem_type, tt.shape))
    return inputs


def make_random_inputs_from_onnx(
    onnx_path: str,
    overrides: dict | None = None,
    seed: int | None = 0,
    device: str = "cpu",
    float_cast: str | None = None,  # one of {None, "fp32", "fp16"}
    mask_names: set[str] | None = None,  # names treated as boolean masks
    mask_prob: float = 0.5,  # Bernoulli p for mask=1
):
    """
    Inspect ONNX model inputs and create random torch tensors that match (dtype, shape).

    Args:
        onnx_path: Path to ONNX.
        overrides: Optional dict: input_name -> shape tuple (fully specifies that input's shape).
        seed: If not None, set RNG seed for reproducibility.
        device: "cpu" or "cuda".
        float_cast: Optional cast for *floating* inputs (others unchanged). {None, "fp32", "fp16"}
        mask_names: Optional set of input names to treat as boolean masks.
        mask_prob: Probability of 1s when generating mask tensors (Bernoulli p).

    Returns:
        inputs: List[torch.Tensor] in the same order as ONNX model inputs.
        names:  List[str] input names (same order).

        Notes:
            Inputs whose names appear in `mask_names` (or contain "mask")
            are generated as {0,1} tensors even if their dtype is floating.
    """
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    model = onnx.load(onnx_path)
    model_inputs = _get_model_inputs(model)

    tensors = []
    names = []
    overrides = overrides or {}

    mask_names = mask_names or set()
    # sanitize mask_prob
    try:
        if not (0.0 <= float(mask_prob) <= 1.0):
            mask_prob = 0.5
    except Exception:
        mask_prob = 0.5

    for name, elem_type, shape_proto in model_inputs:
        # resolve shape
        if name in overrides:
            shape = tuple(int(x) for x in overrides[name])
        else:
            # build shape from dim values/params; fallback to 1 for dynamic
            dims = []
            for d in shape_proto.dim:
                if d.HasField("dim_value") and d.dim_value > 0:
                    dims.append(int(d.dim_value))
                else:
                    # dynamic or unknown -> default to 1
                    dims.append(1)
            shape = tuple(dims) if dims else (1,)

        dtype = _torch_dtype_from_onnx(elem_type)

        # create random tensor according to dtype
        if dtype.is_floating_point:
            if _is_mask_input(name, mask_names):
                # Generate a 0/1 mask with Bernoulli(mask_prob), keep dtype floating
                t = (torch.rand(*shape, device=device) < float(mask_prob)).to(dtype)
            else:
                t = torch.randn(*shape, dtype=dtype, device=device)
            if float_cast is not None:
                if float_cast.lower() == "fp16":
                    t = t.to(torch.float16)
                elif float_cast.lower() == "fp32":
                    t = t.to(torch.float32)
        elif dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            if _is_mask_input(name, mask_names):
                t = torch.randint(low=0, high=2, size=shape, dtype=dtype, device=device)
            else:
                # keep range small to avoid overflow in some ops
                # Use symmetric range [-2, 2)
                high = 3
                low = -2
                t = torch.randint(low=low, high=high, size=shape, dtype=dtype, device=device)
        elif dtype == torch.uint8:
            t = torch.randint(low=0, high=255, size=shape, dtype=dtype, device=device)
        elif dtype == torch.bool:
            t = torch.randint(0, 2, size=shape, dtype=torch.uint8, device=device).bool()
        else:
            # default to float32
            t = torch.randn(*shape, dtype=torch.float32, device=device)
            if float_cast is not None and float_cast.lower() == "fp16":
                t = t.to(torch.float16)

        tensors.append(t)
        names.append(name)

    return tensors, names


def run_quant_and_golden(
    onnx_file: str,
    quant_type: str,
    out_dir: Path,
    device: str = "cuda",
    debug: bool = False,
    seed: int | None = 0,
    overrides: dict | None = None,
    float_cast: str | None = "fp16",
    mask_names: set[str] | None = None,
    mask_prob: float = 0.5,
):
    """Convert ONNX to HMONNX using randomly generated inputs and export golden results."""
    xhquant_init(None, debug=debug)

    target_device = DeviceType.XH2a
    out_hmonnx_file = out_dir / "hmonnx" / f"{Path(onnx_file).stem}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)

    # Prepare example inputs for conversion
    example_inputs, input_names = make_random_inputs_from_onnx(
        onnx_file,
        overrides=overrides,
        seed=seed,
        device="cpu",
        float_cast=None,
        mask_names=mask_names,
        mask_prob=mask_prob,
    )

    quant_scheme = QuantScheme(target_device=target_device, quant_type=quant_type)
    quant_config = create_quant_config(quant_scheme)

    if not os.path.exists(out_hmonnx_file):
        convert_onnx_to_hmonnx(
            onnx_file,
            example_inputs,  # a list of torch tensors
            target_device,
            str(out_hmonnx_file),
            quant_config=quant_config,
        )
        logger = get_root_logger()
        logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}")

    # Golden inference
    hm_model = HMONNXGoldenInference(str(out_hmonnx_file))
    hm_model.save_golden = True
    hm_model.exec_device = device

    golden_dir = out_dir / "golden" / f"{Path(onnx_file).stem}"
    golden_dir.mkdir(exist_ok=True, parents=True)
    hm_model.golden_dir = golden_dir

    # Build runtime inputs on target device with optional float cast for FP tensors
    run_inputs = []
    for t in example_inputs:
        if t.is_floating_point():
            if float_cast and float_cast.lower() == "fp16":
                t = t.to(torch.float16)
            elif float_cast and float_cast.lower() == "fp32":
                t = t.to(torch.float32)
        run_inputs.append(t.to(device))

    with torch.no_grad():
        if len(run_inputs) == 1:
            _ = hm_model(run_inputs[0])
        else:
            _ = hm_model(*run_inputs)

    return str(out_hmonnx_file), str(golden_dir)


def main(args):
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem + "_" + args.quant_type
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)

    overrides = _parse_overrides(args.shape_override)
    mask_names = _parse_mask_inputs(args.mask_inputs)
    mask_prob = args.mask_prob
    out_hmonnx, golden_dir = run_quant_and_golden(
        onnx_file=onnx_file,
        quant_type=args.quant_type,
        out_dir=work_dirs,
        device="cuda" if args.cuda else "cpu",
        debug=args.debug,
        seed=args.seed,
        overrides=overrides,
        float_cast=args.input_dtype,
        mask_names=mask_names,
        mask_prob=mask_prob,
    )

    print(f"HMONNX saved to: {out_hmonnx}")
    print(f"Golden saved under: {golden_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="data/models/xunfei/encoder1d_0.onnx")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--cuda", action="store_true", help="run golden on CUDA if available")
    parser.add_argument("--seed", type=int, default=0, help="random seed for reproducibility")
    parser.add_argument(
        "--input-dtype",
        choices=["fp16", "fp32"],
        default="fp16",
        help="cast floating inputs to this dtype for runtime (golden)",
    )
    parser.add_argument(
        "--shape-override",
        action="append",
        help=("Override input shapes, e.g., --shape-override input0=1x3x48x192." " Can be specified multiple times."),
    )
    parser.add_argument(
        "--mask-inputs",
        type=str,
        default="mask",
        help=(
            "Comma-separated input names to be treated as boolean masks (values restricted to {0,1}). "
            "If omitted, any input whose name contains 'mask' will be treated as a mask."
        ),
    )
    parser.add_argument(
        "--mask-prob",
        type=float,
        default=0.5,
        help=("Probability of 1s when generating mask tensors (Bernoulli p). Must be within [0,1]."),
    )

    args = parser.parse_args()
    main(args)
