"""Short-lived ONNX simplifier used by the Qwen3.5 visual exporter."""

import argparse
import gc
import resource
from pathlib import Path

import onnx
import onnx_graphsurgeon as gs
from onnxsim import simplify

from xhquant.api import get_xhquant_logger
from xhquant.utils.onnxsim_large_model.compress_model import compress_onnx_model, uncompress_onnx_model
from xhquant.utils.onnxsim_large_model.onnx_utils import set_onnx_input_shape
from xhquant.utils.onnxsim_large_model.simplify_large_onnx import constant2initializer


_SKIPPED_OPTIMIZERS = [
    "fuse_pad_into_conv",
    "fuse_consecutive_slices",
    "eliminate_common_subexpression",
    "fuse_qkv",
]


def simplify_visual_onnx(input_file: str, output_file: str) -> None:
    """Simplify and persist one visual ONNX, then let process exit free memory."""
    logger = get_xhquant_logger()
    model = onnx.load(input_file)
    ir_version = model.ir_version

    # This subprocess exclusively owns ``model``, so xhquant's defensive
    # ``copy.deepcopy(model)`` is unnecessary.  It costs another complete
    # 2+ GiB ModelProto for this visual graph.
    model = constant2initializer(model)
    graph = gs.import_onnx(model)
    del model
    gc.collect()
    graph.toposort()
    graph.fold_constants()
    graph.cleanup()
    model = gs.export_onnx(graph)
    del graph
    gc.collect()

    model.ir_version = min(ir_version, 10)
    model, removed_initializers = compress_onnx_model(model)
    model = set_onnx_input_shape(model, False)

    original_onnx_save = onnx.save
    try:
        onnx.save = lambda *args, **kwargs: original_onnx_save(
            *args,
            **kwargs,
            convert_attribute=True,
        )
        simplified_model, succeeded = simplify(
            model,
            skip_fuse_bn=True,
            skipped_optimizers=_SKIPPED_OPTIMIZERS,
        )
    finally:
        onnx.save = original_onnx_save
    if succeeded:
        simplified_model = uncompress_onnx_model(
            simplified_model,
            removed_initializers,
        )
    del model
    gc.collect()

    if not succeeded:
        raise RuntimeError(f"Failed to simplify visual ONNX: {input_file}")

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(
        simplified_model,
        str(output_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=f"{output_path.stem}_external_data",
    )
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    logger.info("Visual ONNX simplify subprocess peak RSS: %.2f MB", peak_rss_mb)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("output_file")
    args = parser.parse_args()
    simplify_visual_onnx(args.input_file, args.output_file)


if __name__ == "__main__":
    main()
