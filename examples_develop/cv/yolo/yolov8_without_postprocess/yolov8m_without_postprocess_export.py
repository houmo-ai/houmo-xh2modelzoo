import argparse
from pathlib import Path

import torch
from xhquant.api import (
    DeviceType,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)

from xh2_model_zoo.utils.onnx.onnx_shape_infer import replace_reshape_minus_one_by_infer


def main(args):
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file = str(out_hmonnx_file)

    xhquant_init(None, debug=args.debug)
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)

    quant_config = create_quant_config(quant_scheme)
    # 在导出hmonnx前，先修正Reshape的-1
    fixed_onnx_path = work_dirs / f"{onnx_name}_fixed_reshape.onnx"
    replace_reshape_minus_one_by_infer(
        onnx_path=onnx_file, output_path=str(fixed_onnx_path), input_shape=[1, 3, 640, 640]  # 可根据实际输入shape调整
    )
    # 后续流程用fixed_onnx_path作为输入
    onnx_file = str(fixed_onnx_path)

    exported_graph_module = convert_onnx_to_hmonnx(
        onnx_file,
        [torch.randn(1, 3, 640, 640, dtype=torch.float32)],
        DeviceType.XH2a,
        out_hmonnx_file,
        quant_config=quant_config,
        input_names=["images"],
        output_names=["outs"],
    )
    logger = get_root_logger()
    logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}")

    exported_graph_module = exported_graph_module.to("cuda")
    exported_graph_module(torch.randn(2, 3, 640, 640, dtype=torch.float16).to("cuda"))
    print(exported_graph_module.graph)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="data/model_zoo2/yolov8m_without_postprocess.onnx")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--image", type=str, default="data/images/000000001490.jpg")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    args = parser.parse_args()
    main(args)
