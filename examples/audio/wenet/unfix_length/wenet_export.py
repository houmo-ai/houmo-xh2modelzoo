import argparse
from pathlib import Path

import onnx
import torch
from xhquant.api import (
    DeviceType,
    QuantScheme,
    convert_onnx_to_hmonnx,
    create_quant_config,
    get_root_logger,
    xhquant_init,
)


def main(args):
    onnx_file = args.onnx
    onnx_name = Path(onnx_file).stem
    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file: str = str(out_hmonnx_file)

    xhquant_init(None, debug=args.debug)
    quant_type = args.quant_type
    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=quant_type)

    quant_config = create_quant_config(quant_scheme)
    model = onnx.load(onnx_file)
    input_names = [input.name for input in model.graph.input]
    output_names = [output.name for output in model.graph.output]
    print(input_names)
    print(output_names)
    input_data = torch.load(args.input_data)

    input_data_list = []
    for input_name in input_names:
        if input_name == "update_indices":
            input_data_list.append(torch.from_numpy(input_data[input_name]).to(torch.int64))
        else:
            input_data_list.append(torch.from_numpy(input_data[input_name]).to(torch.float32))

    convert_onnx_to_hmonnx(
        onnx_file,
        input_data_list,
        DeviceType.XH2a,
        out_hmonnx_file,
        quant_config=quant_config,
        input_names=input_names,
        output_names=output_names,
    )
    logger = get_root_logger()
    logger.info(f"Convert onnx to hmonnx success, out hmonnx file to: {out_hmonnx_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="data/models/wenet/chunk_encoder_v2.onnx")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--quant-type", default="w8a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--input-data", type=str, default="data/wenet/encoder_chunk/chunk_encoder_input_1.pth")
    args = parser.parse_args()
    main(args)
