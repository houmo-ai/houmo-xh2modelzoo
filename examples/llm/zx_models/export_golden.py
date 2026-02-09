      
# copied from xh2modelzoo/examples/llm/zx_models/export_golden.py
import torch
import onnx
from xhquant.api import (
    DeviceType,
    QuantScheme,
    create_quant_config,
    ConfigDict,
    convert_onnx_to_hmonnx,
    HMONNXGoldenInference,
)    
import os
import argparse
import onnxruntime as ort
    
def main(args):
    onnx_model_path = args.model
    output_dir = args.output_dir
    device = "cpu"

    model_name = os.path.split(onnx_model_path)[-1][0:-5]
    quant_model_path = os.path.join(
        output_dir, 'hmquant_' + model_name + '_with_act.onnx',
    )
    onnx_model = onnx.load(onnx_model_path)
    input_names = [_input.name for _input in onnx_model.graph.input]
    output_names = [_output.name for _output in onnx_model.graph.output]
    dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
    input_shape = [dim.dim_value for dim in dims]

    seed = 100
    torch.manual_seed(seed)    
    calib_dataset = [ torch.randn(input_shape, dtype=torch.float32, device=device) ]

    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a, quant_type=args.quant_type,
    )
    quant_config = create_quant_config(quant_scheme)
    quant_config = ConfigDict(quant_config) 

    output_ori = ort.InferenceSession(onnx_model_path).run(None, {"input": calib_dataset[0].cpu().numpy()})

    convert_onnx_to_hmonnx(
        onnx_model,
        calib_dataset,
        device_type=DeviceType.XH2a,
        out_hmonnx_file=quant_model_path,
        quant_config=quant_config,
        input_names=input_names,
        output_names=output_names,
    )
    session = HMONNXGoldenInference(quant_model_path)
    device = 'cpu'
    session.to(device)
    session.save_golden = True
    session.golden_dir = os.path.join(output_dir, 'golden')
    output = session(calib_dataset[0].half())

    print(torch.from_numpy(output_ori[0]) - output)
    print(torch.cosine_similarity(torch.from_numpy(output_ori[0]), output, dim=0))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--model", type=str, default="data/models/L1csi-zgb_b8_fp32.sim.onnx")
    parser.add_argument("--quant-type", default="w4a8h1_sefp", help="quant type, default is w8a8")
    parser.add_argument("--output-dir", default="work_dirs/zx", help="save model path")
    parser.add_argument(
        "--quant-weight",
        type=str,
        default=None,
        help="quant weight path, for example: gptq or quarot, if empty, use w8a8",
    )
    args = parser.parse_args()
    main(args)