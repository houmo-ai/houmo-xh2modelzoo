import os
import os.path as osp
import torch

from xhquant.api import (
    convert_onnx_to_hmonnx,
    QuantScheme,
    create_quant_config,
    DeviceType,
    HMONNXGoldenInference,
)

model_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/speech_tokenizer_v3_3000_3.onnx"
OUTPUT_DIR = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmquant_xh2_fun_cosyvoice3_0.5B_2512_w8a8_20260320/speech_tokenizer_v3/prefill"
GOLDEN_DIR = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmquant_xh2_fun_cosyvoice3_0.5B_2512_w8a8_20260320/speech_tokenizer_v3/prefill/step_0"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GOLDEN_DIR, exist_ok=True)

def main():
    # dummy input
    input = torch.randn(1, 128, 3000)
    mask = torch.randn(1, 20, 750, 750)
    mask1 = torch.randn(1, 750, 1280)

    # quant config
    quant_type = "w8a16_sefp"
    quant_scheme = QuantScheme(
        target_device=DeviceType.XH2a,
        quant_type=quant_type
    )
    quant_config = create_quant_config(quant_scheme)

    # convert（加存在判断）
    prefix = f"hmquant_xh2_speech_tokenizer_v3_w8a16_3000_20260320"
    output_file = osp.join(OUTPUT_DIR, f"{prefix}.onnx")

    if not osp.exists(output_file):
        convert_onnx_to_hmonnx(
            model_path,
            (input, mask, mask1),
            out_hmonnx_file=output_file,
            device_type="XH2A",
            quant_config=quant_config
        )

    # golden
    model = HMONNXGoldenInference(output_file)
    model.save_golden = True
    model.exec_device = torch.device("cuda:0")

    input = input.to(torch.float16)
    mask = mask.to(torch.float16)
    mask1 = mask1.to(torch.float16)

    input_args = (input, mask, mask1)
    model.golden_dir = str(GOLDEN_DIR)

    with torch.no_grad():
        model.forward(*input_args)

if __name__ == "__main__":
    main()