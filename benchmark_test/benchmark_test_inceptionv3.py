import os
from typing import Literal
import allure
import pytest
from pathlib import Path
import onnx


@allure.title("Inceptionv3测试")
@pytest.mark.parametrize(
    "trace_method",
    [
        # "FX",
        # "DFX",
        "ONNX",
    ],
)
@pytest.mark.parametrize("w_bit", [4, 8]) # 
@pytest.mark.parametrize("a_bit", [8]) # , 16
def test_inceptionv3(trace_method: Literal["FX", "DFX", "ONNX"], w_bit: int, a_bit: int):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision.models.resnet import resnet50
    from xhquant.api import (
        DeviceType,
        HMONNXGoldenInference,
        HMONNXInference,
        QuantScheme,
        convert_onnx_to_hmonnx,
        create_quant_config,
        get_root_logger,
        xhquant_init,
    )
    import onnxruntime as ort

    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if w_bit < 8:
        fp_mode="ssfp"
    else:
        fp_mode="sefp"

    quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type=f"w{w_bit}a{a_bit}h1_{fp_mode}")
    quant_config = create_quant_config(quant_scheme)

    onnx_file = "data/models/inceptionv3_model_shape.onnx"
    current_dir = Path(__file__).parent.parent.resolve()  # resolve() 确保是绝对路径
    onnx_file = os.path.join(current_dir, onnx_file)
  
    onnx_model = onnx.load(onnx_file)
    input_shape = [int(dim.dim_value) for dim in onnx_model.graph.input[0].type.tensor_type.shape.dim]

    onnx_name = Path(onnx_file).stem

    work_dirs = Path("work_dirs") / onnx_name
    work_dirs.mkdir(exist_ok=True, parents=True)
    target_device = DeviceType.XH2a
    out_hmonnx_file = work_dirs / "hmonnx" / f"{onnx_name}_{target_device}.onnx"
    out_hmonnx_file.parent.mkdir(exist_ok=True, parents=True)
    out_hmonnx_file: str = str(out_hmonnx_file)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    input = torch.randn(input_shape, dtype=torch.float32)
    
    convert_onnx_to_hmonnx(
        onnx_file,
        [input],
        DeviceType.XH2a,
        out_hmonnx_file,
        quant_config=quant_config,
        input_names=["images"],
        output_names=["cls_score"],
    )

    session = HMONNXGoldenInference(out_hmonnx_file)
    session.to(device)
    session.save_golden = True
    session.golden_dir = work_dirs / "hmonnx/golden"
    session.step = 0
    hm_out = session(input.half().to(device))

    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session_onnx = ort.InferenceSession(onnx_file, providers=providers)
    input_name = session_onnx.get_inputs()[0].name
    outputs = session_onnx.run(None, {input_name:input.cpu().numpy()})
    cos_sim = torch.cosine_similarity(hm_out, torch.from_numpy(outputs[0]).to(hm_out.device) )

    allure.attach(trace_method, "TraceMethod", attachment_type=allure.attachment_type.TEXT)
    allure.attach(str(w_bit), "Wbit", attachment_type=allure.attachment_type.TEXT)
    allure.attach(str(a_bit), "Abit", attachment_type=allure.attachment_type.TEXT)
    allure.attach(str(f"Cosim_{cos_sim}"), "ACC/Cosim_/PPL", attachment_type=allure.attachment_type.TEXT)
    golden_path = os.path.join(current_dir, session.golden_dir)
    allure.attach(str(f"{golden_path}"), 
                  "Golden 地址", attachment_type=allure.attachment_type.TEXT)

if __name__ == "__main__":
    test_inceptionv3("ONNX", w_bit=8, a_bit=16)
