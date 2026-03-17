# Initialize PaddleOCR instance
from paddleocr import PaddleOCR
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

device = "cuda:0"
work_dirs = "work_dirs/paddle_312"

ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu",
)

# Run OCR inference on a sample image 
result = ocr.predict(input="data/models/ocr_onnx/general_ocr_002.png")

# Visualize the results and save the JSON results
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")

from xh_model_zoo.xh_llm.models.paddle_ocrv5.cus_textrec import Cus_TextRecPredictor
from xh_model_zoo.xh_llm.models.paddle_ocrv5.cus_textdet import Cus_TextDetPredictor

det_onnx_path = "work_dirs/paddle_312/hmquant_xh2_paddleocr_det.onnx"
rec_onnx_path = "work_dirs/paddle_312/hmquant_xh2_paddleocr_rec.onnx"

det_model = HMONNXGoldenInference(det_onnx_path)
det_model.to(device)
det_model.save_golden = False
det_model.golden_dir = work_dirs + "/hmonnx/golden"
det_model.step = 0


rec_model = HMONNXGoldenInference(rec_onnx_path)
rec_model.to(device)
rec_model.save_golden = False
rec_model.golden_dir = work_dirs + "/hmonnx/golden"
rec_model.step = 0


ocr.paddlex_pipeline.text_det_model = Cus_TextDetPredictor.to_hf_compatible(
    ocr.paddlex_pipeline.text_det_model, det_model
)

ocr.paddlex_pipeline.text_rec_model = Cus_TextRecPredictor.to_hf_compatible(
    ocr.paddlex_pipeline.text_rec_model, rec_model
)

# Run OCR inference on a sample image 
result = ocr.predict(input="data/models/ocr_onnx/general_ocr_002.png")

# Visualize the results and save the JSON results
for res in result:
    res.print()
    res.save_to_img("output_hm")
    res.save_to_json("output_hm")