# Initialize PaddleOCR instance
from paddleocr import PaddleOCR
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
    device="gpu",
)

# Run OCR inference on a sample image 
result = ocr.predict(input="/data01/home/xuchen/xh2/xh2_model_zoo/examples/llm/groot/general_ocr_002.png")

# Visualize the results and save the JSON results
for res in result:
    res.print()
    res.save_to_img("output")
    res.save_to_json("output")