# float test for paddleocr-vl
# Use flash-attn to boost performance and reduce memory usage
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayoutV3")
output = model.predict(
    "/data01/home/linxiang.wang/xhquant_llm/data/images/ocr_img.png",
    batch_size=1,
    layout_nms=True,
)
for res in output:
    res.print()
    res.save_to_img(
        save_path="/data01/home/linxiang.wang/xhquant_llm/work_dirs/paddleocr_vl_1_5/doclayoutv3_hmonnx_export/res_img.png"
    )
    res.save_to_json(
        save_path="/data01/home/linxiang.wang/xhquant_llm/work_dirs/paddleocr_vl_1_5/doclayoutv3_hmonnx_export/res.json"
    )
