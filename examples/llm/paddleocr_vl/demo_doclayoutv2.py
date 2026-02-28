# float test for paddleocr-vl
# Use flash-attn to boost performance and reduce memory usage
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from paddleocr import LayoutDetection

model = LayoutDetection(model_name="PP-DocLayoutV2")
output = model.predict(
    "xh2modelzoo/data/images/ocr_img.png",
    batch_size=1,
    layout_nms=True,
)
for res in output:
    res.print()
    res.save_to_img(save_path="./v2output/")
    res.save_to_json(save_path="./v2output/res.json")
