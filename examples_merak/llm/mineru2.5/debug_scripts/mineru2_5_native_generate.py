from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from mineru_vl_utils import MinerUClient

# for transformers>=4.56.0
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/data02/datasets/MinerU2.5-Pro-2604-1.2B", dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    "/data02/datasets/MinerU2.5-Pro-2604-1.2B", use_fast=True
)

client = MinerUClient(
    backend="transformers", model=model, processor=processor,
    image_analysis=False # default False, set True to enable image/chart analysis
)

print(client.two_step_extract(Image.open("/data01/home/chuyuan.wei/code/xh2modelzoo/data/images/houmo_logo.jpg")))
