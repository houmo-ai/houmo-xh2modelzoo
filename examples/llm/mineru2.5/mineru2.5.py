from modelscope import AutoProcessor, Qwen2VLForConditionalGeneration
from PIL import Image
from mineru_vl_utils import MinerUClient

# for transformers>=4.56.0
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/data02/datasets/mineru2.5",
    # dtype="auto", # use `torch_dtype` instead of `dtype` for transformers<4.56.0
    device_map="auto"
)

processor = AutoProcessor.from_pretrained(
    "/data02/datasets/mineru2.5",
    use_fast=True
)

client = MinerUClient(
    backend="transformers",
    model=model,
    processor=processor
)

image = Image.open("data/images/houmo_logo.jpg")
extracted_blocks = client.two_step_extract(image)
print(extracted_blocks)