from cv2 import resize
from transformers import AutoModel, AutoTokenizer
from xh_model_zoo.xh_llm.models.deepseek_ocr.modeling_deepseekocr import (
    DeepseekOCRForCausalLM,
    DeepseekOCRModel,
)
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
model_name = '/data02/datasets/DeepSeek-OCR/'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model:DeepseekOCRForCausalLM = DeepseekOCRForCausalLM.from_pretrained(model_name, 
                                _attn_implementation='eager', 
                                trust_remote_code=True, 
                                use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

prompt = "<image>\nFree OCR. "
# prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'examples/llm/deepseek_ocr/data/img3.png'
output_path = 'examples/llm/deepseek_ocr/data'

# # Tiny: base_size = 512, image_size = 512, crop_mode = False
# res = model.infer(tokenizer, 
#                   prompt=prompt, 
#                   image_file=image_file, 
#                   output_path = output_path, 
#                   base_size = 512, 
#                   image_size = 512, 
#                   crop_mode=False, 
#                   save_results = True, 
#                   test_compress = True)

# # Small: base_size = 640, image_size = 640, crop_mode = False
# res = model.infer(tokenizer, 
#                   prompt=prompt, 
#                   image_file=image_file, 
#                   output_path = output_path, 
#                   base_size = 640, 
#                   image_size = 640, 
#                   crop_mode=False, 
#                   save_results = True, 
#                   test_compress = True)

# # Base: base_size = 1024, image_size = 1024, crop_mode = False
# res = model.infer(tokenizer, 
#                   prompt=prompt, 
#                   image_file=image_file, 
#                   output_path = output_path, 
#                   base_size = 1024, 
#                   image_size = 1024, 
#                   crop_mode=False, 
#                   save_results = True, 
#                   test_compress = True)

# # Large: base_size = 1280, image_size = 1280, crop_mode = False
# res = model.infer(tokenizer, 
#                   prompt=prompt, 
#                   image_file=image_file, 
#                   output_path = output_path, 
#                   base_size = 1280, 
#                   image_size = 1280, 
#                   crop_mode=False, 
#                   save_results = True, 
#                   test_compress = True)

# Gundam: base_size = 1024, image_size = 640, crop_mode = True
res = model.infer(tokenizer, 
                  prompt=prompt, 
                  image_file=image_file, 
                  output_path = output_path, 
                  base_size = 1024, 
                  image_size = 640, 
                  crop_mode=True, 
                  save_results = True, 
                  test_compress = True)

# 注意：
    # 1. antialias=True, 我们暂时没管，可能是误差来源
    # 2. resize的mode从bicubic改为了bilinear
    # 3. quick_gelu被我设置为gelu，也可能是误差来源
    # 4. 
