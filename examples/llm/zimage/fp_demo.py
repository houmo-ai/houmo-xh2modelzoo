import torch
from diffusers import ZImagePipeline
from safetensors.torch import load_file, save_file

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "/data02/datasets/zimage",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

weights = load_file("/data02/datasets/zimage/diffusion_pytorch_model.safetensors")
model_dict = pipe.transformer.state_dict()

load_keys = [k for k in weights.keys() if k in model_dict]
for k in load_keys:
    model_dict[k] = weights[k]




# 加载到模型
pipe.transformer.load_state_dict(model_dict, strict=True)

# [Optional] Attention Backend
# Diffusers uses SDPA by default. Switch to Flash Attention for better efficiency if supported:
# pipe.transformer.set_attention_backend("flash")    # Enable Flash-Attention-2
# pipe.transformer.set_attention_backend("_flash_3") # Enable Flash-Attention-3
# [Optional] Model Compilation
# Compiling the DiT model accelerates inference, but the first run will take longer to compile.
# pipe.transformer.compile()
# [Optional] CPU Offloading
# Enable CPU offloading for memory-constrained devices.
# pipe.enable_model_cpu_offload()
prompt = "一朵长得像蝴蝶一样缤纷绚丽的奇异花朵，开在丛林中，散发着柔和的光芒"
# 2. Generate Image
image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    num_inference_steps=9,  # This actually results in 8 DiT forwards
    guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
    generator=torch.Generator("cuda").manual_seed(47),
).images[0]
image.save("example7.png")