import torch
from diffusers import ZImagePipeline
from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
from xhquant.api import (
    ConfigDict,
    DeviceType,
    HMONNXGoldenInference,
    HMONNXInference,
    QuantScheme,
    convert_fx_model_to_quanted_model,
    convert_onnx_to_hmonnx,
    convert_quanted_model_to_hmonnx,
    create_quant_config,
)
from safetensors.torch import load_file, save_file
import torch.nn as nn


class Net_Trace(nn.Module):
    def __init__(self, embedding, transformer):
        super().__init__()
        self.embedding = embedding
        self.transformer = transformer

    def forward(self,
            x,
            attn_mask,
            f_real,
            f_imag,
            t,      
        ):

        adaln_input = self.embedding(t)
        out = self.transformer(x, attn_mask, f_real, f_imag, adaln_input)
        return out

device = "cuda"

# 1. Load the pipeline
# Use bfloat16 for optimal performance on supported GPUs
pipe = ZImagePipeline.from_pretrained(
    "/data02/datasets/zimage",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=False,
)
pipe.to("cuda")

quant_scheme = QuantScheme(target_device=DeviceType.XH2a, quant_type="w4a8h1_ssfp")
quant_config = create_quant_config(quant_scheme)
quant_config = ConfigDict(quant_config)

weights = load_file("/data02/datasets/zimage/diffusion_pytorch_model.safetensors")
model_dict = pipe.transformer.state_dict()

load_keys = [k for k in weights.keys() if k in model_dict]
for k in load_keys:
    model_dict[k] = weights[k]

# 加载到模型
pipe.transformer.load_state_dict(model_dict, strict=False)


from xh_model_zoo.xh_llm.models.zimage._dit_model import register_wrap_cls as llm_register_wrap_cls 
llm_register_wrap_cls(pipe.transformer)
wraped_llm_model = wrap_llm_model(pipe.transformer, {})
wraped_llm_model.cuda()
# wraped_llm_model.to(torch.float16)

model_trace = Net_Trace(pipe.transformer.t_embedder.mlp, pipe.transformer.layers[0])

x = torch.load("examples/llm/zimage/x.pt").cuda()
attn_mask = torch.load("examples/llm/zimage/attn_mask.pt").cuda(),
freqs_cis = torch.load("examples/llm/zimage/freqs_cis.pt").cuda(),
scale_msa = torch.load("examples/llm/zimage/scale_msa.pt").cuda(),
gate_msa = torch.load("examples/llm/zimage/gate_msa.pt").cuda(),
scale_mlp= torch.load("examples/llm/zimage/scale_mlp.pt").cuda(),
gate_mlp = torch.load("examples/llm/zimage/gate_mlp.pt").cuda(),
adaln_input = torch.load("examples/llm/zimage/adaln_input.pt").cuda(),

t = torch.tensor([0.0]).cuda()
t_frep = pipe.transformer.t_embedder.timestep_embedding(t, pipe.transformer.t_embedder.frequency_embedding_size)

freqs_cis_expanded = freqs_cis[0].unsqueeze(2)
f_real = freqs_cis_expanded.real  # 频率的实部，形状匹配x_real
f_imag = freqs_cis_expanded.imag  # 频率的虚部，形状匹配x_imag

input_data = [
    x.float(),
    attn_mask[0].float(),
    f_real.float(),
    f_imag.float(),
    t_frep.float(),
    # scale_msa[0].float(),
    # gate_msa[0].float(),
    # scale_mlp[0].float(),
    # gate_mlp[0].float()
]

hm_onnx_file = "/data01/home/xuchen/xh2/xh2_model_zoo/work_dirs/output_zimage/first_block_honnx.onnx"

# wraped_llm_model.layers[0] = wraped_llm_model.layers[0].float()
# output = wraped_llm_model.layers[0](*input_data)
model_trace = model_trace.float()
output = model_trace(*input_data)

quant_graph_model = convert_fx_model_to_quanted_model(
    model_trace, input_data, DeviceType.XH2a, quant_config
)
convert_quanted_model_to_hmonnx(
    quant_graph_model, input_data, 
    hm_onnx_file,
    # [], 
    # ["output_name"],
)

model_name = "first_block_honnx"

if True:
    session = HMONNXGoldenInference(hm_onnx_file)
    session.to(device)
    session.save_golden = "work_dirs/output_zimage"
    session.golden_dir = "work_dirs" + f"/hmonnx/golden_{model_name}"
    session.step = 0
else:
    session = HMONNXInference(hm_onnx_file)
    session.to(device)

input_data = [
    x.half(),
    attn_mask[0].half(),
    f_real.half(),
    f_imag.half(),
    t_frep.half(),
    # scale_msa[0].half(),
    # gate_msa[0].half(),
    # scale_mlp[0].half(),
    # gate_mlp[0].half()
]

out = session(*input_data)

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


# prompt = "Young Chinese woman in red Hanfu, intricate embroidery. Impeccable makeup, red floral forehead pattern. Elaborate high bun, golden phoenix headdress, red flowers, beads. Holds round folding fan with lady, trees, bird. Neon lightning-bolt lamp (⚡️), bright yellow glow, above extended left palm. Soft-lit outdoor night background, silhouetted tiered pagoda (西安大雁塔), blurred colorful distant lights."
# # 2. Generate Image
# image = pipe(
#     prompt=prompt,
#     height=1024,
#     width=1024,
#     num_inference_steps=9,  # This actually results in 8 DiT forwards
#     guidance_scale=0.0,     # Guidance should be 0 for the Turbo models
#     generator=torch.Generator("cuda").manual_seed(42),
# ).images[0]
# image.save("example.png")