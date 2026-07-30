# Wan2.2 Merak 量化导出与 LoRA 合并示例

本目录保留 Wan2.2 A14B 的结构分析脚本，同时补充 Merak 工作流下的 LoRA 合并与量化导出说明。

可用能力：

- `analyze_wan22.py`：只读盘点 Wan2.2 目录结构，不加载权重。
- `wan22_fp_profile.py`：静态 FP 计算量估算，不加载权重。
- `wan2_2_merge_lora.py`：把 PEFT LoRA adapter 合并到 Wan2.2 high/low noise diffusion safetensors。
- `wan2_2_workflow.py`：Merak Wan2.2 量化导出入口。

## 目录约定

推荐模型目录结构：

```text
Wan_2.2/
  high_noise_model/
  low_noise_model/
  Wan2.1_VAE.pth
  models_t5_umt5-xxl-enc-bf16.pth
  split_files/                         # 可选
    diffusion_models/*.safetensors
    text_encoders/*.safetensors
    vae/*.safetensors
  merged/                              # LoRA 合并后输出
    wan2_2_high_noise_model_xxx_merged.safetensors
    wan2_2_low_noise_model_xxx_merged.safetensors
```

Merak converter 在 `use_resolved_float_loader=true` 时会优先查找：

1. `<model-dir>/merged/*high_noise*merged*.safetensors` 或 `*low_noise*merged*.safetensors`
2. `<model-dir>/split_files/diffusion_models/*high_noise*.safetensors` / `*low_noise*.safetensors`
3. 官方 `high_noise_model/` / `low_noise_model/` 子目录

因此 LoRA 合并后的文件只要放在 `merged/` 下，并且文件名包含 `merged` 和对应 noise model 名称即可被量化导出流程加载。

## 1. 合并 LoRA 权重

示例：只合并 high noise：

```bash
conda run -n xhquant python examples_merak/aigc/wan2.2/wan2_2_merge_lora.py \
  --model-dir /data01/home/xuchen/Wan2.2-main/ckpt/Wan_2.2 \
  --task i2v-A14B \
  --noise-model high_noise_model \
  --lora-dir /path/to/lora_adapter \
  --output-dtype float16 \
  --overwrite
```

同时合并 high/low noise：

```bash
conda run -n xhquant python examples_merak/aigc/wan2.2/wan2_2_merge_lora.py \
  --model-dir /data01/home/xuchen/Wan2.2-main/ckpt/Wan_2.2 \
  --task i2v-A14B \
  --noise-model high_noise_model \
  --noise-model low_noise_model \
  --lora-dir /path/to/lora_adapter \
  --output-dtype float16 \
  --overwrite
```

多个 LoRA adapter 可以重复传 `--lora-dir`，会按命令行顺序依次合并。

脚本支持常见 PEFT safetensors key：

```text
...<module>.lora_A.weight
...<module>.lora_B.weight
```

合并公式：

```text
W_merged = W_base + (B @ A) * (lora_alpha / rank)
```

## 2. 使用合并 LoRA 权重量化导出

导出入口在：

```text
examples_merak/aigc/wan2.2/wan2_2_workflow.py
```

示例：导出 high noise，启用 resolved loader 以加载 `merged/` 权重：

```bash
conda run -n xhquant python examples_merak/aigc/wan2.2/wan2_2_workflow.py \
  --model-dir /data01/home/xuchen/Wan2.2-main/ckpt/Wan_2.2 \
  --output-dir work_dirs/wan2_2_i2v-A14B_merak_lora \
  --task i2v-A14B \
  --components high_noise_model \
  --quant-type w8a8_sefp \
  --torch-dtype float16 \
  --use-resolved-float-loader \
  --overwrite
```

也可以直接在 YAML 中设置：

```yaml
export:
  wan2_2:
    use_resolved_float_loader: true
```

默认配置文件：

```text
configs_merak/workflows/xh2a/other_models/wan2_2/i2v_A14B/wan2_2_i2v_A14B.yaml
```

## 3. 不使用 LoRA 的普通量化导出

如果只想用原始权重，保持 `use_resolved_float_loader: false`，或命令行不传 `--use-resolved-float-loader` 即可。

```bash
conda run -n xhquant python examples_merak/aigc/wan2.2/wan2_2_workflow.py \
  --model-dir /data01/home/xuchen/Wan2.2-main/ckpt/Wan_2.2 \
  --output-dir work_dirs/wan2_2_i2v-A14B_merak \
  --task i2v-A14B \
  --components t5 \
  --quant-type w8a8_sefp \
  --overwrite
```

## 4. 只读盘点与静态分析

```bash
python examples_merak/aigc/wan2.2/analyze_wan22.py \
  --i2v-model-dir /data01/home/xuchen/Wan2.2-main/ckpt/Wan_2.2 \
  --output work_dirs/wan22_i2v_inventory.json

python examples_merak/aigc/wan2.2/wan22_fp_profile.py --help
```

## 注意事项

- LoRA 合并脚本只合并 diffusion transformer 的 Linear 权重，不处理 T5/VAE LoRA。
- 如果 adapter key 无法映射到 WanModel 权重，脚本会报出无法解析的 module path。
- A14B high/low noise 权重很大，合并和导出建议在显存充足的 GPU 上运行。
- 合并后的 safetensors 属于完整权重文件，量化导出时不需要再传 LoRA adapter 路径。
