# Wan2.2 A14B / TI2V-5B 浮点模型分析

> 范围：`/data01/datasets/Wan2.2-T2V-A14B` 本地 T2V 权重、官方 Wan2.2-I2V-A14B 结构，以及官方 Wan2.2-TI2V-5B 配置。本文只分析浮点模型结构、算子、输入输出、数据流 shape 与算力，不涉及量化、HMONNX 导出或 NPU 适配结论。

## 1. 结论摘要

- Wan2.2 A14B 是视频扩散模型，不是自回归 LLM。核心链路为 `T5 文本编码 -> Wan DiT denoiser(high/low expert) -> VAE decode`。
- A14B 的 MoE 不是 token router MoE，而是按扩散 timestep/SNR 切换的双 denoiser expert：高噪阶段用 `high_noise_model`，低噪阶段用 `low_noise_model`。每个采样步只激活一个 expert。
- 每个 Wan DiT expert 约 `14.289B` 参数；本地 FP32 safetensors 每个 expert 约 `53.23 GiB`，high+low 两套约 `106.46 GiB`。官方配置推理 `param_dtype=torch.bfloat16`，若转换为 bf16，单 expert 权重约 `26.62 GiB`。
- T2V 与 I2V 主干相同：`dim=5120`、`ffn_dim=13824`、`num_layers=40`、`num_heads=40`、`head_dim=128`、`text_len=512`。关键差异是 DiT 输入通道：T2V `in_dim=16`，I2V `in_dim=36`。
- I2V 的 `36` 通道来自 `x` 噪声 latent `16ch` 加条件 `y`：参考图 VAE latent `16ch` + mask `4ch`，即 `16 + 20 = 36`。
- 对 81 帧 720P（1280x720），latent patch token 数为 `75,600`。按官方全局时空 attention 估算，单次 WanModel 前向约 `6.52 PFLOPs`，CFG 一个采样步要 cond/uncond 两次前向，约 `13.05 PFLOPs`；40 步约 `521.9 PFLOPs`，不含 T5/VAE。
- TI2V-5B 的官方 720P 档是 `1280x704` / `704x1280` 对齐尺寸，默认 `121` 帧、`24 fps`，输出约 `5.04s`。Wan2.2 VAE latent 为 `48ch`，stride `(4,16,16)`，720P-aligned token 数为 `27,280`，50 步 DiT 去噪约 `52.07 PFLOPs`。
- 按 M50 峰值 `150 TFLOP/s`、三折有效算力 `45 TFLOP/s` 粗估，A14B 720P 5 秒视频约 `195 min`；TI2V-5B 720P-aligned 5 秒视频约 `20.8-20.9 min`。这是 compute-only 下界，未含访存、offload、kernel 效率和调度损耗。

## 2. 证据来源

本地文件：

- T2V 根目录：`/data01/datasets/Wan2.2-T2V-A14B`
- T2V 配置：`configuration.json`
- DiT 配置：`high_noise_model/config.json`、`low_noise_model/config.json`
- Safetensors index：`*/diffusion_pytorch_model.safetensors.index.json`
- T5 权重：`models_t5_umt5-xxl-enc-bf16.pth`
- VAE 权重：`Wan2.1_VAE.pth`

官方源码/配置：

- Wan2.2 GitHub: <https://github.com/Wan-Video/Wan2.2>
- `wan/modules/model.py`: WanModel / WanAttentionBlock
- `wan/text2video.py`: T2V generation flow
- `wan/image2video.py`: I2V generation flow
- `wan/textimage2video.py`: TI2V-5B unified T2V/I2V generation flow
- `wan/configs/wan_ti2v_5B.py`: TI2V-5B dim/layers/fps/steps/frame config
- `wan/modules/vae2_2.py`: Wan2.2 VAE `z_dim=48`
- ModelScope I2V: <https://www.modelscope.cn/models/Wan-AI/Wan2.2-I2V-A14B>

## 3. 模型文件结构

本地 T2V 目录：

```text
Wan2.2-T2V-A14B/
  configuration.json
  high_noise_model/
    config.json
    diffusion_pytorch_model-00001-of-00006.safetensors
    ...
    diffusion_pytorch_model-00006-of-00006.safetensors
    diffusion_pytorch_model.safetensors.index.json
  low_noise_model/
    config.json
    diffusion_pytorch_model-00001-of-00006.safetensors
    ...
    diffusion_pytorch_model-00006-of-00006.safetensors
    diffusion_pytorch_model.safetensors.index.json
  Wan2.1_VAE.pth
  models_t5_umt5-xxl-enc-bf16.pth
```

权重体积统计：

| 组件 | 文件/分片 | 本地大小 | 说明 |
| --- | ---: | ---: | --- |
| high_noise_model | 6 safetensors | 53.23 GiB | 高噪 denoiser expert |
| low_noise_model | 6 safetensors | 53.23 GiB | 低噪 denoiser expert |
| T5 encoder | 1 pth | 10.58 GiB | UMT5-XXL 文本编码器 |
| VAE | 1 pth | 0.47 GiB | Wan2.1 VAE |
| 合计 | 14 权重文件 | 117.51 GiB | 本地 FP 存储量 |

## 4. 架构参数

| 字段 | T2V-A14B | I2V-A14B | TI2V-5B | 说明 |
| --- | ---: | ---: | ---: | --- |
| task | text-to-video-synthesis | image-to-video | any-to-any / ti2v | 根配置/官方 task |
| model_type | t2v | i2v | ti2v | DiT 类型 |
| active DiT | high/low 双 expert | high/low 双 expert | 单 dense 模型 | A14B 按 timestep 切 expert |
| in_dim | 16 | 36 | 48 | TI2V 图生视频通过 latent mask 约束，不再扩成 36ch |
| out_dim | 16 | 16 | 48 | 预测噪声/latent 通道 |
| dim | 5120 | 5120 | 3072 | Transformer hidden |
| ffn_dim | 13824 | 13824 | 14336 | FFN 中间维度 |
| num_layers | 40 | 40 | 30 | DiT blocks |
| num_heads | 40 | 40 | 24 | attention heads |
| head_dim | 128 | 128 | 128 | hidden / heads |
| text_len | 512 | 512 | 512 | T5 context padding length |
| patch_size | (1, 2, 2) | (1, 2, 2) | (1, 2, 2) | DiT Conv3d patch |
| vae_stride | (4, 8, 8) | (4, 8, 8) | (4, 16, 16) | 时间/高/宽压缩 |
| frame_num | 81 | 81 | 121 | 默认帧数 |
| sample_fps | 16 | 16 | 24 | 输出 fps |
| 输出时长 | 5.06s | 5.06s | 5.04s | frames / fps |
| sample_steps | 40 | 40 | 50 | 官方配置 |
| guide_scale | (3.0, 4.0) | (3.5, 3.5) | 5.0 | CFG scale |

## 5. Wan DiT 算子组成

### 5.1 顶层模块

每个 high/low expert 是一个完整 `WanModel`：

1. `patch_embedding`: `Conv3d(in_dim -> 5120, kernel=(1,2,2), stride=(1,2,2))`
2. `text_embedding`: `Linear(4096 -> 5120) -> GELU(tanh) -> Linear(5120 -> 5120)`
3. `time_embedding`: sinusoidal timestep embedding `256` 维，接 `Linear -> SiLU -> Linear`
4. `time_projection`: `SiLU -> Linear(5120 -> 6*5120)`，给每层 modulation 使用
5. `blocks[40]`: 40 个 `WanAttentionBlock`
6. `head`: norm + modulation + `Linear(5120 -> out_dim * prod(patch_size))`
7. `unpatchify`: token 输出还原为 `[16, F_lat, H_lat, W_lat]`

### 5.2 每层 WanAttentionBlock

每层包含：

| 子模块 | 算子 | 输入 shape | 输出 shape |
| --- | --- | --- | --- |
| norm1 + modulation | LayerNorm + affine modulation | `[B,S,5120]` | `[B,S,5120]` |
| self attention QKV | 3 x Linear(5120,5120) | `[B,S,5120]` | Q/K/V `[B,S,40,128]` |
| QK norm | RMSNorm on Q/K | `[B,S,40,128]` | `[B,S,40,128]` |
| 3D RoPE | temporal/height/width rotary | Q/K | Q/K |
| flash attention | global/window attention | Q/K/V | `[B,S,40,128]` |
| self attention O | Linear(5120,5120) | `[B,S,5120]` | `[B,S,5120]` |
| norm3 + cross attention | text cross-attn | x `[B,S,5120]`, text `[B,512,5120]` | `[B,S,5120]` |
| norm2 + FFN | Linear(5120,13824) -> GELU -> Linear(13824,5120) | `[B,S,5120]` | `[B,S,5120]` |
| residual gates | modulation scale | `[B,S,5120]` | `[B,S,5120]` |

主要算子类别：

- Dense/Conv: `Conv3d`, `Linear`
- Norm: `LayerNorm`, `RMSNorm`
- Attention: Q/K/V projection, 3D RoPE, FlashAttention, output projection
- Activation: `GELU(approximate=tanh)`, `SiLU`
- Data movement: pad/concat/flatten/transpose/unpatchify
- Scheduler: UniPC 或 DPM++ step，更新 latent

## 6. 完整数据流图

```text
Prompt string
  -> T5 tokenizer
  -> T5 Encoder UMT5-XXL
  -> context: List[[<=512,4096]]
  -> WanModel text_embedding
  -> context_emb: [B,512,5120]

Negative prompt
  -> T5 tokenizer
  -> T5 Encoder UMT5-XXL
  -> context_null: List[[<=512,4096]]
  -> WanModel text_embedding
  -> context_null_emb: [B,512,5120]

Seed
  -> noise latent x: [16,F_lat,H_lat,W_lat]

I2V only:
  Input image [3,H_img,W_img]
    -> resize by max_area/aspect
    -> first-frame video condition [3,81,H,W]
    -> VAE encode
    -> image latent [16,F_lat,H_lat,W_lat]
  mask build
    -> mask [4,F_lat,H_lat,W_lat]
  image latent + mask
    -> y condition [20,F_lat,H_lat,W_lat]

For each scheduler timestep t:
  if t >= boundary * num_train_timesteps:
    active expert = high_noise_model
  else:
    active expert = low_noise_model

  T2V WanModel input:
    x [16,F_lat,H_lat,W_lat] + context/context_null + seq_len

  I2V WanModel input:
    concat(x [16,F_lat,H_lat,W_lat], y [20,F_lat,H_lat,W_lat])
      -> DiT input [36,F_lat,H_lat,W_lat]

  active expert forward, cond branch
    -> noise_pred_cond [16,F_lat,H_lat,W_lat]
  active expert forward, uncond branch
    -> noise_pred_uncond [16,F_lat,H_lat,W_lat]
  CFG combine
    -> noise_pred [16,F_lat,H_lat,W_lat]
  scheduler step
    -> updated latent [16,F_lat,H_lat,W_lat]

Final latent x0 [16,F_lat,H_lat,W_lat]
  -> VAE decode
  -> RGB video [3,81,H,W]
```

## 7. Shape 分析

默认帧数 `F=81`，VAE stride `(4,8,8)`，DiT patch `(1,2,2)`。

Latent 时间长度：

```text
F_lat = (F - 1) // 4 + 1 = 21
H_lat = H // 8
W_lat = W // 8
S = F_lat * (H_lat // 2) * (W_lat // 2)
```

### 7.1 T2V shape

| 阶段 | 480P, size=(832,480) | 720P, size=(1280,720) |
| --- | --- | --- |
| 输出视频 | `[3,81,480,832]` | `[3,81,720,1280]` |
| 噪声 latent | `[16,21,60,104]` | `[16,21,90,160]` |
| patch grid | `[21,30,52]` | `[21,45,80]` |
| DiT tokens S | `32,760` | `75,600` |
| DiT token hidden | `[1,32760,5120]` | `[1,75600,5120]` |
| text context | `[1,512,4096] -> [1,512,5120]` | 同左 |
| DiT output latent | `[16,21,60,104]` | `[16,21,90,160]` |

### 7.2 I2V shape

I2V 输出分辨率由输入图宽高比和 `max_area` 推导；若输入图比例与 1280x720 一致，则 shape 与 720P T2V 类似。

| 阶段 | Shape | 说明 |
| --- | --- | --- |
| 输入图 | `[3,H_img,W_img]` | PIL -> tensor, normalize 到 `[-1,1]` |
| resize 后首帧 | `[3,H,W]` | 按 `max_area` 和宽高比 |
| 构造视频条件 | `[3,81,H,W]` | 首帧为输入图，其余帧为 0 |
| VAE encode | `[16,F_lat,H_lat,W_lat]` | 图像条件 latent |
| mask | `[4,F_lat,H_lat,W_lat]` | 首段 mask repeat/interleave 后 reshape |
| y | `[20,F_lat,H_lat,W_lat]` | `mask(4) + image_latent(16)` |
| x | `[16,F_lat,H_lat,W_lat]` | 采样噪声 latent |
| DiT input | `[36,F_lat,H_lat,W_lat]` | `cat(x,y)`，对应 `in_dim=36` |
| DiT output | `[16,F_lat,H_lat,W_lat]` | 预测噪声 |

### 7.3 TI2V-5B shape

TI2V-5B 默认 `F=121`、`fps=24`，官方 720P 档为 `1280x704` 或 `704x1280`。Wan2.2 VAE 使用 `z_dim=48`、stride `(4,16,16)`，DiT patch `(1,2,2)`。

```text
F_lat = (121 - 1) // 4 + 1 = 31
H_lat = 704 // 16 = 44
W_lat = 1280 // 16 = 80
S = 31 * (44 // 2) * (80 // 2) = 27,280
```

| 阶段 | Shape | 说明 |
| --- | --- | --- |
| 输出视频 | `[3,121,704,1280]` | 约 `5.04s` |
| 噪声 latent | `[48,31,44,80]` | Wan2.2 VAE latent |
| patch grid | `[31,22,40]` | patch `(1,2,2)` |
| DiT tokens S | `27,280` | 低于 A14B 720P 的 `75,600` |
| DiT token hidden | `[1,27280,3072]` | TI2V hidden dim |
| text context | `[1,512,4096] -> [1,512,3072]` | Wan text embedding |
| DiT output latent | `[48,31,44,80]` | 预测噪声 |

## 8. 参数量与显存

按结构公式计算每个 expert：

| 模块 | 参数量 |
| --- | ---: |
| Patch embedding T2V | 0.333M |
| Patch embedding I2V | 0.742M |
| Text embedding | 47.19M |
| Time embedding + projection | 185.60M |
| 单个 WanAttentionBlock | 351.40M |
| 40 blocks | 14.056B |
| Head + 其他小参数 | 约 0.34M |
| T2V 单 expert 总计 | 14.2887B |
| I2V 单 expert 总计 | 14.2891B |

存储/驻留估算：

| 项 | FP32 | BF16/FP16 |
| --- | ---: | ---: |
| 单 expert DiT 权重 | 53.23 GiB | 26.62 GiB |
| high+low 两 expert | 106.46 GiB | 53.23 GiB |
| T5 encoder 本地文件 | 10.58 GiB | 约 10.58 GiB 文件口径 |
| VAE 本地文件 | 0.47 GiB | 约 0.47 GiB 文件口径 |

说明：

- 本地 safetensors index 的 `metadata.total_size=57153966336 bytes`，与 `14.2887B * 4 bytes` 基本一致，说明本地 DiT 权重文件是 FP32 存储量级。
- 官方推理配置 `param_dtype=torch.bfloat16`，并提供 `--convert_model_dtype` 将 DiT 参数转成 bf16 推理。
- 单卡运行时通常依赖 offload：一次只把当前 timestep 所需 expert 放到 GPU，并可把 T5 放 CPU。

## 9. FLOPs 估算

估算口径：

- 只统计 Wan DiT denoiser 主体，不含 T5 encoder、VAE encode/decode、scheduler 小算子和数据搬运。
- FLOPs 以 matmul 乘加 `2 * M*N*K` 计。
- Attention score + AV 约为 `4 * S^2 * C`。
- Cross attention 文本长度固定 `L_text=512`。
- CFG 每步调用 cond 和 uncond 两次 DiT 前向，因此采样步 FLOPs 是单次前向的 2 倍。

| 模型/分辨率 | tokens S | 单次 DiT 前向 | CFG 单采样步 | 默认步数采样 |
| --- | ---: | ---: | ---: | ---: |
| A14B 480P 832x480 | 32,760 | 1.678 PFLOPs | 3.357 PFLOPs | 134.3 PFLOPs / 40 steps |
| A14B 720P 1280x720 | 75,600 | 6.552 PFLOPs | 13.103 PFLOPs | 524.1 PFLOPs / 40 steps |
| TI2V-5B 720P-aligned 1280x704 | 27,280 | 520.695 TFLOPs | 1.041 PFLOPs | 52.1 PFLOPs / 50 steps |

720P 单层主项近似：

| 单层子项 | FLOPs 量级 | 说明 |
| --- | ---: | --- |
| Self-attn QK/AV | 116.99 TFLOPs | `S^2` 主导 |
| QKV/O projection | 15.86 TFLOPs | 4 个 5120x5120 linear |
| Cross-attn Q/K/V/O + score/AV | 21.33 TFLOPs | text_len=512 |
| FFN | 21.40 TFLOPs | 两个 dense linear |
| 单层合计 | 约 163.1 TFLOPs | 40 层约 6.52 PFLOPs |

关键观察：

- 720P 的全局时空 self-attention 是绝对主项，复杂度随 `S^2` 增长。
- 480P 到 720P，tokens 从 32,760 增到 75,600，self-attention 计算约放大 `(75600/32760)^2 = 5.33x`。
- I2V 相对 T2V 只增加 patch embedding 输入通道和 VAE encode 条件分支；DiT 主体 FLOPs 基本相同。
- TI2V-5B 通过更高 VAE 空间压缩和较小 hidden，把 720P-aligned tokens 降到 `27,280`，即使采样 50 步，DiT 去噪循环仍约为 A14B 的 9.9%。

## 10. 模块级算力拆分与 M50 耗时估算

本节由 `wan22_fp_profile.py` 静态生成，口径如下：

- T5 统计 cond/uncond 两个 prompt 的 encoder 计算。
- Wan DiT 统计单个 active expert；每个采样步因 CFG 需要 cond/uncond 两次 WanModel 前向。
- VAE 为静态卷积近似：官方实现使用 causal 3D conv + temporal chunk cache，真实 kernel 级 FLOPs/带宽需 profiler 复核。
- Scheduler、tokenizer、shape/pad/concat、RoPE、norm、activation 等小算子已单列或并入对应模块；与 DiT attention/FFN 相比不是主导项。
- M50 估算按 `150 TFLOP/s * 0.3 = 45 TFLOP/s`，即 `耗时 = 总 FLOPs / 45e12`。这是 compute-only 估算，不含访存、offload、kernel occupancy、host/device copy。

### 10.1 720P/720P-aligned 模型总表

| 模型 | 输出规格 | 视频时长 | T5 cond+uncond | VAE encode | DiT 单次前向 | DiT 去噪循环 | VAE decode | 端到端总算力 | M50 预计耗时 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Wan2.2-A14B T2V | 1280x720, 81f@16fps | 5.06s | 9.690 TFLOPs | - | 6.552 PFLOPs | 524.130 PFLOPs | 1.193 PFLOPs | 525.333 PFLOPs | 194.6 min |
| Wan2.2-A14B I2V | 1280x720, 81f@16fps | 5.06s | 9.690 TFLOPs | 841.087 TFLOPs | 6.552 PFLOPs | 524.135 PFLOPs | 1.193 PFLOPs | 526.179 PFLOPs | 194.9 min |
| Wan2.2-TI2V-5B T2V | 1280x704, 121f@24fps | 5.04s | 9.690 TFLOPs | - | 520.695 TFLOPs | 52.070 PFLOPs | 3.979 PFLOPs | 56.058 PFLOPs | 20.8 min |
| Wan2.2-TI2V-5B I2V | 1280x704, 121f@24fps | 5.04s | 9.690 TFLOPs | 414.739 TFLOPs | 520.695 TFLOPs | 52.070 PFLOPs | 3.979 PFLOPs | 56.473 PFLOPs | 20.9 min |

说明：

- A14B 的 720P 是官方 `1280x720`，TI2V-5B 的官方 720P 档是 `1280x704`，原因是 TI2V 总压缩/patch 对齐为 `4x32x32`。
- TI2V-5B 虽然帧数更多、步数更多，但 token 数 `27,280` 远小于 A14B 的 `75,600`，全局 self-attention 主项按 `S^2` 下降，因此整体算力约为 A14B 的 10.7%。
- TI2V-5B 的 VAE decode 静态估算高于 A14B，因为 Wan2.2 VAE 使用 `z_dim=48`、输出 `121` 帧；但端到端仍由 50 步 DiT 去噪主导。

### 10.2 A14B T2V 720P 端到端模块拆分

| 模块 | FLOPs |
| --- | ---: |
| T5 encoder cond+uncond | 9.690 TFLOPs |
| Wan DiT single forward | 6.552 PFLOPs |
| Wan DiT CFG per sampling step | 13.103 PFLOPs |
| Wan DiT denoise loop (40 steps) | 524.130 PFLOPs |
| VAE decode final latent (static estimate) | 1.193 PFLOPs |
| End-to-end generation estimate | 525.333 PFLOPs |

### 10.3 A14B I2V 720P 端到端模块拆分

| 模块 | FLOPs |
| --- | ---: |
| T5 encoder cond+uncond | 9.690 TFLOPs |
| Wan DiT single forward | 6.552 PFLOPs |
| Wan DiT CFG per sampling step | 13.103 PFLOPs |
| Wan DiT denoise loop (40 steps) | 524.135 PFLOPs |
| VAE encode image condition (static estimate) | 841.087 TFLOPs |
| VAE decode final latent (static estimate) | 1.193 PFLOPs |
| End-to-end generation estimate | 526.179 PFLOPs |

### 10.4 TI2V-5B T2V 720P-aligned 端到端模块拆分

| 模块 | FLOPs |
| --- | ---: |
| T5 encoder cond+uncond | 9.690 TFLOPs |
| Wan DiT single forward | 520.695 TFLOPs |
| Wan DiT CFG per sampling step | 1.041 PFLOPs |
| Wan DiT denoise loop (50 steps) | 52.070 PFLOPs |
| VAE decode final latent (static estimate) | 3.979 PFLOPs |
| End-to-end generation estimate | 56.058 PFLOPs |

### 10.5 TI2V-5B I2V 720P-aligned 端到端模块拆分

| 模块 | FLOPs |
| --- | ---: |
| T5 encoder cond+uncond | 9.690 TFLOPs |
| Wan DiT single forward | 520.695 TFLOPs |
| Wan DiT CFG per sampling step | 1.041 PFLOPs |
| Wan DiT denoise loop (50 steps) | 52.070 PFLOPs |
| VAE encode image condition (static estimate) | 414.739 TFLOPs |
| VAE decode final latent (static estimate) | 3.979 PFLOPs |
| End-to-end generation estimate | 56.473 PFLOPs |

### 10.6 A14B Wan DiT 单次前向内部拆分（720P）

| 模块 | FLOPs |
| --- | ---: |
| Patch embedding | 0.050 TFLOPs (T2V) / 0.111 TFLOPs (I2V) |
| Wan text embedding | 0.048 TFLOPs |
| Time embedding + modulation | 27.944 TFLOPs |
| Self-attn projections / layer | 15.854 TFLOPs |
| Self-attn QK+AV / layer | 117.051 TFLOPs |
| Cross-attn projections / layer | 7.981 TFLOPs |
| Cross-attn QK+AV / layer | 0.793 TFLOPs |
| FFN / layer | 21.404 TFLOPs |
| Norm + modulation / layer | 0.006 TFLOPs |
| 40 transformer blocks | 6.524 PFLOPs |
| Head + unpatchify linear | 0.050 TFLOPs |
| Single WanModel forward total | 6.552 PFLOPs |

### 10.7 TI2V-5B Wan DiT 单次前向内部拆分（720P-aligned）

| 模块 | FLOPs |
| --- | ---: |
| Patch embedding | 0.032 TFLOPs |
| Wan text embedding | 0.023 TFLOPs |
| Time embedding + modulation | 3.647 TFLOPs |
| Self-attn projections / layer | 2.060 TFLOPs |
| Self-attn QK+AV / layer | 9.145 TFLOPs |
| Cross-attn projections / layer | 1.049 TFLOPs |
| Cross-attn QK+AV / layer | 0.172 TFLOPs |
| FFN / layer | 4.806 TFLOPs |
| Norm + modulation / layer | 0.001 TFLOPs |
| 30 transformer blocks | 516.961 TFLOPs |
| Head + unpatchify linear | 0.032 TFLOPs |
| Single WanModel forward total | 520.695 TFLOPs |

### 10.8 T5 encoder 内部拆分

| 模块 | FLOPs |
| --- | ---: |
| Self-attn projections / layer | 0.069 TFLOPs |
| Self-attn QK+AV / layer | 0.004 TFLOPs |
| Gated FFN / layer | 0.129 TFLOPs |
| Single layer total | 0.202 TFLOPs |
| Single prompt 24-layer encoder | 4.845 TFLOPs |
| Cond + uncond prompts | 9.690 TFLOPs |

### 10.9 VAE 静态近似拆分

| 模块 | A14B Wan2.1 VAE | TI2V-5B Wan2.2 VAE |
| --- | ---: | ---: |
| VAE decode stage 1 | 0.301 TFLOPs | 1.159 TFLOPs |
| VAE decode stage 2 | 16.855 TFLOPs | 55.608 TFLOPs |
| VAE decode stage 3 | 134.842 TFLOPs | 444.867 TFLOPs |
| VAE decode stage 4 | 520.106 TFLOPs | 1.736 PFLOPs |
| VAE decode stage 5 | 520.106 TFLOPs | 1.736 PFLOPs |
| VAE decode stage 6 | 1.161 TFLOPs | 4.522 TFLOPs |
| VAE decode total | 1.193 PFLOPs | 3.979 PFLOPs |
| VAE encode image condition total (I2V only) | 841.087 TFLOPs | 414.739 TFLOPs |

### 10.10 算力热点排序

| 排名 | 热点 | 原因 |
| ---: | --- | --- |
| 1 | Wan DiT self-attn QK+AV | 720P tokens `S=75,600`，复杂度 `O(S^2*C)` |
| 2 | Wan DiT FFN | 40 层，每层 `5120 -> 13824 -> 5120` |
| 3 | Wan DiT QKV/O projection | 每层 4 个 `5120x5120` dense projection |
| 4 | VAE decode | 高分辨率 3D/2D conv，远小于 40 步 DiT 但大于 T5 |
| 5 | T5 encoder | 只运行 prompt/negative prompt 两次，量级约 9.69 TFLOPs |

## 11. 输入输出格式

### 11.1 T2V

输入：

- `input_prompt: str`
- `size: (width, height)`，官方默认可用 `(1280,720)`
- `frame_num=81`
- `seed`
- `sample_solver in {unipc, dpm++}`
- `sampling_steps=40`（A14B 配置）
- `guide_scale=(low, high)`

输出：

- `videos[0]`: RGB video tensor，shape `[3,81,H,W]`

### 11.2 I2V

输入：

- `input_prompt: str`
- `img: PIL.Image`，转 tensor 后 `[3,H_img,W_img]`
- `max_area=720*1280`
- `frame_num=81`
- 其余采样参数同 T2V

输出：

- `videos[0]`: RGB video tensor，shape `[3,81,H,W]`
- `H,W` 由输入图宽高比和 `max_area` 推导，并对 VAE stride/patch 对齐。

### 11.3 TI2V-5B

输入：

- `input_prompt: str`
- T2V: `size=(1280,704)` 或 `(704,1280)`
- I2V: `img: PIL.Image`，输出尺寸由 `best_output_size()` 对齐到 `patch_size * vae_stride = 32` 的倍数，默认 `max_area=704*1280`
- `frame_num=121`
- `sample_fps=24`
- `sampling_steps=50`
- `guide_scale=5.0`

输出：

- T2V: `videos[0]` RGB video tensor，shape `[3,121,704,1280]` 或 `[3,121,1280,704]`
- I2V: `videos[0]` RGB video tensor，shape `[3,121,H,W]`，`H,W` 按输入图宽高比和 `max_area` 推导。

## 12. 后续适配风险

1. Wan2.2 不是 LLM converter 能直接覆盖的模型；适配时应拆成 T5、DiT high/low、VAE 三段。
2. 720P token 数很大，self-attention 是主要算力瓶颈；若目标硬件不支持大 S 的高效 attention，需要先定义分块/sequence parallel/host fallback 策略。
3. I2V 的条件 `y` 是 DiT 输入契约的一部分，不是简单增加 `--image` 参数；导出或编译时必须覆盖 `in_dim=36`。
4. high/low expert 按 timestep 切换，两个 expert 结构相同但权重不同；部署可共享图结构，但不能共享权重。
5. CFG 使每个 timestep 做两次 DiT 前向；性能估算和图调度不能漏算 uncond 分支。
6. 本文未加载完整权重运行原生生成，未验证实际 dtype 转换后显存峰值；当前结论是结构与静态算力分析。
