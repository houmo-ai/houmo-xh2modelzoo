# Gemma4

Gemma4-31B-IT 在 `xhmodel_merak` 中的完整适配，覆盖 wrap → frontend → quant → HMONNX 全链路导出与推理。

## 环境

Gemma4 需要 `transformers>=5.5.0`。使用专用 conda 环境：

```bash
source /data01/home/yujy/miniconda3/etc/profile.d/conda.sh
conda activate gemma4
```

> **注意**：不要 `source env.sh`（它激活 xhquant 环境，transformers 5.3 不支持 Gemma4）。

## 导出

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n gemma4 bash -c \
  'PYTHONPATH=/data01/home/yujy/work/xh2modelzoo:$PYTHONPATH \
   python examples_merak/llm/gemma4/gemma4_xh_export_hmonnx.py \
   --config configs_merak/xh2a/llm_models/gemma4/31b/gemma4_31b_it_xh2a_2k.py'
```

产出目录 `work_dirs/gemma4_31b_it_xh2a_2k/hmquant_xh2_gemma4_31b_it_w8a8_256_2k_<date>/`：
- `visual/` — 视觉编码器 HMONNX
- `prefill/` — 文本 prefill HMONNX
- `decode/` — 文本 decode HMONNX
- `quant_embedding.pt` — 已缩放的 token embedding（含 embed_scale=73.5）
- `golden_meta_info.json` — 推理元信息

## 推理（Golden 模式）

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n gemma4 bash -c \
  'PYTHONPATH=/data01/home/yujy/work/xh2modelzoo:$PYTHONPATH \
   python examples_merak/llm/gemma4/gemma4_xh_hmonnx_generate.py \
   --golden --prompt "Describe this image." --image-path ./data/images/houmo_logo.jpg'
```

不加 `--golden` 则使用 xh2a 后端推理。`--fast` 启用快速模式。

## 模型结构要点

| 特性 | 说明 |
|---|---|
| 层数 | 60（50 sliding + 10 full attention，5:1 交替） |
| Sliding attention | head_dim=256, kv_heads=16, rope_theta=10000 |
| Full attention | head_dim=512, kv_heads=4, rope_theta=1e6 |
| Attention scaling | 1.0（Q/K 已 RMS-normalized，非 1/√d） |
| RMSNorm 变体 | with_scale=True (q/k_norm) 和 with_scale=False (v_norm) |
| Embedding | ScaledWordEmbedding × 73.5，导出时已 bake-in |
| Logit softcapping | tanh(logits/30) × 30 |
| Vision | SigLIP encoder + bidirectional attention + avg pooling |

## 精度验证结果

| 阶段 | 精度 (cosine similarity) |
|---|---|
| Vision wrap | 1.0000 |
| Text wrap prefill | 0.9999 |
| Text wrap decode | 0.9998 |
| Vision frontend | 1.000043 |
| Text frontend prefill | 0.999975 |
| Text frontend decode | 0.999982 |
| E2E text generate | ✅ "I am a large language model, trained by Google..." |
| E2E image generate | ✅ "A vertical, centered image on a white background features..." |

## 已修复的关键问题

1. **Attention scaling**：HF `scaling=1.0`，wrap 原为 `1/√256=0.0625`（16× 误差）
2. **Embedding scale**：`embed_scale=73.5` 需 bake 到权重中以兼容 `nn.Embedding`
3. **Logits padding**：prefill 输出 512 位置需截断到实际 seq_length
4. **RoPE Sin/Cos**：xh2a 不支持，改为预计算表 + `F.embedding` lookup（导出为 Gather）
5. **Vision Clip-on-int32**：xh2a 不支持，改为 `torch.where + F.embedding`
6. **FX tracing**：`Gemma4TextRotaryEmbedding` 注册为 leaf module，动态切片改为常量
