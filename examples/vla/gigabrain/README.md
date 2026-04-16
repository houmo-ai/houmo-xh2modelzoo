# GigaBrain-0.1 XH2A 适配方案

本目录包含 GigaBrain-0.1-3.5B-Base 模型的 XH2A 部署适配代码。

## 模型结构

GigaBrain-0.1 是一个 Vision-Language-Action (VLA) 模型，主要包含以下组件：

```
GigaBrain-0.1
├── Vision Encoder (SigLIP)     # 视觉编码器, 27层, 1152 hidden
│   ├── Embeddings
│   ├── Encoder Layers
│   └── Post LayerNorm
├── Vision Projector            # 视觉投影层: 1152 → 2304
├── PaliGemma2 LLM              # 语言模型主体 (26层, 2304 hidden, GQA 4KV)
│   ├── Embeddings
│   ├── Transformer Layers (Gemma2架构)
│   │   ├── Grouped Query Attention (8Q, 4KV)
│   │   ├── RMSNorm + Gate
│   │   └── MLP (intermediate: 9216)
│   └── LM Head (tied with embed)
├── Action Expert               # 动作专家 (26层, 1024 hidden)
│   ├── Shared Attention (与LLM共享KV)
│   ├── AdaRMS 条件调制
│   └── Expert MLP (intermediate: 2048)
└── Action Heads
    ├── action_in_proj          # 动作输入投影: (B,50,32) → (B,50,1024)
    ├── action_out_proj         # 动作输出投影: (B,50,1024) → (B,50,32)
    └── time_mlp                # 时间步编码: (B,1024) → (B,1024) [AdaRMS cond]
```

## 核心技术特点

### 1. 权重注入方案 (Weight Injection)

为了适配 XH2A 部署框架，我们采用了**权重注入方案**：

- **LLM 导出**: 使用 `create_and_inject_llm_skeleton()` 构建标准 `Gemma2ForCausalLM` 骨架，将 GigaBrain 的 PaliGemma2 权重注入
- **Expert 导出**: 使用 `create_and_inject_expert_skeleton()` 构建包含 AdaRMS 和 Expert MLPs 的标准模型骨架

这种方式避免了直接继承复杂的 HuggingFace 模型，确保了与 XHModel 框架的兼容性。

### 2. 共享 Attention 机制

GigaBrain 的 Action Expert 与 PaliGemma2 LLM **共享 Attention 层**：
- KV Cache 由 LLM 生成并传递给 Expert
- Expert 仅执行自己的 MLP 计算
- 通过 AdaRMS 实现条件调制

### 3. 多机器人支持 (Embodiment)

支持多种机器人平台，通过 `emb_ids` 区分：
- `emb_ids=0`: AgileX Cobot Magic (action_dim=14)
- `emb_ids=1`: Agibot G1 (action_dim=20)
- `emb_ids=2`: Other (action_dim=32)

## 导出流程

### 环境准备

```bash
# 安装依赖
conda activate xhquant

# 确保 giga_models 可访问
export PYTHONPATH="/path/to/giga-models:$PYTHONPATH"
```

### 1. 导出 Vision Encoder

```bash
python gigabrain_export_vision_xh2a.py \
    --model-path /path/to/GigaBrain-0.1-3.5B-Base \
```

**输出文件**:
- `workdir/gigabrain_vision.onnx` - Vision Encoder ONNX 模型
- `hmonnx/vision.onnx` - HMONNX 量化模型

**输入/输出规格**:
- 输入: `pixel_values` (B, 3, 224, 224) float32
- 输出: `image_features` (B, 256, 2304) float32

### 2. 导出 LLM (PaliGemma2)

```bash
python gigabrain_export_llm_xh2a.py \
    --config config/gigabrain/llm/gigabrain_llm_xh2a.py \
```

**核心代码逻辑**:
```python
from xh_model_zoo.xh_llm.models.gigabrain.weight_transfer import create_and_inject_llm_skeleton

# 构建标准 Gemma2 骨架并注入权重
standard_llm = create_and_inject_llm_skeleton(policy)
xh_model.init_wrap_model(standard_llm)
```

**输出文件**:
- `work_dirs/gigabrain_llm_xh2a/prefill_onnx/gigabrain_llm_xh2a_prefill.onnx`
- `work_dirs/gigabrain_llm_xh2a/decode_onnx/gigabrain_llm_xh2a_decode.onnx`
- `work_dirs/gigabrain_llm_xh2a/export_meta_info.json`
- `work_dirs/gigabrain_llm_xh2a/token_embedding.pt`

**输入/输出规格**:
- Prefill: `input_ids` (B, seq_len) → `logits` (B, seq_len, vocab_size)
- Decode: `input_ids` (B, 1) + `past_seq_length` → `logits` (B, 1, vocab_size)

### 3. 导出 Action Expert

```bash
python gigabrain_export_experts_xh2a.py \
    --config config/gigabrain/llm/gigabrain_expert_xh2a.py \
```

**核心代码逻辑**:
```python
from xh_model_zoo.xh_llm.models.gigabrain.weight_transfer import create_and_inject_expert_skeleton

# 构建包含 AdaRMS 的标准 Expert 骨架
standard_expert = create_and_inject_expert_skeleton(policy)
xh_model.init_wrap_model(standard_expert)
```

**输出文件**:
- `work_dirs/gigabrain_expert_xh2a/prefill_onnx/gigabrain_expert_xh2a_prefill.onnx`
- `work_dirs/gigabrain_expert_xh2a/decode_onnx/gigabrain_expert_xh2a_decode.onnx`
- `work_dirs/gigabrain_expert_xh2a/export_meta_info.json`

**输入/输出规格**:
- Prefill: `input_ids` (B, seq_len) → `hidden_states` (B, seq_len, 1024)
- Decode: `input_ids` (B, 50) + `past_seq_length` → `hidden_states` (B, 50, 1024)

### 4. 导出其他模块 (Action Heads, Time MLP)

```bash
python gigabrain_export_other_xh2a.py \
    --model-path /path/to/GigaBrain-0.1-3.5B-Base \
    --output-dir ./workdir
```

**输出文件**:
| 文件 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `action_in_proj.onnx` | action (B,50,32), emb_ids (B) | action_emb (B,50,1024) | 动作输入投影 |
| `action_out_proj.onnx` | hidden (B,50,1024), emb_ids (B) | action (B,50,32) | 动作输出投影 |
| `time_mlp.onnx` | time_emb (B,1024) | adarms_cond (B,1024) | AdaRMS 条件 |
| `embodiment_config.json` | - | - | 机器人类型配置 |

## 文件说明

| 文件 | 说明 |
|------|------|
| `gigabrain_export_vision_xh2a.py` | 导出 SigLIP 视觉编码器和投影层 (ONNX + HMONNX) |
| `gigabrain_export_llm_xh2a.py` | 导出 PaliGemma2 LLM (权重注入 + PTQ 量化) |
| `gigabrain_export_experts_xh2a.py` | 导出 Action Expert (权重注入 + PTQ 量化) |
| `gigabrain_export_other_xh2a.py` | 导出 Action Heads 和 Time MLP |
| `gigabrain_model_analysis.py` | 模型结构分析文档 |
| `config/gigabrain/llm/*.py` | XHModel 配置文件 |

## 与 PI0.5 的区别

| 特性 | PI0.5 | GigaBrain-0.1 |
|------|-------|---------------|
| LLM 架构 | PaliGemma (Gemma) | PaliGemma2 (Gemma2) |
| LLM Layers | 18 | 26 |
| LLM Hidden | 2048 | 2304 |
| LLM Intermediate | 16384 | 9216 |
| KV Heads | 1 (MQA) | 4 (GQA) |
| Expert Hidden | 1024 | 1024 |
| Expert Intermediate | 4096 | 2048 |
| Action Expert | 独立 Gemma 模型 | 共享 Attention |
| Action Dim | 14 (固定) | 14-32 (多机器人) |
| Action Steps | 50 | 50 |
| Embodiment | 单一 | 多机器人支持 |
| LayerNorm | 标准 Gemma | Gemma2 (更多 LN) |
| Attn Softcapping | 无 | 有 (可选) |

## 推理流程

```
1. 图像编码
   images (B, 3, 224, 224) → Vision Encoder → image_features (B, 256, 2304)

2. 语言编码
   task text → Tokenizer → input_ids → Embedding → lang_embeddings

3. LLM Prefill (生成 KV Cache)
   [image_features, lang_embeddings] → LLM Prefill → KV Cache

4. 动作生成 (Flow Matching 扩散)
   for t in timesteps:
       noise = sample_noise()
       action_in = action_in_proj(noise_action, emb_ids)
       time_emb = time_embedding(t)
       cond = time_mlp(time_emb)
       
       # Expert Prefill/Decode (使用 LLM 的 KV Cache)
       hidden = Expert(action_in, cond, KV Cache)
       
       action_out = action_out_proj(hidden, emb_ids)
       v_t = action_out  # 去噪速度

5. 后处理
   action → Unnormalize → Absolute Actions
```

## 输出目录结构

导出完成后的目录结构：

```
work_dirs/
├── gigabrain_llm_xh2a/
│   ├── prefill_onnx/
│   │   └── gigabrain_llm_xh2a_prefill.onnx
│   ├── decode_onnx/
│   │   └── gigabrain_llm_xh2a_decode.onnx
│   ├── hf_config/
│   │   ├── tokenizer.json
│   │   └── tokenizer_config.json
│   ├── token_embedding.pt
│   └── export_meta_info.json
├── gigabrain_expert_xh2a/
│   ├── prefill_onnx/
│   │   └── gigabrain_expert_xh2a_prefill.onnx
│   ├── decode_onnx/
│   │   └── gigabrain_expert_xh2a_decode.onnx
│   └── export_meta_info.json
workdir/
├── gigabrain_vision.onnx
├── gigabrain_action_in_proj.onnx
├── gigabrain_action_out_proj.onnx
├── gigabrain_time_mlp.onnx
└── embodiment_config.json
hmonnx/
├── vision.onnx
├── action_in_proj.onnx
├── action_out_proj.onnx
└── time_mlp.onnx
```

## 注意事项

1. **内存需求**: LLM 和 Expert 导出需要较大内存，建议使用 GPU 进行 PTQ 量化
2. **权重注入**: 确保 `xh_model_zoo.xh_llm.models.gigabrain.weight_transfer` 模块可访问
3. **Embodiment ID**: 不同机器人需要传入对应的 `emb_ids`
4. **KV Cache 共享**: Expert 推理时需要使用 LLM 生成的 KV Cache
