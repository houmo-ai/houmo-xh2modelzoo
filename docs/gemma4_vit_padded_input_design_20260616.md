# Gemma4 ViT 官方 padded 输入适配方案

日期：2026-06-16

## 目标

将当前 Gemma4 ViT 从“固定真实 patch 数 / 固定方图常量化”的适配方式，改成更接近 Transformers 官方前处理的静态 padded 输入方案：

- Host 侧按官方 image processor 做 resize、patchify、pad；
- ViT 图输入固定为 max patches；
- 支持不同宽高比；
- position embedding 和 RoPE 使用 `pixel_position_ids`；
- pooling 使用 Host 侧生成的 `pool_indices`，避免 NPU 图内动态生成 pooling 映射。
- image 与 video 使用两套静态 ViT 导出尺寸；video 不再 pad 到 image 的 2520 patch 图上，避免每帧多算约 4 倍 ViT token。

---

## 1. 官方前处理输出

Gemma4 image processor 参数：

```text
patch_size = 16
pooling_kernel_size = 3
max_soft_tokens = 280
max_patches = max_soft_tokens * pooling_kernel_size^2 = 280 * 9 = 2520
patch_dim = 16 * 16 * 3 = 768
```

Host/CPU 前处理流程：

```text
原图
  -> aspect-ratio preserving resize，H/W 对齐到 48 的倍数
  -> rescale 到 [0, 1]
  -> patchify，patch_size=16
  -> pad 到 max_patches=2520
```

输出：

```text
pixel_values:        [1, 2520, 768]
pixel_position_ids:  [1, 2520, 2]
```

真实 patch：

```text
pixel_position_ids[i] = [x, y]
```

padding patch：

```text
pixel_position_ids[i] = [0, 0]
```

注意：这是本适配方案定义的 NPU 输入协议。HF 原生 processor 用 `[-1, -1]` 表示 padding，但本方案显式传入 `attention_mask` 和 `pool_indices`，不再依赖 `pixel_position_ids` 推导 padding，因此 padding position 可安全填 `[0, 0]`。

同时 Host 侧生成：

```text
pool_indices:             [1, 280, 9]
valid_soft_token_count:   scalar / host side metadata
visual_attention_mask:    [1, 1, 1, 2520]
```

注意：Python `BatchFeature` 中使用 `visual_attention_mask` 避免和文本侧 `attention_mask` 混淆；导出的 ViT ONNX 输入名仍是 `attention_mask`。

其中 `valid_soft_token_count` 用于 LLM 侧截断 ViT 输出。

### 1.1 Video 独立 padded 输出

Gemma4 官方 processor 对 video frame 使用更小的固定 patch 上限。以当前三套 Gemma4 配置的默认 video 前处理为准：

```text
video_max_soft_tokens = 70
video_max_patches = video_max_soft_tokens * pooling_kernel_size^2 = 70 * 9 = 630
```

Host/CPU 对每帧独立输出：

```text
pixel_values_videos:        [1, num_frames, 630, 768]
video_pixel_position_ids:   [1, num_frames, 630, 2]
video_pool_indices:         [1, num_frames, 70, 9]
video_visual_attention_mask:[1, num_frames, 1, 1, 630]
video_soft_token_count:     [1, num_frames]
```

当前默认 224x224 frame 的有效 patch 数是 576，对应 64 个 video soft token；ViT 图仍固定导出为 630 patch / 70 output token，LLM 侧按 `video_soft_token_count` 截断后再替换 `<|video|>` token。

因此导出产物必须包含两套视觉图：

```text
visual/       image VIT: [1, 2520, 768] -> [1, 280, hidden]
video_visual/ video VIT: [1, 630, 768] -> [1, 70, hidden]
```

两套图可以共享同一份 HF 权重来源，编译器/部署侧可继续做权重共享；但图输入 shape 不复用，video 不允许 pad 到 image VIT 上运行。

---

## 2. ViT 图输入

新增业务输入，并显式传入 attention mask；RoPE 在图内由常量 table 根据 `pixel_position_ids` gather。image VIT 的固定输入为：

```text
pixel_values:        [1, 2520, 768]   float
pixel_position_ids:  [1, 2520, 2]     int32
pool_indices:        [1, 280, 9]      int32
attention_mask:      [1, 1, 1, 2520]  float
```

video VIT 使用同一输入名与语义，但静态长度改为：

```text
pixel_values:        [1, 630, 768]   float
pixel_position_ids:  [1, 630, 2]     int32
pool_indices:        [1, 70, 9]      int32
attention_mask:      [1, 1, 1, 630]  float
```

`attention_mask` 语义：

```text
valid key:    0
invalid key:  large negative / -inf equivalent
```

mask 加在 attention logits 的 key 维度：

```python
attn_logits = q @ k.transpose(-2, -1)
attn_logits = attn_logits + attention_mask
attn_probs = softmax(attn_logits)
```

---

## 3. Patch embedding + position embedding

### 3.1 Pixel projection

```python
pixel_values = 2 * (pixel_values - 0.5)
hidden = input_proj(pixel_values)
```

shape：

```text
[1, 2520, 768] -> [1, 2520, 1152]
```

### 3.2 Position embedding

不用 HF 原始的：

```text
one_hot + matmul
```

改成基于 `pixel_position_ids` 的 gather：

```python
x = pixel_position_ids[..., 0]
y = pixel_position_ids[..., 1]

pos_embed = pos_table_x[x] + pos_table_y[y]
hidden = hidden + pos_embed
```

常量：

```text
pos_table_x: [10240, 1152]
pos_table_y: [10240, 1152]
```

注意：本方案中生成 PE 时不做 clamp。padding token 的 `pixel_position_ids` 由 Host 填 `[0, 0]`，不会产生负 index；padding token 的 PE 值不参与有效输出，依赖 attention mask 和 pool_indices 隔离无效 token。

---

## 4. Attention mask

`attention_mask` 作为显式输入传入 ViT：

```text
attention_mask: [1, 1, 1, 2520]
```

mask 只需要屏蔽 key 维度：

```text
valid key:    0
invalid key:  -inf / large negative
```

padding query 产生的输出不会被 pool 阶段读取，因此不需要额外屏蔽 query 行。

---

## 5. RoPE：常量 table + position gather

在线方案可以在 ViT 图内做 `pixel_position_ids * inv_freq -> Cos/Sin`，但当前 HMONNX 链路对图内 `Cos/Sin` 支持不满足稳定交付要求。最新落地方案改为：导出时预计算一维 RoPE `cos/sin` 常量表，运行时在图内用 `pixel_position_ids` 做 gather，既不额外传入 `rope_cos/rope_sin`，也不在图内放 `Cos/Sin`。生成时不做 clamp，padding token 的 `pixel_position_ids` 由 Host 填 `[0, 0]`。

当前配置：

```text
head_dim = 72
ndim = 2  # x/y
spatial_dim = head_dim // 2 = 36
inv_freq length = 18
rope_axis_dim = 36
max_positions = position_embedding_table.shape[1]  # 当前为 10240
```

导出时常量化：

```python
positions = arange(max_positions)[:, None]         # [10240,1]
freqs = positions * inv_freq[None, :]              # [10240,18]
emb = concat(freqs, freqs, dim=-1)                 # [10240,36]
rope_cos_table = cos(emb) * attention_scaling      # [10240,36]
rope_sin_table = sin(emb) * attention_scaling      # [10240,36]
```

运行时图内 gather：

```python
x = pixel_position_ids[..., 0]
y = pixel_position_ids[..., 1]

cos_x = gather(rope_cos_table, x)  # [1,2520,36]
cos_y = gather(rope_cos_table, y)  # [1,2520,36]
sin_x = gather(rope_sin_table, x)  # [1,2520,36]
sin_y = gather(rope_sin_table, y)  # [1,2520,36]

cos = concat(cos_x, cos_y, dim=-1) # [1,2520,72]
sin = concat(sin_x, sin_y, dim=-1) # [1,2520,72]
```

Attention 中：

```text
q/k: [1, 2520, num_heads, 72]
```

RoPE 维度划分：

```text
前 36 维：x position
后 36 维：y position
```

V 不做 RoPE。

padding position 填 `[0, 0]`，对应 RoPE position 0；无效 token 被 attention mask 和 pooling 隔离，不影响有效输出。

---

## 6. Pooling：Host 生成 pool_indices，NPU 做 gather + reduce

不在 NPU 图内通过 `position_ids` 生成 pooling 矩阵，避免：

```text
max / floor_div / one_hot / scatter / dynamic index generation
```

Host 侧生成：

```text
pool_indices: [1, 280, 9]
```

NPU 图内只做：

```python
gathered = gather(hidden, pool_indices, axis=1)
# [1, 280, 9, 1152]

pooled = reduce_sum(gathered, axis=2) * (1 / 9)
# [1, 280, 1152]
```

### 6.1 pool_indices 生成逻辑

根据当前图的有效 patch grid：

```text
grid_h = resized_h / 16
grid_w = resized_w / 16
pooled_h = grid_h / 3
pooled_w = grid_w / 3
valid_soft_token_count = pooled_h * pooled_w
```

生成 3x3 pooling index：

```python
pool_indices = zeros([280, 9])

for py in range(pooled_h):
    for px in range(pooled_w):
        out_idx = py * pooled_w + px
        ids = []
        for dy in range(3):
            for dx in range(3):
                x = px * 3 + dx
                y = py * 3 + dy
                patch_idx = y * grid_w + x
                ids.append(patch_idx)
        pool_indices[out_idx] = ids
```

无效行：

```text
out_idx >= valid_soft_token_count
```

可以填 0，因为 LLM 侧会按 `valid_soft_token_count` 截断。

---

## 7. Pool 后处理和 embed_vision

HF pooler 之后还有 scale：

```python
pooled = pooled * sqrt(vision_hidden_size)
```

当前：

```text
vision_hidden_size = 1152
```

如果 `standardize=True`：

```python
pooled = (pooled - std_bias) * std_scale
```

然后接 `embed_vision`：

```python
pooled = RMSNorm(1152, with_scale=False)(pooled)
image_embeds = Linear(1152 -> 5376)(pooled)
```

输出固定：

```text
image_embeds: [1, 280, 5376]
```

LLM 侧使用：

```python
image_embeds = image_embeds[:, :valid_soft_token_count, :]
```

再替换文本中的 `<image>` token embedding。

---

## 8. 与 HF 的等价关系

HF 原生 ViT 输出动态长度：

```text
[valid_soft_token_count, 5376]
```

本方案输出静态长度：

```text
[1, 280, 5376]
```

LLM 侧截断前 `valid_soft_token_count` 个 token 后，与 HF 语义对齐。

---

## 9. Review

### 9.1 优点

1. 输入 shape 静态：

```text
pixel_values:       [1,2520,768] fp16
pixel_position_ids: [1,2520,2]   int32
pool_indices:       [1,280,9]    int32
attention_mask:     [1,1,1,2520] fp16
```

2. 支持不同宽高比：

不同宽高比只影响：

```text
pixel_position_ids
pool_indices
attention_mask
valid_soft_token_count
```

图结构保持不变。

3. PE 更 NPU 友好：

```text
Gather + Add
```

替代：

```text
OneHot + MatMul
```

4. Pooling 更 NPU 友好：

```text
Gather + ReduceSum
```

替代：

```text
OneHot / dense sparse MatMul
```

5. RoPE 不使用图内 Cos/Sin：

```text
导出时预计算 rope cos/sin 常量表，运行时按 pixel_position_ids gather。
```

### 9.2 风险和待确认点

1. `pixel_position_ids` 中 padding 由 Host 填 `[0,0]`。
   - PE 和 RoPE 本方案不 clamp。
   - padding token 的 PE/RoPE 值不会被有效输出使用。

2. attention mask 必须屏蔽 key 维度。
   - mask shape 推荐 `[1,1,1,2520]`。
   - valid key = 0，invalid key = large negative。

3. `pool_indices` 必须由 Host 侧根据当前 `grid_h/grid_w` 生成。
   - 不能复用固定方图的 indices 到非方图。

4. ViT 输出 `[1,280,5376]` 后，LLM 侧必须按 `valid_soft_token_count` 截断。
   - 文本中 `<image>` token 数也必须等于 `valid_soft_token_count`。

5. RoPE 当前使用常量 table + gather。
   - `pixel_position_ids/pool_indices` 需要按导出 ONNX 协议传 int32；
   - 优点是图内不含 `Cos/Sin`，也不需要额外传 `rope_cos/rope_sin`；
   - 代价是每层 RoPE 前多一次按 x/y 坐标的 table gather。

---

## 10. 建议落地顺序

1. 写 PyTorch reference wrapper：
   - 输入 `pixel_values / pixel_position_ids / pool_indices / attention_mask`；
   - PE gather；
   - RoPE 使用常量 table + `pixel_position_ids` gather；
   - attention mask add；
   - pool gather + reduce。

2. 与 HF 原生 `vision_tower + embed_vision` 对齐：
   - 方图：`224x224 / 448x448 / 896x896`；
   - 非方图：`224x448 / 896x1344`；
   - 比较前 `valid_soft_token_count` 个 image embeddings。

3. 再导出 HMONNX：
   - 验证 Gather / attention mask add / GatherND-like pool / ReduceSum；
   - 确认导出的 visual ONNX 不含 `Cos/Sin`。

---

## 结论

新的 Gemma4 ViT 方案：

```text
官方 padded 2520 patch 输入
+ pixel_position_ids 输入（padding 填 `[0,0]`）
+ pool_indices 输入
+ attention_mask 输入
+ PE gather
+ RoPE 常量 table + pixel_position_ids gather
+ attention logits mask add
+ pool_indices gather/reduce
+ 输出固定 280 image embeddings，LLM 侧按 valid_soft_token_count 截断
+ video 另导出固定 70 video embeddings 的 video_visual 图，不 pad 到 image VIT
```

该方案在保持静态 shape 的同时支持不同宽高比，并尽量避免 NPU 不友好的 one_hot / 动态 pool index 生成 / 图内 SinCos。
