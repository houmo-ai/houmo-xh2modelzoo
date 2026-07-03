# Gemma4 Series 单 Prefill 与 vLLM 对齐设计

## 背景与目标

Gemma4 Series 当前 Merak 推理流程曾采用双 prefill 图：
`prefill(256)` 处理普通 chunk，`prefill_mm(320)` 处理超过 256 的
视觉原子区间。这能覆盖 image 280 token 的场景，但带来了两个问题：

1. 导出和编译需要维护两个 prefill 图，增加产物和调度复杂度。
2. runtime 需要在 `prefill` 与 `prefill_mm` 间切换，不利于和 vLLM 的
   chunked prefill 语义保持一致。

新的目标是对齐 vLLM 的 Gemma4 推理语义：只保留一个固定 shape 的
`prefill` 图，将 `input_sequence_length / prefill_chunk_length` 统一为
320，并通过调度保证 image / video frame 的视觉特征不会被 chunk 切开。

同时，`context_max_length` 不应被限制在 `[2048, 8192]`。它只需要满足
不小于 `input_sequence_length`，例如 `context_max_length=13107` 应合法。
`max_pe_length` 仍按已有独立逻辑处理，不由 `context_max_length` 推导。

## 设计原则

- 只保留一个 `prefill` 图，删除 `prefill_mm` 的导出、meta 和 runtime
  调度语义。
- 默认 `prefill_chunk_length=320`，并硬校验 `prefill_chunk_length >= 280`。
- `context_max_length` 只校验 `context_max_length >= prefill_chunk_length`。
- prefill chunk 尽量 greedy packing 到 320，但不能切开视觉原子 range。
- sliding attention 层应用视觉双向 overlay；full attention 层保持普通
  causal masksoftmax。
- MTP、target decode、target verify 的 cache ABI 统一跟随 320。

## 导出契约

### 单 prefill 图

Gemma4 Series 导出后只应存在一个 LLM prefill 图：

```text
prefill(input_sequence_length=320)
```

不再导出：

```text
prefill_mm
prefill_mm_hmonnx
```

meta / manifest 不应再表达双 prefill 产物。若继续保留 `prefill_graphs`
字段，也只能描述单图：

```json
{
  "prefill": {
    "input_sequence_length": 320
  }
}
```

更推荐 runtime 直接以 `model_config.prefill_chunk_length` 作为唯一 prefill
shape 来源，避免 `prefill_graphs` 被误解为多图调度机制。

### input sequence length 校验

新增固定下限：

```text
prefill_chunk_length >= 280
```

默认和推荐值为 320。原因：Gemma4 image 常见 soft token 数为 280，video
单帧常见为 70。320 能覆盖单张图，同时给 video 多帧与文本 greedy packing
留出空间。

若历史配置仍显式传入 `mm_prefill_chunk_length`，应 fail-fast，并提示：

```text
Gemma4 Series no longer supports mm_prefill_chunk_length/prefill_mm;
use prefill_chunk_length instead.
```

这样能避免旧双图语义在配置层“看起来还有效”。

### context 校验

删除以下限制：

```text
context_max_length in [2048, 8192]
```

改为：

```text
context_max_length >= prefill_chunk_length
```

因此 `context_max_length=13107` 应通过校验。`context_max_length` 只作为
上下文容量/运行约束，不参与 `max_pe_length` 推导。

## Prefill Chunk 调度

### 基础规则

所有 prefill 调用的图 shape 都是 320。每个 chunk 的真实有效长度由
`current_input_length` 表达，padding token 不参与有效 attention。

调度器按 token 顺序 greedy packing：

```text
当前 chunk 剩余空间足够放下下一个原子 range -> 放入当前 chunk
当前 chunk 边界会切开原子 range -> 在该 range 前断开
range 放入后还有空间 -> 继续放后续 text 或 frame
```

text token 可切分；视觉原子 range 不可切分。

### image 原子 range

单张 image 的 visual soft token range 是一个原子 range，通常为 280。

示例：

```text
[image 280][text 40] -> 一个 prefill(320) chunk
[image 280][text 41] -> chunk0: 320, chunk1: 1
```

### video 原子 range

对齐 vLLM：video 的原子单位是“单帧”，不是整段 video clip。

vLLM Gemma4 video replacement 形态为：

```text
timestamp + BOI + video_token * N + EOI
```

`PromptUpdateDetails.select_token_id(..., video_token_id)` 只将连续 video
token 视为 embed 区域；timestamp / BOI / EOI 会自然隔开不同帧。因此
`mm_prefix_range` 是每帧一个 range。

Merak 侧也应保持等价语义：同一帧的约 70 个 video feature token 不能被
切开，多帧可以在帧边界之间切分，也可以多个完整帧放入同一个 320 chunk。

示例：

```text
[frame0 70][frame1 70][frame2 70][frame3 70][text <= 40]
  -> 一个 prefill(320) chunk
```

如果下一帧会跨过 320 边界，则在该帧前断开，下一个 chunk 从该帧开始。

### audio

audio 不参与 vision bidirectional overlay，可按普通 token chunk 处理。E2B 和
E4B 需要覆盖真实 audio generate；31B 和 26B-A4B 按能力矩阵标为不支持。

## Attention Mask 语义

### sliding attention

sliding attention 层使用：

```text
base = causal sliding window
final = base OR same_visual_atomic_range(q, k)
```

其中 `same_visual_atomic_range` 只在同一个 image 或同一帧 video 内成立。
如果一个 chunk 中包含多个 frame range：

- frame 内 token 可双向可见；
- frame 与 frame 之间不双向；
- text token 不获得未来可见性；
- padding 不参与 attention。

因此不能只用“chunk 包含 MM”来构造 mask，mask 必须保留 chunk 内多个
visual atomic range 的边界。

### full attention

full attention 层保持普通 causal mask，不应用 visual bidirectional overlay。
Merak 侧继续走标准 `xhquant.nn.masksoftmax` / causal masksoftmax 路径，不再
恢复 full attention mask 输入。

## MTP 与 KV Cache ABI

MTP 与 target 侧统一跟随 `input_sequence_length=320`。

sliding KV cache 输入/输出长度按：

```text
sliding_window + input_sequence_length
```

在默认 `sliding_window=1024`、`input_sequence_length=320` 下，即：

```text
1024 + 320 = 1344
```

E2B/E4B 若模型 sliding window 为 512，则为：

```text
512 + 320 = 832
```

MTP draft 不需要区分输入 cache 来自 slice 还是 full。KV cache 原值返回，
通过 mask 控制有效范围，与 vLLM 等价。

`accepted_count` 仅在 slice/sliding verify 导出模型中出现，用于编译器在
verify 阶段处理 sliding KV cache rollback。初始上一轮 `accepted_count=0`。
full attention verify 不需要该参数。

## 实现落点

### 导出与配置

- `export_plan.py`
  - 删除 `ALLOWED_CONTEXT_MAX_LENGTHS` 的硬限制。
  - 新增 `MIN_INPUT_SEQUENCE_LENGTH = 280`。
  - 默认/推荐 `prefill_chunk_length = 320`。
  - 校验 `context_max_length >= input_sequence_length`。
  - 拒绝显式 `mm_prefill_chunk_length`。

- workflow YAML / template
  - 将 Gemma4 Series 默认 prefill 配置改为 320。
  - 删除 `mm_prefill_chunk_length` 默认项。

- LLM 导出
  - 删除 `prefill_mm` 导出分支。
  - meta / manifest 不再产生 `prefill_mm_hmonnx`。

### Runtime

- HMONNX runtime 只加载 `prefill`，不再解析 `prefill_mm`。
- 删除按 chunk 长度选择 `prefill` / `prefill_mm` 的逻辑。
- chunk planner 输出的 `graph_name` 恒为 `prefill`，`graph_length` 恒为 320。
- chunk planner 需要输出或保留 chunk 内 visual atomic range 边界，供 sliding
  mask overlay 使用。

## 测试矩阵

### 单元测试

导出契约：

- `prefill_chunk_length=256` 报错。
- `prefill_chunk_length=280` 合法。
- `prefill_chunk_length=320` 合法。
- `context_max_length=13107, prefill_chunk_length=320` 合法。
- `context_max_length < prefill_chunk_length` 报错。
- 显式 `mm_prefill_chunk_length` 报错。

chunk planner：

- text-only 长 prompt 按 320 切。
- image 280 + text 40 -> 一个 chunk。
- image 280 + text 41 -> 两个 chunk。
- 多个 video frame 可 greedy packing 到同一 chunk。
- 下一帧会跨 320 时在帧前断开。
- 单个 visual atomic range > 320 报错。

mask：

- 同一 image/frame 内双向可见。
- 不同 frame 之间不双向。
- text 不可看未来。
- padding 不参与 attention。
- full attention 层保持 causal masksoftmax。

runtime/meta：

- 不再要求 dual `prefill_graphs`。
- 不加载 `prefill_mm_hmonnx`。
- demo 调度日志中不出现 `prefill_mm`。

### 完整验收

四个模型均需覆盖：

```text
weights/gemma-4-E2B-it
weights/gemma-4-E4B-it
weights/gemma-4-31B-it
weights/gemma-4-26B-A4B-it
```

#### 1. 四模型完整非 MTP 导出

每个模型跑完整 target 导出链路，并检查：

- target HMONNX 产物完整；
- image `visual` 产物存在；
- `video_visual` 独立产物存在；
- E2B/E4B `audio` 产物存在，31B/26B-A4B 标 N/A；
- `prefill` 为单图 320；
- 不存在 `prefill_mm` / `prefill_mm_hmonnx`；
- `context_max_length=13107` 不被白名单卡住。

#### 2. 四模型完整 MTP 导出

每个模型跑完整 speculative / MTP 导出链路，并检查：

- MTP draft HMONNX 导出成功；
- target verify ABI 正确；
- sliding KV cache 输入跟随 `sliding_window + 320`；
- MTP KV cache 原值返回，通过 mask 控制有效范围；
- `accepted_count` 只出现在 slice/sliding verify 导出模型；
- `accepted_count` 初始上一轮值为 0。

#### 3. 四模型非 MTP runtime demo

每个模型跑 text / image / video demo。E2B/E4B 额外跑 audio demo；31B 和
26B-A4B audio 标 N/A。

日志需要证明：

- 调度只调用 `prefill`；
- image 不被切开；
- video 单帧不被切开；
- 多帧可以 greedy packing 到 320；
- chunk 内多 visual range 的 mask 独立生效。

#### 4. 四模型 speculative MTP runtime demo

每个模型跑 target HMONNX + MTP draft HMONNX 的 speculative generate，并记录：

- proposed / accepted / rounds；
- `accepted_count_inputs`；
- sliding KV cache rollback 行为；
- 输出文本是否正常。

## 非目标

- 不重新引入 dual prefill。
- 不让 `context_max_length` 参与 max PE 推导。
- 不改变 Gemma4 full attention 为 vision bidirectional。
- 不把整段 video clip 作为不可切原子单位。
- 不把 audio 纳入 vision bidirectional overlay。
