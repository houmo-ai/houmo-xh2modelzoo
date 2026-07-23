# Gemma4 12B Unified 适配设计

> 状态：Phase 0-5 主链路已实现；BF16/HMONNX、W4G64 GPTQModel/AutoRound、
> MTP assistant 与 vllm-merak 多模态部署适配已完成；物理 NPU、量化精度评测和
> 在线服务端到端压测仍待硬件闭环

## 实施验证快照（2026-07-23）

- 权重 LFS 完整性与 safetensors header 已验证，SHA256：
  `5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d`；
- 独立环境 `xhquant_gemma4_12b` 使用 Transformers 5.13.0；
- checkpoint-derived 合同固定为 image=280、video=70/frame、audio=750、
  vision input=6912、audio input=640、position capacity=1120；
- 真实权重 vision/audio adapter 与 HF eager 数值一致；vision HMONNX 仅接收
  `pixel_values + normalized_position_ids`，audio HMONNX 仅接收 `input_features`；
  mask 不进入图，Host 按原始 position/mask 过滤有效输出 token；
- 当前无-mask实现已重新导出真实 side HMONNX，物理 graph input 分别为
  visual/video_visual 两输入和 audio 单输入；
- Unified text 六类核心 HF module 已进入现有 Gemma4 lowering，缩小版 Unified
  模型完成动态 wrap；
- processor 覆盖图像、视频 metadata、音频边界、本地 WAV、image+audio mixed；
- 真实 12B 权重完成 base export，产出 visual/video_visual/audio/prefill/decode
  五类 HMONNX；aligned CUDA runtime 的 text/image/video/audio generate 均成功，
  输出前缀分别包含“thought”“画面中”“这些视频”“这段声音”；
- HM-style Golden 已落盘：visual 1 step、video_visual 16 steps、audio 1 step、
  prefill 1 step、decode 1 step；
- assistant 正式下载目录 `weights/gemma-4-12B-it-assistant` 与镜像权重
  SHA256 一致（`3279c173...6d6df`），确认其为 4 层 Unified assistant；
- MTP draft 真实权重 eager forward 与 W8A8-body/W4A8-head HMONNX smoke export
  通过，输出为 262144-vocab logits 与 3840-dim feedback hidden；
- vllm-merak 新增独立 Unified loader，processor 固定 config-owned 280，并修复
  raw audio `ceil(samples/640)` placeholder 计数；image/audio/video processor smoke 通过；
- GPTQModel W4G64 使用 102 条 IVSG、512 token 校准，48 层量化耗时约 23 分 48 秒；
  quant HF reload 和文本生成通过；
- AutoRound W4G64 使用 128 条 Pile、2048 token、batch=8、iters=200 校准，量化耗时
  2360.5 秒，总耗时 2386.3 秒；quant HF reload 和中文文本生成通过；
- 两份 W4G64 产物均重新导出 visual/video_visual/audio/prefill/decode 五类 HMONNX，
  目录名保留 W4A8，side graph 物理输入无 mask，meta 仍固定 image=280；
- 量化实跑发现并修复三项 Unified 专属问题：sliding/full RoPE 宽度不能跨层复用、
  最后 KV producer 层必须收到 `shared_kv_states` 字典、HF save 不能丢失
  `model_patch_size`/`audio_samples_per_token` 等 checkpoint-only 配置字段；
- GPTQModel/AutoRound 定向 13 项、vllm-merak 全量 390 项、Gemma4 定向/public
  74 项回归通过。
>
> 设计基线：2026-07-23；vLLM `069ba34ed2`；Transformers `5.13.0`；
> checkpoint config commit `a69a4a1`（`max_position_embeddings=262144`）。

## 1. 结论

Gemma4 12B 不是 E2B/E4B 的小变体，也不是 31B/26B-A4B 的同构缩放版。
它使用独立的 HF 架构：

```text
architectures = ["Gemma4UnifiedForConditionalGeneration"]
model_type   = "gemma4_unified"
```

它与现有 Gemma4 Series 可以复用的是 **LLM 推理骨架、KV cache、
full/sliding attention、multimodal scatter、workflow/runtime 外壳**；不能复用的是
**旧 vision tower、3x3 pooling 图、audio tower、audio stride-conv token 计数**。

推荐实现方式：

1. `Gemma4UnifiedForConditionalGeneration` 作为同一个
   `Gemma4SeriesWorkflow` 的第二个 HF architecture alias，不新建另一套 public
   workflow。
2. variant resolver 新增 `12b_unified`，并新增结构维度
   `frontend_kind="encoder_free"`；优先按顶层 `model_type/architectures` 判断，不能再按
   `has_audio` 把它误判为 E4B。
3. 保持现有 text graph，泛化其 HF class 适配；为 Unified 单独实现轻量 vision/audio
   frontend adapter，避免在旧 tower adapter 中堆条件分支。
4. 12B 的 multimodal shape 只从 checkpoint 的 `config.json` 和
   `processor_config.json` 派生，YAML/runtime 不允许覆盖 soft-token 上限。
5. 先完成 BF16/HF/ONNX/HMONNX 正确性闭环，再开放量化。GPTQModel/AutoRound
   必须分别实测，不能由旧 Gemma4 结果推断；当前两条 W4G64 路径均已闭环。

## 2. 权威输入与优先级

发生冲突时使用以下优先级：

1. 12B checkpoint 自带 `config.json`、`processor_config.json`、tokenizer/chat
   template；
2. Transformers 5.13.0 `gemma4_unified` 实现；
3. 当前 vLLM `gemma4_unified.py` 推理路径；
4. 当前 xh2modelzoo Gemma4 Series contract-v1/v2；
5. 旧 Gemma4 Series 飞书文档只作为 tower-based 模型历史参考。

旧飞书文档中的以下合同不适用于 12B Unified：

- image `[1, 2520, 768]` + pooling matrix；
- video `[1, 630, 768]` + pooling matrix；
- audio `[1, 2999, 128]` + 5D audio attention mask；
- 旧的 256-token 双 prefill/full-mask 描述。

## 3. 结构对比

| 项 | 12B Unified | E2B/E4B | 31B | 26B-A4B |
|---|---:|---:|---:|---:|
| HF architecture | `Gemma4UnifiedForConditionalGeneration` | `Gemma4ForConditionalGeneration` | 同左 | 同左 |
| frontend | encoder-free | vision/audio tower | vision tower | vision tower |
| text topology | dense | dense + PLE/shared-KV | dense | MoE |
| hidden/layers | 3840 / 48 | 1536/35、2560/42 | 5376 / 60 | 2816 / 30 |
| sliding/full layers | 40 / 8 | 28/7、35/7 | 50 / 10 | 25 / 5 |
| sliding window | 1024 | 512 | 1024 | 1024 |
| sliding KV heads/dim | 8 / 256 | 1/256、2/256 | 16 / 256 | 8 / 256 |
| full KV heads/dim | 1 / 512 | 1/512、2/512 | 4 / 512 | 2 / 512 |
| `attention_k_eq_v` | true | false | true | true |
| PLE | 无 | 有 | 无 | 无 |
| shared KV | 无 | 有 | 无 | 无 |
| bidirectional | vision | 无 | vision | vision |
| image max soft tokens | 280 | 280 | 280 | 280 |
| video max/frame | 70 | 70 | 70 | 70 |
| audio | raw 640 samples/token | log-mel + tower | 无 | 无 |

### 3.1 Encoder-free vision

Unified vision 不存在 `vision_tower`。单个有效输入 token 已经是合并后的
`48 x 48 x 3 = 6912` RGB patch：

```text
raw merged patch [6912]
  -> LayerNorm
  -> Linear(6912, 3840)
  -> LayerNorm
  -> add factorized x/y position embedding
  -> LayerNorm
  -> RMSNorm(no scale)
  -> Linear(3840, 3840)
  -> LLM soft token
```

因此 12B 不应生成 pooling matrix、ViT attention mask、RoPE table，也不应先生成
2520 个 16x16 patch 再做 3x3 pooling。

### 3.2 Encoder-free audio

Unified audio 不存在 `audio_tower`。16 kHz waveform 每 640 个 sample 分成一个
token frame：

```text
num_audio_tokens = ceil(num_samples / 640)
raw frame [640] -> RMSNorm(no scale) -> Linear(640, 3840)
```

不存在 mel、两层 stride-2 conv，也不存在 5D `audio_attention_mask`。

### 3.3 Text

12B text 是普通 dense Gemma4：无 PLE、无 shared KV、无 MoE。可复用当前
Gemma4 Series dense lowering，但需要注册/泛化到
`Gemma4UnifiedText*`/`Gemma4UnifiedForCausalLM` 类，而不是强制构造旧
`Gemma4ForConditionalGeneration`。

## 4. 固定 multimodal 合同

### 4.1 配置派生规则

新增只读 `Gemma4SeriesModalityContract`，由 checkpoint 派生：

```text
image_soft_tokens = config.vision_config.num_soft_tokens                # 280
image_patch_dim   = config.vision_config.model_patch_size ** 2 * 3      # 6912
position_capacity = config.vision_config.mm_posemb_size                 # 1120
video_soft_tokens = processor_config.video_processor.max_soft_tokens    # 70
audio_soft_tokens = processor_config.audio_seq_length                   # 750
audio_feature_dim = config.audio_config.audio_samples_per_token         # 640
sampling_rate     = processor_config.feature_extractor.sampling_rate     # 16000
```

规则：

- Unified YAML 不提供 `image_seq_length`、`video image_seq_length`、
  `max_soft_tokens`、`audio_seq_length` 的有效 override；
- 若历史字段存在且与 checkpoint 不一致，preflight 直接报错，不静默改写；
- 导出 meta 记录最终派生值及 checkpoint config hash；
- 当前硬件合同只验收 280/70/750/6912/640。未来 checkpoint 值变化时先 fail-fast，
  不能自动沿用旧编译图。

### 4.2 为什么 `mm_posemb_size=1120` 不是 1120 个图像 token

`1120` 是 factorized position embedding 在每个 x/y 轴上的坐标表容量：

```text
pos_embedding.shape = [1120, 2, 3840]
```

它不是 token sequence length。图像 token 上限仍是
`vision_config.num_soft_tokens=280`。vLLM 允许调用方把 `max_soft_tokens` 改为
560/1120 是通用服务能力，不属于本项目的固定模型合同。

### 4.3 Image graph

```text
inputs:
  pixel_values       float/bfloat16 [1, 280, 6912]
  image_position_ids int32          [1, 280, 2]
                                      Host 将 padding [-1, -1] 归一为 [0, 0]
output:
  image_embeds       float/bfloat16 [1, 280, 3840]
```

图内所有算子均为逐 token 操作，不需要 mask。有效 token 数由 Host 在归一化前按
`~(image_position_ids == -1).all(-1)` 计算，图执行后只取对应有效 embedding；padding
位置的数值无需规定，也不得依赖其恰好为零。

### 4.4 Video graph

video 使用同一组 vision 权重，但单独导出单帧静态图：

```text
inputs:
  pixel_values       float/bfloat16 [1, 70, 6912]
  video_position_ids int32          [1, 70, 2]
                                      Host 将 padding [-1, -1] 归一为 [0, 0]
output:
  video_embeds       float/bfloat16 [1, 70, 3840]
```

多帧由 Host 按帧执行并只拼接有效 embedding。每帧是一个 vision atomic range；
frame timestamp、BOI、EOI 保持 hard token，不纳入 embedding range。

预采样帧必须携带 fps/timestamps/frame indices 元数据。缺失元数据时 XH 层
fail-fast，不采用 Transformers 的 `fps=24` fallback，避免 prompt 时间语义漂移。

### 4.5 Audio graph

```text
inputs:
  input_features      float/bfloat16 [1, 750, 640]
output:
  audio_embeds        float/bfloat16 [1, 750, 3840]
```

音频投影同样不跨 token，mask 不进入图。Host 保留 processor 的
`input_features_mask`，执行静态 750-token 图后按该 mask 过滤，scatter 数严格等于
`input_features_mask.sum()`。

30 秒上限为：

```text
750 * 640 = 480000 samples @ 16 kHz
```

超过 480000 samples 默认报错，不静默截断。原因是 Transformers 当前只有调用方
显式传 `max_length=750` 才会截断，而静默截断会让音频语义与用户输入不一致。

## 5. Processor 设计

### 5.1 类边界

保留一个 public loader：

```text
XHGemma4SeriesProcessor.from_pretrained(checkpoint)
```

内部按 architecture 返回两个实现之一：

- tower-based：现有 `Gemma4Processor` adapter；
- encoder-free：新增 `XHGemma4UnifiedProcessor`，基于正确的
  `Gemma4UnifiedProcessor`，不把 Unified 子 processor 重新塞回旧
  `Gemma4Processor`。

不要在当前 `XHGemma4SeriesProcessor.__call__` 中继续叠加 Unified if/else；它目前
固定调用 pooling/audio-conv 后处理，正是 12B image/audio 错误的根源。

### 5.2 输出与校验

Unified processor 必须保证：

- image：patch 和 position pad 到 280；placeholder 数等于有效 position 数；
- video：每帧 pad 到 70；每帧 placeholder 数等于有效 position 数；
- audio：pad 到 750；placeholder 数等于 mask sum；
- mixed：image/video/audio 的 token id、`mm_token_type_ids`、embedding range 顺序
  与 chat template 完全一致；
- audio 只接受 16 kHz 或由仓库已有、可验证的 resample 路径转换到 16 kHz；不新增
  隐式第三方依赖；
- 禁止生成 `pooling_matrix`、`visual_attention_mask`、
  `audio_attention_mask` 等 tower-only 字段。

## 6. LLM attention、chunk 与 cache

### 6.1 Attention 语义

12B config 为 `use_bidirectional_attention="vision"`：

```text
full layer    = causal
sliding layer = (causal OR same_vision_atomic_range) AND sliding_window(1024)
audio         = causal，绝不能加入 vision bidirectional range
```

当前 `data_preprocess.py` 使用 `mm_token_type_ids > 0` 判断 vision，会把 audio 也
包含进去。实现时必须统一显式类型：

```text
IMAGE = 1
VIDEO = 2
AUDIO = 3
is_vision = (type == IMAGE) OR (type == VIDEO)
```

legacy mask 与 contract-v2 compact `mm_prefix_ranges` 两条路径都要修复，并新增
audio-causal 回归测试。

### 6.2 Chunk

- 单一固定 prefill 图，默认 `prefill_chunk_length=320`；
- image 280 token 作为不可切 atomic range；
- video 以单帧有效 token range 为 atomic range；
- audio 是 causal token，可跨 chunk 切分；
- 不允许 image/video atomic range 跨 chunk；
- `context_max_length` 是静态导出容量，不等于模型的
  `max_position_embeddings=262144`。

### 6.3 KV cache

默认 prefill 320 时：

| layer | 数量 | K/V shape（每个 tensor） |
|---|---:|---|
| sliding | 40 | `[1, 8, 1344, 256]`，其中 1344 为对齐后的 1024+320 窗口 |
| full | 8 | `[1, 1, context_max_length, 512]` |

`attention_k_eq_v=true` 允许计算上复用 K，但第一阶段保持现有 K/V 双输入输出 ABI，
不在 12B 首次适配时同时做 cache ABI 优化。

full attention 继续使用当前标准 causal masked-softmax 路径；不恢复旧外部
`full_attention_mask` 输入。contract-v2 继续使用 compact visibility metadata。

## 7. 注册、配置与 meta

### 7.1 Variant resolver

判定优先级：

1. 顶层 `model_type == "gemma4_unified"` 或 architecture 为
   `Gemma4UnifiedForConditionalGeneration` -> `12b_unified`；
2. `enable_moe_block/num_experts` -> `26b_a4b`；
3. PLE/shared-KV/tower audio -> `e2b/e4b`；
4. 其余旧 Gemma4 dense -> `31b`。

`Gemma4SeriesVariantSpec` 增加：

```text
frontend_kind: "tower" | "encoder_free"
hf_architecture: str
```

不能再用 `has_audio` 单独推导 E 系列。

### 7.2 Registry/workflow

- builder 为 `Gemma4UnifiedForConditionalGeneration` 注册独立的
  `XHGemma4UnifiedModel`；它复用 Series 实现，但单独声明 Transformers 5.13.0；
- 原 `Gemma4ForConditionalGeneration` 继续注册 `XHGemma4SeriesModel`，
  单独声明 Transformers 5.5.0；
- workflow 的 allowed model types 接受两个 HF architecture，但内部仍只有一个
  `Gemma4SeriesWorkflow`；
- Transformers AutoConfig/AutoModel 使用上游 Unified 原生注册，不手工把
  `gemma4_unified` 注册到旧 Gemma4 class；
- 推荐配置新增
  `configs_merak/workflows/xh2a/llm_models/gemma4_series/12b_unified/`。

### 7.3 Runtime meta

新增字段：

```json
{
  "variant": "12b_unified",
  "frontend_kind": "encoder_free",
  "hf_architecture": "Gemma4UnifiedForConditionalGeneration",
  "modality_contract": {
    "image_soft_tokens": 280,
    "video_soft_tokens_per_frame": 70,
    "audio_soft_tokens": 750,
    "vision_patch_dim": 6912,
    "audio_feature_dim": 640
  }
}
```

runtime 按 `frontend_kind`/meta 选择子图输入，不能根据目录名或 `variant` 字符串猜
是否需要 pooling/audio attention mask。

## 8. HF model 与权重加载

### 8.1 依赖门禁

项目全局声明保持 `transformers>=4.57,<4.58` 不变。模型运行环境按模型类
选择版本：

1. 原 Gemma4 Series 的 `XHGemma4SeriesModel` 要求 Transformers 5.5.0；
2. Gemma4 12B 的 `XHGemma4UnifiedModel` 要求 Transformers 5.13.0；
3. Unified 专用模块全部延迟导入，5.5.0 环境导入、量化和导出原 Gemma4 时
   不会访问 `transformers.models.gemma4_unified`。

不推荐把 Transformers 的 Unified modeling/processing 文件复制进仓库：这会形成
第二份上游实现，难以维护且容易与 checkpoint 更新漂移。

### 8.2 Model loader

- 两个模型类均通过 `AutoModelForImageTextToText` 按 config 构造；
  `HF_MODEL_CLS` 分别固定为 legacy/Unified 上游 class，用于加载后类型校验；
- text adapter 从 `model.language_model` 获取 text model；
- Unified vision/audio frontend 按实际 HF state dict mapping 提取；
- text graph 构造前移除 frontend 时，仅删除 Unified 实际属性，不假设存在
  `vision_tower/audio_tower`；
- tied `lm_head` 按 `tie_word_embeddings=true` 保持现有处理；
- 完整权重下载后，以 `safe_open` + 实际 `from_pretrained` 结果冻结权重映射测试，
  不能只依赖 safetensors header 猜模块属性。

## 9. 量化策略

已完成顺序：

1. existing HF / weight-only load；
2. GPTQModel text-only W4G64（默认 `quant_nontext_module=false`），已在 GPTQModel
   registry/recipe 中显式注册 `gemma4_unified`；
3. AutoRound text-only W4G64（mode1、无 rotation），已注册 Unified multimodal block
   与 shared-cache keys；
4. HMONNX W8A8/W4A8；
vision/audio frontend 第一版保持 BF16 或沿用现有 subgraph quant scheme，但必须分别
做 golden；不得因为它们只有 Linear 就默认认为量化无损。

GPTQModel 使用独立 `Gemma4UnifiedForConditionalGenerationGPTQ` 定义，量化模块树为
`model.language_model.layers`，顶层 `lm_head` 与旧 Gemma4 mapping 不同，不能回退旧类。

### 9.1 W4G64 实跑与产物

| 方法 | 校准 | 量化耗时 | quant HF | 完整重导出 |
|---|---|---:|---|---|
| GPTQModel | IVSG 102 x 512，batch=1 | 约 23m48s | `work_dirs/gemma4_12b_unified_w4g64_gptq_quant_20260723` | `work_dirs/gemma4_12b_unified_w4g64_gptq_export_20260723` |
| AutoRound | Pile 128 x 2048，batch=8，iters=200 | 2360.5s | `work_dirs/gemma4_12b_unified_w4g64_autoround_quant_20260723` | `work_dirs/gemma4_12b_unified_w4g64_autoround_export_20260723` |

两份量化目录均约 7.5 GiB，`qweight` 为 int32 packed W4、scale 为 FP16、
`bits=4/group_size=64/sym=true`；vision/audio 权重保持 BF16。两份 export 均包含
W4A8 命名的 visual、video_visual、audio、prefill、decode 图；image/video 图只有
`pixel_values + position_ids`，audio 图只有 `input_features`。

实跑不能复用旧 Gemma4 逻辑的原因：

- Unified sliding/full attention 的 RoPE 宽度分别为 256/512，GPTQ block replay 必须按
  layer type 重算 position embeddings；
- `num_kv_shared_layers=0` 时，最后一个 sliding/full 非共享层仍被 HF 标记为 KV
  producer，AutoRound 直接调用 decoder block 时也必须提供空的
  `shared_kv_states` 字典；
- Transformers/AutoRound `save_pretrained` 会省略部分 checkpoint-only multimodal
  字段，保存后必须以量化 config 为上层、原 checkpoint config 为下层做递归合并。

MTP assistant 已完成下载和实现。12B assistant 与 26/31B 的概念流程一致（拼接 target
embedding/backbone hidden、4 层 draft、post projection 回传），但必须使用 Unified text
class；其 3 层 sliding + 1 层 full 的 KV 几何分别为 8x256 和 1x512。导出前校验 target
hidden=3840、layer type mapping、共享 KV head/dim 及 YAML 字段。RoPE 直接使用 HF
`inv_freq`，不得再次套 `partial_rotary_factor`。

真实 assistant config 对比如下，不能沿用 26B/31B 的 target/KV 常量：

| assistant | HF model type | backbone/draft hidden | attention heads | sliding KV | full KV | ordered embedding |
|---|---|---:|---:|---:|---:|---|
| 12B | `gemma4_unified_assistant` | 3840/1024 | 16 | 8x256 | 1x512 | false |
| 26B-A4B | `gemma4_assistant` | 2816/1024 | 16 | 8x256 | 2x512 | false |
| 31B | `gemma4_assistant` | 5376/1024 | 32 | 16x256 | 4x512 | false |

三者的 draft 控制流一致，差异集中在 HF text class、backbone width 和 shared-KV
几何；12B 不需要 E2B/E4B 使用的 ordered/centroid embedding。最新 vLLM 也采用同一
`Gemma4MTP`/`Gemma4Proposer` 控制流，并按 assistant layer type 映射 target 的最后一个
同类非共享 KV layer，因此本实现保留两组显式 shared-cache 输入并在导出前做几何校验。

## 10. 代码落点

| 文件/目录 | 设计改动 |
|---|---|
| `gemma4_series/variants.py` | `12b_unified`、`frontend_kind`、resolver 优先级 |
| `gemma4_series/__init__.py` | 5.13-only Unified symbol 延迟导入与模型注册 |
| `gemma4_series/gemma4_unified_llm_model.py` | 独立 12B 模型类与 Transformers 5.13.0 门禁 |
| `gemma4_series/gemma4_series_llm_model.py` | 原 Gemma4 Transformers 5.5.0 门禁、Auto model load、frontend dispatch/removal |
| `gemma4_series/workflow.py` | architecture alias、推荐配置、preflight |
| `gemma4_series/export_plan.py` | checkpoint-derived modality contract；Unified 不用 2520/630 old patch 数 |
| `gemma4_series/xh_gemma4_series_config.py` | frontend-specific visual/audio config/meta |
| `gemma4_series/gemma4_series_processor.py` | 只保留 factory/legacy adapter；避免 Unified 走 old postprocess |
| `gemma4_series/gemma4_unified_processor.py` | 新增 encoder-free processor adapter |
| `gemma4_series/gemma4_unified_vision_model.py` | 新增 280/70 两输入、Host 过滤的双静态图共享实现 |
| `gemma4_series/gemma4_unified_audio_model.py` | 新增 750x640 单输入图，Host 按 mask 过滤 |
| `gemma4_series/_llm_model_impl.py`, `llm_text.py` | Unified text class lowering/bridge |
| `gemma4_series/data_preprocess.py` | 仅 image/video 构造 bidi range；audio causal |
| `gemma4_series/gemma4_series_hmonnx_inference.py` | 按 frontend meta 分派 encoder-free 子图 |
| `gemma4_series/gemma4_series_mtp_model.py`, `mtp_workflow.py` | Unified assistant、RoPE、共享 KV 合同校验与 MTP draft 导出 |
| `configs_merak/.../gemma4_series/12b_unified/` | base 与 MTP 配置 |
| `/data01/home/yujy/work/gptqmodel/gptqmodel/.../gemma4.py` | Unified GPTQ definition 与 recipe variant |
| `/data01/home/yujy/work/vllm-merak/vllm_merak/.../gemma4.py` | Unified vLLM processor/loader、三模态 side graph 路由 |
| `examples_merak/llm/gemma4_series/` | 12B text/image/video/audio/mixed demo |
| `tests/gemma4/` 与 public workflow tests | 下述回归矩阵 |

## 11. 测试规格

### 11.1 不依赖完整权重的测试

1. **Resolver**
   - Unified config 必须解析为 `12b_unified`，不是 `e4b`；
   - 老 E2B/E4B/31B/26B-A4B 结果不变。
2. **Contract derivation**
   - 从真实 config 得到 280/70/750/6912/640；
   - YAML 传 560/1120 或其他不一致值必须失败；
   - `mm_posemb_size=1120` 不影响 image sequence length。
3. **Processor**
   - image 多种纵横比：placeholder = valid position，shape 固定 280；
   - video 1/4/32 帧：每帧 shape 固定 70，placeholder 逐帧匹配；
   - audio sample 边界：1、640、641、1280、1281、480000；token 数分别为
     1、1、2、2、3、750；
   - 480001 samples 必须失败；
   - image+video+audio mixed 的 placeholder/range 顺序一致；
   - video 缺 metadata 必须失败。
4. **Attention/chunk**
   - image/video range 在 sliding layer 双向且受 1024 window 限制；
   - full layer causal；
   - audio range causal；
   - `[image 280][text 40]` 单 chunk；下一 token 进入第二 chunk；
   - audio 750 可以按 320/320/110 分 chunk。
5. **Registry/config**
   - 两个 HF architecture 都进入同一个 workflow/model class；
   - Unified meta 不出现 pooling/audio attention 字段；
   - 旧四模型 public tests 全部通过。

### 11.2 完整权重到位后的测试

1. `git lfs fsck`，文件大小/sha256 与 pointer 一致；
2. `safe_open` 校验 tensor key、shape、dtype；
3. `AutoModelForImageTextToText.from_pretrained(..., device_map="cpu")` load
   smoke，确认 missing/unexpected keys 为 0 或有白名单；
4. Transformers BF16 text/image/video/audio/mixed forward；
5. frontend PyTorch adapter 对比 HF：只比较有效 token，不约束 padding 输出；
6. ONNX 对比 PyTorch；HMONNX 对比 ONNX；
7. LLM prefill 第一个 chunk logits、decode logits、KV cache shape/value 对比；
8. 至少覆盖：纯文本、单图、极端纵横比图、多帧视频、1-token 音频、30 秒音频、
   三模态 mixed；
9. 与当前 vLLM 做 prompt token/range/generated token 交叉验证；vllm-merak 的
   Unified processing info 必须覆盖旧 tower estimator，使用 `ceil(samples/640)`；
10. NPU 端逐子图 HM-style golden：`visual`、`video_visual`、`audio`、
    `prefill`、`decode`。

### 11.3 验收门槛

- processor 的每个 modality：placeholder 数 == valid frontend embedding 数；
- 0 missing/unexpected critical weights；
- BF16 PyTorch/HF 有效 frontend token 在约定 tolerance 内一致；
- ONNX/HMONNX golden 通过；
- 旧 Gemma4 Series 单测、lint、typecheck、static analysis 无新增失败；
- 所有模态真实 generate 可运行，且 audio 不获得 bidirectional attention；
- 完成上述 BF16 闭环前不开始量化精度归因。

## 12. 实施顺序与停止条件

### Phase 0：下载与环境门禁

- 完成 LFS；验证 config/processor/weights；
- 在 12B 专用环境验证 Transformers 5.13.x，同时在原 Gemma4 5.5.0 环境跑
  import/export 回归；
- 若真实 weight mapping 与设计假设不一致，先更新本设计再编码。

### Phase 1：无权重结构适配

- resolver、registry、contract、processor；
- synthetic frontend 与 attention/chunk 测试；
- 旧模型回归。

### Phase 2：BF16 子图

- Unified vision/audio adapter；
- Unified text lowering；
- HF/ONNX parity。

### Phase 3：HMONNX/runtime

- export/meta/runtime dispatch；
- 五类子图 golden；
- text/image/video/audio/mixed generate。

当前完成度：五类图 export 与 text/image/video/audio aligned CUDA runtime Golden
已通过；物理 NPU、三模态同请求 mixed generate 以及与 vLLM generated-token 的逐 token
交叉验证仍作为硬件验收项保留。

### Phase 4：GPTQModel、MTP 与服务

- GPTQModel 与 AutoRound 的 `gemma4_unified` W4G64 全量量化、HF reload 和五图重导出
  已完成；精度数据集评测仍需独立任务执行；
- MTP 正式 assistant 已通过结构、权重加载、RoPE、target shared-KV 合同、eager
  forward 和 HMONNX smoke export 验证；
- vllm-merak 已支持 Unified architecture dispatch、config-owned processor、无 mask
  image/video/audio side graph、三类 embedding 分 chunk 路由及 audio causal range；
- 当前无-mask代码已完成三个 side HMONNX 的真实导出；在线 OpenAI 服务仍需要把它们
  与 text prefill/decode 重新组装为完整 artifact，并在物理 XH2a 设备做最终启动/请求压测。

停止条件：任一阶段发现 config-derived shape、placeholder 数、HF logits 或 cache
语义不一致，立即停在当前阶段；不得靠 hardcode token 数、吞掉 weight key、放宽
tolerance 或回退旧 Gemma4 class 继续推进。

## 13. 当前已知阻塞与风险

| 风险 | 处理 |
|---|---|
| 12B target/assistant 大权重下载完整性 | 已核验实际大小；assistant 与镜像 SHA256 一致 |
| Unified 5.13-only import 破坏原 Gemma4 5.5 环境 | 已拆分模型类并延迟导入 Unified 模块；双环境回归 |
| 当前 resolver 将 12B 误判成 E4B | 已修复：architecture/model_type 优先 |
| 旧 XH processor 对 Unified image 执行旧 pooling | 已使用独立 Unified processor adapter |
| 旧 XH audio 使用旧 conv token count | 已直接使用 HF `input_features_mask.sum()` |
| 旧 bidi 判断 `mm > 0` 会包含 audio | 已改为显式 image/video token type |
| 上游 vLLM audio estimator 对短边界有偏差 | vllm-merak 覆盖为 `ceil(samples/640)` |
| 上游 vLLM 支持 560/1120 override | XH 与 vllm-merak 固定合同均禁止 override |
| video metadata 缺失时 HF 默认 24 fps | XH fail-fast |
| GPTQModel/AutoRound 新 architecture | 两条 W4G64 路径均已实跑；RoPE/shared-KV/config-save 差异已显式修复 |

## 14. 复核证据

### 14.1 Checkpoint

- `/data01/datasets/gemma-4-12B-it/config.json`
- `/data01/datasets/gemma-4-12B-it/processor_config.json`
- target `model.safetensors`：`23919549408` bytes；
- assistant `weights/gemma-4-12B-it-assistant/model.safetensors`：`845719296`
  bytes，SHA256 `3279c173daddd7186e79d652ad94022415736d3a1370625696c898429b06d6df`。

### 14.2 Transformers 5.13.0

- `modeling_gemma4_unified.py:783-876`：encoder-free vision 与 multimodal
  projection；
- `modeling_gemma4_unified.py:891-953`：无 tower 的主模型与 padding strip；
- `modeling_gemma4_unified.py:1220-1248`：full causal、sliding
  `(causal OR blockwise) AND window`；
- `processing_gemma4_unified.py:195-227`：image/video/audio placeholder；
- `feature_extraction_gemma4_unified.py:67-148`：raw waveform 640-sample framing，且只有
  显式 `max_length` 才截断。

### 14.3 当前 vLLM

- `/data01/home/yujy/work/vllm/vllm/model_executor/models/gemma4_unified.py:73-132`：
  encoder-free vision；
- 同文件 `:220-350`：Unified model 重建、无 tower、复用 language model；
- 同文件 `:356-434`：image/video/audio embedding 流程；
- `vllm/model_executor/models/gemma4.py:498-514`：sliding layer 开启
  `mm_prefix_clamp_sliding_window`；
- `vllm/v1/worker/gpu_model_runner.py:2358-2391`：保留过长 vision range，audio
  排除在 bidi range 外。

### 14.4 已修复的 xh2modelzoo 问题点

- `gemma4_series/variants.py`：12B 不再因有 audio 被误判 E4B；
- `gemma4_series/gemma4_unified_processor.py`：Unified 已与旧 pooling/audio-conv
  processor 隔离；
- `gemma4_series/gemma4_unified_vision_model.py`：不依赖 `vision_tower`、
  pooling matrix 或图内 mask；
- `gemma4_series/gemma4_unified_audio_model.py`：不依赖 `audio_tower` 或 5D
  attention mask；
- `gemma4_series/data_preprocess.py`：bidi range 显式只接受 image/video，audio 保持 causal；
- `gemma4_series/export_plan.py`：Unified 合同由 checkpoint config 派生，不再使用
  2520/630 old tower patch 合同；
- `gemma4_series/gemma4_unified_llm_model.py`：仅 12B 模型类声明
  Transformers 5.13.0，项目全局依赖与原 Gemma4 5.5.0 门禁保持不变。
