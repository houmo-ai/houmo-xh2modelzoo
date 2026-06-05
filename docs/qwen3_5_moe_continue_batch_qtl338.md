# QTL-338 Qwen3.5/3.6 MoE Continue Batch 导出、Demo 与兼容性说明

## 当前结论

QTL-338 的 continue-batch 修改已经覆盖：

- batch=2 / batch=4 HMONNX 导出；
- full attention 的 per-batch KV cache continue 计算；
- linear attention 的 per-batch conv cache / recurrent state / mask 输入输出；
- `split_conv_cache=True` 与 `split_conv_cache=False` 两种路径；
- batch demo 按每个 batch 独立 prompt 做 greedy decode；
- golden-only 与 release package 流程支持 batch>1；
- batch=1 仍保持原始 IO 命名，预期兼容旧导出和旧 demo。

本轮重新导出使用的是正确的量化导出方式：

```bash
--model weights/Qwen3.6-35B-A3B \
--quant-weight weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400 \
--quant-type w4a8h0_ssfp
```

也就是说，`--model` 指向浮点模型目录，`--quant-weight` 指向 GPTQ/量化权重目录；不是把量化权重目录直接当 `--model`。

飞书同步文档：

```text
https://www.feishu.cn/docx/TuM6dvbcmolYngxJtb7cSeYMnhd
```

---

## 代码修改范围

| 文件 | 作用 |
| --- | --- |
| `examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py` | 导出、golden-only、package；补齐 batch>1 golden 输入展开，并按 work_dir/meta.json 的真实 batch size 生成 golden |
| `examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_batch_demo.py` | batch demo；每个 batch 一个 prompt，按 batch 拼回 logits 并更新 cache |
| `xh_model_zoo/xh_llm/models/qwen3_5_moe/_moe_model.py` | continue-batch wrapper；graph 边界单 batch IO，模型内部 concat/split |
| `xh_model_zoo/xh_llm/models/qwen3_5_moe/inference.py` | runtime feed/update；识别 `_batch_N` 输入输出并回写对应 cache |
| `xh_model_zoo/xh_llm/models/qwen3_5_moe/qwen3_5_moe_converter.py` | converter/export config；batch IO、Add domain patch、split conv cache 相关导出逻辑 |
| `tests/qwen3_5_moe/test_moe_resource_tight_and_split_conv.py` | 回归测试；覆盖 batch input slicing、split/non-split conv cache 和 recurrent state 更新 |

本次追加修复的重点是 golden-only：之前 batch=2 单独跑 golden/package 会因为 `_generate_golden()` 没初始化 `input_ids_batch/export_batch` 失败；现在从 `meta.json.wrap_cfg.batch_size` 读取真实 batch，并把单条 prompt expand 到导出 batch。

---

## Continue Batch 的 IO Contract

HMONNX 边界要求单 batch IO，所以 batch>1 时不能直接导出 `[B, ...]` 作为 graph input/output。

实际规则是：

```text
外部 graph input:   多个 [1, ...]，用 _batch_0/_batch_1/... 区分
wrapper 内部:       concat 成 [B, ...]
模型内部:           正常按 batch 计算，必要处 per-batch cache update
外部 graph output:  split 回多个 [1, ...] 输出
```

batch=2 示例：

```text
inputs_embeds_batch_0: [1, ...]
inputs_embeds_batch_1: [1, ...]
logits_batch_0:        [1, ...]
logits_batch_1:        [1, ...]
```

batch=4 示例：

```text
inputs_embeds_batch_0 ... inputs_embeds_batch_3
logits_batch_0        ... logits_batch_3
```

---

## Full Attention 是怎么做的

full attention 的 continue batch 语义是：每个 batch 的 K/V cache 独立更新，每个 batch 的 attention 独立计算，然后再聚合。

更具体地说：

```text
batched hidden_states
  -> q_proj / k_proj / v_proj 先对完整 batch tensor 做 linear
  -> q/k norm、reshape、RoPE
  -> 按 batch slice
  -> 每个 batch 单独更新自己的 K/V cache
  -> 非 BFP flash attention 路径：每个 batch 单独 qk matmul + softmax + pv matmul
  -> torch.cat 聚合回 [B, ...]
  -> gate + o_proj
```

所以之前你问的两点，结论是：

1. **不是一进 full-attention block 就拆 batch。** 先在完整 batch tensor 上做 q/k/v linear、norm、RoPE。
2. **cache 和 attention matmul 阶段是 per-batch 独立的。** 每个 batch 单独做 K/V cache update、QK matmul、softmax、PV matmul，最后再 `cat` 聚合。

如果启用 BFP flash attention，K/V cache 仍按 batch 独立更新；attention kernel 本身走 `bfp_attn`，不再是 Python 显式 per-batch matmul 循环。

---

## Linear Attention 是怎么做的

linear attention 不使用 full attention 的 K/V cache，而是涉及：

- `linear_attn_mask`
- conv cache
- recurrent state
- `split_conv_cache=True/False`

### `split_conv_cache=True`

conv cache 按 q/k/v 三路拆开。batch=2 输入/输出形如：

```text
past_conv_cache_q_0_batch_0
past_conv_cache_q_0_batch_1
past_conv_cache_k_0_batch_0
past_conv_cache_k_0_batch_1
past_conv_cache_v_0_batch_0
past_conv_cache_v_0_batch_1

conv_cache_out_q_0_batch_0
conv_cache_out_q_0_batch_1
conv_cache_out_k_0_batch_0
conv_cache_out_k_0_batch_1
conv_cache_out_v_0_batch_0
conv_cache_out_v_0_batch_1
```

batch=4 同理扩展到 `_batch_0` ~ `_batch_3`。

### `split_conv_cache=False`

conv cache 是 merged 形式。输入/输出形如：

```text
past_conv_cache_0_batch_0
past_conv_cache_0_batch_1
conv_cache_out_0_batch_0
conv_cache_out_0_batch_1
```

这条路径已经有回归测试覆盖，重点验证：

- merged conv cache 能按 batch slice feed；
- recurrent state 能按 batch slice feed；
- per-batch output 能回写到对应 cache；
- 所有喂给 fake HMONNX 的输入 leading dim 都是 1。

---

## 正确导出命令

### batch=2

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 TQDM_DISABLE=1 \
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model weights/Qwen3.6-35B-A3B \
  --quant-weight weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400 \
  --quant-type w4a8h0_ssfp \
  --batch-size 2 \
  --context-length 2048 \
  --input-sequence-length 256 \
  --work-dir work_dirs/qwen36moe-contbatch-w4ssfp-b2
```

### batch=4

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 TQDM_DISABLE=1 \
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model weights/Qwen3.6-35B-A3B \
  --quant-weight weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400 \
  --quant-type w4a8h0_ssfp \
  --batch-size 4 \
  --context-length 2048 \
  --input-sequence-length 256 \
  --work-dir work_dirs/qwen36moe-contbatch-w4ssfp-b4
```

### 导出结果审计

| batch | prefill | decode | MoEBlock |
| --- | --- | --- | --- |
| 2 | inputs=294, outputs=242, bad leading batch dims=0, standard Add=0 | inputs=294, outputs=242, bad leading batch dims=0, standard Add=0 | `mode=ssfp`, `hmfp_weight_man_bit=4` |
| 4 | inputs=588, outputs=484, bad leading batch dims=0, standard Add=0 | inputs=588, outputs=484, bad leading batch dims=0, standard Add=0 | `mode=ssfp`, `hmfp_weight_man_bit=4` |

这说明当前 ONNX 不是旧的 `w8a8h0_sefp`，而是走了 `--quant-weight` 量化权重分支，MoEBlock 为 w4/ssfp。

---

## Golden-only 与 Package 命令

### batch=2

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model weights/Qwen3.6-35B-A3B \
  --batch-size 2 \
  --context-length 2048 \
  --input-sequence-length 256 \
  --quant-type w4a8h0_ssfp \
  --quant-weight weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400 \
  --work-dir work_dirs/qwen36moe-contbatch-w4ssfp-b2 \
  --golden-only \
  --release_xh_version xh2 \
  --release_modelscope_name qwen3_6_35b_a3b \
  --release_wmix_amix wmix_amix \
  --release_date 20260605 \
  --package_release
```

已生成：

```text
work_dirs/qwen36moe-contbatch-w4ssfp-b2/hmquant_xh2_qwen3_6_35b_a3b_wmix_amix_256_2k_20260605/
work_dirs/qwen36moe-contbatch-w4ssfp-b2/hmquant_xh2_qwen3_6_35b_a3b_wmix_amix_256_2k_20260605.zip
```

验证点：

- `Golden export batch: 2`
- prefill next token ids: `[271, 271]`
- `golden_meta_info.json` 中 `wrap_cfg.batch_size=2`
- prefill/decode `step_0` 都有 `inputs_embeds_batch_0/1` 与 `logits_batch_0/1`
- release zip 校验通过，大小约 `84.39 GiB`，zip entries `23731`

### batch=4

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 \
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model weights/Qwen3.6-35B-A3B \
  --batch-size 4 \
  --context-length 2048 \
  --input-sequence-length 256 \
  --quant-type w4a8h0_ssfp \
  --quant-weight weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400 \
  --work-dir work_dirs/qwen36moe-contbatch-w4ssfp-b4 \
  --golden-only \
  --release_xh_version xh2 \
  --release_modelscope_name qwen3_6_35b_a3b \
  --release_wmix_amix wmix_amix \
  --release_date 20260605 \
  --package_release
```

已生成：

```text
work_dirs/qwen36moe-contbatch-w4ssfp-b4/hmquant_xh2_qwen3_6_35b_a3b_wmix_amix_256_2k_20260605/
work_dirs/qwen36moe-contbatch-w4ssfp-b4/hmquant_xh2_qwen3_6_35b_a3b_wmix_amix_256_2k_20260605.zip
```

验证点：

- `Golden export batch: 4`
- prefill next token ids: `[271, 271, 271, 271]`
- prefill/decode `step_0` 有 `_batch_0` ~ `_batch_3` 输入输出 golden 文件
- release zip 校验通过，大小约 `102.15 GiB`，zip entries `26283`

---

## Demo 命令

Demo 脚本：

```text
examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_batch_demo.py
```

它要求 prompt 数量严格等于导出 batch size。

### batch=2 demo

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_batch_demo.py \
  --config work_dirs/qwen36moe-contbatch-w4ssfp-b2/meta.json \
  --max-new-tokens 512 \
  --prompt '请用一句话介绍混合专家模型。' \
  --prompt '请用一句话解释线性注意力。'
```

### batch=4 demo

```bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 \
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_batch_demo.py \
  --config work_dirs/qwen36moe-contbatch-w4ssfp-b4/meta.json \
  --max-new-tokens 512 \
  --prompt '请用一句话介绍混合专家模型。' \
  --prompt '请用一句话解释线性注意力。' \
  --prompt '请用一句话说明KV cache的作用。' \
  --prompt '请用一句话介绍ONNX导出。'
```

Demo 内部流程：

1. 按 prompt 列表构造 `[B, seq]` input ids；
2. prefill 阶段 padding 到导出 `input_sequence_length=256`；
3. runtime 根据 HMONNX 输入名自动把 `[B, ...]` slice 成 `_batch_N` 单 batch feed；
4. runtime 把 `logits_batch_N` 拼回 `[B, ...]`；
5. decode 阶段逐 token 更新每个 batch 对应 cache；
6. 输出时逐 batch decode 文本。

本轮 demo 实测使用 `--max-new-tokens 512`：

- batch=2：`/tmp/qwen36moe_w4ssfp_b2_demo512_gpu4.log`，2 个 batch 都生成了独立回答；
- batch=4：`/tmp/qwen36moe_w4ssfp_b4_demo512_gpu4.log`，4 个 batch 都生成了独立回答。

备注：实测时 GPU0 被其他任务占用，因此实际用 `CUDA_VISIBLE_DEVICES=4` 跑通；命令中的 GPU 编号可按机器空闲卡替换，不影响模型/graph 逻辑。

---

## batch=1 是否兼容

结论：**兼容。**

原因：batch=1 不走 `_batch_N` 命名，不做额外拆分。

### batch=1 导出 IO 仍是原始名字

batch=1 时仍使用：

```text
inputs_embeds
past_seq_length
current_input_length
linear_attn_mask
past_key_cache_0
past_value_cache_0
past_conv_cache_0
past_recurrent_state_0
logits
```

不会变成：

```text
inputs_embeds_batch_0
logits_batch_0
```

### runtime 同时支持两套命名

runtime 现在支持：

```text
原始 batch=1 名字：past_conv_cache_0
continue-batch 名字：past_conv_cache_0_batch_1
```

所以 batch=1 旧图仍能按原路径 feed/update；batch>1 才按 `_batch_N` 解析。

### golden-only 也兼容 batch=1

`_generate_golden()` 当前从 `meta.json.wrap_cfg.batch_size` 读 batch：

- 如果是 1：`input_ids_batch` 就是原 prompt batch，不 expand；
- 如果是 2/4：单条 prompt expand 到导出 batch；
- 如果 prompt batch 和导出 batch 不匹配且不是 1，会报错，避免静默生成错误 golden。

### 注意

本轮完整大模型重新导出和 golden/package 实测覆盖的是 batch=2、batch=4；batch=1 兼容性来自代码分支、命名策略和 runtime 双命名支持。由于 batch=1 不进入 `_batch_N` 分支，本次 continue-batch 修改不会改变 batch=1 graph IO contract。

---

## 验证记录

已完成：

```bash
python -m py_compile examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py
```

已完成 batch=2：

- w4/ssfp export；
- ONNX IO audit；
- golden-only；
- package；
- release zip 校验；
- demo `--max-new-tokens 512`。

已完成 batch=4：

- w4/ssfp export；
- ONNX IO audit；
- golden-only；
- package；
- release zip 校验；
- demo `--max-new-tokens 512`。
