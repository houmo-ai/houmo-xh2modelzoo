# Qwen3.5-MoE XH2a 导出与推理

先在仓库根目录准备环境：

```bash
source env.sh
export PYTHONPATH=./
```

下面默认模型：

- 主模型：`weights/qwen36moe-no-rotate-attn8-shared8-expert45`
- DFlash：`weights/Qwen3.6-35B-A3B-DFlash`

> 对 35B-A3B 这类大 MoE，示例脚本默认关闭 `auto_offload`，优先让 prefill / decode 直接常驻 80G GPU；只有显存确实不够时，再显式加 `--enable-auto-offload` 或 `--resource-tight-mode`。

## Split Conv Cache 导出

`--split-conv-cache` 将线性注意力的 conv_cache 从单个合并 tensor 拆分为 3 个独立 tensor（q, k, v）。默认不开启，保持向后兼容。

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model weights/Qwen3.5-35B-A3B \
  --quant-type w8a8h0_sefp \
  --context-length 2048 \
  --input-sequence-length 256 \
  --split-conv-cache
```

开启后 meta.json 中 `linear_cache.layers` 的每层会输出 `conv_shapes`（3 个 shape 的列表）而非 `conv_shape`（单个 shape）。ONNX 模型的输入/输出命名变化：

| 模式 | 输入名 | 输出名 |
|------|--------|--------|
| 默认 | `past_conv_cache_0` | `conv_cache_out_0` |
| split | `past_conv_cache_q_0` / `past_conv_cache_k_0` / `past_conv_cache_v_0` | `conv_cache_out_q_0` / `conv_cache_out_k_0` / `conv_cache_out_v_0` |

Spec decode 模式（MTP / DFlash）同样支持 `--split-conv-cache`。

## 标准导出

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model weights/qwen36moe-no-rotate-attn8-shared8-expert45 \
  --quant-type w4a8h0_ssfp \
  --context-length 8192 \
  --input-sequence-length 256
```

标准导出产物默认在：

```bash
work_dirs/qwen36moe-no-rotate-attn8-shared8-expert45-XH2a-8k-w4a8h0_ssfp/meta.json
```

## MTP 导出

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model weights/qwen36moe-no-rotate-attn8-shared8-expert45 \
  --quant-type w4a8h0_ssfp \
  --context-length 8192 \
  --input-sequence-length 256 \
  --spec-decode-mode mtp \
  --num-draft-tokens 4
```

默认产物：

```bash
work_dirs/qwen36moe-no-rotate-attn8-shared8-expert45-XH2a-8k-w4a8h0_ssfp-spec_mtp/meta.json
```

## DFlash 导出

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model weights/qwen36moe-no-rotate-attn8-shared8-expert45 \
  --quant-type w4a8h0_ssfp \
  --context-length 8192 \
  --input-sequence-length 256 \
  --spec-decode-mode dflash \
  --num-draft-tokens 16 \
  --dflash-model-dir weights/Qwen3.6-35B-A3B-DFlash
```

默认产物：

```bash
work_dirs/qwen36moe-no-rotate-attn8-shared8-expert45-XH2a-8k-w4a8h0_ssfp-spec_dflash/meta.json
```

## 标准 Demo

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_demo.py \
  --config work_dirs/qwen36moe-no-rotate-attn8-shared8-expert45-XH2a-8k-w4a8h0_ssfp/meta.json \
  --prompt "你好，请用中文介绍一下你自己。" \
  --max-new-tokens 256 \
  --enable_cuda_graph \
  --cuda_graph_modules prefill,decode,draft_prefill,draft_decode
```

交互式：

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_demo.py \
  --config work_dirs/qwen36moe-no-rotate-attn8-shared8-expert45-XH2a-8k-w4a8h0_ssfp/meta.json \
  --interactive
```

## 标准 Benchmark / Smoke Test

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_hmonnx_test.py \
  --config work_dirs/qwen36moe-no-rotate-attn8-shared8-expert45-XH2a-8k-w4a8h0_ssfp/meta.json \
  --prompt "请用中文简要介绍一下混合线性注意力模型。" \
  --max-new-tokens 128 \
  --warmup-runs 1 \
  --benchmark-runs 1
```

如果要打开 CUDA Graph：

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_hmonnx_test.py \
  --config work_dirs/qwen36moe-no-rotate-attn8-shared8-expert45-XH2a-8k-w4a8h0_ssfp/meta.json \
  --enable-cuda-graph \
  --cuda-graph-modules prefill,decode
```

## MTP Demo

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_test.py \
  --config work_dirs/qwen36moe-no-rotate-attn8-shared8-expert45-XH2a-8k-w4a8h0_ssfp-spec_mtp/meta.json \
  --prompt "写一首关于 AI 的诗" \
  --max_new_tokens 128 \
  --warmup_runs 0 \
  --benchmark_runs 1
```

开启 CUDA Graph：

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_test.py \
  --config work_dirs/qwen36moe-no-rotate-attn8-shared8-expert45-XH2a-8k-w4a8h0_ssfp-spec_mtp/meta.json \
  --prompt "写一首关于 AI 的诗" \
  --max_new_tokens 128 \
  --warmup_runs 0 \
  --benchmark_runs 1 \
  --enable_cuda_graph \
  --cuda_graph_modules prefill,decode,draft_prefill,draft_decode
```

## DFlash Demo

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_test.py \
  --config work_dirs/qwen36moe-no-rotate-attn8-shared8-expert45-XH2a-8k-w4a8h0_ssfp-spec_dflash/meta.json \
  --prompt "写一首关于 AI 的诗" \
  --max_new_tokens 128 \
  --warmup_runs 0 \
  --benchmark_runs 1
```

开启 CUDA Graph：

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_test.py \
  --config work_dirs/qwen36moe-no-rotate-attn8-shared8-expert45-XH2a-8k-w4a8h0_ssfp-spec_dflash/meta.json \
  --prompt "写一首关于 AI 的诗" \
  --max_new_tokens 128 \
  --warmup_runs 0 \
  --benchmark_runs 1 \
  --enable_cuda_graph \
  --cuda_graph_modules prefill,decode,draft_context,draft_decode
```

## Spec Decode Bench

基准数据默认直接复用 dense 的评测集：

```bash
examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl
```

### MTP Bench

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_bench.py \
  --meta work_dirs/qwen36moe-no-rotate-attn8-shared8-expert45-XH2a-8k-w4a8h0_ssfp-spec_mtp/meta.json \
  --dataset examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl \
  --output-md work_dirs/qwen36moe_mtp_bench.md \
  --output-json work_dirs/qwen36moe_mtp_bench.json \
  --think-mode both \
  --max-new-tokens 1024 \
  --limit 0
```

### DFlash Bench

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_bench.py \
  --meta work_dirs/qwen36moe-no-rotate-attn8-shared8-expert45-XH2a-8k-w4a8h0_ssfp-spec_dflash/meta.json \
  --dataset examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl \
  --output-md work_dirs/qwen36moe_dflash_bench.md \
  --output-json work_dirs/qwen36moe_dflash_bench.json \
  --think-mode both \
  --max-new-tokens 1024 \
  --limit 0
```

Bench 开启 CUDA Graph：

```bash
export CUDA_VISIBLE_DEVICES=0
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_bench.py \
  --meta work_dirs/qwen36moe-no-rotate-attn8-shared8-expert45-XH2a-8k-w4a8h0_ssfp-spec_dflash/meta.json \
  --dataset examples/llm/qwen3_5/spec_decode_eval_prompts.jsonl \
  --output-md work_dirs/qwen36moe_dflash_bench_cuda_graph.md \
  --output-json work_dirs/qwen36moe_dflash_bench_cuda_graph.json \
  --think-mode both \
  --max-new-tokens 1024 \
  --enable-cuda-graph \
  --cuda-graph-modules prefill,decode,draft_context,draft_decode
```
