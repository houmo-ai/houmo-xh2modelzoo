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


## 推荐完整流程（MoE：导出、Golden、Demo、MTP、DFlash）

若已有导出产物，可以跳过导出步骤，直接将下面 demo 命令的 `--config` 指向已有 `meta.json`。

### 1. 环境与默认兼容参数

```bash
conda activate xhquant
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
MODEL=weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400
DFLASH=weights/Qwen3.6-35B-A3B-DFlash
OUT=work_dirs/qwen3_5_moe_xh2a_$(date +%Y%m%d_%H%M%S)
```

默认兼容参数：`split_conv_cache=True`、`normalize_force_fp32=False`、`use_manual_depthwise_conv1d=False`、`fuse_gdr_ops=False`。MoE demo 建议加 `--resource-tight-mode`，避免 prefill/decode 同时驻留导致显存压力过大。

### 2. 标准导出 + Golden

```bash
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model "$MODEL" \
  --work-dir "$OUT/standard" \
  --context-length 512 \
  --input-sequence-length 128 \
  --split-conv-cache \
  --golden
```

### 3. 标准 HMONNX demo / benchmark

```bash
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_hmonnx_test.py \
  --config "$OUT/standard/meta.json" \
  --prompt "你好，请用几句话介绍你自己，并说明你能做什么。" \
  --max-new-tokens 128 \
  --warmup-runs 0 \
  --benchmark-runs 1 \
  --resource-tight-mode \
  --device cuda \
  --exec-device cuda
```

### 4. MTP 导出 + Golden + 长 token 测试

```bash
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model "$MODEL" \
  --work-dir "$OUT/mtp" \
  --context-length 512 \
  --input-sequence-length 128 \
  --split-conv-cache \
  --spec-decode-mode mtp \
  --num-draft-tokens 2 \
  --golden

python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_test.py \
  --config "$OUT/mtp/meta.json" \
  --prompt "你好，请用几句话介绍你自己，并说明你能做什么。" \
  --max_new_tokens 128 \
  --warmup_runs 0 \
  --benchmark_runs 1 \
  --resource_tight_mode \
  --device cuda \
  --exec_device cuda
```

### 5. DFlash 导出 + Golden + 长 token 测试

```bash
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model "$MODEL" \
  --work-dir "$OUT/dflash" \
  --context-length 512 \
  --input-sequence-length 128 \
  --split-conv-cache \
  --spec-decode-mode dflash \
  --num-draft-tokens 2 \
  --dflash-model-dir "$DFLASH" \
  --golden

python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_test.py \
  --config "$OUT/dflash/meta.json" \
  --prompt "你好，请用几句话介绍你自己，并说明你能做什么。" \
  --max_new_tokens 128 \
  --warmup_runs 0 \
  --benchmark_runs 1 \
  --resource_tight_mode \
  --device cuda \
  --exec_device cuda
```

## QTL-357 实测记录（2026-06-04）

本次 MoE 按任务要求 **未重新导出模型**，直接复测已有 split conv cache 产物；因此下表 MoE 产物不是本次新导出，只是本次实测使用的既有产物。

```bash
BASE=/data01/home/yujy/work/xh2modelzoo/work_dirs/full_goal_20260604_0824
OLD=work_dirs/splitconv_verify_20260603_220839
```

### 本次复测产物

| 类型 | `meta.json` | 说明 |
|------|-------------|------|
| Demo / 标准推理 | `$OLD/qwen35_moe_split/meta.json` | `architecture=Qwen3_5MoeForConditionalGeneration`，`kv=10`，`linear=30`，`max_context_tokens=512` |
| MTP | `$OLD/qwen35_moe_mtp_split/meta.json` | 旧 meta 中 `spec_decode=None`，runtime 从 `draft_onnx/*mtp*` 识别 MTP 并实测生效 |
| DFlash | `$OLD/qwen35_moe_dflash_split_retry/meta.json` | 旧 meta 中 `spec_decode=None`，runtime 从 `draft_onnx/*dflash*` 识别 DFlash 并实测生效 |

> 说明：MoE HMONNX 首次加载时会在 CPU 侧解析 / 编译 ONNX，期间 `nvidia-smi` 可能显示 GPU 利用率为 0；本次 benchmark 表内 latency 是脚本输出的生成耗时，不包含首次 ONNX parse 的墙钟耗时。

### 本次测试命令摘要

```bash
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_hmonnx_test.py \
  --config "$OLD/qwen35_moe_split/meta.json" \
  --prompt "你好，请用几句话介绍你自己，并说明你能做什么。" \
  --max-new-tokens 64 \
  --warmup-runs 0 \
  --benchmark-runs 1 \
  --no-sample

python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_test.py \
  --config "$OLD/qwen35_moe_mtp_split/meta.json" \
  --prompt "你好，请用几句话介绍你自己，并说明你能做什么。" \
  --max_new_tokens 64 \
  --warmup_runs 0 \
  --benchmark_runs 1

python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_spec_decode_test.py \
  --config "$OLD/qwen35_moe_dflash_split_retry/meta.json" \
  --prompt "你好，请用几句话介绍你自己，并说明你能做什么。" \
  --max_new_tokens 64 \
  --warmup_runs 0 \
  --benchmark_runs 1
```

### 本次测试结果

| 类型 | max tokens | 输出 token | latency(s) | tok/s | Spec 统计 | 接收率 |
|------|------------|------------|------------|-------|-----------|--------|
| Demo / 标准推理 | 64 | 64 | 45.0978 | 1.4191 | - | - |
| MTP | 64 | 64 | 12.7042 | 5.0377 | `rounds=23 total=64 draft=46 accepted=41` | `41/46 = 89.13%` |
| DFlash | 64 | 64 | 27.9811 | 2.2873 | `rounds=49 total=64 draft=98 accepted=14` | `14/98 = 14.29%` |

日志：

```bash
$BASE/logs/qwen35_moe_standard_mtp_dflash_long64_20260604.log
```

## Split Conv Cache 导出

`--split-conv-cache` 将线性注意力的 conv_cache 从单个合并 tensor 拆分为 3 个独立 tensor（q, k, v）。当前默认开启；如需旧 merged 格式可显式传 `--no-split-conv-cache`。

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
| merged (`--no-split-conv-cache`) | `past_conv_cache_0` | `conv_cache_out_0` |
| 默认 split | `past_conv_cache_q_0` / `past_conv_cache_k_0` / `past_conv_cache_v_0` | `conv_cache_out_q_0` / `conv_cache_out_k_0` / `conv_cache_out_v_0` |

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
  --config work_dirs/qwen3_6_35b_a3b_xh2a_2k_w4a8h0_ssfp_gptq_norm_fp32_False_20260526_221139/meta.json \
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

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_6_35b_a3b
MODEL_DIR=weights/Qwen3.6-35B-A3B
QUANT_WEIGHT=weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_8k_w4a8h0_ssfp_gptq_norm_fp32_True_${TS}"
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model "$MODEL_DIR" \
  --context-length 2048 \
  --input-sequence-length 256 \
  --quant-type w4a8h0_ssfp \
  --quant-weight "$QUANT_WEIGHT" \
  --golden \
  --work-dir "$WORK_DIR"

export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=2
TS=$(date +%Y%m%d_%H%M%S)
MODEL_KEY=qwen3_6_35b_a3b
MODEL_DIR=weights/Qwen3.6-35B-A3B
QUANT_WEIGHT=weights/qwen36moe-no-rotate-attn8-shared8-n256-iter400
WORK_DIR="work_dirs/${MODEL_KEY}_xh2a_2k_w4a8h0_ssfp_gptq_norm_mtp_${TS}"
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model "$MODEL_DIR" \
  --context-length 2048 \
  --input-sequence-length 256 \
  --quant-type w4a8h0_ssfp \
  --quant-weight "$QUANT_WEIGHT" \
  --work-dir "$WORK_DIR" \
  --golden \
  --package-release \
  --release_xh_version xh2 \
  --release_wmix_amix wmix_amix \
  --spec-decode-mode mtp \
  --num-draft-tokens 4 \
  --split_conv_cache
```

## XH2 规范导出

按照《HM 模型版本发布命名规则》一键导出 + 打包发布产物。导出脚本会把
prefill / decode 按规范布局组织到 `work_dirs/<release_prefix>/` 下，并在末尾
打成 `<release_prefix>.zip`。

`<release_prefix>` 形如：

```
hmquant_<xh1|xh2>_<modelscope_name>_<wmix_amix>_<prefill>_<context>_<date>
```

全部小写。`xh_version` 仅允许 `xh1` / `xh2`；wmix_amix 字段必须是纯定点的
`w<bits>a<bits>`（例如 `w4a8`、`w8a8`），其余一律归并为 `wmix_amix`。
对 MoE 量化（`w8a8h0_sefp` / `w4a8h0_ssfp` 等子模式）默认会归并到 `wmix_amix`。

> 旧版本：`README.md` 顶部沿用的命令使用脚本旧的目录结构（`prefill_onnx/`、
> `decode_onnx/`、`token_embedding.pt`、`-spec_<mode>` 后缀工作目录），保留作为
> 历史参考；新发布请优先使用以下 XH2 规范导出命令。

最小化命令（MoE Qwen3.6-35B-A3B，xh2 规范导出 + 打包）：

```bash
export PYTHONPATH=./
export CUDA_VISIBLE_DEVICES=0
TS=$(date +%Y%m%d)
MODEL_DIR=/data01/nfs_shared/Qwen3.6-35B-A3B
WORK_DIR="work_dirs/qwen3_6_35b_a3b_xh2_release_${TS}"

python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model "$MODEL_DIR" \
  --context-length 2048 \
  --input-sequence-length 256 \
  --quant-type w8a8h0_sefp \
  --work_dir "$WORK_DIR" \
  --golden \
  --release_xh_version xh2 \
  --release_modelscope_name qwen3_6_35b_a3b \
  --release_wmix_amix wmix_amix \
  --release_date "$TS" \
  --package_release
```

导出后产物路径：

- `<WORK_DIR>/<release_prefix>/prefill/<release_prefix>_prefill_with_act.onnx`
- `<WORK_DIR>/<release_prefix>/prefill/<release_prefix>_prefill_external_data`
- `<WORK_DIR>/<release_prefix>/prefill/step_0/`（指向上述两个文件的相对软链）
- `<WORK_DIR>/<release_prefix>/decode/<release_prefix>_decode_with_act.onnx`
- `<WORK_DIR>/<release_prefix>/decode/<release_prefix>_decode_external_data`
- `<WORK_DIR>/<release_prefix>/decode/step_0/`
- `<WORK_DIR>/<release_prefix>/golden_meta_info.json`
- `<WORK_DIR>/<release_prefix>/quant_embedding.pt`
- `<WORK_DIR>/<release_prefix>/<release_prefix>_hmonnx.py`
- `<WORK_DIR>/<release_prefix>/<release_prefix>_hmonnx_debug.log`
- `<WORK_DIR>/<release_prefix>.zip`（`zip -r -y` 保留软链）

仅打包既有 work_dir（不重新导出）：

```bash
python examples/llm/qwen3_5_moe/qwen3_5_moe_xh2a_export_hmonnx.py \
  --model "$MODEL_DIR" \
  --context-length 2048 \
  --input-sequence-length 256 \
  --work_dir "$WORK_DIR" \
  --golden-only \
  --release_xh_version xh2 \
  --release_modelscope_name qwen3_6_35b_a3b \
  --release_wmix_amix wmix_amix \
  --release_date "$TS" \
  --package_release
```

可用的发布 CLI 参数（均可省略，由脚本按规则推导默认值）：

| 参数 | 含义 |
|------|------|
| `--release_xh_version`     | `xh1` 或 `xh2`，argparse 会拒绝 `xh2a` 等取值；脚本内部 `_build_release_prefix_moe` 也会再做一次 ValueError 兜底。 |
| `--release_modelscope_name`| 默认取 `--model` 末段（小写化、`.`/`-` → `_`），例如 `Qwen3.6-35B-A3B` → `qwen3_6_35b_a3b`。 |
| `--release_wmix_amix`      | 纯定点 `w\d+a\d+` 保留；其它一律归并为 `wmix_amix`。 |
| `--release_date`           | 形如 `YYYYMMDD`，默认今天。 |
| `--package_release`        | golden 完成后用 `zip -r -y` 打成 `.zip`。 |

