# Cosmos3-Nano HMONNX 量化工程

本目录用于把 Cosmos3-Nano HF/safetensors 模型目录拆成可导出、可量化、可测试的 HMONNX/ONNX 分图模型包。

当前目标不是导出一个单体 ONNX，也不是把所有 op 统一压成 W8A8。Cosmos3-Nano 是多 surface 模型，量化边界按实际推理链路拆分：

```text
Cosmos3-Nano
  Reasoner
    text/vision embedding
    vision encoder
    AR prefill-KV
    AR decode
    logits head

  Generator
    diffusion denoiser
    CFG scheduler host loop
    VAE encoder/decoder
    sound tokenizer decoder

  Forward Dynamics
    observation/future latent path
    action-conditioned denoiser
    FD scheduler host loop
    VAE decoder reuse

  Policy
    action denoiser backbone / segments
    action head
    action postprocess host logic
```

控制逻辑保留在 Python host runtime，不进量化图：

```text
tokenizer / detokenizer
AR decode loop
diffusion scheduler
CFG combine
KV cache allocation
latent buffer reshape
action unpadding / denormalization / clamp
simulator rollout
```

## 目录结构

```text
examples/vla/cosmos3_nano/
  common/      官方 Cosmos3 模型加载、patch、shape/权重检查
  export/      ONNX/HMONNX 导出入口，按模块拆分
  quantize_all.py              一键量化入口
  demo_quantized_model.py      量化后单样例 demo 入口
  runtime/     分图 host runtime 实现
  quantize/    各组件量化脚本
```

远端仓库只保留可复现工程代码和必要配置；eval/ 是本地验证/仿真入口，依赖本地数据集、仿真环境和运行状态，不随 git push 上传到远端仓库。后续新增本地验证脚本也应放在该目录（eval/）下，并由本目录 .gitignore 排除：

```text
  eval/
```

核心入口：

```text
quantize_all.py
quantize/quantize_reasoner.py
quantize/quantize_generator.py
quantize/quantize_forward_dynamics.py
quantize/quantize_policy.py
demo_quantized_model.py
```

## 环境

推荐从 workspace 根目录运行：

```bash
cd <workspace>
PY=python
export COSMOS3_NANO_MODEL_ROOT=<path-to-Cosmos3-nano>
MODEL=
ROOT=xh2model/xh2modelzoo/examples/vla/cosmos3_nano
```

如果跑 LIBERO/MuJoCo 仿真，需要：

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export NUMBA_DISABLE_JIT=1
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH=<path-to-xhquant-lib>:$LD_LIBRARY_PATH
```

## 一键量化整个模型

Smoke profile 用于快速验证流程和目录产物：

```bash
$PY $ROOT/quantize_all.py \
  --model-root $MODEL \
  --out-dir $ROOT/data/quantized_all \
  --profile smoke \
  --runtime-device cuda \
  --continue-on-error
```

Full profile 会尝试 36 层链路，耗时和显存压力明显更高：

```bash
$PY $ROOT/quantize_all.py \
  --model-root $MODEL \
  --out-dir $ROOT/data/quantized_all_full \
  --profile full \
  --runtime-device cuda \
  --continue-on-error
```

常用参数：

```text
--components reasoner,generator,forward_dynamics,policy
--dry-run                  只打印命令
--force                    已有产物也重跑
--continue-on-error        某个组件失败后继续后续组件
--skip-sound-tokenizer     Generator 跳过 sound tokenizer decoder
--policy-segments ...      Policy 分段方案
--policy-fp-segments ...   Policy 中保持 FP ONNX 的段
--fd-segments ...          Forward Dynamics 分段方案
```

输出目录：

```text
data/quantized_all/
  quantize_all_summary.json
  quantize_all_summary.md
  logs/
  reasoner/
  generator/
  forward_dynamics/
  policy/
```

## 量化后模型单样例 demo

量化产物不是一个单体模型文件，而是 reasoner/generator/forward_dynamics/policy 多个固定 shape 分图。`demo_quantized_model.py` 会读取量化 summary，自动选择一个已记录的 smoke runtime stage，重跑该 stage 内置的固定样例输入，并在终端打印输出 report 中的核心指标。

自动选择可用组件，优先顺序为 policy、forward_dynamics、generator、reasoner：

```bash
$PY $ROOT/demo_quantized_model.py \
  --quant-dir $ROOT/data/quantized_all
```

指定组件运行一个样例：

```bash
$PY $ROOT/demo_quantized_model.py \
  --quant-dir $ROOT/data/quantized_all \
  --component policy \
  --print-command
```

如果只想读取已有 report，不重跑样例：

```bash
$PY $ROOT/demo_quantized_model.py \
  --quant-dir $ROOT/data/quantized_all \
  --component generator \
  --keep-existing-report
```

输出：

```text
data/quantized_all/demo_quantized_model_summary.json
data/quantized_all/demo_logs/
```

`demo_quantized_model_summary.json` 会汇总：

```text
实际运行的 component/stage
report 路径和是否存在
cosine / mean_abs_diff / raw_action_cosine / pc_success 等 JSON report 指标
```

## 各模块使用方式

### Reasoner

量化：

```bash
$PY $ROOT/quantize/quantize_reasoner.py \
  --model-root $MODEL \
  --out-dir $ROOT/data/quantized_reasoner \
  --profile smoke
```

完整 36 层：

```bash
$PY $ROOT/quantize/quantize_reasoner.py \
  --model-root $MODEL \
  --out-dir $ROOT/data/quantized_reasoner_full \
  --profile full
```

拆分链路：

```text
text_embedding
vision_patch_embed
vision_encoder_core
prefill_kv_stack
decode_stack
logits_head
```

Runtime：

```text
runtime/reasoner_runtime.py
```

### Generator

量化：

```bash
$PY $ROOT/quantize/quantize_generator.py \
  --model-root $MODEL \
  --out-dir $ROOT/data/quantized_generator \
  --profile smoke
```

完整 36 层：

```bash
$PY $ROOT/quantize/quantize_generator.py \
  --model-root $MODEL \
  --out-dir $ROOT/data/quantized_generator_full \
  --profile full
```

跳过 sound tokenizer：

```bash
--skip-sound-tokenizer
```

拆分链路：

```text
real-packed diffusion denoiser
CFG scheduler host loop
VAE encoder
VAE decoder pre_mid + up0/up1/up2/up3/head split-chain
sound tokenizer decoder split-chain
generator scheduler -> VAE decoder pipeline smoke
```

Runtime：

```text
runtime/generator_scheduler_runtime.py
runtime/generator_pipeline_runtime.py
runtime/vae_encoder_runtime.py
runtime/vae_decoder_runtime.py
runtime/sound_tokenizer_runtime.py
```

### Forward Dynamics

量化：

```bash
$PY $ROOT/quantize/quantize_forward_dynamics.py \
  --model-root $MODEL \
  --out-dir $ROOT/data/quantized_forward_dynamics \
  --profile smoke
```

分段 36 层：

```bash
$PY $ROOT/quantize/quantize_forward_dynamics.py \
  --model-root $MODEL \
  --out-dir $ROOT/data/quantized_forward_dynamics_segments_36 \
  --segments 0:8,8:8,16:8,24:12 \
  --runtime-device cuda \
  --continue-on-error
```

拆分链路：

```text
und/text condition
future latent tokens
action tokens
shared transformer denoiser
FD scheduler host loop
VAE decoder reuse
```

Runtime：

```text
runtime/forward_dynamics_runtime.py
runtime/forward_dynamics_scheduler_runtime.py
```

### Policy

量化：

```bash
$PY $ROOT/quantize/quantize_policy.py \
  --model-root $MODEL \
  --out-dir $ROOT/data/quantized_policy \
  --profile smoke
```

推荐混精边界：

```text
0-5 W8A16
layer6 mlp.down_proj W16A16
layer7 W8A16
layer8 W8A16
9-35 FP ONNX
action_head FP16
```

当前一键脚本支持分段和 FP 段：

```bash
$PY $ROOT/quantize/quantize_policy.py \
  --model-root $MODEL \
  --out-dir $ROOT/data/quantized_policy_mixed \
  --segments 0:6,6:1,7:2,9:27 \
  --fp-segments 9:27 \
  --runtime-device cuda \
  --continue-on-error
```

拆分链路：

```text
action denoiser init segment
action denoiser stack segment(s)
action head FP16 ONNX
host scheduler / raw_action_dim unpadding / denormalization / clamp
```

Runtime：

```text
runtime/policy_runtime.py
```

## LIBERO 仿真评测

本地 LIBERO/MuJoCo 环境已通过 VLA-JEPA baseline smoke：

```text
task: libero_10/task0
seed: 1000
episodes: 1
pc_success: 100.0
avg_sum_reward: 1.0
avg_max_reward: 1.0
report: data/sim_eval/env_smoke/vla_jepa_pytorch_task0_seed1000_report.json
```

Cosmos3-LIBERO 端到端还不能直接跑，因为当前本地没有 Cosmos3-LIBERO policy checkpoint/config：

```text
COSMOS3_NANO_MODEL_ROOT 指向公开 HF/safetensors 基座；未设置时默认查找 examples/vla/cosmos3_nano/data/Cosmos3-nano
未找到 Cosmos3*Policy* 的 LIBERO checkpoint/config
127.0.0.1:8000 返回 HTTP 503，不是可用的 Cosmos action server
```

已有评测入口：

```text
eval/eval_policy_server_libero.py
```

它用于对接官方 `cosmos_framework.scripts.action_policy_server_libero`：

```bash
$PY $ROOT/eval/eval_policy_server_libero.py \
  --server-url http://127.0.0.1:8000 \
  --domain-name libero \
  --task libero_10 \
  --task-ids 0 \
  --n-episodes 1 \
  --seed 1000 \
  --report $ROOT/data/sim_eval/policy_server/task0_seed1000_report.json
```

要真正比较 Cosmos FP 与 Cosmos quant 的 LIBERO 成功率，需要先准备：

```text
Cosmos3 LIBERO action-policy checkpoint
对应 config.yaml
action stats JSON
可用的 action_policy_server_libero 服务
```

## 当前量化结果

### Reasoner

| 模块 | 量化路径 | 最近结果 | 状态 |
| --- | --- | --- | --- |
| vision patch embed | W8A16 HMONNX | ONNX/HMONNX split chain report 已存在 | 已完成 |
| vision encoder core | W8A16 HMONNX + 4D attention layout | 绕开 RoPE/Transpose rank issue | 已完成 |
| logits head | W8A16 HMONNX | cosine `0.9999178`, mean_abs `0.03218`, top1_match true | 已完成 |
| reasoner E2E | 多图串联 runtime | prefill logits cosine `0.966920256`, decode `0.989707887`, top1_match `1.0` | 已闭环 |

### Generator

| 模块 | 量化路径 | 最近结果 | 状态 |
| --- | --- | --- | --- |
| diffusion denoiser 2-layer | W8A16 real-packed boundary | cosine `0.9998951`, mean_abs `0.014439`, max_abs `0.06148` | 已完成 |
| diffusion denoiser 36-layer | W8A16 real-packed boundary | cosine `0.992977`, mean_abs 约 `0.3669` | 可跑，full 需分段/混精策略 |
| CFG scheduler loop | host runtime + HMONNX denoiser | 36-layer 2-step final_latents cosine `0.998822` | 已完成 |
| VAE encoder | W16A16 HMONNX + Conv3D lowered to 2D | cosine `0.999998033`, mean_abs `0.0008499` | 已完成 |
| VAE decoder | W16A16 split-chain + `Normalize.force_fp32=True` | final cosine `0.999991893`, mean_abs `0.0014573`, max_abs `0.0099882` | 已完成 |
| sound decoder | W16A16 split-chain | chain cosine `0.998910` | decoder 已完成；encoder 权重缺失 |

### Forward Dynamics

| 模块 | 量化路径 | 最近结果 | 状态 |
| --- | --- | --- | --- |
| denoiser 2-layer | W8A16 | future_latent_velocity cosine `0.9998843`, mean_abs `0.0151974` | 已完成 |
| denoiser 4-layer | W8A16 | denoiser cosine `0.9997279`, scheduler final_latents `0.9883814` | 已完成 |
| denoiser 8-layer | W8A16 | denoiser cosine `0.9997327`, scheduler final_latents `0.9831424` | 已完成，误差开始放大 |
| segmented 0:1,1:1 | two HMONNX + host stitch | future_latent_velocity cosine `0.9999123`, mse `0.0002738` | 已完成 |
| segmented 36-layer 0:8,8:8,16:8,24:12 | four HMONNX + host stitch | chained velocity cosine `0.9714049`, scheduler final_latents `0.9643221` | 已跑通，精度需继续优化 |

### Policy

| 配置 | 最近结果 | 结论 |
| --- | --- | --- |
| 2-layer W8A16 + FP16 head | raw_action_cosine `0.9999278`, mse `9.79e-06`, endpoint `0.01036` | 稳定 |
| 4-layer W8A16 + FP16 head | raw_action_cosine `0.9998594`, mse `5.32e-05`, endpoint `0.02246` | 稳定 |
| 8-layer W8A16 + FP16 head | raw_action_cosine `0.9990225`, mse `1.79e-04`, endpoint `0.04942` | 可用 |
| 36-layer W8A16 + FP16 head | raw_action_cosine `0.7657402`, mse `0.217536`, endpoint `1.23055` | 不可交付 |
| `0-5 W8A16 + 6-35 FP` | raw_action_cosine `0.9902307`, mse `0.009736`, endpoint `0.16682` | 保守可用 |
| `0-5 W8A16 + layer6 mlp.down_proj W16A16 + layer7 W8A16 + layer8 W8A16 + 9-35 FP + head FP16` | raw_action_cosine `0.9942794`, mse `0.005637`, endpoint `0.12120` | 当前推荐 |

Policy 误差定位结论：

```text
layer6 的主要敏感点在 shared MLP，尤其 mlp.down_proj。
layer9 直接 W8A16 会推高 endpoint error。
layer9 的 down_proj / shared attention / shared MLP / action branch / full MatMul W16A16 都不能恢复到停在 layer8 的效果。
跳过 layer9 后继续压 10-11、12-19、20-27、28-35 也不能达到当前推荐边界。
```

## 当前默认量化策略

| 模块 | 默认策略 |
| --- | --- |
| Reasoner | W8A16-SEFP |
| Generator denoiser | W8A16-SEFP |
| VAE encoder/decoder | W16A16-SEFP |
| Sound decoder | W16A16-SEFP |
| Forward Dynamics denoiser | W8A16-SEFP |
| Policy denoiser | W8A16-SEFP + 局部 W16A16/FP 混精 |
| Policy action head | FP16 ONNX |

全工程默认保留：

```text
Normalize.force_fp32=True
RMSNorm/Normalize 优先 fp32
```

## 推荐工作流

1. 先看一键命令：

```bash
$PY $ROOT/quantize_all.py \
  --model-root $MODEL \
  --out-dir /tmp/cosmos3_quant_dryrun \
  --profile smoke \
  --dry-run
```

2. 跑 smoke 全组件：

```bash
$PY $ROOT/quantize_all.py \
  --model-root $MODEL \
  --out-dir $ROOT/data/quantized_all \
  --profile smoke \
  --runtime-device cuda \
  --continue-on-error
```

3. 跑一个量化后样例 demo：

```bash
$PY $ROOT/demo_quantized_model.py \
  --quant-dir $ROOT/data/quantized_all \
  --component auto \
  --print-command
```

4. 再按需要单独跑 full：

```bash
$PY $ROOT/quantize/quantize_reasoner.py --model-root $MODEL --out-dir $ROOT/data/quantized_reasoner_full --profile full
$PY $ROOT/quantize/quantize_generator.py --model-root $MODEL --out-dir $ROOT/data/quantized_generator_full --profile full
$PY $ROOT/quantize/quantize_forward_dynamics.py --model-root $MODEL --out-dir $ROOT/data/quantized_fd_full --profile full
$PY $ROOT/quantize/quantize_policy.py --model-root $MODEL --out-dir $ROOT/data/quantized_policy_full --profile full
```

## 已知限制和下一步

1. Policy 36-layer 全 W8A16 当前不可交付，推荐使用混精边界。
2. FD 36-layer 粗分段已能跑通，但 cosine 下降到 `0.9714049`，需要更细分段或混精。
3. Generator 36-layer denoiser 可跑，但 full 仍建议按分段和 pipeline report 追踪。
4. sound tokenizer 只有 decoder 权重，encoder 当前无法闭环。
5. Cosmos3-LIBERO 端到端仿真缺 policy checkpoint/config，当前只能跑 VLA-JEPA baseline smoke 和 Cosmos policy 数值链路。

更详细的历史实验记录保留在：

```text
docs/code_architecture_and_runtime.md
docs/quantize_usage.md
docs/quantization_by_component.md
docs/runtime_and_compare.md
docs/full_quantization_workflow.md
```
