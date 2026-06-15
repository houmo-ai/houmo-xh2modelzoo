# VLA-JEPA HMONNX 导出与评测说明

本目录包含 HMSW-4354 量化流程中使用的 VLA-JEPA 导出、对比、误差定位和 LIBERO 评测脚本。

## 模型拆分

当前运行链路按 HMONNX 可替换模块拆成三部分：

1. `Qwen visual encoder`：把图像特征转换成 visual embeddings。
2. `Qwen/context graph`：根据 prompt、visual embeddings、embodied positions 和 deepstack features 生成 `conditioning_tokens`。
3. `ActionHead`：扩散/去噪动作头，根据 `conditioning_tokens` 和机器人状态预测动作。

video/world-model 相关模块在需要加载完整 policy 时仍会随模型加载，但 Full-HMONNX 评测链路主要替换的是 Qwen visual encoder、context graph 和 ActionHead。

## 目录结构

```text
examples/vla/vla_jepa/
  common/      公共模型加载和结构检查工具
  export/      ONNX/HMONNX 导出入口
  compare/     PyTorch 与 ONNX/HMONNX 数值对比脚本
  eval/        LIBERO rollout 评测和多 GPU 调度器
  debug/       context 误差定位、样本抓取和局部 probe 脚本
  patches/     HMONNX 图后处理工具
```

## 环境准备

LIBERO CPU 渲染通常需要设置：

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 按实际机器路径设置；不设置时脚本会从当前 repo 推导默认 workspace。
export XH2MODELZOO_ROOT=/path/to/workspace/xh2model/xh2modelzoo
export WORKSPACE=/path/to/workspace
export VLA_JEPA_MODEL=$WORKSPACE/models/lerobot/VLA-JEPA-LIBERO
export VLA_JEPA_OUTPUT_ROOT=$WORKSPACE/outputs/vla_jepa
export LIBERO_CONFIG_PATH=$WORKSPACE/.libero
```

模型、输出目录和 LIBERO 配置都可以通过上面的环境变量覆盖；命令行参数仍然优先。下文中的相对路径均相对 `$XH2MODELZOO_ROOT`。

建议从仓库根目录执行命令：

```bash
cd $XH2MODELZOO_ROOT
```

## 1. 检查模型结构

第一步建议先运行 `inspect_policy.py`，确认模块名、hook 点、tensor shape 和需要 patch 的位置。

```bash
python examples/vla/vla_jepa/common/inspect_policy.py \
  --model $VLA_JEPA_MODEL \
  --device cuda
```

## 2. 导出 ActionHead

```bash
python examples/vla/vla_jepa/export/export_action_head.py \
  --model $VLA_JEPA_MODEL \
  --out-dir $VLA_JEPA_OUTPUT_ROOT/standard_export/action_head
```

典型产物：

```text
../../outputs/vla_jepa/standard_export/action_head/vla_jepa_action_head_step_w8a8_sefp.hmonnx.onnx
```

对比 ActionHead HMONNX 和 PyTorch 输出：

```bash
python examples/vla/vla_jepa/compare/compare_action_head_hmonnx.py
python examples/vla/vla_jepa/compare/compare_action_head_hmonnx_libero.py
```

## 3. 导出 Qwen Visual Encoder

```bash
python examples/vla/vla_jepa/export/export_qwen_visual_encoder.py \
  --model $VLA_JEPA_MODEL
```

典型产物：

```text
../../outputs/vla_jepa/context_encoder/qwen_visual_encoder/vla_jepa_qwen_visual_encoder_w8a8_sefp.hmonnx.onnx
```

对比 visual encoder HMONNX 和 PyTorch 输出：

```bash
python examples/vla/vla_jepa/compare/compare_qwen_visual_encoder_hmonnx.py
```

## 4. 导出 Context Graph

Context graph 负责导出生成 `conditioning_tokens` 的 Qwen/context 部分。
当前实际评测使用 `ctx256` 形状，并将部分累计误差敏感的 linear 层保留为 W16。

```bash
python examples/vla/vla_jepa/export/export_context_graph_wrapper.py \
  --model $VLA_JEPA_MODEL
```

典型产物：

```text
../../outputs/vla_jepa/standard_export_ctx256/context/vla_jepa_context_graph_wrapper_float16_w8a16_sefp_linear_w16a16_sefp.fused_rmsnorm.hmonnx.onnx
```

对比 context ONNX/HMONNX 和 PyTorch 输出：

```bash
python examples/vla/vla_jepa/compare/compare_context_graph_onnx.py
python examples/vla/vla_jepa/compare/compare_context_graph_hmonnx.py
```

## 5. 一键导出入口

`vla_jepa_xh2a_export_hmonnx.py` 是标准 ActionHead 和 context 导出流程的组合入口。
如果要定位某个模块的问题，优先使用上面的单模块脚本。

```bash
python examples/vla/vla_jepa/export/vla_jepa_xh2a_export_hmonnx.py \
  --model $VLA_JEPA_MODEL
```

## 6. LIBERO 评测

### Baseline / ActionHead-only 评测脚本

使用 `--mode original` 时是 PyTorch baseline。

```bash
python examples/vla/vla_jepa/eval/eval_action_head_hmonnx_libero.py \
  --model $VLA_JEPA_MODEL \
  --mode original \
  --device cuda:0 \
  --task libero_10 \
  --task-ids 0 \
  --n-episodes 5 \
  --seed 1000 \
  --max-episodes-rendered 0
```

### Full-HMONNX 评测脚本

```bash
python examples/vla/vla_jepa/eval/eval_full_hmonnx_libero.py \
  --model $VLA_JEPA_MODEL \
  --visual-hmonnx $VLA_JEPA_OUTPUT_ROOT/context_encoder/qwen_visual_encoder/vla_jepa_qwen_visual_encoder_w8a8_sefp.hmonnx.onnx \
  --context-hmonnx $VLA_JEPA_OUTPUT_ROOT/standard_export_ctx256/context/vla_jepa_context_graph_wrapper_float16_w8a16_sefp_linear_w16a16_sefp.fused_rmsnorm.hmonnx.onnx \
  --action-hmonnx $VLA_JEPA_OUTPUT_ROOT/standard_export/action_head/vla_jepa_action_head_step_w8a8_sefp.hmonnx.onnx \
  --device cuda:0 \
  --hmonnx-device cuda:0 \
  --context-dtype float16 \
  --task libero_10 \
  --task-ids 0 \
  --n-episodes 2 \
  --seed 1000 \
  --max-episodes-rendered 0
```

## 7. 多 GPU LIBERO-10 调度器

调度器会在 10 个 LIBERO task 上分别运行 baseline 和 Full-HMONNX，每个 task 使用 20 个 seed。
它支持断点续跑，会自动跳过已有的完整 report，并持续写入 summary。

Dry-run，只打印任务计划，不启动评测：

```bash
python examples/vla/vla_jepa/eval/run_libero10_20eps_dynamic.py \
  --gpus 4,5,6,7 \
  --workers-per-gpu 2 \
  --preview-jobs 6
```

正式运行：

```bash
python examples/vla/vla_jepa/eval/run_libero10_20eps_dynamic.py \
  --run \
  --gpus 4,5,6,7 \
  --workers-per-gpu 2
```

主要汇总文件：

```text
../../outputs/vla_jepa/eval/libero10_20eps/summary.json
../../outputs/vla_jepa/eval/libero10_20eps/summary.csv
```

## 8. 评测实验记录

本次提交对应的完整 LIBERO-10 评测使用 `eval/run_libero10_20eps_dynamic.py` 调度完成，目的是对比 PyTorch baseline 和 Full-HMONNX 全链路在实际任务中的成功率。

### 实验配置

```text
任务集: libero_10
任务 ID: 0-9
每个任务 episode 数: 20
seed 范围: 1000-1019
baseline chunk size: 5 episodes/job
Full-HMONNX chunk size: 2 episodes/job
batch size: 1
视频导出: 关闭
渲染后端: osmesa
调度 GPU: 4,5,6,7
workers-per-gpu: 2
baseline workers: 1
Full-HMONNX workers: 7
```

Full-HMONNX 使用的三个 HMONNX 文件：

```text
visual:  ../../outputs/vla_jepa/context_encoder/qwen_visual_encoder/vla_jepa_qwen_visual_encoder_w8a8_sefp.hmonnx.onnx
context: ../../outputs/vla_jepa/standard_export_ctx256/context/vla_jepa_context_graph_wrapper_float16_w8a16_sefp_linear_w16a16_sefp.fused_rmsnorm.hmonnx.onnx
action:  ../../outputs/vla_jepa/standard_export/action_head/vla_jepa_action_head_step_w8a8_sefp.hmonnx.onnx
```

### 执行流程

1. 使用 `--mode original` 跑 PyTorch baseline。
2. 使用 `eval_full_hmonnx_libero.py` 跑 Qwen visual、context graph、ActionHead 三段 HMONNX。
3. 使用动态调度器把 10 个任务、20 个 seed 拆成 140 个 job。
4. 调度器根据已有 report 自动 resume，跳过已经完整完成的 job。
5. 所有 job 完成后汇总到 `summary.json` 和 `summary.csv`。

实际使用的调度命令：

```bash
python examples/vla/vla_jepa/eval/run_libero10_20eps_dynamic.py \
  --run \
  --gpus 4,5,6,7 \
  --workers-per-gpu 2
```

### 结果文件

```text
../../outputs/vla_jepa/eval/libero10_20eps/summary.json
../../outputs/vla_jepa/eval/libero10_20eps/summary.csv
```

每个 job 的单独 report 位于：

```text
../../outputs/vla_jepa/eval/libero10_20eps/baseline/task*/
../../outputs/vla_jepa/eval/libero10_20eps/full_hmonnx/task*/
```

每个 job 的运行日志位于：

```text
../../logs/vla_jepa_libero10_20eps/
```

### 总体结果

```text
Baseline:    191 / 200 = 95.5%
Full-HMONNX: 183 / 200 = 91.5%
差值:        -4.0%
```

### 逐任务成功率

| Task | Baseline | Full-HMONNX | 差值 |
|---:|---:|---:|---:|
| 0 | 19/20 = 95% | 20/20 = 100% | +5% |
| 1 | 20/20 = 100% | 20/20 = 100% | 0% |
| 2 | 20/20 = 100% | 20/20 = 100% | 0% |
| 3 | 17/20 = 85% | 14/20 = 70% | -15% |
| 4 | 19/20 = 95% | 13/20 = 65% | -30% |
| 5 | 20/20 = 100% | 20/20 = 100% | 0% |
| 6 | 18/20 = 90% | 19/20 = 95% | +5% |
| 7 | 20/20 = 100% | 20/20 = 100% | 0% |
| 8 | 20/20 = 100% | 19/20 = 95% | -5% |
| 9 | 18/20 = 90% | 18/20 = 90% | 0% |

主要掉点集中在 task3 和 task4，尤其是 task4 从 95% 降到 65%。后续如果继续优化，优先复查这两个任务对应的 context 输出误差和动作序列差异。

## Debug 与 Patch 脚本

`debug/` 目录用于定位 context graph 误差来源：

- 抓取 dummy 或 LIBERO rollout 中的 context 输入
- 对比 graph prefix、单层、局部 attention、final norm 等部分
- 定位 q/k/v、RoPE、RMSNorm、attention softmax 等位置的误差

`patches/` 目录用于 HMONNX 图后处理，例如保留精度敏感节点或融合 RMSNorm。
这些脚本不是常规评测入口，只有在导出报告或误差分析明确需要时才使用。

## 提交注意事项

不要提交生成产物：

- 模型权重
- `outputs/`
- 视频
- `.onnx.data`
- `__pycache__/`

只提交源码脚本和文档。
