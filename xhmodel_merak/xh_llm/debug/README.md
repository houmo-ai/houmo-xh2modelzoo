# HMFP Analyzer（ManifoldProbe）：WQ + MXINT8 推理流形分析工具

## 0. 摘要

ManifoldProbe 是一个面向低比特大模型部署的精度退化归因工具。它通过同步执行 FP Teacher、Weight-Only Student 和 Deploy Student 三条路径，将最终输出错误回溯到具体 token、layer、operator 和 MXINT8 group，并用 patching / noise injection 验证误差是否具有因果影响。

该工具的核心产出不是单纯的误差表，而是一条可复现的证据链：

```text
错误 token
    -> 首次概率分布偏离
    -> 高风险层和算子
    -> 高风险 MXINT8 group
    -> patch 后恢复 / 注入后退化
    -> QAT / QAD 训练样本
```

第一阶段应优先服务两个目标：

1. 在单样本或小数据集上复现 WQ + MXINT8 部署路径相对 FP16 的 token 分叉。
2. 定位最可能导致分叉的 layer、operator 和 group，并生成可人工复核的 Markdown 报告。

## 1. 文档目的

本文档定义一个面向大模型低比特推理的分析工具，用于分析以下部署路径中的精度退化：

```text
FP16/BF16 原始模型
        |
        v
权重量化模型
        |
        v
推理时激活动态量化为 MXINT8
        |
        v
自研 NPU 数值路径
```

目标模型的实际计算形式为：

\[
\hat{Y}
=
\operatorname{Dequant}
\left[
\operatorname{MatMul}
\left(
Q_{\mathrm{MXINT8}}(X),
Q_W(W)
\right)
\right]
\]

其中：

- 权重已离线量化；
- 激活在推理过程中动态量化为 MXINT8；
- MXINT8 按 `group_size=64` 分组；
- 每组共享 scale 或 exponent；
- MatMul/Linear 使用自研硬件数值规则；
- 中间累加可能使用 FP24；
- 输出重新恢复为 FP16 或其他高精度格式。

该工具不是普通的张量误差统计器，而是一个从硬件数值误差追踪到模型推理行为变化的端到端分析系统。

### 1.1 适用场景

适用于以下问题：

- WQ 模型在 A16 路径正常，但接入 MXINT8 activation 后准确率明显下降；
- 单个 prompt 或少量样本在部署路径上出现稳定 token 分叉；
- 需要判断误差主要来自权重量化、激活量化，还是两者耦合；
- 需要从错误样本中挖掘 QAT / QAD hard examples；
- 需要验证某个硬件数值规则、MXINT8 group 划分或算子实现是否导致精度退化。

### 1.2 非目标

第一版不直接承担以下职责：

- 替代完整模型评测框架；
- 自动修复量化策略或自动训练模型；
- 覆盖所有 serving 系统行为，例如复杂调度、并发抢占和跨请求 cache 复用；
- 在没有硬件一致量化规则的前提下给出最终根因结论。

这些能力可以作为后续扩展，但 MVP 应优先保证离线单样本归因的可信度。

---

## 2. 核心问题

工具必须回答以下问题：

1. 模型从哪个 token 开始与 FP16 模型发生概率分布偏离？
2. 首次 top-1 token 分叉发生在哪里？
3. 表示几何从哪一层开始明显漂移？
4. 精度损失主要来自权重量化，还是 MXINT8 激活量化？
5. 哪个算子导致误差明显放大？
6. 哪个 `group=64` 的激活 block 是关键风险点？
7. 修复某一层、某个算子或某个 block 后，输出是否恢复？
8. 哪些样本、token、层和 block 应进入后续 QAT/QAD 训练集？

### 2.1 建议排查顺序

1. 先定位首次分叉的 token，缩小分析范围。
2. 再定位表示漂移开始出现的层。
3. 接着收敛到具体算子与 `group=64` block。
4. 最后通过 patching 评估修复收益并确定优先级。

---

## 3. 系统定位

系统由四类能力组成：

```text
Numerical Error Analyzer
        +
Representation Geometry Analyzer
        +
Token Trajectory Analyzer
        +
Causal Attribution Engine
```

最终形成以下闭环：

```text
Token divergence
      |
      v
Layer attribution
      |
      v
Operator attribution
      |
      v
MXINT8 group attribution
      |
      v
Hard-example mining
      |
      v
QAT / QAD recovery
      |
      v
Regression evaluation
```

---

## 4. 分析对象

必须同时运行三条模型路径。

### 4.1 FP Teacher

```text
FP16/BF16 weight
FP16/BF16 activation
```

\[
T(x)=F(x;W_{\mathrm{FP}})
\]

用途：

- 作为精度基准；
- 提供 teacher logits；
- 提供 teacher hidden states；
- 提供 activation patching 的参考值。

### 4.2 Weight-Only Student

```text
Quantized weight
FP16/BF16 activation
```

\[
S_W(x)=F(x;Q_W(W))
\]

用途：

- 分离权重量化误差；
- 判断权重量化后表示空间的变化；
- 作为 MXINT8 增量误差分析的基线。

### 4.3 Deploy Student

```text
Quantized weight
MXINT8 activation
```

\[
S_{WA}(x)
=
F_{\mathrm{MXINT8}}
\left(
x;Q_W(W)
\right)
\]

用途：

- 复现最终部署路径；
- 分析权重与激活量化的联合误差；
- 作为最终训练和评测对象。

### 4.4 误差拆分

\[
\Delta_W=S_W-T
\]

\[
\Delta_A=S_{WA}-S_W
\]

\[
\Delta_{\mathrm{total}}=S_{WA}-T
\]

| 项目 | 含义 |
|---|---|
| \(\Delta_W\) | 权重量化误差 |
| \(\Delta_A\) | MXINT8 激活引入的增量误差 |
| \(\Delta_{\mathrm{total}}\) | 最终部署模型相对 FP Teacher 的总误差 |

注意：

\[
\Delta Y
=
X\Delta W
+
\Delta XW
+
\Delta X\Delta W
\]

第三项表示权重和激活量化之间的耦合误差。

---

## 5. 总体架构

```text
Dataset / Prompt Set
        |
        v
+-------------------------------+
| Synchronized Execution Engine |
+-------------------------------+
   |             |             |
   v             v             v
FP Teacher    WQ + A16      WQ + MXINT8
   |             |             |
   +-------------+-------------+
                 |
                 v
        Unified Trace Recorder
                 |
                 v
+---------------------------------------------+
| Metrics / Geometry / Token / Causal Analysis |
+---------------------------------------------+
                 |
                 v
        Report & Hard-Example Store
```

### 5.1 最小可用闭环

MVP 不要求一次性实现全部模块。第一版应形成以下最短链路：

```text
Prompt
    -> 三路径 teacher-forcing
    -> layer output / logits / MX block stats
    -> first divergence 定位
    -> 高风险 layer/operator/group 排序
    -> Markdown 报告
```

只有当统计结果能稳定复现 token 分叉后，再进入 patching 和 noise injection。这样可以避免在基础 trace 尚未对齐时过早引入因果实验。

### 5.2 数据一致性要求

三条模型路径必须满足：

- 相同 tokenizer、prompt template 和 position id；
- 相同 attention mask、cache 策略和 RoPE 配置；
- 相同随机种子和 sampling 配置，teacher-forcing 阶段禁止随机采样；
- 相同 layer/operator 命名规则；
- 相同 token、layer、operator、group 索引语义。

任何路径不一致都会污染后续误差拆分，必须在 run summary 中显式记录。

---

## 6. 模块划分

```text
manifold_probe/
├── config/
├── runtime/
├── quant/
├── recorder/
├── alignment/
├── metrics/
├── attribution/
├── storage/
├── report/
├── training_export/
└── tests/
```

### 6.1 `runtime`

- 构造三种模型变体；
- 保证相同输入和随机种子；
- 执行 teacher-forcing；
- 执行 free-running generation；
- 管理 hook、patch 和 replay。

### 6.2 `quant`

- 实现真实 MXINT8 fake quant；
- 实现权重量化格式；
- 模拟 group、scale、exponent、round、saturation；
- 模拟 NPU 累加和输出反量化。

### 6.3 `recorder`

- 记录层、算子、token、group 级指标；
- 支持轻量统计、选择性张量和完整重放；
- 控制存储开销。

### 6.4 `alignment`

- 对齐三个模型的 token；
- 执行 teacher-forcing；
- 定位 soft divergence；
- 定位 top-1 divergence；
- 对齐推理步骤。

### 6.5 `metrics`

- 数值误差指标；
- MXINT8 block 指标；
- 表示几何指标；
- token 分布指标；
- 推理轨迹指标。

### 6.6 `attribution`

- layer patching；
- operator patching；
- group patching；
- noise injection；
- 因果恢复评分。

### 6.7 `report`

- 输出 Markdown、JSON、Parquet；
- 输出 layer/token/group heatmap；
- 生成根因报告；
- 生成训练样本清单。

### 6.8 `training_export`

- 导出高风险 prompt；
- 导出高风险 token window；
- 导出敏感层和敏感 group；
- 为 QAT/QAD 提供 hard-example 数据。

---

## 7. Trace 数据模型

```python
from dataclasses import dataclass
from enum import Enum


class ModelVariant(str, Enum):
    FP16 = "fp16"
    WQ_A16 = "wq_a16"
    WQ_MXINT8 = "wq_mxint8"


@dataclass(frozen=True)
class TraceKey:
    sample_id: str
    sequence_id: int
    token_id: int
    layer_id: int
    operator_name: str
    tensor_role: str
    group_id: int | None
    model_variant: ModelVariant
```

建议的 `tensor_role`：

```text
layer_input
norm_output
q_proj_input
q_proj_output
k_proj_input
k_proj_output
v_proj_input
v_proj_output
attention_score
attention_probability
o_proj_output
gate_proj_output
up_proj_output
down_proj_output
residual_before_add
residual_after_add
layer_output
logits
```

---

## 8. MXINT8 数据结构

```python
@dataclass
class MXBlockStats:
    exponent: int | None
    scale: float
    max_abs: float
    mean_abs: float
    rms: float
    max_rms_ratio: float
    saturation_ratio: float
    code_utilization: float
    zero_ratio: float
    nmse: float
    cosine_similarity: float
    effective_bits_mean: float | None
    effective_bits_min: float | None
```

\[
r_g
=
\frac{\max|x_g|}
{\operatorname{RMS}(x_g)+\epsilon}
\]

\[
E_g
=
\frac{
\|Q(x_g)-x_g\|_2^2
}{
\|x_g\|_2^2+\epsilon
}
\]

\[
s_g
=
\frac{
\#\{|q_i|=q_{\max}\}
}{
64
}
\]

\[
u_g
=
\frac{
|\{\text{实际使用码字}\}|
}{
|\{\text{可用码字}\}|
}
\]

---

## 9. Layer 数据结构

```python
@dataclass
class LayerStats:
    nmse_fp_vs_wq: float
    nmse_wq_vs_deploy: float
    nmse_fp_vs_deploy: float

    cosine_fp_vs_wq: float
    cosine_wq_vs_deploy: float
    cosine_fp_vs_deploy: float

    cka_fp_vs_wq: float | None
    cka_wq_vs_deploy: float | None
    cka_fp_vs_deploy: float | None

    neighborhood_overlap: float | None
    transition_cosine: float | None
    covariance_spectrum_distance: float | None
```

---

## 10. Token 数据结构

```python
@dataclass
class TokenStats:
    token_id: int
    token_text: str

    kl_fp_vs_wq: float
    kl_wq_vs_deploy: float
    kl_fp_vs_deploy: float

    js_fp_vs_wq: float
    js_wq_vs_deploy: float
    js_fp_vs_deploy: float

    top1_match_fp_vs_wq: bool
    top1_match_wq_vs_deploy: bool
    top1_match_fp_vs_deploy: bool

    topk_overlap_fp_vs_deploy: float

    teacher_target_rank: int
    deploy_target_rank: int

    teacher_margin: float
    weight_only_margin: float
    deploy_margin: float

    teacher_entropy: float
    deploy_entropy: float
```

---

## 11. Execution Recorder

### 11.1 Level 0：统计扫描

只保存：

- mean/std/max/RMS；
- norm；
- NMSE；
- cosine；
- exponent histogram；
- saturation ratio；
- code utilization；
- top-k logits。

用于全数据集扫描和候选筛选。

### 11.2 Level 1：选择性张量

只保存：

- 首次分叉附近 token；
- 高风险层；
- 高风险算子；
- 高风险 group；
- 选定 hidden state；
- 选定 logits。

### 11.3 Level 2：完整重放

保存单个样本的完整必要激活，用于：

- layer patching；
- operator patching；
- group patching；
- noise injection；
- 问题复现。

---

## 12. Token Alignment Engine

### 12.1 Teacher-forcing

三个模型共享相同 prefix：

\[
p_T(y_t\mid y_{<t}^{T})
\]

\[
p_W(y_t\mid y_{<t}^{T})
\]

\[
p_{WA}(y_t\mid y_{<t}^{T})
\]

定义 soft divergence：

\[
t_{\mathrm{soft}}
=
\min
\left\{
t:
D_{\mathrm{JS}}(p_T^t,p_{WA}^t)>\tau
\right\}
\]

定义 top-1 divergence：

\[
t_{\mathrm{top1}}
=
\min
\left\{
t:
\arg\max p_T^t
\neq
\arg\max p_{WA}^t
\right\}
\]

### 12.2 Free-running

三个模型各自生成：

\[
y_t^T,\qquad y_t^W,\qquad y_t^{WA}
\]

记录：

- 首次 token 分叉；
- 分叉后是否恢复；
- 轨迹编辑距离；
- 推理步骤差异；
- 最终答案差异；
- divergence amplification。

### 12.3 Student-prefix

使用部署模型生成的 prefix：

\[
y_{<t}^{WA}
\]

再分别送入 Teacher 和 Deploy Student：

\[
p_T(y_t\mid y_{<t}^{WA},x)
\]

\[
p_{WA}(y_t\mid y_{<t}^{WA},x)
\]

用于分析学生偏离后的恢复能力，以及生成 on-policy QAD 数据。

---

## 13. 指标体系

### 13.1 数值误差

\[
\operatorname{NMSE}(X^T,X^Q)
=
\frac{
\|X^Q-X^T\|_2^2
}{
\|X^T\|_2^2+\epsilon
}
\]

\[
\operatorname{Cos}(X^T,X^Q)
=
\frac{
\langle X^T,X^Q\rangle
}{
\|X^T\|\|X^Q\|
}
\]

\[
\operatorname{SNR}
=
10\log_{10}
\frac{
\|X^T\|_2^2
}{
\|X^Q-X^T\|_2^2+\epsilon
}
\]

数值误差只用于筛选，不直接作为推理影响结论。

### 13.2 表示几何

分别计算：

\[
\operatorname{CKA}(T,S_W)
\]

\[
\operatorname{CKA}(S_W,S_{WA})
\]

\[
\operatorname{CKA}(T,S_{WA})
\]

邻域保持率：

\[
\operatorname{Overlap@k}
=
\frac{
|N_k^T(i)\cap N_k^Q(i)|
}{
k
}
\]

状态转移：

\[
\Delta_l h_t=h_{l+1,t}-h_{l,t}
\]

\[
D_{\mathrm{transition}}
=
1-
\cos
\left(
\Delta_lh_t^T,
\Delta_lh_t^{WA}
\right)
\]

### 13.3 Logit 与决策

- KL divergence；
- JS divergence；
- top-k overlap；
- target-token rank shift；
- entropy shift；
- correct-token margin。

\[
m_t^T
=
z_{t,y^*}^T-
\max_{j\neq y^*}z_{t,j}^T
\]

\[
m_t^{WA}
=
z_{t,y^*}^{WA}-
\max_{j\neq y^*}z_{t,j}^{WA}
\]

\[
\Delta m_t=m_t^{WA}-m_t^T
\]

### 13.4 推理轨迹

记录：

- first soft divergence；
- first top-1 divergence；
- first reasoning-step error；
- reasoning-chain length；
- trajectory edit distance；
- final-answer correctness；
- error type；
- divergence amplification rate。

\[
A_t
=
\frac{
D_{\mathrm{JS}}(p_T^{t+1},p_Q^{t+1})
}{
D_{\mathrm{JS}}(p_T^t,p_Q^t)+\epsilon
}
\]

---

## 14. 因果归因引擎

### 14.1 Layer Patching

\[
h_l^{WA}\leftarrow h_l^T
\]

继续执行后续层，定义：

\[
C_l
=
D_{\mathrm{JS}}^{\mathrm{before}}
-
D_{\mathrm{JS}}^{\mathrm{after}}
\]

### 14.2 Operator Patching

支持：

```text
RMSNorm
q_proj
k_proj
v_proj
attention_score
softmax
o_proj
gate_proj
up_proj
down_proj
residual_add
```

例如：

\[
q_l^{WA}\leftarrow q_l^T
\]

### 14.3 Group Patching

\[
Q(x_g)\leftarrow x_g^{FP}
\]

定义：

\[
C_{l,t,g}
=
D_{\mathrm{JS}}^{\mathrm{original}}
-
D_{\mathrm{JS}}^{\mathrm{patched}}
\]

### 14.4 Noise Injection

\[
x_g^T
\leftarrow
x_g^T+\epsilon_g^{MX}
\]

其中：

\[
\epsilon_g^{MX}
=
Q_{\mathrm{MXINT8}}(x_g)-x_g
\]

完整证据链：

```text
修复该 group -> 输出恢复
注入该误差 -> 输出退化
```

---

## 15. 风险评分

\[
R_{l,t,g}
=
E_{l,t,g}^{\alpha}
\cdot
G_{l,t,g}^{\beta}
\cdot
C_{l,t,g}^{\gamma}
\cdot
M_t^{\delta}
\]

其中：

- \(E\)：相对量化误差；
- \(G\)：梯度或 Jacobian 敏感度；
- \(C\)：causal patching 恢复量；
- \(M\)：token margin 风险。

\[
G_{l,t,g}
=
\left\|
\frac{
\partial D_{\mathrm{KL}}(p_T,p_{WA})
}{
\partial x_{l,t,g}
}
\right\|
\]

\[
M_t
=
\frac{1}{|m_t^{WA}|+\epsilon}
\]

MVP 阶段建议使用归一化后的线性组合：

\[
R
=
w_E\tilde E
+
w_G\tilde G
+
w_C\tilde C
+
w_M\tilde M
\]

---

## 16. 推荐接口

### 16.1 MXINT8 Quantizer

```python
from dataclasses import dataclass
import torch


@dataclass
class MXQuantResult:
    dequantized: torch.Tensor
    quantized: torch.Tensor
    scale: torch.Tensor
    exponent: torch.Tensor | None
    stats: list[MXBlockStats]


class MXINT8Quantizer:
    def __init__(
        self,
        group_size: int = 64,
        round_mode: str = "nearest",
        saturation_mode: str = "clip",
    ) -> None:
        self.group_size = group_size
        self.round_mode = round_mode
        self.saturation_mode = saturation_mode

    def quantize(self, x: torch.Tensor) -> MXQuantResult:
        raise NotImplementedError
```

要求：

- 量化规则与硬件一致；
- group 划分与真实 NPU 一致；
- 支持记录 scale/exponent；
- 支持 fake quant；
- 支持 patch 指定 group；
- 支持注入真实误差。

### 16.2 Probe Hook

```python
class ProbeHook:
    def before_op(
        self,
        key: TraceKey,
        inputs: tuple,
    ) -> tuple:
        return inputs

    def after_op(
        self,
        key: TraceKey,
        inputs: tuple,
        output,
    ):
        return output
```

### 16.3 Recorder

```python
class TraceRecorder:
    def record_tensor_stats(
        self,
        key: TraceKey,
        tensor,
    ) -> None:
        ...

    def record_mx_stats(
        self,
        key: TraceKey,
        stats: list[MXBlockStats],
    ) -> None:
        ...

    def record_token_stats(
        self,
        sample_id: str,
        stats: TokenStats,
    ) -> None:
        ...

    def flush(self) -> None:
        ...
```

### 16.4 Patching API

```python
@dataclass
class PatchSpec:
    sample_id: str
    token_id: int
    layer_id: int
    operator_name: str | None = None
    group_id: int | None = None
    source_variant: ModelVariant = ModelVariant.FP16


class PatchRunner:
    def run(
        self,
        prompt: str,
        patch: PatchSpec,
    ) -> dict:
        ...
```

---

## 17. PyTorch / GraphModule 实现建议

第一版建议基于：

- PyTorch forward hooks；
- `torch.fx.GraphModule`；
- 自定义 `MXINT8FakeQuant`；
- operator wrapper；
- patch callback；
- Parquet/Zarr；
- Matplotlib。

FX 图替换结构：

```text
Linear
  |
  v
MXINT8Quantizer
  |
  v
QuantizedLinearBackend
  |
  v
Recorder
```

目标节点：

```text
call_module:
    nn.Linear
    RMSNorm
    Attention module

call_function:
    torch.matmul
    torch.add
    torch.softmax
    torch.nn.functional.silu

call_method:
    add
    matmul
```

节点必须具有稳定唯一名称，例如：

```text
model.layers.17.self_attn.q_proj
model.layers.17.self_attn.k_proj
model.layers.17.self_attn.v_proj
model.layers.17.self_attn.o_proj
model.layers.17.mlp.gate_proj
model.layers.17.mlp.up_proj
model.layers.17.mlp.down_proj
model.layers.17.residual_add
```

---

## 18. 与 vLLM 的集成

### Offline Analyzer

基于 Transformers 或自定义 GraphModule：

- 单 batch；
- deterministic；
- teacher-forcing；
- 完整 hook；
- group patching；
- 完整重放；
- root-cause analysis。

### vLLM Validation Mode

用于验证：

- PagedAttention；
- KV cache；
- decode；
- scheduler；
- sampling；
- tensor parallel；
- pipeline parallel；
- 实际部署输出。

分工：

```text
GraphModule:
    深入归因
    layer/operator/group patching
    几何分析

vLLM:
    系统级复现
    KV cache
    decode
    sampling
    并行执行
```

---

## 19. 存储设计

建议：

- 元数据：SQLite 或 DuckDB；
- 大规模指标：Parquet；
- 选择性张量：Zarr；
- 报告：Markdown + JSON；
- 可视化：PNG 或 HTML。

```text
runs/
└── run_20260717_001/
    ├── config.yaml
    ├── summary.json
    ├── token_stats.parquet
    ├── layer_stats.parquet
    ├── mx_block_stats.parquet
    ├── causal_scores.parquet
    ├── tensors.zarr
    ├── report.md
    └── figures/
        ├── token_trajectory.png
        ├── layer_token_heatmap.png
        ├── layer_group_heatmap.png
        └── margin_curve.png
```

### 19.1 存储控制策略

Trace 数据容易随样本数、序列长度、层数和 group 数爆炸。默认策略应为：

- Level 0 只保存统计量和 top-k logits，不保存完整 tensor；
- Level 1 只保存 divergence 附近 token window，例如 `[t-4, t+4]`；
- Level 2 只允许单样本或少量样本开启，并要求显式配置；
- 大 tensor 必须按 sample / token / layer 分块写入；
- 报告中记录实际落盘大小和 tensor 数量。

推荐在 recorder 中设置硬限制：

```yaml
storage_limits:
    max_tensor_bytes_per_run: 20000000000
    max_level2_samples: 4
    token_window_before_divergence: 4
    token_window_after_divergence: 4
```

---

## 20. 报告样例

### Run Summary

```text
Model: Qwen-xx
Weight format: INT5 / SSFP
Activation format: MXINT8
Activation group size: 64
Dataset: Math / Code / General
Samples: 1000

Accuracy:
FP16: 78.4%
WQ+A16: 76.9%
WQ+MXINT8: 69.8%

Estimated contribution:
Weight quantization: -1.5%
MXINT8 activation: -7.1%
```

### Token Divergence

```text
Sample: math_00182

First soft divergence:
Token: 74
JS divergence: 0.083

First top-1 divergence:
Token: 81
FP16 token: "12"
WQ+A16 token: "12"
WQ+MXINT8 token: "14"

Correct-token margin:
FP16: +1.83
WQ+A16: +1.52
WQ+MXINT8: -0.21
```

### Root Cause

```text
Root cause candidate #1

Layer: 17
Operator: q_proj
Token: 79
Group: 1432

MX exponent: -3
Max/RMS: 8.7
Relative quantization error: 4.3%
Gradient sensitivity: top 0.2%

Patching recovery:
JS divergence: 0.091 -> 0.018
Correct-token margin: -0.21 -> +1.16

Conclusion:
A single activation outlier raised the shared exponent,
reducing effective precision for the remaining 63 values.
```

---

## 21. Hard-Example Mining

```python
@dataclass
class HardExample:
    sample_id: str
    prompt: str
    teacher_response: str
    student_response: str

    first_soft_divergence_token: int
    first_top1_divergence_token: int

    sensitive_layers: list[int]
    sensitive_operators: list[str]
    sensitive_groups: list[int]

    teacher_logits_path: str | None
    teacher_hidden_path: str | None

    risk_score: float
    error_type: str
```

选择规则：

```text
JS divergence > threshold
OR
correct-token margin sign changed
OR
top-1 token mismatch
OR
CKA drop > threshold
OR
causal patch recovery > threshold
```

---

## 22. MVP 路线

MVP 应以“先证明能定位，再证明能归因，最后进入训练闭环”为顺序推进。

### MVP 1：三模型对比

实现：

- FP16；
- WQ+A16；
- WQ+MXINT8；
- teacher-forcing；
- layer output hook；
- logits KL/JS；
- first-divergence token；
- MX block statistics；
- Markdown 报告。

验收：

- 单样本三路径对比；
- 首次 soft divergence；
- 首次 top-1 divergence；
- 层级误差；
- 高风险 MXINT8 group。

推荐实现顺序：

1. 跑通 FP16 与 WQ+MXINT8 两路径 teacher-forcing。
2. 加入 WQ+A16 路径，拆分权重量化误差和激活量化误差。
3. 只记录 layer output 与 logits，验证 token 对齐和 divergence 计算。
4. 接入 MXINT8 group stats，输出 layer-token-group 风险排序。
5. 生成 Markdown 报告，并保留可复现实验配置。

### MVP 2：因果 Patching

实现：

- layer patch；
- operator patch；
- group patch；
- causal recovery score；
- noise injection。

验收：

- 定位关键层；
- 定位关键算子；
- 判断 group 修复是否恢复 margin；
- 在 FP 路径注入误差后复现退化。

进入 MVP 2 的前置条件：

- MVP 1 能稳定复现同一个样本的 first divergence；
- 三路径 layer output shape 完全一致；
- TraceKey 能唯一定位到 token、layer、operator 和 group；
- Level 2 replay 能在单样本上复现原始 deploy logits。

### MVP 3：Reasoning Manifold

实现：

- CKA；
- neighborhood overlap；
- covariance spectrum；
- transition similarity；
- reasoning-step segmentation；
- margin trajectory。

### MVP 4：训练闭环

实现：

- hard-example mining；
- QAT/QAD 数据导出；
- 训练前后自动回归；
- 风险 group 收敛追踪。

---

## 23. 配置文件示例

```yaml
model:
  teacher_path: /models/fp16
  weight_quantized_path: /models/wq
  architecture: qwen

quantization:
  weight_format: int5
  activation_format: mxint8
  activation_group_size: 64
  round_mode: nearest
  saturation_mode: clip
  accumulator_format: fp24
  output_format: fp16

execution:
  mode: teacher_forcing
  batch_size: 1
  max_sequence_length: 4096
  deterministic: true
  seed: 1234

recording:
  level: 0
  operators:
    - rmsnorm
    - q_proj
    - k_proj
    - v_proj
    - o_proj
    - gate_proj
    - up_proj
    - down_proj
    - residual_add

metrics:
  nmse: true
  cosine: true
  cka: true
  neighborhood_overlap: false
  covariance_spectrum: false
  token_kl: true
  token_js: true
  token_margin: true

attribution:
  layer_patching: false
  operator_patching: false
  group_patching: false
  noise_injection: false

thresholds:
  soft_divergence_js: 0.05
  cka_drop: 0.02
  high_group_nmse: 0.03
  causal_recovery: 0.02

storage:
  output_dir: runs/
  tensor_backend: zarr
  table_backend: parquet
```

---

## 24. 测试要求

### Quantizer

必须验证：

- group 划分；
- scale/exponent；
- round；
- saturation；
- dequant；
- 边界值；
- NaN/Inf；
- 与硬件模拟器逐元素一致。

### Recorder

验证：

- TraceKey 唯一；
- model variant 不混淆；
- token/layer/operator/group 对齐；
- Level 0 不保存完整张量；
- Level 2 可以完整重放。

### Patching

验证：

- layer patch 输出正确；
- operator patch 只影响目标算子；
- group patch 只替换目标 64 元素；
- noise injection 使用真实量化误差；
- patch 前后指标计算正确。

### End-to-End

对小型 Transformer：

1. 运行 FP16；
2. 运行 WQ+A16；
3. 运行 WQ+MXINT8；
4. 生成 token divergence；
5. 定位高风险层；
6. 执行 layer patch；
7. 验证 logits 恢复。

### 24.1 回归测试样本

建议固定三类测试输入：

- 短 prompt：验证基础 trace、logits 对齐和报告生成；
- 长 prompt：验证 position id、RoPE、cache 和 chunk 行为；
- 人工构造 outlier prompt：验证 MXINT8 group saturation、max/RMS 和 patching 归因。

每类样本应保存 golden summary，而不是保存完整 tensor。golden summary 至少包含：

- first soft divergence token；
- first top-1 divergence token；
- top-3 risk layer/operator/group；
- patch 前后 JS divergence；
- patch 前后 correct-token margin。

---

## 25. 风险与约束

### 25.1 硬件一致性风险

MXINT8 fake quant 如果与 NPU 规则不一致，后续归因可能全部失真。必须优先确认：

- group 维度和 group 边界；
- scale 或 exponent 的计算方式；
- rounding mode；
- saturation / clipping 规则；
- accumulator 格式；
- dequant 和输出 cast 规则。

硬件规则未确认时，报告只能标记为 simulation-level analysis，不能作为 hardware root cause。

### 25.2 Hook 和 Graph 改写扰动

hook、FX rewrite 和 fake quant wrapper 可能改变执行顺序或 dtype。每次引入新的 hook 层级后，都应先验证：

```text
无量化 + recorder 开启
    vs
无量化 + recorder 关闭
```

两者 logits 应在容许阈值内一致。

### 25.3 指标误判风险

高 NMSE 不一定导致 token 错误，低 NMSE 也可能翻转 margin。报告中应避免只根据单一指标下结论，至少同时给出：

- token-level JS / margin；
- layer/operator 数值误差；
- group-level MXINT8 stats；
- patching recovery score。

只有 patching 或 noise injection 通过后，才能标记为 causal root cause。

### 25.4 vLLM 集成边界

vLLM validation mode 用于复现系统级输出，不作为第一版深度归因主路径。涉及 PagedAttention、KV cache、scheduler 或并行策略的问题，应先在 GraphModule 离线路径中验证数值根因，再进入 vLLM 对齐。

---

## 26. 关键设计原则

### 保留三条模型路径

没有：

\[
T,\qquad S_W,\qquad S_{WA}
\]

就无法区分权重量化、激活量化和耦合误差。

### 必须实现因果 Patching

低价值输出：

```text
Layer 17 NMSE = 0.038
```

高价值输出：

```text
修复 Layer 17 q_proj group 1432 后，
正确 token margin 从 -0.21 恢复到 +1.16。
```

### 围绕 Token Decision 分析

最终报告应形成：

```text
Layer 17
q_proj
group 1432
MX exponent = -3
max/RMS = 8.7
relative quantization error = 4.3%

该 group 使正确 token "12" 的 margin
从 +1.83 降至 -0.21，
最终输出 token 变为 "14"。
```

### 分析最终部署数值路径

最终结论必须来自：

```text
WQ + MXINT8 activation
```

并且量化规则必须与真实 NPU 一致。

---

## 27. 最终验收标准

系统应支持：

- FP16、WQ+A16、WQ+MXINT8 三路径同步执行；
- teacher-forcing token 对齐；
- soft divergence 和 top-1 divergence；
- 层、算子、group 级 MXINT8 统计；
- CKA 和 transition similarity；
- layer/operator/group patching；
- noise injection；
- causal risk score；
- Markdown 根因报告；
- hard-example 导出；
- 从错误 token 回溯到具体 layer、operator 和 group。

### 27.1 MVP 验收口径

MVP 1 完成时，应能对单个样本输出：

```text
sample_id
first_soft_divergence_token
first_top1_divergence_token
top risk layers
top risk operators
top risk MXINT8 groups
FP16 / WQ+A16 / WQ+MXINT8 token margin
```

MVP 2 完成时，应能证明至少一个候选根因满足：

```text
patch 目标 group 后，JS divergence 明显下降
patch 目标 group 后，correct-token margin 恢复或接近恢复
向 FP 路径注入同类误差后，输出向 deploy 路径退化
```

MVP 3 / MVP 4 完成时，应能将高风险样本导出到训练数据，并在训练或校正后复跑同一报告，比较风险 score、margin 和最终准确率变化。

最终系统应能回答：

> 哪个 MXINT8 block 的什么数值特征，通过哪一层和哪一个算子，破坏了哪个内部表示方向，并最终使哪个 token 的决策 margin 翻转？
