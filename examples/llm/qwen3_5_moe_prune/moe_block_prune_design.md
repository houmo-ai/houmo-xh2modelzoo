# MoeBlock 动态剪枝设计说明

> 剪枝路由和动态专家执行已合并进标准 `MoeBlock`。本文档介绍剪枝属性、内部计算逻辑以及与普通 MoE 模式的差异。

---

## 1. 背景与动机

标准 `MoeBlock` 的 top-k 专家选择是**编译时固定**的：`k` 在构造时指定（`self.k`），每次推理都选择相同数量的专家。

在剪枝（pruning）场景下，不同 token 保留的专家数可能不同。`MoeBlock` 通过可选的剪枝配置同时完成路由筛选和专家执行：

| 模块 | 职责 |
|---|---|
| **`s_scalar`** | buffer，各专家的重要性标量 |
| **`prune_threshold`** | 浮点算子属性；非 `None` 时启用剪枝，不作为网络 tensor 输入 |

---

## 2. MoeBlock 动态剪枝路由

### 2.1 forward 签名

```python
def forward(
    self,
    hidden_states: Tensor,
    routing_weights: Tensor,
    selected_experts: Tensor | None = None,
) -> Tensor:
```

`prune_threshold` 是 `MoeBlock` 算子的浮点属性，在构造/导出时固化，
不作为网络 tensor 输入。

### 2.2 内部计算流程

```
输入: routing_weights [B, S, E], s_scalar [E]
属性: prune_threshold
                    │
                    ▼
          ┌─────────────────────┐
          │  Step 1: top-k 选择  │
          │  torch.topk(..., k=  │
          │    self.top_k)       │
          └─────────┬───────────┘
                    │ topk_weights [B,S,K], selected_experts [B,S,K]
                    ▼
          ┌─────────────────────┐
          │ Step 2: 计算重要性   │
          │ score = weight *    │
          │   s_scalar[expert]  │
          │ score_norm = score /│
          │   sum(score)         │
          └─────────┬───────────┘
                    │ score [B,S,K]
                    ▼
          ┌─────────────────────┐
          │ Step 3: 阈值筛选     │
          │ keep = score_norm >=│
          │ prune_threshold    │
          │ 强制保留下标最大的   │
          │ 那个 slot           │
          └─────────┬───────────┘
                    │ keep_mask [B,S,K] (bool)
                    ▼
          ┌─────────────────────┐
          │ Step 4: 计算动态 k   │
          │ dynamic_k =         │
          │ max(sum(keep,dim=-1))│
          │   .clamp(min=1)     │
          └─────────┬───────────┘
                    │ dynamic_k (标量 int32)
                    ▼
          ┌─────────────────────┐
          │ Step 5: 重排 + 掩码  │
          │ 被剪掉的 slot 权重   │
          │ 置为 0，保持 shape   │
          │ 仍为 [B,S,top_k]    │
          └─────────┬───────────┘
                    │
内部结果: topk_weights [B,S,K'], selected_experts [B,S,K'], dynamic_k
```

**关键算法**：标准的 **Method1** 剪枝算法——先做 top-k 选候选人，再用 `importance = gate_score × s_scalar` 评估每个候选人的"真正重要性"，低于阈值的剪掉，但保证每个 token 至少保留一个专家（importance 最大的 slot 强制保留）。

### 2.3 与标准 MoeBlock 路由的区别

| 方面 | 标准 `MoeBlock` | `PruningRouter` |
|---|---|---|
| 专家选择 | 内嵌在 `MoeBlock.forward()` 中 | 独立模块，输出给下游 |
| `k` 值 | 固定 `self.k` | 内部动态 `dynamic_k`（随 token 和推理步变化） |
| 剪枝能力 | 无 | 有 — 用 score + threshold 做二次筛选 |
| 输出 shape | `[B,S,D]` | `[B,S,D]`，动态路由结果在算子内部消费 |

---

## 3. 动态 k 的 MoE 执行

### 3.1 forward 签名

```python
def forward(
    self,
    hidden_states: torch.Tensor,        # [B, S, D]
    routing_weights: torch.Tensor,      # [B, S, E] — 原始路由权重
    selected_experts: Tensor | None = None,
    fast_mode: bool = True,
) -> torch.Tensor:
```

### 3.2 与标准 MoeBlock 的核心区别

| 方面 | `MoeBlock` | `MoeBlockPrune` |
|---|---|---|
| **`k` 存储** | `self.k` — 构造时固定 | `self.k` — 仅作为 fallback/预分配参考 |
| **实际 `k` 值** | `self.k` | 由内部剪枝结果计算 `dynamic_k` |
| **forward 参数** | 标准输入 | 标准输入；阈值是属性，不增加 tensor 输入 |
| **topk_outside 截断** | 不截断（直接用 `self.k`） | `routing_weights[:, :dynamic_k]`, `selected_experts[:, :dynamic_k]` |
| **非 topk_outside 模式** | `torch.topk(..., k=self.k)` | `torch.topk(..., k=dynamic_k)` |
| **输出 shape** | `[B, S, D]` | `[B, S, D]`（相同） |

### 3.3 内部计算逻辑

```
输入: hidden_states [B,S,D], routing_weights [B,S,K'], 
     selected_experts [B,S,K'], k_tensor
                    │
                    ▼
         ┌──────────────────────┐
         │ 从 k_tensor 提取      │
         │ dynamic_k =           │
         │ int(k_tensor.item())  │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │ 是否 topk_outside?    │
         │ Yes → 上游已选好专家   │
         │   1. 按 selected_    │
         │      experts 对齐    │
         │      routing_weights  │
         │   2. 截断到          │
         │      dynamic_k 列     │
         │                      │
         │ No  → 自己做 topk:    │
         │   torch.topk(...,    │
         │     k=dynamic_k)     │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │ 权重归一化             │
         │ (可选)               │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │ 按专家索引分组排序      │
         │ _get_routing_indices() │
         └──────────┬───────────┘
                    │ token_counts_by_expert, gather_indices
                    ▼
    ╔═══════════════════════════╗
    ║        MLP 计算            ║
    ║                           ║
    ║  ┌─────────────────────┐  ║
    ║  │ Gate Proj:           │  ║
    ║  │ X @ W_gate + b_gate  │  ║
    ║  └─────────┬───────────┘  ║
    ║            ▼              ║
    ║  ┌─────────────────────┐  ║
    ║  │ Activation (SiLU/    │  ║
    ║  │  Gelu/Glu/ReLU²)    │  ║
    ║  └─────────┬───────────┘  ║
    ║            ▼              ║
    ║  ┌─────────────────────┐  ║
    ║  │ Up Proj:             │  ║
    ║  │ X @ W_up + b_up      │  ║
    ║  │ (或 constant_one     │  ║
    ║  │  跳过)               │  ║
    ║  └─────────┬───────────┘  ║
    ║            ▼              ║
    ║  ┌─────────────────────┐  ║
    ║  │ Down Proj:           │  ║
    ║  │ mul_state @ W_down   │  ║
    ║  └─────────┬───────────┘  ║
    ╚═════════════╤═════════════╝
                  ▼
         ┌──────────────────────┐
         │ Routing weight 加权   │
         │ out *= rw[:, None]   │
         └──────────┬───────────┘
                    ▼
         ┌──────────────────────┐
         │ Scatter 回原始 token   │
         │ 顺序 + sum over k     │
         └──────────┬───────────┘
                    ▼
        输出: final_hidden_states [B,S,D]
```

### 3.4 两条计算路径

**路径 A — Fast Mode (Triton grouped GEMM)**：
- 将 token 按专家分组排序，通过 `xh2_grouped_gemm_fp` 单次 kernel 调用完成同一专家的所有 token 计算
- 大幅减少离散的 `F.linear` 调用次数
- 是 GPU 推理的主要路径

**路径 B — Fallback (Python loop)**：
- 按 `dynamic_k` 和专家索引逐层循环
- 每个专家独立调用 `F.linear`
- 用于 CPU 或非 Triton 环境

### 3.5 与标准 MoeBlock 的实现差异对比

```
MoeBlock (fast path)                          MoeBlockPrune (fast path)
──────────────────────────────                ──────────────────────────────
# k 来自 self.k                               # k 来自 k_tensor
torch.topk(..., k=self.k)                     dynamic_k = int(k_tensor.item())
                                              ...
                                              # topk_outside 时多一步截断
selected_experts = selected_experts           selected_experts = selected_experts[:, :dynamic_k]
                                              routing_weights = routing_weights[:, :dynamic_k]

# Grouped GEMM 参数用 self.k                  # Grouped GEMM 参数用 dynamic_k
xh2_grouped_gemm_fp(..., topk=self.k, ...)    xh2_grouped_gemm_fp(..., ...)  # 无 topk 参数

# 输出 reshape 用 self.k                      # 输出 reshape 用 dynamic_k
out.view(..., self.k, hidden_dim)             out.view(..., dynamic_k, hidden_dim)
out.sum(dim=2)                                out.sum(dim=2)
```

---

## 4. 三层流水线数据流

整个 Prune MoE 的计算图由三个阶段的模块串联而成：

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  PruningRouter   │───▶│  MoeBlockPrune    │───▶│  最终输出        │
│                  │    │  (topk_outside    │    │                  │
│  (float / 量化)   │    │   = True)         │    │                  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                       │                       │
        │ routing_weights [B,S,K]                       │
        │ selected_experts [B,S,K]                      │
        │ k_tensor (dynamic_k)   │                       │
        └───────────────────────┘                       │
                                                        │
```

| 阶段 | 实际使用的类 | 量化包装类 | backend IR |
|---|---|---|---|
| 剪枝路由器 | `PruningRouter` | `QPruningRouter` | `xh2a::PruningRouter` |
| 执行模块 | `MoeBlockPrune` | `QMoeBlockPrune` | `xh2a::MoeBlockPrune` |

注意：`MoeBlockPrune` 的 `topk_outside=False` 模式也可独立使用（自行做 topk），但剪枝场景必须用 `topk_outside=True`。

---

## 5. 量化前向路径差异

`QMoeBlockPrune` 继承自 `QMoeBlock`，两者的量化 `_setup` 完全相同。`forward` 的差异仅在于 `k` 的传入方式：

| 路径 | `QMoeBlock` | `QMoeBlockPrune` |
|---|---|---|
| compile 路径 | `torch_ops_xh2a_moeblock(x, ..., k=self.k)` | `torch_ops_xh2a_moeblock_prune(x, ..., k=k_tensor)` |
| FAST 路径 | `moeblock_xh2a_fast(x, ..., k=self.k)` | `moeblock_prune_xh2a_fast(x, ..., k=k_tensor)` |
| fallback | `xhnn.MoeBlock.forward(self, x, rw, sel)` | `xhnn.MoeBlockPrune.forward(self, x, rw, k_tensor, sel)` |

`torch_ops_xh2a_moeblock_prune` 的 backend IR 内部直接将 `k_tensor` 转为 `k_val = int(k.item())`，然后调用 `moeblock_xh2a_default(..., k=k_val, topk_outside=...)`，**不改变后端计算语义**，只是替换了 `k` 的来源方式。

---

## 6. 总结

| 关注点 | `MoeBlock` | `PruningRouter` | `MoeBlockPrune` |
|---|---|---|---|
| **定位** | 端到端 MoE block | 上游剪枝决策模块 | MoE 执行模块（动态 k） |
| **k 值来源** | 编译时固定 | 运行时动态计算 | 运行时外部传入 |
| **专家选择** | 自己做 topk | 做 topk + 阈值剪枝 | 不在自己内部做 topk |
| **依赖 `selected_experts`** | 可选（`topk_outside`） | 输出 | 必须（`topk_outside=True`） |
| **后端 IR** | `xh2a::MoeBlock` | `xh2a::PruningRouter` | `xh2a::MoeBlockPrune` |
| **量化父类** | `QBaseModule` | `QBaseModule` | `QMoeBlock` |
