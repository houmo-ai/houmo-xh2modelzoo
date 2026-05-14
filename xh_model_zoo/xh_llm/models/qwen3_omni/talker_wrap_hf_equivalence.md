# HF 流程 vs Wrap 流程:数学等价性证明

本文针对 Qwen3-Omni talker 在量化导出路径上做的"projection 融合 + 算术 mask"
重构,从数学和有限精度算术两个层面证明:wrap 路径与 HF 浮点路径在 talker
trunk 输入边界处 **bit-exact 相等**(fp16 生产 dtype)。

相关代码:

- HF 浮点参考:`modeling_qwen3_omni_moe.py`
  - `Qwen3OmniMoeForConditionalGeneration._get_talker_user_parts`
  - `Qwen3OmniMoeForConditionalGeneration._get_talker_assistant_parts`
- Wrap 实现:`_talker_model.py:301-331`
  - `_Qwen3OmniMoeTalkerForConditionalGeneration.forward`
- LLMBaseModel 适配层:`qwen3_omni_moe_talker_model.py`
- 单元测试:`tests/qwen3_omni_talker_projection_equivalence_test.py`

---

## 一、用数学符号刻画两个流程

设:

- $H: \mathbb{R}^{d_t} \to \mathbb{R}^{d_v}$ — `hidden_projection`(MLP)
- $T: \mathbb{R}^{d_t} \to \mathbb{R}^{d_v}$ — `text_projection`(MLP)
- $E_c: \mathbb{Z} \to \mathbb{R}^{d_v}$ — `talker.embed`(codec embedding lookup)

输入侧给定:

- $\text{thinker\_embed} \in \mathbb{R}^{1 \times L \times d_t}$
- $\text{thinker\_hidden} \in \mathbb{R}^{1 \times L \times d_t}$
- $\text{mm} \in \{0,1\}^{1 \times L}$(multimodal_mask)
- 几个常数 embedding:$\text{tts\_pad}, \text{tts\_bos} \in \mathbb{R}^{d_v}$

两个流程都要在 $L$ 个位置上各产生一个 $\mathbb{R}^{d_v}$ 向量,送进 talker
trunk。**等价性命题就是:这 $L$ 个 $d_v$ 维向量在两种流程下完全相等。**

---

## 二、HF 流程:控制流分段构造

HF 把 chatml 切成 segment,逐段算输出向量 $y_i$。

### user 段 ($i \in [\text{start}, \text{end})$)

$$
y_i^{\text{HF}} = \begin{cases}
H(\text{thinker\_hidden}_i) & \text{if } \text{mm}_i = 1 \\
T(\text{thinker\_embed}_i) & \text{if } \text{mm}_i = 0
\end{cases}
$$

### assistant 段(开头 4 个 thinker token,扩展为 9 个输出位)

记 $a_j = T(\text{thinker\_embed}_{\text{start}+j})$, $j \in [0,4)$。

$$
y_j^{\text{HF}} = \begin{cases}
a_j & j \in \{0,1,2\} \\
\text{tts\_pad} + E_c(\text{codec\_nothink}) & j = 3 \\
\text{tts\_pad} + E_c(\text{codec\_think\_bos}) & j = 4 \\
\text{tts\_pad} + E_c(\text{codec\_think\_eos}) & j = 5 \\
\text{tts\_pad} + E_c(\text{speaker\_id}) & j = 6 \\
\text{tts\_bos} + E_c(\text{codec\_pad}) & j = 7 \\
a_3 + E_c(\text{codec\_bos}) & j = 8
\end{cases}
$$

### decode 步

$$
y^{\text{HF}} = E_c(\text{prev\_codec\_token})
$$

注意:$H, T, E_c$ 这三个调用在 HF 里是**用 Python 控制流 + tensor
indexing** 选出来的(`if user_mm_mask.any():`、`out[mm] = H(...)`、
`cat([...])`)。

---

## 三、Wrap 流程:统一算术混合

wrap 在每个位置 $i$ 都执行**同一个公式**:

$$
\boxed{\,y_i^{\text{wrap}} = \big[\,H(s_i)\cdot(1-r_i) + T(s_i)\cdot r_i\,\big]\cdot(1-m_i) + b_i \cdot m_i\,}
$$

其中 $s_i \in \mathbb{R}^{d_t}$,$r_i, m_i \in \{0,1\}$,
$b_i \in \mathbb{R}^{d_v}$ 是**外部喂进来的四个张量**:`source` / `role_mask` /
`bypass_embeds` / `bypass_mask`。

注意 $r,m \in \{0,1\}$ 时这个公式是个**两级 2-1 多路选择器(MUX)**:

| $r_i$ | $m_i$ | $y_i^{\text{wrap}}$ |
| ----- | ----- | ------------------- |
| 0     | 0     | $H(s_i)$            |
| 1     | 0     | $T(s_i)$            |
| 0     | 1     | $b_i$               |
| 1     | 1     | $b_i$               |

---

## 四、等价性证明:对每种 HF 情形构造对应的 $(s, r, b, m)$

**这是一个存在性证明**:对每个位置 $i$,我都能在调度方构造一组
$(s_i, r_i, b_i, m_i)$,使 $y_i^{\text{wrap}} = y_i^{\text{HF}}$。

### 4.1 user 段

| HF 情形                              | 设 $s_i$                  | $r_i$ | $b_i$ | $m_i$ | wrap 输出(查 MUX 表)                          |
| ------------------------------------ | ------------------------- | ----- | ----- | ----- | ----------------------------------------------- |
| mm=1, $H(\text{thinker\_hidden}_i)$  | $\text{thinker\_hidden}_i$ | 0     | 任意  | 0     | $H(s_i) = H(\text{thinker\_hidden}_i)$ ✓        |
| mm=0, $T(\text{thinker\_embed}_i)$   | $\text{thinker\_embed}_i$  | 1     | 任意  | 0     | $T(s_i) = T(\text{thinker\_embed}_i)$ ✓         |

构造规则可以写成一个表达式:

$$
s_i = \text{mm}_i \cdot \text{thinker\_hidden}_i + (1-\text{mm}_i) \cdot \text{thinker\_embed}_i,\quad
r_i = 1 - \text{mm}_i,\quad b_i = 0,\quad m_i = 0
$$

### 4.2 assistant 段(9 个位置)

| 位置 $j$ | HF 输出                                              | 设 $s_j$                              | $r_j$ | $b_j$                                                | $m_j$ | wrap        |
| -------- | ---------------------------------------------------- | ------------------------------------- | ----- | ---------------------------------------------------- | ----- | ----------- |
| 0,1,2    | $a_j = T(\text{thinker\_embed}_{\text{start}+j})$    | $\text{thinker\_embed}_{\text{start}+j}$ | 1     | 0                                                    | 0     | $T(s_j) = a_j$ ✓ |
| 3        | $\text{tts\_pad} + E_c(\text{codec\_nothink})$       | 0                                     | 任意  | $\text{tts\_pad} + E_c(\text{codec\_nothink})$       | 1     | $b_j$ ✓     |
| 4        | $\text{tts\_pad} + E_c(\text{codec\_think\_bos})$    | 0                                     | 任意  | $\text{tts\_pad} + E_c(\text{codec\_think\_bos})$    | 1     | $b_j$ ✓     |
| 5,6,7    | (同上模式)                                          | 0                                     | 任意  | 对应 $\text{tts\_*} + E_c(\cdot)$                     | 1     | $b_j$ ✓     |
| 8        | $a_3 + E_c(\text{codec\_bos})$                       | 0                                     | 任意  | $a_3 + E_c(\text{codec\_bos})$                       | 1     | $b_j$ ✓     |

注意位置 8:HF 把 $a_3 = T(\text{thinker\_embed}_{\text{start}+3})$ 也算在
host 侧 — 因为它要和 $E_c(\text{codec\_bos})$ 相加,这种"投影 + codec embed
相加"的混合操作没法表达进 wrap 图($E_c$ 是另一套 lookup)。所以 host 直接
把 $a_3 + E_c(\text{codec\_bos})$ 整体打包进 $b_j$,wrap 走 bypass。

### 4.3 decode 步

| HF 输出                          | $s$ | $r$  | $b$                                | $m$ | wrap   |
| -------------------------------- | --- | ---- | ---------------------------------- | --- | ------ |
| $E_c(\text{prev\_codec\_token})$ | 0   | 任意 | $E_c(\text{prev\_codec\_token})$   | 1   | $b$ ✓  |

至此,**每种 HF 情形都能用 wrap 公式严格复现**。$\square$

---

## 五、为什么 fp16 下是 bit-exact

光"数学相等"不够,还要看**有限精度算术下是不是真的相等**。fp16(IEEE 754)
下有几条**无舍入**的恒等式:

1. $x \cdot 1.0 = x$ — 与 1 相乘是 identity
2. $x \cdot 0.0 = 0.0$(对所有有限 $x$)— 与 0 相乘必为 0,且没有 rounding
3. $x + 0.0 = x$ — 与 0 相加是 identity
4. $1.0 - 0.0 = 1.0$,$1.0 - 1.0 = 0.0$ — 这两个减法 representable 且精确

由于 $r_i, m_i$ 严格取自 $\{0.0, 1.0\}$:

- $1 - r_i$ 总是精确得 $\{1, 0\}$(规则 4)
- $H(s)\cdot(1-r) + T(s)\cdot r$ 在 $r=0$ 时
  $= H(s)\cdot 1 + T(s)\cdot 0 = H(s) + 0 = H(s)$(规则 1+2+3),完全没引入新
  的舍入误差;$r=1$ 类似
- 第二级 bypass mix 同理

所以 **fp16 下每位 $y_i^{\text{wrap}}$ 与 $y_i^{\text{HF}}$ bit-exact 相等**
(`torch.equal()` 通过)。这就是测试
`test_real_wrap_forward_bit_exact_in_fp16` 在证明的事。

---

## 六、那"区别"到底在哪?

数学上没区别。区别只在**计算的物理形态**:

| 维度                  | HF 流程                                  | Wrap 流程                                                     |
| --------------------- | ---------------------------------------- | ------------------------------------------------------------- |
| projection 的位置     | 在 `generate()` 调度器里(host 端 Python) | 在 talker forward 图内部(可被 trace)                          |
| 选择机制              | Python `if` + tensor indexing            | 算术 MUX:$y = a\cdot(1-r) + c\cdot r$                         |
| 张量组织              | 每段一个张量,最后 `cat` 拼接            | 一个张量 + 几个 mask,统一前向                                |
| 两个 projection MLP   | 只在被选中时运行一次                     | **两个都跑**,然后用 mask 把不要的那路 $\times 0$ 抹掉          |
| 量化                  | 不可校准(权重不在图里)                  | 可校准(权重进图,activation 可统计)                          |
| ONNX export           | 不可 trace(控制流)                      | 可 trace(纯算术)                                              |
| prefill / decode 图   | 走 `if` 分支决定走哪条路                  | **同一张图**,通过 `bypass_mask` 切换两种用法                  |

代价是 wrap **多算了一倍 projection FLOPs**(两路都跑),但换来了:量化校
准 / w8a8 / ONNX 导出 / 单图复用 prefill+decode 这一系列工程能力。HF 的
"省一次乘法"只在 generate 调度器里做得到 — 进了图就只能算术化。

---

## 七、用一句话概括

> **HF 的 `if-else + indexing + cat` 与 wrap 的
> $(H(s)(1-r) + T(s)r)(1-m) + bm$ 在 $r,m \in \{0,1\}$ 下定义同一个函数;
> 且因为这些 mask 在 fp16 下是精确的 0/1,这种"数学相等"在生产 dtype 下
> 落地为 bit-exact 相等。**

剩下的工作 — 把 host 上的 segment-wise 构造规则(user mm/text 拆分、
assistant 9 位置打包、decode codec embed)翻译成 $(s, r, b, m)$ 四个张
量 — 已经在 export 脚本的 capture hook 里完成,本次新增的 11 个测试逐位验
证了这个翻译的正确性。

---

## 附:验证用的测试套件

`tests/qwen3_omni_talker_projection_equivalence_test.py` 共 11 个测试,8s
跑完,锁定了三层等价性:

| 层级                     | 测试数 | 证明的事                                                                   |
| ------------------------ | ------ | -------------------------------------------------------------------------- |
| Projection 公式          | 4      | user / assistant / full chatml / decode 的 wrap 算术 mask == HF 控制流      |
| 防漂移 grep              | 1      | `_talker_model.py` 6 行公式被锁,改了立刻 fail                              |
| Adapter 契约             | 3      | 8 元组形状 / padding / legacy 拒绝 / graph forward 一致                     |
| 真实 wrap forward        | 2      | fp32 atol 1e-6 + **fp16 `torch.equal` bit-exact**                          |
| 端到端集成               | 1      | adapter.prepare_inputs → 真实 wrap forward → trunk inputs_embeds 与 HF 等价 |

运行:

```bash
python -m pytest tests/qwen3_omni_talker_projection_equivalence_test.py -v
```
