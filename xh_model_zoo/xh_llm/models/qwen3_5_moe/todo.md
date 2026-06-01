目标：把 xhquant_llm 中的 qwen3_5_moe 模型完整迁移到当前仓库的 LLMConverter 体系内，并形成“代码可合入 + 导出可运行 + hmonnx 可推理 + 有结果佐证”的闭环。

源模型路径：
- /data01/home/huxing/xhquant_llm/xhquant_llm/models/qwen3_5_moe

你要做的不是简单复制代码，而是按当前仓库的实现范式完成迁移：
- 使用 xh2modelzoo 的 ConvertConfig + Converter + LLMConverter.from_pretrained(...) 体系落地。
- 不要保留 xhquant_llm 中旧的 LLMBaseModel 导出范式。
- 迁移完成后，代码、导出脚本、hmonnx 推理、demo、结果汇总都要齐全。

一、优先参考的目标仓实现

请先充分阅读并对照以下实现，再开始迁移：
- xh_model_zoo/xh_llm/models/qwen3_5
- xh_model_zoo/xh_llm/models/qwen3_next
- xh_model_zoo/xh_llm/models/qwen3moe
- examples/llm/qwen3_5

参考关系如下：
- qwen3_5_moe 的整体目录组织、vision 导出流程、示例脚本组织，优先参考 qwen3_5。
- qwen3_5_moe 的 llm 主体结构优先参考 qwen3_next。
- qwen3_5_moe 的 MoE 相关实现、wrap 逻辑、导出与推理注意点，优先参考 qwen3moe。
- 源仓中已经调通的 qwen3_5_moe 实现只能作为“逻辑来源”，不能整体照搬到当前仓库。

二、模型结构认知

在开始编码前，请先统一以下认知，并在实现时严格遵守：
- 该模型在架构上与当前仓库中的 qwen3_5 高度接近，主要差异在 TransformerBlock 的 MLP 部分替换成了 MoE 结构，且同时存在共享专家和非共享专家。
- 该模型的 llm 结构与 qwen3_next 更接近，属于线性注意力 + SelfAttention + MoE 的组合。
- xhquant_llm 侧目前没有完整的 VIT 支持；vision 部分不要从源仓硬搬，而是优先复用当前仓库 qwen3_5 已有的 vision 导出逻辑，并只补齐 qwen3_5_moe 所需的差异。

三、必须完成的迁移内容

请按下面的目标交付，不要遗漏：

1. 模型代码迁移
- 在 xh_model_zoo/xh_llm/models/qwen3_5_moe 下补齐当前仓库所需的模型实现。
- 按当前仓库风格补齐 ConvertConfig、Converter、必要的模型实现文件，以及 __init__.py 导出。
- 如果 hmonnx 推理依赖 inference.py 或 hf_compatible.py，也一并补齐。

2. converter 注册
- 在 xh_model_zoo/xh_llm/llm_converter.py 中注册 qwen3_5_moe 对应的 architecture 分支。
- 不接受“模型代码存在但 LLMConverter 无法命中”的半迁移状态。

3. 导出脚本迁移
- 参考 examples/llm/qwen3_5，为 qwen3_5_moe 补齐 llm 导出、vision 导出、hmonnx 测试、demo 等脚本能力。
- 如果 qwen3_5 目录下存在公共辅助脚本或公共运行逻辑，请优先复用，不要无意义复制。
- 最终效果应至少覆盖 qwen3_5 目录下当前核心脚本提供的主要功能。

4. hmonnx 推理闭环
- 参考 /data01/home/huxing/xhquant_llm/examples/qwen3_5_moe/dev/qwen3_5_moe_test_ppl_hmonnx.py，完成当前仓库下导出产物的 hmonnx 推理适配。
- 迁移后的产物必须能够基于 meta.json 正常加载并完成至少一轮有效推理，而不是只导出 ONNX 文件但不能跑。

5. 浮点与量化权重双路径打通
- 浮点权重路径：/data01/nfs_shared/Qwen3.5-35B-A3B/
- GPTQModel 量化权重路径：/data01/home/huxing/gptqmodel/work_dirs/Qwen35_35B_A3B_attn4_e4_se4_0324
- 对这两个模型目录都要完成完整的导出验证。
- 如果两条路径在加载逻辑、量化配置、meta.json 字段、导出参数上存在差异，需要明确处理，不要只调通其中一条。

四、执行顺序要求

请严格按下面顺序推进，避免返工：

1. 先阅读并梳理 qwen3_5、qwen3_next、qwen3moe、源仓 qwen3_5_moe 的差异。
2. 明确 qwen3_5_moe 对应的 architecture 名称、模型入口、vision/llm 子模块边界。
3. 先完成 models/qwen3_5_moe 下的 converter 体系迁移。
4. 再补 llm_converter.py 的 architecture 注册。
5. 再补 examples/llm/qwen3_5_moe 下的导出、测试、demo 脚本。
6. 先用浮点权重跑通导出和 hmonnx 推理，再验证 GPTQModel 权重。
7. 最后整理验证结果和风险点。

五、验收标准

以下项目全部满足，才算迁移完成：

1. 代码层
- qwen3_5_moe 在当前仓库中已具备完整 converter 体系。
- llm_converter.py 已正确注册 architecture。
- examples/llm/qwen3_5_moe 下已有可运行脚本。

2. 导出层
- llm 导出可运行。
- vision 导出可运行。
- 成功生成 prefill/decode ONNX 以及 meta.json。

3. 推理层
- hmonnx 测试可正常加载导出结果。
- 至少完成 1 组有效 demo 对话或生成结果，输出非空、非报错、非明显异常。

4. 回归结果
- 至少给出一组能证明迁移成功的 demo 对话结果。
- 至少给出 ppl 或等价评测结果，并说明是基于哪套权重、哪套导出产物得到的。
- 如果浮点和 GPTQModel 两条路径的结果不同，需要分别汇总。

六、硬约束和禁止项

以下事项必须遵守：
- 不要把源仓脚本原样拷贝到当前仓库后做少量改名；必须收敛到当前仓库的 LLMConverter 工作流。
- 不要只迁 models 目录而漏掉 llm_converter.py 注册。
- 不要只导出 llm 而遗漏 vision；如果当前模型需要 vision 闭环，则 llm 和 vision 都必须验证。
- 不要只在一种权重形态下验证成功；浮点和 GPTQModel 都要实际跑通。
- 不要只汇报“代码已完成”或“导出成功”，必须给出真实的推理/对话/ppl 结果作为证据。
- 遇到 meta.json、cache shape、quant config、architecture 分派问题时，要优先从 converter 和注册链路排查，不要绕过框架打补丁。

七、最终输出要求

任务完成后，请按下面格式给我汇总：

1. 改了哪些文件
- 只列出关键文件，并说明每个文件承担的作用。

2. 迁移策略摘要
- 说明 qwen3_5、qwen3_next、qwen3moe、源仓 qwen3_5_moe 各自借鉴了什么。

3. 验证命令
- 给出浮点权重和 GPTQModel 权重各自使用的导出、测试、demo 命令。

4. 验证结果
- 给出 demo 对话样例。
- 给出 ppl 或其他关键评测结果。
- 给出 ONNX/meta.json 的产物位置。

5. 未解决问题
- 如果还有未完全解决的问题，必须明确说明阻塞点、影响范围、下一步建议；不要含糊带过。

如果中途发现我的要求与仓库现状冲突，请先基于当前仓库真实代码结构做出最合理的实现方案，再把冲突点和你的处理方式明确写出来。
