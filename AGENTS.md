<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.


## 代码提交
commit时候应该遵守规则：https://houmo.feishu.cn/wiki/B7Yeww5MKixxaSkqaJ7cOlF5n2c

## Gemma4 Merak 验收规则

- Gemma4 大模型导出/生成验收必须使用全量规格：`context_max_length=2048`、`prefill_chunk_length/input_sequence_length=256`。
- E4B、31B、26B-A4B 三个统一 API 预设都必须按同一验收标准覆盖，不能只验证其中一个模型。
- 每个模型的 demo/e2e 验收至少必须覆盖 `text` 和 `image` 两种输入；E4B 额外具备的 audio/video 能力需要在对应阶段单独验证。
- text e2e 的 prompt token 长度必须 **超过 1024**，且 `prompt_tokens + max_new_tokens <= context_max_length`，用于覆盖 sliding-window/slice-window 边界；短 prompt 不能作为 text e2e 完成证据。
- e2e prompt 必须是真实问答或指令，不允许只用 `1+1`、`你好`、纯占位串等弱问题；长 prompt 应包含可回溯资料卡/约束，并在尾部提出明确问题，验证长上下文 retrieval 和 generate。
- image e2e 必须是真实图像问答：输入图像需要包含可观察内容（例如文字、颜色、形状、布局或数字），prompt 必须询问图像内容；只把图片塞进 prompt 但问题与图片无关，不算 image e2e。
- E4B video/audio 阶段必须跑真实 `generate`：video 走多帧输入并经过独立 `video_visual` HMONNX，audio 走本地音频输入并经过 `audio` HMONNX；processor dry-run、meta 文件存在、子图导出成功都不能替代 generate 证据。
- 不要把 smoke test、小模型 shape、`py_compile`、dry-run 当作完成证据；这些只能作为定位或快速回归的辅助证据。
- 完成阶段必须跑真实全量链路：`to_quanted_aligned`、HMONNX export、generate，并检查生成结果。
- base 导出可以跳过 workflow quant stage（`config_overrides={"quant": None}`），但不得把 `export.model.quant_scheme` 改成 `None`；默认保持 `w8a8h1_sefp`。
- Gemma4 video 必须走独立 `video_visual` ViT 尺寸，不允许 pad 到 image 的 2520-patch ViT 上作为验收路径。
- 验证调度允许使用空闲 GPU 并行，但必须保持 **单 GPU 同时只跑一个 Gemma4 导出/生成任务**，避免多任务抢同一张卡导致结论不稳定。
