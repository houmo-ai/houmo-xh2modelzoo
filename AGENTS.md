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

| Tool                        | Use when                                               |
| --------------------------- | ------------------------------------------------------ |
| `detect_changes`            | Reviewing code changes — gives risk-scored analysis    |
| `get_review_context`        | Need source snippets for review — token-efficient      |
| `get_impact_radius`         | Understanding blast radius of a change                 |
| `get_affected_flows`        | Finding which execution paths are impacted             |
| `query_graph`               | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes`     | Finding functions/classes by name or keyword           |
| `get_architecture_overview` | Understanding high-level codebase structure            |
| `refactor_tool`             | Planning renames, finding dead code                    |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.

## 代码审查（OpenCodeReview）

进行任何代码 review 时，必须使用本机安装的 OpenCodeReview CLI，可执行命令为 `ocr`（不是 `opencode-review`）。

### 命令选择

- 审查当前工作区的 staged、unstaged 和 untracked 变更：`ocr review --repo <repo-root> --audience agent`
- 审查分支差异：`ocr review --repo <repo-root> --from <base-ref> --to <target-ref> --audience agent`
- 审查单个 commit：`ocr review --repo <repo-root> --commit <commit> --audience agent`
- 仅在没有 Git diff、需要审查完整文件或目录时使用：`ocr scan --repo <repo-root> --path <repo-relative-path> --audience agent`
- 需要确认某文件命中的 review rule 时使用：`ocr rules check --repo <repo-root> <repo-relative-file>`

### 执行规则

1. 先按上文的 CodeGraph 流程了解变更及影响范围，再执行 `ocr`。
2. 范围不明确时，先使用相同参数加 `--preview` 确认待审查文件，再移除 `--preview` 执行实际 review。
3. 有需求、设计或业务背景时，`ocr review` 通过 `--background <text>` 或 `--background-file <markdown-file>` 传入，两者可以同时使用；`ocr scan` 仅使用 `--background <text>`。
4. 面向 Agent 执行时使用 `--audience agent`；需要结构化结果时另加 `--format json`。
5. 必须根据实际代码、调用关系和测试复核 `ocr` 的结论；只报告可复现、可定位且对正确性有实际影响的问题。
6. 审查结果中说明实际执行的 `ocr` 命令和审查范围；若 `ocr` 无法执行，明确报告原因，不得声称已使用。

## 代码提交

commit时候应该遵守规则：https://houmo.feishu.cn/wiki/B7Yeww5MKixxaSkqaJ7cOlF5n2c
