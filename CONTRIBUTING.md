# xh2modelzoo Contribution Guide

## xh2modelzoo 提交与开源合规规范

### 1. 目的
为保证 `xh2modelzoo` 的开源合规性，所有提交必须明确代码归属，并确保不引入 AGPL、非商业许可或来源不明的代码。

### 2. 基本原则
提交代码前，开发者必须先判断该代码属于以下哪一类：

1. **自研代码**
2. **第三方开源代码直接引入**
3. **第三方开源代码修改/派生**

凡无法明确归属、来源或许可证的代码，一律不得提交。

### 3. 自研代码要求
只有在满足以下条件时，代码才可标记为“自研”：

- 由 HOUMO 独立开发；
- 未复制、改写、粘贴第三方源码实现；
- 不包含第三方源码片段、预计算表、特殊矩阵、版权头、来源注释或受版权保护的实现细节。

自研 Python 文件建议使用如下头部：

```python
# Copyright 2025 HOUMO AI
# SPDX-License-Identifier: Apache-2.0
```

### 4. 第三方开源代码直接引入要求
若代码直接来自第三方开源项目，必须满足以下要求：

1. 保留原始版权声明、来源说明和许可证声明；
2. 将对应许可证文件保存到：

   ```text
   licenses/<component>/LICENSE
   ```

3. 在 `THIRD_PARTY_NOTICES` 中补充对应组件信息；
4. 在提交说明或 PR 描述中写明：
   - 上游项目名称
   - 上游版本、commit 或 release
   - 许可证类型
   - 引入文件范围

### 5. 第三方开源代码修改/派生要求
若代码基于开源项目修改而来，必须满足以下要求：

1. 不得删除、隐藏或覆盖上游 attribution；
2. 必须同时保留上游版权信息和 HOUMO 修改信息；
3. 必须补充对应许可证文件到 `licenses/<component>/LICENSE`；
4. 必须在提交说明或 PR 描述中写明修改来源和修改范围。

建议头部格式如下：

```python
# Copyright (c) <upstream copyright holder>
# Copyright 2025 HOUMO AI (modifications)
# Licensed under the <license name>
# See licenses/<component>/LICENSE for full license text
# Source: <upstream url or repo>
```

### 6. 禁止提交的代码
以下内容禁止提交到 `xh2modelzoo`：

- AGPL / AGPL-3.0
- 非商业用途许可（Non-Commercial / CC-BY-NC / research-only / personal-use-only / evaluation-only）
- 来源不明、许可证不明的代码
- 删除或篡改原始 LICENSE / attribution / copyright 的代码
- 明知来源于第三方却标记为“自研”的代码

以下内容默认禁止，需专项确认后才能提交：

- GPL / LGPL / SSPL / BSL / source-available 但有限制的协议
- 许可证声明与项目实际分发内容不一致的代码
- 无法明确上游许可证是否兼容 Apache-2.0 的代码

**判断原则：**

- 许可证不明确：**禁止提交**
- 来源不明确：**禁止提交**
- attribution 不完整：**禁止提交**

### 7. 提交前自查清单
提交前，开发者必须确认以下事项：

- 这段代码是自研，还是第三方代码？
- 如果是第三方，来源仓库、版本或 commit 是什么？
- 许可证类型是什么？是否允许在本仓库中使用和分发？
- 是否已补充 `licenses/<component>/LICENSE`？
- 是否已保留或补充上游 attribution？
- 是否已更新 `THIRD_PARTY_NOTICES`？
- 是否包含 AGPL、非商业或未知许可证内容？

若任一问题无法回答，禁止提交。

### 8. PR / 提交说明要求
每个 PR 必须包含如下信息：

```md
## 代码归属说明

- 类型：自研 / 第三方直接引入 / 第三方修改派生
- 文件范围：
- 上游项目：
- 上游版本/commit：
- 许可证：
- LICENSE 文件路径：
- THIRD_PARTY_NOTICES 是否已更新：是 / 否
- 是否包含 AGPL / 非商业 / 未知许可证内容：否
- 结论：允许提交 / 禁止提交
```

### 9. Reviewer 审核要求
Reviewer 必须将“代码归属与许可证合规”作为必审项，不得只审核功能正确性。

出现以下情况必须直接打回：

- 未说明代码归属
- 声称自研，但存在明显第三方来源痕迹
- 引入第三方代码但未补充 LICENSE
- 删除或覆盖上游 attribution
- 许可证不明确
- 含 AGPL、非商业或其他受限许可证内容

### 10. 兜底规则
若开发者无法确认代码是否可提交，必须按以下流程处理：

1. 暂停提交；
2. 查明上游来源和许可证；
3. 如仍无法确认，则按“禁止提交”处理；
4. 必要时由项目负责人或合规负责人确认后再决定是否提交。

### 11. 一句话原则
**自研代码明确标自研；引用开源代码必须保留来源和 LICENSE；AGPL、非商业、来源不明代码一律不得进入仓库。**
