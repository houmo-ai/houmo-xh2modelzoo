# Eagle3 VL Model Notice

## 特别事项

本项目仅包含上述模型的接口存根（stub），不发布任何预训练模型权重：

- 模型名称: Isaac-GR00T / Eagle3-VL / starforce (stub only)
- 来源: https://github.com/NVIDIA/Isaac-GR00T 等
- 许可: Non-Commercial License (NVIDIA Isaac GR00T) / 上游 starforce 许可不明 （https://github.com/NVIDIA/Isaac-GR00T/blob/n1.6-release/LICENSE）

预训练模型、配置文件和脚本在运行时下载，工程发布**不包含**该下载包。
同时，原有的接口存根代码（Eagle3_VLConfig 等运行时抛出 NotImplementedError）已进行适配性修改，并包含在项目提交文件中。特此声明。

## Status
**Eagle3 VL implementation removed pending upstream license clarification.**

## Reason
The `Eagle3_VLConfig` and `Eagle3_VLForConditionalGeneration` classes in this directory have been stubbed out and are not available in the xh2modelzoo Apache-2.0 release.

## Upstream License Issue
- **Upstream source**: PyPI package `starforce` (versions 1.0.0-1.1.0)
- **Problem**: The starforce PyPI sdist distribution lacks a LICENSE file
- **Metadata conflict**: setup.py metadata historically indicated GPL-3.0, but current status unclear
- **Mixed provenance**: Code contains segments with NVIDIA MIT file-level notices, but overall licensing status of starforce package remains unresolved

## What remains
- Stub classes that maintain import compatibility but raise `NotImplementedError` at runtime
- NVIDIA MIT copyright notice preserved in file header for code segments originally derived from NVIDIA sources

## Usage impact
Any attempt to instantiate or use `Eagle3_VLConfig` or `Eagle3_VLForConditionalGeneration` will result in:
```python
NotImplementedError: Implementation removed pending upstream starforce license clarification. 
This model is not available in the Apache-2.0 release.
```

## Resolution path
This implementation may be restored if:
1. Upstream starforce maintainers provide clear LICENSE declaration, or
2. Alternative clean-room implementation is developed, or
3. HOUMO legal/P10 explicitly approves inclusion under documented risk acceptance

## Contact
For questions or to request Eagle3 VL functionality, escalate to xh2modelzoo maintainers.

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
