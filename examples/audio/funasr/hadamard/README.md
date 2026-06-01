# Hadamard Transform Implementation for FunASR

## 特别事项

本项目使用以下第三方开源代码与算法：

- 模型名称: fast-hadamard-transform / GPTQModel (Hadamard utilities)
- 来源: https://github.com/Dao-AILab/fast-hadamard-transform 与 https://github.com/ModelCloud/GPTQModel
- 许可: Apache License 2.0 （https://www.apache.org/licenses/LICENSE-2.0）

预训练模型、配置文件和脚本在运行时下载，工程发布**不包含**该下载包。
同时，原有的hadamard_utils.py 等模块已进行适配性修改，并包含在项目提交文件中。特此声明。

## Algorithm Description

The Hadamard transform is an orthogonal transformation used here for:
- Audio feature preprocessing in FunASR models
- Efficient signal rotation for quantization-aware inference
- Matrix operations optimized for xh2 hardware deployment

Functions adapted from upstream:
- `hadamard_transform()` - Core Hadamard matrix transformation
- `apply_online_hadamard()` - Streaming Hadamard for real-time audio
- Utility functions for matrix block processing

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
