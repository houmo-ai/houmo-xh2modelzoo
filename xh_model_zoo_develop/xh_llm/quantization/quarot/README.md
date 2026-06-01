# QuaRot Quantization Implementation

This directory contains code adapted from the QuaRot quantization algorithm for deployment on xh2 hardware.

## 特别事项

本项目使用以下第三方开源代码与算法：

- 模型名称: QuaRot / GPTQModel / fast-hadamard-transform
- 来源: https://github.com/spcl/QuaRot 等
- 许可: Apache License 2.0 （https://www.apache.org/licenses/LICENSE-2.0）

预训练模型、配置文件和脚本在运行时下载，工程发布**不包含**该下载包。
同时，原有的QuaRot 量化与 Hadamard 变换相关模块已进行适配性修改，并包含在项目提交文件中。特此声明。

## 算法说明

QuaRot 是一种基于旋转的大语言模型量化算法，通过Hadamard变换和在线旋转策略实现低比特量化同时保持模型精度。

本实现在上游QuaRot基础上进行了以下改进：
- 适配xh2硬件的量化参数
- 集成GPTQ量化策略
- 优化Hadamard变换实现以支持批处理

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。
