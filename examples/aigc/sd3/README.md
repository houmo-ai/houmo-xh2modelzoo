# SD3 (Stable Diffusion 3)

本示例展示如何将 Stable Diffusion 3 模型导出为xh2硬件可用格式并进行推理验证。

## 特别事项

本项目使用从 Hugging Face 下载的预训练模型、配置文件和脚本：

- 模型名称: Stable Diffusion 3 (SD3)
- 来源: https://huggingface.co/stabilityai/stable-diffusion-3-medium
- 许可: Stability AI Community License （https://huggingface.co/stabilityai/stable-diffusion-3-medium/blob/main/LICENSE.md）

预训练模型、配置文件和脚本在运行时下载，工程发布**不包含**该下载包。
同时，原有的导出和验证脚本已进行适配性修改，并包含在项目提交文件中。特此声明。

## 依赖

```bash
pip install transformers==4.47.0
diffusers       0.29.2
sentencepiece
peft            0.16.0
```

## 导出SD3-2b模型

```bash
python examples/aigc/sd3/sd3_export.py --model data/models/stable-diffusion-3-medium-diffusers --guidance-scale 7 --width 512 --height 512 
```

## 验证

```bash
python examples/aigc/sd3/sd3_hmonnx_test.py --config work_dirs/stable-diffusion-3-medium-diffusers_XH2a_512x512/meta.json --hf-model data/models/stable-diffusion-3-medium-diffusers --steps 28
```

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

