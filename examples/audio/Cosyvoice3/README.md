# cosyvoice3-0.5B模型

导出该模型需要同时使用xhquanttool环境。

模型主体可分为五部分：campplus, speech_tokenizer_v3, llm, flow_decoder, hift；除llm部分之外，其余模块均通过onnx可直接导出。

## 配置参数

可修改对应config文件中的参数。

```llm
context-length
input-sequence-length
quant-type: for example: w8a8h1-sefp
```

## ONNX模型预处理以及导出HMONNX

替换相应的模型路径即可。

### campplus

```bash
python campplus_export_hmonnx.py
```

### speech_tokenizer_v3

```bash
python speech_tokenizer_v3_convert.py
python speech_tokenizer_v3_export_hmonnx.py
```
### llm(qwen2 0.5B)

```bash
python qwen2_xh2a_export_0.5B.py
```

### flow decoder 

```bash
python decoder_export_hmonnx.py
```

### hift

```bash
python hift_export_hmonnx.py
```

### other module

```bash
python other_export_hmonnx.py
```

## demo运行

```bash
python demo_new.py
```

## cv3_eval 评估

```bash
python cv3_eval.py
```

## 特别事项

本项目使用从 ModelScope/Hugging Face 下载的预训练模型、配置文件和脚本：

- 模型名称: CosyVoice3 (cosyvoice3-0.5B)
- 来源: https://github.com/FunAudioLLM/CosyVoice
- 许可: Apache License 2.0 （https://www.apache.org/licenses/LICENSE-2.0）

预训练模型、配置文件和脚本在运行时下载，工程发布**不包含**该下载包。
同时，原有的导出与推理脚本（campplus、speech_tokenizer_v3、llm、flow_decoder、hift 等模块）已进行适配性修改，并包含在项目提交文件中。特此声明。

## 4 免责声明

您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

