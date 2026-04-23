# VoxCPM2 XH2a 

## 功能

模型可以实现 Text-to-Speech、自然语言描述声音、参考音频音色克隆以及参考音频/文本续生成，模型以 patches 作为基本分块单位。最基本用法有 zero-shot 与 参考音频生成。

## 架构

对 TTS 模型 `VoxCPM2` 量化适配导出，根据其架构拆分为：vae encoder，vae decoder，locdit，locenc，以及 basellm 与 residualllm 几个部分。其中 basellm 与 residualllm 分别有 prefill 与 decode。每个程序的用户附在文件首。

## 依赖

`pip install voxcpm`
