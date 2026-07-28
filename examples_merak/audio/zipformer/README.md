# Streaming Zipformer ASR Encoder（XH2a/M50）

本例把 sherpa-onnx 发布的中文 14M streaming Zipformer Transducer
Encoder 转成适合 XH2a/M50 的定长 HMONNX。它完成了三件事：

1. 把真实 utterance batch 固定为 `N=1`；
2. 把 5 组 cache 的 encoder-layer 首维 `2/3` 拆成 84 路首维为
   `1` 的独立输入和输出；
3. 把公开 `cached_len` 从 INT64 改为 INT32，并在图内保持原始 INT64
   位置计算。

这里转换的是 Transducer 的 **Encoder**，不是完整 ASR。CPU 仍需负责
16 kHz 音频、80 维 fbank、39/32 帧窗口调度、Decoder、Joiner、greedy/beam
search 和文本拼接。

## 1. 开源来源与模型身份

- 推理工程：[k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)
- 模型说明：[Streaming Zipformer Transducer models](https://k2-fsa.github.io/sherpa/onnx/pretrained_models/online-transducer/zipformer-transducer-models.html)
- 发布包：`sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2`
- 本例 encoder：`encoder-epoch-99-avg-1.onnx`
- Encoder SHA256：
  `84c6a8f372686faa5b8f45f2d79f0816f76dcd9f547acb9a90eba2772d7eda8b`
- 发布包 SHA256：
  `2cbd71b640d9c37d3784f29367333a4577b0398b62e9deeed418170b081cba8b`

按下面的通用目录准备模型；`--model-dir` 指向解压目录：

```text
<zipformer-model-dir>/
├── encoder-epoch-99-avg-1.onnx
├── decoder-epoch-99-avg-1.onnx
├── joiner-epoch-99-avg-1.onnx
├── tokens.txt
└── test_wavs/
    ├── 0.wav
    ├── 1.wav
    └── 8k.wav
```

下载示例：

```bash
curl -L \
  -o sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2 \
  https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2

sha256sum sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2
tar -xjf sherpa-onnx-streaming-zipformer-zh-14M-2023-02-23.tar.bz2
```

同名 `encoder-epoch-99-avg-1.onnx` 在不同 Zipformer 发布包中可能代表不同
模型，不能只按文件名判断。客户模型必须先与上面的 SHA256 对齐。

## 2. 原始模型在做什么

一次调用输入 39 帧 fbank：

```text
16 kHz PCM
  -> CPU online fbank，80 bins
  -> [1,39,80]，每次向前移动 32 帧
  -> Zipformer Encoder
  -> encoder_out [1,8,320]
  -> Transducer Decoder + Joiner + 搜索
  -> token ids -> 文本
```

39 帧不是每次的步长。相邻窗口重叠 7 帧；调用方每次只消费 32 个新
fbank 帧。Encoder 内部包含 embedding、5 个多尺度 Zipformer stack、
attention、feed-forward、卷积模块、跨尺度组合以及输出投影。一次调用会把
39 帧变成 8 个 320 维 encoder frame。

流式上下文不在模型权重中，而在 35 个逻辑 cache 张量中。每次输出的
`new_cached_*` 必须回灌给下一次同名 `cached_*`；新流开始或 reset 时全部
清零。

## 3. 原始 cache shape 的真实含义

原模型输入的第一维 `2/3` 是每个 stack 中的 **encoder layer 数**，不是
utterance batch。真正的 batch 是 shape 中的 `N`，本例已固定为 1。

| stack `i` | layer 数 | `len` | `avg` | `key` | `val` / `val2` | `conv1` / `conv2` |
|---:|---:|---|---|---|---|---|
| 0 | 2 | `[2,N]` | `[2,N,160]` | `[2,64,N,96]` | `[2,64,N,48]` | `[2,N,160,30]` |
| 1 | 3 | `[3,N]` | `[3,N,160]` | `[3,32,N,96]` | `[3,32,N,48]` | `[3,N,160,30]` |
| 2 | 2 | `[2,N]` | `[2,N,160]` | `[2,16,N,96]` | `[2,16,N,48]` | `[2,N,160,30]` |
| 3 | 2 | `[2,N]` | `[2,N,160]` | `[2,8,N,96]` | `[2,8,N,48]` | `[2,N,160,30]` |
| 4 | 3 | `[3,N]` | `[3,N,160]` | `[3,32,N,96]` | `[3,32,N,48]` | `[3,N,160,30]` |

每一类状态的意义：

- `cached_len_i`：每层已经处理的位置计数，原模型公开类型为 INT64；
- `cached_avg_i`：每层统计/平均状态，通道数 160；
- `cached_key_i`：attention key 左上下文，attention 维 96；
- `cached_val_i`、`cached_val2_i`：两路 attention value 左上下文，维 48；
- `cached_conv1_i`、`cached_conv2_i`：两路卷积模块的 30 帧历史，通道数
  160。

## 4. 为什么拆成 84 路首维为 1 的 cache

原模型有 `5 stacks × 7 cache family = 35` 个 cache 输入。每个输入又把
该 stack 的 2 或 3 层堆在 axis 0。M50 部署图里，这种 `2/3` 容易被误读
成模型 batch，而且设备侧状态绑定时还需记住各 stack 的特殊首维。

本例沿 axis 0 拆分：

```text
cached_key_1 [3,32,1,96]
  -> cached_key_1_layer0 [1,32,1,96]
  -> cached_key_1_layer1 [1,32,1,96]
  -> cached_key_1_layer2 [1,32,1,96]
```

12 个 encoder layer（`2+3+2+2+3`）乘以 7 类状态，共 84 路 cache，
加 `x` 后一共 85 个输入；输出同样是 `encoder_out + 84 cache = 85`。
每个公开 cache 的首维都是 1，图内先用 `Concat(axis=0)` 恢复官方布局，
输出端再用 `Split(axis=0)` 拆回各层。因此：

- 内部 Zipformer 数学、权重和 layer 顺序完全不变；
- 真实 batch 仍固定为 1；
- 输入数量变多，但设备侧每层状态 shape 统一、绑定关系直观；
- 拆分不减少 cache 内存，也不允许漏传某一路；
- 这是接口重排，不是模型剪枝。

拆分后的每层 shape 如下：

| stack `i` | 每层 `len` | 每层 `avg` | 每层 `key` | 每层 `val/val2` | 每层 `conv1/conv2` |
|---:|---|---|---|---|---|
| 0 | `[1,1]` | `[1,1,160]` | `[1,64,1,96]` | `[1,64,1,48]` | `[1,1,160,30]` |
| 1 | `[1,1]` | `[1,1,160]` | `[1,32,1,96]` | `[1,32,1,48]` | `[1,1,160,30]` |
| 2 | `[1,1]` | `[1,1,160]` | `[1,16,1,96]` | `[1,16,1,48]` | `[1,1,160,30]` |
| 3 | `[1,1]` | `[1,1,160]` | `[1,8,1,96]` | `[1,8,1,48]` | `[1,1,160,30]` |
| 4 | `[1,1]` | `[1,1,160]` | `[1,32,1,96]` | `[1,32,1,48]` | `[1,1,160,30]` |

## 5. 一步一步的图改造

实现位于
`xhmodel_merak/xh_other_model/models/zipformer/graph.py`：

1. 读取官方 encoder，不改权重；
2. 把符号维 `N` 固定为 `1`，并刷新所有输出 shape；
3. 用 onnxsim 做常量折叠和 shape 简化，并运行 3 组数值检查；
4. 把五路公开 `cached_len_i/new_cached_len_i` 改为 INT32；
5. 在输入后插入 INT32→INT64 `Cast`，保持官方位置运算；
6. 在输出前插入 INT64→INT32 `Cast`；
7. 把每个原始 cache input 替换成 2/3 个 `_layer{k}` 输入；
8. 用 `Concat(axis=0)` 接回原图内部 tensor；
9. 把每个原始 `new_cached_*` 用 `Split(axis=0)` 拆成 `_layer{k}`；
10. 写入固定 batch、39/32 窗口、layer 数和状态回灌 metadata；
11. 用 ONNX checker 检查模型；
12. 对官方图和改造图连续回灌多段随机输入，逐项比较
    `encoder_out` 和全部 cache。

静态化后随机 3 段连续回灌的最大浮点差为 `1.81198e-5`，所有整数状态
完全一致。真实音频 18 段的 source/static 最差最大绝对误差为
`5.07385e-6`，最小 cosine 为 `0.9999999999990793`。

## 6. Merak 目录与职责

```text
configs_merak/workflows/xh2a/other_models/zipformer/
└── zipformer_xh2a_w16.yaml

xhmodel_merak/xh_other_model/models/zipformer/
├── model.py       # register_other_model 注册
├── workflow.py    # quant/export/dump_golden 三阶段
├── graph.py       # Batch=1、INT32、84 路 cache 拆分和等价验证
└── runtime.py     # ORT/HMONNX runner、状态管理、fbank 窗口

examples_merak/audio/zipformer/
├── zipformer_workflow.py
├── real_audio_eval.py
└── README.md
```

工作流严格分离：

- `quant()`：返回 skipped；本例的权重量化发生在 ONNX→HMONNX 导出；
- `export()`：生成静态 ONNX、XH2a HMONNX、落盘配置和
  `export_meta_info.json`；
- `dump_golden()`：对已生成 HMONNX 单独执行一次 85 输入 golden。

## 7. 环境

Merak/XHQuant 环境需具备：

- `onnx`、`onnxruntime`、`onnxsim`；
- `torch`、`xhquant`；
- 真实 ASR 验证额外需要 `kaldi-native-fbank`、`soundfile`、
  `sherpa-onnx`。

示例命令使用环境占位符：

```bash
conda activate <merak-env>
export PYTHONPATH=.
```

## 8. 导出 XH2a W16 HMONNX

```bash
python examples_merak/audio/zipformer/zipformer_workflow.py \
  --model-dir <zipformer-model-dir> \
  --output-dir work_dirs/zipformer_merak/export_xh2a_w16a16 \
  --device cuda:0 \
  --dump-golden
```

已有输出目录默认不会删除。如明确需要重建，增加 `--overwrite`；它只会
删除这次指定的 `--output-dir`。

主要产物：

```text
<output-dir>/
├── zipformer_xh2a_w16.yaml
├── export_meta_info.json
├── onnx/
│   └── zipformer_encoder_b1_layer_cache.onnx
├── hmonnx/
│   ├── zipformer_encoder_b1_layer_cache_XH2a_w16a16_sefp.onnx
│   └── zipformer_encoder_b1_layer_cache_XH2a_w16a16_sefp.log
└── golden/
    └── manifest.json
```

`export_meta_info.json` 记录源/目标 SHA256、量化类型、85 入/85 出、84 路
cache、layer 数、官方配套 Decoder/Joiner/tokens 身份和静态等价结果。

## 9. 真实音频精度验证

```bash
python examples_merak/audio/zipformer/real_audio_eval.py \
  --model-dir <zipformer-model-dir> \
  --export-dir work_dirs/zipformer_merak/export_xh2a_w16a16
```

脚本对同一段真实 16 kHz 音频执行：

1. sherpa-onnx 官方完整 recognizer；
2. 官方 source encoder + 官方 Decoder/Joiner；
3. 85 口 static encoder + 官方 Decoder/Joiner；
4. XH2a W16 HMONNX encoder + 官方 Decoder/Joiner；
5. 每段 `encoder_out` 误差、连续 cache 回灌、最终文本和 CER。

本仓验证数据是发布包 `test_wavs/0.wav`：

| 项目 | 结果 |
|---|---:|
| 采样率 / 时长 | 16 kHz / 5.6115 s |
| fbank 窗口 | `[1,39,80]`，shift 32 |
| 流式 chunk 数 | 18 |
| 参考字符数 | 25 |
| source/static/HMONNX 文本 | 全部完全一致 |
| source/static/HMONNX CER | 全部 0 |
| static encoder 最差 max abs | `5.07385e-6` |
| static encoder 最小 cosine | `0.9999999999990793` |
| W16 HMONNX encoder 最差 max abs | `0.0169362` |
| W16 HMONNX encoder 最小 cosine | `0.9999502678` |
| W16 HMONNX encoder 最小 SNR | `39.8265 dB` |

这是一条真实音频功能和数值回归，不等价于业务测试集 WER。量产前仍需在
客户领域测试集上跑长音频、噪声、口音、空白段、断流/reset 和 WER/CER。

## 10. 推荐量化配置

首发推荐配置已经写在
`configs_merak/workflows/xh2a/other_models/zipformer/zipformer_xh2a_w16.yaml`：

```yaml
quant_type: w16a16_sefp
```

推荐 W16 的原因：

- streaming state 会跨 chunk 累积误差；
- Transducer token 一次排名翻转会改变后续 Decoder 状态；
- W16 已通过 18 个连续 chunk、完整 Decoder/Joiner 和 CER=0 闭环；
- cache 拆口只改变接口，不应与更激进的降精度同时引入。

W8/A8 只能作为后续优化项。必须重新标定并至少验证：

- 多段连续 cache 回灌的 encoder 数值；
- 完整业务测试集 WER/CER；
- 长音频 state 漂移；
- partial result 稳定性；
- reset 后与新 stream 隔离；
- 真机性能和 cache 常驻/拷贝策略。

## 11. M50 运行时约束

每个 stream 独立维护一份 84 路 cache：

```python
output = encoder.run({"x": fbank_39x80, **cache})
for name in cache:
    cache[name] = output[f"new_{name}"]
encoder_out = output["encoder_out"]
```

工程上建议：

- cache 尽量常驻设备，用 ping-pong 或可证明安全的原位绑定；
- `cached_len` 保持 INT32，其他公开 cache/feature 按 HMONNX 使用 FP16；
- 每次输入 `[1,39,80]`，每次推进 32 帧；
- 同一 stream 严格按顺序调度，不并发改写其 cache；
- 不同 stream 不能共享 cache；
- 新流、取消、异常结束后必须清零/reset；
- 只把 `encoder_out [1,8,320]` 送给 Decoder/Joiner，不把 84 路 cache
  当成业务输出回传 Host（若 M50 runtime 支持设备内回灌）。

最常见错误是把 layer 首维当成 batch、按 39 帧步进、漏回灌某一路
`val2/conv2`、拿错同名发布包，或者只比较单个 chunk。
