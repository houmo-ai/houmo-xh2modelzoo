# Silero VAD 8 kHz / 16 kHz（CPU 或 XH2a/M50）

本例适配 Silero VAD 官方流式 ONNX。它不是“选择器”：最外层 `If` 只根据
采样率选择 8 kHz/16 kHz 两套实际 VAD 子图，分支内部包含特征提取、卷积
encoder、LSTM state 和语音概率输出。

默认工程建议直接在 CPU 运行，因为模型很小，Host 本来就要管理音频 context、
LSTM state、阈值和端点状态机。若客户要求把统一音频链路放到 M50，本例也提供
两张独立的 XH2a W16 HMONNX。

## 1. 开源来源和模型身份

- 官方工程：[snakers4/silero-vad](https://github.com/snakers4/silero-vad)
- 官方 release：[silero-vad/releases](https://github.com/snakers4/silero-vad/releases)
- 本例固定验证 commit：
  `76e3dc408eb2a5c655c34e230d2d5459b4439daa`
- 源模型相对路径：
  `src/silero_vad/data/silero_vad.onnx`
- 源模型 SHA256：
  `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`

客户 `silero_vad.onnx` 与上述官方文件逐字节相同。准备源码：

```bash
git clone https://github.com/snakers4/silero-vad.git <silero-model-dir>
git -C <silero-model-dir> checkout \
  76e3dc408eb2a5c655c34e230d2d5459b4439daa
sha256sum \
  <silero-model-dir>/src/silero_vad/data/silero_vad.onnx
```

正式交付必须同时记录 commit 和 ONNX SHA，不能只按文件名识别。

## 2. 模型作用和原图接口

VAD（Voice Activity Detection）对每个流式音频 frame 输出一个
`[0,1]` 语音概率。产品通常再以 `0.5` 或业务阈值驱动开始说话、继续说话、
结束说话等状态机。

官方整图接口：

| 名称 | shape/type | 含义 |
|---|---|---|
| `input` | `[B,S]` FP32 | 历史 context + 当前新 frame |
| `state` | `[2,B,128]` FP32 | LSTM hidden/cell state |
| `sr` | scalar INT64 | `8000` 或 `16000`，控制最外层 `If` |
| `output` | `[B,1]` FP32 | 当前 frame 的语音概率 |
| `stateN` | `[2,B,128]` FP32 | 下一 frame 的 LSTM state |

官方固定流式协议：

| 采样率 | Host 新 frame | 历史 context | 模型 `input` |
|---:|---:|---:|---|
| 8 kHz | 256 samples / 32 ms | 32 samples | `[1,288]` |
| 16 kHz | 512 samples / 32 ms | 64 samples | `[1,576]` |

两个采样率每次都推进 32 ms。`input` 不是只含新 PCM；Host 必须把上一次
input 尾部的 context 拼在新 frame 前面。

## 3. 一步一步的数据流

```text
mono PCM
  -> 必要时 CPU resample 到 8 kHz 或 16 kHz
  -> 每 32 ms 切 256/512 个新 sample
  -> 拼接 32/64 个历史 context
  -> VAD feature/STFT branch
  -> convolution encoder
  -> recurrent LSTM(state)
  -> probability [1,1]
  -> threshold / endpoint state machine
  -> 回灌 stateN，保存最新 context
```

一次实际调用：

1. 新 stream 时 `context` 和 `state` 全 0；
2. 读取 256/512 个新 sample，不足一帧时由业务决定补 0 或丢弃；
3. `model_input = concat(context, new_frame)`；
4. 执行对应采样率模型；
5. 读取 `output[0,0]`；
6. `state = stateN`；
7. `context = model_input` 的最后 32/64 个 sample；
8. 下一帧重复；
9. stream 结束、采样率切换或 reset 时同时清零两类状态。

不能只回灌 LSTM state 而漏掉音频 context，也不能让不同 stream 共享状态。

## 4. 为什么拆成两个模型

官方外层 `If(sr == 16000)` 是运行时采样率选择。产品在一个 stream 开始前
已经知道采样率，不需要每 32 ms 在 NPU 图里重复选择。拆分后：

- `silero_vad_8000_b1_ifless.onnx`：只保留 8 kHz 子图；
- `silero_vad_16000_b1_ifless.onnx`：只保留 16 kHz 子图；
- 移除公开 `sr` 输入；
- batch 固定为 1；
- 每张图的音频长度、context 和所有中间 shape 固定；
- CPU scheduler 按 stream 采样率选择一次模型。

这样既消除控制流算子，又避免用一个最大 576 输入配 mask 去模拟 8 kHz。
两套分支的 STFT/卷积配置不同，直接分别导出语义更清晰。

## 5. 子图内的 `If` 怎么处理

分支内部仍有由 `Shape/Rank` 产生的 `If`，主要用于兼容不同输入 rank 和
padding 分支。固定 `B=1` 与固定 sample 数后，这些条件都可以静态求值。

`graph.py` 的处理顺序：

1. 读取官方整图；
2. 根据 `sr=8000/16000` 选择外层 branch；
3. 把 branch graph 内联到主图；
4. 固定 `input/state/output/stateN` shape；
5. 对内部 shape/rank 条件做常量传播；
6. 选择确定的 then/else branch 并内联，直到没有 `If`；
7. 把 xhquant 不接受的 rank-2 reflect Pad 等价改写成
   `Unsqueeze -> rank-3 Pad -> Squeeze`；
8. ONNX checker 检查；
9. 连续多帧比较官方 source graph 与静态图，逐帧回灌 state。

所以答案是：子图也必须按 8 kHz/16 kHz 固定输入尺寸；固定后内部 `If`
不是留给运行时执行，而是在 modelzoo 导出阶段消除。

## 6. 静态图接口

8 kHz：

| 输入/输出 | shape | 类型 |
|---|---|---|
| `input` | `[1,288]` | FP32 / HMONNX FP16 |
| `state` | `[2,1,128]` | FP32 / HMONNX FP16 |
| `output` | `[1,1]` | FP32 / HMONNX FP16 |
| `stateN` | `[2,1,128]` | FP32 / HMONNX FP16 |

16 kHz 只有 `input` 改为 `[1,576]`，其他接口相同。

静态图输入仍包含 context。M50 runtime 不应给 8 kHz 图传 `[1,256]`，
也不应给 16 kHz 图传 `[1,512]`。

## 7. Merak 目录

```text
configs_merak/workflows/xh2a/other_models/silero_vad/
└── silero_vad_xh2a_w16.yaml

xhmodel_merak/xh_other_model/models/silero_vad/
├── model.py       # register_other_model
├── graph.py       # 两分支提取、If 消除、Pad 改写、连续帧等价
├── workflow.py    # quant/export/dump_golden
└── runtime.py     # ORT/HMONNX runner、resample、context/state 调度

examples_merak/audio/silero_vad/
├── silero_vad_workflow.py
├── real_audio_eval.py
└── README.md
```

工作流严格分离：

- `quant()`：skipped，PTQ 在 ONNX→HMONNX 导出时完成；
- `export()`：生成 8 kHz/16 kHz 静态 ONNX、W16 HMONNX、落盘配置和
  顶层 `export_meta_info.json`；
- `dump_golden()`：为两个已导出的 HMONNX 分别生成 golden。

## 8. 环境与导出

```bash
conda activate <merak-env>
pip install onnx onnxruntime onnxsim scipy soundfile
export PYTHONPATH=.

python examples_merak/audio/silero_vad/silero_vad_workflow.py \
  --model-dir <silero-model-dir> \
  --output-dir work_dirs/silero_vad_merak/export_xh2a_w16a16 \
  --device cuda:0 \
  --dump-golden
```

已有输出目录默认保留。明确重建时增加 `--overwrite`，只删除指定
`--output-dir`。

配置：
`configs_merak/workflows/xh2a/other_models/silero_vad/silero_vad_xh2a_w16.yaml`。

产物：

```text
<output-dir>/
├── silero_vad_xh2a_w16.yaml
├── export_meta_info.json
├── onnx/
│   ├── silero_vad_8000_b1_ifless.onnx
│   └── silero_vad_16000_b1_ifless.onnx
├── hmonnx/
│   ├── silero_vad_8000_b1_ifless_XH2a_w16a16_sefp.onnx
│   └── silero_vad_16000_b1_ifless_XH2a_w16a16_sefp.onnx
└── golden/
    ├── 8000/
    ├── 16000/
    └── manifest.json
```

`export_meta_info.json` 记录 source commit/SHA、两图 frame/context、ONNX/
HMONNX 路径、量化类型和产物 SHA。

## 9. 真实音频精度

```bash
python examples_merak/audio/silero_vad/real_audio_eval.py \
  --model-dir <silero-model-dir> \
  --export-dir work_dirs/silero_vad_merak/export_xh2a_w16a16 \
  --audio <silero-model-dir>/tests/data/test.wav \
  --seconds 3
```

同一段官方真实 WAV 会分别 resample 到 8 kHz/16 kHz，对官方整图、静态
ONNX 和 XH2a W16 HMONNX 连续回灌 94 帧。

| 采样率 | frames | static vs source max abs | W16 max abs | W16 cosine | 0.5 判决一致 |
|---:|---:|---:|---:|---:|---:|
| 8 kHz | 94 | `0` | `0.00231409` | `0.9999998267` | 100% |
| 16 kHz | 94 | `0` | `0.00440711` | `0.9999996620` | 100% |

两种采样率均无 false-speech frame、无 missed-speech frame。静态化图与官方
图逐值一致，说明拆分/If/Pad 改写没有引入数值误差。

这是 3 秒官方样例的回归基线，不等价于业务 VAD 全量验收。量产前还要在
静音、噪声、远场、电话、回声、音乐、多人和边界语音上统计 precision/
recall/F1、误检率、漏检率和 endpoint delay。

## 10. 推荐部署与量化

推荐优先级：

1. **CPU FP32**：模型很小，context/state/阈值本就由 Host 管理，通常是
   最简单且能耗/调度合理的方案；
2. **M50 W16A16**：若音频计算必须统一到 NPU，使用已验证的
   `w16a16_sefp` 两图；
3. **W8/A8**：只能在客户真实数据集重新标定并通过任务指标后使用。

不能用单帧 cosine 代替 VAD 精度。W8 必须连续回灌 state，并至少比较
frame probability、阈值判决、segment start/end、F1 和端点偏移。

## 11. M50 运行约束

- stream 创建时按采样率选择一次 8 kHz 或 16 kHz HMONNX；
- 采样率不匹配时先在 CPU resample，不要向 8 kHz 图喂 16 kHz PCM；
- 每次推进恰好 32 ms；
- context 和 state 尽量常驻同一调度对象；
- 每个 stream 独占状态，不并发改写；
- 输出概率回 Host 后再执行可配置阈值、hangover 和 endpoint 状态机；
- reset/切换采样率时 context/state 必须一起清零；
- 尾帧补 0 策略必须和离线评测一致。
