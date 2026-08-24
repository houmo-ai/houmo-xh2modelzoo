# Kokoro v1.1-zh：XH2a ModelZoo 最佳适配方案

本目录对应 `HMSW-4679`、`HMSW-4680` 和 `CMS-706`。正式部署采用独立的 T/F
bucket、三张 NPU 图，以及一段很窄的 CPU FP32 相位计算：

```text
ZHG2P / tokenization / voice pack                              Host
        |
        | 选择最小 T bucket
        v
text_duration: Front Base + Duration/Text Encoder             NPU W16A16
        |
        | duration_features / text_encoded / duration_logits
        v
duration -> 真实 F -> 选择最小 F bucket -> index/Gather      Host
        |
        v
frame_acoustic: Shared BiLSTM + F0/Noise + Decoder            NPU W16A16
                + F0 upsample + phase increments
        |
        v
phase_core: CumSum -> phase interpolation -> Sin               CPU FP32
        |
        v
generator_istft: SourceMerge + Harmonic STFT + Generator      NPU W16A16
                 + static iSTFT -> waveform
```

Host 还负责 duration 归约、bucket 路由以及用 frame-to-token index 执行 Gather；
需要保留 FP32 的模型计算只有 `CumSum -> phase interpolation -> Sin`。F0 Predictor、
F0 上采样、谐波频率构造、SourceMerge、Harmonic STFT、Generator 和 iSTFT 全部在 NPU。

## 1. 为什么不是一张静态图

Kokoro 有两个不同时间域：文本域 `T` 和声学帧域 `F`。执行文本图前只知道 token
数，`F=sum(duration)` 必须等 Duration Predictor 完成后才能得到。把 T/F 同时固化
到一张图会产生两个问题：

1. 为不同文本长度和语速覆盖全部组合，需要导出 T×F 笛卡尔积；
2. 若只绑定少量 T/F 配对，F 溢出时必须连已经正确完成的文本图一起重跑。

因此第一处边界放在 duration 后。Host 根据真实 F 选择声学 bucket，生成
frame-to-token index，再用两次 Gather 完成 Length Regulator，后续 F 图不再依赖 T。

第二处边界来自精度。W16A16 中很小的频率误差会被长序列 CumSum 持续积分，最终
形成明显错相。真实 T32/F120 HMONNX 中，相位全上 NPU 的 STFT magnitude cosine
只有 `0.780179`；只把相位核心留在 CPU FP32 后恢复到 `0.995189`。所以正式方案是
三张 NPU 图，不再把两图路径作为生产选择。

ZHG2P 和 voice 查表本来就是 Host 前处理，也不进入 ONNX。voice pack 独立保存，
Host 输出 `style[1,256]`，不会为每个音色重复导出模型。

## 2. 资产

适配锁定以下上游版本：

- 源码：[hexgrad/kokoro](https://github.com/hexgrad/kokoro)，commit
  `dfb907a02bba8152ca444717ca5d78747ccb4bec`；
- PyTorch 模型：[hexgrad/Kokoro-82M-v1.1-zh](https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh)；
- 参考 ONNX：[xun/kokoro-v1.1-zh-onnx](https://huggingface.co/xun/kokoro-v1.1-zh-onnx)；
- 默认 voice：`zf_001`。

```bash
pip install huggingface_hub
bash examples_merak/audio/kokoro/download_assets.sh \
  work_dirs/model_assets/kokoro
python -m pip install work_dirs/model_assets/kokoro/source/kokoro
```

工作流校验源码 commit、checkpoint、配置、voice 和参考 ONNX 的 SHA256，身份不匹配
时直接停止。

## 3. T/F bucket

Kokoro 的采样率为 24 kHz，每个声学帧最终对应 600 个采样点：

```text
frames_per_second = 24000 / 600 = 40
F = audio_seconds * 40
waveform_max_samples = F * 600
```

生产只保留四个 T/F 容量档位：

| route | T bucket | F bucket | 最大音频 |
|---|---:|---:|---:|
| 1 | 32 | 160 | 4 秒 |
| 2 | 64 | 320 | 8 秒 |
| 3 | 128 | 640 | 16 秒 |
| 4 | 256 | 1280 | 32 秒 |

T 图和 F 图在文件上仍分别保存，因此不是 4×4 的笛卡尔积；运行时按上表成对路由。
先执行 token 能放下的最小 route，duration 若超过该 route 的 F 容量，再用下一档 T 图
重算并继续，确保实际选择始终是表中的合法组合。T 超过 256 或 F 超过 1280 时由上层
按自然句边界切句，不做静默截断。

所有 attention mask 的有效位置为 0，无效位置为 `-65504`，不使用 `-inf`。T/F
padding 特征显式清零，最终只保留 `valid_frames*600` 个波形采样点。

## 4. 三张 NPU 图和 Host 接口

### 4.1 text_duration（T 域）

包含 Front Base、3 层 Duration Encoder BiLSTM、Duration Predictor BiLSTM 和
Text Encoder BiLSTM，共 5 个逻辑双向 LSTM。

```text
输入：
input_ids         [1,T]       INT32
attention_mask    [1,1,T,T]   FP32
style             [1,256]     FP32
valid_len         [1]         INT32
reverse_indices   [T]         ONNX INT64 / HMONNX INT32

输出：
duration_features [1,T,640]
text_encoded      [1,512,T]
duration_logits   [1,T,50]
```

### 4.2 Host duration、F 路由和 alignment

```text
duration[i] = max(round(sum(sigmoid(duration_logits[i])) / speed), 1)
duration[padding] = 0
valid_frames = sum(duration)
F_bucket = smallest(F_bucket >= valid_frames)

alignment[i,j] = 1  if start[i] <= j < end[i] else 0
encoded = duration_features^T @ alignment  -> [1,640,F]
asr     = text_encoded @ alignment         -> [1,512,F]
```

Alignment 放 Host 后会彻底消化 T 轴，避免 F 图重新携带固定 T。

### 4.3 frame_acoustic（F 域）

包含 Shared Prosody BiLSTM、F0 Predictor、Noise Predictor、Decoder、F0 上采样
和 9 路谐波增量构造，共 1 个逻辑双向 LSTM。F0 保持官方 AdaIN 语义。

```text
输入：
encoded           [1,640,F]
asr               [1,512,F]
style             [1,256]
valid_frames      [1]
reverse_indices   [F]         ONNX INT64 / HMONNX INT32

输出：
decoder_feature   [1,512,2F]
f0                [1,2F]
phase_increments  [1,2F,9]
```

### 4.4 phase_core（CPU FP32）

```text
phase_increments[1,2F,9]
  -> CumSum(FP32)
  -> * 2π
  -> 300x linear interpolation
  -> Sin * 0.1
  -> sine[1,600F,9]
```

F0 上采样、谐波序号相乘、除以 24 kHz、remainder 和初始相位注入已经在
`frame_acoustic` 完成。CPU 不执行 F0 Predictor、SourceMerge 或 STFT。

### 4.5 generator_istft（F 域）

```text
输入：
decoder_feature   [1,512,2F]
sine              [1,600F,9]
f0                [1,2F]
style             [1,256]
valid_frames      [1]

输出：
waveform          [1,600F]
```

图内完成 voiced/unvoiced 门控、固定噪声、SourceMerge、长度感知 Harmonic STFT、
Generator 和静态 iSTFT。reflect padding 按真实 valid length 计算，不把 bucket 尾部
的 0 当作真实音频反射。

## 5. 双向 LSTM

Kokoro 主路径使用一个 `num_directions=2` 的逻辑双向 LSTM，不用两个单向节点
拼接。有效区契约固定为 prefix reverse index：

```text
L=5, capacity=8
reverse_indices = [4,3,2,1,0,5,6,7]
```

索引只反转有效前缀，padding 后缀留在原位。这个置换满足 `P^-1=P`，所以反向分支
输入前和输出恢复时可以使用同一组 Gather；整条静态序列全反转会把 padding 搬到
前面并污染反向 hidden state，因此不能使用。

xhquant 的 QLSTM 内部由 QLinear、Add、Mul、Sigmoid、Tanh、Split 和 Gather 组成。
LSTM 使用与 Linear 相同的量化配置传递：一个 quant type 同时控制 X/H 激活以及 W、R
两组权重。PTQ 固化时，W 和 R 都执行 `weight_static_quant()`，生成
`qweight + scale_or_exp`。native 和 decomposed 只是导出表示不同，使用的是同一份量化
权重；native 大算子不再携带 FP16 W/R initializer。

同一套量化实现支持两种导出表示：

| 形态 | torch.export 结果 | 用途 |
|---|---|---|
| native | 保留带量化 W/R 输入的 `ai.houmo.xh2a::LSTM` | 编译器原生支持后的正式小图 |
| decomposed | 展开为量化 Linear、门控、状态更新和 Gather | 默认兼容路径 |

选择方式：

```text
Python API: decompose_lstm=False / True
环境变量:  XHQUANT_LSTM_EXPORT_MODE=native|decomposed
```

Python 参数优先级最高；未传参数、也未设置环境变量时默认导出 decomposed。需要 native
时显式传 `decompose_lstm=False`，或设置环境变量为 `native`。

展开参考 RoIAlign 的 torch.export 机制，在导出阶段决定，不再先生成 native HMONNX
再调用 `decompose_hmonnx_lstm()` 手工改最终文件。T 图 native 应有 5 个 BiLSTM，
`frame_acoustic` 应有 1 个；`generator_istft` 不含 LSTM。decomposed 节点数随静态
序列长度线性增长，大 bucket 文件更大、编译更慢，失败会按 bucket 记录，不伪装成功。

`torch_onnx_internal_optimize` 必须保持 `true`。T32/F120 的关闭实验中，native
数值不变，但两个 decomposed 图都残留 `pkg.onnxscript.torch_lib` function domain，
HMONNX runtime 不能解析。这里启用的是 PyTorch exporter 将 function 正规化为标准
节点的必要步骤；Kokoro 自己的 post-export graph rewrite 仍为 0。

## 6. Merak 导出

主配置：

`configs_merak/workflows/xh2a/other_models/kokoro/kokoro_xh2a_bucketed_w16.yaml`

完整四档导出：

```bash
PYTHONPATH=. python examples_merak/audio/kokoro/kokoro_workflow.py \
  --model-dir work_dirs/model_assets/kokoro \
  --output-dir work_dirs/kokoro_merak/bucketed_w16
```

选择一组配对 T/F 做开发烟测：

```bash
PYTHONPATH=. python examples_merak/audio/kokoro/kokoro_workflow.py \
  --model-dir work_dirs/model_assets/kokoro \
  --output-dir work_dirs/kokoro_merak/smoke_t32_f160 \
  --token-bucket 32 \
  --audio-seconds-bucket 4 \
  --lstm-variants native decomposed
```

只导出 FP32 ONNX 可加 `--onnx-only`；导出一组后生成节点 golden 可加
`--dump-golden`。golden 会为命中的 T/F bucket 分别保存 native/decomposed 输入，
CPU phase_core 保存 FP32 ONNX 输入输出，不遍历 T×F。

分片导出时，用 `--token-bucket` 或 `--audio-seconds-bucket` 选择对应 route 子集，完成
后使用：

```bash
PYTHONPATH=. python examples_merak/audio/kokoro/merge_bucket_exports.py \
  --shard-dir work_dirs/kokoro_merak/shards/a \
  --shard-dir work_dirs/kokoro_merak/shards/b \
  --config-path configs_merak/workflows/xh2a/other_models/kokoro/kokoro_xh2a_bucketed_w16.yaml \
  --output-dir work_dirs/kokoro_merak/bucketed_w16
```

合并器要求资产、量化、phase boundary、mask 和 LSTM 契约一致，检查四个 T 和四个
F bucket 是否完整，并重新验证 native/decomposed LSTM 节点。相同且一致的重复 bucket
可去重；内容冲突会直接报错。

## 7. 产物布局

```text
work_dirs/kokoro_merak/bucketed_w16/
├── export_meta_info.json
├── assets/voices/zf_001.npy                    Host NumPy 常量
├── onnx/
│   ├── text_duration/t0032 ... t0256/          4 份
│   ├── frame_acoustic/f0160 ... f1280/         4 份
│   ├── phase_core/f0160 ... f1280/             4 份 CPU FP32
│   └── generator_istft/f0160 ... f1280/        4 份
└── hmonnx/
    ├── text_duration/native|decomposed/tXXXX/
    ├── frame_acoustic/native|decomposed/fXXXX/
    └── generator_istft/fXXXX/
```

每张含 LSTM 的 FP32 导出同时保存标准 ONNX companion，供 ONNX Runtime 做语义对照；
部署 ONNX 的 LSTM 第五输入改为外部 `reverse_indices`。大权重可能使用同目录 external
data，复制产物时不能只复制 `.onnx`。

## 8. Inference demo

```bash
PYTHONPATH=. python examples_merak/audio/kokoro/bucketed_inference_demo.py \
  --export-dir work_dirs/kokoro_merak/bucketed_w16 \
  --backend hmonnx \
  --lstm-variant native \
  --output-wav work_dirs/kokoro_merak/bucketed_w16/native_demo.wav
```

查看 bucket：

```bash
PYTHONPATH=. python examples_merak/audio/kokoro/bucketed_inference_demo.py \
  --export-dir work_dirs/kokoro_merak/bucketed_w16 \
  --list-buckets
```

遍历四个配对 route：

```bash
PYTHONPATH=. python examples_merak/audio/kokoro/bucketed_inference_demo.py \
  --export-dir work_dirs/kokoro_merak/bucketed_w16 \
  --backend hmonnx \
  --lstm-variant native \
  --exercise-all-buckets \
  --route-report work_dirs/kokoro_merak/bucketed_w16/native_all_buckets.json
```

也可以用 `--token-bucket 128` 或 `--audio-seconds 16` 强制 T128/F640。生产模式总是
从最小合法配对档位开始；duration 溢出当前 F 容量时才重跑下一档 T 图。
新产物自带完整 `zf_001.npy` voice pack，demo 不依赖 PyTorch 源模型目录；只有读取旧
产物时才需要用 `--model-dir` 回退到上游 `.pt` voice 文件。

## 9. 当前精度基线

固定中文用例包含 28 个有效 token、110 个声学帧、66,000 个有效采样点，生产路由为
T32/F160。以下精度表来自同一用例的早期 F120 边界实验，用于说明 CPU phase 的必要性：

| 对比 | waveform cosine | max abs | MAE | STFT magnitude cosine | log-spectrum cosine |
|---|---:|---:|---:|---:|---:|
| PyTorch vs FP32 ONNX 三图链 | 0.999463 | 0.048433 | 0.000494 | 0.999943 | 0.999872 |
| FP32 ONNX vs 相位全 NPU HMONNX | 0.420311 | 0.638173 | 0.028927 | 0.780179 | 0.846850 |
| FP32 ONNX vs CPU phase HMONNX | 0.727428 | 0.623035 | 0.014746 | 0.995189 | 0.989713 |

CPU phase 不能消除上游量化 F0 已经产生的频率差异，所以逐点 waveform cosine 仍受
相位漂移影响；但它避免了 W16A16 CumSum 自身继续放大误差，频谱主体明显恢复。
发布 gate 必须同时检查 duration、F0、phase increments、waveform、STFT magnitude、
log spectrum、ASR、说话人相似度和人工听测，不能只看一个 cosine。

## 10. 全 bucket 导出与路由实测

最终 `bucketed_w16` 产物只包含 T32/F160、T64/F320、T128/F640、T256/F1280
四个配对档位。native 图保留双向 LSTM 大算子；decomposed 图的 LSTM 数为 0，节点数
随 T/F 线性增长。28 token 用例自动路由到 T32/F160，输出 110 帧、66,000 个采样点
（2.75 秒）。`--exercise-all-routes` 会逐一验证四个 route，而不是组合出 16 条路径。
报告与 WAV 保存在：

```text
work_dirs/kokoro_merak/bucketed_w16/
├── ort_all_buckets.json
├── native_all_buckets.json
├── decomposed_all_buckets.json
├── ort_demo.wav
├── native_demo.wav
└── decomposed_demo.wav
```

PCM16 WAV 上，ORT 对 native 的 STFT magnitude cosine 为 0.995189，ORT 对 decomposed
为 0.994514；后者逐点 waveform cosine 为 0.267819，但 log-spectrum cosine 仍为
0.986546。这是上游量化 F0 微差经长时间相位积分后的漂移，不能据此把频谱正确的声音
误判为整体失真。`--exercise-all-routes` 会在每个 route 结束后释放 runner cache，避免
验证四档时把所有 decomposed 大图同时常驻 GPU；生产单请求路径仍按需缓存已选择的图。

## 11. 回归

```bash
PYTHONPATH=. python -m pytest -q tests/kokoro/test_kokoro_support.py

cd "${XHQUANT_REPO}"
PYTHONPATH=. python -m pytest -q tests/testing/xh2a/ops/test_lstm.py
```

核心代码：

```text
xhmodel_merak/xh_other_model/models/kokoro/
├── buckets.py             # 四档配对 T/F 预设
├── independent_split.py   # 分离的 T/F NPU 图和 CPU phase_core wrapper
├── bucketed_runtime.py    # duration 驱动的配对档位回退
├── graph.py               # 静态 wrapper、BiLSTM、mask
├── static_dsp.py          # Harmonic STFT 和静态 iSTFT
└── workflow.py            # Merak 导出、HMONNX、校验和 metadata
```
