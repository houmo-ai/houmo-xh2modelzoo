# MeloTTS Chinese+English 44.1 kHz（XH2a/M50）

本例把 sherpa-onnx 发布的 `vits-melo-tts-zh_en/model.onnx` 对应到官方
MeloTTS PyTorch checkpoint，并按严格数值对齐方案拆成：

```text
CPU 文本前端
  -> M50 Graph 1: Text Encoder + Duration Predictor
  -> CPU duration/path/noise bridge
  -> M50 Graph 2: inverse Flow + mask-aware Generator
  -> CPU 按有效长度裁剪
  -> 44.1 kHz waveform
```

不是简单按原 ONNX 的某个节点机械切开。原始发布 ONNX 把文本编码、时长、
路径展开、随机 latent、inverse flow 和 waveform generator 混在一张动态图
里；变长 shape、`Ceil`、路径构造和随机数不适合 M50 静态图。这里从官方
PyTorch 权重重新导出两张定长计算图，并逐层处理 padding mask，避免固定
`Tmax` 的尾部通过卷积和 speaker condition 反向污染有效音频。

## 1. 开源来源与身份

- 官方工程：[myshell-ai/MeloTTS](https://github.com/myshell-ai/MeloTTS)
- 固定源码 commit：
  `209145371cff8fc3bd60d7be902ea69cbdb7965a`
- 官方安装文档：[docs/install.md](https://github.com/myshell-ai/MeloTTS/blob/main/docs/install.md)
- sherpa-onnx 发布模型：
  `vits-melo-tts-zh_en/model.onnx`
- 发布模型 SHA256：
  `bf30582eb1b012250a35b1a4a80e7dfbcf8485e7bb9de0d95efbbeef0e4ad86d`
- ZH `config.json` SHA256：
  `d58b5acdab89ad2bbd65325affab309ae3cb964834b02f9a60587474e81c8bb9`
- ZH `checkpoint.pth` SHA256：
  `a74e9eadffff065c75eb6dfa040efa72cad23e72cfea70d39190bc174fb97093`

客户 `model.onnx` 与上述 sherpa-onnx 发布文件逐字节相同。不能把其他语言、
speaker 或 MeloTTS 版本的 config/checkpoint 混用。

## 2. 准备官方资产

运行本例提供的固定版本下载脚本：

```bash
bash examples_merak/audio/melotts/download_assets.sh \
  work_dirs/model_assets/melotts
```

得到：

```text
<melotts-model-dir>/
├── source/
│   └── MeloTTS/                         # 固定 commit
├── melotts_weights/
│   ├── config.json                      # 官方 ZH config
│   └── checkpoint.pth                   # 官方 ZH checkpoint
├── downloads/
│   └── vits-melo-tts-zh_en.tar.bz2
└── packages/
    └── vits-melo-tts-zh_en/
        ├── model.onnx                   # 客户整图对应的发布模型
        ├── lexicon.txt
        ├── tokens.txt
        ├── dict/
        └── *.fst
```

脚本只在指定 `<melotts-model-dir>` 下创建资产，并对 config、checkpoint、
archive 和 release ONNX 做 SHA256 校验。工作流的 `--model-dir` 指向这一
根目录。

## 3. 原始整图数据流

原 `model.onnx` 的主要输入是：

| 输入 | 原始 shape/type | 含义 |
|---|---|---|
| `x` | `[1,L]` INT64 | phoneme/token id，含 blank |
| `x_lengths` | `[1]` INT64 | 有效文本长度 |
| `tones` | `[1,L]` INT64 | 与 token 对齐的 tone id |
| `sid` | `[1]` INT64 | speaker id |
| `noise_scale` | `[1]` FP32 | acoustic latent 随机强度 |
| `length_scale` | `[1]` FP32 | 语速/时长缩放 |
| `noise_scale_w` | `[1]` FP32 | stochastic duration 参数 |

整图内部依次执行：

1. token、tone、language、speaker embedding；
2. Text Encoder 产生每个文本位置的 acoustic mean/log-scale；
3. Duration Predictor 产生 `logw`；
4. `ceil(exp(logw) × text_mask × length_scale)` 得到整数时长；
5. 按时长展开 token latent，构造单调对齐/path；
6. 注入 acoustic noise；
7. 逆向 normalizing flow；
8. 多级 ConvTranspose1d/ResBlock Generator；
9. 输出 `[1,1,T×512]` 的 44.1 kHz 音频。

这里 `T` 是 acoustic frame 数，不是文本长度。`hop_length=512`，所以每个
acoustic frame 对应 512 个 waveform sample。

## 4. 拆图原则

本例默认：

- batch `B=1`；
- 文本桶 `Lmax=32`；
- acoustic 桶 `Tmax=64`；
- speaker id `1`；
- Chinese+English language id 规则：blank 位置 0，真实音素位置 3；
- `noise_scale=0`，建立可复现的严格精度基线；
- 公开整数输入用 INT32，图内显式转为 PyTorch embedding 所需 INT64。

### 4.1 Graph 1：Encoder + deterministic Duration Predictor

输入：

| 输入 | 固定 shape | 类型 | 含义 |
|---|---|---|---|
| `x` | `[1,32]` | INT32 | 右侧补 0 的 token |
| `x_lengths` | `[1]` | INT32 | 实际有效 token 数 `L` |
| `tones` | `[1,32]` | INT32 | 右侧补 0 的 tone |
| `sid` | `[1]` | INT32 | speaker id |

输出：

| 输出 | 固定 shape | 含义 |
|---|---|---|
| `m_p` | `[1,192,32]` | 每个文本位置的 acoustic prior mean |
| `logs_p` | `[1,192,32]` | 每个文本位置的 log scale |
| `logw` | `[1,1,32]` | deterministic duration log prediction |
| `x_mask` | `[1,1,32]` | 前 `L` 个 1、padding 为 0 |
| `g` | `[1,256,1]` | speaker embedding |

为什么 Duration Predictor 放在 encoder 图：它是固定 `Lmax` 的密集神经网络，
适合 NPU；而从 `logw` 到整数 `T` 的 `Exp/Ceil/Repeat` 会改变 tensor shape，
放 CPU 更直接。

### 4.2 CPU bridge

bridge 必须按下列顺序和 FP32 规则执行：

```text
duration_f = exp(logw.float()) * x_mask.float() * length_scale
durations  = ceil(duration_f).int64()             # [32]
T          = max(sum(durations), 1)
token_idx  = repeat_interleave(arange(32), durations)
z_valid    = m_p[:,:,token_idx]
             + epsilon * exp(logs_p[:,:,token_idx]) * noise_scale
z_p        = right_pad(z_valid, Tmax)              # [1,192,64]
y_mask     = [1] * T + [0] * (Tmax-T)             # [1,1,64]
```

若 `T > Tmax`，不能截断后继续合成；应拆句、拒绝，或导出更大的 decoder
bucket。CPU bridge 不构造原整图的 `L×T` 二维 path，直接用
`repeat_interleave/index_select`，避免二次方内存。

`noise_scale=0` 时 epsilon 全零，PyTorch、ONNX 和 HMONNX 可以严格复现。
若产品需要随机音色细节，CPU 用固定 seed 生成 FP32 noise，并重新执行主观
MOS/ABX 测试。

### 4.3 Graph 2：inverse Flow + mask-aware Generator

输入：

| 输入 | 固定 shape | 类型 | 含义 |
|---|---|---|---|
| `z_p` | `[1,192,64]` | FP16/FP32 | 已展开并右补 0 的 latent |
| `y_mask` | `[1,1,64]` | FP16/FP32 | acoustic 有效区 |
| `g` | `[1,256,1]` | FP16/FP32 | speaker embedding |

输出：

| 输出 | 固定 shape | 含义 |
|---|---|---|
| `y` | `[1,1,32768]` | `64×512` 的 padded 44.1 kHz waveform |

CPU 最终只保留前 `T×512` 个样本。

## 5. 为什么 decoder 必须每一级 mask

只在输入 `z_p` 末尾补 0 不够。官方 Generator 包含 speaker condition、
有 bias 的卷积、ConvTranspose1d 和多层带 padding 的 ResBlock。padding
区域会在中间层变成非零，再沿卷积感受野泄漏回有效区边界，使“固定
`Tmax` 后裁剪”的音频与动态长度 PyTorch 不一致。

`MaskAwareGenerator` 保留官方每一层和全部权重，但在这些位置乘 mask：

1. `conv_pre + speaker condition` 后；
2. 每次 ConvTranspose1d 后，把 mask 按同样 stride 上采样；
3. 每个 ResBlock 内已有的 `x_mask`；
4. 多分支 ResBlock 求平均后；
5. `conv_post` 前后和 `tanh` 后。

因此 padded tail 严格为 0，且 padding 不会反向污染有效 waveform。这是
“严格对齐方案 1”的关键，不是普通 encode/decode 切图。

## 6. modelzoo 侧静态化

实现位于
`xhmodel_merak/xh_other_model/models/melotts/graph.py`：

1. 用官方 `SynthesizerTrn` 类和 config 实例化模型；
2. strict 加载官方 checkpoint；
3. 构造固定 `Lmax/Tmax` wrapper；
4. PyTorch 导出两张 opset 17 ONNX；
5. 把公开整数输入固定为 INT32，图内显式 Cast；
6. 对固定 shape ONNX 执行 onnxsim 三轮等价检查；
7. 常量折叠 shape 子图，使 Unsqueeze/Slice/Reshape 参数成为 initializer；
8. 替换残留静态 `ConstantOfShape`；
9. 固定公开输出 shape 并写入 mask/duration metadata；
10. 用 ONNX checker 检查；
11. 对 `L=3/5/9` 比较动态 PyTorch、padded PyTorch、官方发布 ONNX 和
    双静态 ONNX。

这些兼容处理全部发生在 xh2modelzoo 导出的 ONNX 上，不修改 xhquant。

本仓导出时的严格随机用例最大误差：

| `L` | `T` | samples | 所有严格对齐项最大 abs |
|---:|---:|---:|---:|
| 3 | 20 | 10240 | `7.15e-7` |
| 5 | 30 | 15360 | `1.91e-6` |
| 9 | 52 | 26624 | `1.59e-6` |

三组所有检查的总最大误差为 `1.90735e-6`，低于 `2e-5` 门限。

## 7. Merak 结构

```text
configs_merak/workflows/xh2a/other_models/melotts/
└── melotts_xh2a_w16_l32_t64.yaml

xhmodel_merak/xh_other_model/models/melotts/
├── model.py       # register_other_model
├── assets.py      # 统一发现源码/config/checkpoint/release package
├── modeling.py    # 严格双图 wrapper 和 PyTorch bridge
├── graph.py       # 两图导出、静态简化、严格等价检查
├── workflow.py    # quant/export/dump_golden
└── runtime.py     # ORT/HMONNX runner 和 CPU bridge

examples_merak/audio/melotts/
├── download_assets.sh
├── melotts_workflow.py
├── real_text_eval.py
└── README.md
```

`quant()`、`export()`、`dump_golden()` 严格分离。顶层
`export_meta_info.json` 记录官方源码 commit、三类源文件 SHA、两个
ONNX/HMONNX SHA、静态桶、量化配置和 PyTorch 严格校验结果。

## 8. 环境

先按 MeloTTS 官方文档安装源码依赖，再安装 Merak/XHQuant 运行依赖：

```bash
conda activate <merak-env>
pip install -e <melotts-model-dir>/source/MeloTTS
pip install onnx onnxruntime onnxsim jieba soundfile
export PYTHONPATH=.
```

本例从 `melo.models` 直接加载网络，不要求 BERT 文本栈参与图导出；真实
文本精度脚本使用发布包 lexicon/tokens 和 jieba 复现 sherpa-onnx 中文前端。

## 9. 导出两张 XH2a W16 HMONNX

```bash
python examples_merak/audio/melotts/melotts_workflow.py \
  --model-dir <melotts-model-dir> \
  --output-dir work_dirs/melotts_merak/export_xh2a_w16_l32_t64 \
  --device cuda:0 \
  --dump-golden
```

已有输出目录默认保留。明确重建时增加 `--overwrite`，只删除指定
`--output-dir`。

产物：

```text
<output-dir>/
├── melotts_xh2a_w16_l32_t64.yaml
├── export_meta_info.json
├── onnx/
│   ├── melotts_encoder_dp_b1_l32.onnx
│   └── melotts_flow_masked_decoder_b1_t64.onnx
├── hmonnx/
│   ├── melotts_encoder_dp_b1_l32_XH2a_w16a16_sefp.onnx
│   └── melotts_flow_masked_decoder_b1_t64_XH2a_w16a16_sefp.onnx
└── golden/
    ├── encoder/
    ├── decoder/
    └── manifest.json
```

## 10. 真实文本精度

```bash
python examples_merak/audio/melotts/real_text_eval.py \
  --model-dir <melotts-model-dir> \
  --export-dir work_dirs/melotts_merak/export_xh2a_w16_l32_t64 \
  --text "你好。"
```

脚本输出四份可听 WAV，并比较：

- 官方动态 PyTorch；
- sherpa-onnx 官方发布整图；
- 双静态 ONNX + CPU bridge；
- 两张 XH2a W16 HMONNX + CPU bridge。

真实结果：

| 项目 | 结果 |
|---|---:|
| 文本 | `你好。` |
| token 长度 `L` | 11 |
| acoustic 长度 `T` | 51 |
| 有效 samples | 26112 |
| 时长向量 ONNX/HMONNX | 完全一致 |
| release ONNX vs PyTorch cosine | `0.999999999983092` |
| static ONNX vs PyTorch cosine | `0.999999999990734` |
| W16 HMONNX vs PyTorch max abs | `0.00219719` |
| W16 HMONNX vs PyTorch cosine | `0.9994688902` |
| W16 HMONNX vs PyTorch SNR | `29.7026 dB` |
| W16 active-band log spectral MAE | `0.553866 dB` |

全频谱 MAE 会被静音/极低能量 bin 的 dB 放大，因此同时报告参考谱峰下
60 dB 内的 active-band 指标。量产验收还必须加入多说话人、多文本长度、
中英混合、数字/标点、MOS/ABX 和听感边界检查。

## 11. 推荐量化

首发两张图都推荐 `w16a16_sefp`：

```yaml
components:
  encoder:
    quant_type: w16a16_sefp
  decoder:
    quant_type: w16a16_sefp
```

原因：

- duration 在 `Ceil` 附近很敏感，一个 token 变化会改变 `T` 和全部后续
  waveform 对齐；
- Generator 的多级上采样会放大量化误差；
- 当前 W16 已完成严格 PyTorch、发布整图、双静态图、双 HMONNX 和真实
  WAV 闭环；
- 客户初始模型无精度数据，不能直接以 W8 转换成功代替音质验收。

若后续尝试 W8，至少分别评估 encoder duration exact rate、decoder
waveform/spectral、MOS/ABX、爆音/尾音、不同 `L/T` 桶和真实 M50 性能。
可以独立探索 encoder W8 + decoder W16，但在 duration exact rate 和业务
听感通过前不推荐。

## 12. M50 运行约束

- CPU 前端的 normalization、lexicon、token、tone、blank 插入必须与
  checkpoint 配套；
- `L` 包含 blank，不能拿原始中文字数与 `Lmax=32` 比；
- Graph 1 的 padding 必须右对齐为 0，`x_lengths` 填实际 `L`；
- CPU bridge 的 `Exp/Ceil` 固定用 FP32，不要先转 FP16；
- `T>Tmax` 必须分句或选大桶，不能静默截断；
- `y_mask` 必须是连续前 1 后 0；
- decoder 输出只保留前 `T×512`；
- 44.1 kHz 样本不能按 16 kHz 播放；
- 若改 `speaker_id/language_id/noise_scale/length_scale`，必须重新验证；
- 推荐按多个 `Lmax/Tmax` bucket 部署，CPU 选择能容纳输入的最小桶。
