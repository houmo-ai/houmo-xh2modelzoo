# Zipformer CTC 流式中文 ASR

基于 [sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13](https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models) 的流式中文语音识别模型量化示例。

- 模型：Zipformer-M CTC，~64M 参数
- 输入：80 维 fbank，chunk_length=45 帧（流式推理）
- 输出：CTC log_probs（2000 tokens）

## 目录结构

```
├── README.md            # 本文件
├── ptq_cn.py            # 量化导出脚本
├── eval_final_v2.py     # 精度评测脚本（sherpa / ORT FP32 / HMONNX 对比）
├── data/cn/             # 测试音频和参考文本
│   ├── 0.wav / 1.wav / 8k.wav
│   └── trans.txt
└── models/              # 原始 ONNX 模型（需自行下载）
```

## 1. 下载模型

```bash
cd examples/audio/HMSW-3885_zipformer_ctc
mkdir -p models && cd models
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
tar xf sherpa-onnx-streaming-zipformer-ctc-multi-zh-hans-2023-12-13.tar.bz2
cd ..
```

## 2. 量化导出 HMONNX

```bash
# w8a8h1_sefp（默认）
python examples/audio/HMSW-3885_zipformer_ctc/ptq_cn.py \
    --quant-type w8a8h1_sefp \
    --output_dir work_dirs/w8a8

# w16a16h1_sefp
python examples/audio/HMSW-3885_zipformer_ctc/ptq_cn.py \
    --quant-type w16a16h1_sefp \
    --output_dir work_dirs/w16a16
```

## 3. GPU 仿真评测

```bash
python examples/audio/HMSW-3885_zipformer_ctc/eval_final_v2.py \
    --hmonnx-w16a16 work_dirs/w16a16/ctc-epoch-20-avg-1-chunk-16-left-128_sim_XH2a.onnx \
    --hmonnx-w8a8   work_dirs/w8a8/ctc-epoch-20-avg-1-chunk-16-left-128_sim_XH2a.onnx
```

## 4. 精度对比

| 方案 | 0.wav | 1.wav | 8k.wav | Corpus CER |
|------|-------|-------|--------|------------|
| sherpa-onnx (FP32 基线) | 0.00% | 0.00% | 0.00% | **0.00%** |
| ORT FP32 (流式复现) | 0.00% | 0.00% | 0.00% | **0.00%** |
| HMONNX w16a16h1_sefp | 0.00% | 4.17% | 0.00% | **1.47%** |
| HMONNX w8a8h1_sefp | 0.00% | 4.17% | 0.00% | **1.47%** |

示例识别结果：

| 音频 | 参考文本 | HMONNX w8a8 输出 |
|------|---------|-----------------|
| 0.wav | 对我做了介绍那么我想说的是大家如果对我的研究感兴趣 | 对我做了介绍那么我想说的是大家如果对我的研究感兴趣 ✓ |
| 1.wav | 重点想谈三个问题首先就是这一轮全球金融动荡的表现 | 重点**呢**想谈三个问题首先就是这一轮全球金融动荡的表现 |
| 8k.wav | 深度的分析这一次全球金融动荡背后的根源 | 深度的分析这一次全球金融动荡背后的根源 ✓ |

## 5. 关键实现备注

- **流式推理参数**：`chunk_length=45, chunk_shift=32, tail_pad=45`
- **Fbank 特征**：`dither=0, snip_edges=False, high_freq=-400`（匹配 sherpa-onnx）
- **BPE 解码**：需处理 `<0xHH>` byte fallback 序列
- **Softplus 融合**：原始 ONNX 中 `nn.Softplus` 被分解为 6 个算子（Sub→Exp→Add→Log→Equal→Where），在 FP16 下 Exp 溢出。xhquant 的 `FuseSoftplusTransformer` 自动融合为标准 Softplus 算子，使用 LUT 查表避免溢出
