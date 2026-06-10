# CAM++ — ONNX / HMONNX 导出

```text
feats (B,T,80) -> embedding (B,192)
```

本目录支持三种产物，固定长度由命令行 `--fixed-t` 控制，默认是 `1000`；HMONNX 默认量化配置是 `w8a8_sefp`：

1. `onnx/campplus.onnx`：动态 `T`，适合 ONNXRuntime。
2. `onnx/campplus_${fixed_t}.onnx`：固定 `(1,fixed_t,80)`，适合原 HMONNX 定长路径。
3. `onnx/campplus_masked_${fixed_t}.onnx`：固定 shape 但带 mask，padding 不参与 embedding 计算，适合变长语音落到固定 NPU 模型。

## 文件说明

| 文件 | 作用 |
|------|------|
| `campplus_components.py` | CAM++ 组件。默认不传 mask 时保持原逻辑；传 mask 时使用 masked pooling |
| `campplus_model.py` | `CAMPPlus` 主模型，接口为 `forward(feats, feat_mask=None, mask=None)` |
| `campplus_export_onnx.py` | 导出普通动态 ONNX 和普通 fixed-t ONNX |
| `campplus_export_hmonnx.py` | 普通 fixed-t ONNX 转 HMONNX |
| `campplus_export_masked.py` | masked 路径统一入口，支持 `--stage onnx/hmonnx/test/all` |

实际常用脚本只有三个：`campplus_export_onnx.py`、`campplus_export_hmonnx.py`、`campplus_export_masked.py`。

权重路径在脚本顶部：

```text
/data01/nfs_shared/ASR_TTS/CAM++/campplus_cn_common.bin
```

## 环境

```bash
PY=/data01/home/she.gao/miniconda3/envs/xhquant/bin/python
cd /data01/home/she.gao/xh2modelzoo_new/examples/audio/campplus
```

## 固定长度参数

所有 fixed-shape 导出脚本都支持 `--fixed-t`：

```bash
--fixed-t 1000   # 默认值
--fixed-t 1500   # 例如导出 1500 帧定长模型
```

`mask` 的长度永远是 `(fixed_t + 1) // 2`，因为 CAMPPlus 第一个 TDNN 是 `stride=2`。

## 路径和量化参数

脚本里的模型路径都可以通过 argparse 覆盖，同时保留默认值：

```bash
# PyTorch 权重路径
--bin-path /data01/nfs_shared/ASR_TTS/CAM++/campplus_cn_common.bin

# ONNX 输入/输出
--dynamic-onnx onnx/campplus.onnx
--fixed-onnx onnx/campplus_1000.onnx
--masked-onnx onnx/campplus_masked_1000.onnx
--model-path onnx/campplus_masked_1000.onnx
--simplified-path onnx/campplus_masked_simplify.onnx

# HMONNX 输出
--output-root campplus
--output-file campplus/campplus_masked/prefill/hmquant_xh2_campplus_masked_w8a8_sefp_1000.onnx
--golden-dir campplus/campplus_masked/prefill/step_0

# 默认量化配置
--quant-type w8a8_sefp
```

## 命令一：导出普通 ONNX

```bash
$PY campplus_export_onnx.py --fixed-t 1000
```

产物：

```text
onnx/campplus.onnx       # 动态 batch/time，输入 feats=(B,T,80)
onnx/campplus_1000.onnx  # 固定 shape，输入 feats=(1,1000,80)；如果 --fixed-t 1500，则为 campplus_1500.onnx
```

脚本会自动验证：

```text
dynamic ONNX: T=137/200/311，Torch vs ONNXRuntime
fixed ONNX  : T=1000，Torch vs ONNXRuntime
```

## 命令二：普通 fixed-t HMONNX

```bash
$PY campplus_export_hmonnx.py --fixed-t 1000 --quant-type w8a8_sefp
```

输入：

```text
onnx/campplus_1000.onnx
```

产物：

```text
campplus/prefill/hmquant_xh2_campplus_w8a8_sefp_1000.onnx
campplus/prefill/step_0/
```

这个路径只有一个输入 `feats=(1,fixed_t,80)`。如果语音不足 fixed_t 帧只能 padding，padding 会参与统计池化，因此变长语音建议用下面的 masked 路径。

## 命令三：masked fixed-t ONNX + HMONNX

一条命令跑完整 masked 流程：

```bash
$PY campplus_export_masked.py --fixed-t 1000 --quant-type w8a8_sefp
```

只导出 masked ONNX：

```bash
$PY campplus_export_masked.py --stage onnx --fixed-t 1000
```

只做 masked HMONNX 转换：

```bash
$PY campplus_export_masked.py --stage hmonnx --fixed-t 1000 --quant-type w8a8_sefp
```

产物：

```text
onnx/campplus_masked_1000.onnx
onnx/campplus_masked_simplify.onnx
campplus/campplus_masked/prefill/hmquant_xh2_campplus_masked_w8a8_sefp_1000.onnx
campplus/campplus_masked/prefill/step_0/

# 如果 --fixed-t 1500，则产物会带 1500 后缀：
onnx/campplus_masked_1500.onnx
onnx/campplus_masked_1500_simplify.onnx
campplus/campplus_masked_1500/prefill/hmquant_xh2_campplus_masked_w8a8_sefp_1500.onnx
```

masked 模型固定 3 个输入：

| 输入 | shape | dtype | 含义 |
|------|-------|-------|------|
| `feats` | `(1,fixed_t,80)` | fp32 / fp16 | padding 或 crop 后的 fbank |
| `feat_mask` | `(1,1,fixed_t)` | fp32 / fp16 | fbank/head 时间维 mask |
| `mask` | `(1,1,ceil(fixed_t/2))` | fp32 / fp16 | 第一个 TDNN stride=2 后的时间维 mask |

脚本已验证：

```text
原长输入 vs padding 到 1000 后带 mask：T=137/200/311/999/1000，max|Δ|≈4e-6
Torch masked vs ONNXRuntime masked：max|Δ|≈7e-6
```

## 命令四：真实 wav 样例测试

`campplus_export_masked.py` 内置了旧 Cosyvoice3 目录里的两个真实短音频样例，用来验证动态 ONNX 原长输入、masked ONNX padding 输入、masked HMONNX padding 输入是否一致：

```bash
$PY campplus_export_masked.py --stage test --fixed-t 1000 --quant-type w8a8_sefp
```

默认测试音频：

```text
test_wavs/zero_shot_prompt.wav
test_wavs/xiaotian_chunk_000.wav
```

也可以传自己的音频，`--test-wav` 可以重复：

```bash
$PY campplus_export_masked.py \
  --stage test \
  --fixed-t 1000 \
  --test-wav /path/to/a.wav \
  --test-wav /path/to/b.wav
```

如果音频帧数超过 `fixed_t`，脚本会额外打印 `dynamic_crop` 对比。全长动态模型和 fixed-t masked 模型不同是正常的，因为 fixed-t 路径已经裁掉了后半段。

测试阶段默认查找这个 HMONNX：

```text
campplus/campplus_masked/prefill/hmquant_xh2_campplus_masked_w8a8_sefp_1000.onnx
```

如果文件存在，会额外打印：

```text
cos(masked_onnx, hmonnx)
hmonnx_vs_onnx_max_abs_diff
hmonnx_vs_onnx_mean_abs_diff
cos(dynamic_ref, hmonnx)
```

其中 `dynamic_ref` 的含义是：短音频用动态 ONNX 全长结果；长音频超过 `fixed_t` 时，用动态 ONNX 裁剪到 `fixed_t` 后的结果。这样对比对象和 fixed-shape HMONNX 的真实输入长度保持一致。

如果 HMONNX 文件还没生成，测试脚本会打印 `[hmonnx] skip`，仍然会完成动态 ONNX 和 masked ONNX 的对比。先执行下面命令即可生成 HMONNX：

```bash
$PY campplus_export_masked.py --stage hmonnx --fixed-t 1000 --quant-type w8a8_sefp
```

## 部署端前处理

前处理顺序要保持一致：

```text
wav -> 16k mono -> kaldi.fbank(num_mel_bins=80, dither=0, sample_frequency=16000)
    -> 按有效帧做逐句 CMN -> pad/crop 到 fixed_t -> 构造 feat_mask 和 mask
```

注意：CMN 要在原始有效帧上做，不要把 padding 的 0 算进去。

示例代码：

```python
import numpy as np

FIXED_T = 1000
FEAT_DIM = 80


def pad_or_crop_feats(feats, fixed_t=FIXED_T):
    # feats: (T, 80), after CMN
    valid_t = min(feats.shape[0], fixed_t)
    out = np.zeros((1, fixed_t, FEAT_DIM), dtype=np.float32)
    out[0, :valid_t, :] = feats[:valid_t].astype(np.float32)
    return out, valid_t


def make_campplus_masks(valid_t, fixed_t=FIXED_T):
    valid_t = min(int(valid_t), fixed_t)

    # head/FCM 使用原始 1000 帧时间分辨率
    feat_mask = np.zeros((1, 1, fixed_t), dtype=np.float32)
    feat_mask[:, :, :valid_t] = 1.0

    # CAMPPlus 的第一个 TDNN 是 stride=2，所以后续时间维是 ceil(T/2)
    mask_t = (fixed_t + 1) // 2       # 1000 -> 500
    valid_mask_t = (valid_t + 1) // 2
    mask = np.zeros((1, 1, mask_t), dtype=np.float32)
    mask[:, :, :valid_mask_t] = 1.0
    return feat_mask, mask


# fbank: (T,80)
# fbank = fbank - fbank.mean(axis=0, keepdims=True)
feats, valid_t = pad_or_crop_feats(fbank)
feat_mask, mask = make_campplus_masks(valid_t)

# ONNXRuntime 输入
inputs = {
    "feats": feats,
    "feat_mask": feat_mask,
    "mask": mask,
}
```

如果接 HMONNX golden/inference，一般把三路输入都转成 fp16：

```python
feats = torch.from_numpy(feats).to(torch.float16)
feat_mask = torch.from_numpy(feat_mask).to(torch.float16)
mask = torch.from_numpy(mask).to(torch.float16)
output = model.forward(feats, feat_mask, mask)
```

## 什么时候用哪个模型

| 场景 | 推荐模型 |
|------|----------|
| CPU / ONNXRuntime，输入天然变长 | `onnx/campplus.onnx` |
| NPU 固定 shape，能接受 padding 影响 | 普通 fixed-t HMONNX |
| NPU 固定 shape，要求 padding 不影响结果 | masked fixed-t HMONNX |
| 语音经常超过当前 fixed_t | 调大 `--fixed-t`、做多 bucket，或滑窗提 embedding 后融合 |
