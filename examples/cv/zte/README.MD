# 带loop的模型

## 1. 固定batch为32
    ```shell
        python -m onnxsim weights/zte/model_hexiaoxi.onnx weights/zte/model_hexiaoxi_sim.onnx --input-shape serving_default_lstm_input:0:32,9,13
    ```
## 2. 将loop模型拆开，lstm拆开
    ```shell
        python examples/cv/zte/convert_loop.py --in weights/zte/model_hexiaoxi_sim.onnx --out weights/zte/model_hexiaoxi_sim_unloop.onnx --loop_name while_loop --trip_count 9
    ```
## 3. onnxsim下
    ```shell 
        python -m onnxsim weights/zte/model_hexiaoxi_sim_unloop.onnx weights/zte/model_hexiaoxi_sim_unloop_sim.onnx --skip-optimization fuse_matmul_add_bias_into_gemm
    ```
## 4. 量化生成golden
    ```shell
        python examples/cv/zte/loop_model_quant.py
    ```

# model_timi_cjb

## 1. 固定src和src_key_padding_mask的batch_size为1
    ```shell
    python -m onnxsim weights/zte/model_timi_cjb.onnx weights/zte/model_timi_cjb_sim.onnx --input-shape src:1,512,64 src_key_padding_mask:1,512

    python -m onnxsim weights/zte/L3模型待测试项/L3精度测试用例/timi/model_timi.onnx weights/zte/L3模型待测试项/L3精度测试用例/timi/model_timi_sim.onnx --input-shape src:1,512,64 src_key_padding_mask:1,512
    ```
## 2. 量化生成golden
    ```shell
        python examples/cv/zte/model_timi_cjb_quant.py
    ```

# 带LSTM的模型

## 1. 固定batch为32
    ```shell
        python -m onnxsim weights/zte/model_v4_hxx.onnx weights/zte/model_v4_hxx_sim.onnx --input-shape lstm_input:32,9,13
    ```

# MLP模型

## 1. 固定batch

### 1.1 batch 1
    ```shell
        python -m onnxsim weights/zte/L3模型待测试项/L3并发测试用例/model_MLP.onnx weights/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch1.onnx --input-shape numeric:1,4
    ```

### 1.2 batch 16
    ```shell
        python -m onnxsim weights/zte/L3模型待测试项/L3并发测试用例/model_MLP.onnx weights/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch16.onnx --input-shape numeric:16,4
    ```

### 1.3 batch 64
    ```shell
        python -m onnxsim weights/zte/L3模型待测试项/L3并发测试用例/model_MLP.onnx weights/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch64.onnx --input-shape numeric:64,4
    ```

### 1.4 batch 128
    ```shell
        python -m onnxsim weights/zte/L3模型待测试项/L3并发测试用例/model_MLP.onnx weights/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch128.onnx --input-shape numeric:128,4
    ```

## 2. 量化生成golden
    ```shell
       hmquant2 export --onnx weights/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch1.onnx --quant-type w16a16_sefp --output-dir work_dirs/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch1_quant

       hmquant2 export --onnx weights/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch16.onnx --quant-type w16a16_sefp --output-dir work_dirs/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch16_quant

       hmquant2 export --onnx weights/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch64.onnx --quant-type w16a16_sefp --output-dir work_dirs/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch64_quant

       hmquant2 export --onnx weights/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch128.onnx --quant-type w16a16_sefp --output-dir work_dirs/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch128_quant
    ```

## 3. debug不同量化精度

使用 `hmquant2 debug` 对比不同量化类型下模型输出与 onnxruntime fp32 推理结果的精度差异。

### 3.1 w8a8
    ```shell
        hmquant2 debug --onnx weights/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch1.onnx --quant-type w8a8 --output-dir work_dirs/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch1_debug_w8a8
    ```

### 3.2 w8a16
    ```shell
        hmquant2 debug --onnx weights/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch1.onnx --quant-type w8a16 --output-dir work_dirs/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch1_debug_w8a16
    ```

### 3.3 w16a16
    ```shell
        hmquant2 debug --onnx weights/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch1.onnx --quant-type w16a16 --output-dir work_dirs/zte/L3模型待测试项/L3并发测试用例/model_MLP_batch1_debug_w16a16
    ```

### 3.4 精度对比结果（batch1）

| 量化类型 | Max AE | Max RE | MSE | Cos Similarity |
|---------|--------|--------|-----|----------------|
| w8a8 | 0.002660 | 0.000680 | 0.000007 | 1.000000 |
| w8a16 | 0.012662 | 0.003204 | 0.000160 | 1.000000 |
| w16a16 | 0.004762 | 0.001215 | 0.000023 | 1.000000 |

> 以上为 hmonnx vs onnxruntime 的 output 精度对比。三种量化类型 Cosine Similarity 均为 1.0，精度损失极小。其中 w8a8 精度最优，w16a16 次之，w8a16 误差最大。

# L3精度测试（timi / hxx_lstm）

用于重新量化对方最新提供的 L3 精度测试模型，基于对方输入跑 HMONNX 输出，并调用对方 compare 脚本生成精度报告。

> 说明：上面的 `model_timi_cjb` 是旧的 batch1 固定流程。本次 L3 精度测试使用的是 `<l3-test-root>/timi/model_timi.onnx` 和对方提供的 batch64 输入，已由一键脚本自动处理，不需要手工再执行 batch1 的 `onnxsim`。

下面统一用 `<l3-test-root>` 表示 `L3精度测试用例` 目录。这个目录不要求固定放在 `weights/zte` 下，只要把路径通过 `--l3-test-root` 传给脚本，或者设置环境变量 `ZTE_L3_TEST_ROOT` 即可。

## 1. 一键脚本

脚本路径：`examples/cv/zte/l3_precision_runner.py`

脚本会自动完成以下步骤：

1. `timi` 根据 `inference_results.jsonl` 的首条输入，将原始模型预处理成 batch64 静态 shape。
2. `hxx_lstm` 先把 `model_hxx_v4.onnx` 固定成 batch1，生成 `model_hxx_v4_batch1.onnx`。
3. 按给定量化配置导出 hmonnx。
4. 使用对方输入数据跑 HMONNX 推理，runtime 优先使用 `HMONNXGrapInference`，失败时回退到 `HMONNXInference`。
5. 调用对方 compare 脚本输出报告，并生成统一的 `summary.json` 和 `summary.md`。
6. 也支持按 `export / infer / compare` 三个 stage 分开执行，方便把 `infer` 替换成芯片推理。

## 2. 执行命令

`hxx_lstm` 的 batch1 模型不是对方直接提供的，需要先执行：

```shell
python -m onnxsim \
  <l3-test-root>/hxx_lstm/model_hxx_v4.onnx \
  <l3-test-root>/hxx_lstm/model_hxx_v4_batch1.onnx \
  --input-shape lstm_input:1,9,13
```

然后按需要选择以下两种方式。

### 2.1 本地全流程

显式传 `--l3-test-root`，一次性执行 `export infer compare`：

```shell
conda activate xhquant

python examples/cv/zte/l3_precision_runner.py \
  --stages export infer compare \
  --l3-test-root <l3-test-root> \
  --cases timi hxx_lstm \
  --quant-types w8a8_sefp w8a16_sefp w16a16_sefp \
  --output-root /data01/datasets/xh2a_model/qwen2_5_vl/zte/l3_precision_final
```

也可以先设置环境变量，再执行：

```shell
export ZTE_L3_TEST_ROOT=<l3-test-root>

conda activate xhquant

python examples/cv/zte/l3_precision_runner.py \
  --stages export infer compare \
  --cases timi hxx_lstm \
  --quant-types w8a8_sefp w8a16_sefp w16a16_sefp \
  --output-root work_dirs/zte/l3_precision
```

### 2.2 推荐给 FAE 的分阶段流程

先导出 hmonnx 和 manifest：

```shell
conda activate xhquant

python examples/cv/zte/l3_precision_runner.py \
  --stages export \
  --l3-test-root <l3-test-root> \
  --cases timi hxx_lstm \
  --quant-types w8a8_sefp w8a16_sefp w16a16_sefp \
  --output-root work_dirs/zte/l3_precision
```

如果要继续用本地 HMONNX 推理，再执行：

```shell
conda activate xhquant

python examples/cv/zte/l3_precision_runner.py \
  --stages infer \
  --l3-test-root <l3-test-root> \
  --cases timi hxx_lstm \
  --quant-types w8a8_sefp w8a16_sefp w16a16_sefp \
  --output-root work_dirs/zte/l3_precision
```

如果要替换成芯片推理，则不要执行 `infer`。直接读取 `<output-root>/run_manifest.json`，把芯片输出写到每个 quant 对应的 `expected_output_path`：

- `timi`：写成 `jsonl`
- `hxx_lstm`：写成 `xlsx`

格式建议：

- `timi`：每行至少包含 `iteration` 和 `output` 字段，`latency_ms` 可选。
- `hxx_lstm`：保存成 `xlsx`，sheet 中放纯数值输出列即可；可直接参考脚本生成的 `hmonnx_output.xlsx` 格式。

最后执行对比和 summary：

```shell
conda activate xhquant

python examples/cv/zte/l3_precision_runner.py \
  --stages compare \
  --l3-test-root <l3-test-root> \
  --cases timi hxx_lstm \
  --quant-types w8a8_sefp w8a16_sefp w16a16_sefp \
  --output-root work_dirs/zte/l3_precision
```

如果你的数据目录刚好就在仓库默认位置，脚本也会自动尝试搜索，此时可以省略 `--l3-test-root`：

```shell
conda activate xhquant

python examples/cv/zte/l3_precision_runner.py \
  --stages export infer compare \
  --cases timi hxx_lstm \
  --quant-types w8a8_sefp w8a16_sefp w16a16_sefp \
  --output-root work_dirs/zte/l3_precision
```

## 3. 输入与对比文件

目录约定：

- `<l3-test-root>/timi/...`
- `<l3-test-root>/hxx_lstm/...`

- `timi`
  - 模型：`<l3-test-root>/timi/model_timi.onnx`
  - 输入：`<l3-test-root>/timi/inference_results.jsonl`
  - 对方 GPU 输出：`<l3-test-root>/timi/64Batch_output_gpu.jsonl`
  - 对比脚本：`<l3-test-root>/timi/compare_inference_results.py`
- `hxx_lstm`
  - 原始模型：`<l3-test-root>/hxx_lstm/model_hxx_v4.onnx`
  - batch1 模型：`<l3-test-root>/hxx_lstm/model_hxx_v4_batch1.onnx`
  - 输入：`<l3-test-root>/hxx_lstm/model_io.xlsx`
  - 对方 GPU 输出：`<l3-test-root>/hxx_lstm/gpu_inference_output.xlsx`
  - 对比脚本：`<l3-test-root>/hxx_lstm/compare_model_outputs.py`

## 4. 输出目录

`<output-root>` 下的目录结构如下：

- `<output-root>/summary.json`
- `<output-root>/summary.md`
- `<output-root>/run_manifest.json`
- `<output-root>/timi/preprocessed/`
- `<output-root>/<case>/<quant_type>/hmonnx/`
- `<output-root>/<case>/<quant_type>/outputs/`
- `<output-root>/<case>/<quant_type>/compare/`
- `<output-root>/<case>/vendor_baseline/`

`run_manifest.json` 会记录：

- 每个 case/quant 对应的 `hmonnx_path`
- 外部推理需要回填的 `expected_output_path`
- `expected_output_format` 和 `expected_output_note`
- reference、vendor baseline、compare script 的实际路径

本次实际结果汇总在：

- `/data01/datasets/xh2a_model/qwen2_5_vl/zte/l3_precision_final/summary.json`
- `/data01/datasets/xh2a_model/qwen2_5_vl/zte/l3_precision_final/summary.md`

## 5. 精度结果

### 5.0 指标解释

- `Pass Rate`
  表示通过 compare 脚本判定的样本比例。`timi` 的通过条件是单样本 `max_diff <= 1e-5`，并且标签准确率不低于 `99%`。这个阈值非常严，所以即使是对方 GPU 输出，对 `timi` 也可能是 `0%`。
- `Avg Max Diff`
  先对每个样本计算一个“最大绝对误差”，再对全部样本取平均。这个值越小，说明整体上越接近 reference output。
- `Peak Max Diff`
  全部样本、全部输出位置里的最大绝对误差峰值，反映最坏情况下偏了多少。
- `Mean Label Acc`
  对输出最后一维做 `argmax`，比较预测标签是否一致，再对全部样本求平均。这个值越高，说明最终类别判断越稳定。
- `MAE`
  Mean Absolute Error，平均绝对误差。它看的是整体数值差多少，不只看最坏点。
- `MAPE(%)`
  Mean Absolute Percentage Error，平均绝对百分比误差。它反映相对误差大小，越小越好。

### 5.1 timi

`compare_inference_results.py` 的通过条件是 `max_diff <= 1e-5` 且标签准确率不低于 `99%`。由于对方自己的 `64Batch_output_gpu.jsonl` 在这个阈值下也是 `0%` pass rate，所以这个 case 主要看 `Avg Max Diff` 和 `Mean Label Acc`，不要只看 `Pass Rate`。表中的 `vendor_gpu_vs_pytorch_ref` 表示“对方 GPU 输出”相对于 `inference_results.jsonl` 中 PyTorch reference output 的对比结果。

| 配置 | 样本数 | Pass Rate | Avg Max Diff | Peak Max Diff | Mean Label Acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| vendor_gpu_vs_pytorch_ref | 32 | 0.00% | 1.491546e-03 | 1.831889e-03 | 99.9756% |
| w8a8_sefp | 32 | 0.00% | 4.671969e-02 | 6.374991e-02 | 99.3439% |
| w8a16_sefp | 32 | 0.00% | 1.831133e-02 | 2.378440e-02 | 99.4019% |
| w16a16_sefp | 32 | 0.00% | 2.118231e-03 | 2.865791e-03 | 99.9817% |

结论：`w16a16_sefp` 明显最好，已经接近对方 GPU baseline；`w8a16_sefp` 次之；`w8a8_sefp` 误差最大。

### 5.2 hxx_lstm

`compare_model_outputs.py` 输出的是 `MAE` 和 `MAPE(%)`，这两个值越小越好。表中的 `vendor_gpu_vs_model_io_base` 表示“对方 GPU 输出”相对于 `model_io.xlsx` 基准输出的对比结果。

| 配置 | MAE | MAPE(%) |
| --- | ---: | ---: |
| vendor_gpu_vs_model_io_base | 9.515639e-06 | 0.008534 |
| w8a8_sefp | 4.478217e-04 | 0.392493 |
| w8a16_sefp | 3.848738e-04 | 0.336273 |
| w16a16_sefp | 1.302664e-04 | 0.119731 |

结论：`w16a16_sefp` 同样是最优配置，`w8a16_sefp` 好于 `w8a8_sefp`。

## 6. 推荐用法

- 如果目标是尽量贴近对方 GPU 输出，优先使用 `w16a16_sefp`。
- 如果只是复现本次完整流程，直接运行一键脚本即可，不需要分别手工量化、手工推理、手工执行 compare。
- 每次重跑后，优先查看 `<output-root>/summary.md`，再根据需要进入各 case 的 `compare/` 目录看明细报告。
