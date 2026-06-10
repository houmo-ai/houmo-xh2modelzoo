# DINOv3 LT-DETR XH2a Quantization

本目录保留 DINOv3/LT-DETR 检测模型的当前主线流程：

1. 从 LightlyTrain 模型导出 ONNX。
2. 用 `xhquant` 混合精度搜索得到 **77 个 weighted/TE W16 节点**。
3. 导出 HMONNX。
4. 用 COCO sample200 评估 mixed77 baseline，以及 mixed77 + 3 个 bbox refinement sigmoid FP fallback。

中间 ablation、range probe、softmax/sigmoid 定位、QuaRot 等实验结果已经归档到 `examples/cv/dinov3/debug/`。
旧版 backbone-only 导出、普通量化、range probe、QuaRot 导出等非主线脚本已归档到
`examples/cv/dinov3/debug/legacy_scripts/`。

## 路径与环境

```bash
PYTHON=/data01/home/xuzk/anaconda3/envs/xh2/bin/python
NOPROXY="env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u all_proxy"
COCO_IMG=/data01/datasets/coco2017/val2017
COCO_ANN=/data01/datasets/coco2017/annotations/instances_val2017.json
```

默认文件：

- ONNX：`examples/cv/dinov3/onnx/dinov3-vitt16-ltdetr-coco_is640.onnx`
- 搜索主目录：`examples/cv/dinov3/mixed_precision_search/auto_search_sample8_top40_v2`
- 77 W16 配置：`examples/cv/dinov3/mixed_precision_search/auto_search_sample8_top40_v2/mixed_precision_weighted_w16.json`
- HMONNX：`examples/cv/dinov3/mixed_precision_search/auto_search_sample8_top40_v2/hmonnx/dinov3-vitt16-ltdetr-coco_is640_mix_search_w8a8h1_sefp_XH2a.onnx`

## 脚本职责

- `examples/cv/dinov3/1.lightly_ltdetr_export_onnx.py`：只负责加载 LightlyTrain 模型并导出默认 ONNX。
- `examples/cv/dinov3/dinov3_common.py`：共享工具库，包含 ONNX IO、COCO 采样、量化配置、HMONNX runner 和 COCO 输出转换。
- `examples/cv/dinov3/2.mix_search_xh2_export_hmonnx.py`：一键执行 PTQ、混合精度搜索、写出 `mixed_precision_weighted_w16.json`、导出 HMONNX。
- `examples/cv/dinov3/mix_search_tool.py`：混合精度搜索工具库，继承 `xhquant.mix_precision.MixPrecisionSearch`，只搜索 6 类 weighted/TE 候选。
- `examples/cv/dinov3/3.hmonnx_eval_coco.py`：COCO mAP 评估，支持 `torch` / `onnx` / `frontend` / `quantgraph` / `hmonnx`。

## Step 1：导出 ONNX

```bash
$NOPROXY $PYTHON examples/cv/dinov3/1.lightly_ltdetr_export_onnx.py
```

无参数运行默认等价于 `export-onnx`，默认导出到 `examples/cv/dinov3/onnx/dinov3-vitt16-ltdetr-coco_is640.onnx`，使用 `precision=fp32`、`batch_size=1`、`opset=17`，并把 LightlyTrain 缓存放到 `examples/cv/dinov3/models` 和 `examples/cv/dinov3/data_cache`。

## Step 2：搜索 77 个 weighted/TE W16 节点并导出 HMONNX

当前保留的主线是 top40 weighted 搜索结果，对应 77 个 weighted/TE W16 节点。重新跑脚本会输出 canonical 配置：

- `search_result.pt`
- `search_summary.json`
- `mixed_precision_weighted_w16.json`
- `hmonnx/*.onnx`

```bash
$NOPROXY CUDA_VISIBLE_DEVICES=2 $PYTHON examples/cv/dinov3/2.mix_search_xh2_export_hmonnx.py
```

无参数运行默认使用 `examples/cv/dinov3/onnx/dinov3-vitt16-ltdetr-coco_is640.onnx`，COCO 路径为 `/data01/datasets/coco2017/val2017` 和 `/data01/datasets/coco2017/annotations/instances_val2017.json`，输出到 `examples/cv/dinov3/mixed_precision_search/auto_search_sample8_top40_v2`，搜索参数为 `sample_size=8`、`sample_seed=4200`、`mix_policy=topk`、`mix_topk=0.4`、`mix_weight_bits=8,16`、`mix_act_bits=8,16`、`mix_label_weight=0.25`。

确认节点数：

```bash
$PYTHON - <<'PY'
import json
p = "examples/cv/dinov3/mixed_precision_search/auto_search_sample8_top40_v2/mixed_precision_weighted_w16.json"
print(len(json.load(open(p))["nodes"]))
PY
```

期望输出：`77`。

## Step 3：评估 mixed77 baseline

该评估使用 `mixed_precision_weighted_w16.json`，只把搜索出的 77 个 weighted/TE 节点设置为 W16/A16/O16，其余节点保持默认 `w8a8h1_sefp`。

```bash
$NOPROXY CUDA_VISIBLE_DEVICES=2 \
  $PYTHON examples/cv/dinov3/3.hmonnx_eval_coco.py \
    --backend quantgraph \
    --quantgraph-mode aligned \
    --quant-type w8a8h1_sefp \
    --mixed-precision-config examples/cv/dinov3/mixed_precision_search/auto_search_sample8_top40_v2/mixed_precision_weighted_w16.json \
    --onnx examples/cv/dinov3/onnx/dinov3-vitt16-ltdetr-coco_is640.onnx \
    --images-dir $COCO_IMG \
    --annotations $COCO_ANN \
    --sample-size 200 \
    --sample-seed 4200 \
    --log-every 20 \
    --out-dir examples/cv/dinov3/coco_eval/mixed_top40v2_weighted_w16_baseline_sample200_seed4200
```

当前保留结果：

- 指标：`examples/cv/dinov3/coco_eval/mixed_top40v2_weighted_w16_baseline_sample200_seed4200/quantgraph_metrics.json`
- AP50:95：`0.514826564163`

## Step 4：评估 mixed77 + 3 个 bbox sigmoid FP fallback

定位结果显示当前 `xhquant` 的 `QSigmoid` 仍走 LUT，A16/O16 不能消除这 3 个 bbox refinement sigmoid 的误差。`3.hmonnx_eval_coco.py` 现在提供命名 preset，不需要手写正则：

```bash
--quantgraph-torch-sigmoid-preset decoder-bbox-refine-top3
```

该 preset 等价于：

```text
^(_decoder_decoder_sigmoid_2|_decoder_decoder_sigmoid_3|_decoder_decoder_sigmoid_4)$
```

评估命令：

```bash
$NOPROXY CUDA_VISIBLE_DEVICES=2 \
  $PYTHON examples/cv/dinov3/3.hmonnx_eval_coco.py \
    --backend quantgraph \
    --quantgraph-mode aligned \
    --quant-type w8a8h1_sefp \
    --mixed-precision-config examples/cv/dinov3/mixed_precision_search/auto_search_sample8_top40_v2/mixed_precision_weighted_w16.json \
    --onnx examples/cv/dinov3/onnx/dinov3-vitt16-ltdetr-coco_is640.onnx \
    --images-dir $COCO_IMG \
    --annotations $COCO_ANN \
    --sample-size 200 \
    --sample-seed 4200 \
    --log-every 20 \
    --quantgraph-torch-sigmoid-preset decoder-bbox-refine-top3 \
    --out-dir examples/cv/dinov3/coco_eval/mixed_top40v2_weighted_w16_sigmoid234_fp_sample200_seed4200
```

当前保留结果：

- 指标：`examples/cv/dinov3/coco_eval/mixed_top40v2_weighted_w16_sigmoid234_fp_sample200_seed4200/quantgraph_metrics.json`
- AP50:95：`0.527019770753`
- 对比 torch AMP fp16 sample200 AP50:95：`0.530563686826`
- 差距：约 `0.00354 AP`

只保留 `_decoder_decoder_sigmoid_3/_4` 的对照结果：

- 指标：`examples/cv/dinov3/coco_eval/mixed_top40v2_weighted_w16_sigmoid34_fp16_sample200_seed4200/quantgraph_metrics.json`
- AP50:95：`0.524847211330`
- AP75：`0.562635193261`
- 比 2/3/4 三个节点少约 `0.00217 AP`

## Step 5：评估 HMONNX

HMONNX 运行评估：

```bash
$NOPROXY CUDA_VISIBLE_DEVICES=2 $PYTHON examples/cv/dinov3/3.hmonnx_eval_coco.py
```

无参数运行默认使用 `backend=hmonnx`，HMONNX 路径为 `examples/cv/dinov3/mixed_precision_search/auto_search_sample8_top40_v2/hmonnx/dinov3-vitt16-ltdetr-coco_is640_mix_search_w8a8h1_sefp_XH2a.onnx`，COCO 路径为 `/data01/datasets/coco2017/val2017` 和 `/data01/datasets/coco2017/annotations/instances_val2017.json`，评估 `sample_size=200`、`sample_seed=4200`、`log_every=20`，输出到 `examples/cv/dinov3/coco_eval/hmonnx_mix77_sample200_seed4200`。

当前目录中仍保留了一个 HMONNX W16A16 full-val baseline：

- `examples/cv/dinov3/coco_eval/hmonnx_w16_full/hmonnx_metrics.json`

注意：`quantgraph` 的 `decoder-bbox-refine-top3` preset 是评估期 monkey patch，用来验证 `QSigmoid` LUT 误差来源；它不会自动改变 HMONNX 导出的 sigmoid 实现。要让 HMONNX 真正复现该效果，需要在导出/运行时增加 FP sigmoid 或高精度 sigmoid op 支持。

## 当前保留目录

```text
examples/cv/dinov3/
├── README.md
├── 1.lightly_ltdetr_export_onnx.py
├── 2.mix_search_xh2_export_hmonnx.py
├── 3.hmonnx_eval_coco.py
├── dinov3_common.py
├── mix_search_tool.py
├── coco_eval/
│   ├── hmonnx_w16_full/
│   ├── mixed_top40v2_weighted_w16_baseline_sample200_seed4200/
│   ├── mixed_top40v2_weighted_w16_sigmoid34_fp_sample200_seed4200/
│   ├── mixed_top40v2_weighted_w16_sigmoid34_fp16_sample200_seed4200/
│   └── mixed_top40v2_weighted_w16_sigmoid234_fp_sample200_seed4200/
├── debug/
│   ├── coco_eval/
│   ├── legacy_scripts/
│   ├── mixed_precision_search/
│   └── onnx/
├── mixed_precision_search/
│   └── auto_search_sample8_top40_v2/
│       ├── hmonnx/
│       ├── mix_search_config.json
│       ├── mixed_precision_weighted_w16.json
│       ├── search_result.pt
│       └── search_summary.json
└── onnx/
    └── dinov3-vitt16-ltdetr-coco_is640.onnx
```

## 关键结论

- 混合精度搜索主线只保留 6 类 weighted/TE 候选，不再搜索 activation-only 节点。
- 当前 77 weighted/TE W16 配置的 QuantGraph AP50:95 为 `0.514826564163`。
- 加上 3 个 bbox refinement sigmoid 的 FP fallback 后 AP50:95 为 `0.527019770753`。
- `xhquant` 现有 sigmoid 是 LUT 路径；A16/O16 只改变输入/输出量化边界，不能替代 FP sigmoid 函数本身。
