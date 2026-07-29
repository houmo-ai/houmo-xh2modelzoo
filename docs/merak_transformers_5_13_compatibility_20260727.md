# Merak 模型族 Transformers 5.13 兼容性结论

## 结论

Qwen3-Next、Qwen3.5/Qwen3.6（dense 与 MoE）和 Gemma4 Series 的 Merak
路径可以统一运行在 `transformers==5.13.0`。量化导出和部署必须使用各自固定
环境，不能为另一个环境的 Torch/Triton 行为在导出代码中增加兼容分支：

- 量化和 HMONNX 导出只使用 `xhquant_55`（Torch 2.8）；
- vLLM Merak 部署和性能测试只使用 `vllm`（Torch 2.11）；
- 项目级 `pyproject.toml` 继续保持 `transformers>=4.57,<4.58`，避免把尚未
  回归的历史模型一起切换；
- Transformers 5.13 环境必须使用稳定版 `safetensors>=0.8.0`，不能使用
  `0.8.0rc0`；
- vLLM 0.24 环境中的 `compressed-tensors 0.17` 不再从顶层导出
  `has_offloaded_params`。两个历史 `base_model.py` 镜像统一从其实际所有者
  `accelerate.utils.has_offloaded_params` 导入；该入口也兼容旧环境。

## 已验证矩阵

2026-07-28 在同一份源码上执行：

| 环境 | Transformers | Safetensors | Torch | 结果 |
| --- | ---: | ---: | ---: | --- |
| `vllm` | 5.13.0 | 0.8.0 | 2.11.0+cu129 | 53 passed |
| `xhquant_55` | 5.13.0 | 0.8.0 | 2.8.0+cu128 | 53 passed |

测试范围：

```bash
pytest -q \
  tests/qwen3_next \
  tests/qwen3_5 \
  tests/qwen3_5_moe \
  tests/gemma4
```

证据日志：

```text
work_dirs/transformers_5_13_xhquant55_family_tests_20260728.log
work_dirs/transformers_5_13_vllm_family_tests_20260728.log
```

另外已直接导入 Qwen3-Next、Qwen3.5、Gemma4 和 Gemma4 Unified 的 config/model
入口。这里的“通过”表示这些模型族的 Merak 配置、wrap、导出合同和无权重 smoke
通过，不代表仓库中所有旧模型、旧数据路径和历史脚本都已经完成 5.13 回归。

同一份当前源码还在原 Transformers 5.5 环境通过 152 个 Qwen3.5/Qwen3-Next
配置、cache transaction、workflow 与 MTP 合同测试，确认 5.13 兼容改动没有把
现有 Qwen 导出路径升级为 5.13-only：

```text
/data01/home/yujy/work/vllm-merak/work_dirs/qtl428_tf513_final_services/review_points_qwen_tf55_20260730.log
```

## GPTQModel 与全量链路验证

Transformers 升级后，checkpoint key conversion 属于通用加载语义，不应在
xh2modelzoo 为 Gemma4-12B 增加特判加载器。GPTQModel 的公共 checkpoint loader
现在遵循以下规则：

- 先读取 Transformers 注册的 ordered conversion mapping；
- 只有 checkpoint key 确实需要 rename/converter 时才走 Transformers loader；
- 已与 runtime key 对齐的 checkpoint 继续走 Accelerate exact-key 快路径；
- Transformers 5.5 与 5.13 的 exact-key/conversion 路径均已验证；
- xh2modelzoo 不包含 Gemma4 Unified 专用 GPTQModel loader 或前端权重修补。

GPTQModel 5.8 在 `xhquant_55` 中完成了 47 个 Qwen3.5、Qwen3-Next 和 Gemma4
量化 recipe/adapter 测试。随后用 Qwen3.5-0.8B 做了不是“只量几层”的全量
W4G64 示例：

- 24 个 decoder layer 的 150 个目标线性模块全部进入 `quant_log.csv`；
- 保存后的 GPTQ checkpoint 可由 Transformers 5.13/GPTQModel 重新加载；
- 在同一个 `xhquant_55` 环境导出 256K、FlashAttention、融合 GDR 的视觉
  HMONNX，以及 prefill/decode HMONNX；
- 在 `vllm` 环境以 `max_model_len=262144` 加载并通过文本、图片请求；
- runtime 在线 pack 覆盖 132 个 W4 节点，单份权重从 383,778,816 bytes
  降为 191,889,408 bytes，prefill/decode 共用同一份 packed initializer。

GPTQModel checkpoint conversion 的相同 7 个单测还在现有
`gemma4`（Torch 2.8、Transformers 5.5）环境中通过，确认新 loader 不要求
Transformers 5.13。这里仅验证公共 checkpoint conversion；Gemma4 Unified
模型类型本身是 Transformers 5.13 新增能力，不声明可在 5.5 中量化或加载：

```text
/data01/home/yujy/work/gptqmodel/work_dirs/qtl428_tf513_review/gptq_conversion_tf55_final_85bac8e_20260730.log
```

证据目录：

```text
/data01/home/yujy/work/gptqmodel/work_dirs/qtl428_tf513_qwen35_0_8b_full_w4g64_20260728
work_dirs/qtl428_tf513_full_quant_export_0_8b
/data01/home/yujy/work/vllm-merak/work_dirs/qtl428_tf513_smoke
```

## 固定执行方式

导出：

```bash
source /data01/home/yujy/miniconda3/etc/profile.d/conda.sh
conda activate xhquant_55
# quantize / export
```

部署：

```bash
source /data01/home/yujy/miniconda3/etc/profile.d/conda.sh
conda activate vllm
# vLLM Merak serve / benchmark
```

Transformers 升级后必须先运行上面的模型族测试，再用真实权重完成一次全量量化、
HMONNX 导出和 vLLM Merak smoke；只通过无权重单测不能视为兼容完成。
