# stella-mrl 导出示例

## 1. 下载模型

已通过 ModelScope 下载到：

- /data02/datasets/stella-mrl/t3lsss/stella-mrl-large-zh-v3___5-1792d

## 2. 导出 HMONNX

```bash
source env.sh
/data01/home/xuchen/miniconda3/envs/xhquant/bin/python examples/llm/stella-mrl/stella_mrl_xh2a_export_hmonnx.py \
  --model /data02/datasets/stella-mrl/t3lsss/stella-mrl-large-zh-v3___5-1792d \
  --batch-size 1 \
  --context-length 512 \
  --quant-type w8a8_sefp
```

默认输出到：

- work_dirs/stella-mrl-large-zh-v3___5-1792d-XH2a/

meta.json 中包含：

- hmonnx_file
- output_dim（默认 1792）
- output_normalized（是否图内做 L2 归一化）
