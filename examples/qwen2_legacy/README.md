# Qwen2

## 配置参数

```text
batch_size
context-length
input-sequence-length
quant-type: for example: w4a8-ssfp、w8a8-sefp
quant-weight : GPTQ、Quarot量化后的权重文件
```

## 准备模型

将Qwen2.5-7B-Instruct模型放到data/models/Qwen2.5-7B-Instruct目录下  

## 导出HMONNX

transformers库需要升级到4.51.0以上版本，否则会报错。  

### w8a8

#### 1. 导出

```bash
python examples/qwen2_legacy/qwen2_legacy_xh2a_export_hmonnx.py --model data/models/Qwen2.5-7B-Instruct --context-length 2048 --input-sequence-length 256 --quant-type w8a8_sefp
```

#### 2. GPU仿真

```bash
python examples/qwen2_legacy/qwen2_legacy_xh2a_hmonnx_test.py --config work_dirs/Qwen2.5-7B-Instruct-XH2a-2k-w8a8h1_sefp/meta.json
```

### w4a8

#### 1. Quarot、GPTQ量化  

lm_head 层一般不使用4bit int 量化，使用w8a8h1_sefp量化。

```bash
python examples/qwen2_legacy/qwen2_xh2a_common_quant.py  --model data/models/Qwen2.5-7B-Instruct  --w-bits 4 --out-dir work_dirs/
默认做Quarot+GPTQ量化，weight bit = 4
```

#### 2. 导出

```bash
python examples/qwen2_legacy/qwen2_legacy_xh2a_export_hmonnx.py --model data/models/Qwen2.5-7B-Instruct --context-length 2048 --input-sequence-length 256 --quant-type w4a8_ssfp --quant-weight work_dirs/Qwen2.5-7B-Instruct_quarot_gptq/quarot_gptq-state-dict.safetensors
```

#### 3. GPU仿真

```bash
python examples/qwen2_legacy/qwen2_legacy_xh2a_hmonnx_test.py --config work_dirs/Qwen2.5-7B-Instruct-XH2a-2k-w4a8_ssfp/meta.json 
```
