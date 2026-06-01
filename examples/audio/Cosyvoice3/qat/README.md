# CosyVoice3 LLM+Flow QAT 训练与部署指南

基于 xhquant 的量化感知训练（QAT），仅保留 LLM + Flow 端到端流程。

## 目录结构

```
qat/
├── qat_utils.py                       # 共享 QAT 基础设施（7 阶段流程 + xhquant API）
├── qat_module_llm.py                  # LLM 模块加载器
├── qat_module_flow.py                 # Flow 模块加载器
├── cosyvoice3_qat_e2e_llm_pred100.py  # LLM+Flow Pred100 QAT（主训练脚本）
├── cosyvoice3_qat_flow_distill.py     # Flow 蒸馏 QAT（可选精修）
├── predict_tokens_offline.py          # 离线 LLM token 预测（Flow distill 数据准备）
├── export_all.py                      # QAT 权重导出（反量化）
├── setup_eval.py                      # 构建评测模型目录
├── run_qat.sh                         # 训练启动脚本
├── README.md                          # 本文件
├── eval/                              # 评测
│   ├── eval_qat.py                    # 统一评测脚本（WER + Spk Sim + DNSMOS）
│   └── plot_training_loss.py          # 训练 loss 可视化
├── prepare/                           # 数据准备
│   ├── prepare_data.py                # 统一入口 (--dataset librispeech|thchs30|zero_shot|libritts)
│   └── prepare_synthetic.py           # 合成数据（冒烟测试）
└── trash/                             # 废弃脚本
```

## 完整流程

### Step 0: 数据准备

```bash
conda activate xhquant
cd qat/prepare/

# LibriSpeech (本地 tar.gz, 多 GPU)
python prepare_data.py --dataset librispeech \
    --train_tgz /path/to/train-clean-100.tar.gz \
    --dev_tgz /path/to/dev-clean.tar.gz \
    --output_dir ../../data/data_librispeech --gpu_ids 0,1,2,3

# THCHS-30 中文
python prepare_data.py --dataset thchs30 \
    --tgz_path /path/to/data_thchs30.tgz \
    --output_dir ../../data/data_thchs30 --gpu_ids 0,1

# CV3-Eval 多语言
python prepare_data.py --dataset zero_shot \
    --data_root /path/to/CV3-Eval/data/zero_shot \
    --cv3_eval_root /path/to/CV3-Eval \
    --output_dir ../../data/data_zero_shot --gpu_ids 0,1

# LibriTTS (HuggingFace, 单进程)
python prepare_data.py --dataset libritts \
    --output_dir ../../data/data_libritts --max_train 5000
```

多数据集混合训练只需合并 list：
```bash
cat data_librispeech/train.list data_thchs30/train.list > data_mixed/train.list
```

### Step 1: LLM+Flow Pred100 QAT 训练

```bash
cd qat/
conda activate xhquant

export MODEL_DIR=/data01/nfs_shared/ASR_TTS/CosyVoice3-0.5B-2512
export HF_MODEL_DIR=$MODEL_DIR/CosyVoice-BlankEN
export YAML_PATH=$MODEL_DIR/cosyvoice3.yaml
export TRAIN_DATA=../data/data_librispeech/train.list
export CV_DATA=../data/data_librispeech/dev.list

CUDA_VISIBLE_DEVICES=1 python -u cosyvoice3_qat_e2e_llm_pred100.py
```

或使用 `run_qat.sh` 一键启动。

### Step 2: 导出（反量化）

```bash
python export_all.py \
    --model_dir $MODEL_DIR \
    --qat_dir ./output_cosyvoice3_qat_pred100 \
    --qat_steps 5000
```

### Step 3: 构建评测目录

```bash
python setup_eval.py \
    --src_model_dir $MODEL_DIR \
    --qat_dir ./output_cosyvoice3_qat_pred100 \
    --output_dir ./eval_model_qat \
    --qat_steps 5000
```

### Step 4: 评测

```bash
cd eval/

# CV3-Eval 评测 (中文 zero-shot, 含量化推理)
conda activate xhquant
python eval_qat.py \
    --data_source cv3_eval --model_type qat \
    --model_dir $MODEL_DIR \
    --qat_ckpt_dir ../output_cosyvoice3_qat_pred100 --qat_steps 5000

# Parquet dev 评测 (快速验证, 无量化)
conda activate cosyvoice-deploy
python eval_qat.py \
    --data_source parquet --model_type prebuilt \
    --model_dir ../eval_model_qat \
    --dev_parquet ../../data/data_librispeech_full_tokw8a16/cv/dev_w8a16.list \
    --max_samples 100
```

## 可选: Flow 蒸馏精修

Pred100 QAT 完成后，可进一步蒸馏 Flow，提升对量化 LLM 输出的鲁棒性。

### 数据准备

```bash
# FP pred tokens (teacher)
python predict_tokens_offline.py \
    --model_dir $MODEL_DIR \
    --input_list ../data/train.list \
    --output_dir ../data/data_fp_pred_tokens

# QAT pred tokens (student)
python predict_tokens_offline.py \
    --model_dir $MODEL_DIR \
    --llm_weights ./output_cosyvoice3_qat_pred100/llm/dequant_steps5000.pt \
    --quantize \
    --input_list ../data/train.list \
    --output_dir ../data/data_qat_pred_tokens
```

### 训练 + 组合

```bash
export TRAIN_DATA=../data/data_qat_pred_tokens/train_abs_pred.list
export GT_TRAIN_DATA=../data/data_fp_pred_tokens/train_abs_pred.list
export CV_DATA=../data/data_qat_pred_tokens/cv/train_abs_pred.list
export GT_CV_DATA=../data/data_fp_pred_tokens/train_abs_pred.list
export TEACHER_FLOW_WEIGHTS=$MODEL_DIR/flow.pt

CUDA_VISIBLE_DEVICES=2 python -u cosyvoice3_qat_flow_distill.py \
    W_FLOW=1.0 W_DISTILL=5.0 W_MU=0.1 TRAIN_STEPS=5000 SKIP_EVAL=1

# 组合 LLM (pred100) + Flow (distill)
mkdir -p output_qat_combined/{llm,flow}
ln -s $(realpath output_cosyvoice3_qat_pred100/llm/qat_steps5000.pt) output_qat_combined/llm/
ln -s $(realpath output_cosyvoice3_qat_flow_distill/flow/qat_steps5000.pt) output_qat_combined/flow/
python export_all.py --qat_dir ./output_qat_combined --qat_steps 5000
```

## 环境变量（训练）

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MODEL_DIR` | `/data01/nfs_shared/ASR_TTS/CosyVoice3-0.5B-2512` | 模型目录 |
| `HF_MODEL_DIR` | `{MODEL_DIR}/CosyVoice-BlankEN` | Qwen2 HF 权重 |
| `YAML_PATH` | `{MODEL_DIR}/cosyvoice3.yaml` | 模型配置 |
| `TRAIN_DATA` / `CV_DATA` | (必填) | 训练/验证数据 list（每行一个 parquet 路径） |
| `BATCH_SIZE` | 1 | 批大小 |
| `TRAIN_STEPS` | 200 | 训练步数 |
| `EVAL_INTERVAL` | 50 | 评估间隔 |
| `LEARNING_RATE` | 1e-5 | 学习率 |
| `W_MAN_BIT` | 8 | 权重尾数位数（8=w8a8, 4=w4a8） |
| `GRAD_ACCUM_STEPS` | 2 | 梯度累积步数 |
| `LLM_LOSS_WEIGHT` | 1.0 | LLM cross-entropy loss 权重 |
| `FLOW_LOSS_WEIGHT` | 1.0 | Flow matching loss 权重 |
| `SKIP_EVAL` | 0 | 设为 1 跳过训练中评估，大幅加速 |

## QAT 适配新模型

### 1. 写模块加载器

参考 `qat_module_llm.py` / `qat_module_flow.py`：

```python
MODULE_NAME = "your_module"
CKPT_FILE = "your_module.pt"

def load_your_module(yaml_path, hf_model_dir):
    overrides = {"qwen_pretrain_path": hf_model_dir, ...}
    with open(yaml_path) as f:
        configs = load_hyperpyyaml(f, overrides=overrides)
    return configs["your_module"]
```

### 2. 写训练 Wrapper

确保 `model.forward(batch, device)` 返回 `{"loss": tensor}`：

```python
class YourWrapper(nn.Module):
    def __init__(self, your_module):
        super().__init__()
        self.your_module = your_module

    def forward(self, batch, device):
        result = self.your_module(batch, device)
        return {"loss": result["loss"]}
```

### 3. 配置量化参数

SEFP 量化配置（在 qat_utils.py 中定义）：

```python
{
    "precision_mode": "aligned",
    "w_schema": {"fp_mode": "sefp", "man_bit": 8, "nshare": 64, "rounding": "rne"},
    "act_schema": {"fp_mode": "sefp", "man_bit": 8, "nshare": 64, "rounding": "rne"},
}
```

### 4. 调用训练流程

使用 `qat_utils.qat_train_module()` 跑 7 阶段标准流程，或参考
`cosyvoice3_qat_e2e_llm_pred100.py` 编写多模块联合训练。

## 依赖

```
# QAT 训练 (conda activate xhquant)
xhquant
torch >= 2.0
hyperpyyaml
transformers

# 评测 (conda activate cosyvoice-deploy)
cosyvoice
funasr, jiwer, zhconv   # WER 计算
onnxruntime             # Speaker similarity
```
