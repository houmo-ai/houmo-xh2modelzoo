"""
CosyVoice3 QAT 共享工具库
==========================

提供完整的 7 阶段 QAT 训练基础设施，供各模块训练脚本复用。

包含:
    - 环境变量 / 通用工具函数
    - 数据管线 (CosyVoiceDataset, SyntheticDataset, collate_fn)
    - 训练工具 (warmup, train_one_step, evaluate)
    - QAT 工具 (dequantize, save, clip, sanitize)
    - 单模块 7 阶段 QAT 流程 (qat_train_module)

xhquant API 概览
----------------
QAT 流程使用三个 xhquant 入口:

    1. prepare_quanted_model_to_compile(name, model, device_type, config)
       - 将模型中每个 nn.Linear/nn.Conv 包装为 QBaseModule
       - 返回 (quantized_model, backend)
       - config 关键字段: precision_mode, enable_qat_compiler_ops,
         w_schema (权重量化), act_schema (激活量化)
       - enable_qat_compiler_ops=True → 训练模式 (插入 STE fake-quant)
       - enable_qat_compiler_ops=False → 部署模式 (真实量化, 无 STE)

    2. torch_compile_quanted_model(qmodel, backend)
       - 编译 STE 计算图以提升训练效率
       - 必须在第一次 forward 之前调用

    3. 反量化: 遍历 named_modules 找到 QBaseModule 实例,
       调用 module.w_quantizer.true_2_fake_convert(qweight, scale, ...)
       还原为与原始模型结构兼容的 FP 权重

7 阶段 QAT 流程
----------------
    Stage 0: FP 基线评测
    Stage 1: PTQ prepare (deepcopy 模型 → prepare_quanted_model_to_compile)
    Stage 2: QAT compile + warmup (torch_compile_quanted_model → forward+backward)
    Stage 3: QAT 训练 (Adam, cosine LR, gradient accumulation, STE 更新)
    Stage 4: 训练后评测
    Stage 5: 反量化导出 FP 权重
    Stage 6: 部署一致性验证 (enable_qat_compiler_ops=False 模式下)

适配新模型
----------
    1. 写模块加载器 (参考 qat_module_llm.py / qat_module_flow.py)
    2. 确保模型 forward(batch, device) 返回 {"loss": ...}
    3. 定义量化配置 (w_schema, act_schema)
    4. 调用 qat_train_module() 或基于阶段工具函数编写自定义训练循环
"""

import copy
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ================================================================
#  CosyVoice / Matcha-TTS 路径 — LLM 和 Flow 需要
# ================================================================
_COSYVOICE_ROOT = os.getenv(
    "COSYVOICE_ROOT",
    os.path.expanduser("~/workspace/repo/develop/CosyVoice"),
)
sys.path.insert(0, _COSYVOICE_ROOT)
_MATCHA_PATH = os.path.join(_COSYVOICE_ROOT, "third_party", "Matcha-TTS")
if os.path.isdir(_MATCHA_PATH):
    sys.path.insert(0, _MATCHA_PATH)

from xhquant.api import prepare_quanted_model_to_compile
from xhquant.core import QBaseModule
from xhquant.quantization import torch_compile_quanted_model


# ================================================================
#  XHConv1d 兼容补丁
# ================================================================
try:
    from xhquant.nn.modules.conv1d import XHConv1d as _XHConv1d

    if not hasattr(_XHConv1d, "kernel_size"):
        _XHConv1d.kernel_size = property(
            lambda self: self.conv2d.kernel_size[:1]
            if self.conv2d is not None else None
        )
        _XHConv1d.stride = property(
            lambda self: self.conv2d.stride[:1]
            if self.conv2d is not None else None
        )
        _XHConv1d.padding = property(
            lambda self: self.conv2d.padding[:1]
            if self.conv2d is not None else None
        )
except ImportError:
    pass


# ================================================================
#  默认路径 & 环境变量工具
# ================================================================
DEFAULT_MODEL_DIR = "/data01/nfs_shared/ASR_TTS/CosyVoice3-0.5B-2512"
DEFAULT_OUTPUT_DIR = "./output_cosyvoice3_qat"


def get_env_int(name: str, default: int) -> int:
    return int(os.getenv(name) or default)


def get_env_float(name: str, default: float) -> float:
    return float(os.getenv(name) or default)


def get_env_str(name: str, default: str) -> str:
    return os.getenv(name) or default


def print_stage(title: str):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


# ================================================================
#  数据管线 — CosyVoice parquet 加载
# ================================================================

class CosyVoiceDataset(torch.utils.data.Dataset):
    """从 CosyVoice 预处理 parquet 文件加载训练数据。

    parquet 文件需包含字段:
        text, audio_data, spk_embedding, speech_token

    __getitem__ 时完成:
        - 文本分词 (Qwen tokenizer)
        - 音频解码 + mel 频谱计算
        - pitch 提取
    """

    def __init__(
        self,
        data_list_path: str,
        tokenizer,
        mel_fn,
        sample_rate: int = 24000,
        hop_size: int = 480,
        max_samples: int = -1,
    ):
        self.tokenizer = tokenizer
        self.mel_fn = mel_fn
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.samples = self._load_parquet_index(data_list_path, max_samples)

    @staticmethod
    def _load_parquet_index(data_list_path: str, max_samples: int) -> list:
        """读取 data.list 指向的所有 parquet 文件，构建样本索引。"""
        import pyarrow.parquet as pq

        parquet_files = []
        with open(data_list_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    parquet_files.append(line)

        samples = []
        for pf in parquet_files:
            table = pq.read_table(pf)
            df = table.to_pandas()
            for idx in range(len(df)):
                row = df.iloc[idx]
                samples.append({
                    "text": str(row.get("text", "")),
                    "spk_embedding": row.get("spk_embedding"),
                    "speech_token": row.get("speech_token"),
                    "audio_data": row.get("audio_data"),
                })
                if 0 < max_samples <= len(samples):
                    return samples
        return samples

    def _tokenize_text(self, text: str):
        """将文本拆分为 instruct_token 和 text_token。"""
        import io

        EOP_ID = 151646
        eop_marker = "<|endofprompt|>"
        if eop_marker in text:
            instruct_text, content_text = text.split(eop_marker, 1)
            instruct_ids = self.tokenizer.encode(
                instruct_text, add_special_tokens=False
            )
            content_ids = self.tokenizer.encode(
                content_text, add_special_tokens=False
            )
            instruct_token = torch.tensor(
                instruct_ids + [EOP_ID], dtype=torch.long
            )
            text_token = torch.tensor(content_ids, dtype=torch.long)
        else:
            instruct_token = torch.tensor([], dtype=torch.long)
            text_token = torch.tensor(
                self.tokenizer.encode(text, add_special_tokens=False),
                dtype=torch.long,
            )
        return text_token, instruct_token

    def _compute_speech_feat(self, audio_bytes: bytes):
        """从原始音频字节计算 80 维 mel 频谱。"""
        import io

        import torchaudio

        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                sr, self.sample_rate
            )(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return self.mel_fn(waveform)

    def _extract_f0(self, audio_bytes: bytes):
        """提取 F0 pitch 特征。"""
        try:
            import io

            import numpy as np
            import pyworld as pw
            import torchaudio

            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
            if sr != self.sample_rate:
                waveform = torchaudio.transforms.Resample(
                    sr, self.sample_rate
                )(waveform)
            wav_np = waveform.squeeze(0).numpy().astype(np.float64)
            hop = self.hop_size
            f0, _ = pw.harvest(
                wav_np, sr, frame_period=hop / sr * 1000
            )
            return torch.tensor(f0, dtype=torch.float32).unsqueeze(-1)
        except ImportError:
            return None

    def _load_raw_speech(self, audio_bytes: bytes):
        """加载原始波形。"""
        import io

        import torchaudio

        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(
                sr, self.sample_rate
            )(waveform)
        return waveform.squeeze(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text_token, instruct_token = self._tokenize_text(sample["text"])

        # 说话人 embedding
        embedding = sample["spk_embedding"]
        if embedding is not None:
            if isinstance(embedding, bytes):
                import numpy as np
                embedding = torch.from_numpy(
                    np.frombuffer(embedding, dtype=np.float32)
                )
            elif not isinstance(embedding, torch.Tensor):
                embedding = torch.tensor(embedding, dtype=torch.float32)
        else:
            embedding = torch.zeros(192, dtype=torch.float32)

        # 语音 token
        speech_token = sample["speech_token"]
        if speech_token is not None:
            if isinstance(speech_token, bytes):
                import numpy as np
                speech_token = torch.from_numpy(
                    np.frombuffer(speech_token, dtype=np.int32)
                )
            elif not isinstance(speech_token, torch.Tensor):
                speech_token = torch.tensor(speech_token, dtype=torch.long)
            else:
                speech_token = speech_token.long()
        else:
            speech_token = torch.tensor([], dtype=torch.long)

        # mel / pitch / 原始波形
        speech_feat = torch.zeros(0, 80, dtype=torch.float32)
        pitch_feat = None
        speech = torch.zeros(0, dtype=torch.float32)
        if sample.get("audio_data") is not None:
            audio_data = sample["audio_data"]
            if isinstance(audio_data, memoryview):
                audio_data = bytes(audio_data)
            speech_feat = self._compute_speech_feat(audio_data)
            pitch_feat = self._extract_f0(audio_data)
            speech = self._load_raw_speech(audio_data)

            # 对齐 mel 与 speech_token 时序维度
            target_mel_len = len(speech_token) * 2
            if target_mel_len > 0 and speech_feat.shape[0] > 0:
                if speech_feat.shape[0] > target_mel_len:
                    speech_feat = speech_feat[:target_mel_len]
                    if pitch_feat is not None:
                        pitch_feat = pitch_feat[:target_mel_len]
                    speech = speech[:target_mel_len * self.hop_size]
                elif speech_feat.shape[0] < target_mel_len:
                    pad_len = target_mel_len - speech_feat.shape[0]
                    speech_feat = torch.cat(
                        [speech_feat, torch.zeros(pad_len, 80)], dim=0
                    )

        return {
            "text_token": text_token,
            "text_token_len": torch.tensor(len(text_token), dtype=torch.int32),
            "instruct_token": instruct_token,
            "instruct_token_len": torch.tensor(
                len(instruct_token), dtype=torch.int32
            ),
            "speech_token": speech_token,
            "speech_token_len": torch.tensor(
                len(speech_token), dtype=torch.int32
            ),
            "speech_feat": speech_feat,
            "speech_feat_len": torch.tensor(
                speech_feat.shape[0], dtype=torch.int32
            ),
            "embedding": embedding,
            "pitch_feat": (
                pitch_feat if pitch_feat is not None
                else torch.zeros(0, 1)
            ),
            "speech": speech,
        }


class SyntheticDataset(torch.utils.data.Dataset):
    """从 .pt 文件加载预特征化数据。"""

    def __init__(self, pt_path: str):
        self.samples = torch.load(pt_path, map_location="cpu", weights_only=False)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ================================================================
#  Collate & DataLoader
# ================================================================

def _pad(tensors, pad_value=0):
    """将不等长 tensor 列表 pad 到等长并 stack。"""
    max_len = max(t.shape[0] for t in tensors)
    ndim = tensors[0].ndim
    if ndim == 1:
        return torch.stack([
            F.pad(t, (0, max_len - t.shape[0]), value=pad_value)
            for t in tensors
        ])
    return torch.stack([
        F.pad(t, (0, max_len - t.shape[0]), value=pad_value)
        for t in tensors
    ])


def collate_fn(batch: list) -> dict:
    """将变长样本 collate 为 batch dict，1D 和 2D tensor 分别 pad。"""
    result = {}
    for key in batch[0]:
        tensors = [s[key] for s in batch]
        if not isinstance(tensors[0], torch.Tensor):
            result[key] = tensors
            continue
        if tensors[0].ndim == 0:
            result[key] = torch.stack(tensors)
        else:
            result[key] = _pad(tensors)
    return result


def build_dataloader(dataset, batch_size: int, shuffle: bool):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_fn,
    )


def get_next_batch(loader_iter, loader):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


# ================================================================
#  QAT 训练工具
# ================================================================

def warmup_model(model: nn.Module, batch: dict, device):
    """触发 STE 图编译（训练态 forward + backward）。"""
    model.train()
    model.zero_grad(set_to_none=True)
    result = model.forward(batch, device)
    loss = result["loss"]
    loss.backward()
    total_grad = sum(
        p.grad.norm().item() for p in model.parameters()
        if p.grad is not None
    )
    model.zero_grad(set_to_none=True)
    return float(loss.detach()), total_grad


def train_one_step(model, batch, device, grad_accum_steps=1):
    """单步训练，支持梯度累积。返回 (loss, total_grad, extra)。"""
    result = model.forward(batch, device)
    loss = result["loss"] / grad_accum_steps
    loss.backward()
    total_grad = sum(
        p.grad.norm().item() for p in model.parameters()
        if p.grad is not None
    )
    extra = {}
    if "llm_loss" in result:
        extra["llm_loss"] = float(result["llm_loss"])
    if "flow_loss" in result:
        extra["flow_loss"] = float(result["flow_loss"])
    if "tokenizer_loss" in result:
        extra["tokenizer_loss"] = float(result["tokenizer_loss"])
    return float(loss.detach()), total_grad, extra


@torch.no_grad()
def evaluate_model(model, dataloader, device, max_batches=-1):
    """评估模型 loss，返回平均 loss。"""
    was_training = model.training
    model.eval()
    total_loss, count = 0.0, 0
    for i, batch in enumerate(dataloader):
        if 0 < max_batches <= i:
            break
        result = model.forward(batch, device)
        total_loss += float(result["loss"])
        count += 1
    if was_training:
        model.train()
    return total_loss / max(count, 1)


# ================================================================
#  工具函数
# ================================================================

def _sanitize_numpy_attrs(module: nn.Module):
    """将 Module 中 numpy 标量属性转换为 Python 原生类型。"""
    import numpy as np

    for name in dir(module):
        if name.startswith("_"):
            continue
        try:
            val = getattr(module, name)
        except AttributeError:
            continue
        if isinstance(val, (np.integer, np.floating)):
            object.__setattr__(module, name, val.item())
        elif isinstance(val, np.ndarray) and val.ndim == 0:
            object.__setattr__(module, name, val.item())
    for child in module.children():
        _sanitize_numpy_attrs(child)


@torch.no_grad()
def clip_weights_to_fp16(model: nn.Module):
    """将 FP16 权重投影到安全范围，防止 SEFP 累加器溢出。"""
    fp16_max = torch.finfo(torch.float16).max
    for param in model.parameters():
        if param.is_floating_point() and param.dtype == torch.float16:
            param.clamp_(-fp16_max * 0.5, fp16_max * 0.5)


def summarize_qmodules(model: nn.Module):
    qmodule_names, qw_names = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, QBaseModule):
            qmodule_names.append(name)
            if hasattr(mod, "qweight") and getattr(mod, "qweight") is not None:
                qw_names.append(name)
    return qmodule_names, qw_names


def save_model_weights(model: nn.Module, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(state_dict, save_path)
    print(f"saved weights to: {save_path}")


def build_float_compatible_state_dict(source, target):
    src, tgt = source.state_dict(), target.state_dict()
    return {
        k: src[k].detach().to(v.dtype)
        if k in src and src[k].shape == v.shape else v
        for k, v in tgt.items()
    }


def load_checkpoint(model: nn.Module, ckpt_path: str, tag: str = ""):
    """加载 .pt 权重文件到模型。"""
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[{tag}] missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[{tag}] unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    print(f"[{tag}] loaded checkpoint from {ckpt_path}")


# ================================================================
#  SEFP 反量化
# ================================================================

@torch.no_grad()
def dequantize_module_state_dict(quant_model, original_model):
    """从量化模型反量化所有 QBaseModule 权重。

    使用 HMFPQuantizer.true_2_fake_convert 做 SEFP 反量化，
    结果的 key 集合与 original_model.state_dict() 完全一致。
    """
    orig_sd = original_model.state_dict()
    quant_sd = quant_model.state_dict()
    result = {}

    for name, module in quant_model.named_modules():
        if not isinstance(module, QBaseModule):
            continue
        if not hasattr(module, "qweight") or module.qweight is None:
            continue

        weight_key = f"{name}.weight"
        if weight_key not in orig_sd:
            continue

        orig_shape = orig_sd[weight_key].shape
        orig_dtype = orig_sd[weight_key].dtype

        try:
            if getattr(module, "is_fast_quant", False):
                result[weight_key] = module.qweight.to(orig_dtype)
            else:
                in_features = orig_shape[1] if len(orig_shape) == 2 else orig_shape[0]
                x_fp, _ = module.w_quantizer.true_2_fake_convert(
                    module.qweight, module.scale_or_exp,
                    dim=0, is_padded=True, return_padded=False,
                    channel=in_features,
                )
                result[weight_key] = x_fp.T.contiguous().to(orig_dtype)
        except Exception as e:
            print(f"  [WARN] dequantize {weight_key} failed: {e}")
            result[weight_key] = orig_sd[weight_key]

    for key, val in orig_sd.items():
        if key not in result:
            if key in quant_sd and quant_sd[key].shape == val.shape:
                result[key] = quant_sd[key].detach().to(val.dtype)
            else:
                result[key] = val

    return result


# ================================================================
#  mel 频谱构建 (CosyVoice3 标准: 80-bin @24kHz)
# ================================================================

def make_mel_fn():
    """创建 mel 频谱提取函数 (n_fft=1920, num_mels=80, sr=24000)。"""
    import torchaudio.compliance.kaldi as kaldi

    def mel_fn(waveform):
        return kaldi.fbank(
            waveform,
            num_mel_bins=80,
            dither=0,
            sample_frequency=24000,
            frame_length=1920 / 24000 * 1000,
            frame_shift=480 / 24000 * 1000,
        )
    return mel_fn


# ================================================================
#  QAT 7 阶段流程 — 单模块
# ================================================================

def qat_train_module(
    module_name: str,
    model: nn.Module,
    train_loader,
    val_loader,
    device,
    quanted_config: dict,
    steps: int,
    eval_interval: int,
    eval_max_batches: int,
    grad_accum_steps: int,
    learning_rate: float,
    grad_clip_norm: float,
    output_dir: Path = None,
):
    """对单个模块执行完整的 QAT 训练流程 (7 阶段)。"""
    if output_dir is None:
        output_dir = Path(DEFAULT_OUTPUT_DIR)

    tag = module_name.upper()

    # ---- 阶段 0: FP 基线评估 ----
    print_stage(f"{tag}: 阶段 0 — FP 基线评估")
    fp_loss = evaluate_model(model, val_loader, device, eval_max_batches)
    print(f"[{tag}] FP baseline loss = {fp_loss:.6f}")

    # ---- 阶段 1: PTQ 准备 ----
    print_stage(f"{tag}: 阶段 1 — PTQ 准备")
    quant_model = copy.deepcopy(model).to("cpu")
    qmodel, backend = prepare_quanted_model_to_compile(
        f"cosyvoice3_{module_name}_qat", quant_model, "xh2a", quanted_config,
    )
    qmodel = qmodel.to(device)

    qm_names, qw_names = summarize_qmodules(qmodel)
    print(f"[{tag}] qmodule count = {len(qm_names)}, "
          f"modules with qweight = {len(qw_names)}")
    if qm_names:
        print(f"[{tag}] sample qmodules = {qm_names[:5]}")

    ptq_loss = float("inf")
    try:
        ptq_loss = evaluate_model(qmodel, val_loader, device, eval_max_batches)
        print(f"[{tag}] PTQ loss = {ptq_loss:.6f} "
              f"(delta vs FP = {(ptq_loss - fp_loss):+.6f})")
    except Exception as e:
        print(f"[{tag}] PTQ eval skipped: {e}")

    # ---- 阶段 2: QAT 编译 + warmup ----
    print_stage(f"{tag}: 阶段 2 — QAT 编译 + warmup")
    qmodel.train()
    compiled_model = torch_compile_quanted_model(qmodel, backend)

    train_iter = iter(train_loader)
    warmup_batch, train_iter = get_next_batch(train_iter, train_loader)
    warmup_loss, warmup_grad = warmup_model(compiled_model, warmup_batch, device)
    print(f"[{tag}] warmup loss = {warmup_loss:.6f}, "
          f"total_grad = {warmup_grad:.6f}")

    pre_loss = evaluate_model(
        compiled_model, val_loader, device, eval_max_batches
    )
    print(f"[{tag}] compiled loss before QAT = {pre_loss:.6f} "
          f"(delta vs FP = {(pre_loss - fp_loss):+.6f})")

    # ---- 阶段 3: QAT 训练 ----
    print_stage(f"{tag}: 阶段 3 — QAT 训练")
    trainable = [p for p in compiled_model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)

    num_optim_steps = math.ceil(steps / grad_accum_steps)
    warmup_steps = max(1, num_optim_steps // 10)
    eta_min = 1e-8

    def _lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, num_optim_steps - warmup_steps)
        return max(eta_min / learning_rate,
                   0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    for group in optimizer.param_groups:
        group["lr"] = learning_rate * _lr_lambda(0)
    optimizer.zero_grad(set_to_none=True)

    for step in range(steps):
        batch, train_iter = get_next_batch(train_iter, train_loader)
        train_loss, total_grad, _extra = train_one_step(
            compiled_model, batch, device, grad_accum_steps,
        )

        if math.isnan(train_loss):
            print(f"[{tag}] step={step} WARNING: NaN loss, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                compiled_model.parameters(), max_norm=grad_clip_norm
            )
            optimizer.step()
            clip_weights_to_fp16(compiled_model)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        lr = scheduler.get_last_lr()[0]
        print(f"[{tag}] step={step} lr={lr:.2e} loss={train_loss:.6f} "
              f"grad={total_grad:.6f}")

        if (step + 1) % eval_interval == 0:
            eval_loss = evaluate_model(
                compiled_model, val_loader, device, eval_max_batches
            )
            print(f"[{tag}] eval@step={step + 1} loss={eval_loss:.6f} "
                  f"(delta_vs_fp={(eval_loss - fp_loss):+.6f})")

    # ---- 阶段 4: QAT 后评估 ----
    print_stage(f"{tag}: 阶段 4 — QAT 后评估")
    post_loss = evaluate_model(
        compiled_model, val_loader, device, eval_max_batches
    )
    print(f"[{tag}] post-QAT loss = {post_loss:.6f} "
          f"(delta vs FP = {(post_loss - fp_loss):+.6f})")

    # ---- 阶段 5: 反量化权重导出 ----
    print_stage(f"{tag}: 阶段 5 — 反量化权重导出")
    export_model = copy.deepcopy(model).to("cpu")
    dequant_sd = dequantize_module_state_dict(compiled_model, export_model)
    export_model.load_state_dict(dequant_sd, strict=False)
    dequant_path = output_dir / module_name / f"dequant_steps{steps}.pt"
    save_model_weights(export_model, dequant_path)
    del export_model

    # ---- 阶段 6: 部署一致性验证 ----
    print_stage(f"{tag}: 阶段 6 — 部署一致性验证")
    qat_state_dict = {
        k: v.detach().cpu() for k, v in compiled_model.state_dict().items()
    }
    del compiled_model, optimizer
    torch.cuda.empty_cache()

    deploy_model = copy.deepcopy(model).to("cpu")
    target_sd = deploy_model.state_dict()
    deploy_sd = {
        k: qat_state_dict[k].to(v.dtype)
        if k in qat_state_dict and qat_state_dict[k].shape == v.shape else v
        for k, v in target_sd.items()
    }
    deploy_model.load_state_dict(deploy_sd, strict=False)

    deploy_config = copy.deepcopy(quanted_config)
    deploy_config["enable_qat_compiler_ops"] = False
    deploy_qmodel, _ = prepare_quanted_model_to_compile(
        f"cosyvoice3_{module_name}_deploy",
        deploy_model, "xh2a", deploy_config,
    )
    deploy_qmodel = deploy_qmodel.to(device)

    deploy_loss = float("inf")
    try:
        deploy_loss = evaluate_model(
            deploy_qmodel, val_loader, device, eval_max_batches
        )
    except Exception as e:
        print(f"[{tag}] deploy eval skipped: {e}")

    print(f"[{tag}] deploy loss = {deploy_loss:.6f}")
    print(f"[{tag}] delta vs FP  = {(deploy_loss - fp_loss):+.6f}")
    print(f"[{tag}] delta vs PTQ = {(deploy_loss - ptq_loss):+.6f}")
    print(f"[{tag}] QAT gain     = {(ptq_loss - deploy_loss):+.6f}")

    save_path = output_dir / module_name / f"qat_steps{steps}.pt"
    save_model_weights(deploy_qmodel, save_path)

    del deploy_qmodel, deploy_model
    torch.cuda.empty_cache()

    return {
        "module": module_name,
        "fp_loss": fp_loss,
        "ptq_loss": ptq_loss,
        "pre_qat_loss": pre_loss,
        "post_qat_loss": post_loss,
        "deploy_loss": deploy_loss,
    }
