"""
CosyVoice3 Flow 独立 QAT + 蒸馏训练
====================================

在 Pred100 QAT 之后可选运行，进一步提升 Flow 对量化 LLM 输出的鲁棒性。

核心思路:
    Teacher (FP, frozen)          Student (QAT, trainable)
         │                              │
         ├─ 输入: FP LLM 预测 token     ├─ 输入: QAT LLM 预测 token
         │                              │
         └─ 输出: velocity_pred ──KD──→│─ 输出: velocity_pred

    Loss = w_flow * flow_matching_MSE          (标准 flow matching)
         + w_distill * velocity_distill_MSE     (核心蒸馏 loss)
         + w_mu * encoder_distill_MSE           (辅助 encoder 对齐)

数据准备: 先用 predict_tokens_offline.py 分别生成 FP 和 QAT 的 pred token，
          本脚本同时加载两份数据。

用法:
    export TRAIN_DATA=.../qat_pred_tokens/train_abs_pred.list   # student
    export GT_TRAIN_DATA=.../fp_pred_tokens/train_abs_pred.list  # teacher
    export MODEL_DIR=/data01/nfs_shared/ASR_TTS/CosyVoice3-0.5B-2512
    python -u cosyvoice3_qat_flow_distill.py
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qat_utils import (
    DEFAULT_MODEL_DIR, DEFAULT_OUTPUT_DIR,
    get_env_int, get_env_float, get_env_str,
    print_stage, load_checkpoint, _sanitize_numpy_attrs,
    build_dataloader, get_next_batch,
    warmup_model, evaluate_model,
    clip_weights_to_fp16, dequantize_module_state_dict,
    save_model_weights, summarize_qmodules, make_mel_fn,
    prepare_quanted_model_to_compile, torch_compile_quanted_model,
)
from qat_module_flow import load_flow_model


# ================================================================
#  数据集 — 双 token 来源
# ================================================================

class FlowDistillDataset(torch.utils.data.Dataset):
    """加载两套 parquet：QAT LLM pred token (student) + FP LLM pred token (teacher)。

    batch 输出:
        speech_token      — QAT LLM pred → student Flow
        speech_token_gt   — FP LLM pred → teacher Flow
        speech_token_len  — 共享
        speech_feat       — 共享 80-bin mel
        speech_feat_len   — 共享
        embedding         — 共享 speaker embedding
    """

    def __init__(self, student_list, teacher_list, mel_fn,
                 sample_rate=24000, max_audio_sec=10.0, max_samples=-1):
        self.mel_fn = mel_fn
        self.sample_rate = sample_rate
        self.max_samples = int(max_audio_sec * sample_rate)
        self.samples_s = self._load_index(student_list, max_samples)
        self.samples_t = self._load_index(teacher_list, max_samples)
        assert len(self.samples_s) == len(self.samples_t), \
            f"student ({len(self.samples_s)}) != teacher ({len(self.samples_t)})"

    @staticmethod
    def _load_index(data_list_path, max_samples):
        import pyarrow.parquet as pq

        files = [l.strip() for l in open(data_list_path) if l.strip()]
        samples = []
        for pf in files:
            for _, row in pq.read_table(pf).to_pandas().iterrows():
                samples.append(row)
                if 0 < max_samples <= len(samples):
                    return samples
        return samples

    @staticmethod
    def _bytes_to_tensor(val, dtype):
        import numpy as np
        if isinstance(val, (bytes, memoryview)):
            raw = bytes(val)
            # parquet 中 token 保存为 int32，embedding 保存为 float32
            # 按最小单元读取再转换到目标 dtype
            base_np = np.float32 if dtype.is_floating_point else np.int32
            arr = np.frombuffer(raw, dtype=base_np).copy()
            return torch.from_numpy(arr).to(dtype)
        if isinstance(val, np.ndarray):
            return torch.from_numpy(val.copy()).to(dtype)
        return torch.tensor(val, dtype=dtype)

    @staticmethod
    def _load_audio(audio_bytes):
        import io, torchaudio
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes))
        if sr != 24000:
            waveform = torchaudio.transforms.Resample(sr, 24000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        return waveform.squeeze(0)

    def __len__(self):
        return len(self.samples_s)

    def __getitem__(self, idx):
        rs, rt = self.samples_s[idx], self.samples_t[idx]
        audio_data = rs.get("audio_data")
        if isinstance(audio_data, memoryview):
            audio_data = bytes(audio_data)

        speech = self._load_audio(audio_data)
        if self.max_samples > 0 and speech.shape[0] > self.max_samples:
            speech = speech[:self.max_samples]

        speech_feat = self.mel_fn(speech.unsqueeze(0))  # (T_mel, 80) — kaldi.fbank 返回 2D
        st_s = self._bytes_to_tensor(rs["speech_token"], torch.long)
        st_t = self._bytes_to_tensor(rt["speech_token"], torch.long)
        emb = self._bytes_to_tensor(rs["spk_embedding"], torch.float32)

        # 对齐 mel 与 student token 长度 (token_mel_ratio=2)
        target_mel_len = len(st_s) * 2
        if speech_feat.shape[0] > target_mel_len:
            speech_feat = speech_feat[:target_mel_len, :]
        elif speech_feat.shape[0] < target_mel_len:
            pad = target_mel_len - speech_feat.shape[0]
            speech_feat = torch.cat([speech_feat, torch.zeros(pad, 80)], dim=0)

        return {
            "speech_token": st_s,
            "speech_token_gt": st_t,
            "speech_token_len": torch.tensor(len(st_s), dtype=torch.int32),
            "speech_feat": speech_feat,  # (T_mel, 80)
            "speech_feat_len": torch.tensor(speech_feat.shape[0], dtype=torch.int32),
            "embedding": emb,
        }


# ================================================================
#  Flow encoder 辅助函数
# ================================================================

def _encode_flow(flow, token, token_len, device):
    """token → (mu, mask)，通过 Flow 的 encoder 路径。"""
    from cosyvoice.utils.mask import make_pad_mask

    mask = (~make_pad_mask(token_len)).float().unsqueeze(-1).to(device)
    tok = flow.input_embedding(torch.clamp(token, min=0).to(device)) * mask
    h = flow.pre_lookahead_layer(tok)
    h = h.repeat_interleave(flow.token_mel_ratio, dim=1)
    mask = mask.repeat_interleave(flow.token_mel_ratio, dim=1).squeeze(-1)
    return h.transpose(1, 2).contiguous(), mask.unsqueeze(1)


def _sample_conds(feat, feat_len, device):
    """随机采样 mel prefix 条件。"""
    conds = torch.zeros(feat.shape, device=device)
    for i, j in enumerate(feat_len):
        if random.random() < 0.5:
            continue
        index = random.randint(0, int(0.3 * j.item()))
        conds[i, :index] = feat[i, :index]
    return conds.transpose(1, 2)


# ================================================================
#  Flow 蒸馏 Wrapper
# ================================================================

class FlowDistillWrapper(nn.Module):
    """Flow teacher-student QAT wrapper.

    Student: QAT-prepared Flow (trainable)
    Teacher: FP32 Flow (frozen, via object.__setattr__)
    """

    def __init__(self, student_flow, w_flow=1.0, w_distill=1.0, w_mu=0.1):
        super().__init__()
        self.student_flow = student_flow
        self.w_flow = w_flow
        self.w_distill = w_distill
        self.w_mu = w_mu
        object.__setattr__(self, '_teacher_flow', None)

    def set_teacher(self, teacher_flow):
        teacher_flow.eval()
        for p in teacher_flow.parameters():
            p.requires_grad = False
        object.__setattr__(self, '_teacher_flow', teacher_flow)

    def forward(self, batch, device):
        token_s = batch["speech_token"].to(device)
        token_t = batch["speech_token_gt"].to(device)
        token_len = batch["speech_token_len"].to(device)
        feat = batch["speech_feat"].to(device)
        feat_len = batch["speech_feat_len"].to(device)
        embedding = batch["embedding"].to(device)

        # ---- 共享随机量 ----
        streaming = random.random() < 0.5
        embedding_norm = F.normalize(embedding, dim=1)
        conds = _sample_conds(feat, feat_len, device)

        # ---- Student 编码 ----
        spks_s = self.student_flow.spk_embed_affine_layer(embedding_norm)
        mu_s, mask = _encode_flow(self.student_flow, token_s, token_len, device)

        # ---- Teacher 编码 (no_grad) ----
        teacher = object.__getattribute__(self, '_teacher_flow')
        with torch.no_grad():
            spks_t = teacher.spk_embed_affine_layer(embedding_norm)
            mu_t, _ = _encode_flow(teacher, token_t, token_len, device)

        # ---- 共享 z, t ----
        x1 = feat.transpose(1, 2).contiguous()
        z = torch.randn_like(x1)
        t = torch.rand([x1.shape[0], 1, 1], device=device, dtype=x1.dtype)
        sigma = self.student_flow.decoder.sigma_min
        y = (1 - (1 - sigma) * t) * z + t * x1
        u = x1 - (1 - sigma) * z

        # ---- 共享 cfg_mask ----
        b = x1.shape[0]
        cfg_mask = torch.rand(b, device=device) > self.student_flow.decoder.training_cfg_rate
        mu_s_cfg = mu_s * cfg_mask.view(-1, 1, 1)
        mu_t_cfg = mu_t * cfg_mask.view(-1, 1, 1)
        spks_s_cfg = spks_s * cfg_mask.view(-1, 1)
        spks_t_cfg = spks_t * cfg_mask.view(-1, 1)
        conds_cfg = conds * cfg_mask.view(-1, 1, 1)

        # ---- Student velocity ----
        pred_s = self.student_flow.decoder.estimator(
            y, mask, mu_s_cfg, t.squeeze(), spks_s_cfg, conds_cfg,
            streaming=streaming,
        )

        # ---- Teacher velocity (no_grad) ----
        with torch.no_grad():
            pred_t = teacher.decoder.estimator(
                y, mask, mu_t_cfg, t.squeeze(), spks_t_cfg, conds_cfg,
                streaming=streaming,
            )

        # ---- Losses ----
        mask_sum = torch.sum(mask) * u.shape[1]
        flow_loss = F.mse_loss(pred_s * mask, u * mask, reduction="sum") / mask_sum
        distill_loss = F.mse_loss(pred_s * mask, pred_t * mask, reduction="sum") / mask_sum
        mu_loss = F.mse_loss(mu_s * mask, mu_t * mask, reduction="sum") / mask_sum

        total = (self.w_flow * flow_loss
                 + self.w_distill * distill_loss
                 + self.w_mu * mu_loss)

        return {
            "loss": total,
            "flow_loss": flow_loss.detach(),
            "distill_loss": distill_loss.detach(),
            "mu_loss": mu_loss.detach(),
        }


# ================================================================
#  QAT 训练流程
# ================================================================

def _quant_config(w_man_bit=8, deploy=False):
    return {
        "precision_mode": "aligned",
        "enable_qat_compiler_ops": not deploy,
        "w_schema": {
            "fp_mode": "sefp", "man_bit": w_man_bit,
            "hidden_bit": True, "nshare": 64, "rounding": "rne",
        },
        "act_schema": {
            "fp_mode": "sefp", "man_bit": 8,
            "hidden_bit": True, "nshare": 64, "rounding": "rne",
            "max_exp_boost": 0,
        },
    }


def qat_train_flow_distill(
    student_flow,
    teacher_flow,
    train_loader,
    val_loader,
    device,
    quanted_config,
    steps,
    eval_interval,
    eval_max_batches,
    grad_accum_steps,
    learning_rate,
    grad_clip_norm,
    w_flow,
    w_distill,
    w_mu,
    output_dir,
):
    tag = "FLOW-DISTILL"

    # ---- 组装 wrapper ----
    wrapper = FlowDistillWrapper(student_flow, w_flow, w_distill, w_mu)
    wrapper.set_teacher(teacher_flow)
    wrapper = wrapper.to(device)
    # teacher 不是 registered submodule，需手动移到 device
    object.__setattr__(wrapper, '_teacher_flow',
        object.__getattribute__(wrapper, '_teacher_flow').to(device))

    # ---- 阶段 0: FP 基线 ----
    skip_eval = get_env_int("SKIP_EVAL", 0)
    if not skip_eval:
        print_stage(f"{tag}: 阶段 0 — FP 基线评估")
        fp_loss = evaluate_model(wrapper, val_loader, device, eval_max_batches)
        print(f"[{tag}] FP baseline loss = {fp_loss:.6f}")
    else:
        print(f"[{tag}] 阶段 0 — FP 基线评估 (SKIPPED)")

    # ---- 阶段 1: PTQ prepare ----
    print_stage(f"{tag}: 阶段 1 — PTQ prepare")
    teacher_fp32 = object.__getattribute__(wrapper, '_teacher_flow')
    object.__setattr__(wrapper, '_teacher_flow', None)

    quant_model = copy.deepcopy(wrapper).to("cpu")
    qmodel, backend = prepare_quanted_model_to_compile(
        "cosyvoice3_flow_distill", quant_model, "xh2a", quanted_config)
    qmodel = qmodel.to(device)
    object.__setattr__(qmodel, '_teacher_flow', teacher_fp32.to(device))
    object.__setattr__(wrapper, '_teacher_flow', teacher_fp32)

    qm_names, qw_names = summarize_qmodules(qmodel)
    print(f"[{tag}] qmodule={len(qm_names)}, qweight={len(qw_names)}")

    if not skip_eval:
        try:
            ptq_loss = evaluate_model(qmodel, val_loader, device, eval_max_batches)
            print(f"[{tag}] PTQ loss = {ptq_loss:.6f}")
        except Exception as e:
            print(f"[{tag}] PTQ eval skipped: {e}")
    else:
        print(f"[{tag}] PTQ 评估 (SKIPPED)")

    # ---- 阶段 2: QAT compile + warmup ----
    print_stage(f"{tag}: 阶段 2 — QAT compile + warmup")
    qmodel.train()
    compiled = torch_compile_quanted_model(qmodel, backend)

    train_iter = iter(train_loader)
    wb, train_iter = get_next_batch(train_iter, train_loader)
    wl, wg = warmup_model(compiled, wb, device)
    print(f"[{tag}] warmup loss={wl:.6f} grad={wg:.6f}")

    # ---- 阶段 3: QAT 训练 ----
    print_stage(f"{tag}: 阶段 3 — QAT 训练 ({steps} steps)")
    trainable = [p for p in compiled.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)

    num_optim_steps = math.ceil(steps / grad_accum_steps)
    warmup_steps = max(1, num_optim_steps // 10)

    def _lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, num_optim_steps - warmup_steps)
        return max(1e-8 / learning_rate,
                   0.5 * (1 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    for g in optimizer.param_groups:
        g["lr"] = learning_rate * _lr_lambda(0)
    optimizer.zero_grad(set_to_none=True)

    for step in range(steps):
        batch, train_iter = get_next_batch(train_iter, train_loader)
        result = compiled.forward(batch, device)
        loss = result["loss"] / grad_accum_steps
        loss.backward()

        if math.isnan(loss.item()):
            print(f"[{tag}] step={step} NaN, skip")
            optimizer.zero_grad(set_to_none=True)
            continue

        total_grad = sum(p.grad.norm().item()
                         for p in compiled.parameters() if p.grad is not None)

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(compiled.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            clip_weights_to_fp16(compiled)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        lr = scheduler.get_last_lr()[0]
        ex = {k: f"{v:.4f}" for k, v in result.items() if k != "loss"}
        extra_str = " ".join(f"{k}={v}" for k, v in ex.items())
        print(f"[{tag}] step={step} lr={lr:.2e} loss={loss.item():.6f} "
              f"grad={total_grad:.6f} {extra_str}")

        if not skip_eval and (step + 1) % eval_interval == 0:
            ev = evaluate_model(compiled, val_loader, device, eval_max_batches)
            print(f"[{tag}] eval@{step + 1} loss={ev:.6f}")

    # ---- 阶段 4: QAT 后评估 ----
    if not skip_eval:
        print_stage(f"{tag}: 阶段 4 — QAT 后评估")
        post_loss = evaluate_model(compiled, val_loader, device, eval_max_batches)
        print(f"[{tag}] post-QAT loss = {post_loss:.6f}")
    else:
        print(f"[{tag}] 阶段 4 — QAT 后评估 (SKIPPED)")

    # ---- 阶段 5: 反量化导出 ----
    print_stage(f"{tag}: 阶段 5 — 反量化导出")
    export_model = copy.deepcopy(student_flow).to("cpu")
    compiled_submodules = dict(compiled.named_modules())
    compiled_sub = compiled_submodules.get("student_flow")
    if compiled_sub is not None:
        dequant_sd = dequantize_module_state_dict(compiled_sub, export_model)
    else:
        print(f"[{tag}] WARN: student_flow not in compiled, raw sd")
        dequant_sd = export_model.state_dict()
    export_model.load_state_dict(dequant_sd, strict=False)

    save_path = output_dir / "flow" / f"dequant_steps{steps}.pt"
    save_model_weights(export_model, save_path)
    print(f"[{tag}] saved: {save_path}")
    del export_model

    # ---- 阶段 6: 部署验证 ----
    print_stage(f"{tag}: 阶段 6 — 部署验证")
    qat_sd = {k: v.detach().cpu() for k, v in compiled.state_dict().items()}
    del compiled, optimizer
    torch.cuda.empty_cache()

    deploy_config = _quant_config(quanted_config["w_schema"]["man_bit"], deploy=True)
    deploy_model = copy.deepcopy(student_flow).to("cpu")
    target_sd = deploy_model.state_dict()
    deploy_sd = {}
    for k, v in target_sd.items():
        full_key = "student_flow." + k
        if full_key in qat_sd and qat_sd[full_key].shape == v.shape:
            deploy_sd[k] = qat_sd[full_key].to(v.dtype)
        else:
            deploy_sd[k] = v
    deploy_model.load_state_dict(deploy_sd, strict=False)

    deploy_qmodel, _ = prepare_quanted_model_to_compile(
        "cosyvoice3_flow_distill_deploy", deploy_model, "xh2a", deploy_config)
    qat_path = output_dir / "flow" / f"qat_steps{steps}.pt"
    save_model_weights(deploy_qmodel.cpu(), qat_path)
    print(f"[{tag}] saved: {qat_path}")

    print(f"\n[{tag}] 完成")


# ================================================================
#  主函数
# ================================================================

def main():
    seed = get_env_int("SEED", 42)
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = get_env_str("MODEL_DIR", DEFAULT_MODEL_DIR)
    hf_model_dir = get_env_str("HF_MODEL_DIR", os.path.join(model_dir, "CosyVoice-BlankEN"))
    yaml_path = get_env_str("YAML_PATH", os.path.join(model_dir, "cosyvoice3.yaml"))

    train_data = get_env_str("TRAIN_DATA", "")
    cv_data = get_env_str("CV_DATA", "")
    gt_train_data = get_env_str("GT_TRAIN_DATA", "")
    gt_cv_data = get_env_str("GT_CV_DATA", "")

    batch_size = get_env_int("BATCH_SIZE", 1)
    train_steps = get_env_int("TRAIN_STEPS", 5000)
    eval_interval = get_env_int("EVAL_INTERVAL", 500)
    eval_max_batches = get_env_int("EVAL_MAX_BATCHES", 20)
    grad_accum = get_env_int("GRAD_ACCUM_STEPS", 2)
    grad_clip = get_env_float("GRAD_CLIP_NORM", 5.0)
    lr = get_env_float("LEARNING_RATE", 1e-5)
    w_man_bit = get_env_int("W_MAN_BIT", 8)

    w_flow = get_env_float("W_FLOW", 0.5)
    w_distill = get_env_float("W_DISTILL", 1.0)
    w_mu = get_env_float("W_MU", 0.1)

    output_dir = Path(get_env_str("OUTPUT_DIR", "./output_cosyvoice3_qat_flow_distill"))

    # ---- 打印配置 ----
    print_stage("配置信息")
    print(f"device          = {device}")
    print(f"model_dir       = {model_dir}")
    print(f"train_data      = {train_data or '(未指定)'}")
    print(f"gt_train_data   = {gt_train_data or '(未指定)'}")
    print(f"cv_data         = {cv_data or '(未指定)'}")
    print(f"gt_cv_data      = {gt_cv_data or '(未指定)'}")
    print(f"batch_size      = {batch_size}")
    print(f"train_steps     = {train_steps}")
    print(f"learning_rate   = {lr}")
    print(f"w_flow={w_flow} w_distill={w_distill} w_mu={w_mu}")
    print(f"output_dir      = {output_dir}")

    for req in ("TRAIN_DATA", "CV_DATA", "GT_TRAIN_DATA", "GT_CV_DATA"):
        if not get_env_str(req, ""):
            print(f"\n[ERROR] 必须指定 {req} 环境变量")
            return

    teacher_flow_weights = get_env_str("TEACHER_FLOW_WEIGHTS", "")

    # ---- 加载 student Flow ----
    print_stage("加载 Student Flow")
    student_flow = load_flow_model(yaml_path, hf_model_dir)
    load_checkpoint(student_flow, os.path.join(model_dir, "flow.pt"), tag="FLOW-STUDENT")
    _sanitize_numpy_attrs(student_flow)
    print(f"  params = {sum(p.numel() for p in student_flow.parameters()):,}")

    # ---- 加载 teacher Flow (FP) ----
    print_stage("加载 Teacher Flow (FP)")
    teacher_flow = load_flow_model(yaml_path, hf_model_dir)
    teacher_ckpt = teacher_flow_weights or os.path.join(model_dir, "flow.pt")
    load_checkpoint(teacher_flow, teacher_ckpt, tag="FLOW-TEACHER")
    _sanitize_numpy_attrs(teacher_flow)
    print(f"  params = {sum(p.numel() for p in teacher_flow.parameters()):,}")
    print(f"  weights = {teacher_ckpt}")

    # ---- 数据 ----
    print_stage("加载数据")
    mel_fn = make_mel_fn()
    train_ds = FlowDistillDataset(train_data, gt_train_data, mel_fn)
    val_ds = FlowDistillDataset(cv_data, gt_cv_data, mel_fn)
    print(f"  train = {len(train_ds)}, val = {len(val_ds)}")

    train_loader = build_dataloader(train_ds, batch_size, shuffle=True)
    val_loader = build_dataloader(val_ds, batch_size, shuffle=False)

    # ---- 训练 ----
    quanted_config = _quant_config(w_man_bit)
    qat_train_flow_distill(
        student_flow=student_flow,
        teacher_flow=teacher_flow,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        quanted_config=quanted_config,
        steps=train_steps,
        eval_interval=eval_interval,
        eval_max_batches=eval_max_batches,
        grad_accum_steps=grad_accum,
        learning_rate=lr,
        grad_clip_norm=grad_clip,
        w_flow=w_flow,
        w_distill=w_distill,
        w_mu=w_mu,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
