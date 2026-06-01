"""
CosyVoice3 LLM + Flow 端到端 QAT (Pred100 模式)
=================================================

主训练脚本。联合训练 LLM 和 Flow，核心设计：Flow 接收 LLM argmax 预测
的 speech token 而非 ground truth，使训练/推理分布对齐。

架构:
    LLMWithPredTokens — 在 criterion_ce 上挂 hook 捕获 logits，
                        提取 speech token 位置的 argmax 预测
    LLMFlowPred100Wrapper — 联合 wrapper:
        1. LLM forward → loss + predicted speech tokens
        2. predicted tokens (.detach()) → Flow forward → loss
        3. total_loss = w_llm * llm_loss + w_flow * flow_loss

关键: pred_tokens 经过 .detach()，梯度不会从 Flow 回传到 LLM，
两个模块独立优化。

用法:
    conda activate xhquant
    export MODEL_DIR=/path/to/CosyVoice3-0.5B-2512
    export TRAIN_DATA=data/train.list CV_DATA=data/dev.list
    python cosyvoice3_qat_e2e_llm_pred100.py
"""

import copy
import math
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qat_utils import (
    DEFAULT_MODEL_DIR,
    DEFAULT_OUTPUT_DIR,
    get_env_int,
    get_env_float,
    get_env_str,
    print_stage,
    load_checkpoint,
    _sanitize_numpy_attrs,
    build_dataloader,
    get_next_batch,
    warmup_model,
    train_one_step,
    evaluate_model,
    clip_weights_to_fp16,
    dequantize_module_state_dict,
    save_model_weights,
    summarize_qmodules,
    make_mel_fn,
    prepare_quanted_model_to_compile,
    torch_compile_quanted_model,
    CosyVoiceDataset,
)
from qat_module_llm import load_llm_model
from qat_module_flow import load_flow_model


class LLMWithPredTokens(nn.Module):
    """在 LLM forward 上挂 hook，从 criterion_ce 的输入中捕获 logits + lm_target，
    提取 speech token 位置的 argmax 预测，暴露给 LLMFlowPred100Wrapper。

    CosyVoice3LM.forward 内部流程:
        logits = self.llm_decoder(lm_output)   # (B, T_total, vocab)
        loss   = self.criterion_ce(logits, lm_target)
    其中 lm_target 在 speech token 位置为真实 ID，其余为 IGNORE_ID=-1。
    本 adapter 通过 hook 捕获这两个张量，提取 speech 位置的 argmax 预测。
    """

    _IGNORE_ID = -1

    def __init__(self, llm):
        super().__init__()
        self.llm = llm
        self._logits = None
        self._target = None
        llm.criterion_ce.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        self._logits = inputs[0]   # (B, T_total, vocab)
        self._target = inputs[1]   # (B, T_total)

    def forward(self, batch, device):
        out = self.llm.forward(batch, device)

        if self._logits is None or self._target is None:
            return out

        pred_all = self._logits.argmax(dim=-1)                    # (B, T_total)

        # LLM 词表 = speech_token_size + 200（含 sos/eos/task_id 等特殊 token），
        # 但 Flow 的 input_embedding 只有 speech_token_size 个条目，
        # 超出范围的预测 token 会导致 embedding 越界。
        max_token_id = self.llm.speech_token_size - 1
        pred_all = pred_all.clamp(max=max_token_id)

        speech_mask = self._target != self._IGNORE_ID             # (B, T_total)
        speech_token_len = batch['speech_token_len'].to(device)

        pred_tokens = []
        for i in range(speech_token_len.shape[0]):
            pos = speech_mask[i].nonzero(as_tuple=True)[0]
            n = int(speech_token_len[i].item())
            if len(pos) >= n:
                pred_tokens.append(pred_all[i, pos[:n]])
            else:
                # 兜底: 回退到 GT token
                gt = batch['speech_token'][i, :n].to(device)
                pred_tokens.append(gt)

        out['predict_speech_token'] = torch.nn.utils.rnn.pad_sequence(
            pred_tokens, batch_first=True, padding_value=0,
        )
        return out


class LLMFlowPred100Wrapper(nn.Module):
    def __init__(self, llm, flow, w_llm=1.0, w_flow=1.0, allow_gt_fallback=False):
        super().__init__()
        self.llm = llm
        self.flow = flow
        self.w_llm = float(w_llm)
        self.w_flow = float(w_flow)
        self.allow_gt_fallback = bool(allow_gt_fallback)

    def _extract_from_out(self, llm_out):
        token_keys = (
            "pred_speech_token",
            "speech_token_pred",
            "predict_speech_token",
            "generated_speech_token",
            "speech_token",
        )
        for key in token_keys:
            val = llm_out.get(key)
            if torch.is_tensor(val):
                if val.ndim == 1:
                    val = val.unsqueeze(0)
                if val.ndim == 2:
                    return val.long()

        logit_keys = (
            "speech_token_logits",
            "speech_logits",
            "logits",
            "llm_logits",
        )
        for key in logit_keys:
            val = llm_out.get(key)
            if torch.is_tensor(val) and val.ndim >= 3:
                return val.argmax(dim=-1).long()
        return None

    def _extract_from_method(self, batch, device):
        method_names = (
            "predict_speech_token",
            "inference_speech_token",
            "generate_speech_token",
        )
        for name in method_names:
            fn = getattr(self.llm, name, None)
            if fn is None:
                continue
            try:
                out = fn(batch, device)
            except TypeError:
                try:
                    out = fn(batch)
                except Exception:
                    continue
            except Exception:
                continue
            if torch.is_tensor(out):
                return out.long()
            if isinstance(out, dict):
                for key in ("speech_token", "pred_speech_token", "speech_token_pred"):
                    val = out.get(key)
                    if torch.is_tensor(val):
                        return val.long()
        return None

    def _get_pred_tokens(self, llm_out, batch, device):
        pred = self._extract_from_out(llm_out)
        if pred is None:
            pred = self._extract_from_method(batch, device)
        if pred is None:
            if self.allow_gt_fallback:
                pred = batch["speech_token"].to(device).long()
            else:
                keys = sorted(list(llm_out.keys()))
                raise RuntimeError(
                    f"LLM 输出中未找到预测 token 或 logits，keys={keys}"
                )
        if pred.ndim == 1:
            pred = pred.unsqueeze(0)
        return pred

    def forward(self, batch, device):
        llm_out = self.llm.forward(batch, device)
        llm_loss = llm_out["loss"]

        pred_token = self._get_pred_tokens(llm_out, batch, device)
        pred_len = batch["speech_token_len"].to(device).long().clamp(
            min=0, max=pred_token.shape[1]
        )

        flow_batch = dict(batch)
        flow_batch["speech_token"] = pred_token.detach()
        flow_batch["speech_token_len"] = pred_len

        flow_out = self.flow.forward(flow_batch, device)
        flow_loss = flow_out["loss"]

        total = self.w_llm * llm_loss + self.w_flow * flow_loss
        return {
            "loss": total,
            "llm_loss": llm_loss.detach(),
            "flow_loss": flow_loss.detach(),
        }


def qat_train_llm_flow_pred100(
    wrapper,
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
    output_dir,
    skip_eval=False,
):
    tag = "LLM+FLOW-PRED100"

    # ---- 阶段 0: FP 基线评估 ----
    fp_loss = float("nan")
    if not skip_eval:
        print_stage(f"{tag}: 阶段 0 — FP 基线评估")
        fp_loss = evaluate_model(wrapper, val_loader, device, eval_max_batches)
        print(f"[{tag}] FP baseline loss = {fp_loss:.6f}")
    else:
        print_stage(f"{tag}: 阶段 0 — FP 基线评估 (跳过)")

    # ---- 阶段 1: PTQ 准备 ----
    print_stage(f"{tag}: 阶段 1 — PTQ 准备")
    quant_model = copy.deepcopy(wrapper).to("cpu")
    qmodel, backend = prepare_quanted_model_to_compile(
        "cosyvoice3_llm_flow_pred100_qat", quant_model, "xh2a", quanted_config
    )
    qmodel = qmodel.to(device)

    qm_names, qw_names = summarize_qmodules(qmodel)
    print(f"[{tag}] qmodule count = {len(qm_names)}, modules with qweight = {len(qw_names)}")

    ptq_loss = float("inf")
    if not skip_eval:
        try:
            ptq_loss = evaluate_model(qmodel, val_loader, device, eval_max_batches)
            print(f"[{tag}] PTQ loss = {ptq_loss:.6f}")
        except Exception as e:
            print(f"[{tag}] PTQ eval skipped: {e}")

    # ---- 阶段 2: QAT 编译 + warmup ----
    print_stage(f"{tag}: 阶段 2 — QAT 编译 + warmup")
    qmodel.train()
    compiled = torch_compile_quanted_model(qmodel, backend)

    train_iter = iter(train_loader)
    wb, train_iter = get_next_batch(train_iter, train_loader)
    wl, wg = warmup_model(compiled, wb, device)
    print(f"[{tag}] warmup loss = {wl:.6f}, grad = {wg:.6f}")

    if not skip_eval:
        pre_loss = evaluate_model(compiled, val_loader, device, eval_max_batches)
        print(f"[{tag}] compiled pre-QAT loss = {pre_loss:.6f}")

    print_stage(f"{tag}: 阶段 3 — QAT 训练")
    trainable = [p for p in compiled.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=learning_rate)

    num_steps = math.ceil(steps / grad_accum_steps)
    warmup_steps = max(1, num_steps // 10)

    def _lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        prog = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        return max(1e-8 / learning_rate, 0.5 * (1 + math.cos(math.pi * prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    for g in optimizer.param_groups:
        g["lr"] = learning_rate * _lr_lambda(0)
    optimizer.zero_grad(set_to_none=True)

    for step in range(steps):
        batch, train_iter = get_next_batch(train_iter, train_loader)
        tl, tg, ex = train_one_step(compiled, batch, device, grad_accum_steps)

        if math.isnan(tl):
            print(f"[{tag}] step={step} NaN, skip")
            optimizer.zero_grad(set_to_none=True)
            continue

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(compiled.parameters(), max_norm=grad_clip_norm)
            optimizer.step()
            clip_weights_to_fp16(compiled)
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        lr = scheduler.get_last_lr()[0]
        extra_str = " ".join(f"{k}={v:.4f}" for k, v in ex.items())
        print(f"[{tag}] step={step} lr={lr:.2e} loss={tl:.6f} grad={tg:.6f} {extra_str}")

        if (step + 1) % eval_interval == 0:
            ev = evaluate_model(compiled, val_loader, device, eval_max_batches)
            print(f"[{tag}] eval@{step + 1} loss={ev:.6f}")

    # ---- 阶段 4: QAT 后评估 ----
    post_loss = float("nan")
    if not skip_eval:
        print_stage(f"{tag}: 阶段 4 — QAT 后评估")
        post_loss = evaluate_model(compiled, val_loader, device, eval_max_batches)
        print(f"[{tag}] post-QAT loss = {post_loss:.6f}")
    else:
        print_stage(f"{tag}: 阶段 4 — QAT 后评估 (跳过)")

    print_stage(f"{tag}: 阶段 5 — 反量化权重导出")
    modules_to_export = {
        "llm": ("llm", wrapper.llm),
        "flow": ("flow", wrapper.flow),
    }
    compiled_submodules = dict(compiled.named_modules())

    for mod_name, (prefix, orig_module) in modules_to_export.items():
        export_model = copy.deepcopy(orig_module).to("cpu")
        compiled_sub = compiled_submodules.get(prefix)
        if compiled_sub is not None:
            dequant_sd = dequantize_module_state_dict(compiled_sub, export_model)
        else:
            print(f"[{tag}] WARN: {prefix} not found in compiled model, using raw sd")
            dequant_sd = export_model.state_dict()
        export_model.load_state_dict(dequant_sd, strict=False)
        save_model_weights(export_model, output_dir / mod_name / f"dequant_steps{steps}.pt")
        del export_model

    print_stage(f"{tag}: 阶段 6 — 部署一致性验证 + 保存")
    qat_sd = {k: v.detach().cpu() for k, v in compiled.state_dict().items()}
    del compiled, optimizer
    torch.cuda.empty_cache()

    deploy_config = copy.deepcopy(quanted_config)
    deploy_config["enable_qat_compiler_ops"] = False

    for mod_name, (prefix, orig_module) in modules_to_export.items():
        deploy_model = copy.deepcopy(orig_module).to("cpu")
        target_sd = deploy_model.state_dict()
        deploy_sd = {}
        full_prefix = prefix + "."
        for k, v in target_sd.items():
            full_key = full_prefix + k
            if full_key in qat_sd and qat_sd[full_key].shape == v.shape:
                deploy_sd[k] = qat_sd[full_key].to(v.dtype)
            else:
                deploy_sd[k] = v
        deploy_model.load_state_dict(deploy_sd, strict=False)

        deploy_qmodel, _ = prepare_quanted_model_to_compile(
            f"cosyvoice3_{mod_name}_pred100_deploy", deploy_model, "xh2a", deploy_config
        )
        save_model_weights(deploy_qmodel.cpu(), output_dir / mod_name / f"qat_steps{steps}.pt")
        del deploy_model, deploy_qmodel

    print(f"[{tag}] 完成。LLM/Flow 权重已保存到 {output_dir}")
    return {
        "fp_loss": fp_loss,
        "ptq_loss": ptq_loss,
        "post_qat_loss": post_loss,
    }


def main():
    seed = get_env_int("SEED", 42)
    torch.manual_seed(seed)
    random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = get_env_str("MODEL_DIR", DEFAULT_MODEL_DIR)
    hf_model_dir = get_env_str("HF_MODEL_DIR", os.path.join(model_dir, "CosyVoice-BlankEN"))
    yaml_path = get_env_str("YAML_PATH", os.path.join(model_dir, "cosyvoice3.yaml"))
    output_dir = Path(get_env_str("OUTPUT_DIR", DEFAULT_OUTPUT_DIR + "_pred100"))

    train_data = get_env_str("TRAIN_DATA", "")
    cv_data = get_env_str("CV_DATA", "")
    batch_size = get_env_int("BATCH_SIZE", 1)
    train_steps = get_env_int("TRAIN_STEPS", 200)
    eval_interval = get_env_int("EVAL_INTERVAL", 50)
    eval_max_batches = get_env_int("EVAL_MAX_BATCHES", 20)
    grad_accum = get_env_int("GRAD_ACCUM_STEPS", 2)
    grad_clip = get_env_float("GRAD_CLIP_NORM", 5.0)
    lr = get_env_float("LEARNING_RATE", 1e-5)

    w_llm = get_env_float("LLM_LOSS_WEIGHT", 1.0)
    w_flow = get_env_float("FLOW_LOSS_WEIGHT", 1.0)
    allow_gt_fallback = get_env_int("ALLOW_GT_FALLBACK", 0) == 1
    skip_eval = get_env_int("SKIP_EVAL", 0) == 1

    w_man_bit = get_env_int("W_MAN_BIT", 8)
    quanted_config = {
        "precision_mode": "aligned",
        "enable_qat_compiler_ops": True,
        "w_schema": {
            "fp_mode": "sefp",
            "man_bit": w_man_bit,
            "hidden_bit": True,
            "nshare": 64,
            "rounding": "rne",
        },
        "act_schema": {
            "fp_mode": "sefp",
            "man_bit": 8,
            "hidden_bit": True,
            "nshare": 64,
            "rounding": "rne",
            "max_exp_boost": 0,
        },
    }

    print_stage("配置信息")
    print(f"device          = {device}")
    print(f"model_dir       = {model_dir}")
    print(f"output_dir      = {output_dir}")
    print(f"train_data      = {train_data or '(未指定)'}")
    print(f"cv_data         = {cv_data or '(未指定)'}")
    print(f"batch_size      = {batch_size}")
    print(f"train_steps     = {train_steps}")
    print(f"weights         = llm={w_llm} flow={w_flow}")
    print(f"allow_gt_fallback = {allow_gt_fallback}")
    print(f"skip_eval        = {skip_eval}")

    if not train_data or not cv_data:
        print("\n[ERROR] 必须指定 TRAIN_DATA 和 CV_DATA")
        return

    print_stage("加载数据")
    from transformers import AutoTokenizer

    text_tokenizer = AutoTokenizer.from_pretrained(hf_model_dir, trust_remote_code=True)
    mel_fn = make_mel_fn()

    train_ds = CosyVoiceDataset(train_data, text_tokenizer, mel_fn)
    val_ds = CosyVoiceDataset(cv_data, text_tokenizer, mel_fn)
    print(f"train samples = {len(train_ds)}")
    print(f"val samples   = {len(val_ds)}")

    train_loader = build_dataloader(train_ds, batch_size, shuffle=True)
    val_loader = build_dataloader(val_ds, batch_size, shuffle=False)

    print_stage("加载模型")
    print("[LLM] 加载 ...")
    llm = load_llm_model(yaml_path, hf_model_dir)
    load_checkpoint(llm, os.path.join(model_dir, "llm.pt"), tag="LLM")
    _sanitize_numpy_attrs(llm)
    llm = llm.to(device)

    # hook 捕获 logits + lm_target，提取 speech token 位置的预测
    llm = LLMWithPredTokens(llm)
    print("[LLM] 已挂载 PredTokens adapter (hook on criterion_ce)")

    print("[Flow] 加载 ...")
    flow = load_flow_model(yaml_path, hf_model_dir)
    load_checkpoint(flow, os.path.join(model_dir, "flow.pt"), tag="FLOW")
    _sanitize_numpy_attrs(flow)
    flow = flow.to(device)

    print_stage("组装 Wrapper")
    wrapper = LLMFlowPred100Wrapper(
        llm=llm,
        flow=flow,
        w_llm=w_llm,
        w_flow=w_flow,
        allow_gt_fallback=allow_gt_fallback,
    ).to(device)

    qat_train_llm_flow_pred100(
        wrapper=wrapper,
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
        output_dir=output_dir,
        skip_eval=skip_eval,
    )


if __name__ == "__main__":
    main()
