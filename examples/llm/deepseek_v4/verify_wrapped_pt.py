# ================================================================== #
#  File: _verify_wrapped_pt.py                                        #
#  Description:                                                       #
#    HF float vs wrapped PT float 的 prefill last-token cosine。       #
#    二分定位：根因在 wrapping（attention / HC / MoE clamp）还是       #
#    下游（量化 / 导出 / NPU runtime）。                                #
#                                                                     #
#    Phase 1 HF : accelerate 分片到多 GPU（单卡装不下 46GB；HF MoE 用   #
#                 aten::_grouped_mm，仅 CUDA 实现）。                    #
#    Phase 2 wrapped : CPU（xh2modelzoo MoeBlock 非 grouped_mm，可跑；  #
#                 且 dispatch_model 对 wrapped 的 KV-cache-list/多 kwarg #
#                 forward 兼容性未知，CPU 最稳）。                        #
# ================================================================== #

import faulthandler
import gc
import sys
import traceback
from pathlib import Path

import torch
import torch.nn.functional as F


REPO = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, REPO)
sys.path.insert(0, str(Path(REPO).parent))  # xhquanttool

HF_MODEL_PATH = "/data01/nfs_shared/llm/DeepSeek-V4-Flash-slim5l"
SEQ_LEN = 256
SEED = 42

# ------------------------------------------------------------------ #
#  诊断：确保崩溃时 traceback 一定可见（曾因 stdout 缓冲被吞）           #
# ------------------------------------------------------------------ #
faulthandler.enable()
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
print("=== SCRIPT START ===", flush=True)


def cos(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def load_and_fix():
    """加载 HF + FP4 dequant（保留 HF 原生 dtype 分配）。"""
    from xh_model_zoo.xh_llm.models.deepseek_v4.deepseek_v4_converter import (
        DeepseekV4ConverterXH2a,
        _register_std_parsers,
    )

    _register_std_parsers()
    conv = DeepseekV4ConverterXH2a.__new__(DeepseekV4ConverterXH2a)
    m = conv.load_hf_model(HF_MODEL_PATH)
    m = conv._fix_fp4_experts(m, HF_MODEL_PATH)
    m.eval()
    return m


def dispatch_to_gpus(model):
    """accelerate 自动按各 GPU 空闲内存分片。"""
    from accelerate import dispatch_model, infer_auto_device_map

    dm = infer_auto_device_map(
        model,
        max_memory=None,
        no_split_module_classes=["DeepseekV4DecoderLayer"],
    )
    return dispatch_model(model, dm), dm


def main():
    torch.manual_seed(SEED)
    input_ids = torch.randint(0, 1000, (1, SEQ_LEN), dtype=torch.long)
    print(f"input_ids[:10]: {input_ids[0, :10].tolist()}", flush=True)

    # ================================================================ #
    #  Phase 1: HF float baseline（多 GPU 分片）                         #
    # ================================================================ #
    print("\n=== Phase 1: HF float ===", flush=True)
    hf_model = load_and_fix()
    hf_model, dm = dispatch_to_gpus(hf_model)
    print(f"HF device_map: {dm}", flush=True)
    in_dev = next(hf_model.parameters()).device
    with torch.no_grad():
        logits_hf = hf_model(input_ids.to(in_dev)).logits[0, -1, :].float().cpu()
    print(f"HF top1={logits_hf.argmax().item()}  norm={logits_hf.norm().item():.3f}", flush=True)
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()

    # ================================================================ #
    #  Phase 2: wrapped PT float（CPU）                                  #
    # ================================================================ #
    print("\n=== Phase 2: wrapped PT float (CPU) ===", flush=True)
    from xh_model_zoo.xh_llm.models.builder import wrap_llm_model
    from xh_model_zoo.xh_llm.models.deepseek_v4._layers import register_wrap_modules
    from xhquant.api import CacheTensor, Config

    dev = "cpu"
    raw = load_and_fix()
    cksl = {}
    for i, lt in enumerate(raw.config.layer_types):
        if lt != "sliding_attention":
            cksl[i] = SEQ_LEN // raw.model.layers[i].self_attn.compressor.compress_rate

    register_wrap_modules()
    wrap_cfg = Config(
        dict(
            batch_size=1,
            max_sequence_length=2048,
            input_sequence_length=SEQ_LEN,
            use_cache=True,
            num_logits_to_keep=0,
            kv_cache=dict(cache_axis=2, context_length=2048),
            compressed_kv_seq_lens=cksl,
        )
    )
    wrapped = wrap_llm_model(raw, wrap_cfg).eval()
    # eager forward 要求 dtype 一致：HF 的 fp32 weight（o_a_projs 等 keep_in_fp32
    # 模块）→ bf16。保留 int/bool buffer（tid2eid / causal_threshold / future_mask）。
    # 仅服务于本二分工具的 eager 运行；导出时量化器另行定 dtype，与此无关。
    for p in wrapped.parameters():
        if p.is_floating_point() and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    for buf in wrapped.buffers():
        if buf.is_floating_point() and buf.dtype != torch.bfloat16:
            buf.data = buf.data.to(torch.bfloat16)
    wrapped = wrapped.to(dev)

    # Phase 2 eager dtype 统一：activation 经 softmax/norm 后可能停留在 fp32，
    # 而 weight 已 cast bf16 → matmul/linear 报 dtype mismatch。
    # 量化器导出时会插 Cast 对齐 dtype（故导出不受影响）；eager 下用
    # monkey-patch 模拟：cast 第二参数(weight)→ 第一参数(activation)dtype。
    _mm, _lin = torch.matmul, torch.nn.functional.linear

    def _matmul(a, b, *args, **kw):
        if a.is_floating_point() and b.is_floating_point() and a.dtype != b.dtype:
            b = b.to(a.dtype)
        return _mm(a, b, *args, **kw)

    def _linear(x, w, bias=None):
        if x.is_floating_point() and w.is_floating_point() and x.dtype != w.dtype:
            w = w.to(x.dtype)
        return _lin(x, w, bias)

    torch.matmul = _matmul
    torch.nn.functional.linear = _linear

    n_layers = wrapped.config.num_hidden_layers
    head_dim = wrapped.config.head_dim
    nkv = wrapped.config.num_key_value_heads

    embed = wrapped.model.embed_tokens(input_ids.to(dev))
    pos = torch.arange(SEQ_LEN, device=dev).unsqueeze(0)
    psl = torch.tensor([0], dtype=torch.int32, device=dev)
    cil = torch.tensor([SEQ_LEN], dtype=torch.int32, device=dev)
    iid = input_ids.to(dev)
    kv = [CacheTensor(torch.zeros([1, nkv, 2048, head_dim], dtype=torch.bfloat16, device=dev)) for _ in range(n_layers)]

    with torch.no_grad():
        hs, _ = wrapped.model(
            inputs_embeds=embed,
            position_ids=pos,
            past_seq_length=psl,
            current_input_length=cil,
            input_ids=iid,
            past_key_caches=kv,
        )
        logits_wr = wrapped.lm_head(hs)[0, -1, :].float().cpu()
    print(f"wrapped top1={logits_wr.argmax().item()}  norm={logits_wr.norm().item():.3f}", flush=True)

    # ================================================================ #
    #  对比                                                             #
    # ================================================================ #
    print("\n=== HF vs wrapped PT (float) ===", flush=True)
    c = cos(logits_hf, logits_wr)
    print(f"Cosine : {c:.6f}", flush=True)
    print(f"MaxDiff: {(logits_hf - logits_wr).abs().max().item():.6f}", flush=True)
    verdict = (
        "wrapping 一致（根因在下游量化/导出）" if c > 0.99 else "wrapping 发散（根因在 wrapping：attention/HC/MoE）"
    )
    print(f"判定: {verdict}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        print("=== SCRIPT CRASHED ===", flush=True)
        sys.exit(1)
