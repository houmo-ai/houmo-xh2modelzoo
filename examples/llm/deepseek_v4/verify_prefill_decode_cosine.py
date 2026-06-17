# teacher forcing decode：onnx 用 hf token 输入，排除 argmax 分叉干扰，看真实量化 cosine
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


repo = str(Path(__file__).resolve().parents[3])
sys.path.insert(0, repo)
sys.path.insert(0, str(Path(repo).parent))  # xhquanttool


def cos_sim(a, b):
    a, b = a.flatten().float(), b.flatten().float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main(args):
    from accelerate import dispatch_model, infer_auto_device_map
    from transformers import AutoTokenizer

    from xh_model_zoo.xh_llm.models.deepseek_v4 import DeepseekV4Inference
    from xh_model_zoo.xh_llm.models.deepseek_v4.deepseek_v4_converter import DeepseekV4ConverterXH2a

    with open(args.config) as f:
        meta = json.load(f)
    work_dir = Path(args.config).parent
    hf_config = str(work_dir / meta["hf_config"])
    hf_model_path = meta.get("hf_model_path") or args.hf_model
    isl = meta["wrap_cfg"]["input_sequence_length"]
    tok = AutoTokenizer.from_pretrained(hf_config, trust_remote_code=True)
    prompt = (
        "DeepSeek-V4 is a large language model developed by DeepSeek. "
        "It features MLA attention with Hyper-Connections, CSA and HCA "
        "compressors, and FP4-quantized MoE experts. The model supports "
        "long context with sliding window attention and compressed KV cache. "
        "This test verifies that the exported HMONNX model matches the "
        "original HuggingFace float model with high cosine similarity."
    )
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    real_len = ids.shape[1]
    assert real_len <= isl

    # ---- HF baseline：自回归得 hf_tokens ----
    print("--- HF baseline ---", flush=True)
    converter = DeepseekV4ConverterXH2a.__new__(DeepseekV4ConverterXH2a)
    model = converter.load_hf_model(hf_model_path)
    model = converter._fix_fp4_experts(model, hf_model_path)
    model.eval()
    # HF dispatch 避开 ONNX 所在 device（args.device），并用 free 显存（减其他进程占用），防 OOM
    _onnx_idx = int(args.device.split(":")[1]) if ":" in args.device else 0
    _max_mem = {i: torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count()) if i != _onnx_idx}
    model = dispatch_model(
        model, infer_auto_device_map(model, max_memory=_max_mem, no_split_module_classes=["DeepseekV4DecoderLayer"])
    )
    hf_tokens = []
    hf_ids = ids.clone()
    with torch.no_grad():
        out = model(hf_ids.to(model.device))
    hf_logits0 = out.logits[0, -1, :].float().cpu()
    hf_tokens.append(hf_logits0.argmax().item())
    hf_all = [hf_logits0]
    for _ in range(args.steps):
        hf_ids = torch.cat([hf_ids, torch.tensor([[hf_tokens[-1]]])], dim=1)
        with torch.no_grad():
            out = model(hf_ids.to(model.device))
        lg = out.logits[0, -1, :].float().cpu()
        hf_all.append(lg)
        hf_tokens.append(lg.argmax().item())
    del model
    torch.cuda.empty_cache()

    # ---- ONNX prefill + decode（teacher forcing: onnx 用 hf_tokens）----
    print("--- ONNX (teacher forcing) ---", flush=True)
    engine = DeepseekV4Inference(args.config, fast_mode=False, device=args.device, execution_device=args.device)
    data_pf = dict(input_ids=ids, past_seq_length=0)
    embeds, pos_ids, past_sl, cur_il, iid, kv_caches = engine.prepare_inputs(data_pf, isl)
    engine.init_prefill()
    with torch.no_grad():
        pf_out = engine.prefill_session(embeds, pos_ids, past_sl, cur_il, iid, *kv_caches)
    ckv_meta = meta.get("compressed_kv_cache", {})
    ckv_layer_indices = ckv_meta.get("layer_indices", [])
    ckv_shapes = ckv_meta.get("shapes", {})
    head_dim = meta["kv_cache"]["shape"][-1]
    num_layers = meta["kv_cache"]["num_decoder_layers"]
    compressed_kv_caches = [
        torch.zeros(ckv_shapes.get(str(i), [1, 1, 1, head_dim]), dtype=torch.float16) for i in range(num_layers)
    ]
    if isinstance(pf_out, tuple):
        pf_logits = pf_out[0]
        n_ckv = len(ckv_layer_indices)
        ckv_outputs = pf_out[-n_ckv:] if len(pf_out) > n_ckv + 1 else pf_out[1 : 1 + n_ckv]
        for k, li in enumerate(ckv_layer_indices):
            compressed_kv_caches[li] = ckv_outputs[k]
    else:
        pf_logits = pf_out
    del engine.prefill_session
    engine.prefill_session = None
    torch.cuda.empty_cache()
    engine.init_decode()
    pf = pf_logits.float().cpu()
    pf_last = pf[0, -1, :] if pf.dim() == 3 else pf[0]
    onnx_all = [pf_last]
    cur_token = torch.tensor([[hf_tokens[0]]])  # teacher forcing：用 hf 的 next token
    for step in range(args.steps):
        data_dec = dict(input_ids=cur_token, past_seq_length=real_len + step)
        e, p, ps, ci, ii, _ = engine.prepare_inputs(data_dec, 1)
        with torch.no_grad():
            dl = engine.decode_session(e, p, ps, ci, ii, *list(kv_caches), *compressed_kv_caches)
        onnx_all.append(dl[0, -1, :].float().cpu())
        if step + 1 < args.steps:
            cur_token = torch.tensor([[hf_tokens[step + 1]]])  # 继续用 hf token

    print(f"\n{'Step':>7} | {'CosSim':>9} | {'MaxDiff':>9} | argmax_match")
    print("-" * 50)
    for i in range(len(onnx_all)):
        c = cos_sim(onnx_all[i], hf_all[i])
        md = (onnx_all[i] - hf_all[i]).abs().max().item()
        onnx_tok = onnx_all[i].argmax().item()
        hf_tok = hf_all[i].argmax().item()
        label = "prefill" if i == 0 else f"dec {i}"
        print(f"{label:>7} | {c:>9.6f} | {md:>9.4f} | {'✓' if onnx_tok == hf_tok else '✗'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--hf-model", default=None)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--device", default="cuda:7")
    main(p.parse_args())
