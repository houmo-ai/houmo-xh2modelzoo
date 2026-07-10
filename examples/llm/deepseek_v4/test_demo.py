#!/usr/bin/env python
"""DeepSeek-V4 hmonnx demo: 用正确的 chat 格式推理, 支持 pipeline parallel (多卡加速).

用法:
  # 单卡 (auto-offload, 权重CPU, 算子GPU, 慢但稳定)
  CUDA_VISIBLE_DEVICES=1 python test_demo.py --config <meta.json>

  # 多卡 pipeline (2卡, 权重常驻, 快约 Nx)
  #   HMONNX_PIPELINE_DEVICES 给出 CUDA_VISIBLE_DEVICES 重映射后的 cuda index。
  #   --exec-device 应等于 pipeline 第 0 级（stage 0，放 embed/lm_head/首层）。
  CUDA_VISIBLE_DEVICES=1,5 HMONNX_PIPELINE_DEVICES=0,1 python test_demo.py \
      --config <meta.json> --exec-device cuda:0

  # 也可用 --pipeline-devices 显式指定（与 env 二选一）
  CUDA_VISIBLE_DEVICES=1,5 python test_demo.py --config <meta.json> \
      --pipeline-devices 0,1 --exec-device cuda:0
"""
import sys, os, argparse, torch, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
import xhquant.xhonnxruntime.config as _cfg; _cfg.disable_progress = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="meta.json 路径")
    ap.add_argument("--prompt", default="你好", help="用户输入")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--device", default="cpu", help="权重存储设备 (pipeline 模式下忽略)")
    ap.add_argument("--exec-device", default="cuda:0", help="计算设备 (pipeline 模式下须=stage0)")
    ap.add_argument(
        "--pipeline-devices",
        default=None,
        help="pipeline 并行设备 (如 '0,1')；不填则读 HMONNX_PIPELINE_DEVICES env",
    )
    args = ap.parse_args()

    meta = json.load(open(args.config))
    isl = meta["wrap_cfg"]["input_sequence_length"]

    from transformers import AutoTokenizer
    hf_config_dir = os.path.join(os.path.dirname(args.config), meta["hf_config"])
    tok = AutoTokenizer.from_pretrained(hf_config_dir, trust_remote_code=True)

    from xh_model_zoo.xh_llm.models.deepseek_v4 import DeepseekV4Inference
    from xh_model_zoo.xh_llm.models.deepseek_v4.deepseek_v4_converter import _register_std_parsers
    _register_std_parsers()
    engine = DeepseekV4Inference(
        args.config,
        fast_mode=False,
        device=args.device,
        execution_device=args.exec_device,
        pipeline_devices=args.pipeline_devices,
    )
    if engine._pp_enabled:
        print(f"=== pipeline parallel: devices={engine._pipeline_devices!r} ===", flush=True)

    # DeepSeek chat 格式
    prompt = f"<｜begin▁of▁sentence｜>User\n{args.prompt}\n\nAssistant\n"
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    real_len = ids.shape[1]
    print(f"=== prompt ({real_len} tokens): {repr(args.prompt)} ===", flush=True)

    # === prefill ===
    print("--- prefill ---", flush=True)
    data_pf = dict(input_ids=ids, past_seq_length=0)
    embeds, pos_ids, past_sl, cur_il, iid, kv_caches = engine.prepare_inputs(data_pf, isl)
    engine.init_prefill()
    with torch.no_grad():
        pf_out = engine.prefill_session(embeds, pos_ids, past_sl, cur_il, iid, *kv_caches)

    # compressed_kv
    ckv_meta = meta.get("compressed_kv_cache", {})
    ckv_layer_indices = ckv_meta.get("layer_indices", [])
    ckv_shapes = ckv_meta.get("shapes", {})
    head_dim = meta["kv_cache"]["shape"][-1]
    num_layers = meta["kv_cache"]["num_decoder_layers"]
    compressed_kv_caches = [
        torch.zeros(ckv_shapes.get(str(i), [1, 1, 1, head_dim]), dtype=torch.float16)
        for i in range(num_layers)
    ]
    if isinstance(pf_out, tuple):
        pf_logits = pf_out[0]
        n_ckv = len(ckv_layer_indices)
        ckv_outputs = pf_out[-n_ckv:] if len(pf_out) > n_ckv + 1 else pf_out[1 : 1 + n_ckv]
        for k, li in enumerate(ckv_layer_indices):
            compressed_kv_caches[li] = ckv_outputs[k]
    else:
        pf_logits = pf_out

    logits = pf_logits[0, -1, :].float().cpu() if pf_logits.dim() == 3 else pf_logits[0].float().cpu()
    next_token = logits.argmax()
    print(f"prefill token: {next_token.item()} -> {repr(tok.decode([next_token.item()]))}", flush=True)

    # 释放 prefill, 初始化 decode
    del engine.prefill_session; engine.prefill_session = None
    torch.cuda.empty_cache()
    engine.init_decode()

    # === decode ===
    generated = [next_token.item()]
    cur_token = torch.tensor([[next_token.item()]])
    print(f"\n--- decode loop (max {args.max_new_tokens}) ---", flush=True)
    for step in range(args.max_new_tokens):
        data_dec = dict(input_ids=cur_token, past_seq_length=real_len + step)
        e, p, ps, ci, ii, _ = engine.prepare_inputs(data_dec, 1)
        with torch.no_grad():
            dec_out = engine.decode_session(e, p, ps, ci, ii, *list(kv_caches), *compressed_kv_caches)
        lg = dec_out[0, -1, :].float().cpu() if dec_out.dim() == 3 else dec_out[0].float().cpu()
        next_token = lg.argmax()
        generated.append(next_token.item())
        text = tok.decode(generated)
        print(f"  step {step}: {next_token.item()} -> {repr(tok.decode([next_token.item()]))} | so_far: {text}", flush=True)
        if next_token.item() == tok.eos_token_id:
            print("  (EOS)", flush=True)
            break
        cur_token = torch.tensor([[next_token.item()]])

    print(f"\n=== 最终输出 ===\n{args.prompt}{tok.decode(generated)}", flush=True)


if __name__ == "__main__":
    main()
