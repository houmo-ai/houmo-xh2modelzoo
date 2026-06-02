#!/usr/bin/env python3
"""Stream-decode a single quantized xh2a LoRA model on a given prompt.
Pass: model meta.json (or HF model dir) + optional prompt text.
Prints tokens as they are generated (streaming), plus per-step timing.

Examples:
  # Use the 5-docx system prompt + a specific docx
  python stream_decode.py \\
      --model work_dirs/qwen3_xf_base-XH2a-16k-w8a16h1_sefp-lora-common/meta.json \\
      --docx /data01/home/yujy/work/xunfei/xf_data/2026津0114行初1号-2-全文笔录-改-标注_I5NfCYP.docx \\
      --max-new-tokens 4096

  # Use a custom prompt
  python stream_decode.py \\
      --model work_dirs/qwen3_xf_base-XH2a-16k-w8a16h1_sefp-lora-common/meta.json \\
      --prompt "请用100字介绍中华人民共和国的首都。" \\
      --max-new-tokens 200
"""
import argparse
import os
import sys
import time
from pathlib import Path

# ---- CLI ----
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="path to meta.json (xh2a) OR HF model dir (hf)")
    ap.add_argument("--backend", choices=["xh2a", "hf"], default="xh2a")
    ap.add_argument("--prompt", default=None, help="raw prompt text (overrides --docx)")
    ap.add_argument("--docx", default=None, help="docx file; uses default docx dir if just a stem")
    ap.add_argument("--prompt-md", default="/data01/home/yujy/work/xunfei/xf_data/prompt.md",
                    help="system prompt source")
    ap.add_argument("--docx-dir", default="/data01/home/yujy/work/xunfei/xf_data",
                    help="where to look for docx by stem if --docx is not a path")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--use-cuda-graph", action="store_true",
                    help="enable HMONNXCUDAGraphInference (XH2A_CUDA_GRAPH=1)")
    ap.add_argument("--save", default=None,
                    help="if set, save full output to this path")
    ap.add_argument("--token-log", default=None,
                    help="if set, append per-token (id, t_ms) CSV lines to this path")
    return ap.parse_args()


# ---- Prompt construction (mirrors eval_customer_docx) ----
def load_system_prompt(prompt_md_path):
    import re
    text = Path(prompt_md_path).read_text(encoding="utf-8")
    m = re.search(r"SYSTEM_PROMPT\s*=\s*(?P<q>\"\"\"|\'\'\')(?P<body>.*?)(?P=q)", text, re.DOTALL)
    if not m:
        raise ValueError(f"no SYSTEM_PROMPT in {prompt_md_path}")
    return m.group("body").strip()


def build_messages(system_prompt, transcript):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ]


def extract_docx_text(docx_path):
    from docx import Document
    doc = Document(docx_path)
    return "\n".join(p.text for p in doc.paragraphs if p.text and p.text.strip())


# ---- Streaming wrapper ----
# Subclass the standard `transformers.TextStreamer` (HF 4.x `BaseStreamer`
# interface: `put(value)` + `end()`). This aligns with
# `qwen3_legacy_xh2a_hmonnx_test.py:25,98` which already uses the standard
# TextStreamer. We override `on_finalized_text` to additionally append a
# (token_id, t_ms) CSV row per generated token (preserves the old `--token-log`
# capability).
from transformers import TextStreamer  # noqa: E402  (placed after helpers for clarity)


class TokenCsvStreamer(TextStreamer):
    def __init__(self, tokenizer, log_path=None, **decode_kwargs):
        # skip_prompt=True: the first `put()` call from `.generate()` carries the
        # full input_ids (prompt); TextStreamer drops it, and we likewise never
        # want prompt tokens to show up in the CSV.
        super().__init__(tokenizer, skip_prompt=True, **decode_kwargs)
        self.log_f = open(log_path, "a") if log_path else None
        self.t_start = None
        self._pending_ids = []  # token ids waiting to be flushed to CSV

    def put(self, value):
        # Capture the new token ids BEFORE delegating to super (which mutates
        # internal cache state). `value` is a 1D tensor of new tokens (one step
        # worth; could be >1 for batched/beam cases).
        if value.dim() > 1:
            value = value[0]
        new_ids = value.tolist()
        # Replicate the parent's "first call = prompt, drop it" gate locally so
        # we never accumulate prompt ids into `_pending_ids`.
        is_prompt_call = self.skip_prompt and self.next_tokens_are_prompt
        if not is_prompt_call:
            self._pending_ids.extend(new_ids)
            if self.t_start is None:
                self.t_start = time.time()
        super().put(value)

    def on_finalized_text(self, text: str, stream_end: bool = False):
        # Mirror the default TextStreamer text output: print to stdout, newline
        # only on stream_end.
        print(text, flush=True, end="" if not stream_end else None)
        # Flush any token ids that have now been "finalized" — one CSV row each.
        if self.log_f is not None and self._pending_ids:
            t_ms = (time.time() - self.t_start) * 1000
            for tid in self._pending_ids:
                self.log_f.write(f"{tid},{t_ms:.1f}\n")
            self._pending_ids = []
            self.log_f.flush()
        if stream_end and self.log_f is not None and not self.log_f.closed:
            self.log_f.close()

    def close(self):
        # Backward-compat shim with the previous `streamer.close()` call site;
        # safe to call even after `end()` (idempotent).
        if self.log_f is not None and not self.log_f.closed:
            self.log_f.close()


# ---- Main ----
def main():
    args = parse_args()

    # Optional CUDA graph
    if args.use_cuda_graph and args.backend == "xh2a":
        os.environ["XH2A_CUDA_GRAPH"] = "1"

    import torch
    from xhquant.api import xhquant_init

    if args.backend == "xh2a":
        from xh_model_zoo.xh_llm.models.qwen3_legacy_lora import (
            Qwen3LegacyLoRAHFCompatible, Qwen3LegacyLoRAInference,
        )
    else:
        from transformers import AutoModelForCausalLM

    xhquant_init(None, False)

    # ---- Build prompt ----
    if args.prompt:
        prompt_text = args.prompt
    else:
        # try as full docx path
        p = Path(args.docx) if args.docx else None
        if p and p.is_file():
            docx_path = p
        elif p:
            # treat as stem
            docx_path = Path(args.docx_dir) / f"{p.name}.docx"
            if not docx_path.exists():
                # try matching by partial stem
                cands = list(Path(args.docx_dir).glob(f"*{p.name}*.docx"))
                if not cands:
                    raise FileNotFoundError(f"no docx matches {args.docx}")
                docx_path = cands[0]
        else:
            raise SystemExit("either --prompt or --docx required")
        sys_prompt = load_system_prompt(args.prompt_md)
        transcript = extract_docx_text(str(docx_path))
        msgs = build_messages(sys_prompt, transcript)
        # build chat template via tokenizer at engine init time
        # we need tokenizer first, so load engine below then template
        prompt_text = None
        transcript_msg = msgs

    # ---- Load engine ----
    t0 = time.time()
    if args.backend == "xh2a":
        engine = Qwen3LegacyLoRAInference(args.model, fast_mode=True,
                                         device=args.device, execution_device=args.device)
        engine.enable_lora = True
        tok = engine.tokenizer
        hf_model_path = engine.meta_info.get("hf_model_path", None)
        if hf_model_path is None or not Path(hf_model_path).exists():
            raise FileNotFoundError("meta.json hf_model_path not resolvable")
        wrapped = Qwen3LegacyLoRAHFCompatible.to_hf_compatible(hf_model_path, engine)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.model)
        wrapped = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)
    wrapped.eval(); wrapped.to(args.device)
    print(f"[load] engine ready in {time.time()-t0:.1f}s", file=sys.stderr)
    print(f"[load] device={args.device} cuda_graph={'ON' if args.use_cuda_graph else 'OFF'}", file=sys.stderr)

    # ---- Finalize prompt via tokenizer chat template ----
    if prompt_text is None:
        prompt_text = tok.apply_chat_template(
            transcript_msg, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
    mi = tok(prompt_text, return_tensors="pt").to(args.device)
    n_in = mi.input_ids.shape[-1]
    print(f"[load] prompt in-tokens: {n_in}", file=sys.stderr)

    # ---- Stream ----
    streamer = TokenCsvStreamer(tok, log_path=args.token_log)
    t0 = time.time()
    print(f"\n[gen] start (max_new_tokens={args.max_new_tokens})\n---", flush=True)
    try:
        with torch.no_grad():
            out = wrapped.generate(
                **mi,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                streamer=streamer,
            )
    finally:
        streamer.close()
    dt = time.time() - t0
    n_out = out.shape[-1] - n_in
    print(f"\n---\n[gen] done: {n_out} new tokens in {dt:.1f}s = {n_out/max(dt,1e-3):.1f} tok/s", file=sys.stderr)

    # save full output
    if args.save:
        text = tok.decode(out[0][n_in:], skip_special_tokens=True).strip()
        Path(args.save).write_text(text, encoding="utf-8")
        print(f"[save] wrote {args.save} ({len(text)} chars)", file=sys.stderr)


if __name__ == "__main__":
    main()
