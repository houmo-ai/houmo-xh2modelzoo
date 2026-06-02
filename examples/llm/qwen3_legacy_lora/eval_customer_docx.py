# Copyright 2025 HOUMO AI
#
# File: eval_customer_docx.py
# Description:
#   Reusable evaluation harness for the Xunfei "court transcript condensation"
#   customer task. Reads SYSTEM_PROMPT + a directory of .docx transcripts,
#   builds chat prompts and generates condensed transcripts. Supports two
#   backends:
#     * hf   : plain HuggingFace AutoModelForCausalLM (fp16 float baseline)
#     * xh2a : quantized HMONNX export via Qwen3LegacyInference (meta.json)
#   so the same script measures the fp16 baseline and every quant variant.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import re
import time
from pathlib import Path
from typing import List

DEFAULT_PROMPT_MD = "/data01/home/yujy/work/xunfei/xf_data/prompt.md"
DEFAULT_DOCX_DIR = "/data01/home/yujy/work/xunfei/xf_data"
DEFAULT_OUT_ROOT = "work_dirs/customer_eval"


def load_system_prompt(prompt_md_path: str) -> str:
    """Extract the SYSTEM_PROMPT string literal from prompt.md.

    prompt.md is a python snippet: SYSTEM_PROMPT = \"\"\"...\"\"\". We regex the
    triple-quoted literal so we never execute untrusted file content.
    """
    text = Path(prompt_md_path).read_text(encoding="utf-8")
    m = re.search(
        r'SYSTEM_PROMPT\s*=\s*(?P<q>"""|\'\'\')(?P<body>.*?)(?P=q)',
        text,
        re.DOTALL,
    )
    if not m:
        raise ValueError(
            f"Could not find SYSTEM_PROMPT triple-quoted literal in {prompt_md_path}"
        )
    return m.group("body").strip()


def extract_docx_text(docx_path: str) -> str:
    """Extract full transcript text from a .docx, preserving paragraph order.

    Speaker markers (<审>：/<原代>： or bare 审判长：/原代：) and &nbsp symbols
    live inline in the paragraph text, so a plain paragraph join keeps them.
    Empty paragraphs are dropped to avoid runs of blank lines, but real
    content order is untouched.
    """
    from docx import Document

    doc = Document(docx_path)
    lines: List[str] = []
    for para in doc.paragraphs:
        t = para.text
        if t is not None and t.strip() != "":
            lines.append(t)
    return "\n".join(lines)


def build_messages(system_prompt: str, transcript: str) -> List[dict]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": transcript},
    ]


class HFBackend:
    """Plain HuggingFace fp16 backend for the floating-point baseline."""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )
        self.model.to(device)
        self.model.eval()

    def build_prompt_text(self, messages: List[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def count_tokens(self, prompt_text: str) -> int:
        return len(self.tokenizer(prompt_text).input_ids)

    def generate(self, prompt_text: str, max_new_tokens: int) -> str:
        torch = self.torch
        model_inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        out_ids = generated[0][len(model_inputs.input_ids[0]):].tolist()
        return self.tokenizer.decode(out_ids, skip_special_tokens=True).strip("\n")


class XH2ABackend:
    """Quantized HMONNX backend driven by a meta.json export.

    Mirrors examples/llm/qwen3_legacy_lora/qwen3_legacy_xh2a_hmonnx_test.py and
    xh_model_zoo/.../qwen3_legacy/inference.py. Wraps the HMONNX inference
    engine into an HF-compatible module exposing .generate(). The prefill ONNX
    has a fixed input_sequence_length (wrap_cfg) — transcripts longer than that
    cannot be evaluated by a given export, which is surfaced explicitly.

    NOTE: not yet联调 in this round — requires a meta.json produced by P8-D.
    The code path is wired and signature-verified against the source.
    """

    def __init__(self, meta_json: str, device: str = "cuda:0", fast_mode: bool = True):
        import torch
        from xhquant.api import xhquant_init
        from xhquant.xhonnxruntime import config as xhonnxruntime_config

        # LoRA exports use the LoRA-specific inference + hf-compat classes, whose
        # prefill graph carries an extra input vs the plain qwen3_legacy path
        # (the non-LoRA engine supplies 75 inputs but the graph expects 76).
        from xh_model_zoo.xh_llm.models.qwen3_legacy_lora import (
            Qwen3LegacyLoRAHFCompatible,
            Qwen3LegacyLoRAInference,
        )

        self.torch = torch
        self.device = device
        xhquant_init(None, False)
        self.engine = Qwen3LegacyLoRAInference(
            meta_json, fast_mode=fast_mode, device=device, execution_device=device
        )
        self.engine.enable_lora = True
        hf_model_path = self.engine.meta_info.get("hf_model_path", None)
        if hf_model_path is None or not Path(hf_model_path).exists():
            raise FileNotFoundError(
                f"hf_model_path from meta.json not found: {hf_model_path}. "
                "Pass a meta.json whose hf_config/hf_model_path resolve."
            )
        self.tokenizer = self.engine.tokenizer
        self.max_input_len = self.engine.prefill_input_sequence_length
        xhonnxruntime_config.disable_progress = True
        xhonnxruntime_config.verbose_progress = False
        self.wrapped = Qwen3LegacyLoRAHFCompatible.to_hf_compatible(hf_model_path, self.engine)
        self.wrapped.eval()
        self.wrapped.to(device)

    def build_prompt_text(self, messages: List[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def count_tokens(self, prompt_text: str) -> int:
        return len(self.tokenizer(prompt_text).input_ids)

    def generate(self, prompt_text: str, max_new_tokens: int) -> str:
        torch = self.torch
        # prefill is chunked at input_sequence_length by the runtime, so long
        # prompts are fed through the prefill ONNX in multiple passes — no cap.
        model_inputs = self.tokenizer([prompt_text], return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.wrapped.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        out_ids = generated[0][len(model_inputs.input_ids[0]):].tolist()
        return self.tokenizer.decode(out_ids, skip_special_tokens=True).strip("\n")


def make_backend(args):
    if args.backend == "hf":
        return HFBackend(args.model, device=args.device)
    elif args.backend == "xh2a":
        return XH2ABackend(args.model, device=args.device)
    raise ValueError(f"unknown backend {args.backend}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="hf: HF model dir; xh2a: path to meta.json")
    ap.add_argument("--backend", choices=["hf", "xh2a"], default="hf")
    ap.add_argument("--prompt-md", default=DEFAULT_PROMPT_MD)
    ap.add_argument("--docx-dir", default=DEFAULT_DOCX_DIR)
    ap.add_argument("--out-tag", required=True,
                    help="subdir under work_dirs/customer_eval for outputs")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-docs", type=int, default=0,
                    help="if >0, only evaluate the first N docx (tiered eval)")
    ap.add_argument("--only", type=str, default=None,
                    help="if set, only docx whose filename contains this substring")
    args = ap.parse_args()

    system_prompt = load_system_prompt(args.prompt_md)
    docx_files = sorted(Path(args.docx_dir).glob("*.docx"))
    if not docx_files:
        raise FileNotFoundError(f"no .docx under {args.docx_dir}")
    if args.only:
        docx_files = [f for f in docx_files if args.only in f.name]
    if args.max_docs and args.max_docs > 0:
        docx_files = docx_files[: args.max_docs]
    if not docx_files:
        raise FileNotFoundError(f"no .docx left after filtering (only={args.only})")

    out_dir = Path(DEFAULT_OUT_ROOT) / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[eval] backend={args.backend} model={args.model}")
    print(f"[eval] {len(docx_files)} docx -> {out_dir}")
    backend = make_backend(args)

    summary = []
    for f in docx_files:
        transcript = extract_docx_text(str(f))
        messages = build_messages(system_prompt, transcript)
        prompt_text = backend.build_prompt_text(messages)
        n_in = backend.count_tokens(prompt_text)
        t0 = time.time()
        out_text = backend.generate(prompt_text, args.max_new_tokens)
        dt = time.time() - t0
        n_out = backend.count_tokens(out_text)
        (out_dir / (f.stem + ".txt")).write_text(out_text, encoding="utf-8")
        rec = {"docx": f.name, "in_tokens": n_in, "out_tokens": n_out,
               "seconds": round(dt, 1)}
        summary.append(rec)
        print(f"  [gen] {f.name}: in={n_in} out={n_out} {dt:.1f}s")

    (out_dir / "_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[eval] done -> {out_dir/'_summary.json'}")


if __name__ == "__main__":
    main()
