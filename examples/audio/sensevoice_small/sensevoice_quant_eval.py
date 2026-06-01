# Copyright 2025 HOUMO AI
#
# File: sensevoice_quant_eval.py
# Description:
#   Example script: audio/sensevoice_small/sensevoice_quant_eval.py
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
import time
from pathlib import Path
from typing import Any, Dict, List
import os
os.environ["USE_TRITON_MATMUL"] = "1"

import sensevoice_common as sc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-dir", type=str, default="/data01/nfs_shared/ASR_TTS/SenseVoiceSmall")
    p.add_argument("--hmonnx", type=str, required=True)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--tokens", type=str, default="")
    p.add_argument("--strip-tags", action="store_true")
    p.add_argument("--limit", type=int, default=0)

    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest-jsonl", type=str, default="")
    g.add_argument("--wav-scp", type=str, default="")
    g.add_argument("--hf-dataset", type=str, default="")

    p.add_argument("--text", type=str, default="", help="required when using --wav-scp")
    p.add_argument("--hf-config", type=str, default="")
    p.add_argument("--hf-split", type=str, default="test")
    p.add_argument("--hf-streaming", action="store_true")
    p.add_argument("--hf-audio-field", type=str, default="audio")
    p.add_argument("--hf-text-field", type=str, default="")
    p.add_argument("--hf-text-path", type=str, default="")

    p.add_argument("--report", type=str, default="work_dirs/sensevoice_small/report/quant_report.json")
    p.add_argument("--fast", action="store_true", help="Enable fast mode (disable progress bars)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    
    from xhquant.xhonnxruntime import config as xhonnxruntime_config
    xhonnxruntime_config.disable_progress = True
    xhonnxruntime_config.verbose_progress = False

    model_dir = Path(args.model_dir).expanduser().resolve()
    frontend = sc.build_frontend(model_dir)
    target_sr = int(frontend.cfg.fs)

    if args.manifest_jsonl:
        samples = sc.read_manifest_jsonl(Path(args.manifest_jsonl).expanduser().resolve())
    elif args.wav_scp:
        if not args.text:
            raise ValueError("--text is required when using --wav-scp")
        samples = sc.read_wav_scp_text(
            Path(args.wav_scp).expanduser().resolve(), Path(args.text).expanduser().resolve()
        )
    else:
        samples = sc.load_hf_dataset(
            dataset=args.hf_dataset,
            config=args.hf_config,
            split=args.hf_split,
            limit=int(args.limit),
            streaming=bool(args.hf_streaming),
            audio_field=args.hf_audio_field,
            text_field=args.hf_text_field,
        )

    if int(args.limit) > 0:
        samples = samples[: int(args.limit)]

    text_map: Dict[str, str] = {}
    if args.hf_text_path:
        text_path = Path(args.hf_text_path).expanduser().resolve()
        with text_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                k, v = line.split(maxsplit=1)
                text_map[k] = v

    tokens_path = Path(args.tokens).expanduser().resolve() if args.tokens else (model_dir / "tokens.json")
    token_list = sc.load_tokens(tokens_path)

    hmonnx_path = Path(args.hmonnx).expanduser().resolve()

    total_cer_dist = 0
    total_cer_len = 0
    total_wer_dist = 0
    total_wer_len = 0
    n = 0
    per_sample: List[Dict[str, Any]] = []

    last_t = time.time()
    log_interval = 10

    for s in samples:
        ref = s.text
        if text_map and s.audio_id:
            utt_id = s.audio_id.split("/")[-1]
            ref = text_map.get(utt_id, ref)
        if not ref.strip():
            print(f"Skipping empty reference text for sample {s.audio_id}")
            continue

        wav = sc.load_audio_any(s, target_sr=target_sr)
        feat, feat_len = sc.extract_features(frontend, wav)
        inputs = sc.make_inputs_for_sample(feat, feat_len, s.language, s.textnorm)

        logits, lens = sc.run_hmonnx(hmonnx_path, inputs, device=args.device, fast=args.fast)
        out_len = int(lens[0]) if hasattr(lens, "__len__") else int(lens)
        token_ids = sc.ctc_greedy_decode(logits[0], out_len)
        hyp = sc.decode_token_ids(token_ids, token_list)
        
        # Evaluation normalization: strip tags and lowercase
        hyp_norm = sc.strip_rich_tags(hyp)
        ref_norm = sc.strip_rich_tags(ref)
        # print(f"ref_norm: {ref_norm}")
        # print(f"hyp_norm: {hyp_norm}")
        
        # Calculate CER stats (corpus level)
        r_chars = list(ref_norm.replace(" ", ""))
        h_chars = list(hyp_norm.replace(" ", ""))
        cer_dist = sc.edit_distance(r_chars, h_chars)
        total_cer_dist += cer_dist
        total_cer_len += len(r_chars)
        
        # Calculate WER stats (corpus level)
        r_words = [w for w in ref_norm.lower().split() if w]
        h_words = [w for w in hyp_norm.lower().split() if w]
        wer_dist = sc.edit_distance(r_words, h_words)
        total_wer_dist += wer_dist
        total_wer_len += len(r_words)

        cer_v = cer_dist / max(1, len(r_chars))
        wer_v = wer_dist / max(1, len(r_words))

        n += 1
        if n % log_interval == 0:
            now = time.time()
            print(
                f"processed {n}/{len(samples)} avg_sec_per_sample={(now - last_t) / log_interval:.3f} last_id={s.audio_id}",
                flush=True,
            )
            last_t = now
        per_sample.append(
            {
                "audio": s.audio_id or str(s.audio),
                "ref": ref,
                "hyp": hyp,
                "token_ids": token_ids,
                "cer": cer_v,
                "wer": wer_v,
                "feat_len": int(feat_len),
                "out_len": int(out_len),
                "language": s.language,
                "textnorm": s.textnorm,
            }
        )

    summary = {
        "num_samples": n,
        "cer_avg": total_cer_dist / max(1, total_cer_len),
        "wer_avg": total_wer_dist / max(1, total_wer_len),
    }
    report = {
        "summary": summary,
        "per_sample": per_sample,
        "hmonnx": str(hmonnx_path),
        "tokens": str(tokens_path),
        "model_dir": str(model_dir),
        "hf_dataset": args.hf_dataset,
        "hf_config": args.hf_config,
        "hf_split": args.hf_split,
        "hf_text_path": args.hf_text_path,
        "device": args.device,
    }

    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved report: {report_path}")
    if args.hf_streaming:
        sc.close_hf_streaming()
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
