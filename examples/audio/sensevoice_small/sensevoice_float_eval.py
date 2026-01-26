import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List
import jiwer

import sensevoice_common as sc


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-dir", type=str, default="/data01/nfs_shared/ASR_TTS/SenseVoiceSmall")
    p.add_argument("--onnx", type=str, required=True)
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

    p.add_argument("--report", type=str, default="work_dirs/sensevoice_small/report/float_report.json")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    t0 = time.time()
    print("=== SenseVoice float eval start ===", flush=True)
    print(
        f"model_dir={args.model_dir} onnx={args.onnx} strip_tags={args.strip_tags} limit={args.limit}",
        flush=True,
    )
    model_dir = Path(args.model_dir).expanduser().resolve()
    frontend = sc.build_frontend(model_dir)
    target_sr = int(frontend.cfg.fs)
    print(
        f"frontend: fs={frontend.cfg.fs} n_mels={frontend.cfg.n_mels} lfr_m={frontend.cfg.lfr_m} lfr_n={frontend.cfg.lfr_n}",
        flush=True,
    )

    print("loading dataset ...", flush=True)
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
    print(
        f"dataset loaded: count={len(samples)} source={('hf' if args.hf_dataset else 'local')} streaming={args.hf_streaming}",
        flush=True,
    )

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
    print(f"tokens: {tokens_path} loaded={token_list is not None}", flush=True)

    onnx_path = Path(args.onnx).expanduser().resolve()
    print(f"onnx: {onnx_path}", flush=True)

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

        if n == 0:
            print(f"first sample: id={s.audio_id} lang={s.language} textnorm={s.textnorm}", flush=True)
        wav = sc.load_audio_any(s, target_sr=target_sr)
        feat, feat_len = sc.extract_features(frontend, wav)
        inputs = sc.make_inputs_for_sample(feat, feat_len, s.language, s.textnorm)

        logits, lens = sc.run_onnx(onnx_path, inputs)
        out_len = int(lens[0]) if hasattr(lens, "__len__") else int(lens)
        token_ids = sc.ctc_greedy_decode(logits[0], out_len)
        hyp = sc.decode_token_ids(token_ids, token_list)

        # Evaluation normalization: strip tags and lowercase
        hyp_norm = sc.strip_rich_tags(hyp)
        ref_norm = sc.strip_rich_tags(ref)
        # print(f"hyp_norm: {hyp_norm}")
        # print(f"ref_norm: {ref_norm}")
        
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
        "onnx": str(onnx_path),
        "tokens": str(tokens_path),
        "model_dir": str(model_dir),
        "hf_dataset": args.hf_dataset,
        "hf_config": args.hf_config,
        "hf_split": args.hf_split,
        "hf_text_path": args.hf_text_path,
    }

    report_path = Path(args.report).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved report: {report_path}")
    print(f"total_time_sec={time.time() - t0:.2f}", flush=True)
    if args.hf_streaming:
        sc.close_hf_streaming()
        import os
        import sys

        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
