import argparse
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
from xhquant.api import set_random_seed

from hmonnx_utils import DEFAULT_TEXT, infer_request, load_export_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--work-dir", type=str, default=None, help="optional exported work dir used for defaults")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--language", type=str, default="Chinese")
    parser.add_argument("--speaker", type=str, default=None)
    parser.add_argument("--instruct", type=str, default=None)
    parser.add_argument("--ref-audio", type=str, default=None)
    parser.add_argument("--ref-text", type=str, default=None)
    parser.add_argument("--xvec-only", action="store_true")
    parser.add_argument("--output", type=str, default="native_output.wav")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--attn", type=str, default="sdpa")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=2048)
    parser.add_argument("--do-sample", "--do_sample", dest="do_sample", action="store_true", default=True)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=50)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", "--repetition_penalty", dest="repetition_penalty", type=float, default=1.05)
    parser.add_argument("--subtalker-dosample", "--subtalker_dosample", dest="subtalker_dosample", action="store_true", default=True)
    parser.add_argument("--subtalker-top-k", "--subtalker_top_k", dest="subtalker_top_k", type=int, default=50)
    parser.add_argument("--subtalker-top-p", "--subtalker_top_p", dest="subtalker_top_p", type=float, default=1.0)
    parser.add_argument("--subtalker-temperature", "--subtalker_temperature", dest="subtalker_temperature", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=1024)
    args = parser.parse_args()

    export_meta = load_export_meta(args.work_dir) if args.work_dir else {}
    model_dir = args.model_dir or export_meta.get("hf_model")
    if not model_dir:
        raise ValueError("--model-dir is required when --work-dir is not provided")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    set_random_seed(args.seed)

    request = infer_request(export_meta or {"tts_text": DEFAULT_TEXT, "tts_mode": args.mode or "custom_voice"}, args)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    model = Qwen3TTSModel.from_pretrained(
        model_dir,
        device_map=args.device,
        dtype=dtype_map[args.dtype],
        attn_implementation=args.attn,
    )
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_dosample": args.subtalker_dosample,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }
    mode = str(request["mode"]).replace("-", "_").lower()
    if mode in {"voice_design", "voicedesign"}:
        wavs, sr = model.generate_voice_design(
            request["text"],
            language=request["language"],
            instruct=request["instruct"],
            **gen_kwargs,
        )
    elif mode in {"voice_clone", "voiceclone", "base"}:
        wavs, sr = model.generate_voice_clone(
            request["text"],
            language=request["language"],
            ref_audio=request["ref_audio"],
            ref_text=request["ref_text"],
            x_vector_only_mode=args.xvec_only,
            **gen_kwargs,
        )
    else:
        wavs, sr = model.generate_custom_voice(
            request["text"],
            language=request["language"],
            speaker=request["speaker"],
            **gen_kwargs,
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, wavs[0], sr)
    print(f"audio saved to {output}")


if __name__ == "__main__":
    main()
