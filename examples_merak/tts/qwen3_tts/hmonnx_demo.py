import argparse
from pathlib import Path

import soundfile as sf
import torch
from xhquant.api import set_random_seed

from hmonnx_utils import build_hmonnx_model, build_voice_clone_prompt, infer_request, load_export_meta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default=None, help="custom_voice, voice_clone/base, or voice_design")
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--language", type=str, default="Chinese")
    parser.add_argument("--speaker", type=str, default=None)
    parser.add_argument("--instruct", type=str, default=None)
    parser.add_argument("--ref-audio", type=str, default=None)
    parser.add_argument("--ref-text", type=str, default=None)
    parser.add_argument("--xvec-only", action="store_true")
    parser.add_argument("--max-new-tokens", "--max_new_tokens", dest="max_new_tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=1024)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"
    set_random_seed(args.seed)
    work_dir = Path(args.work_dir)
    export_meta = load_export_meta(work_dir)
    request = infer_request(export_meta, args)
    voice_clone_prompt = build_voice_clone_prompt(work_dir, export_meta, request, args.device, args.xvec_only)

    model = build_hmonnx_model(work_dir, args.device)
    kwargs = {"max_new_tokens": args.max_new_tokens}
    if voice_clone_prompt is not None:
        kwargs["voice_clone_prompt"] = voice_clone_prompt
        request["ref_audio"] = None
    wavs, sr = model.generate_by_mode(**request, **kwargs)
    output = Path(args.output) if args.output else work_dir / f"output_{request['mode']}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output, wavs[0], sr)
    print(f"audio saved to {output}")


if __name__ == "__main__":
    main()
