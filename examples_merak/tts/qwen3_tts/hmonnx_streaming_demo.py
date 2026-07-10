import argparse
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from xhquant.api import set_random_seed

from hmonnx_utils import (
    build_hmonnx_model,
    build_voice_clone_prompt,
    infer_request,
    load_export_meta,
    resolve_stateful_decoder_meta,
)
from xhmodel_merak.xh_other_model.models.qwen3_tts.qwen3_tts_stateful_decoder import Qwen3TTSStatefulDecoderInference


def _normalize_codes(codes) -> torch.Tensor:
    if not torch.is_tensor(codes):
        codes = torch.as_tensor(np.asarray(codes), dtype=torch.long)
    else:
        codes = codes.detach().cpu().long()
    if codes.dim() == 3:
        codes = codes.reshape(-1, codes.shape[-1])
    if codes.dim() == 1:
        codes = codes.view(-1, 16)
    if codes.dim() != 2 or codes.shape[-1] != 16:
        raise ValueError(f"expected codec codes with shape [N, 16], got {tuple(codes.shape)}")
    return codes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--mode", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--language", type=str, default="Chinese")
    parser.add_argument("--speaker", type=str, default=None)
    parser.add_argument("--instruct", type=str, default=None)
    parser.add_argument("--ref-audio", type=str, default=None)
    parser.add_argument("--ref-text", type=str, default=None)
    parser.add_argument("--xvec-only", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
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
    decoder = Qwen3TTSStatefulDecoderInference(model_cfg=str(resolve_stateful_decoder_meta(work_dir, export_meta)))
    decoder.to(args.device)
    decoder.initialize()
    decoder.decode(
        torch.zeros((decoder.chunk_size, 16), dtype=torch.int32),
        state=decoder.create_state(decoder.kv_cache_window, device=args.device, dtype=torch.float16),
        is_final=True,
    )
    state = None

    gen_kwargs = {"max_new_tokens": args.max_new_tokens}
    if voice_clone_prompt is not None:
        gen_kwargs["voice_clone_prompt"] = voice_clone_prompt
        request["ref_audio"] = None

    chunks = []
    decoded_final_chunk = False
    start = time.time()
    for codes in model.generate_code_stream(chunk_size=args.chunk_size, **request, **gen_kwargs):
        codes = _normalize_codes(codes)
        if codes.numel() == 0:
            continue
        is_final = codes.shape[0] < int(decoder.chunk_size)
        decoded_final_chunk = decoded_final_chunk or is_final
        audio, state = decoder.decode(codes.to(torch.int32), state=state, is_final=is_final)
        audio_np = audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if audio_np.size:
            print(f"AUDIO frames={codes.shape[0]} samples={audio_np.size}")
            chunks.append(audio_np)
    if not decoded_final_chunk:
        audio, state = decoder.decode(torch.zeros((0, 16), dtype=torch.int32), state=state, is_final=True)
        audio_np = audio.detach().cpu().numpy().astype(np.float32).reshape(-1)
        if audio_np.size:
            print(f"AUDIO final samples={audio_np.size}")
            chunks.append(audio_np)

    if not chunks:
        raise RuntimeError("streaming generated no audio")
    output = Path(args.output) if args.output else work_dir / f"output_streaming_{request['mode']}.wav"
    output.parent.mkdir(parents=True, exist_ok=True)
    wav = np.concatenate(chunks)
    sf.write(output, wav, 24000)
    print(f"FINISH chunks={len(chunks)} seconds={len(wav) / 24000:.2f} elapsed={time.time() - start:.2f}")
    print(f"audio saved to {output}")


if __name__ == "__main__":
    main()
