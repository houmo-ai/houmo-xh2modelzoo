"""Export FunASR-Nano audio encoder, adaptor and optional CTC branch to ONNX."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from funasr_nano_common import (
    extract_fbank,
    load_audio_for_frontend,
    load_funasr_nano_model,
    low_frame_rate_len,
    output_path_in_work_dir,
    save_token_embedding_from_model,
    write_json,
)
from xh_model_zoo.xh_llm.models.funasr_nano.mask_utils import (
    attention_additive_mask,
    downsample_lengths,
    downsample_time,
    sequence_mask,
)
from xh_model_zoo.xh_llm.models.funasr_nano.warp import (
    apply_funasr_audio_warp,
)


class AudioEncoderExport(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = apply_funasr_audio_warp(encoder)

    def forward(self, speech: torch.Tensor, speech_mask: torch.Tensor, speech_att_mask: torch.Tensor):
        encoder_out, encoder_out_lens = self.encoder(speech, speech_mask, speech_att_mask)
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]
        return encoder_out, encoder_out_lens.to(torch.int32)


class AudioAdaptorExport(torch.nn.Module):
    def __init__(self, adaptor: torch.nn.Module, use_low_frame_rate: bool = True):
        super().__init__()
        self.adaptor = adaptor
        self.use_low_frame_rate = use_low_frame_rate

    def forward(self, encoder_out: torch.Tensor, adaptor_mask: torch.Tensor, adaptor_att_mask: torch.Tensor):
        adaptor_out, adaptor_out_lens = self.adaptor(encoder_out, adaptor_mask, adaptor_att_mask)
        if self.use_low_frame_rate:
            adaptor_out_lens = low_frame_rate_len(adaptor_mask.squeeze(1).sum(1).to(torch.int32))
        return adaptor_out, adaptor_out_lens.to(torch.int32)


class CTCDecoderExport(torch.nn.Module):
    def __init__(self, ctc_decoder: torch.nn.Module, ctc: torch.nn.Module):
        super().__init__()
        self.ctc_decoder = ctc_decoder
        self.ctc = ctc

    def forward(self, encoder_out: torch.Tensor, ctc_mask: torch.Tensor, ctc_att_mask: torch.Tensor):
        decoder_out, decoder_out_lens = self.ctc_decoder(encoder_out, ctc_mask, ctc_att_mask)
        logits = self.ctc.log_softmax(decoder_out)
        return logits, decoder_out_lens.to(torch.int32)


def _export_onnx(module, inputs, out_file: Path, input_names, output_names, opset: int, dynamic: bool) -> Path:
    module.eval().cpu()
    dynamic_axes = None
    if dynamic:
        dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
        for name in input_names:
            if name.endswith("out") or name == "speech" or name == "encoder_out":
                dynamic_axes[name][1] = "time"
        for name in output_names:
            if name.endswith("out") or name == "ctc_logits":
                dynamic_axes[name][1] = "time"
    torch.onnx.export(
        module,
        tuple(x.cpu() for x in inputs),
        str(out_file),
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    return out_file


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-dir", default="/data01/datasets/Funasr/Fun-ASR-Nano-2512")
    parser.add_argument("--audio", default=None, help="Optional sample audio for dummy shape")
    parser.add_argument("--work-dir", default="work_dirs/funasr_nano_xh2a")
    parser.add_argument("--max-frames", type=int, default=512, help="Pad/truncate fbank frames for static export")
    parser.add_argument("--opset", type=int, default=14)
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--skip-ctc", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    work_dir = Path(args.work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    model, kwargs = load_funasr_nano_model(args.model_dir, device="cpu")
    model.audio_adaptor = apply_funasr_audio_warp(model.audio_adaptor)
    if getattr(model, "ctc_decoder", None) is not None:
        model.ctc_decoder = apply_funasr_audio_warp(model.ctc_decoder)
    frontend = kwargs["frontend"]
    wav = load_audio_for_frontend(args.audio, frontend)
    speech, speech_lengths = extract_fbank(wav, frontend)
    if not args.dynamic:
        target_frames = int(args.max_frames)
        cur_frames = int(speech.shape[1])
        if cur_frames < target_frames:
            speech = torch.nn.functional.pad(speech, (0, 0, 0, target_frames - cur_frames))
            speech_lengths = torch.tensor([target_frames], dtype=torch.int32)
        elif cur_frames > target_frames:
            speech = speech[:, :target_frames, :]
            speech_lengths = torch.tensor([target_frames], dtype=torch.int32)

    encoder_export = AudioEncoderExport(model.audio_encoder)
    speech_mask = sequence_mask(speech_lengths.to(torch.int32), maxlen=int(speech.shape[1]))
    speech_att_mask = attention_additive_mask(speech_mask)
    encoder_onnx = output_path_in_work_dir(work_dir, "Encoder", "funasr_nano_encoder.onnx")
    _export_onnx(
        encoder_export,
        (speech.float(), speech_mask.float(), speech_att_mask.float()),
        encoder_onnx,
        ["speech", "speech_mask", "speech_att_mask"],
        ["encoder_out", "encoder_out_lens"],
        args.opset,
        args.dynamic,
    )

    with torch.no_grad():
        encoder_out, encoder_out_lens = encoder_export(speech.float(), speech_mask.float(), speech_att_mask.float())

    adaptor_export = AudioAdaptorExport(model.audio_adaptor, getattr(model, "use_low_frame_rate", True))
    adaptor_lens = downsample_lengths(encoder_out_lens.to(torch.int32), getattr(model.audio_adaptor, "k", 1))
    adaptor_mask = sequence_mask(adaptor_lens, maxlen=downsample_time(int(encoder_out.shape[1]), getattr(model.audio_adaptor, "k", 1)))
    adaptor_att_mask = attention_additive_mask(adaptor_mask)
    adaptor_onnx = output_path_in_work_dir(work_dir, "Adaptor", "funasr_nano_audio_adaptor.onnx")
    _export_onnx(
        adaptor_export,
        (encoder_out.float(), adaptor_mask.float(), adaptor_att_mask.float()),
        adaptor_onnx,
        ["encoder_out", "adaptor_mask", "adaptor_att_mask"],
        ["audio_embeds", "audio_embed_lens"],
        args.opset,
        args.dynamic,
    )

    ctc_onnx = None
    if not args.skip_ctc and getattr(model, "ctc_decoder", None) is not None and getattr(model, "ctc", None) is not None:
        ctc_export = CTCDecoderExport(model.ctc_decoder, model.ctc)
        ctc_lens = downsample_lengths(encoder_out_lens.to(torch.int32), getattr(model.ctc_decoder, "k", 1))
        ctc_mask = sequence_mask(ctc_lens, maxlen=downsample_time(int(encoder_out.shape[1]), getattr(model.ctc_decoder, "k", 1)))
        ctc_att_mask = attention_additive_mask(ctc_mask)
        ctc_onnx = output_path_in_work_dir(work_dir, "CTC", "funasr_nano_ctc.onnx")
        _export_onnx(
            ctc_export,
            (encoder_out.float(), ctc_mask.float(), ctc_att_mask.float()),
            ctc_onnx,
            ["encoder_out", "ctc_mask", "ctc_att_mask"],
            ["ctc_logits", "ctc_lens"],
            args.opset,
            args.dynamic,
        )

    token_embedding_file = save_token_embedding_from_model(model, work_dir / "token_embedding.pt")
    meta = {
        "funasr_model_dir": str(Path(args.model_dir).expanduser().resolve()),
        "encoder_onnx_file": str(encoder_onnx.relative_to(work_dir)),
        "audio_adaptor_onnx_file": str(adaptor_onnx.relative_to(work_dir)),
        "ctc_onnx_file": str(ctc_onnx.relative_to(work_dir)) if ctc_onnx else None,
        "token_embedding_file": str(token_embedding_file.relative_to(work_dir)),
        "use_low_frame_rate": bool(getattr(model, "use_low_frame_rate", True)),
        "audio_adaptor_downsample_rate": int(getattr(model.audio_adaptor, "k", 1)),
        "ctc_decoder_downsample_rate": int(getattr(model.ctc_decoder, "k", 1)) if getattr(model, "ctc_decoder", None) is not None else 1,
        "blank_id": int(getattr(model, "blank_id", -1)),
        "ctc_tokenizer_source": "funasr_model" if getattr(model, "ctc_tokenizer", None) is not None else None,
        "max_frames": int(args.max_frames),
        "opset": int(args.opset),
    }
    write_json(work_dir / "export_meta_info.json", meta)
    print(f"Exported FunASR-Nano audio modules to {work_dir}")


if __name__ == "__main__":
    main()
