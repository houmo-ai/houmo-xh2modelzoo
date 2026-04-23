import copy
import types
from pathlib import Path

import torch

from funasr.register import tables
from funasr.utils.load_utils import extract_fbank, load_audio_text_image_video


_STATIC_ENCODER_DUMMY = None
_STATIC_DECODER_DUMMY = None


def _lengths_to_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = int(lengths.max().item())
    positions = torch.arange(max_len, device=lengths.device)
    return (positions.unsqueeze(0) < lengths.unsqueeze(1)).to(dtype=torch.float32)


def _mask_to_lengths(mask: torch.Tensor) -> torch.Tensor:
    if isinstance(mask, (tuple, list)):
        mask = mask[0]
    if mask.dim() == 3:
        if mask.shape[1] == 1:
            mask = mask.squeeze(1)
        elif mask.shape[2] == 1:
            mask = mask.squeeze(-1)
    return mask.sum(dim=-1).to(dtype=torch.int32)


def _to_predictor_mask(mask: torch.Tensor) -> torch.Tensor:
    if isinstance(mask, (tuple, list)):
        mask = mask[0]
    if mask.dim() == 2:
        return mask[:, None, :]
    if mask.dim() == 3 and mask.shape[2] == 1:
        return mask.transpose(1, 2)
    return mask


def configure_static_export_from_audio(auto_model, audio_path: str) -> None:
    global _STATIC_DECODER_DUMMY, _STATIC_ENCODER_DUMMY

    cfg = {"is_final": False}
    audio_sample = load_audio_text_image_video(
        audio_path,
        fs=auto_model.kwargs["frontend"].fs,
        audio_fs=auto_model.kwargs.get("fs", 16000),
        data_type=auto_model.kwargs.get("data_type", "sound"),
        tokenizer=auto_model.kwargs.get("tokenizer"),
        cache=cfg,
    )
    if isinstance(audio_sample, (tuple, list)):
        audio_sample = audio_sample[0]
    if audio_sample.ndim == 0:
        audio_sample = audio_sample.reshape(1)
    if audio_sample.ndim > 1:
        audio_sample = audio_sample.reshape(-1)

    speech, speech_lengths = extract_fbank(
        [audio_sample],
        data_type=auto_model.kwargs.get("data_type", "sound"),
        frontend=auto_model.kwargs["frontend"],
    )
    speech = speech.to(dtype=torch.float32)
    speech_lengths = speech_lengths.to(dtype=torch.int32)
    speech_mask = _lengths_to_mask(speech_lengths, max_len=speech.shape[1])

    ref_encoder_onnx = Path(__file__).with_name("exports") / "mask_safe" / "model.onnx"
    if ref_encoder_onnx.exists():
        from input_utils import build_encoder_inputs_from_arrays
        from runtime_utils import OrtInferSession, cif_chunk

        encoder_session = OrtInferSession(str(ref_encoder_onnx), device_id=-1, intra_op_num_threads=1)
        encoder_inputs = build_encoder_inputs_from_arrays(ref_encoder_onnx, speech.numpy(), speech_lengths.numpy())
        enc_np, enc_len_np, alphas_np = encoder_session([encoder_inputs[name] for name in encoder_session.get_input_names()])[:3]
        enc = torch.from_numpy(enc_np.astype("float32"))
        enc_lengths = torch.from_numpy(enc_len_np.astype("int32"))
        acoustic_embeds_np, acoustic_lengths_np = cif_chunk(
            enc_np,
            alphas_np,
            {
                "cif_hidden": None,
                "cif_alphas": None,
            },
            [0, 10, 5],
            True,
            auto_model.model.predictor.tail_threshold,
        )
        acoustic_embeds = torch.from_numpy(acoustic_embeds_np.astype("float32"))
        acoustic_lengths = torch.from_numpy(acoustic_lengths_np.astype("int32"))
    else:
        with torch.no_grad():
            enc, enc_lengths = auto_model.model.encode(speech, speech_lengths)
            enc_mask = _lengths_to_mask(enc_lengths.to(dtype=torch.int32), max_len=enc.shape[1])
            predictor_mask = _to_predictor_mask(enc_mask).to(dtype=torch.float32)
            acoustic_embeds, acoustic_lengths, _, _ = auto_model.model.predictor(enc, mask=predictor_mask)
            acoustic_lengths = acoustic_lengths.floor().to(dtype=torch.int32)
            acoustic_max_len = int(acoustic_lengths.max().item())
            acoustic_embeds = acoustic_embeds[:, :acoustic_max_len, :].to(dtype=torch.float32)

    enc_mask = _lengths_to_mask(enc_lengths.to(dtype=torch.int32), max_len=enc.shape[1])
    pre_token_mask = _lengths_to_mask(acoustic_lengths, max_len=acoustic_embeds.shape[1])

    cache_num = len(auto_model.model.decoder.decoders)
    if auto_model.model.decoder.decoders2 is not None:
        cache_num += len(auto_model.model.decoder.decoders2)
    cache_len = auto_model.model.decoder.decoders[0].self_attn.kernel_size - 1
    caches = [
        torch.zeros((1, auto_model.model.decoder.decoders[0].size, cache_len), dtype=torch.float32)
        for _ in range(cache_num)
    ]

    _STATIC_ENCODER_DUMMY = (speech, speech_mask)
    _STATIC_DECODER_DUMMY = (enc.to(dtype=torch.float32), enc_mask, acoustic_embeds, pre_token_mask, *caches)


def _use_static_dummy() -> bool:
    return _STATIC_ENCODER_DUMMY is not None and _STATIC_DECODER_DUMMY is not None


def _encoder_forward_with_mask(
    self,
    speech: torch.Tensor,
    speech_mask: torch.Tensor,
    online: bool = False,
):
    if not online:
        speech = speech * self.output_size()**0.5

    xs_pad = speech if self.embed is None else self.embed(speech)
    mask = self.prepare_mask(speech_mask)

    encoder_outs = self.model.encoders0(xs_pad, mask)
    xs_pad = encoder_outs[0]

    encoder_outs = self.model.encoders(xs_pad, mask)
    xs_pad = encoder_outs[0]

    xs_pad = self.model.after_norm(xs_pad)

    if self.ctc_linear is not None:
        xs_pad = self.ctc_linear(xs_pad)
        xs_pad = torch.softmax(xs_pad, dim=2)

    enc_lens = _mask_to_lengths(speech_mask)
    return xs_pad, enc_lens


def _decoder_forward_with_mask(
    self,
    hs_pad: torch.Tensor,
    enc_mask: torch.Tensor,
    ys_in_pad: torch.Tensor,
    tgt_mask: torch.Tensor,
    *args,
):
    x = ys_in_pad
    tgt_mask, _ = self.prepare_mask(tgt_mask)

    memory = hs_pad
    if isinstance(enc_mask, (tuple, list)):
        _, memory_mask = enc_mask
    else:
        _, memory_mask = self.prepare_mask(enc_mask)

    out_caches = []
    for index, decoder in enumerate(self.model.decoders):
        in_cache = args[index]
        x, tgt_mask, memory, memory_mask, out_cache = decoder(
            x,
            tgt_mask,
            memory,
            memory_mask,
            cache=in_cache,
        )
        out_caches.append(out_cache)

    if self.model.decoders2 is not None:
        for index, decoder in enumerate(self.model.decoders2):
            cache_index = index + len(self.model.decoders)
            in_cache = args[cache_index]
            x, tgt_mask, memory, memory_mask, out_cache = decoder(
                x,
                tgt_mask,
                memory,
                memory_mask,
                cache=in_cache,
            )
            out_caches.append(out_cache)

    x, tgt_mask, memory, memory_mask, _ = self.model.decoders3(x, tgt_mask, memory, memory_mask)
    x = self.after_norm(x)
    x = self.output_layer(x)
    return x, out_caches


def export_rebuild_model(model, **kwargs):
    is_onnx = kwargs.get("type", "onnx") == "onnx"
    encoder_class = tables.encoder_classes.get(kwargs["encoder"] + "Export")
    model.encoder = encoder_class(model.encoder, onnx=is_onnx)

    predictor_class = tables.predictor_classes.get(kwargs["predictor"] + "Export")
    model.predictor = predictor_class(model.predictor, onnx=is_onnx)

    decoder_name = kwargs["decoder"]
    if decoder_name == "ParaformerSANMDecoder":
        decoder_name = "ParaformerSANMDecoderOnline"
    decoder_class = tables.decoder_classes.get(decoder_name + "Export")
    model.decoder = decoder_class(model.decoder, onnx=is_onnx)

    if hasattr(model.encoder, "prepare_mask"):
        model.encoder.forward = types.MethodType(_encoder_forward_with_mask, model.encoder)
    model.decoder.forward = types.MethodType(_decoder_forward_with_mask, model.decoder)

    encoder_model = copy.copy(model)
    decoder_model = copy.copy(model)

    encoder_model.forward = types.MethodType(export_encoder_forward, encoder_model)
    encoder_model.export_dummy_inputs = types.MethodType(export_encoder_dummy_inputs, encoder_model)
    encoder_model.export_input_names = types.MethodType(export_encoder_input_names, encoder_model)
    encoder_model.export_output_names = types.MethodType(export_encoder_output_names, encoder_model)
    encoder_model.export_dynamic_axes = types.MethodType(export_encoder_dynamic_axes, encoder_model)
    encoder_model.export_name = "model"

    decoder_model.forward = types.MethodType(export_decoder_forward, decoder_model)
    decoder_model.export_dummy_inputs = types.MethodType(export_decoder_dummy_inputs, decoder_model)
    decoder_model.export_input_names = types.MethodType(export_decoder_input_names, decoder_model)
    decoder_model.export_output_names = types.MethodType(export_decoder_output_names, decoder_model)
    decoder_model.export_dynamic_axes = types.MethodType(export_decoder_dynamic_axes, decoder_model)
    decoder_model.export_name = "decoder"

    return encoder_model, decoder_model


def export_encoder_forward(
    self,
    speech: torch.Tensor,
    speech_mask: torch.Tensor,
):
    enc, enc_len = self.encoder(speech, speech_mask, online=True)
    predictor_mask = _to_predictor_mask(speech_mask).to(dtype=torch.float32)
    alphas, _ = self.predictor.forward_cnn(enc, predictor_mask)
    return enc, enc_len, alphas


def export_encoder_dummy_inputs(_self):
    if _STATIC_ENCODER_DUMMY is not None:
        return _STATIC_ENCODER_DUMMY
    speech = torch.randn(2, 30, 560, dtype=torch.float32)
    speech_lengths = torch.tensor([6, 30], dtype=torch.int32)
    speech_mask = _lengths_to_mask(speech_lengths, max_len=speech.shape[1])
    return (speech, speech_mask)


def export_encoder_input_names(_self):
    return ["speech", "speech_mask"]


def export_encoder_output_names(_self):
    return ["enc", "enc_len", "alphas"]


def export_encoder_dynamic_axes(_self):
    if _use_static_dummy():
        return {}
    return {
        "speech": {0: "batch_size", 1: "feats_length"},
        "speech_mask": {0: "batch_size", 1: "feats_length"},
        "enc": {0: "batch_size", 1: "feats_length"},
        "enc_len": {0: "batch_size"},
        "alphas": {0: "batch_size", 1: "feats_length"},
    }


def export_decoder_forward(
    self,
    enc: torch.Tensor,
    enc_mask: torch.Tensor,
    acoustic_embeds: torch.Tensor,
    pre_token_mask: torch.Tensor,
    *args,
):
    decoder_out, out_caches = self.decoder(enc, enc_mask, acoustic_embeds, pre_token_mask, *args)
    sample_ids = decoder_out.argmax(dim=-1)
    return decoder_out, sample_ids, out_caches


def export_decoder_dummy_inputs(self):
    if _STATIC_DECODER_DUMMY is not None:
        return _STATIC_DECODER_DUMMY
    enc_size = self.encoder._output_size
    enc = torch.randn(2, 100, enc_size, dtype=torch.float32)
    enc_lengths = torch.tensor([30, 100], dtype=torch.int32)
    enc_mask = _lengths_to_mask(enc_lengths, max_len=enc.shape[1])

    acoustic_embeds = torch.randn(2, 10, enc_size, dtype=torch.float32)
    acoustic_lengths = torch.tensor([5, 10], dtype=torch.int32)
    pre_token_mask = _lengths_to_mask(acoustic_lengths, max_len=acoustic_embeds.shape[1])

    cache_num = len(self.decoder.model.decoders)
    if self.decoder.model.decoders2 is not None:
        cache_num += len(self.decoder.model.decoders2)
    cache_len = self.decoder.model.decoders[0].self_attn.kernel_size - 1
    cache = [
        torch.zeros((2, self.decoder.model.decoders[0].size, cache_len), dtype=torch.float32)
        for _ in range(cache_num)
    ]
    return (enc, enc_mask, acoustic_embeds, pre_token_mask, *cache)


def export_decoder_input_names(self):
    cache_num = len(self.decoder.model.decoders)
    if self.decoder.model.decoders2 is not None:
        cache_num += len(self.decoder.model.decoders2)
    return ["enc", "enc_mask", "acoustic_embeds", "pre_token_mask"] + [
        f"in_cache_{index}" for index in range(cache_num)
    ]


def export_decoder_output_names(self):
    cache_num = len(self.decoder.model.decoders)
    if self.decoder.model.decoders2 is not None:
        cache_num += len(self.decoder.model.decoders2)
    return ["logits", "sample_ids"] + [f"out_cache_{index}" for index in range(cache_num)]


def export_decoder_dynamic_axes(_self):
    if _use_static_dummy():
        return {}
    dynamic_axes = {
        "enc": {0: "batch_size", 1: "enc_length"},
        "enc_mask": {0: "batch_size", 1: "enc_length"},
        "acoustic_embeds": {0: "batch_size", 1: "token_length"},
        "pre_token_mask": {0: "batch_size", 1: "token_length"},
        "logits": {0: "batch_size", 1: "token_length"},
        "sample_ids": {0: "batch_size", 1: "token_length"},
    }
    for index in range(16):
        dynamic_axes[f"in_cache_{index}"] = {0: "batch_size"}
        dynamic_axes[f"out_cache_{index}"] = {0: "batch_size"}
    return dynamic_axes


def install_mask_safe_export() -> None:
    import funasr.models.paraformer_streaming.export_meta as export_meta

    export_meta.export_rebuild_model = export_rebuild_model