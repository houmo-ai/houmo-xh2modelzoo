import argparse
import logging

from export_utils import resolve_decoder_onnx, resolve_encoder_onnx, resolve_manifest
from input_utils import build_encoder_inputs_from_arrays, get_onnx_input_specs
from runtime_utils import DEFAULT_CHUNK_SIZE, DEFAULT_DECODER_LOOK_BACK, DEFAULT_ENCODER_LOOK_BACK, OrtInferSession, build_decoder_inputs_from_encoder_outputs, extract_full_features, load_model, tokens_to_text, decode_ids

LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/data02/datasets/funasr/Paraformer")
    parser.add_argument("--model-revision", default="v2.0.4")
    parser.add_argument("--audio", default="/data02/datasets/funasr/Paraformer/example/asr_example.wav")
    parser.add_argument("--model-dir", default="examples/audio/paraformer/exports/mask_safe_static")
    parser.add_argument("--export-manifest", default="")
    parser.add_argument("--encoder", default="")
    parser.add_argument("--decoder", default="")
    parser.add_argument("--device-id", type=int, default=-1)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()

    manifest_path = resolve_manifest(args.export_manifest)
    encoder_path = resolve_encoder_onnx(args.encoder, args.model_dir, manifest_path)
    decoder_path = resolve_decoder_onnx(args.decoder, args.model_dir, manifest_path)
    model = load_model(args.model, args.model_revision, device="cpu")
    baseline = model.generate(input=args.audio, chunk_size=DEFAULT_CHUNK_SIZE, encoder_chunk_look_back=DEFAULT_ENCODER_LOOK_BACK, decoder_chunk_look_back=DEFAULT_DECODER_LOOK_BACK)
    LOGGER.info("baseline: %s", baseline)

    encoder_session = OrtInferSession(str(encoder_path), device_id=args.device_id, intra_op_num_threads=args.threads)
    decoder_session = OrtInferSession(str(decoder_path), device_id=args.device_id, intra_op_num_threads=args.threads)
    encoder_input_names = [name for name, _ in get_onnx_input_specs(encoder_path)]
    decoder_input_names = [name for name, _ in get_onnx_input_specs(decoder_path)]
    speech, speech_lengths = extract_full_features(model, args.audio)
    encoder_inputs = build_encoder_inputs_from_arrays(encoder_path, speech, speech_lengths)
    encoder_outputs = encoder_session([encoder_inputs[name] for name in encoder_input_names])
    enc, enc_len, alphas = encoder_outputs[:3]
    prepared = build_decoder_inputs_from_encoder_outputs(model, decoder_path, enc, enc_len, alphas)
    decoder_inputs = prepared["decoder_inputs"]
    decoder_outputs = decoder_session([decoder_inputs[name] for name in decoder_input_names])
    sample_ids = decoder_outputs[1]
    text = tokens_to_text(decode_ids(model, sample_ids[0], int(prepared["acoustic_embeds_len"][0])))
    print(text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()