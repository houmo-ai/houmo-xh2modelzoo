import argparse

# from pathlib import Path
# from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.pipelines.audio_utils import ffmpeg_read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="./data/models/whisper-large-v3-turbo",
        # default="./data/models/whisper-medium",
        # "--model",
        # type=str,
        # default="./data/models/whisper-medium",
    )
    parser.add_argument("--audio", type=str, default="./examples/llm/whisper/audio.mp3")
    args = parser.parse_args()

    # load model and processor
    hf_model = args.model
    processor = WhisperProcessor.from_pretrained(hf_model)
    model = WhisperForConditionalGeneration.from_pretrained(hf_model)
    model.config.forced_decoder_ids = None

    # load dummy dataset and read audio files
    # ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    # sample = ds[0]["audio"]
    audo_file = args.audio
    sampling_rate = 16000
    with open(audo_file, "rb") as f:
        inputs = f.read()
    if isinstance(inputs, bytes):
        inputs = ffmpeg_read(inputs, sampling_rate)
    sample = {
        "array": inputs,
        "sampling_rate": sampling_rate,
    }
    input_features = processor(
        sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt"
    ).input_features
    # [1,80,3000]

    model.model.encoder.decoder_m = model.model.decoder

    # generate token ids
    predicted_ids = model.generate(input_features)  # [1,21]
    # decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
    # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    print(transcription)


if __name__ == "__main__":
    main()
