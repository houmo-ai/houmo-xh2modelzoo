from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

# load model and processor
processor = WhisperProcessor.from_pretrained("data/models/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("data/models/whisper-medium")
model.config.forced_decoder_ids = None

# load dummy dataset and read audio files
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]
input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features
# [1,80,3000]

model.model.encoder.decoder_m = model.model.decoder

# generate token ids
predicted_ids = model.generate(input_features)  # [1,21]
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
