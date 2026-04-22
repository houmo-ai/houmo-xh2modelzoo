#!/usr/bin/python3
# -*- coding: utf-8 -*-

from loguru import logger
# import whisper_api
import soundfile as sf
import torch
import time

# ---------- CLI / 示例用法 ----------
if __name__ == "__main__":
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    hf_model = "/data02/datasets/whisper_medium"
    processor = WhisperProcessor.from_pretrained(hf_model)
    model = WhisperForConditionalGeneration.from_pretrained(hf_model)
    model.model.encoder.decoder_m = model.model.decoder

    model.config.forced_decoder_ids = None
    # whisper_api.init_model("whisper-medium")

    waveform, sr = sf.read("examples/llm/whisper/test.wav")  # or wav
    waveform = torch.from_numpy(waveform)

    # 如果是双声道，转成 mono
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=1)

    # whisper模型一次最大20S,对音频文件进行切片
    total_samples = waveform.shape[0]
    chunk_len = int(25 * sr)
    ranges = []
    start = 0
    while start < total_samples:
        end = start + chunk_len
        if end > total_samples:
            end = total_samples
        ranges.append((start, end))
        start = end
   
   
    # 分片转换
    start = time.time()
    for i, (s, e) in enumerate(ranges):
        time1 = time.time()
        chunk = waveform[s:e]  # 1-D tensor

        input_features = processor(
            chunk, sampling_rate=sr, return_tensors="pt"
        ).input_features

        # generate token ids
        predicted_ids = model.generate(input_features)  # [1,21]
        # decode token ids to text
        text = processor.batch_decode(predicted_ids, skip_special_tokens=False)

        if isinstance(text, (list, tuple)):
            chunk_text = text[0] if len(text) > 0 else ""
        else:
            chunk_text = str(text)
        time2 = time.time()
        
        logger.info(f"Processing chunk {i+1}/{len(ranges)}: samples [{s}:{e}], Cost {(time2 - time1)*1000:.3f} ms")
        logger.info(f"Chunk {i+1} text (raw): {chunk_text!r}")
    end = time.time()
    logger.info(f"audio: {total_samples/sr}s, Cost {(end - start)*1000:.3f} ms")
  