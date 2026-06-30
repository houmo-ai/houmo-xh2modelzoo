from funasr.models.fun_asr_nano.inference_vllm_streaming import FunASRNanoStreamingVLLM

if __name__ == "__main__":
    engine = FunASRNanoStreamingVLLM.from_pretrained(
        model="/data01/datasets/Funasr/Fun-ASR-Nano-2512",
        chunk_ms=720,
        rollback_chars=8,
    )

    for result in engine.streaming_generate("audio.wav", language="中文"):
        if result["is_final"]:
            print(f"最终: {result['text']}")
        else:
            print(f"[{result['audio_duration_ms']:.0f}ms] 确认: {result['fixed_text']}")