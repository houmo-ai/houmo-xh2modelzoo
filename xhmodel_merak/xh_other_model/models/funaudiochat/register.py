def register_funaudiochat() -> None:
    from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

    from .configuration_funaudiochat import FunAudioChatAudioEncoderConfig, FunAudioChatConfig
    from .modeling_funaudiochat import FunAudioChatForConditionalGeneration
    from .processing_funaudiochat import FunAudioChatProcessor

    AutoConfig.register("funaudiochat", FunAudioChatConfig, exist_ok=True)
    AutoConfig.register("funaudiochat_audio_encoder", FunAudioChatAudioEncoderConfig, exist_ok=True)
    AutoProcessor.register(FunAudioChatConfig, FunAudioChatProcessor, exist_ok=True)
    AutoModelForSeq2SeqLM.register(FunAudioChatConfig, FunAudioChatForConditionalGeneration, exist_ok=True)
