# Shared Model Matrix (xhquant_llm -> xh2modelzoo)

- xh2modelzoo: `/home/jiangyong.yu/xh2_work/xh2modelzoo`
- xhquant_llm: `/home/jiangyong.yu/xh2_work/xhquant_llm`

## Overlap Models (15)

| model | archetype | xhquant_llm key files | xh2modelzoo key files | src example | dst example |
|---|---|---|---|---|---|
| cogvlm2 | vlm-ocr | cogvlm2_llm_model.py<br>cogvlm2_onnx_model.py<br>cogvlm2_vision_model.py<br>_llm_model_impl.py<br>_vision_model_impl.py | _llm_model_impl.py<br>_vision_model_impl.py<br>cogvlm2_convert_config.py<br>cogvlm2_vl_converter.py | - | - |
| fm9g | llm | fm9g_llm_model.py<br>_model_impl.py | _model_impl.py<br>fm9g_hf_compatible.py<br>fm9g_convert_config.py<br>fm9g_converter.py | fm9g | fm9g |
| glm_ocr | vlm-ocr | glm_ocr_llm_model.py<br>glm_ocr_onnx_model.py<br>glm_ocr_vision_model.py<br>_llm_model_impl.py<br>_vision_model_impl.py<br>glm_ocr_hf_compatible.py | glm_ocr_llm_model.py<br>glm_ocr_onnx_model.py<br>glm_ocr_vision_model.py<br>_llm_model_impl.py<br>_vision_model_impl.py<br>glm_ocr_hf_compatible.py | glm_ocr | glm_ocr |
| gpt_oss_with_mask | llm | _model.py<br>gpt_oss_llm_model.py<br>gpt_oss_hf_compatible.py | _model.py<br>gpt_oss_llm_model.py<br>gpt_oss_hf_compatible.py<br>gpt_oss_convert_config.py<br>gpt_oss_converter.py | gpt_oss_with_mask | - |
| llama | llm | _model.py<br>llama_batch_hmonnx_model.py<br>llama_llm_model.py | _model.py<br>llama_convert_config.py<br>llama_converter.py | llama | llama |
| minicpmo | multi-component | minicpmo_audio_model.py<br>minicpmo_base_model.py<br>minicpmo_llm_model.py<br>minicpmo_tts_dvae_model.py<br>minicpmo_tts_model.py<br>minicpmo_tts_vocos_model.py | minicpmo_audio_model.py<br>minicpmo_base_model.py<br>minicpmo_llm_model.py<br>minicpmo_tts_dvae_model.py<br>minicpmo_tts_model.py<br>minicpmo_tts_vocos_model.py | minicpmo | minicpmo |
| paddleocr_vl | vlm-ocr | paddleocr_vl_llm_model.py<br>paddleocr_vl_onnx_model.py<br>paddleocr_vl_vision_model.py<br>_llm_model_impl.py<br>_vision_model_impl.py<br>paddleocr_vl_hf_compatible.py | paddleocr_vl_llm_model.py<br>paddleocr_vl_onnx_model.py<br>paddleocr_vl_vision_model.py<br>_llm_model_impl.py<br>_vision_model_impl.py<br>paddleocr_vl_hf_compatible.py | paddleocr_vl | paddleocr_vl |
| paddleocr_vl_1_5 | vlm-ocr | paddleocr_vl_llm_model.py<br>paddleocr_vl_onnx_model.py<br>paddleocr_vl_vision_model.py<br>_llm_model_impl.py<br>_vision_model_impl.py<br>paddleocr_vl_hf_compatible.py | paddleocr_vl_llm_model.py<br>paddleocr_vl_onnx_model.py<br>paddleocr_vl_vision_model.py<br>_llm_model_impl.py<br>_vision_model_impl.py<br>paddleocr_vl_hf_compatible.py | paddleocr_vl_1_5 | paddleocr_vl_1_5 |
| qwen2_5_vl | vlm-ocr | qwen2_5_vl_llm_model.py<br>qwen2_5_vl_onnx_model.py<br>qwen2_5_vl_vision_model.py<br>_llm_model_impl.py<br>_vision_model_impl.py<br>qwe2_5_vl_hf_compatible.py | llm_onnx_model.py<br>qwen2_5_vl_onnx_model.py<br>_llm_model_impl.py<br>_vision_model_impl.py<br>qwen2_5_vl_convert_config.py<br>qwen2_5_vl_converter.py | qwen2.5-vl | qwen2_5_vl |
| qwen2_legacy | llm | _model.py<br>gguf_model.py<br>qwen_llm_hf_gptq_model.py<br>qwen_llm_model.py<br>qwen2_hf_compatible.py | _model.py<br>qwen2_hf_compatible.py<br>qwen2_convert_config.py<br>qwen2_converter.py | qwen2_legacy | qwen2_legacy |
| qwen2_vl | vlm-ocr | qwen2_vl_awq_llm_model.py<br>qwen2_vl_llm_model.py<br>qwen2_vl_vision_model.py<br>qwen2vl_onnx_model.py<br>xh_model.py<br>_llm_model_impl.py | _llm_model_impl.py<br>_vision_model_impl.py<br>qwen2_vl_hf_compatible.py<br>qwen2_vl_convert_config.py<br>qwen2_vl_awq_converter.py<br>qwen2_vl_converter.py | qwen2-vl | qwen2-vl |
| qwen3_legacy | llm | _model.py<br>gguf_model.py<br>qwen_llm_model.py<br>qwen3_hf_compatible.py | _model.py<br>qwen3_hf_compatible.py<br>qwen3_convert_config.py<br>qwen3_converter.py | qwen3_legacy | qwen3_legacy |
| qwen3_legacy_lora | llm | _model.py<br>qwen_llm_model.py<br>qwen3_hf_compatible.py | _model.py<br>qwen3_hf_compatible.py<br>qwen3_convert_config.py<br>qwen3_converter.py | qwen3_legacy_lora | qwen3_legacy_lora |
| qwen3_vl | vlm-ocr | qwen3_vl_llm_model.py<br>qwen3_vl_onnx_model.py<br>qwen3_vl_vision_model.py<br>_llm_model_impl.py<br>_vision_model_impl.py<br>qwen3_vl_hf_compatible.py | llm_onnx_model.py<br>qwen3_vl_onnx_model.py<br>_llm_model_impl.py<br>_vision_model_impl.py<br>qwen3_vl_convert_config.py<br>qwen3_vl_converter.py | qwen3_vl | - |
| qwen3moe | moe | _moe_model.py<br>qwen_llm_hf_gptq_model.py<br>qwen_moe_batch_hmonnx_model.py<br>qwen_moe_llm_model.py<br>qwen3moe_hf_compatible.py | _moe_model.py<br>qwen_moe_hf_compatible.py<br>qwen_moe_convert_config.py<br>qwen_moe_converter.py | qwen3moe | qwen3moe |

## Only In xhquant_llm (27)

cogvlm2_legacy, common, deepseekocr, fast_eval_wrap, gemma3, gemma3_mask, glm4v, glm_flash, gpt_oss, hunyuan_moe, intervl3, intervl3_5, llm_pipeline, minicpmv, qwen2_batch, qwen3, qwen3_5, qwen3_5_moe, qwen3_legacy_opt, qwen3_omni, qwen3_tts, qwen3_with_mask, qwen3moe_pipe, qwen3moe_raccoon, qwen3moe_vl, qwen3next, s2t_transformers

## Only In xh2modelzoo (17)

_base_, _qwen2, _qwen3, bert, bge_reranker, cosyvoice3, deepseek_ocr, groot, gte_paddle, pi05, qwen2_ste, qwen3_embeding, qwen3_vl_moe, qwen_image, whisper, xvla, zimage
