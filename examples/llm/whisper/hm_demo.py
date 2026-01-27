import argparse
import json
import gc
import torch
import soundfile as sf
from pathlib import Path
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from xhquant.api import HMONNXInference

MAX_GEN_LEN = 448

def clean_memory():
    """强制清理显存和内存"""
    gc.collect()
    torch.cuda.empty_cache()

def main(args):
    device = torch.device("cuda")
    
    # 1. 准备配置
    work_dir = Path(args.hmonnx_model)
    meta_info = json.load((work_dir / "meta_info.json").open("r", encoding="utf-8"))
    
    num_heads = meta_info["model_cfg"]["num_heads"]
    head_dim = meta_info["model_cfg"]["head_dim"]
    num_decode_layers = meta_info["model_cfg"]["num_decode_layers"]
    
    processor = WhisperProcessor.from_pretrained(args.hf_model)
    model_config = WhisperForConditionalGeneration.from_pretrained(args.hf_model).config

    # 获取 Prompt Tokens ID
    sot_id = model_config.decoder_start_token_id
    lang_id = processor.tokenizer.convert_tokens_to_ids("<|zh|>")
    transcribe_id = processor.tokenizer.convert_tokens_to_ids("<|transcribe|>")
    notime_id = processor.tokenizer.convert_tokens_to_ids("<|notimestamps|>")
    eos_id = model_config.eos_token_id

    prompt_tokens = [sot_id, lang_id, transcribe_id, notime_id]

    print(f">>> 加载音频文件: {args.audio_file}")
    audio_array, sr = sf.read(args.audio_file)
    
    # --- Encoder ---
    print(">>> 运行 Encoder...")
    encoder = HMONNXInference(str(work_dir / meta_info["encoder"]))
    encoder.to_fast_mode()
    encoder.to(device)
    
    input_features = processor(audio_array, sampling_rate=16000, return_tensors="pt").input_features.half().to(device)
    enc_out = encoder(input_features)
    
    # 动态获取序列长度
    enc_seq_len = enc_out[0].shape[2] # 1500
    # print(f"Encoder 输出范围: {enc_out[0].min().item():.4f} ~ {enc_out[0].max().item():.4f}")
    
    # enc_out 交错结构
    # [k0, v0, k1, v1, k2, v2, k3, v3]
    # 这里已经将 hidden states 切分成了 key 和 value
    k_list = enc_out[0::2]
    v_list = enc_out[1::2]
    
    del encoder
    clean_memory()

    # --- Decoder ---
    print(">>> 准备 Decoder...")
    decoder_model_path = str(work_dir / meta_info["decoder"])
    decoder = HMONNXInference(decoder_model_path)
    decoder.to(device)
    decoder.to_fast_mode()
    
    prefill_model_path = str(work_dir / meta_info["prefill"])
    prefill = HMONNXInference(prefill_model_path)
    prefill.to(device)
    prefill.to_fast_mode()
    
    dec_names = decoder.get_input_names()
    
    # 预分配 self-attention 的 K/V Cache
    CACHE_MAX_LEN = 1280 
    k_cache = [torch.zeros([1, num_heads, CACHE_MAX_LEN, head_dim], device=device, dtype=torch.float16) for _ in range(num_decode_layers)]
    v_cache = [torch.zeros([1, num_heads, CACHE_MAX_LEN, head_dim], device=device, dtype=torch.float16) for _ in range(num_decode_layers)]
    
    # 构造 Encoder Mask
    # 和 enc_seq_len 对齐
    encoder_attention_mask = torch.zeros((1, 1, 1, enc_seq_len), device=device, dtype=torch.float16)

    all_tokens_list = []
    
    logits = None
    step = 0 
    
    # === 阶段 1: Prefill ===
    print(">>> 开始并行处理 Prefill ...")
    
    # [sot_id, lang_id, transcribe_id, notime_id]
    input_ids = torch.tensor([prompt_tokens], device=device, dtype=torch.int32)
    prompt_len = len(prompt_tokens) # 4
    # 2. Position IDs: [1, 4] -> [[0, 1, 2, 3]]
    position_ids = torch.arange(prompt_len, device=device, dtype=torch.int32).unsqueeze(0)
    # 构造 self-attention Mask，作用在 logits 上
    mask_atten = torch.full((1, num_heads, prompt_len, CACHE_MAX_LEN), -65504.0, device=device, dtype=torch.float16)
    
    # 将可见部分填为 0.0 (下三角)
    # Token 0 看 [0]
    # Token 1 看 [0, 1]
    # Token 2 看 [0, 1, 2]
    # Token 3 看 [0, 1, 2, 3]
    for i in range(prompt_len):
        mask_atten[:, :, i, :i+1] = 0.0
    
    # This block of code is creating a dictionary named `inputs` that contains various tensors used as
    # input for the model during the Prefill stage of the decoding process. Here's a breakdown of each
    # key-value pair in the `inputs` dictionary:
    inputs = {
        dec_names[0]: input_ids,
        dec_names[1]: position_ids,                                          
        dec_names[2]: torch.tensor([0], device=device, dtype=torch.int32),   # past_key_values_length = 0
        dec_names[3]: torch.tensor([prompt_len], device=device, dtype=torch.int32), # current_sequence_length = 4
        dec_names[4]: mask_atten,
        dec_names[5]: encoder_attention_mask
    }
    
    base_idx = 6
    for l in range(num_decode_layers):
        inputs[dec_names[base_idx + l]] = k_cache[l]
        inputs[dec_names[base_idx + num_decode_layers + l]] = v_cache[l]
        inputs[dec_names[base_idx + num_decode_layers*2 + l]] = k_list[l]
        inputs[dec_names[base_idx + num_decode_layers*3 + l]] = v_list[l]

    # 运行 Prefill
    out = prefill.run(inputs)
    
    logits = out[0]
    k_cache = out[1:1+num_decode_layers]
    v_cache = out[1+num_decode_layers:1+2*num_decode_layers]
    
    step += prompt_len

    print(">>> Prefill 完成，开始 Generate...")

    # === 阶段 2: Decode ===
    next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
    
    while step < MAX_GEN_LEN:
        all_tokens_list.append(next_token)
        
        current_tensor = torch.tensor([all_tokens_list], device=device)
        decoded_text = processor.batch_decode(current_tensor, skip_special_tokens=True)[0]
        print(f"\rStep {step}: {decoded_text}", end="", flush=True)
        
        if next_token == eos_id:
            print(f"\nStep {step}: {decoded_text} [EOS]")
            break
            
        input_ids = torch.tensor([[next_token]], device=device, dtype=torch.int32)
        
        mask_atten = torch.zeros(([1, num_heads, 1, CACHE_MAX_LEN]), device=device, dtype=torch.float16)
        if step + 1 < CACHE_MAX_LEN:
            mask_atten[:, :, :, step+1:] = -65504
        
        inputs = {
            dec_names[0]: input_ids,
            dec_names[1]: torch.tensor([[step]], device=device, dtype=torch.int32),
            dec_names[2]: torch.tensor([step], device=device, dtype=torch.int32),
            dec_names[3]: torch.tensor([1], device=device, dtype=torch.int32),
            dec_names[4]: mask_atten,
            dec_names[5]: encoder_attention_mask
        }
        
        base_idx = 6
        for l in range(num_decode_layers):
            inputs[dec_names[base_idx + l]] = k_cache[l]
            inputs[dec_names[base_idx + num_decode_layers + l]] = v_cache[l]
            inputs[dec_names[base_idx + num_decode_layers*2 + l]] = k_list[l]
            inputs[dec_names[base_idx + num_decode_layers*3 + l]] = v_list[l]
            
        out = decoder.run(inputs)
        logits = out[0]
        k_cache = out[1:1+num_decode_layers]
        v_cache = out[1+num_decode_layers:1+2*num_decode_layers]
        
        next_token = torch.argmax(logits[:, -1, :], dim=-1).item()
        step += 1

    print("\n🎉 推理完成！")
    print(f"最终结果: {decoded_text}")
    
    del decoder
    clean_memory()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf-model", type=str, default="/data01/home/USER/models/openai/whisper-large-v3-turbo")
    parser.add_argument("--hmonnx-model", type=str, default="/data01/home/USER/xh2modelzoo/examples/llm/whisper/new_work_dirs/whisper-large-v3-turbo_XH2a")
    # parser.add_argument("--audio-file", type=str, default="/data01/home/USER/DATA/audio.mp3", help="Path to audio file")
    args = parser.parse_args()
    main(args)