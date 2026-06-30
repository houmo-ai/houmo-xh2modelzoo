# Copyright (c) 2025, Alibaba Cloud and its affiliates;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import librosa
import torch
import sys
import math
import os
import uuid
import torchaudio
from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoProcessor

from xh_model_zoo.xh_llm.models.funaudiochat.register import register_funaudiochat

# from funaudiochat.register import register_funaudiochat
register_funaudiochat()

from xh_model_zoo.xh_llm.models.funaudiochat.cosyvoice_detokenizer import get_audio_detokenizer, token2wav
# from utils.cosyvoice_detokenizer import get_audio_detokenizer, token2wav
from xh_model_zoo.xh_llm.models.funaudiochat.constant import (
    DEFAULT_S2M_GEN_KWARGS,
    DEFAULT_SP_GEN_KWARGS,
    DEFAULT_S2M_PROMPT,
    SPOKEN_S2M_PROMPT,
    AUDIO_TEMPLATE,
)
import soundfile as sf
device = "cuda:0" if torch.cuda.is_available() else "cpu"
os.environ["DIFFUSERS_NO_XFORMERS"] = "1"


def infer_example(model_path, audio_path):
    """
    推理示例函数
    
    Args:
        model_path: 模型路径
        audio_path: 输入音频路径
    """
    # 加载模型和处理器
    config = AutoConfig.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)

    config.audio_config.crq_transformer_config["torch_dtype"] = torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, device_map=device)

    # 生成参数
    sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
    sp_gen_kwargs['text_greedy'] = True
    gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
    gen_kwargs['max_new_tokens'] = 2048
    model.sp_gen_kwargs.update(sp_gen_kwargs)

    # 构建audio样例
    audio = [librosa.load(audio_path, sr=16000)[0]]
    
    conversation = [
        {"role": "system", "content": SPOKEN_S2M_PROMPT},
        {"role": "user", "content": AUDIO_TEMPLATE},
    ]

    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(model.device)
    generate_ids, audio_ids = model.generate(**inputs, **gen_kwargs)
    generate_ids = generate_ids[:, inputs.input_ids.size(1):]
    generate_text = processor.decode(generate_ids[0], skip_special_tokens=True)
    generate_audio = processor.speech_tokenizer.decode(audio_ids[0])

    print("generate_text: ", generate_text)
    print("generate_audio_token: ", generate_audio)
    
    token_for_cosyvoice = list(filter(lambda x: 0 <= x < 6561, audio_ids[0].tolist()))

    # 加载CosyVoice detokenizer用于将token转换为wav
    print("Loading CosyVoice detokenizer...")
    cosyvoice_model = get_audio_detokenizer(model_path='/data01/datasets/Fun-CosyVoice3-0.5B-2512')
    
    # (使用默认的中文女声，你可以根据需要修改）
    print("Converting audio tokens to wav...")
    speech = token2wav(cosyvoice_model, token_for_cosyvoice, embedding=None, token_hop_len=25 * 30, pre_lookahead_len=3)
    
    # 保存wav文件
    output_uuid = str(uuid.uuid4())
    os.makedirs('saves', exist_ok=True)
    # torchaudio.set_audio_backend("sox_io")
    output_path = f'saves/output_audio_{output_uuid}.wav'
    # torchaudio.save(output_path, speech.cpu(), cosyvoice_model.sample_rate)
    audio_np = speech.cpu().squeeze().detach().numpy()
    sf.write(output_path, audio_np, cosyvoice_model.sample_rate)
    print(f"Audio saved to: {output_path}")

def infer_multiturn_example(model_path, audio_paths):
    """
    多轮对话推理示例函数，这里我们采用assistant生成的text作为下一轮的输入
    
    Args:
        model_path: 模型路径
        audio_paths: 输入音频路径列表
    """
    # 加载模型和处理器
    config = AutoConfig.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, config=config, torch_dtype=torch.float16, device_map=device)

    # 加载CosyVoice detokenizer用于将token转换为wav
    print("Loading CosyVoice detokenizer...")
    cosyvoice_model = get_audio_detokenizer()

    # 生成参数
    sp_gen_kwargs = DEFAULT_SP_GEN_KWARGS.copy()
    sp_gen_kwargs['text_greedy'] = True
    gen_kwargs = DEFAULT_S2M_GEN_KWARGS.copy()
    gen_kwargs['max_new_tokens'] = 2048
    model.sp_gen_kwargs.update(sp_gen_kwargs)

    conversation = [
        {"role": "system", "content": SPOKEN_S2M_PROMPT},
    ]
    audio = []
    for index, audio_path in enumerate(audio_paths):
        # ignore assistant ground truth audio
        if index % 2 == 1:
            continue
        
        audio.append(librosa.load(audio_path, sr=16000)[0])
        conversation.append({"role": "user", "content": AUDIO_TEMPLATE})

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = processor(text=text, audio=audio, return_tensors="pt", return_token_type_ids=False).to(model.device)
        generate_ids, audio_ids = model.generate(**inputs, **gen_kwargs)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]
        generate_text = processor.decode(generate_ids[0], skip_special_tokens=True)
        generate_audio = processor.speech_tokenizer.decode(audio_ids[0])

        print("generate_text: ", generate_text)
        print("generate_audio_token: ", generate_audio)
        
        token_for_cosyvoice = list(filter(lambda x: 0 <= x < 6561, audio_ids[0].tolist()))

        # 使用默认的中文女声，你可以根据需要修改
        print("Converting audio tokens to wav...")
        speech = token2wav(cosyvoice_model, token_for_cosyvoice, embedding=None, token_hop_len=25 * 30, pre_lookahead_len=3)
        
        # 保存wav文件
        output_uuid = str(uuid.uuid4())
        os.makedirs('saves', exist_ok=True)
        output_path = f'saves/output_audio_{output_uuid}.wav'
        torchaudio.save(output_path, speech.cpu(), cosyvoice_model.sample_rate)
        print(f"Audio saved to: {output_path}")

        # add assistant response to conversation
        conversation.append({"role": "assistant", "content": generate_text})

if __name__ == "__main__":
    model_path = "/data01/datasets/Fun-Audio-Chat-8B"
    audio_path = "examples/audio/fun_audio_chat/asr_example_hotword.wav"
    infer_example(model_path, audio_path)
