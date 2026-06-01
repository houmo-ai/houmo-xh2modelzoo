# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: cv3_eval.py
# Description:
#   CosyVoice3 HMONNX evaluation glue script (HOUMO eval pipeline).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import re
import os
import json
import time
import inflect
import whisper
import argparse

import torch
import torchaudio
import numpy as np
import torch.nn as nn
import onnxruntime as ort
import torch.nn.functional as F
from tqdm import tqdm
from typing import Generator
from typing import Callable, List
from functools import partial

try:
    import ttsfrd
    use_ttsfrd = True
except ImportError:
    print("failed to import ttsfrd, use wetext instead")
    from wetext import Normalizer as ZhNormalizer
    from wetext import Normalizer as EnNormalizer
    use_ttsfrd = False
from frontend_utils import contains_chinese, replace_blank, replace_corner_mark, remove_bracket, spell_out_number, split_paragraph, is_only_punctuation 

from xhquant.api import HMONNXInference, Config
from xh_model_zoo.xh_llm.models.builder import MODELS
from transformers import Qwen2ForCausalLM
from xh_model_zoo.xh_llm.models.cosyvoice3 import Qwen2_HFCompatible, XHQwen2HMONNXModel

from hyperpyyaml import load_hyperpyyaml
import torchaudio.compliance.kaldi as kaldi

import logging
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))  # CosyVoice 目录
matcha_path = os.path.join(script_dir, "third_party/Matcha-TTS")
sys.path.insert(0, matcha_path)  # 插到列表开头，优先级最高

from torch.multiprocessing import spawn, set_start_method
from types import GeneratorType

try:
    set_start_method('spawn')
except RuntimeError:
    pass

def worker(rank, gpus_to_use, args, configs, prompt_wav_dict, prompt_text_dict, text_dict, cv3_eval_path, dataset_exp_dir):
    gpu_id = gpus_to_use[rank]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    frontend = CosyVoiceFrontEnd(args, configs['get_tokenizer'], configs['feat_extractor'], configs['allowed_special'])
    
    utts = list(prompt_wav_dict.keys())
    for idx in tqdm(range(rank, len(utts), len(gpus_to_use)), desc=f"GPU {gpu_id} 处理数据"):
        utt = utts[idx]
        
        wav_path_out = os.path.join(dataset_exp_dir, f"{utt}.wav")
        # 如果已经存在就跳过
        if os.path.exists(wav_path_out):
            logging.info(f"{wav_path_out} 已存在，跳过生成")
            continue
        
        if utt not in prompt_text_dict or utt not in text_dict:
            logging.warning(f"utt {utt} 缺少prompt_text或target_text，跳过")
            continue

        prompt_text = 'You are a helpful assistant.<|endofprompt|>' + prompt_text_dict[utt]
        prompt_wav_path = prompt_wav_dict[utt]
        text = text_dict[utt]

        prompt_text = frontend.text_normalize(prompt_text, split=False, text_frontend=True)
        for i in frontend.text_normalize(text, split=True, text_frontend=True):
            if not isinstance(i, str) and len(i) < 0.5 * len(prompt_text):
                logging.warning(
                    f'synthesis text {i} too short than prompt text {prompt_text}, this may lead to bad performance'
                )
            
            model_input = frontend.frontend_zero_shot(
                i, prompt_text, os.path.join(cv3_eval_path, prompt_wav_path), configs['sample_rate']
            )
            logging.info(f'synthesis text {i}')
            
            # 推理
            this_tts_speech_token = llm_inference(args, model_input, device)
            this_tts_speech_token_un = torch.tensor(this_tts_speech_token).unsqueeze(0).to(device)
            
            # token -> wav
            this_tts_speech = token2wav(
                args,
                token=this_tts_speech_token_un,
                prompt_token=model_input['flow_prompt_speech_token'].to(device),
                prompt_feat=model_input['prompt_speech_feat'].to(device),
                embedding=model_input['flow_embedding'].to(device),
                token_offset=0,
                finalize=True,
                speed=1.0
            )
            this_tts_speech = this_tts_speech.to(torch.float32).to("cpu")
        
        # 保存 wav
        os.makedirs(dataset_exp_dir, exist_ok=True)
        torchaudio.save(os.path.join(dataset_exp_dir, f"{utt}.wav"), this_tts_speech, 24000)

class CosyVoiceFrontEnd:

    def __init__(self,
                 args,
                 get_tokenizer: Callable,
                 feat_extractor: Callable,
                 allowed_special: str = 'all'):
        self.tokenizer = get_tokenizer()
        self.feat_extractor = feat_extractor
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.allowed_special = allowed_special
        self.use_ttsfrd = use_ttsfrd
        if self.use_ttsfrd:
            self.frd = ttsfrd.TtsFrontendEngine()
            assert self.frd.initialize('/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/CosyVoice-ttsfrd') is True, \
                'failed to initialize ttsfrd resource'
            self.frd.set_lang_type('pinyinvg')
        else:
            self.zh_tn_model = ZhNormalizer(remove_erhua=False)
            self.en_tn_model = EnNormalizer()
            self.inflect_parser = inflect.engine()
        
        self.campplus_session = HMONNXInference(args.campplus)
        self.speech_tokenizer_session = HMONNXInference(args.speech_tokenizer_v3)

    def _extract_text_token(self, text):
        text_token = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        text_token = torch.tensor([text_token], dtype=torch.int32).to(self.device)
        text_token_len = torch.tensor([text_token.shape[1]], dtype=torch.int32).to(self.device)
        return text_token, text_token_len

    def _extract_speech_token(self, speech):
        speech = load_wav(speech, 16000)
        assert speech.shape[1] / 16000 <= 30, 'do not support extract speech token for audio longer than 30s'
        feat = whisper.log_mel_spectrogram(speech, n_mels=128)

        #生成hm speech_tokenizer——input
        feat = feat.half()
        feat_len = feat.shape[2]
        padded_input = torch.zeros((1, 128, 3000), dtype=torch.float16)
        padded_input[:, :, :feat_len] = feat
        mask_shape = (1, 20, 750, 750)
        mask = torch.full(mask_shape, torch.finfo(torch.float16).min, dtype=torch.float16)
        mask[:, :, :, :feat_len//4] = 0
        mask1 = torch.zeros((1, 750, 1280), dtype=torch.float16)
        mask1[:,0:feat_len//4,:] = 1.0
        
        input_names = self.speech_tokenizer_session.get_input_names()
        self.speech_tokenizer_session.save_golden = False
        self.speech_tokenizer_session.to(self.device)
        inputs = {
            input_names[0]:padded_input,
            input_names[1]:mask,
            input_names[2]:mask1,
        }
        speech_token = self.speech_tokenizer_session.run(inputs)
        speech_token = speech_token[:, :feat_len//4]
        speech_token = speech_token.to(self.device)
        speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        
        # import torch
        # import numpy as np
        # import onnxruntime

        # feat_len = feat.shape[2]

        # padded_input = torch.zeros((1, 128, 3000), dtype=torch.float32)
        # padded_input[:, :, :feat_len] = feat
        # padded_input = padded_input.cpu().numpy().astype(np.float32)

        # mask_shape = (1, 20, 750, 750)
        # mask = torch.full(mask_shape, -1e9, dtype=torch.float32)
        # mask[:, :, :, :feat_len//4] = 0.0
        # mask = mask.cpu().numpy().astype(np.float32)

        # mask1 = torch.zeros((1, 750, 1280), dtype=torch.float32)
        # mask1[:, 0:feat_len//4, :] = 1.0
        # mask1 = mask1.cpu().numpy().astype(np.float32)

        # onnx_model_path = "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/onnx/speech_tokenizer_v3_3000_3.onnx"
        # session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

        # input_names = [inp.name for inp in session.get_inputs()]

        # inputs = {
        #     input_names[0]: padded_input,
        #     input_names[1]: mask,
        #     input_names[2]: mask1,
        # }

        # outputs = session.run(None, inputs)
        # speech_token = torch.from_numpy(outputs[0]).to(torch.int64)
        # speech_token = speech_token[:, :feat_len//4]
        # speech_token = speech_token.to(self.device)
        # speech_token_len = torch.tensor([speech_token.shape[1]], dtype=torch.int32).to(self.device)
        return speech_token, speech_token_len

    def _extract_spk_embedding(self, speech):
        speech = load_wav(speech, 16000)
        feat = kaldi.fbank(speech,
                           num_mel_bins=80,
                           dither=0,
                           sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)

        T_fixed = 1000
        T = feat.shape[0]
        if T < T_fixed:
            feat = F.pad(feat, (0, 0, 0, T_fixed - T))  # 时间维度补零
        else:
            feat = feat[:T_fixed]
        feat = feat.half()
        input_names = self.campplus_session.get_input_names()
        self.campplus_session.save_golden = False
        self.campplus_session.to(self.device)
        inputs = {
            input_names[0]:feat.unsqueeze(0)
        }
        embedding = self.campplus_session.run(inputs)
        embedding = embedding.to(self.device)

        # import torch
        # import numpy as np
        # import onnxruntime

        # onnx_model_path = "/data01/home/she.gao/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/campplus.onnx"
        # session = onnxruntime.InferenceSession(onnx_model_path, providers=['CPUExecutionProvider'])

        # input_names = [inp.name for inp in session.get_inputs()]
        # feat = feat.unsqueeze(0).cpu().numpy().astype(np.float32)
        # inputs = {
        #     input_names[0]: feat,
        # }

        # outputs = session.run(None, inputs)
        # embedding = torch.from_numpy(outputs[0]).to(torch.float16).to(self.device)
        return embedding

    def _extract_speech_feat(self, speech):
        speech = load_wav(speech, 24000)
        speech_feat = self.feat_extractor(speech).squeeze(dim=0).transpose(0, 1).to(self.device)
        speech_feat = speech_feat.unsqueeze(dim=0)
        speech_feat_len = torch.tensor([speech_feat.shape[1]], dtype=torch.int32).to(self.device)
        return speech_feat, speech_feat_len

    def text_normalize(self, text, split=True, text_frontend=True):
        if isinstance(text, Generator):
            logging.info('get tts_text generator, will skip text_normalize!')
            return [text]
        if text_frontend is False or text == '':
            return [text] if split is True else text
        text = text.strip()
        if self.use_ttsfrd:
            texts = [i["text"] for i in json.loads(self.frd.do_voicegen_frd(text))["sentences"]]
            text = ''.join(texts)
        else:
            if contains_chinese(text):
                text = self.zh_tn_model.normalize(text)
                text = text.replace("\n", "")
                text = replace_blank(text)
                text = replace_corner_mark(text)
                text = text.replace(".", "。")
                text = text.replace(" - ", "，")
                text = remove_bracket(text)
                text = re.sub(r'[，,、]+$', '。', text)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "zh", token_max_n=80,
                                                token_min_n=60, merge_len=20, comma_split=False))
            else:
                text = self.en_tn_model.normalize(text)
                text = spell_out_number(text, self.inflect_parser)
                texts = list(split_paragraph(text, partial(self.tokenizer.encode, allowed_special=self.allowed_special), "en", token_max_n=80,
                                                token_min_n=60, merge_len=20, comma_split=False))
        texts = [i for i in texts if not is_only_punctuation(i)]
        return texts if split is True else text
    
    def frontend_zero_shot(self, tts_text, prompt_text, prompt_speech_16k, resample_rate):
        tts_text_token, tts_text_token_len = self._extract_text_token(tts_text)

        prompt_text_token, prompt_text_token_len = self._extract_text_token(prompt_text)
        # prompt_speech_resample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=resample_rate)(prompt_speech_16k)
        speech_feat, speech_feat_len = self._extract_speech_feat(prompt_speech_16k)
        speech_token, speech_token_len = self._extract_speech_token(prompt_speech_16k)
        if resample_rate == 24000:
            # cosyvoice2, force speech_feat % speech_token = 2
            token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
            speech_feat, speech_feat_len[:] = speech_feat[:, :2 * token_len], 2 * token_len
            speech_token, speech_token_len[:] = speech_token[:, :token_len], token_len
        embedding = self._extract_spk_embedding(prompt_speech_16k)
        model_input = {'prompt_text': prompt_text_token, 'prompt_text_len': prompt_text_token_len,
                        'llm_prompt_speech_token': speech_token, 'llm_prompt_speech_token_len': speech_token_len,
                        'flow_prompt_speech_token': speech_token, 'flow_prompt_speech_token_len': speech_token_len,
                        'prompt_speech_feat': speech_feat, 'prompt_speech_feat_len': speech_feat_len,
                        'llm_embedding': embedding, 'flow_embedding': embedding}
        model_input['text'] = tts_text_token
        model_input['text_len'] = tts_text_token_len
        return model_input


def load_wav(wav, target_sr):
    speech, sample_rate = torchaudio.load(wav, backend='soundfile')
    speech = speech.mean(dim=0, keepdim=True)
    if sample_rate != target_sr:
        # assert sample_rate > target_sr, 'wav sample rate {} must be greater than {}'.format(sample_rate, target_sr)
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)
    return speech

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0,
                             max_len,
                             dtype=torch.int64,
                             device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask

def llm_inference(args, model_input, device):
    this_tts_speech_token = []
    sampling = 25
    max_token_text_ratio = 20
    min_token_text_ratio = 2
    cur_silent_token_num, max_silent_token_num = 0, 5
    silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]
    
    # ---- 准备输入 ----
    text = model_input['text'].to(device)
    text_len = model_input['text_len']
    
    if 'prompt_text' not in model_input:
        prompt_text = torch.zeros(1, 0, dtype=torch.int32, device=device)
        prompt_text_len = torch.tensor([0], dtype=torch.int32, device=device)
    else:
        prompt_text = model_input['prompt_text'].to(device)
        prompt_text_len = model_input.get(
            'prompt_text_len',
            torch.tensor([prompt_text.shape[1]], dtype=torch.int32, device=device)
        )
    
    prompt_speech_token = model_input.get('llm_prompt_speech_token', 
                                          torch.zeros(1, 0, dtype=torch.int32, device=device)).to(device)

    # ---- 初始化模型 ----
    cfg = Config.fromfile(args.llm_config)
    qwen2_model: XHQwen2HMONNXModel = MODELS.build(cfg.model.llm)
    hf_model_native = Qwen2ForCausalLM.from_pretrained(args.qwen2_hf_model)
    hf_model = Qwen2_HFCompatible.to_hf_compatible(hf_model_native, qwen2_model)
    hf_model._llm_model._set_device(device)
    
    # ---- 将所有 embedding 放到同一 device ----
    text = torch.concat([prompt_text, text], dim=1)
    text_len += prompt_text_len
    text_emb = hf_model._llm_model.token_embedding(text).to(device)
    
    sos_eos_emb = torch.load(args.sos_eos_emb, map_location=device)
    task_id_emb = torch.load(args.task_id_emb, map_location=device)
    prompt_speech_token_emb = hf_model._llm_model.speech_embedding(prompt_speech_token).to(device)
    
    lm_input = torch.concat([sos_eos_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1)

    # ---- 计算 min/max length ----
    min_len = int((text_len - prompt_text_len) * min_token_text_ratio)
    max_len = int((text_len - prompt_text_len) * max_token_text_ratio)

    # ---- step by step decode ----
    lm_input = lm_input.to(torch.float16)
    token_generator = hf_model.generate(min_len=min_len, max_len=max_len, inputs_embeds=lm_input)
    
    for i in token_generator:
        if i in silent_tokens:
            cur_silent_token_num += 1
            if cur_silent_token_num > max_silent_token_num:
                continue
        else:
            cur_silent_token_num = 0
        this_tts_speech_token.append(i)
    
    return this_tts_speech_token

def token2wav(args, token, prompt_token, prompt_feat, embedding, token_offset, finalize=False, speed=1.0):
    assert token.shape[0] == 1
    inference_cfg_rate = 0.7
    token_mel_ratio = 2
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # encoder
    token = token.to(device)
    prompt_token = prompt_token.to(device)
    token_len = torch.tensor([token.shape[1]], dtype=torch.int32)
    prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32)
    token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
    input_embedding_state_dict = torch.load(
            "/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx/input_embedding.pt", map_location="cpu", weights_only=True
        )
    input_embedding = nn.Embedding(
        input_embedding_state_dict.shape[0],
        input_embedding_state_dict.shape[1],
    )
    input_embedding_state_dict_ = {"weight": input_embedding_state_dict}    
    input_embedding.load_state_dict(input_embedding_state_dict_)
    input_embedding.to("cuda")
    token = input_embedding(token)
    token = F.pad(token, (0, 0, 0, 1024 - token.shape[1]), value=0)
    token = token.to(torch.float16)
    
    flow_encoder_session = HMONNXInference(args.pre_lookahead_layer)
    input_names = flow_encoder_session.get_input_names()
    flow_encoder_session.save_golden = False
    flow_encoder_session.to(device)
    inputs = {
        input_names[0]:token
    }
    h = flow_encoder_session.run(inputs)
    h = h.repeat_interleave(token_mel_ratio, dim=1)

    #decoder
    sol = []
    embedding = F.normalize(embedding, dim=1)
    spk_embed_affine_layer = HMONNXInference(args.spk_embed_affine_layer)
    input_names = spk_embed_affine_layer.get_input_names()
    spk_embed_affine_layer.save_golden = False
    spk_embed_affine_layer.to(device)
    inputs = {
        input_names[0]:embedding
    }
    embedding = spk_embed_affine_layer.run(inputs)
    
    mel_len1, mel_len2 = prompt_feat.shape[1], token_len*2 - prompt_feat.shape[1]
    conds = torch.zeros([1, 2048, 80], device=token.device).to(h.dtype)
    conds[:, :mel_len1] = prompt_feat
    conds = conds.transpose(1, 2)
    mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]), 2048)).to(h)
    mask = mask.unsqueeze(1)
    mu = h.transpose(1, 2).contiguous()
    rand_noise = torch.randn([1, 80, 50 * 300])
    x = rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * 1.0
    t_span = torch.linspace(0, 1, 10 + 1, device=mu.device, dtype=mu.dtype)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
    t = t.unsqueeze(dim=0)
    x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
    mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=x.dtype)
    mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
    t_in = torch.zeros([2], device=x.device, dtype=x.dtype)
    cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=x.dtype)
    spks_in = torch.zeros([2, 80], device=x.device, dtype=x.dtype)

    decoder_session = HMONNXInference(args.flow_decoder)
    input_names_decoder = decoder_session.get_input_names()
    decoder_session.save_golden = False
    decoder_session.to(device)
    for step in range(1, len(t_span)):
        # Classifier-Free Guidance inference introduced in VoiceBox
        x_in[:] = x
        mask_in[:] = mask
        mu_in[0] = mu
        t_in[:] = t.unsqueeze(0)
        spks_in[0] = embedding
        cond_in[0] = conds
        input_decoder = {
            input_names_decoder[0]:x_in,
            input_names_decoder[1]:mask_in,
            input_names_decoder[2]:mu_in,
            input_names_decoder[3]:t_in,
            input_names_decoder[4]:spks_in,
            input_names_decoder[5]:cond_in,
        }
        dphi_dt = decoder_session.run(input_decoder)
        dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = ((1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt)
        x = x + dt * dphi_dt
        t = t + dt
        sol.append(x)
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t
    feat = sol[-1].float()
    feat = feat[:, :, mel_len1:mel_len1+mel_len2]

    #hift
    tts_mel = feat[:, :, 0:]
    needed = 1024 - tts_mel.size(2)
    if needed > 0:
        tts_mel = F.pad(tts_mel, (0, needed), value=0)
    else:
        tts_mel = tts_mel[:, :, :1024]
    if speed != 1.0:
        tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
    hift_session = HMONNXInference(args.hift)
    input_names_hift = hift_session.get_input_names()
    hift_session.save_golden = False
    hift_session.to(device)
    # hift_session.to_fast_mode()
    inputs_hift = {
        input_names_hift[0]:tts_mel.to(torch.float16),
    }
    tts_speech = hift_session.run(inputs_hift)
    tts_speech = tts_speech[:, :480*mel_len2]

    # import onnxruntime as ort

    # tts_mel = feat[:, :, 0:]
    # needed = 1024 - tts_mel.size(2)
    # if needed > 0:
    #     tts_mel = F.pad(tts_mel, (0, needed), value=0)
    # else:
    #     tts_mel = tts_mel[:, :, :1024]

    # if speed != 1.0:
    #     tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')

    # tts_mel_np = tts_mel.to(torch.float).cpu().numpy()

    # providers = ["CUDAExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    # session = ort.InferenceSession("/data01/home/she.gao/xhquant_llm/examples/CosyVoice/onnx/hift_1024_no_uni_fixed.onnx", providers=providers)

    # input_name = session.get_inputs()[0].name
    # inputs_hift = {input_name: tts_mel_np}

    # outputs = session.run(None, inputs_hift)
    # tts_speech = outputs[0]
    # tts_speech = tts_speech[:, :480 * mel_len2]
    # tts_speech = torch.from_numpy(tts_speech).to(device)
    return tts_speech

def load_wav_new(wav, target_sr, original_sr=None):
    """
    wav: numpy array 或文件路径
    target_sr: 目标采样率
    original_sr: 如果 wav 是 numpy array，需要提供原始采样率
    """
    # 如果输入是 numpy array 或 torch tensor
    if isinstance(wav, (np.ndarray, torch.Tensor)):
        if isinstance(wav, np.ndarray):
            speech = torch.from_numpy(wav).float()
        else:
            speech = wav.float()
        # 确保是 [1, N] 形状
        if speech.dim() == 1:
            speech = speech.unsqueeze(0)
        if original_sr is None:
            raise ValueError("For numpy array input, original_sr must be provided.")
        sample_rate = original_sr
    else:
        # 文件路径
        speech, sample_rate = torchaudio.load(wav, backend='soundfile')
        speech = speech.mean(dim=0, keepdim=True)  # 多通道转单通道

    # 重采样
    if sample_rate != target_sr:
        speech = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)(speech)

    return speech

def main(args):
    hyper_yaml_path = args.config_yaml
    if not os.path.exists(hyper_yaml_path):
        raise ValueError('{} not found!'.format(hyper_yaml_path))
    with open(hyper_yaml_path, 'r') as f:
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': os.path.join(args.cosyvoice_path, 'CosyVoice-BlankEN')})
    # frontend = CosyVoiceFrontEnd(args, configs['get_tokenizer'], configs['feat_extractor'], configs['allowed_special'])
    
    data_path = "/data01/home/she.gao/CV3-Eval/data/zero_shot"
    cv3_eval_path = "/data01/home/she.gao/CV3-Eval"
    exp_dir = "zeroshot_CosyVoice3-0.5B"
    os.makedirs(exp_dir, exist_ok=True)
    dataset_dirs = [d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))]
    
    dataset_dirs = ['zh']  # 或 ['zh', 'en', ...]
    # 选择使用的 GPU
    gpus_to_use = [0, 1, 2, 3, 4, 5, 6, 7]  # 只使用 GPU 0 和 GPU 2

    for dataset in dataset_dirs:
        seed_eval_path = os.path.join(data_path, dataset)

        required_files = ["prompt_wav.scp", "prompt_text", "text"]
        missing_files = [f for f in required_files if not os.path.exists(os.path.join(seed_eval_path, f))]
        if missing_files:
            logging.warning(f"未找到以下文件: {missing_files}，跳过该数据集")
            continue

        dataset_exp_dir = os.path.join(exp_dir, dataset)
        os.makedirs(dataset_exp_dir, exist_ok=True)

        # ==== 读取数据 ====
        prompt_wav_dict = {}
        with open(os.path.join(seed_eval_path, "prompt_wav.scp"), "r") as f:
            for line in f:
                utt, wav_path = line.strip().split(maxsplit=1)
                prompt_wav_dict[utt] = wav_path

        prompt_text_dict = {}
        with open(os.path.join(seed_eval_path, "prompt_text"), "r") as f:
            for line in f:
                utt, prompt_text = line.strip().split(maxsplit=1)
                prompt_text_dict[utt] = prompt_text

        text_dict = {}
        with open(os.path.join(seed_eval_path, "text"), "r") as f:
            for line in f:
                utt, target_text = line.strip().split(maxsplit=1)
                text_dict[utt] = target_text

        # ==== 启动多卡推理 ====
        world_size = len(gpus_to_use)
        spawn(
            worker,
            args=(gpus_to_use, args, configs, prompt_wav_dict, prompt_text_dict, text_dict, cv3_eval_path, dataset_exp_dir),
            nprocs=world_size,
            join=True
        )
    # for dataset in dataset_dirs:
    #     seed_eval_path = os.path.join(data_path, dataset)

    #     required_files = ["prompt_wav.scp", "prompt_text", "text"]
    #     missing_files = [f for f in required_files if not os.path.exists(os.path.join(seed_eval_path, f))]
    #     if missing_files:
    #         logging.warning(f"未找到以下文件: {missing_files}，跳过该数据集")
    #         continue

    #     dataset_exp_dir = os.path.join(exp_dir, dataset)
    #     os.makedirs(dataset_exp_dir, exist_ok=True)

    #     prompt_wav_dict = {}
    #     with open(os.path.join(seed_eval_path, "prompt_wav.scp"), "r") as f:
    #         for line in f:
    #             utt, wav_path = line.strip().split(maxsplit=1)  # 按第一个空格分割utt和路径
    #             prompt_wav_dict[utt] = wav_path

    #     prompt_text_dict = {}
    #     with open(os.path.join(seed_eval_path, "prompt_text"), "r") as f:
    #         for line in f:
    #             utt, prompt_text = line.strip().split(maxsplit=1)  # 按第一个空格分割utt和文本
    #             prompt_text_dict[utt] = prompt_text

    #     text_dict = {}
    #     with open(os.path.join(seed_eval_path, "text"), "r") as f:
    #         for line in f:
    #             utt, target_text = line.strip().split(maxsplit=1)  # 按第一个空格分割utt和文本
    #             text_dict[utt] = target_text

    #     for utt in tqdm(prompt_wav_dict.keys(), desc=f"处理数据集: {dataset}"):
    #         # 检查当前utt是否在所有数据字典中
    #         if utt not in prompt_text_dict or utt not in text_dict:
    #             logging.warning(f"utt {utt} 缺少prompt_text或target_text，跳过")
    #             continue
                
    #         # 获取当前utt的所有必要数据
    #         prompt_text = 'You are a helpful assistant.<|endofprompt|>' + prompt_text_dict[utt]
    #         prompt_wav_path = prompt_wav_dict[utt]
    #         text = text_dict[utt]

    #         prompt_text = frontend.text_normalize(prompt_text, split=False, text_frontend=True)
    #         for i in tqdm(frontend.text_normalize(text, split=True, text_frontend=True)):
    #             if (not isinstance(i, Generator)) and len(i) < 0.5 * len(prompt_text):
    #                     logging.warning('synthesis text {} too short than prompt text {}, this may lead to bad performance'.format(i, prompt_text))
    #             model_input = frontend.frontend_zero_shot(i, prompt_text, os.path.join(cv3_eval_path, prompt_wav_path), configs['sample_rate'])
    #             logging.info('synthesis text {}'.format(i))
    #             this_tts_speech_token = llm_inference(args, model_input)
    #             this_tts_speech_token_un = torch.tensor(this_tts_speech_token).unsqueeze(dim=0)
    #             this_tts_speech = token2wav(args,
    #                                         token=this_tts_speech_token_un,
    #                                         prompt_token=model_input['flow_prompt_speech_token'],
    #                                         prompt_feat=model_input['prompt_speech_feat'],
    #                                         embedding=model_input['flow_embedding'],
    #                                         token_offset=0,
    #                                         finalize=True,
    #                                         speed=1.0)
    #             this_tts_speech = this_tts_speech.to(torch.float32).to("cpu")
    #         torchaudio.save(os.path.join(dataset_exp_dir, f"{utt}.wav"), this_tts_speech, 24000)

def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--prompt_speech_16k",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/zero_shot_prompt.wav",
    )
    parser.add_argument(
        "--config_yaml",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/cosyvoice3.yaml",
    )
    parser.add_argument(
        "--cosyvoice_path",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3",
    )
    parser.add_argument(
        "--campplus",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx/campplus_1000.onnx",
    )
    parser.add_argument(
        "--speech_tokenizer_v3",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx/speech_tokenizer_v3_3000.onnx",
    )
    parser.add_argument(
        "--llm_config",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/config/qwen2_05b/qwen2_05b_instruct_xh2a_2k_hf.py",
    )
    parser.add_argument(
        "--qwen2_hf_model",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/CosyVoice-BlankEN",
    )
    parser.add_argument(
        "--sos_eos_emb",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx/sos_emb.pt",
    )
    parser.add_argument(
        "--task_id_emb",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx/task_id_emb.pt",
    )
    parser.add_argument(
        "--pre_lookahead_layer",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx/pre_lookahead_layer.onnx",
    )
    parser.add_argument(
        "--spk_embed_affine_layer",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx/spk.onnx",
    )
    parser.add_argument(
        "--input_embedding",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx/input_embedding.pt",
    )
    parser.add_argument(
        "--flow_decoder",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx/decoder_2048.onnx",
    )
    parser.add_argument(
        "--hift",
        type=str,
        default="/data01/home/she.gao/xh2modelzoo/examples/audio/Cosyvoice3/hmonnx/hift_1024.onnx",
    )
    return parser

if __name__ == "__main__":
    parser = parse_arguments()
    args = parser.parse_args()
    main(args)