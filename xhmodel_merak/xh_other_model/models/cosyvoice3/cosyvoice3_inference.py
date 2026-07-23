# Copyright 2025 HOUMO AI
#
# File: cosyvoice3_inference.py
# Description:
#   CosyVoice3 unified HMONNX inference model. Encapsulates the full TTS
#   pipeline (frontend → LLM generate → token2wav) into a single registered
#   class with generate_by_mode(), mirroring qwen3_tts's Qwen3TTSHMONNXInference.
#
#   De-cosyvoice: uses transformers.AutoTokenizer + torchaudio mel instead of
#   hyperpyyaml/cosyvoice/matcha/whisper. sos_eos_emb/task_id_emb are optional
#   (not in checkpoint; must be provided by user or extracted via cosyvoice lib).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import re
from functools import partial
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, Qwen2ForCausalLM

from xhquant.api import HMONNXInference

from ...builder import MODELS, register_other_model
from .frontend_utils import (
    contains_chinese,
    is_only_punctuation,
    remove_bracket,
    replace_blank,
    replace_corner_mark,
    spell_out_number,
    split_paragraph,
)
from .llm_hmonnx_model import XHQwen2HMONNXModel
from .qwen2_hf_compatible import Qwen2_HFCompatible


# ---------------------------------------------------------------------------
# Local replacements for cosyvoice / matcha library functions
# ---------------------------------------------------------------------------

_COSYVOICE3_SPECIAL_TOKENS = [
    "<|im_start|>",
    "<|im_end|>",
    "<|endofprompt|>",
    "[breath]",
    "<strong>",
    "</strong>",
    "[noise]",
    "[laughter]",
    "[cough]",
    "[clucking]",
    "[accent]",
    "[quick_breath]",
    "<laughter>",
    "</laughter>",
    "[hissing]",
    "[sigh]",
    "[vocalized-noise]",
    "[lipsmack]",
    "[mn]",
    "<|endofsystem|>",
    "[AA]",
    "[AA0]",
    "[AA1]",
    "[AA2]",
    "[AE]",
    "[AE0]",
    "[AE1]",
    "[AE2]",
    "[AH]",
    "[AH0]",
    "[AH1]",
    "[AH2]",
    "[AO]",
    "[AO0]",
    "[AO1]",
    "[AO2]",
    "[AW]",
    "[AW0]",
    "[AW1]",
    "[AW2]",
    "[AY]",
    "[AY0]",
    "[AY1]",
    "[AY2]",
    "[B]",
    "[CH]",
    "[D]",
    "[DH]",
    "[EH]",
    "[EH0]",
    "[EH1]",
    "[EH2]",
    "[ER]",
    "[ER0]",
    "[ER1]",
    "[ER2]",
    "[EY]",
    "[EY0]",
    "[EY1]",
    "[EY2]",
    "[F]",
    "[G]",
    "[HH]",
    "[IH]",
    "[IH0]",
    "[IH1]",
    "[IH2]",
    "[IY]",
    "[IY0]",
    "[IY1]",
    "[IY2]",
    "[JH]",
    "[K]",
    "[L]",
    "[M]",
    "[N]",
    "[NG]",
    "[OW]",
    "[OW0]",
    "[OW1]",
    "[OW2]",
    "[OY]",
    "[OY0]",
    "[OY1]",
    "[OY2]",
    "[P]",
    "[R]",
    "[S]",
    "[SH]",
    "[T]",
    "[TH]",
    "[UH]",
    "[UH0]",
    "[UH1]",
    "[UH2]",
    "[UW]",
    "[UW0]",
    "[UW1]",
    "[UW2]",
    "[V]",
    "[W]",
    "[Y]",
    "[Z]",
    "[ZH]",
    "[a]",
    "[ai]",
    "[an]",
    "[ang]",
    "[ao]",
    "[b]",
    "[c]",
    "[ch]",
    "[d]",
    "[e]",
    "[ei]",
    "[en]",
    "[eng]",
    "[f]",
    "[g]",
    "[h]",
    "[i]",
    "[ian]",
    "[in]",
    "[ing]",
    "[iu]",
    "[ià]",
    "[iàn]",
    "[iàng]",
    "[iào]",
    "[iá]",
    "[ián]",
    "[iáng]",
    "[iáo]",
    "[iè]",
    "[ié]",
    "[iòng]",
    "[ióng]",
    "[iù]",
    "[iú]",
    "[iā]",
    "[iān]",
    "[iāng]",
    "[iāo]",
    "[iē]",
    "[iě]",
    "[iōng]",
    "[iū]",
    "[iǎ]",
    "[iǎn]",
    "[iǎng]",
    "[iǎo]",
    "[iǒng]",
    "[iǔ]",
    "[j]",
    "[k]",
    "[l]",
    "[m]",
    "[n]",
    "[o]",
    "[ong]",
    "[ou]",
    "[p]",
    "[q]",
    "[r]",
    "[s]",
    "[sh]",
    "[t]",
    "[u]",
    "[uang]",
    "[ue]",
    "[un]",
    "[uo]",
    "[uà]",
    "[uài]",
    "[uàn]",
    "[uàng]",
    "[uá]",
    "[uái]",
    "[uán]",
    "[uáng]",
    "[uè]",
    "[ué]",
    "[uì]",
    "[uí]",
    "[uò]",
    "[uó]",
    "[uā]",
    "[uāi]",
    "[uān]",
    "[uāng]",
    "[uē]",
    "[uě]",
    "[uī]",
    "[uō]",
    "[uǎ]",
    "[uǎi]",
    "[uǎn]",
    "[uǎng]",
    "[uǐ]",
    "[uǒ]",
    "[vè]",
    "[w]",
    "[x]",
    "[y]",
    "[z]",
    "[zh]",
    "[à]",
    "[ài]",
    "[àn]",
    "[àng]",
    "[ào]",
    "[á]",
    "[ái]",
    "[án]",
    "[áng]",
    "[áo]",
    "[è]",
    "[èi]",
    "[èn]",
    "[èng]",
    "[èr]",
    "[é]",
    "[éi]",
    "[én]",
    "[éng]",
    "[ér]",
    "[ì]",
    "[ìn]",
    "[ìng]",
    "[í]",
    "[ín]",
    "[íng]",
    "[ò]",
    "[òng]",
    "[òu]",
    "[ó]",
    "[óng]",
    "[óu]",
    "[ù]",
    "[ùn]",
    "[ú]",
    "[ún]",
    "[ā]",
    "[āi]",
    "[ān]",
    "[āng]",
    "[āo]",
    "[ē]",
    "[ēi]",
    "[ēn]",
    "[ēng]",
    "[ě]",
    "[ěi]",
    "[ěn]",
    "[ěng]",
    "[ěr]",
    "[ī]",
    "[īn]",
    "[īng]",
    "[ō]",
    "[ōng]",
    "[ōu]",
    "[ū]",
    "[ūn]",
    "[ǎ]",
    "[ǎi]",
    "[ǎn]",
    "[ǎng]",
    "[ǎo]",
    "[ǐ]",
    "[ǐn]",
    "[ǐng]",
    "[ǒ]",
    "[ǒng]",
    "[ǒu]",
    "[ǔ]",
    "[ǔn]",
    "[ǘ]",
    "[ǚ]",
    "[ǜ]",
]


class _CosyVoice3Tokenizer:
    """Replaces cosyvoice.tokenizer.tokenizer.get_qwen_tokenizer."""

    def __init__(self, token_path: str, skip_special_tokens: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(token_path)
        # Qwen2 tokenizer already has correct
        # Only add CosyVoice3 pinyin special tokens.
        self.tokenizer.add_special_tokens({"additional_special_tokens": _COSYVOICE3_SPECIAL_TOKENS})
        self.skip_special_tokens = skip_special_tokens

    def encode(self, text, **kwargs):
        return self.tokenizer([text], return_tensors="pt")["input_ids"][0].cpu().tolist()


_mel_basis_cache = {}
_hann_window_cache = {}


def _matcha_mel(y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
    """Replaces matcha.utils.audio.mel_spectrogram."""
    from librosa.filters import mel as librosa_mel_fn

    key = f"{fmax}_{y.device}"
    if key not in _mel_basis_cache:
        mel = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
        _mel_basis_cache[key] = torch.from_numpy(mel).float().to(y.device)
        _hann_window_cache[str(y.device)] = torch.hann_window(win_size).to(y.device)

    y = F.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)), mode="reflect")
    y = y.squeeze(1)

    spec = torch.view_as_real(
        torch.stft(
            y,
            n_fft,
            hop_length=hop_size,
            win_length=win_size,
            window=_hann_window_cache[str(y.device)],
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=True,
        )
    )
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
    spec = torch.matmul(_mel_basis_cache[key], spec)
    return torch.log(torch.clamp(spec, min=1e-5))


def _load_wav(path: str, target_sr: int) -> torch.Tensor:
    import torchaudio

    speech, sr = torchaudio.load(path, backend="soundfile")
    speech = speech.mean(dim=0, keepdim=True)
    if sr != target_sr:
        speech = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(speech)
    return speech


def _make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else int(lengths.max().item())
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    mask = seq_range.unsqueeze(0).expand(batch_size, max_len) >= lengths.unsqueeze(-1)
    return mask


@register_other_model("CosyVoice3HMONNXInference", master=False)
class CosyVoice3HMONNXInference:
    """Unified CosyVoice3 HMONNX inference model.

    Mirrors qwen3_tts's Qwen3TTSHMONNXInference: a single registered class that
    builds all HMONNX sessions from export_meta_info and exposes generate_by_mode().
    """

    def __init__(
        self,
        hf_model: str,
        llm: dict,
        campplus: dict,
        speech_tokenizer_v3: dict,
        flow_decoder: dict,
        hift: dict,
        pre_lookahead_layer: dict,
        spk_embed_affine_layer: dict,
        input_embedding: dict,
        sos_eos_emb: Optional[dict] = None,
        task_id_emb: Optional[dict] = None,
    ) -> None:
        self.hf_model_dir = hf_model
        self.device = torch.device("cpu")

        self.tokenizer = _CosyVoice3Tokenizer(token_path=hf_model, skip_special_tokens=True)
        self.allowed_special = "all"
        try:
            from wetext import Normalizer as _WetextNormalizer
        except ImportError as e:
            raise ImportError(
                "CosyVoice3 inference requires 'wetext' for text normalization. Install: pip install wetext"
            ) from e
        self.zh_tn_model = _WetextNormalizer(remove_erhua=False)
        self.en_tn_model = _WetextNormalizer()
        try:
            import inflect

            self.inflect_parser = inflect.engine()
        except ImportError as e:
            raise ImportError(
                "CosyVoice3 inference requires 'inflect' for English number-to-word conversion. "
                "Install: pip install inflect"
            ) from e

        self.llm_model: XHQwen2HMONNXModel = MODELS.build(
            {
                "type": "XHCosyVoice3LLMHMONNX",
                "model_dir": llm["model_dir"],
            }
        )
        hf_native = Qwen2ForCausalLM.from_pretrained(hf_model, torch_dtype=torch.float16)
        self.hf_model_wrapped = Qwen2_HFCompatible.to_hf_compatible(hf_native, self.llm_model)

        def _sess(cfg):
            onnx_file = cfg.get("onnx_file")
            if onnx_file is None:
                return None
            s = HMONNXInference(onnx_file)
            s.save_golden = False
            return s

        self.campplus_session = _sess(campplus)
        self.speech_tokenizer_session = _sess(speech_tokenizer_v3)
        self.flow_decoder_session = _sess(flow_decoder)
        self.hift_session = _sess(hift)
        self.pre_lookahead_session = _sess(pre_lookahead_layer)
        self.spk_aff_session = _sess(spk_embed_affine_layer)

        in_emb_w = torch.load(input_embedding["file"], map_location="cpu", weights_only=True)
        w = in_emb_w["weight"] if isinstance(in_emb_w, dict) else in_emb_w
        self.input_embedding = nn.Embedding(w.shape[0], w.shape[1])
        self.input_embedding.load_state_dict({"weight": w})

        self.sos_eos_emb = torch.load(sos_eos_emb["file"], map_location="cpu") if sos_eos_emb else None
        self.task_id_emb = torch.load(task_id_emb["file"], map_location="cpu") if task_id_emb else None

    def to(self, device: str) -> "CosyVoice3HMONNXInference":
        self.device = torch.device(device)
        self.llm_model._set_device(self.device)
        self.hf_model_wrapped._llm_model._set_device(self.device)
        for s in [
            self.campplus_session,
            self.speech_tokenizer_session,
            self.flow_decoder_session,
            self.hift_session,
            self.pre_lookahead_session,
            self.spk_aff_session,
        ]:
            s.to(self.device)
        self.input_embedding.to(self.device)
        if self.sos_eos_emb is not None:
            self.sos_eos_emb = self.sos_eos_emb.to(self.device)
        if self.task_id_emb is not None:
            self.task_id_emb = self.task_id_emb.to(self.device)
        return self

    # ------------------------------------------------------------------
    # Frontend (de-cosyvoice: AutoTokenizer + torchaudio mel + kaldi.fbank)
    # ------------------------------------------------------------------

    def _extract_text_token(self, text: str):
        tokens = self.tokenizer.encode(text, allowed_special=self.allowed_special)
        return torch.tensor([tokens], dtype=torch.int32, device=self.device)

    def _extract_speech_token(self, speech_16k: torch.Tensor):
        try:
            import whisper
        except ImportError as e:
            raise ImportError(
                "CosyVoice3 inference requires 'openai-whisper' for speech tokenization. "
                "Install: pip install openai-whisper"
            ) from e
        feat = whisper.log_mel_spectrogram(speech_16k, n_mels=128).half()
        feat_len = feat.shape[2]
        padded = torch.zeros((1, 128, 3000), dtype=torch.float16, device=self.device)
        padded[:, :, :feat_len] = feat
        mask = torch.full((1, 20, 750, 750), torch.finfo(torch.float16).min, dtype=torch.float16, device=self.device)
        mask[:, :, :, : feat_len // 4] = 0
        mask1 = torch.zeros((1, 750, 1280), dtype=torch.float16, device=self.device)
        mask1[:, 0 : feat_len // 4, :] = 1.0
        names = self.speech_tokenizer_session.get_input_names()
        out = self.speech_tokenizer_session.run({names[0]: padded, names[1]: mask, names[2]: mask1})
        return out[:, : feat_len // 4].to(self.device)

    def _extract_spk_embedding(self, speech_16k: torch.Tensor):
        import torchaudio.compliance.kaldi as kaldi

        feat = kaldi.fbank(speech_16k, num_mel_bins=80, dither=0, sample_frequency=16000)
        feat = feat - feat.mean(dim=0, keepdim=True)
        t_fixed = 1000
        if feat.shape[0] < t_fixed:
            feat = F.pad(feat, (0, 0, 0, t_fixed - feat.shape[0]))
        else:
            feat = feat[:t_fixed]
        feat = feat.half()
        names = self.campplus_session.get_input_names()
        return self.campplus_session.run({names[0]: feat.unsqueeze(0)}).to(self.device)

    def _extract_speech_feat(self, speech_24k: torch.Tensor):
        feat = _matcha_mel(
            speech_24k,
            n_fft=1920,
            num_mels=80,
            sampling_rate=24000,
            hop_size=480,
            win_size=1920,
            fmin=0,
            fmax=None,
            center=False,
        )
        feat = feat.squeeze(0).transpose(0, 1)
        return feat.unsqueeze(0).to(self.device)

    def _text_normalize(self, text: str, split: bool = True):
        text = text.strip()
        if not text:
            return [text] if split else text
        if contains_chinese(text):
            text = self.zh_tn_model.normalize(text)
            text = text.replace("\n", "")
            text = replace_blank(text)
            text = replace_corner_mark(text)
            text = text.replace(".", "。")
            text = text.replace(" - ", "，")
            text = remove_bracket(text)
            text = re.sub(r"[，,、]+$", "。", text)
            tokenize = partial(self.tokenizer.encode, allowed_special=self.allowed_special)
            sentences = list(
                split_paragraph(text, tokenize, "zh", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False)
            )
        else:
            text = self.en_tn_model.normalize(text)
            if self.inflect_parser is not None:
                text = spell_out_number(text, self.inflect_parser)
            tokenize = partial(self.tokenizer.encode, allowed_special=self.allowed_special)
            sentences = list(
                split_paragraph(text, tokenize, "en", token_max_n=80, token_min_n=60, merge_len=20, comma_split=False)
            )
        sentences = [s for s in sentences if not is_only_punctuation(s)]
        return sentences if split else text

    def _frontend_zero_shot(self, tts_text: str, prompt_text: str, prompt_wav: str):
        prompt_16k = _load_wav(prompt_wav, 16000)
        prompt_24k = _load_wav(prompt_wav, 24000)
        tts_token = self._extract_text_token(tts_text)
        prompt_token = self._extract_text_token(prompt_text)
        speech_token = self._extract_speech_token(prompt_16k)
        speech_feat = self._extract_speech_feat(prompt_24k)
        token_len = min(int(speech_feat.shape[1] / 2), speech_token.shape[1])
        speech_feat = speech_feat[:, : 2 * token_len]
        speech_token = speech_token[:, :token_len]
        embedding = self._extract_spk_embedding(prompt_16k)
        return {
            "text": tts_token,
            "prompt_text": prompt_token,
            "llm_prompt_speech_token": speech_token,
            "flow_prompt_speech_token": speech_token,
            "prompt_speech_feat": speech_feat,
            "llm_embedding": embedding,
            "flow_embedding": embedding,
        }

    # ------------------------------------------------------------------
    # LLM inference (XHQwen2HMONNXModel + Qwen2_HFCompatible + generate)
    # ------------------------------------------------------------------

    def _llm_inference(self, model_input: dict) -> List[int]:
        if self.sos_eos_emb is None or self.task_id_emb is None:
            raise ValueError(
                "sos_eos_emb and task_id_emb are required for LLM generation but "
                "not provided. These are CosyVoice-specific embeddings not in the "
                "checkpoint. Provide them via hmonnx_utils config or extract from "
                "a cosyvoice library environment."
            )
        device = self.device
        inner = self.hf_model_wrapped._llm_model
        inner.set_input_sequence_length(inner.prefill_input_sequence_length)
        inner.token_embedding = inner.token_embedding.to(device)
        inner.speech_embedding = inner.speech_embedding.to(device)
        text = model_input["text"].to(device)
        prompt_text = model_input["prompt_text"].to(device)
        prompt_speech_token = model_input["llm_prompt_speech_token"].to(device)

        prompt_text_len = prompt_text.shape[1]
        text = torch.concat([prompt_text, text], dim=1)
        text_emb = self.hf_model_wrapped._llm_model.token_embedding(text).to(device)
        prompt_speech_token_emb = self.hf_model_wrapped._llm_model.speech_embedding(prompt_speech_token).to(device)
        lm_input = torch.concat([self.sos_eos_emb, text_emb, self.task_id_emb, prompt_speech_token_emb], dim=1).to(
            torch.float16
        )

        text_len = text.shape[1]
        min_len = int((text_len - prompt_text_len) * 2)
        max_len = int((text_len - prompt_text_len) * 20)

        silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]
        out: List[int] = []
        cur_silent = 0
        for tok in self.hf_model_wrapped.generate(min_len=min_len, max_len=max_len, inputs_embeds=lm_input):
            if tok in silent_tokens:
                cur_silent += 1
                if cur_silent > 5:
                    continue
            else:
                cur_silent = 0
            out.append(int(tok))
        return out

    def _llm_token_generator(self, model_input: dict):
        """Generator version of _llm_inference: yields tokens one by one."""
        if self.sos_eos_emb is None or self.task_id_emb is None:
            raise ValueError(
                "sos_eos_emb and task_id_emb are required for LLM generation but "
                "not provided. Provide them via hmonnx_utils config or extract from "
                "a cosyvoice library environment."
            )
        device = self.device
        inner = self.hf_model_wrapped._llm_model
        inner.set_input_sequence_length(inner.prefill_input_sequence_length)
        inner.token_embedding = inner.token_embedding.to(device)
        inner.speech_embedding = inner.speech_embedding.to(device)
        text = model_input["text"].to(device)
        prompt_text = model_input["prompt_text"].to(device)
        prompt_speech_token = model_input["llm_prompt_speech_token"].to(device)

        prompt_text_len = prompt_text.shape[1]
        text = torch.concat([prompt_text, text], dim=1)
        text_emb = self.hf_model_wrapped._llm_model.token_embedding(text).to(device)
        prompt_speech_token_emb = self.hf_model_wrapped._llm_model.speech_embedding(prompt_speech_token).to(device)
        lm_input = torch.concat([self.sos_eos_emb, text_emb, self.task_id_emb, prompt_speech_token_emb], dim=1).to(
            torch.float16
        )

        text_len = text.shape[1]
        min_len = int((text_len - prompt_text_len) * 2)
        max_len = int((text_len - prompt_text_len) * 20)

        silent_tokens = [1, 2, 28, 29, 55, 248, 494, 2241, 2242, 2322, 2323]
        cur_silent = 0
        for tok in self.hf_model_wrapped.generate(min_len=min_len, max_len=max_len, inputs_embeds=lm_input):
            if tok in silent_tokens:
                cur_silent += 1
                if cur_silent > 5:
                    continue
            else:
                cur_silent = 0
            yield int(tok)

    # ------------------------------------------------------------------
    # token2wav (pre_lookahead → spk_aff → flow_decoder CFM → hift)
    # ------------------------------------------------------------------

    def _token2wav(
        self,
        token: torch.Tensor,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        speed: float = 1.0,
    ) -> torch.Tensor:
        device = self.device
        cfg_rate = 0.7
        token_mel_ratio = 2

        token = token.to(device)
        prompt_token = prompt_token.to(device)
        token_len = torch.tensor([token.shape[1]], dtype=torch.int32)
        prompt_token_len = torch.tensor([prompt_token.shape[1]], dtype=torch.int32)
        token = torch.concat([prompt_token, token], dim=1)
        token_len = prompt_token_len + token_len

        token = self.input_embedding(token)
        token = F.pad(token, (0, 0, 0, 1024 - token.shape[1]), value=0).to(torch.float16)

        n = self.pre_lookahead_session.get_input_names()
        h = self.pre_lookahead_session.run({n[0]: token}).repeat_interleave(token_mel_ratio, dim=1)

        embedding = F.normalize(embedding, dim=1)
        n = self.spk_aff_session.get_input_names()
        embedding = self.spk_aff_session.run({n[0]: embedding})

        mel_len1 = prompt_feat.shape[1]
        mel_len2 = token_len * 2 - prompt_feat.shape[1]
        conds = torch.zeros([1, 2048, 80], device=device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)
        mask = (~_make_pad_mask(torch.tensor([mel_len1 + mel_len2]), 2048)).to(h).unsqueeze(1)
        mu = h.transpose(1, 2).contiguous()
        rand_noise = torch.randn([1, 80, 50 * 300])
        x = rand_noise[:, :, : mu.size(2)].to(mu.device).to(mu.dtype) * 1.0
        t_span = torch.linspace(0, 1, 11, device=device, dtype=mu.dtype)
        t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(0)

        x_in = torch.zeros([2, 80, x.size(2)], device=device, dtype=x.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=device, dtype=x.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=device, dtype=x.dtype)
        t_in = torch.zeros([2], device=device, dtype=x.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=device, dtype=x.dtype)
        spks_in = torch.zeros([2, 80], device=device, dtype=x.dtype)

        nd = self.flow_decoder_session.get_input_names()
        sol = []
        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = embedding
            cond_in[0] = conds
            d = self.flow_decoder_session.run(
                {
                    nd[0]: x_in,
                    nd[1]: mask_in,
                    nd[2]: mu_in,
                    nd[3]: t_in,
                    nd[4]: spks_in,
                    nd[5]: cond_in,
                }
            )
            d, cfg_d = torch.split(d, [x.size(0), x.size(0)], dim=0)
            d = (1.0 + cfg_rate) * d - cfg_rate * cfg_d
            x = x + dt * d
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t

        feat = sol[-1].float()
        feat = feat[:, :, mel_len1 : mel_len1 + mel_len2]

        tts_mel = feat[:, :, 0:]
        needed = 1024 - tts_mel.size(2)
        if needed > 0:
            tts_mel = F.pad(tts_mel, (0, needed), value=0)
        else:
            tts_mel = tts_mel[:, :, :1024]
        if speed != 1.0:
            tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode="linear")

        nh = self.hift_session.get_input_names()
        wav = self.hift_session.run({nh[0]: tts_mel.to(torch.float16)})
        return wav[:, : 480 * mel_len2]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        text: str,
        ref_text: Optional[str] = None,
        prompt_wav: Optional[str] = None,
        prompt_text: Optional[str] = None,
        **kwargs,
    ):
        """Generate TTS audio via zero-shot mode (prompt_wav + prompt_text).

        ``prompt_wav`` is required. ``prompt_text`` is the transcription of the
        prompt audio. ``ref_text`` is an alias for ``prompt_text``.
        """
        if not prompt_wav:
            raise ValueError("CosyVoice3 generate requires prompt_wav (zero-shot mode)")
        prompt_text = prompt_text or ref_text or ""
        prompt_text_full = "You are a helpful assistant.<|endofprompt|>" + prompt_text
        prompt_text_full = self._text_normalize(prompt_text_full, split=False)

        wavs = []
        for sent in self._text_normalize(text, split=True):
            if not isinstance(sent, str) or not sent.strip():
                continue
            model_input = self._frontend_zero_shot(sent, prompt_text_full, prompt_wav)
            tokens = self._llm_inference(model_input)
            token_t = torch.tensor(tokens).unsqueeze(0).to(self.device)
            wav = self._token2wav(
                token=token_t,
                prompt_token=model_input["flow_prompt_speech_token"].to(self.device),
                prompt_feat=model_input["prompt_speech_feat"].to(self.device),
                embedding=model_input["flow_embedding"].to(self.device),
            )
            wavs.append(wav.to(torch.float32).to("cpu"))

        if not wavs:
            raise RuntimeError("No audio generated; check input text/prompt")
        final = torch.cat(wavs, dim=1)
        return [final.numpy().astype("float32")], 24000

    # ------------------------------------------------------------------
    # Streaming generation (25-token chunk, ported from cv3_stream.py)
    # ------------------------------------------------------------------

    HIFT_FINALIZE_TAIL_CUT = 3840
    _PRE_LA_CAP = 1024

    def generate_stream(
        self,
        text: str,
        prompt_wav: str,
        prompt_text: str = "",
        *,
        token_hop_len: int = 25,
        pre_lookahead_len: int = 3,
        v3_align: bool = False,
        fade_ms: float = 0.0,
        sample_rate: int = 24000,
    ):
        """Generator: yields (chunk_wav_np, is_first, is_final) per 25-token chunk.

        Implements intra-sentence token-level streaming on fixed-length HMONNX
        graphs using a sliding-window approach: each chunk feeds the accumulated
        token prefix to token2wav and extracts the newly generated audio segment.

        Args:
            text: Text to synthesize (single sentence recommended).
            prompt_wav: Path to prompt wav file (16kHz).
            prompt_text: Prompt text transcription.
            token_hop_len: Tokens per chunk (default 25).
            pre_lookahead_len: Pre-lookahead tokens for flow decoder (default 3).
            v3_align: Cut hift tail 3840 samples on intermediate chunks.
            fade_ms: Linear fade-in/out per chunk boundary (ms).
            sample_rate: Output sample rate (24000).
        """
        prompt_text_full = "You are a helpful assistant.<|endofprompt|>" + prompt_text
        prompt_text_full = self._text_normalize(prompt_text_full, split=False)

        for sent in self._text_normalize(text, split=True):
            if not isinstance(sent, str) or not sent.strip():
                continue
            yield from self._stream_one_sentence(
                sent,
                prompt_text_full,
                prompt_wav,
                token_hop_len=token_hop_len,
                pre_lookahead_len=pre_lookahead_len,
                v3_align=v3_align,
                fade_ms=fade_ms,
                sample_rate=sample_rate,
            )

    def _stream_one_sentence(
        self,
        sentence: str,
        prompt_text_full: str,
        prompt_wav: str,
        *,
        token_hop_len: int,
        pre_lookahead_len: int,
        v3_align: bool,
        fade_ms: float,
        sample_rate: int,
    ):
        device = self.device
        model_input = self._frontend_zero_shot(sentence, prompt_text_full, prompt_wav)
        prompt_token = model_input["flow_prompt_speech_token"].to(device)
        prompt_feat = model_input["prompt_speech_feat"].to(device)
        embedding = model_input["flow_embedding"].to(device)

        prompt_token_len = prompt_token.shape[1]
        if prompt_token_len % token_hop_len == 0:
            prompt_token_pad = 0
        else:
            prompt_token_pad = (
                (prompt_token_len + token_hop_len - 1) // token_hop_len
            ) * token_hop_len - prompt_token_len

        onnx_token_cap = self._PRE_LA_CAP - pre_lookahead_len - 4

        speech_tokens: List[int] = []
        token_offset = 0
        speech_offset = 0
        is_first_yield = True

        def _emit(tok_offset_curr: int, finalize: bool):
            nonlocal speech_offset, is_first_yield
            cur_tokens = torch.tensor(speech_tokens[:tok_offset_curr], dtype=torch.long).unsqueeze(0).to(device)
            wav_full = self._token2wav(
                token=cur_tokens,
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=embedding,
            )
            wav_full = wav_full.to(torch.float32).to("cpu")
            if v3_align and not finalize:
                if wav_full.shape[1] > self.HIFT_FINALIZE_TAIL_CUT:
                    wav_full = wav_full[:, : -self.HIFT_FINALIZE_TAIL_CUT]
            if wav_full.shape[1] <= speech_offset:
                return None, is_first_yield, finalize
            new_wav = wav_full[:, speech_offset:]
            speech_offset = wav_full.shape[1]
            is_f = is_first_yield
            is_first_yield = False
            return new_wav, is_f, finalize

        for tok in self._llm_token_generator(model_input):
            speech_tokens.append(tok)
            if prompt_token_len + len(speech_tokens) >= onnx_token_cap:
                break
            this_hop = token_hop_len + prompt_token_pad if token_offset == 0 else token_hop_len
            if len(speech_tokens) - token_offset >= this_hop + pre_lookahead_len:
                new_offset = token_offset + this_hop
                new_wav, is_f, _ = _emit(new_offset + pre_lookahead_len, finalize=False)
                token_offset = new_offset
                if new_wav is not None and new_wav.shape[1] > 0:
                    if fade_ms > 0:
                        fade_n = int(fade_ms / 1000.0 * sample_rate)
                        if fade_n > 0 and new_wav.shape[1] > 2 * fade_n:
                            ramp_in = torch.linspace(0, 1, fade_n).unsqueeze(0)
                            ramp_out = torch.linspace(1, 0, fade_n).unsqueeze(0)
                            new_wav[:, :fade_n] *= ramp_in
                            new_wav[:, -fade_n:] *= ramp_out
                    yield new_wav.numpy().astype("float32"), is_f, False

        if len(speech_tokens) > token_offset:
            new_wav, is_f, _ = _emit(len(speech_tokens), finalize=True)
            if new_wav is not None and new_wav.shape[1] > 0:
                if fade_ms > 0:
                    fade_n = int(fade_ms / 1000.0 * sample_rate)
                    if fade_n > 0 and new_wav.shape[1] > 2 * fade_n:
                        ramp_in = torch.linspace(0, 1, fade_n).unsqueeze(0)
                        ramp_out = torch.linspace(1, 0, fade_n).unsqueeze(0)
                        new_wav[:, :fade_n] *= ramp_in
                        new_wav[:, -fade_n:] *= ramp_out
                yield new_wav.numpy().astype("float32"), is_f, True
