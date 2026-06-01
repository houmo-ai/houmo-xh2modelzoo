# Copyright 2025 HOUMO AI
#
# File: sensevoice_hmonnx_demo.py
# Description:
#   Example script: audio/sensevoice_small/sensevoice_hmonnx_demo.py
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


import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
import yaml
import librosa
from xhquant.api import HMONNXInference

# --- Constants ---
LANGUAGE_MAP: Dict[str, int] = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
TEXTNORM_MAP: Dict[str, int] = {"withitn": 14, "woitn": 15}

# --- Frontend Logic (from sensevoice_frontend.py) ---

def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def _load_cmvn(cmvn_file: Path) -> Any:
    lines = cmvn_file.read_text(encoding="utf-8").splitlines()
    means_list = []
    vars_list = []
    for i in range(len(lines)):
        line_item = lines[i].split()
        if not line_item:
            continue
        if line_item[0] == "<AddShift>":
            line_item = lines[i + 1].split()
            if line_item and line_item[0] == "<LearnRateCoef>":
                means_list = list(line_item[3 : len(line_item) - 1])
        elif line_item[0] == "<Rescale>":
            line_item = lines[i + 1].split()
            if line_item and line_item[0] == "<LearnRateCoef>":
                vars_list = list(line_item[3 : len(line_item) - 1])

    means = np.array(means_list, dtype=np.float64)
    vars_ = np.array(vars_list, dtype=np.float64)
    if means.size == 0 or vars_.size == 0:
        raise ValueError(f"failed to parse cmvn file: {cmvn_file}")
    return np.stack([means, vars_], axis=0)

def _apply_cmvn(feat: Any, cmvn: Any) -> Any:
    frame, dim = feat.shape
    means = np.tile(cmvn[0:1, :dim], (frame, 1))
    vars_ = np.tile(cmvn[1:2, :dim], (frame, 1))
    return (feat + means) * vars_

def _apply_lfr(inputs: Any, lfr_m: int, lfr_n: int) -> Any:
    if lfr_m == 1 and lfr_n == 1:
        return inputs.astype(np.float32)
    lfr_inputs = []
    t = inputs.shape[0]
    t_lfr = int(np.ceil(t / lfr_n))
    left_padding = np.tile(inputs[0], ((lfr_m - 1) // 2, 1))
    inputs = np.vstack((left_padding, inputs))
    t = t + (lfr_m - 1) // 2
    for i in range(t_lfr):
        if lfr_m <= t - i * lfr_n:
            lfr_inputs.append((inputs[i * lfr_n : i * lfr_n + lfr_m]).reshape(1, -1))
        else:
            num_padding = lfr_m - (t - i * lfr_n)
            frame = inputs[i * lfr_n :].reshape(-1)
            for _ in range(num_padding):
                frame = np.hstack((frame, inputs[-1]))
            lfr_inputs.append(frame)
    return np.vstack(lfr_inputs).astype(np.float32)

@dataclass(frozen=True)
class FrontendConfig:
    fs: int = 16000
    window: str = "hamming"
    n_mels: int = 80
    frame_length: int = 25
    frame_shift: int = 10
    lfr_m: int = 7
    lfr_n: int = 6
    dither: float = 1.0
    cmvn_file: str = ""

class SenseVoiceFrontend:
    def __init__(self, cfg: FrontendConfig):
        self.cfg = cfg
        self.cmvn = _load_cmvn(Path(cfg.cmvn_file)) if cfg.cmvn_file else None

    @classmethod
    def from_model_dir(cls, model_dir: Union[str, Path]) -> "SenseVoiceFrontend":
        model_dir = Path(model_dir).expanduser().resolve()
        cfg_path = model_dir / "config.yaml"
        if not cfg_path.exists():
             print(f"Warning: {cfg_path} not found. Using default config but CMVN might fail.")
             cfg_obj = FrontendConfig(cmvn_file=str(model_dir / "am.mvn"))
        else:
            cfg = _read_yaml(cfg_path)
            frontend_conf = dict(cfg.get("frontend_conf") or {})
            cfg_obj = FrontendConfig(
                fs=int(frontend_conf.get("fs", 16000)),
                window=str(frontend_conf.get("window", "hamming")),
                n_mels=int(frontend_conf.get("n_mels", 80)),
                frame_length=int(frontend_conf.get("frame_length", 25)),
                frame_shift=int(frontend_conf.get("frame_shift", 10)),
                lfr_m=int(frontend_conf.get("lfr_m", 7)),
                lfr_n=int(frontend_conf.get("lfr_n", 6)),
                dither=float(frontend_conf.get("dither", 1.0)),
                cmvn_file=str(model_dir / "am.mvn"),
            )
        return cls(cfg_obj)

    def fbank(self, waveform: Any) -> Tuple[Any, int]:
        wav = torch.as_tensor(waveform, dtype=torch.float32)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.shape[0] != 1:
            wav = wav.mean(dim=0, keepdim=True)

        feat = torchaudio.compliance.kaldi.fbank(
            wav * (1 << 15),
            num_mel_bins=self.cfg.n_mels,
            sample_frequency=self.cfg.fs,
            frame_length=float(self.cfg.frame_length),
            frame_shift=float(self.cfg.frame_shift),
            dither=float(self.cfg.dither),
            window_type=self.cfg.window,
            snip_edges=True,
            energy_floor=0.0,
            use_energy=False,
        )
        feat = feat.cpu().numpy().astype("float32")
        return feat, int(feat.shape[0])

    def extract(self, waveform: Any) -> Tuple[Any, int]:
        feat, feat_len = self.fbank(waveform)
        feat = _apply_lfr(feat, self.cfg.lfr_m, self.cfg.lfr_n)
        if self.cmvn is not None:
            feat = _apply_cmvn(feat, self.cmvn).astype("float32")
        return feat, int(feat.shape[0])

def resolve_tag(v: str, mapping: Dict[str, int]) -> int:
    if v.isdigit():
        return int(v)
    key = v.lower().strip()
    if key not in mapping:
        raise ValueError(f"unsupported value: {v}; supported: {sorted(mapping.keys())}")
    return int(mapping[key])

def load_tokens(tokens_path: Path) -> Optional[List[str]]:
    if not tokens_path.exists():
        return None
    obj = json.loads(tokens_path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        return None
    return [str(x) for x in obj]

def decode_token_ids(token_ids: List[int], token_list: Optional[List[str]]) -> str:
    if not token_list:
        return " ".join(str(x) for x in token_ids)
    toks = [token_list[i] if 0 <= i < len(token_list) else "" for i in token_ids]
    s = "".join(toks)
    s = s.replace("▁", " ").strip()
    s = re.sub(r"\\s+", " ", s)
    return s

def ctc_greedy_decode(logits: Any, out_len: int, blank_id: int = 0) -> List[int]:
    x = torch.as_tensor(logits)
    x = x[:out_len]
    y = x.argmax(dim=-1)
    y = torch.unique_consecutive(y, dim=-1)
    y = y[y != blank_id]
    return [int(v) for v in y.cpu().tolist()]

def strip_rich_tags(s: str) -> str:
    return re.sub(r"<\|.*?\|>", "", s)

def make_inputs_for_sample(feat: Any, feat_len: int, language: str, textnorm: str) -> Dict[str, Any]:
    speech = feat[None, :, :].astype("float32")
    speech_lengths = np.array([feat_len], dtype="int32")
    lang = np.array([resolve_tag(language, LANGUAGE_MAP)], dtype="int32")
    norm = np.array([resolve_tag(textnorm, TEXTNORM_MAP)], dtype="int32")
    return {"speech": speech, "speech_lengths": speech_lengths, "language": lang, "textnorm": norm}

def load_audio(path: str, target_sr: int) -> Any:
    wav, _ = librosa.load(path, sr=target_sr, mono=True)
    return wav

def run_inference(hmonnx_path: Path, inputs: Dict[str, Any], device: str) -> Tuple[Any, Any]:
    sess = HMONNXInference(str(hmonnx_path))
    dev = torch.device(device)
    sess.to_fast_mode()
    sess.to(dev)
    
    input_info = {info.name: info for info in sess.inputs}
    feed: Dict[str, Any] = {}
    
    for k, v in inputs.items():
        t = torch.as_tensor(v)
        
        if k in input_info:
            target_info = input_info[k]
            target_dtype = target_info.dtype
            if t.dtype != target_dtype:
                t = t.to(dtype=target_dtype)
            
            target_shape = target_info.shape
            if k == "speech" and len(target_shape) == 3 and t.ndim == 3:
                 cur_t = t.shape[1]
                 tgt_t = target_shape[1]
                 if cur_t != tgt_t:
                     if cur_t < tgt_t:
                         t = torch.nn.functional.pad(t, (0, 0, 0, tgt_t - cur_t))
                     else:
                         t = t[:, :tgt_t, :]
        
        feed[k] = t.to(dev)
    
    outs = sess.run(feed)
    return outs[0], outs[1]

# --- Main ---

def _default_assets_dir() -> Path:
    candidates = [
        Path(__file__).parent / "model",
        Path(__file__).parent / "work_dirs/sensevoice_small/export_xh2a_libri_128_minmax_v2/model",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]

def main():
    default_assets_dir = _default_assets_dir()
    parser = argparse.ArgumentParser(description="SenseVoiceSmall HMONNX Demo")
    parser.add_argument("audio_files", nargs="+", help="Audio files to transcribe")
    parser.add_argument("--hmonnx", type=str, required=True, help="Path to HMONNX model")
    parser.add_argument(
        "--assets-dir",
        type=str,
        default=str(default_assets_dir),
        help="Dir with config.yaml/am.mvn/tokens.json",
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Inference device")
    parser.add_argument("--language", type=str, default="auto", choices=LANGUAGE_MAP.keys())
    parser.add_argument("--textnorm", type=str, default="woitn", choices=TEXTNORM_MAP.keys())
    
    args = parser.parse_args()
    
    assets_dir = Path(args.assets_dir).expanduser().resolve()
    hmonnx_path = Path(args.hmonnx).expanduser().resolve()
    
    if not hmonnx_path.exists():
        print(f"Error: HMONNX model not found: {hmonnx_path}")
        sys.exit(1)
        
    print(f"Loading frontend assets from {assets_dir}")
    frontend = SenseVoiceFrontend.from_model_dir(assets_dir)
    target_sr = frontend.cfg.fs
    
    tokens_path = assets_dir / "tokens.json"
    print(f"Loading tokens from {tokens_path}")
    token_list = load_tokens(tokens_path)
    if token_list is None:
        print("Warning: Failed to load tokens. Output will be token IDs.")

    print(f"Running inference on device: {args.device}")
    
    for audio_file in args.audio_files:
        # Handle potential Windows-style paths
        audio_file = audio_file.replace("\\", "/")
        audio_path = Path(audio_file).expanduser().resolve()
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_path}")
            continue
            
        print(f"\nProcessing: {audio_path}")
        try:
            # 1. Load Audio
            wav = load_audio(str(audio_path), target_sr)
            
            # 2. Extract Features
            feat, feat_len = frontend.extract(wav)
            
            # 3. Prepare Inputs
            inputs = make_inputs_for_sample(feat, feat_len, args.language, args.textnorm)
            
            # 4. Run Inference
            logits, lens = run_inference(hmonnx_path, inputs, args.device)
            
            # 5. Decode
            out_len = int(lens[0]) if hasattr(lens, "__len__") else int(lens)
            token_ids = ctc_greedy_decode(logits[0], out_len)
            text = decode_token_ids(token_ids, token_list)
            
            # 6. Post-process (strip tags)
            clean_text = strip_rich_tags(text)
            
            print(f"Raw Output: {text}")
            print(f"Clean Text: {clean_text}")
            
        except Exception as e:
            print(f"Failed to process {audio_path}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
