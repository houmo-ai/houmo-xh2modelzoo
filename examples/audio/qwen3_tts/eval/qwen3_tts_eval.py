#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-TTS accuracy test script

Supports Native PyTorch and HMONNX inference modes.
Generates audio over the CV3-Eval zero_shot/zh dataset for accuracy evaluation.

Native mode uses the original Qwen3TTSModel (bf16 + sdpa) to avoid fp16 numeric issues.
HMONNX mode uses the quantized model for stable, efficient inference.

Usage:
    # native mode, first 20 samples
    PYTHONPATH=/path/to/xh2modelzoo python qwen3_tts_eval.py \
        --mode native --gpus 0,1,2,3 --max-samples 20

    # hmonnx mode, all 500 samples
    PYTHONPATH=/path/to/xh2modelzoo python qwen3_tts_eval.py \
        --mode hmonnx --gpus 0,1,2,3,4,5,6,7
"""

import os
import sys
import logging
import argparse
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from torch.multiprocessing import spawn, set_start_method

QWEN3_TTS_DIR = Path(__file__).resolve().parents[1]
if str(QWEN3_TTS_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN3_TTS_DIR))

from xhquant.api import Config
from xh_model_zoo.api import xhquant_llm_init
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.qwen3_tts import (
    Qwen3TTSHMONNXInference
)
# use the original Qwen3TTSModel, not the XH wrapper
from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram

# configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# set multiprocessing start method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

# predefined speaker list
SPEAKERS = [
    "serena",
    "vivian",
    "uncle_fu",
    "ryan",
    "aiden",
    "ono_anna",
    "sohee",
    "eric",
    "dylan"
]

DEFAULT_CUSTOMVOICE_MODEL = "./data/models/Qwen3-TTS-12Hz-0.6B-CustomVoice"
DEFAULT_BASE_MODEL = "./data/models/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_FRONTEND_HMONNX_DIR = "./work_dirs/qwen3_tts_12hz_0_6B_base_frontend_xh2a"


def load_text_data(data_path: str, max_samples: int = None) -> Dict[str, str]:
    """
    Load the CV3-Eval text file

    Args:
        data_path: CV3-Eval dataset path
        max_samples: max number of samples, None for all

    Returns:
        dict {uttid: text}
    """
    text_file = os.path.join(data_path, "text")
    if not os.path.exists(text_file):
        raise FileNotFoundError(f"Text file not found: {text_file}")

    text_dict = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                logging.warning(f"Invalid line format: {line.strip()}")
                continue
            utt, text = parts
            text_dict[utt] = text

            if max_samples and len(text_dict) >= max_samples:
                break

    logging.info(f"Loaded {len(text_dict)} samples from {text_file}")
    return text_dict


def load_voice_clone_data(data_path: str, max_samples: int = None) -> List[Dict[str, str]]:
    """Load CV3-Eval voice-clone samples with target text, prompt text and prompt wav."""
    text_file = os.path.join(data_path, "text")
    prompt_text_file = os.path.join(data_path, "prompt_text")
    prompt_wav_file = os.path.join(data_path, "prompt_wav.scp")

    for file_path in [text_file, prompt_text_file, prompt_wav_file]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}")

    text_dict = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                text_dict[parts[0]] = parts[1]

    prompt_text_dict = {}
    with open(prompt_text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                prompt_text_dict[parts[0]] = parts[1]

    prompt_wav_dict = {}
    cv3_eval_root = Path(data_path).parent.parent.parent
    with open(prompt_wav_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                continue
            utt, wav_path = parts
            if not os.path.isabs(wav_path):
                wav_path = os.path.join(cv3_eval_root, wav_path)
            prompt_wav_dict[utt] = wav_path

    samples = []
    for utt, text in text_dict.items():
        if utt not in prompt_text_dict or utt not in prompt_wav_dict:
            logging.warning(f"Missing voice-clone prompt for {utt}, skipping")
            continue
        if not os.path.exists(prompt_wav_dict[utt]):
            logging.warning(f"Reference audio not found: {prompt_wav_dict[utt]}, skipping {utt}")
            continue
        samples.append(
            {
                "uttid": utt,
                "text": text,
                "ref_text": prompt_text_dict[utt],
                "ref_audio": prompt_wav_dict[utt],
            }
        )
        if max_samples and len(samples) >= max_samples:
            break

    logging.info(f"Loaded {len(samples)} voice-clone samples from {data_path}")
    return samples


def _resolve_frontend_hmonnx_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    frontend_dir = Path(args.frontend_hmonnx_dir)
    encode_hmonnx = Path(args.speech_tokenizer_encode_hmonnx) if args.speech_tokenizer_encode_hmonnx else (
        frontend_dir / "hmonnx" / "speech_tokenizer_encode_XH2a.onnx"
    )
    speaker_hmonnx = Path(args.speaker_encoder_hmonnx) if args.speaker_encoder_hmonnx else (
        frontend_dir / "hmonnx" / "speaker_encoder_XH2a.onnx"
    )
    if not encode_hmonnx.exists():
        raise FileNotFoundError(f"speech_tokenizer.encode HMONNX not found: {encode_hmonnx}")
    if not speaker_hmonnx.exists():
        raise FileNotFoundError(f"speaker_encoder HMONNX not found: {speaker_hmonnx}")
    return encode_hmonnx, speaker_hmonnx


def _pad_or_trim_1d(x: torch.Tensor, target_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_len = min(int(x.numel()), target_len)
    out = torch.zeros(target_len, dtype=torch.float32)
    mask = torch.zeros(target_len, dtype=torch.int32)
    if valid_len > 0:
        out[:valid_len] = x[:valid_len].to(torch.float32)
        mask[:valid_len] = 1
    return out.unsqueeze(0), mask.unsqueeze(0)


def _pad_or_trim_mels(mels: torch.Tensor, target_frames: int, target_dim: int) -> torch.Tensor:
    if mels.shape[-1] != target_dim:
        raise ValueError(f"speaker mels dim mismatch: expected {target_dim}, got {mels.shape[-1]}")
    out = torch.zeros((mels.shape[0], target_frames, target_dim), dtype=torch.float32)
    valid_frames = min(int(mels.shape[1]), target_frames)
    if valid_frames > 0:
        out[:, :valid_frames, :] = mels[:, :valid_frames, :].to(torch.float32)
    return out


class VoiceCloneFrontendHMONNX:
    """Run exported speech_tokenizer.encode and speaker_encoder HMONNX modules."""

    def __init__(self, args: argparse.Namespace, device: torch.device):
        from xhquant.xhonnxruntime.hmonnx_inference import HMONNXInference

        encode_hmonnx, speaker_hmonnx = _resolve_frontend_hmonnx_paths(args)
        logging.info(f"Loading speech_tokenizer.encode HMONNX: {encode_hmonnx}")
        self.encode_session = HMONNXInference(str(encode_hmonnx))
        self.encode_session.to(device)
        logging.info(f"Loading speaker_encoder HMONNX: {speaker_hmonnx}")
        self.speaker_session = HMONNXInference(str(speaker_hmonnx))
        self.speaker_session.to(device)
        self.device = device

        self.audio_samples = int(self.encode_session.inputs[0].shape[1])
        self.mel_frames = int(self.speaker_session.inputs[0].shape[1])
        self.mel_dim = int(self.speaker_session.inputs[0].shape[2])
        self.encode_input_dtype = self.encode_session.inputs[0].dtype
        self.speaker_input_dtype = self.speaker_session.inputs[0].dtype
        self.sample_rate = int(args.frontend_sample_rate)

    def _load_audio(self, wav_path: str) -> torch.Tensor:
        wav, sr = torchaudio.load(wav_path)
        wav = wav.to(torch.float32)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != self.sample_rate:
            wav = torchaudio.functional.resample(wav, int(sr), self.sample_rate)
        return wav.squeeze(0).contiguous()

    def _speaker_mels(self, wav: torch.Tensor) -> torch.Tensor:
        mels = mel_spectrogram(
            wav.unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=self.sample_rate,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000,
        ).transpose(1, 2)
        return _pad_or_trim_mels(mels, self.mel_frames, self.mel_dim)

    @torch.no_grad()
    def build_prompt(self, ref_audio: str, ref_text: str, xvec_only: bool) -> List[VoiceClonePromptItem]:
        wav = self._load_audio(ref_audio)
        if wav.numel() > self.audio_samples:
            logging.warning(
                f"Reference audio {ref_audio} has {wav.numel()} samples; "
                f"truncating to {self.audio_samples} for static speech_tokenizer.encode HMONNX"
            )

        input_values, padding_mask = _pad_or_trim_1d(wav, self.audio_samples)
        input_values = input_values.to(device=self.device, dtype=self.encode_input_dtype)
        padding_mask = padding_mask.to(device=self.device, dtype=torch.int32)
        encode_out = self.encode_session(input_values, padding_mask)
        audio_codes, valid_frames = encode_out if isinstance(encode_out, (tuple, list)) else (encode_out, None)
        if valid_frames is None:
            valid_len = audio_codes.shape[1]
        else:
            valid_len = int(valid_frames.detach().cpu().reshape(-1)[0])
        ref_code = None if xvec_only else audio_codes[0, :valid_len, :].detach().cpu().to(torch.long)

        mels = self._speaker_mels(wav).to(device=self.device, dtype=self.speaker_input_dtype)
        speaker_out = self.speaker_session(mels)
        if isinstance(speaker_out, (tuple, list)):
            speaker_out = speaker_out[0]
        ref_spk_embedding = speaker_out[0].detach().cpu().to(torch.float32)

        return [
            VoiceClonePromptItem(
                ref_code=ref_code,
                ref_spk_embedding=ref_spk_embedding,
                x_vector_only_mode=bool(xvec_only),
                icl_mode=bool(not xvec_only),
                ref_text=ref_text,
            )
        ]


def select_speaker(utt: str, mode: str, fixed_speaker: str) -> str:
    """
    Pick a speaker according to the mode

    Args:
        utt: uttid (e.g. "uttid_1")
        mode: "fixed", "random", "round-robin"
        fixed_speaker: speaker used in fixed mode

    Returns:
        speaker name
    """
    if mode == "fixed":
        return fixed_speaker
    elif mode == "random":
        return random.choice(SPEAKERS)
    elif mode == "round-robin":
        # extract the number from e.g. uttid_1
        try:
            uttid_index = int(utt.split('_')[1])
        except (IndexError, ValueError):
            logging.warning(f"Cannot parse uttid '{utt}' for round-robin, using index 0")
            uttid_index = 1
        return SPEAKERS[(uttid_index - 1) % len(SPEAKERS)]
    else:
        raise ValueError(f"Unknown speaker mode: {mode}")


def worker_native(rank: int, gpus_to_use: List[int], args: argparse.Namespace,
                  dataset: Any, output_dir: str):
    """
    Native PyTorch inference worker

    Args:
        rank: worker index
        gpus_to_use: list of GPU ids
        args: command-line args
        text_dict: dict {uttid: text}
        output_dir: output directory
    """
    gpu_id = gpus_to_use[rank]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    # set random seed
    torch.manual_seed(1024 + rank)
    torch.cuda.manual_seed(1024 + rank)
    torch.cuda.manual_seed_all(1024 + rank)

    logging.info(f"Worker {rank} (GPU {gpu_id}): Loading Native model...")

    # load the original Qwen3TTSModel (no XH wrapper)
    # use bfloat16 to avoid fp16 multinomial NaN
    model = Qwen3TTSModel.from_pretrained(
        args.hf_model,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa"
    )

    logging.info(f"Worker {rank} (GPU {gpu_id}): Model loaded")
    os.makedirs(output_dir, exist_ok=True)

    is_voice_clone = args.tts_mode == "voice_clone"
    total = len(dataset) if is_voice_clone else len(dataset.keys())
    utts = None if is_voice_clone else list(dataset.keys())
    for idx in tqdm(range(rank, total, len(gpus_to_use)),
                    desc=f"GPU {gpu_id} Native {args.tts_mode}"):
        if is_voice_clone:
            sample = dataset[idx]
            utt = sample["uttid"]
            text = sample["text"]
            ref_audio = sample["ref_audio"]
            ref_text = sample["ref_text"]
        else:
            utt = utts[idx]
            text = dataset[utt]
            ref_audio = None
            ref_text = None

        # skip if output already exists
        wav_path_out = os.path.join(output_dir, f"{utt}.wav")
        if os.path.exists(wav_path_out):
            logging.debug(f"{wav_path_out} already exists, skipping")
            continue

        # generate audio
        try:
            generate_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                repetition_penalty=1.05,
                subtalker_dosample=True,
                subtalker_top_k=50,
                subtalker_top_p=1.0,
                subtalker_temperature=0.9,
            )
            if is_voice_clone:
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language="Chinese",
                    ref_audio=ref_audio,
                    ref_text=ref_text,
                    x_vector_only_mode=args.xvec_only,
                    **generate_kwargs,
                )
                log_suffix = "voice_clone"
            else:
                speaker = select_speaker(utt, args.speaker_mode, args.speaker)
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language="Chinese",
                    speaker=speaker,
                    **generate_kwargs,
                )
                log_suffix = f"speaker={speaker}"

            # save
            # ensure wav tensor is 2D (channels, samples)
            wav = wavs[0]
            # handle numpy array or torch tensor
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # add channel dim
            torchaudio.save(wav_path_out, wav, sr)
            logging.info(f"✓ generated {utt} ({log_suffix})")

        except Exception as e:
            import traceback
            logging.error(f"✗ failed to generate {utt}: {str(e)}")
            logging.error(f"Traceback:\n{traceback.format_exc()}")
            continue

    # free GPU memory
    del model
    torch.cuda.empty_cache()
    logging.info(f"Worker {rank} (GPU {gpu_id}): Finished and cleaned up")


def worker_hmonnx(rank: int, gpus_to_use: List[int], args: argparse.Namespace,
                  dataset: Any, output_dir: str):
    """
    HMONNX inference worker

    Args:
        rank: worker index
        gpus_to_use: list of GPU ids
        args: command-line args
        text_dict: dict {uttid: text}
        output_dir: output directory
    """
    gpu_id = gpus_to_use[rank]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    # set random seed
    torch.manual_seed(1024 + rank)
    torch.cuda.manual_seed(1024 + rank)
    torch.cuda.manual_seed_all(1024 + rank)

    # init logger (required by HMONNX inference)
    import tempfile
    log_file = tempfile.mktemp(suffix=f"_worker_{rank}.log")
    xhquant_llm_init(log_file, debug=False)

    # create output dir
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Worker {rank} (GPU {gpu_id}): Loading HMONNX model...")

    # load config and model
    cfg = Config.fromfile(args.hmonnx_config)
    if getattr(args, "variant", None):
        from config.llm._components import apply_variant_hmonnx
        apply_variant_hmonnx(cfg, args.variant)
    model = MODELS.build(cfg.model)
    assert isinstance(model, Qwen3TTSHMONNXInference)
    model.to(device)
    voice_clone_frontend = VoiceCloneFrontendHMONNX(args, device) if args.tts_mode == "voice_clone" else None

    logging.info(f"Worker {rank} (GPU {gpu_id}): Model loaded")

    # process this rank shard
    is_voice_clone = args.tts_mode == "voice_clone"
    total = len(dataset) if is_voice_clone else len(dataset.keys())
    utts = None if is_voice_clone else list(dataset.keys())
    for idx in tqdm(range(rank, total, len(gpus_to_use)),
                    desc=f"GPU {gpu_id} HMONNX {args.tts_mode}"):
        if is_voice_clone:
            sample = dataset[idx]
            utt = sample["uttid"]
            text = sample["text"]
            ref_audio = sample["ref_audio"]
            ref_text = sample["ref_text"]
        else:
            utt = utts[idx]
            text = dataset[utt]
            ref_audio = None
            ref_text = None

        # skip if output already exists
        wav_path_out = os.path.join(output_dir, f"{utt}.wav")
        if os.path.exists(wav_path_out):
            logging.debug(f"{wav_path_out} already exists, skipping")
            continue

        # generate audio
        try:
            generate_kwargs = dict(
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                repetition_penalty=1.05,
                subtalker_dosample=True,
                subtalker_top_k=50,
                subtalker_top_p=1.0,
                subtalker_temperature=0.9,
            )
            if is_voice_clone:
                assert voice_clone_frontend is not None
                voice_clone_prompt = voice_clone_frontend.build_prompt(ref_audio, ref_text, args.xvec_only)
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language="Chinese",
                    ref_audio=None,
                    ref_text=ref_text,
                    voice_clone_prompt=voice_clone_prompt,
                    **generate_kwargs,
                )
                log_suffix = "voice_clone_hmonnx_frontend"
            else:
                speaker = select_speaker(utt, args.speaker_mode, args.speaker)
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language="Chinese",
                    speaker=speaker,
                    **generate_kwargs,
                )
                log_suffix = f"speaker={speaker}"

            # save
            # ensure wav tensor is 2D (channels, samples)
            wav = wavs[0]
            # handle numpy array or torch tensor
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # add channel dim
            torchaudio.save(wav_path_out, wav, sr)
            logging.info(f"✓ generated {utt} ({log_suffix})")

        except Exception as e:
            import traceback
            logging.error(f"✗ failed to generate {utt}: {str(e)}")
            logging.error(f"Traceback:\n{traceback.format_exc()}")
            continue

    # free GPU memory
    del model
    if voice_clone_frontend is not None:
        del voice_clone_frontend
    torch.cuda.empty_cache()
    logging.info(f"Worker {rank} (GPU {gpu_id}): Finished and cleaned up")


def main(args: argparse.Namespace):
    """Main."""
    args.tts_mode = "voice_clone" if args.variant == "0_6B_base" else "custom_voice"

    if args.tts_mode == "voice_clone":
        if args.hf_model == DEFAULT_CUSTOMVOICE_MODEL:
            logging.warning("voice-clone uses Base model; switching --hf-model to Qwen3-TTS-12Hz-0.6B-Base")
            args.hf_model = DEFAULT_BASE_MODEL

    # load data
    logging.info(f"Loading {args.tts_mode} data from {args.data_path}")
    if args.tts_mode == "voice_clone":
        dataset = load_voice_clone_data(args.data_path, args.max_samples)
    else:
        dataset = load_text_data(args.data_path, args.max_samples)

    if not dataset:
        logging.error("No valid samples loaded. Exiting.")
        sys.exit(1)

    # parse GPU list
    try:
        gpus_to_use = [int(g.strip()) for g in args.gpus.split(',') if g.strip()]
        if not gpus_to_use:
            raise ValueError("No valid GPU IDs provided")
        if any(g < 0 for g in gpus_to_use):
            raise ValueError("GPU IDs must be non-negative")
        # validate GPU availability
        if torch.cuda.is_available():
            available_gpus = torch.cuda.device_count()
            if any(g >= available_gpus for g in gpus_to_use):
                raise ValueError(f"GPU IDs must be < {available_gpus} (available GPUs on this system)")
        else:
            raise RuntimeError("CUDA is not available on this system")
    except (ValueError, RuntimeError) as e:
        logging.error(f"Invalid GPU configuration '{args.gpus}': {e}")
        sys.exit(1)

    world_size = len(gpus_to_use)
    logging.info(f"Using {world_size} GPUs: {gpus_to_use}")

    # determine output dir
    if args.mode == "native":
        output_dir = os.path.join(args.exp_dir, "native_voice_clone" if args.tts_mode == "voice_clone" else "native_fp16")
    elif args.mode == "hmonnx":
        output_dir = os.path.join(args.exp_dir, "hmonnx_voice_clone" if args.tts_mode == "voice_clone" else "hmonnx")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # native mode runs sequentially (avoids CUDA sampling errors from multiprocessing)
    if args.mode == "native":
        logging.info(f"Running native mode sequentially on {world_size} GPU(s)...")
        for rank in range(world_size):
            worker_native(rank, gpus_to_use, args, dataset, output_dir)
    else:
        # hmonnx mode uses multiprocessing
        logging.info(f"Starting {world_size} workers in hmonnx mode...")
        spawn(
            worker_hmonnx,
            args=(gpus_to_use, args, dataset, output_dir),
            nprocs=world_size,
            join=True
        )

    logging.info("All workers finished!")


def parse_arguments():
    """Parse command-line args."""
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS accuracy test script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # required args
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["native", "hmonnx"],
        help="inference mode"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="comma-separated GPU list, e.g. '0,1,2,3'"
    )

    # data args
    parser.add_argument(
        "--data-path",
        type=str,
        default="/data01/home/she.gao/CV3-Eval/data/zero_shot/zh",
        help="CV3-Eval dataset path"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="max samples to process, None for all"
    )

    # output args
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="qwen3tts_eval_zh",
        help="output root directory"
    )

    # speaker args
    parser.add_argument(
        "--speaker",
        type=str,
        default="vivian",
        choices=SPEAKERS,
        help="fixed speaker name (only used when --speaker-mode fixed)"
    )
    parser.add_argument(
        "--speaker-mode",
        type=str,
        default="fixed",
        choices=["fixed", "random", "round-robin"],
        help="speaker selection strategy"
    )
    parser.add_argument(
        "--xvec-only",
        action="store_true",
        help="voice-clone x-vector only mode; ignore ref_code/ICL prompt"
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="max generated talker tokens"
    )

    # model paths
    parser.add_argument(
        "--hf-model",
        type=str,
        default=DEFAULT_CUSTOMVOICE_MODEL,
        help="HuggingFace model path for native mode"
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["0_6B_base", "0_6B_customvoice", "1_7B_voicedesign"],
        default=None,
        help="TTS variant; injects work_dirs paths into the unified hmonnx config"
    )
    parser.add_argument(
        "--hmonnx-config",
        type=str,
        default="./config/llm/qwen3_tts_12hz_xh2a_hmonnx.py",
        help="config file path for hmonnx mode"
    )
    parser.add_argument(
        "--frontend-hmonnx-dir",
        type=str,
        default=DEFAULT_FRONTEND_HMONNX_DIR,
        help="work dir containing exported voice-clone frontend HMONNX files"
    )
    parser.add_argument(
        "--speech-tokenizer-encode-hmonnx",
        type=str,
        default=None,
        help="explicit speech_tokenizer.encode HMONNX path; overrides --frontend-hmonnx-dir"
    )
    parser.add_argument(
        "--speaker-encoder-hmonnx",
        type=str,
        default=None,
        help="explicit speaker_encoder HMONNX path; overrides --frontend-hmonnx-dir"
    )
    parser.add_argument(
        "--frontend-sample-rate",
        type=int,
        default=24000,
        help="sample rate expected by exported voice-clone frontend modules"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
