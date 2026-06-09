#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-TTS voice-clone accuracy test script

Supports Native PyTorch and HMONNX inference modes.
Runs voice-clone over the CV3-Eval zero_shot/zh dataset using reference audio.

Native mode uses the original Qwen3TTSModel (bf16 + sdpa) to avoid fp16 numeric issues.
HMONNX mode uses the quantized model for stable, efficient inference.

Usage:
    # native mode, first 20 samples
    PYTHONPATH=/path/to/xh2modelzoo python qwen3_tts_eval_voice_clone.py \
        --mode native --gpus 0,1,2,3 --max-samples 20

    # hmonnx mode, all 500 samples
    PYTHONPATH=/path/to/xh2modelzoo python qwen3_tts_eval_voice_clone.py \
        --mode hmonnx --gpus 0,1,2,3,4,5,6,7
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from torch.multiprocessing import spawn, set_start_method

from xhquant.api import Config
from xh_model_zoo.api import xhquant_llm_init
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.qwen3_tts import (
    Qwen3TTSHMONNXInference
)
# use the original Qwen3TTSModel, not the XH wrapper
from qwen_tts import Qwen3TTSModel

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


def load_voice_clone_data(data_path: str, max_samples: int = None) -> List[Dict[str, str]]:
    """
    Load CV3-Eval voice-clone data

    Args:
        data_path: CV3-Eval dataset path
        max_samples: max number of samples, None for all

    Returns:
        list of {uttid, text, ref_text, ref_audio}
    """
    text_file = os.path.join(data_path, "text")
    prompt_text_file = os.path.join(data_path, "prompt_text")
    prompt_wav_file = os.path.join(data_path, "prompt_wav.scp")

    # check files exist
    for f in [text_file, prompt_text_file, prompt_wav_file]:
        if not os.path.exists(f):
            raise FileNotFoundError(f"Required file not found: {f}")

    # load target text
    text_dict = {}
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                logging.warning(f"Invalid text line: {line.strip()}")
                continue
            utt, text = parts
            text_dict[utt] = text

    # load reference text
    prompt_text_dict = {}
    with open(prompt_text_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                logging.warning(f"Invalid prompt_text line: {line.strip()}")
                continue
            utt, ref_text = parts
            prompt_text_dict[utt] = ref_text

    # load reference audio paths
    prompt_wav_dict = {}
    with open(prompt_wav_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(maxsplit=1)
            if len(parts) != 2:
                logging.warning(f"Invalid prompt_wav line: {line.strip()}")
                continue
            utt, ref_wav_path = parts
            # convert relative path to absolute
            if not os.path.isabs(ref_wav_path):
                # paths in prompt_wav.scp are relative to the CV3-Eval root
                cv3_eval_root = Path(data_path).parent.parent.parent
                ref_wav_path = os.path.join(cv3_eval_root, ref_wav_path)
            prompt_wav_dict[utt] = ref_wav_path

    # assemble samples
    data_list = []
    for utt in text_dict.keys():
        if utt not in prompt_text_dict:
            logging.warning(f"Missing prompt_text for {utt}, skipping")
            continue
        if utt not in prompt_wav_dict:
            logging.warning(f"Missing prompt_wav for {utt}, skipping")
            continue

        ref_audio_path = prompt_wav_dict[utt]
        if not os.path.exists(ref_audio_path):
            logging.warning(f"Reference audio not found: {ref_audio_path}, skipping {utt}")
            continue

        data_list.append({
            "uttid": utt,
            "text": text_dict[utt],
            "ref_text": prompt_text_dict[utt],
            "ref_audio": ref_audio_path
        })

        if max_samples and len(data_list) >= max_samples:
            break

    logging.info(f"Loaded {len(data_list)} valid voice clone samples from {data_path}")
    return data_list


def worker_native(rank: int, gpus_to_use: List[int], args: argparse.Namespace,
                  data_list: List[Dict[str, str]], output_dir: str):
    """
    Native PyTorch inference worker (voice clone)

    Args:
        rank: worker index
        gpus_to_use: list of GPU ids
        args: command-line args
        data_list: list of voice-clone samples
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

    # process this rank shard
    for idx in tqdm(range(rank, len(data_list), len(gpus_to_use)),
                    desc=f"GPU {gpu_id} Native Voice Clone"):
        sample = data_list[idx]
        uttid = sample["uttid"]
        text = sample["text"]
        ref_text = sample["ref_text"]
        ref_audio = sample["ref_audio"]

        # skip if output already exists
        wav_path_out = os.path.join(output_dir, f"{uttid}.wav")
        if os.path.exists(wav_path_out):
            logging.debug(f"{wav_path_out} already exists, skipping")
            continue

        # generate audio (voice clone)
        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="Chinese",
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=args.xvec_only,
                max_new_tokens=2048,
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                repetition_penalty=1.05,
                subtalker_dosample=True,
                subtalker_top_k=50,
                subtalker_top_p=1.0,
                subtalker_temperature=0.9
            )

            # save
            # ensure wav tensor is 2D (channels, samples)
            wav = wavs[0]
            # handle numpy array or torch tensor
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # add channel dim
            torchaudio.save(wav_path_out, wav, sr)
            logging.info(f"✓ generated {uttid}")

        except Exception as e:
            import traceback
            logging.error(f"✗ failed to generate {uttid}: {str(e)}")
            logging.error(f"Traceback:\n{traceback.format_exc()}")
            continue

    # free GPU memory
    del model
    torch.cuda.empty_cache()
    logging.info(f"Worker {rank} (GPU {gpu_id}): Finished and cleaned up")


def worker_hmonnx(rank: int, gpus_to_use: List[int], args: argparse.Namespace,
                  data_list: List[Dict[str, str]], output_dir: str):
    """
    HMONNX inference worker (voice clone)

    Args:
        rank: worker index
        gpus_to_use: list of GPU ids
        args: command-line args
        data_list: list of voice-clone samples
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

    logging.info(f"Worker {rank} (GPU {gpu_id}): Model loaded")

    # process this rank shard
    for idx in tqdm(range(rank, len(data_list), len(gpus_to_use)),
                    desc=f"GPU {gpu_id} HMONNX Voice Clone"):
        sample = data_list[idx]
        uttid = sample["uttid"]
        text = sample["text"]
        ref_text = sample["ref_text"]
        ref_audio = sample["ref_audio"]

        # skip if output already exists
        wav_path_out = os.path.join(output_dir, f"{uttid}.wav")
        if os.path.exists(wav_path_out):
            logging.debug(f"{wav_path_out} already exists, skipping")
            continue

        # generate audio (voice clone)
        try:
            wavs, sr = model.generate_voice_clone(
                text=text,
                language="Chinese",
                ref_audio=ref_audio,
                ref_text=ref_text,
                x_vector_only_mode=args.xvec_only,
                max_new_tokens=2048,
                do_sample=True,
                top_k=50,
                top_p=1.0,
                temperature=0.9,
                repetition_penalty=1.05,
                subtalker_dosample=True,
                subtalker_top_k=50,
                subtalker_top_p=1.0,
                subtalker_temperature=0.9
            )

            # save
            # ensure wav tensor is 2D (channels, samples)
            wav = wavs[0]
            # handle numpy array or torch tensor
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # add channel dim
            torchaudio.save(wav_path_out, wav, sr)
            logging.info(f"✓ generated {uttid}")

        except Exception as e:
            import traceback
            logging.error(f"✗ failed to generate {uttid}: {str(e)}")
            logging.error(f"Traceback:\n{traceback.format_exc()}")
            continue

    # free GPU memory
    del model
    torch.cuda.empty_cache()
    logging.info(f"Worker {rank} (GPU {gpu_id}): Finished and cleaned up")


def main(args: argparse.Namespace):
    """Main."""
    # load data
    logging.info(f"Loading voice clone data from {args.data_path}")
    data_list = load_voice_clone_data(args.data_path, args.max_samples)

    if not data_list:
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
        output_dir = os.path.join(args.exp_dir, "native_voice_clone")
    elif args.mode == "hmonnx":
        output_dir = os.path.join(args.exp_dir, "hmonnx_voice_clone")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # native mode runs sequentially (avoids CUDA sampling errors from multiprocessing)
    if args.mode == "native":
        logging.info(f"Running native voice clone mode sequentially on {world_size} GPU(s)...")
        for rank in range(world_size):
            worker_native(rank, gpus_to_use, args, data_list, output_dir)
    else:
        # hmonnx mode uses multiprocessing
        logging.info(f"Starting {world_size} workers in hmonnx voice clone mode...")
        spawn(
            worker_hmonnx,
            args=(gpus_to_use, args, data_list, output_dir),
            nprocs=world_size,
            join=True
        )

    logging.info("All workers finished!")


def parse_arguments():
    """Parse command-line args."""
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS voice-clone accuracy test script",
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
        default="qwen3tts_eval_zh_voice_clone",
        help="output root directory"
    )

    # voice-clone args
    parser.add_argument(
        "--xvec-only",
        action="store_true",
        help="x-vector only mode (ignore fine-grained features of the reference audio)"
    )

    # model paths
    parser.add_argument(
        "--hf-model",
        type=str,
        default="./data/models/Qwen3-TTS-12Hz-0.6B-Base",
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

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
