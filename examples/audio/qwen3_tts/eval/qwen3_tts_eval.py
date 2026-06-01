#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-TTS 精度测试脚本

支持 Native PyTorch 和 HMONNX 两种推理模式。
使用 CV3-Eval 的 zero_shot/zh 数据集生成音频文件供精度评估。

使用方法:
    # Native 模式，测试前 20 条
    PYTHONPATH=/path/to/xh2modelzoo python qwen3_tts_eval.py \
        --mode native --gpus 0,1,2,3 --max-samples 20

    # HMONNX 模式，全部 500 条
    PYTHONPATH=/path/to/xh2modelzoo python qwen3_tts_eval.py \
        --mode hmonnx --gpus 0,1,2,3,4,5,6,7
"""

import os
import sys
import logging
import argparse
import random
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
    XHQwen3TTSModel,
    Qwen3TTSHMONNXInference
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)

# 设置多进程启动方式
try:
    set_start_method('spawn')
except RuntimeError:
    pass

# 预定义 speaker 列表
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


def load_text_data(data_path: str, max_samples: int = None) -> Dict[str, str]:
    """
    加载 CV3-Eval 的 text 文件

    Args:
        data_path: CV3-Eval 数据集路径
        max_samples: 最大样本数，None 表示全部

    Returns:
        {uttid: text} 字典
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


def select_speaker(utt: str, mode: str, fixed_speaker: str) -> str:
    """
    根据模式选择 speaker

    Args:
        utt: uttid (如 "uttid_1")
        mode: "fixed", "random", "round-robin"
        fixed_speaker: 固定模式使用的 speaker

    Returns:
        speaker 名称
    """
    if mode == "fixed":
        return fixed_speaker
    elif mode == "random":
        return random.choice(SPEAKERS)
    elif mode == "round-robin":
        # 从 uttid_1 提取数字 1
        try:
            uttid_index = int(utt.split('_')[1])
        except (IndexError, ValueError):
            logging.warning(f"Cannot parse uttid '{utt}' for round-robin, using index 0")
            uttid_index = 1
        return SPEAKERS[(uttid_index - 1) % len(SPEAKERS)]
    else:
        raise ValueError(f"Unknown speaker mode: {mode}")


def worker_native(rank: int, gpus_to_use: List[int], args: argparse.Namespace,
                  text_dict: Dict[str, str], output_dir: str):
    """
    Native PyTorch 推理 worker

    Args:
        rank: worker 编号
        gpus_to_use: GPU 列表
        args: 命令行参数
        text_dict: {uttid: text} 字典
        output_dir: 输出目录
    """
    gpu_id = gpus_to_use[rank]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    # 设置随机种子
    torch.manual_seed(1024 + rank)
    torch.cuda.manual_seed(1024 + rank)
    torch.cuda.manual_seed_all(1024 + rank)

    logging.info(f"Worker {rank} (GPU {gpu_id}): Loading Native model...")

    # 加载模型
    model = XHQwen3TTSModel.from_pretrained(
        args.hf_model,
        device_map=device,
        dtype=torch.float16,
        attn_implementation="sdpa"
    )

    logging.info(f"Worker {rank} (GPU {gpu_id}): Model loaded")
    os.makedirs(output_dir, exist_ok=True)

    # 处理数据分片
    utts = list(text_dict.keys())
    for idx in tqdm(range(rank, len(utts), len(gpus_to_use)),
                    desc=f"GPU {gpu_id} Native"):
        utt = utts[idx]
        text = text_dict[utt]

        # 检查输出文件是否已存在
        wav_path_out = os.path.join(output_dir, f"{utt}.wav")
        if os.path.exists(wav_path_out):
            logging.debug(f"{wav_path_out} 已存在，跳过生成")
            continue

        # 选择 speaker
        speaker = select_speaker(utt, args.speaker_mode, args.speaker)

        # 生成音频
        try:
            wavs, sr = model.generate_custom_voice(
                text=text,
                language="Chinese",
                speaker=speaker,
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

            # 保存
            # 确保 wav tensor 是 2D (channels, samples)
            wav = wavs[0]
            # 处理 numpy array 或 torch tensor
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # 添加 channel 维度
            torchaudio.save(wav_path_out, wav, sr)
            logging.info(f"✓ 成功生成 {utt} (speaker={speaker})")

        except Exception as e:
            import traceback
            logging.error(f"✗ 生成 {utt} 失败: {str(e)}")
            logging.error(f"Traceback:\n{traceback.format_exc()}")
            continue

    # 清理 GPU 内存
    del model
    torch.cuda.empty_cache()
    logging.info(f"Worker {rank} (GPU {gpu_id}): Finished and cleaned up")


def worker_hmonnx(rank: int, gpus_to_use: List[int], args: argparse.Namespace,
                  text_dict: Dict[str, str], output_dir: str):
    """
    HMONNX 推理 worker

    Args:
        rank: worker 编号
        gpus_to_use: GPU 列表
        args: 命令行参数
        text_dict: {uttid: text} 字典
        output_dir: 输出目录
    """
    gpu_id = gpus_to_use[rank]
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')

    # 设置随机种子
    torch.manual_seed(1024 + rank)
    torch.cuda.manual_seed(1024 + rank)
    torch.cuda.manual_seed_all(1024 + rank)

    # 初始化 logger（HMONNX 推理需要）
    import tempfile
    log_file = tempfile.mktemp(suffix=f"_worker_{rank}.log")
    xhquant_llm_init(log_file, debug=False)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Worker {rank} (GPU {gpu_id}): Loading HMONNX model...")

    # 加载配置和模型
    cfg = Config.fromfile(args.hmonnx_config)
    model = MODELS.build(cfg.model)
    assert isinstance(model, Qwen3TTSHMONNXInference)
    model.to(device)

    logging.info(f"Worker {rank} (GPU {gpu_id}): Model loaded")

    # 处理数据分片
    utts = list(text_dict.keys())
    for idx in tqdm(range(rank, len(utts), len(gpus_to_use)),
                    desc=f"GPU {gpu_id} HMONNX"):
        utt = utts[idx]
        text = text_dict[utt]

        # 检查输出文件是否已存在
        wav_path_out = os.path.join(output_dir, f"{utt}.wav")
        if os.path.exists(wav_path_out):
            logging.debug(f"{wav_path_out} 已存在，跳过生成")
            continue

        # 选择 speaker
        speaker = select_speaker(utt, args.speaker_mode, args.speaker)

        # 生成音频
        try:
            wavs, sr = model.generate_custom_voice(
                text=text,
                language="Chinese",
                speaker=speaker,
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

            # 保存
            # 确保 wav tensor 是 2D (channels, samples)
            wav = wavs[0]
            # 处理 numpy array 或 torch tensor
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)  # 添加 channel 维度
            torchaudio.save(wav_path_out, wav, sr)
            logging.info(f"✓ 成功生成 {utt} (speaker={speaker})")

        except Exception as e:
            import traceback
            logging.error(f"✗ 生成 {utt} 失败: {str(e)}")
            logging.error(f"Traceback:\n{traceback.format_exc()}")
            continue

    # 清理 GPU 内存
    del model
    torch.cuda.empty_cache()
    logging.info(f"Worker {rank} (GPU {gpu_id}): Finished and cleaned up")


def main(args: argparse.Namespace):
    """主函数"""
    # 加载数据
    logging.info(f"Loading data from {args.data_path}")
    text_dict = load_text_data(args.data_path, args.max_samples)

    # 解析 GPU 列表
    try:
        gpus_to_use = [int(g.strip()) for g in args.gpus.split(',') if g.strip()]
        if not gpus_to_use:
            raise ValueError("No valid GPU IDs provided")
        if any(g < 0 for g in gpus_to_use):
            raise ValueError("GPU IDs must be non-negative")
        # 验证 GPU 可用性
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

    # 确定输出目录
    if args.mode == "native":
        output_dir = os.path.join(args.exp_dir, "native_fp16")
    elif args.mode == "hmonnx":
        output_dir = os.path.join(args.exp_dir, "hmonnx")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")

    # Native 模式使用顺序处理（避免 multiprocessing 导致的 CUDA 采样错误）
    if args.mode == "native":
        logging.info(f"Running native mode sequentially on {world_size} GPU(s)...")
        for rank in range(world_size):
            worker_native(rank, gpus_to_use, args, text_dict, output_dir)
    else:
        # HMONNX 模式使用多进程
        logging.info(f"Starting {world_size} workers in hmonnx mode...")
        spawn(
            worker_hmonnx,
            args=(gpus_to_use, args, text_dict, output_dir),
            nprocs=world_size,
            join=True
        )

    logging.info("All workers finished!")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS 精度测试脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # 必需参数
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["native", "hmonnx"],
        help="推理模式"
    )
    parser.add_argument(
        "--gpus",
        type=str,
        required=True,
        help="GPU 列表，逗号分隔，如 '0,1,2,3'"
    )

    # 数据相关
    parser.add_argument(
        "--data-path",
        type=str,
        default="/data01/home/she.gao/CV3-Eval/data/zero_shot/zh",
        help="CV3-Eval 数据集路径"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="最大处理样本数，None 表示全部"
    )

    # 输出相关
    parser.add_argument(
        "--exp-dir",
        type=str,
        default="qwen3tts_eval_zh",
        help="输出根目录"
    )

    # Speaker 相关
    parser.add_argument(
        "--speaker",
        type=str,
        default="vivian",
        choices=SPEAKERS,
        help="固定 speaker 名称（仅在 --speaker-mode fixed 时有效）"
    )
    parser.add_argument(
        "--speaker-mode",
        type=str,
        default="fixed",
        choices=["fixed", "random", "round-robin"],
        help="Speaker 选择策略"
    )

    # 模型路径
    parser.add_argument(
        "--hf-model",
        type=str,
        default="./data/models/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        help="Native 模式的 HuggingFace 模型路径"
    )
    parser.add_argument(
        "--hmonnx-config",
        type=str,
        default="./config/llm/qwen3_tts_12hz_0_6B_customvoice_xh2a_hmonnx.py",
        help="HMONNX 模式的配置文件路径"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
