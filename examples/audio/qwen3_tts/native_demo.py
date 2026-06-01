#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-TTS Native PyTorch 推理示例

支持两种模型加载方式：
1. XH Wrapped Model（默认）：使用 XHQwen3TTSModel，经过 XH 优化
2. Original Model：使用原始 Qwen3TTSModel（需要 --use-original 参数）

支持三种推理模式：
1. voice-clone: 基于参考音频克隆音色（0.6B-Base 模型）
2. voice-design: 基于文本描述生成音色（1.7B-VoiceDesign 模型）
3. custom-voice: 使用预定义 speaker（0.6B-CustomVoice 模型）

使用方法:
    # Voice Clone 模式（XH Wrapped，默认）
    python native_demo.py \
        --mode voice-clone \
        --model ./data/models/Qwen3-TTS-12HZ_0.6B-Base/ \
        --ref_audio /path/to/reference.wav \
        --ref_text "参考音频的文本内容" \
        --text "要生成的文本" \
        --out output_clone.wav

    # Voice Clone 模式（使用原始模型）
    python native_demo.py \
        --mode voice-clone \
        --use-original \
        --model ./data/models/Qwen3-TTS-12HZ_0.6B-Base/ \
        --ref_audio /path/to/reference.wav \
        --ref_text "参考音频的文本内容" \
        --text "要生成的文本" \
        --out output_clone_original.wav

    # Voice Design 模式（1.7B）
    python native_demo.py \
        --mode voice-design \
        --model ./data/models/Qwen3-TTS-12Hz-1.7B-VoiceDesign/ \
        --text "要生成的文本" \
        --instruct "音色描述：体现温柔甜美的女声" \
        --out output_design.wav

    # Custom Voice 模式（0.6B-CustomVoice）
    python native_demo.py \
        --mode custom-voice \
        --model ./data/models/Qwen3-TTS-12HZ_0.6B-CustomVoice/ \
        --text "要生成的文本" \
        --speaker serena \
        --out output_custom.wav
"""

import argparse
from typing import cast, Union

import soundfile as sf
import torch
from loguru import logger

from xh_model_zoo.xh_llm.models.qwen3_tts import XHQwen3TTSModel

try:
    from qwen_tts import Qwen3TTSModel
    ORIGINAL_MODEL_AVAILABLE = True
except ImportError:
    ORIGINAL_MODEL_AVAILABLE = False
    Qwen3TTSModel = None


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS Native PyTorch Inference Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # 基础参数
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["voice-clone", "voice-design", "custom-voice"],
        help="推理模式"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="模型路径"
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="要生成的文本内容"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Chinese",
        help="语言（默认：Chinese）"
    )
    parser.add_argument(
        "--out",
        type=str,
        default="output.wav",
        help="输出音频文件路径"
    )

    # Voice Clone 模式参数
    parser.add_argument(
        "--ref_audio",
        type=str,
        help="参考音频路径（voice-clone 模式必需）"
    )
    parser.add_argument(
        "--ref_text",
        type=str,
        help="参考音频的文本内容（voice-clone 模式必需）"
    )
    parser.add_argument(
        "--xvec_only",
        action="store_true",
        help="仅使用 x-vector 模式（voice-clone）"
    )

    # Voice Design 模式参数
    parser.add_argument(
        "--instruct",
        type=str,
        help="音色描述指令（voice-design 模式必需）"
    )

    # Custom Voice 模式参数
    parser.add_argument(
        "--speaker",
        type=str,
        default="serena",
        help="预定义 speaker 名称（custom-voice 模式）"
    )

    # 模型配置参数
    parser.add_argument(
        "--use-original",
        action="store_true",
        help="使用原始 Qwen3TTSModel（不使用 XH wrapper）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="设备（默认：cuda:0）"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="fp16",
        choices=["fp16", "fp32", "bf16"],
        help="数据类型（默认：fp16）"
    )
    parser.add_argument(
        "--attn",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
        help="注意力实现（默认：sdpa）"
    )

    # 生成参数
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--do_sample", action="store_true", default=True)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--subtalker_dosample", action="store_true", default=True)
    parser.add_argument("--subtalker_top_k", type=int, default=50)
    parser.add_argument("--subtalker_top_p", type=float, default=1.0)
    parser.add_argument("--subtalker_temperature", type=float, default=0.9)

    args = parser.parse_args()

    # 参数验证
    if args.mode == "voice-clone" and (not args.ref_audio or not args.ref_text):
        parser.error("voice-clone 模式需要 --ref_audio 和 --ref_text 参数")
    if args.mode == "voice-design" and not args.instruct:
        parser.error("voice-design 模式需要 --instruct 参数")
    if args.use_original and not ORIGINAL_MODEL_AVAILABLE:
        parser.error("--use-original 需要安装 qwen_tts 包: pip install qwen_tts")

    # 加载模型
    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    logger.info(f"Loading model from {args.model}")
    logger.info(f"Device: {args.device}, dtype: {args.dtype}, attn: {args.attn}")
    logger.info(f"Model type: {'Original Qwen3TTSModel' if args.use_original else 'XH Wrapped Model'}")

    if args.use_original:
        # 使用原始 Qwen3TTSModel
        model = Qwen3TTSModel.from_pretrained(
            args.model,
            device_map=args.device,
            torch_dtype=dtype,
            attn_implementation=args.attn,
        )
        logger.info(f"Model loaded: Original Qwen3TTSModel")
    else:
        # 使用 XH wrapper
        model = XHQwen3TTSModel.from_pretrained(
            args.model,
            device_map=args.device,
            dtype=dtype,
            attn_implementation=args.attn,
        )
        model = cast(XHQwen3TTSModel, model)
        logger.info(f"Model loaded: type={model.model.tts_model_type}, size={model.model.tts_model_size}")

    # 生成参数
    gen_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "top_k": args.top_k,
        "top_p": args.top_p,
        "temperature": args.temperature,
        "repetition_penalty": args.repetition_penalty,
        "subtalker_dosample": args.subtalker_dosample,
        "subtalker_top_k": args.subtalker_top_k,
        "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }

    # 根据模式生成音频
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Text: {args.text}")

    if args.mode == "voice-clone":
        logger.info(f"Reference audio: {args.ref_audio}")
        logger.info(f"Reference text: {args.ref_text}")
        wavs, sr = model.generate_voice_clone(
            text=args.text,
            language=args.language,
            ref_audio=args.ref_audio,
            ref_text=args.ref_text,
            x_vector_only_mode=args.xvec_only,
            **gen_kwargs
        )
    elif args.mode == "voice-design":
        logger.info(f"Instruct: {args.instruct}")
        wavs, sr = model.generate_voice_design(
            text=args.text,
            language=args.language,
            instruct=args.instruct,
            **gen_kwargs
        )
    elif args.mode == "custom-voice":
        logger.info(f"Speaker: {args.speaker}")
        wavs, sr = model.generate_custom_voice(
            text=args.text,
            language=args.language,
            speaker=args.speaker,
            **gen_kwargs
        )

    # 保存音频
    sf.write(args.out, wavs[0], sr)
    duration = len(wavs[0]) / sr
    logger.info(f"✓ Audio saved to {args.out}")
    logger.info(f"  Sample rate: {sr} Hz")
    logger.info(f"  Samples: {len(wavs[0])}")
    logger.info(f"  Duration: {duration:.2f}s")


if __name__ == "__main__":
    main()
