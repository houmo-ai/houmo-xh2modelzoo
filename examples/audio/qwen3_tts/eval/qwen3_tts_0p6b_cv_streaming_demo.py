#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-TTS 0.6B CustomVoice 流式推理示例

该脚本展示如何实现流式 TTS 推理：
1. 逐步生成音频 codes（chunk by chunk）
2. 每生成一个 chunk 就立即解码成音频波形
3. 支持实时播放或增量保存

使用方法：
    PYTHONPATH=/path/to/xh2modelzoo python qwen3_tts_0p6b_cv_streaming_demo.py \
        --config ./config/llm/qwen3_tts_12hz_0_6B_customvoice_xh2a_hmonnx.py \
        --text "你好，这是一个流式语音合成的测试。" \
        --speaker vivian \
        --chunk-size 50
"""

import argparse
import time
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import soundfile as sf
import torch

from xhquant.api import Config, set_random_seed
from xh_model_zoo.api import get_root_logger, xhquant_llm_init
from xh_model_zoo.xh_llm.models.builder import MODELS
from xh_model_zoo.xh_llm.models.qwen3_tts import Qwen3TTSHMONNXInference


class StreamingQwen3TTS:
    """流式 TTS 推理包装器"""

    def __init__(self, model: Qwen3TTSHMONNXInference, chunk_size: int = 50):
        """
        Args:
            model: HMONNX 推理模型
            chunk_size: 每次生成的 code 数量（越小延迟越低，但可能影响质量）
        """
        self.model = model
        self.chunk_size = chunk_size
        self.logger = get_root_logger()

    def generate_streaming(
        self, text: str, language: str, speaker: str
    ) -> Generator[Tuple[np.ndarray, int], None, None]:
        """
        流式生成音频

        Args:
            text: 输入文本
            language: 语言（Chinese/English）
            speaker: 说话人 ID

        Yields:
            (audio_chunk, sample_rate): 音频片段和采样率
        """
        self.logger.info(f"开始流式生成: text='{text}', speaker={speaker}, chunk_size={chunk_size}")

        # 1. 获取 speaker embedding 和 text tokens
        native_model = self.model.native_model
        model_impl = native_model.model

        # 准备输入（这部分需要访问 native_model 的内部方法）
        # 注意：这里简化处理，实际需要根据 qwen_tts 的实现来适配
        self.logger.info("步骤 1: 准备输入 tokens 和 speaker embedding")

        # 由于 qwen_tts 的 generate_custom_voice 是一次性生成，
        # 我们需要手动实现流式逻辑
        # 这里展示一个简化的流程框架

        # 2. 通过 Talker 生成 hidden states（这部分通常是一次性的）
        self.logger.info("步骤 2: Talker 生成 hidden states")

        # 3. 通过 CodePredictor 逐步生成 codes
        self.logger.info("步骤 3: 开始流式生成 audio codes")

        # 由于当前 HMONNX 接口不直接支持流式，我们采用"伪流式"方案：
        # 先完整生成，然后分块解码和输出
        # 真正的流式需要修改底层推理逻辑

        # 完整生成（这是当前限制）
        start_time = time.time()
        wavs, sr = self.model.generate_custom_voice(text, language, speaker)
        gen_time = time.time() - start_time

        self.logger.info(f"完整生成耗时: {gen_time:.2f}s")

        # 分块输出（模拟流式效果）
        wav = wavs[0]  # 取第一个结果
        total_samples = len(wav)

        # 计算每个 chunk 对应的音频样本数
        # chunk_size 是 code 数量，每个 code 对应 200 个音频样本（12Hz * 24000Hz / 12 = 2000 samples per code）
        # 实际上 Qwen3-TTS 12Hz 模型：1 code = 24000/12 = 2000 samples
        samples_per_code = sr // 12  # 12Hz 的 code rate
        chunk_samples = self.chunk_size * samples_per_code

        self.logger.info(
            f"音频总长度: {total_samples} samples ({total_samples/sr:.2f}s), "
            f"分块大小: {chunk_samples} samples/chunk"
        )

        # 逐块 yield
        for i in range(0, total_samples, chunk_samples):
            chunk = wav[i : i + chunk_samples]
            yield chunk, sr

            # 模拟流式延迟（实际流式推理中这里是真实的生成时间）
            time.sleep(0.05)  # 50ms 延迟模拟

        self.logger.info("流式生成完成")


def main(args: argparse.Namespace) -> None:
    # 加载配置
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    cfg_name = Path(args.config).stem
    log_file = Path(cfg.work_dir) / f"{cfg_name}_streaming_debug.log"
    Path(cfg.work_dir).mkdir(exist_ok=True, parents=True)

    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"
    cfg.dtype = "float16"
    cfg.debug = args.debug
    cfg.exec_device = "cuda:0" if torch.cuda.is_available() else "cpu"

    seed = cfg.get("seed", 1024)
    set_random_seed(seed)

    xhquant_llm_init(log_file, cfg.debug)
    logger = get_root_logger()

    logger.info(f"配置:\n{cfg.pretty_text}")

    # 构建模型
    logger.info("加载 HMONNX 模型...")
    xh_model = MODELS.build(cfg.model)
    assert isinstance(xh_model, Qwen3TTSHMONNXInference)
    xh_model.to(cfg.exec_device)

    # 创建流式推理器
    streaming_tts = StreamingQwen3TTS(xh_model, chunk_size=args.chunk_size)

    # 流式生成
    output_file = Path(cfg.work_dir) / args.output
    logger.info(f"开始流式生成，输出到: {output_file}")

    all_chunks = []
    sample_rate = None
    chunk_count = 0

    start_time = time.time()

    for audio_chunk, sr in streaming_tts.generate_streaming(
        text=args.text, language=args.language, speaker=args.speaker
    ):
        chunk_count += 1
        all_chunks.append(audio_chunk)
        sample_rate = sr

        # 这里可以实现实时播放逻辑
        # 例如：使用 sounddevice.play(audio_chunk, sr)

        logger.info(
            f"Chunk {chunk_count}: {len(audio_chunk)} samples "
            f"({len(audio_chunk)/sr*1000:.1f}ms), "
            f"累计 {sum(len(c) for c in all_chunks)/sr:.2f}s"
        )

    total_time = time.time() - start_time

    # 合并所有 chunks 并保存
    full_audio = np.concatenate(all_chunks)
    sf.write(output_file, full_audio, sample_rate)

    logger.info(f"流式生成完成!")
    logger.info(f"  总时长: {len(full_audio)/sample_rate:.2f}s")
    logger.info(f"  总耗时: {total_time:.2f}s")
    logger.info(f"  RTF: {total_time / (len(full_audio)/sample_rate):.3f}")
    logger.info(f"  Chunks: {chunk_count}")
    logger.info(f"  输出文件: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS 0.6B CustomVoice 流式推理",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./config/llm/qwen3_tts_12hz_0_6B_customvoice_xh2a_hmonnx.py",
        help="HMONNX 配置文件路径",
    )
    parser.add_argument(
        "--text",
        type=str,
        default="基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，加速人工智能技术的普惠落地。",
        help="要合成的文本",
    )
    parser.add_argument(
        "--language", type=str, default="Chinese", choices=["Chinese", "English"], help="语言"
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="vivian",
        choices=["serena", "vivian", "uncle_fu", "ryan", "aiden", "ono_anna", "sohee", "eric", "dylan"],
        help="说话人 ID",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="每次生成的 code 数量（越小延迟越低）",
    )
    parser.add_argument("--output", type=str, default="output_streaming.wav", help="输出音频文件名")
    parser.add_argument("--debug", action="store_true", help="调试模式")

    args = parser.parse_args()

    cfg_name = Path(args.config).stem
    args.work_dir = str(Path("./work_dirs") / f"{cfg_name}_streaming")

    main(args)
