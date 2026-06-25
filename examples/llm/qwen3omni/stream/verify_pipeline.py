#!/usr/bin/env python
# Copyright 2025 HOUMO AI
"""验证三阶段流水并行推理的脚本"""

import sys
import time as _time
from pathlib import Path
from types import SimpleNamespace

# Add stream directory to path
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "tests"))

import torch
import torch.nn as nn
from _fakes import FakeCode2Wav, FakeHMONNXSession
from code2wav_stage import HMONNXCode2WavStage
from connectors import Talker2Code2WavConnector, Thinker2TalkerConnector
from orchestrator import OmniOrchestrator
from talker_stage import HMONNXTalkerStage
from thinker_stage import HMONNXThinkerStage


class SlowFakeHMONNXSession:
    """带延迟的 FakeHMONNXSession，用于展示流水线并行效果"""

    def __init__(self, delay=0.01, output_shapes=None, return_logits=True, return_hidden=True):
        self.inputs = [SimpleNamespace(name=f"input_{i}") for i in range(10)]
        self.output_shapes = output_shapes
        self.return_logits = return_logits
        self.return_hidden = return_hidden
        self.call_count = 0
        self.delay = delay

    def forward(self, *args, **kwargs):
        self.call_count += 1
        _time.sleep(self.delay)  # 模拟推理延迟
        seq_len = 1
        hidden_size = 64
        for arg in args:
            if isinstance(arg, torch.Tensor) and arg.dim() >= 2:
                seq_len = max(seq_len, int(arg.shape[1]))
                hidden_size = int(arg.shape[-1])
                break

        vocab_size = 128
        outputs = []
        if self.return_logits:
            outputs.append(torch.randn(1, seq_len, vocab_size, dtype=torch.float32))
        if self.return_hidden:
            outputs.append(torch.randn(1, seq_len, hidden_size, dtype=torch.float16))

        if len(outputs) == 1:
            return outputs[0]
        return outputs


class SlowFakeCode2Wav:
    """带延迟的 FakeCode2Wav"""

    def __init__(self, delay=0.005, total_upsample=600):
        self.total_upsample = total_upsample
        self.delay = delay
        self.hmonnx = FakeHMONNXSession(return_logits=False, return_hidden=False)
        self.hmonnx_max_code_len = 512

    def forward(self, codes):
        _time.sleep(self.delay)  # 模拟推理延迟
        if isinstance(codes, torch.Tensor):
            code_len = int(codes.shape[-1])
        else:
            code_len = 1
        wav_len = code_len * self.total_upsample
        return torch.randn(1, wav_len, dtype=torch.float32)


def build_pipeline(hidden_size=64, vocab_size=128, num_code_groups=2, use_slow_fakes=False):
    """构建三阶段流水线

    Parameters
    ----------
    use_slow_fakes : bool
        如果为 True，使用带延迟的 Fake session 来展示流水线并行效果
    """
    # 选择使用快速还是慢速的 Fake session
    if use_slow_fakes:
        ThinkerSession = lambda: SlowFakeHMONNXSession(delay=0.01)
        TalkerSession = lambda: SlowFakeHMONNXSession(delay=0.008)
        PredictorSession = lambda: SlowFakeHMONNXSession(delay=0.005)
        Code2WavSession = lambda: SlowFakeCode2Wav(delay=0.003)
    else:
        ThinkerSession = FakeHMONNXSession
        TalkerSession = FakeHMONNXSession
        PredictorSession = FakeHMONNXSession
        Code2WavSession = FakeCode2Wav

    # Thinker
    token_emb = nn.Embedding(vocab_size, hidden_size)
    token_emb.eval()
    thinker = HMONNXThinkerStage(
        prefill_session=ThinkerSession(),
        decode_session=ThinkerSession(),
        token_embedding=token_emb,
        kv_cache_info={'shape': [1, 4, 128, 128], 'num_decoder_layers': 4},
        input_sequence_length=32,
        accept_hidden_layer=3,
        eos_token_ids={99},
        supports_hidden_states=True,
    )

    # Talker
    talker = HMONNXTalkerStage(
        talker_prefill_session=TalkerSession(),
        talker_decode_session=TalkerSession(),
        predictor_prefill_session=PredictorSession(),
        predictor_decode_session=PredictorSession(),
        talker_kv_cache_info={'shape': [1, 4, 64, 128], 'num_decoder_layers': 4},
        predictor_kv_cache_info={'shape': [1, 2, 32, 128], 'num_decoder_layers': 2},
        static_prefill_len=16,
        static_predictor_prefill_len=4,
        num_code_groups=num_code_groups,
        num_lm_heads=8,
        thinker_hidden_size=hidden_size,
    )

    # Code2Wav
    fake_c2w = Code2WavSession()
    code2wav = HMONNXCode2WavStage(
        code2wav_session=fake_c2w,
        total_upsample=600,
    )

    # Connectors
    t2t = Thinker2TalkerConnector(
        accept_hidden_layer=3,
        thinker_hidden_size=hidden_size,
        talker_hidden_size=hidden_size,
    )
    t2c = Talker2Code2WavConnector(
        codec_chunk_frames=5,
        codec_left_context_frames=2,
    )

    return OmniOrchestrator(thinker, talker, code2wav, t2t, t2c)


def run_verification(use_slow_fakes=False, max_new_tokens=5):
    """运行验证"""
    orchestrator = build_pipeline(use_slow_fakes=use_slow_fakes)
    input_ids = torch.randint(0, 128, (1, 8))

    mode_str = "慢速模拟 (展示流水线并行)" if use_slow_fakes else "快速模拟"
    print(f'\n模式: {mode_str}')
    print(f'输入: {input_ids.shape}')
    print(f'最大新 token 数: {max_new_tokens}')
    print('-' * 70)

    events = []
    start_time = _time.time()

    for event in orchestrator.generate_stream(input_ids, max_new_tokens=max_new_tokens):
        elapsed = _time.time() - start_time
        if event.type == 'thinker_token':
            print(f'[{elapsed:.4f}s] THINKER_TOKEN: token_id={event.data.get("token_id")}, step={event.data.get("step")}')
        elif event.type == 'audio_chunk':
            audio = event.data.get("audio")
            samples = audio.shape[-1] if audio is not None else 0
            print(f'[{elapsed:.4f}s] AUDIO_CHUNK: chunk_index={event.data.get("chunk_index")}, samples={samples}')
        elif event.type == 'complete':
            text_ids = event.data.get("text_ids")
            audio = event.data.get("audio")
            print(f'[{elapsed:.4f}s] COMPLETE: tokens={text_ids.shape[-1] if text_ids is not None else 0}, audio_samples={audio.shape[-1] if audio is not None else 0}')
        events.append(event)

    total_elapsed = _time.time() - start_time

    print('-' * 70)
    print(f'总事件数: {len(events)}')
    print(f'总耗时: {total_elapsed:.4f}s')

    # 分析事件顺序
    thinker_tokens = [e for e in events if e.type == 'thinker_token']
    audio_chunks = [e for e in events if e.type == 'audio_chunk']
    complete = [e for e in events if e.type == 'complete']

    print(f'\n事件统计:')
    print(f'  - Thinker tokens: {len(thinker_tokens)}')
    print(f'  - Audio chunks: {len(audio_chunks)}')
    print(f'  - Complete: {len(complete)}')

    # 验证流水并行
    if audio_chunks and thinker_tokens:
        first_audio_time = next(e.data.get('elapsed', 0) for e in events if e.type == 'audio_chunk')
        last_thinker_time = max(e.data.get('elapsed', 0) for e in events if e.type == 'thinker_token')
        print(f'\n流水并行验证:')
        print(f'  - 首个 audio_chunk 时间: {first_audio_time:.4f}s')
        print(f'  - 末个 thinker_token 时间: {last_thinker_time:.4f}s')
        if first_audio_time < last_thinker_time:
            print('  ✓ 验证通过: Audio 在 Thinker 完成前开始 (流水线并行)')
        else:
            print('  ✗ 验证失败: Audio 在 Thinker 完成后才开始')

    # 验证事件顺序
    print(f'\n事件顺序验证:')
    event_types = [e.type for e in events]
    if 'thinker_token' in event_types and 'audio_chunk' in event_types:
        first_token_idx = event_types.index('thinker_token')
        first_audio_idx = event_types.index('audio_chunk')
        complete_idx = event_types.index('complete') if 'complete' in event_types else -1

        print(f'  - 首个 thinker_token 索引: {first_token_idx}')
        print(f'  - 首个 audio_chunk 索引: {first_audio_idx}')
        print(f'  - complete 索引: {complete_idx}')

        if first_token_idx < first_audio_idx < complete_idx:
            print('  ✓ 事件顺序正确: thinker_token → audio_chunk → complete')
        else:
            print('  ✗ 事件顺序异常')

    return events, total_elapsed


def main():
    print('=' * 70)
    print('三阶段流水并行推理验证')
    print('=' * 70)

    # 1. 快速模式验证
    print('\n' + '=' * 70)
    print('测试 1: 快速模拟 (验证基本功能)')
    print('=' * 70)
    events_fast, time_fast = run_verification(use_slow_fakes=False, max_new_tokens=5)

    # 2. 慢速模式验证 (展示流水线并行)
    print('\n' + '=' * 70)
    print('测试 2: 慢速模拟 (展示流水线并行效果)')
    print('=' * 70)
    events_slow, time_slow = run_verification(use_slow_fakes=True, max_new_tokens=5)

    # 3. 总结
    print('\n' + '=' * 70)
    print('验证总结')
    print('=' * 70)
    print(f'快速模式: {time_fast:.4f}s')
    print(f'慢速模式: {time_slow:.4f}s')

    # 计算理论串行时间 vs 实际时间
    # Thinker: 1 prefill + 5 decode = 6 steps * 0.01s = 0.06s
    # Talker: 1 prefill + 5 decode = 6 steps * 0.008s = 0.048s
    # Predictor: 5 decode * 0.005s = 0.025s
    # Code2Wav: ~3 chunks * 0.003s = 0.009s
    # 理论串行时间: 0.06 + 0.048 + 0.025 + 0.009 = 0.142s
    # 理论流水线时间: max(Thinker, Talker, Code2Wav) ≈ 0.06s
    print(f'\n理论串行时间 (慢速): ~0.14s')
    print(f'实际流水线时间 (慢速): {time_slow:.4f}s')
    if time_slow < 0.12:
        print('✓ 流水线并行生效: 实际时间远小于理论串行时间')
    else:
        print('✗ 流水线并行效果不明显')

    print('\n' + '=' * 70)
    print('三阶段流水并行推理验证完成!')
    print('=' * 70)

    return 0


if __name__ == '__main__':
    sys.exit(main())