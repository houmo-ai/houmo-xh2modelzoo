# Qwen3-Omni HMONNX End-to-End Streaming Pipeline

参考 vLLM-Omni 的 async-chunk/stage pipeline 架构，为 HMONNX 推理实现 Qwen3-Omni 端到端流式支持。

## 架构

```
┌─────────────────────────────────────────────────────────────┐
│                    OmniOrchestrator                          │
│  ┌──────────┐   Connector1   ┌──────────┐   Connector2     │
│  │ Stage 0  │ ─────────────▶ │ Stage 1  │ ─────────────▶   │
│  │ Thinker  │                │ Talker   │                   │
│  │ (HMONNX) │                │ (HMONNX) │    ┌──────────┐  │
│  └──────────┘                └──────────┘ ─▶ │ Stage 2  │  │
│                                              │ Code2Wav │  │
│                                              │ (HMONNX) │  │
│                                              └──────────┘  │
│                                                    │       │
│                                              StreamEvent   │
│                                              (to consumer) │
└─────────────────────────────────────────────────────────────┘
```

### Stage 0 – Thinker (`thinker_stage.py`)

独立的 HMONNX text prefill/decode 循环，不依赖 HuggingFace `model.generate()`。
每一步 emit 一个 `ThinkerPrefillChunk`（prefill）或 `ThinkerDecodeChunk`（decode），
完全对应 vLLM-Omni 的 `thinker2talker_async_chunk`。

### Stage 1 – Talker (`talker_stage.py`)

独立的 HMONNX talker + code-predictor prefill/decode 循环。
消费 Thinker→Talker connector 的输出，每步生成 `num_code_groups` 个 residual audio codec codes。

### Stage 2 – Code2Wav (`code2wav_stage.py`)

HMONNX code-to-waveform 解码。消费 Talker→Code2Wav connector 的 code chunk，
输出 audio waveform chunk。

### Connectors (`connectors.py`)

- **Thinker2TalkerConnector**: 将 Thinker per-step output 转换为 Talker fused guidance inputs
  （hidden_state, role_mask, bypass_embeds, bypass_mask），对应 vLLM 的 `thinker2talker_async_chunk`。
- **Talker2Code2WavConnector**: 累积 Talker residual codes，按 chunk_size 分块并加入
  left-context overlap，对应 vLLM 的 `talker2code2wav_async_chunk`。

### Orchestrator (`orchestrator.py`)

三线程并发编排器，镜像 vLLM-Omni 的 `AsyncOmni` + `Orchestrator`：
- Thread 1: Thinker → Connector1 → talker_in_q
- Thread 2: talker_in_q → Talker → Connector2 → audio_out_q
- Thread 3: audio_out_q → Code2Wav → consumer
- Main thread: yield `StreamEvent` objects

### Events (`events.py`)

类型安全的数据类：
- `ThinkerPrefillChunk` / `ThinkerDecodeChunk`
- `TalkerInputPrefill` / `TalkerInputDecode`
- `TalkerStepOutput`
- `Code2WavInput`
- `AudioChunk`
- `StreamEvent`

## 使用

### CLI 生成

```bash
python -m examples.llm.qwen3omni.stream.generate_stream \
    --model /path/to/Qwen3-Omni \
    --work-dir work_dirs/qwen3omni \
    --case multimodal \
    --max-new-tokens 128
```

### 编程接口

```python
from stream.orchestrator import OmniOrchestrator
from stream.thinker_stage import HMONNXThinkerStage
from stream.talker_stage import HMONNXTalkerStage
from stream.code2wav_stage import HMONNXCode2WavStage
from stream.connectors import Thinker2TalkerConnector, Talker2Code2WavConnector

# 构建 pipeline
orchestrator = OmniOrchestrator(thinker, talker, code2wav, t2t, t2c)

# 流式生成
for event in orchestrator.generate_stream(input_ids, max_new_tokens=100):
    if event.type == "thinker_token":
        print(f"Token: {event.data['token_id']}")
    elif event.type == "audio_chunk":
        save_audio(event.data["audio"], event.data["chunk_index"])
    elif event.type == "complete":
        save_final(event.data["text_ids"], event.data["audio"])
```

## 测试

```bash
cd examples/llm/qwen3omni/stream
python -m pytest tests/ -v
```

测试覆盖：
- 事件数据类（12 tests）
- Thinker2Talker / Talker2Code2Wav connectors（14 tests）
- Thinker stage（5 tests）
- Talker stage（5 tests）
- Code2Wav stage（5 tests）
- Orchestrator（8 tests）
- 端到端集成（7 tests）
- **共 55 tests**

## 文件结构

```
examples/llm/qwen3omni/stream/
├── __init__.py              # 包文档
├── _utils.py                # HMONNX 工具函数（session、KV cache、padding）
├── events.py                # 阶段数据类和 StreamEvent
├── connectors.py            # Thinker→Talker / Talker→Code2Wav connectors
├── thinker_stage.py         # Stage 0: HMONNX Thinker generation loop
├── talker_stage.py          # Stage 1: HMONNX Talker + code-predictor loop
├── code2wav_stage.py        # Stage 2: HMONNX Code2Wav decode
├── orchestrator.py          # 三线程异步编排器
├── generate_stream.py       # CLI 入口脚本
├── conftest.py              # Root conftest（sys.path + fake re-exports）
├── README.md                # 本文档
└── tests/
    ├── __init__.py
    ├── _fakes.py            # FakeHMONNXSession / FakeCode2Wav
    ├── conftest.py           # 测试 fixtures
    ├── test_events.py        # 事件数据类测试
    ├── test_connectors.py    # Connector 测试
    ├── test_thinker_stage.py # Thinker stage 测试
    ├── test_talker_stage.py  # Talker stage 测试
    ├── test_code2wav_stage.py # Code2Wav stage 测试
    ├── test_orchestrator.py  # Orchestrator 测试
    └── test_integration.py   # 端到端集成测试
```

## 与 vLLM-Omni 的对应关系

| vLLM-Omni | 本实现 |
|---|---|
| `Stage 0: Thinker (LLM_AR)` | `HMONNXThinkerStage` |
| `Stage 1: Talker (LLM_AR)` | `HMONNXTalkerStage` |
| `Stage 2: Code2Wav (Generation)` | `HMONNXCode2WavStage` |
| `thinker2talker_async_chunk` | `Thinker2TalkerConnector.process()` |
| `talker2code2wav_async_chunk` | `Talker2Code2WavConnector.process()` |
| `AsyncOmni` | `OmniOrchestrator.generate_stream()` |
| `OmniChunkTransferAdapter` | `queue.Queue` + `threading.Thread` |
| `codec_chunk_frames=25` | `Talker2Code2WavConnector(codec_chunk_frames=25)` |
| `codec_left_context_frames=25` | `Talker2Code2WavConnector(codec_left_context_frames=25)` |
| `response.audio.delta` | `StreamEvent.audio_chunk()` |
