import importlib.util
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace

import torch


STREAM_MODULE_PATH = Path("examples/llm/qwen3omni/qwen3_omni_hmonnx_generate_stream.py")


def test_stream_example_script_exists():
    assert STREAM_MODULE_PATH.exists()


def _load_stream_module(monkeypatch):
    fake_pipeline = types.ModuleType("_hmonnx_pipeline")
    fake_pipeline._ensure_hm_pixel_values = lambda inputs: inputs
    fake_pipeline.apply_artifact_replacements = lambda *args, **kwargs: None
    fake_pipeline.discover_artifacts = lambda *args, **kwargs: {}
    fake_pipeline.save_json = lambda *args, **kwargs: None
    fake_pipeline.validate_golden_outputs = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "_hmonnx_pipeline", fake_pipeline)

    fake_api = types.ModuleType("xhquant.api")
    fake_api.get_root_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None)
    fake_api.set_random_seed = lambda *args, **kwargs: None
    fake_api.xhquant_init = lambda *args, **kwargs: None

    fake_xhquant = types.ModuleType("xhquant")
    fake_xhquant.api = fake_api
    monkeypatch.setitem(sys.modules, "xhquant", fake_xhquant)
    monkeypatch.setitem(sys.modules, "xhquant.api", fake_api)

    fake_runtime_pkg = types.ModuleType("xhquant.xhonnxruntime")
    fake_hmonnx_module = types.ModuleType("xhquant.xhonnxruntime.hmonnx_inference")

    class DummyHMONNXInference:
        def __init__(self, onnx_file):
            self.onnx_file = onnx_file

        def forward(self, values):
            return values

    fake_hmonnx_module.HMONNXInference = DummyHMONNXInference
    fake_runtime_pkg.hmonnx_inference = fake_hmonnx_module
    monkeypatch.setitem(sys.modules, "xhquant.xhonnxruntime", fake_runtime_pkg)
    monkeypatch.setitem(sys.modules, "xhquant.xhonnxruntime.hmonnx_inference", fake_hmonnx_module)

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Qwen3OmniMoeForConditionalGeneration = object
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_processing = types.ModuleType("xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe")
    fake_processing.Qwen3OmniMoeProcessor = object
    monkeypatch.setitem(
        sys.modules,
        "xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe",
        fake_processing,
    )

    fake_soundfile = types.ModuleType("soundfile")
    fake_soundfile.SoundFile = type("DummySoundFile", (), {})
    fake_soundfile.write = lambda *args, **kwargs: None
    fake_soundfile.__spec__ = ModuleSpec("soundfile", loader=None)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    spec = importlib.util.spec_from_file_location(
        "qwen3_omni_hmonnx_generate_stream_testmod",
        STREAM_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_stream_module_for_main(
    monkeypatch,
    *,
    resolved_device_map="cuda:4",
    validation_max_memory=None,
    ensure_hm_pixel_values_fn=None,
):
    if validation_max_memory is None:
        validation_max_memory = {0: "40.0GiB", "cpu": "64.0GiB"}
    if ensure_hm_pixel_values_fn is None:
        ensure_hm_pixel_values_fn = lambda inputs: inputs

    patch_calls = []
    fake_pipeline = types.ModuleType("_hmonnx_pipeline")
    fake_pipeline._ensure_hm_pixel_values = ensure_hm_pixel_values_fn
    fake_pipeline._resolve_validation_device_map = lambda device_map, logger=None: resolved_device_map
    fake_pipeline._build_safe_validation_max_memory = lambda logger=None: validation_max_memory
    fake_pipeline._patch_inputs_embeds_generation_device = lambda module, module_name, logger=None: patch_calls.append(
        ("inputs_embeds", module_name)
    )
    fake_pipeline._patch_runtime_device_property = lambda module, module_name, logger=None: patch_calls.append(
        ("runtime_device", module_name)
    )
    fake_pipeline.apply_artifact_replacements = lambda *args, **kwargs: None
    fake_pipeline.discover_artifacts = lambda *args, **kwargs: {"talker": {}, "talker_prediction": {}, "code2wav": {}}
    fake_pipeline.save_json = lambda *args, **kwargs: None
    fake_pipeline.validate_golden_outputs = lambda *args, **kwargs: {}
    monkeypatch.setitem(sys.modules, "_hmonnx_pipeline", fake_pipeline)

    fake_api = types.ModuleType("xhquant.api")
    fake_api.get_root_logger = lambda: SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)
    fake_api.set_random_seed = lambda *args, **kwargs: None
    fake_api.xhquant_init = lambda *args, **kwargs: None

    fake_xhquant = types.ModuleType("xhquant")
    fake_xhquant.api = fake_api
    monkeypatch.setitem(sys.modules, "xhquant", fake_xhquant)
    monkeypatch.setitem(sys.modules, "xhquant.api", fake_api)

    fake_runtime_pkg = types.ModuleType("xhquant.xhonnxruntime")
    fake_hmonnx_module = types.ModuleType("xhquant.xhonnxruntime.hmonnx_inference")

    class DummyHMONNXInference:
        def __init__(self, onnx_file):
            self.onnx_file = onnx_file

        def forward(self, values):
            return values

    fake_hmonnx_module.HMONNXInference = DummyHMONNXInference
    fake_runtime_pkg.hmonnx_inference = fake_hmonnx_module
    monkeypatch.setitem(sys.modules, "xhquant.xhonnxruntime", fake_runtime_pkg)
    monkeypatch.setitem(sys.modules, "xhquant.xhonnxruntime.hmonnx_inference", fake_hmonnx_module)

    fake_soundfile = types.ModuleType("soundfile")

    class DummySoundFile:
        def __init__(self, *args, **kwargs):
            self.writes = []

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def write(self, data):
            self.writes.append(data)

    fake_soundfile.SoundFile = DummySoundFile
    fake_soundfile.write = lambda *args, **kwargs: None
    fake_soundfile.__spec__ = ModuleSpec("soundfile", loader=None)
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    captured = {}

    class DummySubmodule:
        def __init__(self):
            self._parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))

        def parameters(self):
            yield self._parameter

    class DummyModel:
        def __init__(self):
            self._parameter = torch.nn.Parameter(torch.zeros(1, dtype=torch.float16))
            self.talker = DummySubmodule()
            self.talker.code_predictor = DummySubmodule()
            self.code2wav = SimpleNamespace(
                total_upsample=1,
                device=torch.device("cpu"),
                chunked_decode=lambda *args, **kwargs: torch.zeros(1, 1),
            )

        def eval(self):
            return self

        def parameters(self):
            yield self._parameter

        def generate(self, **kwargs):
            return SimpleNamespace(sequences=torch.tensor([[1, 2]])), torch.zeros(1, 2)

    class DummyModelClass:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            captured["model_path"] = model_path
            captured["kwargs"] = kwargs
            return DummyModel()

    class DummyProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return DummyProcessor()

        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

        def __call__(self, *args, **kwargs):
            class DummyInputs(dict):
                def to(self, *args, **kwargs):
                    return self

            return DummyInputs(
                {
                    "input_ids": torch.ones((1, 1), dtype=torch.long),
                    "attention_mask": torch.ones((1, 1), dtype=torch.long),
                }
            )

        def batch_decode(self, *args, **kwargs):
            return ["hello"]

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Qwen3OmniMoeForConditionalGeneration = DummyModelClass
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_processing = types.ModuleType("xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe")
    fake_processing.Qwen3OmniMoeProcessor = DummyProcessor
    monkeypatch.setitem(
        sys.modules,
        "xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe",
        fake_processing,
    )

    spec = importlib.util.spec_from_file_location(
        "qwen3_omni_hmonnx_generate_stream_main_testmod",
        STREAM_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, captured, patch_calls


class _FakeCode2Wav:
    def __init__(self, total_upsample=2):
        self.total_upsample = total_upsample
        self.device = torch.device("cpu")
        self.calls = []

    def forward(self, codes):
        self.calls.append(codes.clone())
        sample_count = codes.shape[-1] * self.total_upsample
        return torch.arange(sample_count, dtype=torch.float32).view(1, -1)

    def chunked_decode(self, codes, chunk_size=300, left_context_size=25):
        return self.forward(codes)


class _FakeFloat16Code2Wav(_FakeCode2Wav):
    def forward(self, codes):
        self.calls.append(codes.clone())
        sample_count = codes.shape[-1] * self.total_upsample
        return torch.arange(sample_count, dtype=torch.float16).view(1, -1)


def test_streaming_decoder_emits_chunks_and_flushes_tail(monkeypatch):
    module = _load_stream_module(monkeypatch)
    emitted = []
    decoder = module.StreamingCode2WavDecoder(
        code2wav=_FakeCode2Wav(),
        chunk_size=2,
        left_context_size=1,
        on_audio_chunk=lambda chunk, chunk_index, code_steps: emitted.append((chunk.clone(), chunk_index, code_steps)),
    )

    for step in range(5):
        decoder.add_residual_codes(torch.tensor([[step, step + 10]], dtype=torch.long))
    final_audio = decoder.finalize()

    assert [item[1] for item in emitted] == [0, 1, 2]
    assert [item[2] for item in emitted] == [2, 4, 5]
    assert [item[0].shape[-1] for item in emitted] == [4, 4, 2]
    assert [call.shape[-1] for call in decoder.code2wav.calls] == [2, 3, 2]
    assert final_audio.shape[-1] == 10


def test_streaming_decoder_converts_audio_chunks_to_float32(monkeypatch):
    module = _load_stream_module(monkeypatch)
    emitted = []
    decoder = module.StreamingCode2WavDecoder(
        code2wav=_FakeFloat16Code2Wav(),
        chunk_size=2,
        left_context_size=0,
        on_audio_chunk=lambda chunk, *_: emitted.append(chunk.clone()),
    )

    decoder.add_residual_codes(torch.tensor([[1, 11]], dtype=torch.long))
    decoder.add_residual_codes(torch.tensor([[2, 12]], dtype=torch.long))
    final_audio = decoder.finalize()

    assert emitted
    assert emitted[0].dtype == torch.float32
    assert final_audio.dtype == torch.float32


def test_patch_talker_step_callback_forwards_residual_codes(monkeypatch):
    module = _load_stream_module(monkeypatch)
    seen = []

    class FakeTalker:
        def _update_model_kwargs_for_generation(
            self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1
        ):
            updated = dict(model_kwargs)
            updated["num_new_tokens"] = num_new_tokens
            return updated

    talker = FakeTalker()
    original = talker._update_model_kwargs_for_generation

    with module.patch_talker_step_callback(talker, lambda residual: seen.append(residual.clone())):
        updated = talker._update_model_kwargs_for_generation(
            SimpleNamespace(hidden_states=("unused", torch.tensor([[1, 2, 3]], dtype=torch.long))),
            {"seed": 1},
            num_new_tokens=2,
        )

    assert updated["num_new_tokens"] == 2
    assert len(seen) == 1
    assert torch.equal(seen[0], torch.tensor([[1, 2, 3]], dtype=torch.long))
    assert talker._update_model_kwargs_for_generation.__func__ is original.__func__


def test_generate_stream_yields_audio_chunks_and_complete_result(monkeypatch):
    module = _load_stream_module(monkeypatch)

    class FakeTalker:
        def _update_model_kwargs_for_generation(
            self, outputs, model_kwargs, is_encoder_decoder=False, num_new_tokens=1
        ):
            return dict(model_kwargs)

    class FakeModel:
        def __init__(self):
            self.talker = FakeTalker()
            self.code2wav = _FakeCode2Wav()

        def generate(self, **kwargs):
            first = SimpleNamespace(hidden_states=("unused", torch.tensor([[1, 11]], dtype=torch.long)))
            second = SimpleNamespace(hidden_states=("unused", torch.tensor([[2, 12]], dtype=torch.long)))
            self.talker._update_model_kwargs_for_generation(first, {})
            self.talker._update_model_kwargs_for_generation(second, {})
            audio = self.code2wav.chunked_decode(
                torch.tensor([[[1, 2], [11, 12]]], dtype=torch.long),
                chunk_size=1,
                left_context_size=0,
            )
            return SimpleNamespace(sequences=torch.tensor([[1, 2, 3]], dtype=torch.long)), audio

    model = FakeModel()
    decoder = module.StreamingCode2WavDecoder(
        code2wav=model.code2wav,
        chunk_size=1,
        left_context_size=0,
    )
    events = list(module.generate_stream(model, stream_decoder=decoder, speaker="Ethan"))

    assert [event["type"] for event in events] == ["audio_chunk", "audio_chunk", "complete"]
    assert events[0]["audio"].shape[-1] == 2
    assert events[1]["audio"].shape[-1] == 2
    assert events[-1]["audio"].shape[-1] == 4
    assert torch.equal(events[-1]["text_ids"].sequences, torch.tensor([[1, 2, 3]], dtype=torch.long))


def test_main_uses_safe_loading_and_runtime_patches(monkeypatch, tmp_path):
    module, captured, patch_calls = _load_stream_module_for_main(
        monkeypatch,
        resolved_device_map="auto",
        validation_max_memory={0: "40.0GiB", 1: "41.0GiB", "cpu": "64.0GiB"},
    )
    monkeypatch.setattr(
        module,
        "generate_stream",
        lambda *args, **kwargs: iter(
            [
                {"type": "audio_chunk", "audio": torch.zeros(1, 1), "chunk_index": 0, "code_steps": 1},
                {"type": "complete", "text_ids": SimpleNamespace(sequences=torch.tensor([[1, 2]])), "audio": torch.zeros(1, 1)},
            ]
        ),
    )

    args = SimpleNamespace(
        model="/tmp/fake-model",
        work_dir=str(tmp_path),
        auto_discover=True,
        code2wav_hmonnx=None,
        code2wav_static_code_len=126,
        cases="text",
        speaker="Ethan",
        max_new_tokens=8,
        stream_chunk_size=2,
        stream_left_context_size=1,
        device_map="auto",
        seed=1234,
        debug=False,
    )

    module.main(args)

    assert captured["model_path"] == "/tmp/fake-model"
    assert captured["kwargs"]["device_map"] == "auto"
    assert captured["kwargs"]["max_memory"] == {0: "40.0GiB", 1: "41.0GiB", "cpu": "64.0GiB"}
    assert ("inputs_embeds", "talker") in patch_calls
    assert ("inputs_embeds", "talker.code_predictor") in patch_calls
    assert ("runtime_device", "code2wav") in patch_calls


def test_main_drops_hmonnx_only_pixel_kwargs_before_stream_generate(monkeypatch, tmp_path):
    def _inject_hm_pixel_values(inputs):
        inputs["hm_pixel_values"] = torch.ones((1, 1), dtype=torch.float16)
        inputs["hm_pixel_values_videos"] = torch.ones((1, 1), dtype=torch.float16)
        return inputs

    module, _, _ = _load_stream_module_for_main(
        monkeypatch,
        ensure_hm_pixel_values_fn=_inject_hm_pixel_values,
    )
    captured_generate_kwargs = {}

    def _fake_generate_stream(*args, **kwargs):
        captured_generate_kwargs.update(kwargs)
        return iter(
            [
                {"type": "audio_chunk", "audio": torch.zeros(1, 1), "chunk_index": 0, "code_steps": 1},
                {"type": "complete", "text_ids": SimpleNamespace(sequences=torch.tensor([[1, 2]])), "audio": torch.zeros(1, 1)},
            ]
        )

    monkeypatch.setattr(module, "generate_stream", _fake_generate_stream)

    args = SimpleNamespace(
        model="/tmp/fake-model",
        work_dir=str(tmp_path),
        auto_discover=True,
        code2wav_hmonnx=None,
        code2wav_static_code_len=126,
        cases="vision",
        speaker="Ethan",
        max_new_tokens=8,
        stream_chunk_size=2,
        stream_left_context_size=1,
        device_map="cuda:0",
        seed=1234,
        debug=False,
    )

    module.main(args)

    assert "hm_pixel_values" not in captured_generate_kwargs
    assert "hm_pixel_values_videos" not in captured_generate_kwargs
