import importlib.util
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def test_gemma4_module_files_exist_and_are_exported():
    base = Path("xhmodel_merak/xh_llm/models/gemma4e")
    expected_files = [
        "__init__.py",
        "_llm_model_impl.py",
        "_vision_model_impl.py",
        "_audio_model_impl.py",
        "data_preprocess.py",
        "gemma4_processor.py",
        "gemma4_llm_model.py",
        "gemma4_vision_model.py",
        "gemma4_audio_model.py",
        "gemma4_hmonnx_inference.py",
        "xh_gemma4_config.py",
    ]
    for name in expected_files:
        assert (base / name).exists(), f"missing {name}"

    init_text = (base / "__init__.py").read_text(encoding="utf-8")
    for exported_name in [
        "Gemma4ForConditionalGeneration",
        "XHGemma4Model",
        "XHGemma4VisionModel",
        "XHGemma4AudioModel",
        "XHGemma4_HMONNXModel",
        "XHGemma4ModelConfig",
    ]:
        assert exported_name in init_text


def test_expected_gemma4_examples_exist():
    base = Path("examples_merak/llm/gemma4_e")
    expected_files = [
        "gemma4_e_xh_export_hmonnx.py",
        "gemma4_e_xh_hmonnx_generate.py",
    ]
    for name in expected_files:
        assert (base / name).exists(), f"missing {name}"


def test_gemma4_generate_script_imports_without_soundfile_when_audio_unused():
    script_path = Path("examples_merak/llm/gemma4_e/gemma4_e_xh_hmonnx_generate.py")
    spec = importlib.util.spec_from_file_location("gemma4_e_xh_hmonnx_generate", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._load_audio(None) is None


def test_gemma4_generate_script_loads_wav_without_soundfile(tmp_path, monkeypatch):
    import builtins

    script_path = Path("examples_merak/llm/gemma4_e/gemma4_e_xh_hmonnx_generate.py")
    spec = importlib.util.spec_from_file_location("gemma4_e_xh_hmonnx_generate_fallback", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    wav_path = tmp_path / "fallback.wav"
    with wave.open(str(wav_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        samples = [0, 8192, -8192, 4096]
        wav_file.writeframes(b"".join(struct.pack("<h", sample) for sample in samples))

    original_import = builtins.__import__

    def _missing_soundfile(name, *args, **kwargs):
        if name == "soundfile":
            raise ModuleNotFoundError("soundfile not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing_soundfile)

    audio = module._load_audio(str(wav_path))

    assert isinstance(audio, np.ndarray)
    assert audio.dtype == np.float32
    assert audio.shape == (4,)


def test_gemma4_generate_script_moves_hmonnx_model_to_the_same_device_as_inputs(monkeypatch):
    script_path = Path("examples_merak/llm/gemma4_e/gemma4_e_xh_hmonnx_generate.py")
    spec = importlib.util.spec_from_file_location("gemma4_e_xh_hmonnx_generate", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class FakeImage:
        def convert(self, mode):
            assert mode == "RGB"
            return self

    class FakeModelInputs(dict):
        def __init__(self):
            input_ids = [[11, 12]]
            super().__init__(input_ids=input_ids)
            self.input_ids = input_ids
            self.device = None

        def to(self, device):
            self.device = device
            return self

    class FakeProcessor:
        def __init__(self, model_inputs):
            self._model_inputs = model_inputs

        def apply_chat_template(self, messages):
            assert messages[0]["content"][-1]["text"] == "Describe the image."
            return self._model_inputs

    class FakeTokenizer:
        eos_token_id = 0

        def decode(self, token_ids, skip_special_tokens=True):
            assert skip_special_tokens is True
            return "decoded output"

    class FakeModel:
        def __init__(self, processor, tokenizer):
            self._processor = processor
            self._tokenizer = tokenizer
            self.to_calls = []

        def get_tokenizer(self):
            return self._tokenizer

        def get_tf_processor(self):
            return self._processor

        def to(self, device):
            self.to_calls.append(device)
            return self

        def generate(self, **kwargs):
            return torch.tensor([kwargs["input_ids"][0] + [13]])

    class FakeContextManager:
        def __init__(self, model):
            self.model = model

        def __enter__(self):
            return self.model

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

    model_inputs = FakeModelInputs()
    fake_model = FakeModel(FakeProcessor(model_inputs), FakeTokenizer())

    monkeypatch.setattr(module, "xhquant_init", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "TextStreamer", lambda tokenizer: ("streamer", tokenizer))
    monkeypatch.setattr(module, "LLMInferenceContextManager", FakeContextManager)
    monkeypatch.setattr(module.Image, "open", lambda path: FakeImage())
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(module.AutoLLMHONNXModel, "from_pretrained", lambda config: fake_model)

    module.main(
        SimpleNamespace(
            config="fake-config.json",
            image_path="fake.png",
            audio_path="",
            audio_sampling_rate=16000,
            prompt="Describe the image.",
            max_new_tokens=1,
            debug=False,
        )
    )

    assert model_inputs.device == "cuda"
    assert fake_model.to_calls == ["cuda"]
