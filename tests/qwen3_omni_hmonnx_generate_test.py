import importlib.util
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import SimpleNamespace

import torch


MODULE_PATH = Path("examples/llm/qwen3omni/qwen3_omni_hmonnx_generate.py")


def _load_module(monkeypatch, *, resolved_device_map="auto", validation_max_memory=None, ensure_hm_pixel_values_fn=None):
    if validation_max_memory is None:
        validation_max_memory = {0: "40.0GiB", "cpu": "64.0GiB"}
    if ensure_hm_pixel_values_fn is None:
        ensure_hm_pixel_values_fn = lambda inputs: inputs

    patch_calls = []
    saved_audio = {}
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
    monkeypatch.setitem(sys.modules, "xhquant.api", fake_api)

    fake_hmonnx_module = types.ModuleType("xhquant.xhonnxruntime.hmonnx_inference")
    fake_hmonnx_module.HMONNXInference = lambda onnx_file: SimpleNamespace(forward=lambda values: values)
    monkeypatch.setitem(sys.modules, "xhquant.xhonnxruntime.hmonnx_inference", fake_hmonnx_module)

    fake_soundfile = types.ModuleType("soundfile")
    fake_soundfile.write = lambda path, data, samplerate=24000: saved_audio.update(
        {"path": path, "dtype": str(data.dtype), "samplerate": samplerate}
    )
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
            captured["generate_kwargs"] = kwargs
            return SimpleNamespace(sequences=torch.tensor([[1, 2]])), torch.zeros(1, 4, dtype=torch.float16)

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
        "qwen3_omni_hmonnx_generate_testmod",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module, captured, patch_calls, saved_audio


def test_main_uses_safe_loading_runtime_patches_and_float32_audio(monkeypatch, tmp_path):
    module, captured, patch_calls, saved_audio = _load_module(
        monkeypatch,
        resolved_device_map="auto",
        validation_max_memory={0: "40.0GiB", 1: "41.0GiB", "cpu": "64.0GiB"},
    )

    args = SimpleNamespace(
        model="/tmp/fake-model",
        work_dir=str(tmp_path),
        auto_discover=True,
        code2wav_hmonnx=None,
        code2wav_static_code_len=126,
        cases="text",
        max_new_tokens=8,
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
    assert saved_audio["dtype"] == "float32"


def test_main_drops_hmonnx_only_pixel_kwargs_before_generate(monkeypatch, tmp_path):
    def _inject_hm_pixel_values(inputs):
        inputs["hm_pixel_values"] = torch.ones((1, 1), dtype=torch.float16)
        inputs["hm_pixel_values_videos"] = torch.ones((1, 1), dtype=torch.float16)
        return inputs

    module, captured, _, _ = _load_module(
        monkeypatch,
        ensure_hm_pixel_values_fn=_inject_hm_pixel_values,
    )

    args = SimpleNamespace(
        model="/tmp/fake-model",
        work_dir=str(tmp_path),
        auto_discover=True,
        code2wav_hmonnx=None,
        code2wav_static_code_len=126,
        cases="vision",
        max_new_tokens=8,
        device_map="cuda:0",
        seed=1234,
        debug=False,
    )

    module.main(args)

    assert "hm_pixel_values" not in captured["generate_kwargs"]
    assert "hm_pixel_values_videos" not in captured["generate_kwargs"]
