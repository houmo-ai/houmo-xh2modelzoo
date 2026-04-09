import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


EXPORT_MODULE_PATH = Path("examples/llm/qwen3omni/qwen3_omni_xh2a_export_talker_model.py")
PIPELINE_MODULE_PATH = Path("examples/llm/qwen3omni/_hmonnx_pipeline.py")


def _load_export_module(
    monkeypatch,
    *,
    resolved_device_map="cuda:2",
    max_memory=None,
    patch_calls=None,
    run_dialogue_validation_fn=None,
):
    if patch_calls is None:
        patch_calls = []
    if run_dialogue_validation_fn is None:
        run_dialogue_validation_fn = lambda *args, **kwargs: {}

    fake_pipeline = types.ModuleType("_hmonnx_pipeline")
    fake_pipeline.run_dialogue_validation = run_dialogue_validation_fn
    fake_pipeline.save_json = lambda *args, **kwargs: None
    fake_pipeline._resolve_validation_device_map = lambda device_map, logger=None: resolved_device_map
    fake_pipeline._build_safe_validation_max_memory = lambda logger=None: max_memory
    fake_pipeline._patch_runtime_device_property = lambda module, module_name, logger=None: None
    fake_pipeline._patch_inputs_embeds_generation_device = (
        lambda module, module_name, logger=None: patch_calls.append((module, module_name))
    )
    monkeypatch.setitem(sys.modules, "_hmonnx_pipeline", fake_pipeline)

    fake_base_converter = types.ModuleType("xh_model_zoo.xh_llm.models.base_converter")
    fake_base_converter.BaseConverter = type(
        "BaseConverter",
        (),
        {"xh1_hmonnx_compatible": staticmethod(lambda names: names)},
    )
    monkeypatch.setitem(sys.modules, "xh_model_zoo.xh_llm.models.base_converter", fake_base_converter)

    fake_builder = types.ModuleType("xh_model_zoo.xh_llm.models.builder")
    fake_builder.wrap_llm_model = lambda model, cfg: model
    monkeypatch.setitem(sys.modules, "xh_model_zoo.xh_llm.models.builder", fake_builder)

    fake_memory_tracker = types.ModuleType("xh_model_zoo.utils.memory_tracker")
    fake_memory_tracker.MemoryTracker = type(
        "MemoryTracker",
        (),
        {
            "__init__": lambda self, *args, **kwargs: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
        },
    )
    monkeypatch.setitem(sys.modules, "xh_model_zoo.utils.memory_tracker", fake_memory_tracker)

    fake_time_profiler = types.ModuleType("xh_model_zoo.utils.time_profiler")
    fake_time_profiler.TimeProfiler = type(
        "TimeProfiler",
        (),
        {
            "__init__": lambda self, *args, **kwargs: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
        },
    )
    monkeypatch.setitem(sys.modules, "xh_model_zoo.utils.time_profiler", fake_time_profiler)

    fake_api = types.ModuleType("xhquant.api")
    fake_api.CacheTensor = lambda value: value
    fake_api.Config = dict
    fake_api.ConfigDict = dict
    fake_api.DeviceType = SimpleNamespace(XH2a="XH2a")
    fake_api.QuantScheme = lambda **kwargs: kwargs
    fake_api.convert_fx_model_to_quanted_model = lambda *args, **kwargs: None
    fake_api.convert_quanted_model_to_hmonnx = lambda *args, **kwargs: None
    fake_api.create_quant_config = lambda quant_scheme: quant_scheme
    fake_api.get_root_logger = lambda: None
    fake_api.xhquant_init = lambda *args, **kwargs: None
    fake_xhquant = types.ModuleType("xhquant")
    fake_xhquant.api = fake_api
    monkeypatch.setitem(sys.modules, "xhquant", fake_xhquant)
    monkeypatch.setitem(sys.modules, "xhquant.api", fake_api)

    spec = importlib.util.spec_from_file_location(
        "qwen3_omni_xh2a_export_talker_model_testmod",
        EXPORT_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, patch_calls


def _install_pipeline_import_stubs(monkeypatch, *, model_class, processor_class):
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Qwen3OmniMoeForConditionalGeneration = model_class
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_soundfile = types.ModuleType("soundfile")

    def _fake_write(path, *args, **kwargs):
        Path(path).write_bytes(b"stub")

    fake_soundfile.write = _fake_write
    monkeypatch.setitem(sys.modules, "soundfile", fake_soundfile)

    fake_api = types.ModuleType("xhquant.api")
    fake_api.CacheTensor = lambda value: value
    monkeypatch.setitem(sys.modules, "xhquant.api", fake_api)

    fake_hmonnx_module = types.ModuleType("xhquant.xhonnxruntime.hmonnx_inference")

    class DummyHMONNXInference:
        def __init__(self, onnx_file):
            self.onnx_file = onnx_file
            self.inputs = [SimpleNamespace(shape=[1, 1, 1])]

        def forward(self, *args, **kwargs):
            return torch.zeros(1, 1, 1)

    fake_hmonnx_module.HMONNXInference = DummyHMONNXInference
    monkeypatch.setitem(sys.modules, "xhquant.xhonnxruntime.hmonnx_inference", fake_hmonnx_module)

    fake_modeling = types.ModuleType(
        "xh_model_zoo.xh_llm.models.qwen3_omni.modeling_qwen3_omni_moe"
    )
    fake_modeling._get_feat_extract_output_lengths = lambda value: value
    monkeypatch.setitem(
        sys.modules,
        "xh_model_zoo.xh_llm.models.qwen3_omni.modeling_qwen3_omni_moe",
        fake_modeling,
    )

    fake_monkey_patch = types.ModuleType(
        "xh_model_zoo.xh_llm.models.qwen3_omni.monkey_patch"
    )
    fake_monkey_patch.Qwen3OmniMoeThinkerForConditionalGeneration_forward = lambda self, *args, **kwargs: None
    monkeypatch.setitem(
        sys.modules,
        "xh_model_zoo.xh_llm.models.qwen3_omni.monkey_patch",
        fake_monkey_patch,
    )

    fake_processing = types.ModuleType(
        "xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe"
    )
    fake_processing.Qwen3OmniMoeProcessor = processor_class
    monkeypatch.setitem(
        sys.modules,
        "xh_model_zoo.xh_llm.models.qwen3_omni.processing_qwen3_omni_moe",
        fake_processing,
    )


def _load_pipeline_module(monkeypatch, *, model_class, processor_class):
    _install_pipeline_import_stubs(
        monkeypatch,
        model_class=model_class,
        processor_class=processor_class,
    )
    spec = importlib.util.spec_from_file_location(
        "qwen3_omni_hmonnx_pipeline_testmod",
        PIPELINE_MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_load_native_model_for_capture_uses_safe_single_gpu(monkeypatch):
    module, patch_calls = _load_export_module(monkeypatch, resolved_device_map="cuda:2", max_memory=None)
    captured = {}

    class DummySubmodule:
        pass

    class DummyModel:
        def __init__(self):
            self.talker = DummySubmodule()
            self.talker.code_predictor = DummySubmodule()

        def eval(self):
            captured["eval_called"] = True
            return self

    class DummyModelClass:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            captured["model_path"] = model_path
            captured["kwargs"] = kwargs
            return DummyModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Qwen3OmniMoeForConditionalGeneration = DummyModelClass
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    model = module._load_native_model_for_capture("/tmp/fake-model", logger)

    assert captured["model_path"] == "/tmp/fake-model"
    assert captured["kwargs"]["device_map"] == "cuda:2"
    assert "max_memory" not in captured["kwargs"]
    assert captured["eval_called"] is True
    assert len(patch_calls) == 2
    assert patch_calls[0][1] == "talker"
    assert patch_calls[1][1] == "talker.code_predictor"
    assert model is not None


def test_load_native_model_for_capture_passes_max_memory_when_auto_remains(monkeypatch):
    validation_max_memory = {0: "55.0GiB", "cpu": "128.0GiB"}
    module, patch_calls = _load_export_module(
        monkeypatch,
        resolved_device_map="auto",
        max_memory=validation_max_memory,
    )
    captured = {}

    class DummySubmodule:
        pass

    class DummyModel:
        def __init__(self):
            self.talker = DummySubmodule()
            self.talker.code_predictor = DummySubmodule()

        def eval(self):
            return self

    class DummyModelClass:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            captured["kwargs"] = kwargs
            return DummyModel()

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.Qwen3OmniMoeForConditionalGeneration = DummyModelClass
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    logger = SimpleNamespace(info=lambda *args, **kwargs: None)

    module._load_native_model_for_capture("/tmp/fake-model", logger)

    assert captured["kwargs"]["device_map"] == "auto"
    assert captured["kwargs"]["max_memory"] == validation_max_memory
    assert len(patch_calls) == 2


def test_run_talker_dialogue_validation_passes_talker_token_limit(monkeypatch):
    captured = {}

    def fake_run_dialogue_validation(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return {}

    module, _ = _load_export_module(
        monkeypatch,
        run_dialogue_validation_fn=fake_run_dialogue_validation,
    )
    logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    work_dir = Path("/tmp/qwen3omni_talker_validation")
    meta_file = work_dir / "meta_talker.json"
    meta_info = {"module": "talker_model"}

    module._run_talker_dialogue_validation(
        "/tmp/fake-model",
        work_dir,
        logger,
        meta_info,
        meta_file,
        max_new_tokens=8,
        talker_max_new_tokens=128,
    )

    assert captured["args"][:3] == ("/tmp/fake-model", work_dir, logger)
    assert captured["kwargs"]["case"] == "multimodal"
    assert captured["kwargs"]["max_new_tokens"] == 8
    assert captured["kwargs"]["talker_max_new_tokens"] == 128
    assert captured["kwargs"]["report_name"] == "talker_dialogue_validation.json"
    assert captured["kwargs"]["output_prefix"] == "talker_dialogue"


def test_run_dialogue_validation_uses_concrete_code2wav_device_when_module_is_meta(monkeypatch, tmp_path):
    class FakeBatch(dict):
        def to(self, *args, **kwargs):
            return self

    class DummyProcessor:
        @staticmethod
        def from_pretrained(model_path):
            return DummyProcessor()

        def apply_chat_template(self, conversation, add_generation_prompt=True, tokenize=False):
            return "prompt"

        def __call__(self, **kwargs):
            return FakeBatch({"input_ids": torch.tensor([[1, 2, 3]], dtype=torch.long)})

        def batch_decode(self, *args, **kwargs):
            return ["decoded"]

    class DummyPredictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))
            self.model = SimpleNamespace()

        def _maybe_initialize_input_ids_for_generation(self, inputs=None, bos_token_id=None, model_kwargs=None):
            return torch.zeros((1, 0), dtype=torch.long)

    class DummyTalker(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))
            self.code_predictor = DummyPredictor()

        def _maybe_initialize_input_ids_for_generation(self, inputs=None, bos_token_id=None, model_kwargs=None):
            return torch.zeros((1, 0), dtype=torch.long)

    class DummyCode2Wav(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.empty(1, device="meta"))
            self._hf_hook = SimpleNamespace(execution_device="cpu")

        @property
        def device(self):
            return next(self.parameters()).device

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.root_param = nn.Parameter(torch.zeros(1))
            self.talker = DummyTalker()
            self.code2wav = DummyCode2Wav()
            self.observed_code2wav_device = None

        def eval(self):
            return self

        def generate(self, **kwargs):
            self.observed_code2wav_device = self.code2wav.device
            assert self.observed_code2wav_device.type == "cpu"
            return SimpleNamespace(sequences=torch.tensor([[1, 2, 3, 4]], dtype=torch.long)), torch.zeros(1, 8)

    model_holder = {}

    class DummyModelClass:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            model_holder["model"] = DummyModel()
            return model_holder["model"]

    pipeline = _load_pipeline_module(
        monkeypatch,
        model_class=DummyModelClass,
        processor_class=DummyProcessor,
    )
    logger = SimpleNamespace(info=lambda *args, **kwargs: None, warning=lambda *args, **kwargs: None)

    report = pipeline.run_dialogue_validation(
        "/tmp/fake-model",
        tmp_path,
        logger,
        case="multimodal",
        device_map="cpu",
    )

    assert model_holder["model"].observed_code2wav_device.type == "cpu"
    assert report["output_text"] == ["decoded"]
    assert (tmp_path / "dialogue_multimodal.wav").exists()


def test_patch_runtime_device_property_keeps_concrete_parameter_device_without_recursion(monkeypatch):
    class DummyProcessor:
        @staticmethod
        def from_pretrained(model_path):
            return DummyProcessor()

    class DummyModelClass:
        @staticmethod
        def from_pretrained(model_path, **kwargs):
            raise AssertionError("from_pretrained should not be called in this unit test")

    pipeline = _load_pipeline_module(
        monkeypatch,
        model_class=DummyModelClass,
        processor_class=DummyProcessor,
    )

    class DummyCode2Wav(nn.Module):
        def __init__(self):
            super().__init__()
            self.param = nn.Parameter(torch.zeros(1))

        @property
        def device(self):
            return next(self.parameters()).device

    module = DummyCode2Wav()
    pipeline._patch_runtime_device_property(
        module,
        "code2wav",
        logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )

    assert module.device.type == "cpu"
