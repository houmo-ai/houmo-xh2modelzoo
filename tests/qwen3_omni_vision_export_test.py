import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace


MODULE_PATH = Path("examples/llm/qwen3omni/qwen3_omni_xh2a_export_vision.py")


def _load_module(monkeypatch, *, run_dialogue_validation_fn=None):
    captured = {}
    if run_dialogue_validation_fn is None:

        def run_dialogue_validation_fn(*args, **kwargs):
            captured["args"] = args
            captured["kwargs"] = kwargs
            return {}

    fake_pipeline = types.ModuleType("_hmonnx_pipeline")
    fake_pipeline._create_hmonnx_session = lambda *args, **kwargs: None
    fake_pipeline.discover_artifacts = lambda *args, **kwargs: {}
    fake_pipeline.run_dialogue_validation = run_dialogue_validation_fn
    fake_pipeline.run_text_hmonnx_chain_forward = lambda *args, **kwargs: {}
    fake_pipeline.save_json = lambda *args, **kwargs: None
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

    fake_api = types.ModuleType("xhquant.api")
    fake_api.Config = dict
    fake_api.DeviceType = SimpleNamespace(XH2a="XH2a")
    fake_api.QuantScheme = lambda **kwargs: kwargs
    fake_api.convert_fx_model_to_hmonnx = lambda *args, **kwargs: None
    fake_api.convert_onnx_to_hmonnx = lambda *args, **kwargs: None
    fake_api.get_root_logger = lambda: None
    fake_api.xhquant_init = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "xhquant.api", fake_api)

    fake_mem = types.ModuleType("xh_model_zoo.utils.memory_tracker")
    fake_mem.MemoryTracker = type(
        "MemoryTracker",
        (),
        {
            "__init__": lambda self, *args, **kwargs: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
        },
    )
    monkeypatch.setitem(sys.modules, "xh_model_zoo.utils.memory_tracker", fake_mem)

    fake_prof = types.ModuleType("xh_model_zoo.utils.time_profiler")
    fake_prof.TimeProfiler = type(
        "TimeProfiler",
        (),
        {
            "__init__": lambda self, *args, **kwargs: None,
            "__enter__": lambda self: self,
            "__exit__": lambda self, *args: None,
        },
    )
    monkeypatch.setitem(sys.modules, "xh_model_zoo.utils.time_profiler", fake_prof)

    fake_simplify = types.ModuleType("xhquant.utils.onnxsim_large_model.simplify_large_onnx")
    fake_simplify.simplify_large_onnx = lambda model: (model, True)
    monkeypatch.setitem(sys.modules, "xhquant.utils.onnxsim_large_model.simplify_large_onnx", fake_simplify)

    spec = importlib.util.spec_from_file_location(
        "qwen3_omni_xh2a_export_vision_testmod",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, captured


def test_run_vision_dialogue_validation_passes_talker_token_limit(monkeypatch):
    module, captured = _load_module(monkeypatch)
    logger = SimpleNamespace(info=lambda *args, **kwargs: None)
    work_dir = Path("/tmp/qwen3omni_vision_validation")
    golden_dir = work_dir / "golden"
    meta_info = {"module": "vision_encoder"}
    meta_file = work_dir / "meta_vision.json"

    module._run_vision_dialogue_validation(
        "/tmp/fake-model",
        work_dir,
        golden_dir,
        logger,
        meta_info,
        meta_file,
        case="vision",
        max_new_tokens=64,
        talker_max_new_tokens=96,
        valid_device="auto",
        save_golden=False,
    )

    assert captured["args"][:3] == ("/tmp/fake-model", work_dir, logger)
    assert captured["kwargs"]["case"] == "vision"
    assert captured["kwargs"]["max_new_tokens"] == 64
    assert captured["kwargs"]["talker_max_new_tokens"] == 96
    assert captured["kwargs"]["device_map"] == "auto"
    assert captured["kwargs"]["report_name"] == "vision_dialogue_validation.json"
    assert captured["kwargs"]["output_prefix"] == "vision_dialogue"
