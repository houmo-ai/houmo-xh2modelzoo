import contextlib
import importlib.util
import sys
import types
from pathlib import Path


MODULE_PATH = Path("xh_model_zoo/xh_llm/models/qwen3moe/qwen_moe_hf_compatible.py")


def test_qwen3moe_hf_compatible_imports_without_transformers_no_init_weights(monkeypatch):
    observed = {}

    fake_accelerate = types.ModuleType("accelerate")

    @contextlib.contextmanager
    def fake_init_empty_weights():
        observed["init_empty_weights_entered"] = observed.get("init_empty_weights_entered", 0) + 1
        yield

    fake_accelerate.init_empty_weights = fake_init_empty_weights
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)

    fake_transformers = types.ModuleType("transformers")

    class DummyGenerationMixin:
        pass

    class DummyQwen3MoeForCausalLM:
        pass

    class DummyDynamicCache:
        def get_seq_length(self):
            return 0

    class DummyAutoConfig:
        @staticmethod
        def from_pretrained(model_dir):
            observed["config_model_dir"] = model_dir
            return {"model_dir": model_dir}

    class DummyAutoModelForCausalLM:
        @staticmethod
        def from_config(config, **kwargs):
            observed["from_config"] = {"config": config, "kwargs": kwargs}
            return {"ok": True}

    fake_transformers.AutoConfig = DummyAutoConfig
    fake_transformers.AutoModelForCausalLM = DummyAutoModelForCausalLM
    fake_transformers.DynamicCache = DummyDynamicCache
    fake_transformers.GenerationMixin = DummyGenerationMixin
    fake_transformers.Qwen3MoeForCausalLM = DummyQwen3MoeForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    fake_cache_utils = types.ModuleType("transformers.cache_utils")
    fake_cache_utils.Cache = object
    monkeypatch.setitem(sys.modules, "transformers.cache_utils", fake_cache_utils)

    fake_modeling_outputs = types.ModuleType("transformers.modeling_outputs")
    fake_modeling_outputs.CausalLMOutputWithPast = object
    monkeypatch.setitem(sys.modules, "transformers.modeling_outputs", fake_modeling_outputs)

    fake_modeling_utils = types.ModuleType("transformers.modeling_utils")
    monkeypatch.setitem(sys.modules, "transformers.modeling_utils", fake_modeling_utils)

    fake_package = types.ModuleType("xh_model_zoo.xh_llm.models.qwen3moe")
    fake_package.__path__ = [str(MODULE_PATH.parent.resolve())]
    monkeypatch.setitem(sys.modules, "xh_model_zoo.xh_llm.models.qwen3moe", fake_package)

    fake_inference = types.ModuleType("xh_model_zoo.xh_llm.models.qwen3moe.inference")
    fake_inference.Qwen3MoeInference = object
    monkeypatch.setitem(sys.modules, "xh_model_zoo.xh_llm.models.qwen3moe.inference", fake_inference)

    spec = importlib.util.spec_from_file_location(
        "xh_model_zoo.xh_llm.models.qwen3moe.qwen_moe_hf_compatible_testmod",
        MODULE_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    model = module.get_empty_hf_model("/tmp/fake-model")

    assert callable(module.no_init_weights)
    assert observed["config_model_dir"] == "/tmp/fake-model"
    assert observed["from_config"]["kwargs"]["torch_dtype"] is not None
    assert observed["init_empty_weights_entered"] == 1
    assert model == {"ok": True}
