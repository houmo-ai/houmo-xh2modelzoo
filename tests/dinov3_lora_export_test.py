import hashlib
import importlib.util
import sys
import types
from pathlib import Path

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "examples/cv/dinov3_lora/dinov3_lora_export.py"
)


def _install_import_stubs(monkeypatch):
    peft = types.ModuleType("peft")
    peft.LoraConfig = object
    peft.TaskType = types.SimpleNamespace(
        FEATURE_EXTRACTION=types.SimpleNamespace(value="FEATURE_EXTRACTION")
    )
    peft.get_peft_model = lambda model, _config: model
    monkeypatch.setitem(sys.modules, "peft", peft)

    transformers = types.ModuleType("transformers")
    transformers.AutoImageProcessor = object
    transformers.AutoModel = object
    image_utils = types.ModuleType("transformers.image_utils")
    image_utils.load_image = lambda image_url: image_url
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.image_utils", image_utils)

    xhquant = types.ModuleType("xhquant")
    xhquant_api = types.ModuleType("xhquant.api")
    xhquant_api.DeviceType = types.SimpleNamespace(XH2a="XH2a")
    xhquant_api.HMONNXGoldenInference = object
    xhquant_api.QuantScheme = object
    xhquant_api.convert_onnx_to_hmonnx = lambda *args, **kwargs: None
    xhquant_api.create_quant_config = lambda *args, **kwargs: None
    xhquant_api.xhquant_init = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "xhquant", xhquant)
    monkeypatch.setitem(sys.modules, "xhquant.api", xhquant_api)


def _load_export_module(monkeypatch):
    _install_import_stubs(monkeypatch)
    spec = importlib.util.spec_from_file_location(
        "dinov3_lora_export_under_test", MODULE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TinyLoraModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = torch.nn.Linear(3, 4, bias=False)
        self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(3, 2, bias=False)})
        self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 4, bias=False)})
        torch.nn.init.constant_(self.lora_A["default"].weight, 1.0)
        torch.nn.init.zeros_(self.lora_B["default"].weight)


def _lora_hash(model):
    digest = hashlib.sha256()
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        digest.update(name.encode())
        digest.update(param.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def test_seeded_lora_init_is_reproducible(monkeypatch):
    module = _load_export_module(monkeypatch)
    first = TinyLoraModel()
    second = TinyLoraModel()

    assert module.initialize_lora_adapter_weights(first, seed=123, std=0.02) == 2
    assert module.initialize_lora_adapter_weights(second, seed=123, std=0.02) == 2

    assert _lora_hash(first) == _lora_hash(second)
    assert torch.count_nonzero(first.lora_B["default"].weight).item() > 0


def test_different_seeds_produce_different_lora_hashes(monkeypatch):
    module = _load_export_module(monkeypatch)
    first = TinyLoraModel()
    second = TinyLoraModel()

    module.initialize_lora_adapter_weights(first, seed=123, std=0.02)
    module.initialize_lora_adapter_weights(second, seed=124, std=0.02)

    assert _lora_hash(first) != _lora_hash(second)


def test_default_or_keep_default_does_not_force_lora_weights(monkeypatch):
    module = _load_export_module(monkeypatch)
    no_seed = TinyLoraModel()
    keep_default = TinyLoraModel()
    no_seed_hash = _lora_hash(no_seed)
    keep_default_hash = _lora_hash(keep_default)

    assert module.initialize_lora_adapter_weights(no_seed, seed=None, std=0.02) == 0
    assert (
        module.initialize_lora_adapter_weights(
            keep_default, seed=123, std=0.02, keep_default_init=True
        )
        == 0
    )

    assert _lora_hash(no_seed) == no_seed_hash
    assert _lora_hash(keep_default) == keep_default_hash
    assert torch.count_nonzero(no_seed.lora_B["default"].weight).item() == 0
    assert torch.count_nonzero(keep_default.lora_B["default"].weight).item() == 0
