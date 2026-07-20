import torch.nn as nn

from xhmodel_merak.xh_llm.wrap_model import _collect_wrapping_modules, traceable_module_placeholder_context
from xhquant.utils.registry import DynamicModule, _DMRegistryCls


class LeafChild(nn.Module):
    pass


class Root(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([LeafChild(), LeafChild()])


class LeafParent(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = LeafChild()


class ParentRoot(nn.Module):
    def __init__(self):
        super().__init__()
        self.parent = LeafParent()


class CallbackWrap(nn.Module):
    pass


class LeafParentDM(DynamicModule):
    @classmethod
    def convert(cls, module, config):
        module.__class__ = cls
        return module


class LeafChildDM(DynamicModule):
    @classmethod
    def convert(cls, module, config):
        module.__class__ = cls
        return module


class CallbackWrapDM(DynamicModule):
    @classmethod
    def convert(cls, module, config):
        module.__class__ = cls
        return module


class Logger:
    def debug(self, message):
        pass


def test_collect_wrapping_modules_descends_into_module_list():
    registry = _DMRegistryCls("test_collect_module_list")
    registry.register_module({LeafChild: "LeafChild"}, LeafChildDM)
    model = Root()

    modules = _collect_wrapping_modules(model, registry, Logger())

    assert modules == list(model.layers)


def test_collect_wrapping_modules_stops_at_non_container_fx_leaf():
    from xhquant.nn import FX_LEAF_MODULES

    registry = _DMRegistryCls("test_collect_leaf_parent")
    registry.register_module({LeafParent: "LeafParent"}, LeafParentDM)
    registry.register_module({LeafChild: "LeafChild"}, LeafChildDM)
    FX_LEAF_MODULES.register_module(module=LeafParent)
    model = ParentRoot()

    modules = _collect_wrapping_modules(model, registry, Logger())

    assert modules == [model.parent]


def test_traceable_module_placeholder_context_temporarily_removes_registered_types():
    registry = _DMRegistryCls("test_placeholder_context")
    registry.register_module({LeafParent: "LeafParent"}, LeafParentDM)
    registry.register_module({LeafChild: "LeafChild"}, LeafChildDM)

    assert LeafParent in registry
    assert LeafChild in registry
    assert registry.get(LeafParent) is not None


def test_traceable_module_placeholder_context_callback_registers_temporary_wrap():
    registry = _DMRegistryCls("test_placeholder_context_callback")
    registry.register_module({LeafParent: "LeafParent"}, LeafParentDM)
    registry.register_module({LeafChild: "LeafChild"}, LeafChildDM)
    callback_events = []

    def callback(registry):
        callback_events.append((LeafParent in registry, LeafChild in registry, CallbackWrap in registry))
        registry.register_module({CallbackWrap: "CallbackWrap"}, CallbackWrapDM)

    with traceable_module_placeholder_context(["LeafParent"], registry=registry, callback=callback):
        assert LeafParent not in registry
        assert LeafChild in registry
        assert CallbackWrap in registry
        assert registry.get(CallbackWrap) is not None

    assert LeafParent in registry
    assert LeafChild in registry
    assert CallbackWrap not in registry
    assert registry.get(CallbackWrap) is None
    assert callback_events == [(False, True, False)]

    with traceable_module_placeholder_context(["LeafParent"], registry=registry):
        assert LeafParent not in registry
        assert LeafChild in registry
        assert registry.get(LeafParent) is None

    assert LeafParent in registry
    assert LeafChild in registry
    assert registry.get(LeafParent) is not None
