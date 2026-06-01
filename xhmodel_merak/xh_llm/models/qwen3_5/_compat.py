"""
Compatibility shims for transformers API differences between versions.
The local modeling_qwen3_5.py was copied from a newer transformers version
and requires some APIs not present in transformers 4.57.x.
"""

import contextlib
import functools

import torch
import torch.nn as nn


# ─── transformers.initialization ──────────────────────────────────────────────
# Newer transformers has an initialization module with helper functions.
# We provide minimal stubs that match the usage in modeling_qwen3_5.py.


class _Initialization:
    @staticmethod
    def ones_(tensor):
        nn.init.ones_(tensor)

    @staticmethod
    def zeros_(tensor):
        nn.init.zeros_(tensor)

    @staticmethod
    def copy_(tensor, source):
        with torch.no_grad():
            tensor.copy_(source)


init = _Initialization()

# ─── transformers.utils.torch_compilable_check ────────────────────────────────
# A simple assertion-like check that is compilable by torch.compile.


def torch_compilable_check(condition, message=""):
    if not condition:
        raise ValueError(message)


# ─── transformers.utils.generic — missing functions ───────────────────────────


def is_flash_attention_requested(config):
    """Check if the config requests flash attention."""
    attn_impl = getattr(config, "_attn_implementation", None)
    return attn_impl == "flash_attention_2"


@contextlib.contextmanager
def maybe_autocast(device_type=None, enabled=True, dtype=None):
    """Context manager for optional autocast."""
    if enabled:
        with torch.autocast(device_type=device_type or "cuda", dtype=dtype):
            yield
    else:
        yield


def merge_with_config_defaults(fn):
    """Decorator stub — in newer transformers this merges kwargs with config defaults.
    In our case we just pass through since we control inputs explicitly."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


# ─── transformers.utils.output_capturing ──────────────────────────────────────


def capture_outputs(fn):
    """Decorator stub — in newer transformers this captures output types.
    We just pass through."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper


# ─── transformers.integrations.use_kernelized_func ────────────────────────────


def use_kernelized_func(original_func=None):
    """Decorator stub — in newer transformers this registers a kernel function.
    We just return the class/function as-is."""

    def decorator(cls_or_fn):
        return cls_or_fn

    if original_func is not None:
        # Called as @use_kernelized_func(some_func) — returns decorator
        return decorator
    return decorator


# ─── transformers.utils.auto_docstring ────────────────────────────────────────
# The older transformers auto_docstring can't handle newer union type syntax (X | Y).
# Provide a no-op version.


def auto_docstring(fn=None, **kwargs):
    """No-op stub for auto_docstring — avoids compat issues with union types.
    Supports both @auto_docstring and @auto_docstring(custom_intro=...) usage."""
    if fn is not None:
        # Used as @auto_docstring without arguments
        return fn

    # Used as @auto_docstring(custom_intro=...) — return a decorator
    def decorator(obj):
        return obj

    return decorator


# ─── transformers.configuration_utils.PreTrainedConfig ────────────────────────
# In transformers 4.57.x the class is called PretrainedConfig (lowercase 't')
# Newer versions renamed it to PreTrainedConfig.

try:
    from transformers.configuration_utils import PreTrainedConfig
except ImportError:
    from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig  # noqa: F401


# ─── transformers.configuration_utils.layer_type_validation ───────────────────

try:
    from transformers.configuration_utils import layer_type_validation
except ImportError:

    def layer_type_validation(layer_types, num_hidden_layers):
        """Validate that layer_types list matches num_hidden_layers."""
        if layer_types is not None and len(layer_types) != num_hidden_layers:
            raise ValueError(f"layer_types has {len(layer_types)} entries but num_hidden_layers={num_hidden_layers}")


# ─── transformers.modeling_rope_utils.RopeParameters ──────────────────────────

try:
    from transformers.modeling_rope_utils import RopeParameters
except ImportError:
    # Stub — only used as type annotation in config
    RopeParameters = dict
