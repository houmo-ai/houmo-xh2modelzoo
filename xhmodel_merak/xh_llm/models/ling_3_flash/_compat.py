"""Compatibility shims for Ling-3's checkpoint-side Transformers code."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


def patch_ling_remote_code_compatibility() -> None:
    """Bridge the Ling checkpoint code to the workspace Transformers build.

    The checkpoint was authored against a Transformers revision that exposed
    ``is_torch_fx_available`` and registered the default RoPE initializer.
    Transformers 5.13 removed both public entries while retaining compatible
    behavior.  Patch the missing names before importing the remote model; no
    checkpoint source file is modified.
    """

    import transformers.modeling_rope_utils as rope_utils
    import transformers.utils.import_utils as import_utils

    if not hasattr(import_utils, "is_torch_fx_available"):
        import_utils.is_torch_fx_available = lambda: True

    if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:

        def _default_rope(config=None, device=None, seq_len=None, layer_type=None):
            del seq_len, layer_type
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            partial = float(getattr(config, "partial_rotary_factor", 1.0))
            dim = int(head_dim * partial)
            theta = float(getattr(config, "rope_theta", 10000.0))
            positions = torch.arange(0, dim, 2, dtype=torch.int64, device=device).float()
            inv_freq = 1.0 / (theta ** (positions / dim))
            return inv_freq, 1.0

        rope_utils.ROPE_INIT_FUNCTIONS["default"] = _default_rope

    # Once AutoConfig has imported the checkpoint configuration module,
    # preserve the serialized ``rope_scaling: null`` semantics. Transformers
    # 5.x stores it as ``{rope_type: default}``; the Ling remote model treats
    # every non-null mapping as YaRN and unconditionally reads ``factor``.
    # A class property also covers GPTQModel/AutoRound, whose loaders own the
    # config object and cannot receive our normalized instance directly.
    for module_name, module in tuple(sys.modules.items()):
        if "configuration_bailing_moe_v3" not in module_name or module is None:
            continue
        config_cls = getattr(module, "BailingMoeV3Config", None)
        if not isinstance(config_cls, type):
            continue
        # The checkpoint's remote class accidentally declares model_type="".
        # AutoConfig preserves the serialized instance value while
        # PretrainedConfig.to_dict()/save_pretrained writes the class value,
        # silently producing an unsupported checkpoint after quantization.
        config_cls.model_type = "bailing_hybrid"
        if getattr(config_cls, "_xh_ling_rope_compat", False):
            continue

        def _get_rope_scaling(instance):
            value = instance.__dict__.get("rope_scaling", None)
            if (
                isinstance(value, dict)
                and value.get("rope_type") == "default"
                and "factor" not in value
            ):
                return None
            return value

        def _set_rope_scaling(instance, value):
            instance.__dict__["rope_scaling"] = value

        config_cls.rope_scaling = property(_get_rope_scaling, _set_rope_scaling)
        config_cls._xh_ling_rope_compat = True

    # Transformers 5.13 changed ``_tied_weights_keys`` from a list to a
    # mapping. Ling does not tie embeddings in this checkpoint; normalize the
    # remote class metadata so save_pretrained (used by both quantizers) works.
    for module_name, module in tuple(sys.modules.items()):
        if "modeling_bailing_moe_v3" not in module_name or module is None:
            continue
        # Check every checkpoint-defined class rather than just the two base
        # classes. Transformers recursively inspects this attribute on all
        # child modules while saving, and the remote module may be imported
        # under a dynamically generated package name by GPTQModel.
        for model_cls in vars(module).values():
            if (
                isinstance(model_cls, type)
                and model_cls.__module__ == module_name
                and isinstance(getattr(model_cls, "_tied_weights_keys", None), list)
            ):
                model_cls._tied_weights_keys = {}


def normalize_ling_config(config):
    """Undo a Transformers 5.x normalization incompatible with remote code."""

    # AutoConfig imports the checkpoint module as part of constructing this
    # instance, so a second pass can now patch its class metadata as well.
    patch_ling_remote_code_compatibility()

    rope_scaling = getattr(config, "rope_scaling", None)
    if (
        isinstance(rope_scaling, dict)
        and rope_scaling.get("rope_type") == "default"
        and "factor" not in rope_scaling
    ):
        # The serialized Ling config contains null.  Transformers 5.x expands
        # it to {rope_type: default}, while the remote MLA constructor assumes
        # every non-null mapping contains a scaling factor.
        config.rope_scaling = None
    return config


def load_ling_config(hf_model_dir: str | Path):
    """Load Ling's remote config after fixing its empty class model_type.

    Calling AutoConfig directly emits a misleading architecture-mismatch
    warning because the checkpoint code declares ``model_type = ""`` even
    though config.json correctly stores ``bailing_hybrid``. Import and patch
    that class first, then let its normal ``from_pretrained`` path run.
    """

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    patch_ling_remote_code_compatibility()
    config_cls = get_class_from_dynamic_module(
        "configuration_bailing_moe_v3.BailingMoeV3Config",
        str(hf_model_dir),
    )
    config_cls.model_type = "bailing_hybrid"
    patch_ling_remote_code_compatibility()
    config = normalize_ling_config(config_cls.from_pretrained(str(hf_model_dir)))
    # The checkpoint serializes an empty ``_name_or_path``.  AutoModel's
    # trusted-code resolver later uses this field as the repository id when
    # building the compatibility model for ``generate``.  Rebind it to the
    # local checkpoint/export directory so ``auto_map`` resolves locally.
    config._name_or_path = str(Path(hf_model_dir).expanduser().resolve())
    return config


__all__ = [
    "load_ling_config",
    "normalize_ling_config",
    "patch_ling_remote_code_compatibility",
]
