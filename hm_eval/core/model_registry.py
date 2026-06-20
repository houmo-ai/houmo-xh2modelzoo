"""Model registry — discovers and loads model configurations from YAML files."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

logger = logging.getLogger(__name__)

_HM_EVAL_ROOT = Path(__file__).resolve().parent.parent
_MODEL_CONFIGS_DIR = _HM_EVAL_ROOT / "model_configs"
_DEFAULT_MODEL_ROOT = Path("/data01/datasets")


@dataclass
class BackendConfig:
    """Configuration for a single backend."""
    type: str = ""  # "float", "gptqmodel", or "hmonnx"
    dtype: str = "bfloat16"
    device_map: str = "auto"
    # float-specific
    model_class: str = ""
    processor_class: str = "AutoTokenizer"
    use_processor: bool = False
    chat_template: str = "tokenizer"  # "tokenizer" or "processor"
    experts_implementation: str = "eager"
    # hmonnx-specific
    onnx_model_type: str = ""
    xhquant_config: str = ""
    export_config: str = ""
    export_configs: list[str] = field(default_factory=list)  # available config files for export
    export_script: str = ""
    export_meta_info: str = ""
    vision_export_meta_info: str = ""
    auto_offload: bool = False
    sliding_window: bool = False
    kv_cache_heterogeneous: bool = False
    resource_tight_mode: bool = False
    # extra arbitrary args
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Full configuration for an evaluatable model."""
    config_id: str
    display_name: str
    model_family: str
    hf_model_dir: str
    transformers_version: str
    model_class: str
    source_hf_model_dir: str = ""
    processor_class: str = "AutoTokenizer"
    backends: dict[str, BackendConfig] = field(default_factory=dict)
    recommended_datasets: list[str] = field(default_factory=list)
    normalize_choice_range: str = "A-J"
    disable_thinking: bool = True
    system_prompt: str = ""
    max_tokens: int = 512
    # runtime state
    hf_model_available: bool = False
    hmonnx_available: bool = False

    def check_availability(self, repo_root: Path) -> None:
        """Check if model weights and HMONNX exports exist on disk."""
        self.hf_model_available = Path(self.hf_model_dir).is_dir()
        hmonnx_cfg = self.backends.get("hmonnx")
        if hmonnx_cfg:
            # HMONNX export path must be provided manually at submit time.
            self.hmonnx_available = False


def _parse_backend_config(raw: dict[str, Any]) -> BackendConfig:
    extra_args = raw.get("extra_args", {})
    return BackendConfig(
        type=raw.get("type", ""),
        dtype=extra_args.get("torch_dtype", raw.get("dtype", "bfloat16")),
        device_map=extra_args.get("device_map", raw.get("device_map", "auto")),
        model_class=raw.get("model_class", ""),
        processor_class=raw.get("processor_class", "AutoTokenizer"),
        use_processor=raw.get("use_processor", False),
        chat_template=raw.get("chat_template", "tokenizer"),
        experts_implementation=raw.get("experts_implementation", "eager"),
        onnx_model_type=raw.get("onnx_model_type", ""),
        xhquant_config=raw.get("xhquant_config", ""),
        export_config=raw.get("export_config", ""),
        export_configs=raw.get("export_configs", []),
        export_script=raw.get("export_script", ""),
        export_meta_info=raw.get("export_meta_info", ""),
        vision_export_meta_info=raw.get("vision_export_meta_info", ""),
        auto_offload=raw.get("auto_offload", False),
        sliding_window=raw.get("sliding_window", False),
        kv_cache_heterogeneous=raw.get("kv_cache_heterogeneous", False),
        resource_tight_mode=raw.get("resource_tight_mode", False),
        extra_args=extra_args,
    )


def load_model_config(yaml_path: Path) -> ModelConfig:
    """Load a single model config from a YAML file."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    backends: dict[str, BackendConfig] = {}
    for backend_name, backend_raw in raw.get("backends", {}).items():
        backends[backend_name] = _parse_backend_config(backend_raw)

    return ModelConfig(
        config_id=raw.get("config_id", yaml_path.stem),
        display_name=raw.get("display_name", yaml_path.stem),
        model_family=raw.get("model_family", ""),
        hf_model_dir=raw.get("hf_model_dir", ""),
        source_hf_model_dir=raw.get("hf_model_dir", ""),
        transformers_version=raw.get("transformers_version", ""),
        model_class=raw.get("model_class", raw.get("architecture", "")),
        processor_class=raw.get("processor_class", "AutoTokenizer"),
        backends=backends,
        recommended_datasets=raw.get("recommended_datasets", []),
        normalize_choice_range=raw.get("normalize_choice_range", "A-J"),
        disable_thinking=raw.get("disable_thinking", True),
        system_prompt=raw.get("system_prompt", ""),
        max_tokens=raw.get("max_tokens", 512),
    )


class ModelRegistry:
    """Registry that loads all model configs from model_configs/ directory."""

    def __init__(self, repo_root: Optional[Path] = None) -> None:
        self.repo_root = repo_root or _HM_EVAL_ROOT.parent
        self.configs_dir = _MODEL_CONFIGS_DIR
        self._models: dict[str, ModelConfig] = {}
        self._model_root_override: Optional[Path] = None

    def _resolve_hf_model_dir(self, raw_hf_model_dir: str) -> str:
        if not raw_hf_model_dir or self._model_root_override is None:
            return raw_hf_model_dir

        raw_path = Path(raw_hf_model_dir)
        try:
            relative_path = raw_path.relative_to(_DEFAULT_MODEL_ROOT)
            return str(self._model_root_override / relative_path)
        except ValueError:
            return str(self._model_root_override / raw_path.name)

    def set_model_root(self, model_root: Optional[str]) -> None:
        normalized_root = (model_root or "").strip()
        self._model_root_override = Path(normalized_root).expanduser() if normalized_root else None
        self.scan()

    def get_active_model_root(self) -> Optional[str]:
        if self._model_root_override is None:
            return None
        return str(self._model_root_override)

    def scan(self) -> None:
        """Scan model_configs/ directory and load all YAML files."""
        self._models.clear()
        if not self.configs_dir.is_dir():
            logger.warning("Model configs dir not found: %s", self.configs_dir)
            return

        for yaml_file in sorted(self.configs_dir.glob("*.yaml")):
            try:
                config = load_model_config(yaml_file)
                config.hf_model_dir = self._resolve_hf_model_dir(config.source_hf_model_dir)
                config.check_availability(self.repo_root)
                self._models[config.config_id] = config
                logger.info(
                    "Loaded model config: %s (HF=%s, HMONNX=%s)",
                    config.display_name,
                    "✓" if config.hf_model_available else "✗",
                    "✓" if config.hmonnx_available else "✗",
                )
            except Exception:
                logger.exception("Failed to load model config: %s", yaml_file)

    def list_models(self, available_only: bool = False) -> list[ModelConfig]:
        """Return all loaded model configs."""
        models = list(self._models.values())
        if available_only:
            return [model for model in models if model.hf_model_available]
        return models

    def get_model(self, config_id: str) -> Optional[ModelConfig]:
        """Get a model config by its ID."""
        return self._models.get(config_id)

    def get_model_choices(self, available_only: bool = True) -> list[str]:
        """Return model display names for UI dropdown."""
        return [m.display_name for m in self.list_models(available_only=available_only)]

    def get_model_by_display_name(self, display_name: str) -> Optional[ModelConfig]:
        """Lookup model config by display name."""
        for m in self._models.values():
            if m.display_name == display_name:
                return m
        return None

    def available_backends(self, config_id: str) -> list[str]:
        """Return available backends for a model (checking disk availability)."""
        m = self._models.get(config_id)
        if m is None:
            return []
        backends = []
        if "float" in m.backends and m.hf_model_available:
            backends.append("float")
        if "hmonnx" in m.backends and m.hmonnx_available:
            backends.append("hmonnx")
        # Float is always listable (even if model not on disk, show with warning)
        if "float" in m.backends and "float" not in backends:
            backends.append("float")
        return backends
