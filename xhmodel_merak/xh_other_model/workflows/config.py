import copy
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class WorkflowConfig:
    data: dict[str, Any]
    source: str

    @classmethod
    def from_file(cls, config_file: str) -> "WorkflowConfig":
        path = Path(config_file)
        if path.suffix not in {".yaml", ".yml"}:
            raise ValueError(f"Workflow config only supports YAML files for now: {path}")
        with path.open("r", encoding="utf-8") as fin:
            data = yaml.safe_load(fin) or {}
        if not isinstance(data, dict):
            raise TypeError(f"Workflow config must be a YAML mapping: {path}")
        cls._validate_workflow_data(data, str(path))
        return cls(data=data, source=str(path))

    def dump(self, config_file: str) -> str:
        if config_file is None:
            raise ValueError("config_file must be provided")
        path = Path(config_file)
        if path.suffix not in {".yaml", ".yml"}:
            raise ValueError(f"Workflow config only supports YAML files for now: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fout:
            yaml.safe_dump(self.data, fout, allow_unicode=True, sort_keys=False)
        return str(path)

    @property
    def name(self) -> str:
        if self.source is None:
            raise ValueError("WorkflowConfig source can't be None!")
        return Path(self.source).stem

    @property
    def quant(self) -> dict[str, Any] | None:
        quant = self.data["quant"]
        if quant is None:
            return None
        if not isinstance(quant, dict):
            raise TypeError("workflow quant config must be a mapping or null")
        return quant

    @property
    def export(self) -> dict[str, Any]:
        export = self.data["export"]
        if not isinstance(export, dict) or not export:
            raise ValueError("workflow export config must be a non-empty mapping")
        return export

    def build_export_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self.export)

    def with_overrides(self, overrides: Mapping[str, Any] | None = None) -> "WorkflowConfig":
        if not overrides:
            return self
        self._validate_override_paths(overrides)
        data = copy.deepcopy(self.data)
        for path, value in overrides.items():
            self._set_existing_path(data, path, value)

        p = Path(self.source)
        override_source = str(p.with_stem(p.stem + "_override"))
        self._validate_workflow_data(data, override_source)
        return WorkflowConfig(data=data, source=override_source)

    # TODO valid逻辑简化？
    def _validate_override_paths(self, overrides: Mapping[str, Any]) -> None:
        for path, value in overrides.items():
            if not isinstance(path, str) or not path:
                raise ValueError(f"Override path must be a non-empty string, got: {path!r}")
            # Top-level quant may be replaced as a whole, e.g. {"quant": None}
            # for base export or {"quant": {...}} for an externally quantized
            # HF model. Other overrides keep strict existing-key validation so
            # typos such as export.model.foo are still rejected.
            if path == "quant":
                continue
            if path == "export":
                raise ValueError("Top-level export override is not allowed; use dotted export.* paths")
            existing_value = self._get_existing_path(self.data, path)
            self._validate_override_value(existing_value, value, path)

    @staticmethod
    def _get_existing_path(data: dict[str, Any], path: str) -> Any:
        current: Any = data
        parts = path.split(".")
        for index, part in enumerate(parts):
            if not part:
                raise ValueError(f"Invalid override path with empty segment: {path!r}")
            if not isinstance(current, dict):
                prefix = ".".join(parts[:index])
                raise ValueError(f"Override path {path!r} cannot descend into non-mapping field {prefix!r}")
            if part not in current:
                raise KeyError(f"Override path {path!r} does not exist in workflow config")
            current = current[part]
        return current

    @staticmethod
    def _set_existing_path(data: dict[str, Any], path: str, value: Any) -> None:
        current: Any = data
        parts = path.split(".")
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1]] = value

    @classmethod
    def _validate_override_value(cls, existing_value: Any, override_value: Any, path: str) -> None:
        if not isinstance(override_value, Mapping):
            return
        if not isinstance(existing_value, Mapping):
            raise ValueError(f"Override path {path!r} cannot replace a non-mapping field with a mapping")
        for key, value in override_value.items():
            if key not in existing_value:
                raise KeyError(f"Override path {path}.{key} does not exist in workflow config")
            cls._validate_override_value(existing_value[key], value, f"{path}.{key}")

    @staticmethod
    def _validate_workflow_data(data: dict[str, Any], source: str) -> None:
        for field in ("quant", "export"):
            if field not in data:
                raise ValueError(f"workflow config {source} must contain {field!r}")

        quant = data["quant"]
        if quant is not None and not isinstance(quant, dict):
            raise TypeError(f"workflow config {source} field 'quant' must be a mapping or null")

        export = data["export"]
        if not isinstance(export, dict) or not export:
            raise ValueError(f"workflow config {source} field 'export' must be a non-empty mapping")
        
        target_device = export.get("target_device")
        if not isinstance(target_device, str) or not target_device:
            raise ValueError(f"workflow config {source} must specify export.target_device")
        
        model = export.get("model")
        if not isinstance(model, dict) or not model:
            raise ValueError(f"workflow config {source} must contain non-empty export.model")
        if not isinstance(model.get("type"), str) or not model["type"]:
            raise ValueError(f"workflow config {source} must specify non-empty export.model.type")
