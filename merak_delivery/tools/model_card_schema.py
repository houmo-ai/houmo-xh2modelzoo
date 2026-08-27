from __future__ import annotations

import re
from copy import deepcopy
from typing import Any


PARAMETER_STATUS = {"verified", "inferred", "missing"}
COMPUTE_STATUS = {"verified", "inferred", "missing"}


class ModelCardSchemaValidationError(ValueError):
    def __init__(self, path: tuple[str | int, ...], message: str) -> None:
        self.field = ".".join(str(part) for part in path) or "<root>"
        super().__init__(message)


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return False


def _resolve_ref(root_schema: dict[str, Any], reference: str) -> dict[str, Any]:
    if not reference.startswith("#/"):
        raise ValueError(f"unsupported schema reference: {reference}")
    resolved: Any = root_schema
    for part in reference[2:].split("/"):
        resolved = resolved[part.replace("~1", "/").replace("~0", "~")]
    if not isinstance(resolved, dict):
        raise ValueError(f"schema reference does not resolve to a mapping: {reference}")
    return resolved


def _validate_schema_value(
    value: Any,
    schema: dict[str, Any],
    root_schema: dict[str, Any],
    path: tuple[str | int, ...],
) -> None:
    if "$ref" in schema:
        _validate_schema_value(value, _resolve_ref(root_schema, schema["$ref"]), root_schema, path)
        return

    if "const" in schema and value != schema["const"]:
        raise ModelCardSchemaValidationError(path, f"must equal {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        raise ModelCardSchemaValidationError(path, f"must be one of {schema['enum']!r}")

    expected_types = schema.get("type")
    if isinstance(expected_types, str):
        expected_types = [expected_types]
    if isinstance(expected_types, list) and not any(_matches_type(value, item) for item in expected_types):
        raise ModelCardSchemaValidationError(path, f"must have type {' or '.join(expected_types)}")

    if isinstance(value, dict):
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        for key in required:
            if key not in value:
                raise ModelCardSchemaValidationError(path + (key,), "is required")
        additional = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in properties:
                _validate_schema_value(item, properties[key], root_schema, path + (key,))
            elif additional is False:
                raise ModelCardSchemaValidationError(path + (key,), "is not allowed")
            elif isinstance(additional, dict):
                _validate_schema_value(item, additional, root_schema, path + (key,))

    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ModelCardSchemaValidationError(path, f"must contain at least {schema['minItems']} item(s)")
        if schema.get("uniqueItems") and any(
            value[index] == value[other]
            for index in range(len(value))
            for other in range(index + 1, len(value))
        ):
            raise ModelCardSchemaValidationError(path, "must contain unique items")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                _validate_schema_value(item, item_schema, root_schema, path + (index,))

    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ModelCardSchemaValidationError(path, f"must contain at least {schema['minLength']} character(s)")
        if schema.get("pattern") and re.search(schema["pattern"], value) is None:
            raise ModelCardSchemaValidationError(path, f"must match pattern {schema['pattern']!r}")

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ModelCardSchemaValidationError(path, f"must be at least {schema['minimum']}")


def validate_schema_instance(value: Any, schema: dict[str, Any]) -> None:
    """Validate the model-card schema subset without adding a runtime dependency."""
    _validate_schema_value(value, schema, schema, ())


def default_parameters() -> dict[str, Any]:
    return {"total": None, "active": None, "unit": "parameters", "status": "missing"}


def default_compute() -> dict[str, Any]:
    return {
        "input_shape": None,
        "prefill": None,
        "decode": None,
        "unit": "TFLOPs",
        "status": "missing",
    }


def _main_component_from_legacy(
    *,
    model: dict[str, Any],
    workflow: dict[str, Any],
    frontend: dict[str, Any],
) -> dict[str, Any]:
    precision = workflow.get("precision", {}) if isinstance(workflow.get("precision"), dict) else {}
    return {
        "id": "model",
        "display_name": model.get("display_name", ""),
        "type": workflow.get("category", ""),
        "precision": precision.get("overall", ""),
        "inputs": deepcopy(frontend.get("inputs", [])),
        "outputs": deepcopy(frontend.get("outputs", [])),
    }


def normalize_model_card(card: dict[str, Any]) -> dict[str, Any]:
    """Return a legacy-compatible in-memory card while preserving v2 sections."""
    if card.get("schema_version") != 2:
        legacy = deepcopy(card)
        if "external" not in legacy or "internal" not in legacy:
            legacy.update(to_v2_model_card(legacy))
        return legacy

    raw = deepcopy(card)
    external = raw.get("external", {}) if isinstance(raw.get("external"), dict) else {}
    internal = raw.get("internal", {}) if isinstance(raw.get("internal"), dict) else {}
    external_model = external.get("model", {}) if isinstance(external.get("model"), dict) else {}
    internal_model = internal.get("model", {}) if isinstance(internal.get("model"), dict) else {}
    precision = external.get("precision", {}) if isinstance(external.get("precision"), dict) else {}
    presentation = internal.get("presentation", {}) if isinstance(internal.get("presentation"), dict) else {}
    io = internal.get("io", {}) if isinstance(internal.get("io"), dict) else {}
    release = deepcopy(internal.get("release", {}) if isinstance(internal.get("release"), dict) else {})
    release["owner"] = external.get("owner", "")

    frontend = {
        "summary": presentation.get("summary", ""),
        "inputs": deepcopy(io.get("inputs", [])),
        "outputs": deepcopy(io.get("outputs", [])),
        "submodels": deepcopy(internal.get("components", [])),
        "demo": deepcopy(presentation.get("demo", {})),
        "limitations": deepcopy(presentation.get("limitations", [])),
        "evidence": deepcopy(presentation.get("evidence", {})),
    }
    if not any(isinstance(item, dict) and item.get("id") == "model" for item in frontend["submodels"]):
        frontend["submodels"].insert(
            0,
            _main_component_from_legacy(
                model={"display_name": external_model.get("display_name", "")},
                workflow=internal.get("workflow", {}) if isinstance(internal.get("workflow"), dict) else {},
                frontend=frontend,
            ),
        )

    legacy = {
        "schema_version": 2,
        "model": {
            "id": external_model.get("id", ""),
            "family": external_model.get("family", ""),
            "display_name": external_model.get("display_name", ""),
            "modality": internal_model.get("modality", []),
            "task": internal_model.get("task", []),
            "tags": internal_model.get("tags", []),
        },
        "source": deepcopy(internal.get("source", {})),
        "workflow": deepcopy(internal.get("workflow", {})),
        "runtime": deepcopy(internal.get("runtime", {})),
        "frontend": frontend,
        "accuracy": deepcopy(precision.get("comparisons", [])),
        "parameters": deepcopy(external.get("parameters", default_parameters())),
        "compute": deepcopy(external.get("compute", default_compute())),
        "external_precision": deepcopy(precision),
        "release": release,
        "external": raw.get("external", {}),
        "internal": raw.get("internal", {}),
    }
    if isinstance(internal.get("benchmark"), dict):
        legacy["benchmark"] = deepcopy(internal["benchmark"])
    return legacy


def to_v2_model_card(card: dict[str, Any]) -> dict[str, Any]:
    """Convert a legacy in-memory card into the v2 on-disk contract."""
    if card.get("schema_version") == 2 and set(card).issubset({"schema_version", "external", "internal"}):
        return deepcopy(card)

    source = deepcopy(card.get("source", {}) if isinstance(card.get("source"), dict) else {})
    workflow = deepcopy(card.get("workflow", {}) if isinstance(card.get("workflow"), dict) else {})
    runtime = deepcopy(card.get("runtime", {}) if isinstance(card.get("runtime"), dict) else {})
    frontend = deepcopy(card.get("frontend", {}) if isinstance(card.get("frontend"), dict) else {})
    release = deepcopy(card.get("release", {}) if isinstance(card.get("release"), dict) else {})
    model = deepcopy(card.get("model", {}) if isinstance(card.get("model"), dict) else {})
    workflow_precision = workflow.get("precision", {}) if isinstance(workflow.get("precision"), dict) else {}
    current_external = card.get("external", {}) if isinstance(card.get("external"), dict) else {}
    current_external_precision = (
        card.get("external_precision", {})
        if isinstance(card.get("external_precision"), dict)
        else current_external.get("precision", {})
        if isinstance(current_external.get("precision"), dict)
        else {}
    )

    internal_release = deepcopy(release)
    owner = internal_release.pop("owner", "")

    components = deepcopy(frontend.get("submodels", []))
    if not any(isinstance(item, dict) and item.get("id") == "model" for item in components):
        components.insert(0, _main_component_from_legacy(model=model, workflow=workflow, frontend=frontend))

    internal: dict[str, Any] = {
        "model": {
            "type": workflow.get("category", ""),
            "modality": deepcopy(model.get("modality", [])),
            "task": deepcopy(model.get("task", [])),
            "tags": deepcopy(model.get("tags", [])),
        },
        "source": source,
        "components": components,
        "io": {
            "inputs": deepcopy(frontend.get("inputs", [])),
            "outputs": deepcopy(frontend.get("outputs", [])),
        },
        "workflow": workflow,
        "runtime": runtime,
        "release": internal_release,
        "presentation": {
            "summary": frontend.get("summary", ""),
            "demo": deepcopy(frontend.get("demo", {})),
            "limitations": deepcopy(frontend.get("limitations", [])),
            "evidence": deepcopy(frontend.get("evidence", {})),
        },
    }
    benchmark = card.get("benchmark")
    if isinstance(benchmark, dict):
        internal["benchmark"] = deepcopy(benchmark)

    return {
        "schema_version": 2,
        "external": {
            "model": {
                "id": model.get("id", ""),
                "family": model.get("family", ""),
                "display_name": model.get("display_name", ""),
            },
            "parameters": deepcopy(
                card.get("parameters", current_external.get("parameters", default_parameters()))
            ),
            "compute": deepcopy(card.get("compute", current_external.get("compute", default_compute()))),
            "precision": {
                "float_format": current_external_precision.get("float_format", ""),
                "quantized_format": current_external_precision.get(
                    "quantized_format", workflow_precision.get("overall", "")
                ),
                "comparisons": deepcopy(
                    card.get("accuracy", current_external_precision.get("comparisons", []))
                ),
            },
            "owner": owner,
        },
        "internal": internal,
    }
