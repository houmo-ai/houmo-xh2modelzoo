"""Declarative test inputs; loading this module never imports model/test code.

A ``test_name.impact.json`` belongs ONLY to the adjacent ``test_name.py``.
Stable, filename-derived ownership makes declaration edits/removals selectable
even when Jenkins provides just changed paths and no Git history.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


MANIFEST_SUFFIX = ".impact.json"


@dataclass(frozen=True)
class ImpactRule:
    id: str
    patterns: tuple[str, ...]
    test: str


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_object)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def validate_paths(values, *, field: str, allow_empty: bool = False) -> tuple[str, ...]:
    if not isinstance(values, list) or (not values and not allow_empty):
        raise ValueError(f"{field} must be a {'possibly empty' if allow_empty else 'non-empty'} list")
    for value in values:
        if not isinstance(value, str) or not value or value.startswith(("/", "!", "./")):
            raise ValueError(f"{field} must contain positive repo-relative paths: {value!r}")
        if "\\" in value or ":" in value or ".." in PurePosixPath(value).parts:
            raise ValueError(f"{field} has an invalid repo-relative path: {value!r}")
        if any(character in value for character in "\n\r\0"):
            raise ValueError(f"{field} has an invalid path: {value!r}")
    return tuple(values)


def manifest_test(path: str, suite_relative: str) -> str | None:
    """Resolve ownership from the path, including for a deleted manifest."""
    if not path.endswith(MANIFEST_SUFFIX):
        return None
    if not path.startswith(f"{suite_relative}/"):
        return None
    test = path.removesuffix(MANIFEST_SUFFIX) + ".py"
    name = PurePosixPath(test).name
    if not (name.startswith("test_") or name.endswith("_test.py")):
        raise ValueError(f"Impact declaration must be adjacent to a test: {path}")
    return test


def load_manifests(repo_root: Path, suite_relative: str, candidates: set[str]) -> tuple[ImpactRule, ...]:
    rules = []
    for path in sorted((repo_root / suite_relative).rglob(f"*{MANIFEST_SUFFIX}")):
        relative = path.relative_to(repo_root).as_posix()
        test = manifest_test(relative, suite_relative)
        if test not in candidates:
            raise ValueError(f"Orphan impact declaration {relative}: missing test {test}")
        data = read_json(path)
        if set(data) != {"version", "rules"} or type(data["version"]) is not int or data["version"] != 1:
            raise ValueError(f"Invalid impact declaration schema in {relative}; expected version 1 and rules")
        entries = data["rules"]
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Impact declaration {relative} must have non-empty rules")
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) != {"id", "patterns"}:
                raise ValueError(f"Each rule in {relative} must contain only id and patterns")
            name = entry["id"]
            if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) or name in seen:
                raise ValueError(f"Invalid/duplicate rule id in {relative}: {name!r}")
            seen.add(name)
            patterns = validate_paths(entry["patterns"], field=f"{relative}:{name}.patterns")
            rules.append(ImpactRule(name, patterns, test))
    return tuple(rules)
