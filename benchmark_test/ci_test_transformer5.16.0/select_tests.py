#!/usr/bin/env python3
"""Select CI tests affected by a set of changed files.

Selection is deterministic and auditable:

1. reject changes under frozen legacy directories;
2. apply global safety policy and test-local ``*.impact.json`` declarations;
3. follow reverse Python-import dependencies when no explicit rule owns the file;
4. use model-family naming as a fallback within the active CI suite;
5. fail closed when guarded code has no matching test.

The selector deliberately does not depend on pytest, PyYAML, CodeGraph, or a
network service so it can be the first CI gate in a clean checkout.
"""
from __future__ import annotations

import argparse
import ast
import fnmatch
import functools
import json
import os
import subprocess
import sys
import warnings
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

from _impact_manifest import load_manifests, manifest_test, read_json, validate_paths


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SUITE_RELATIVE = SCRIPT_DIR.relative_to(REPO_ROOT).as_posix()
POLICY_RELATIVE = f"{SUITE_RELATIVE}/impact_policy.json"
SOURCE_ROOTS = ("xhmodel_merak", "examples_merak", SUITE_RELATIVE)


class FrozenPathError(RuntimeError):
    """Raised when a change touches a frozen legacy directory."""


class MissingCoverageError(RuntimeError):
    """Raised when guarded code has no affected test."""


@dataclass(frozen=True)
class Selection:
    changed_files: tuple[str, ...]
    tests: tuple[str, ...]
    reasons: Mapping[str, tuple[str, ...]]
    uncovered_files: tuple[str, ...] = ()
    full_suite: bool = False
    full_suite_reasons: tuple[str, ...] = ()


def _normalized(path: str | os.PathLike[str]) -> str:
    value = str(path).replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    return PurePosixPath(value).as_posix()


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def _looks_like_test_path(path: str) -> bool:
    candidate = PurePosixPath(path)
    if candidate.suffix != ".py":
        return False
    if not path.startswith(f"{SUITE_RELATIVE}/"):
        return False
    return candidate.name.startswith("test_") or candidate.name.endswith("_test.py")


@functools.lru_cache(maxsize=4)
def _test_candidates(repo_root: Path) -> set[str]:
    candidates: set[str] = set()
    root = repo_root / SUITE_RELATIVE
    for path in root.rglob("*.py"):
        name = path.name
        if path.is_file() and (name.startswith("test_") or name.endswith("_test.py")):
            candidates.add(path.relative_to(repo_root).as_posix())
    return candidates


@functools.lru_cache(maxsize=4)
def _default_suite(repo_root: Path) -> set[str]:
    return set(_test_candidates(repo_root))


def _module_name(path: Path, repo_root: Path) -> str | None:
    try:
        relative = path.relative_to(repo_root)
    except ValueError:
        return None
    if relative.suffix != ".py":
        return None
    parts = list(relative.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) if parts else None


@functools.lru_cache(maxsize=4)
def _python_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    for root_name in SOURCE_ROOTS:
        root = repo_root / root_name
        if root.is_dir():
            files.extend(path for path in root.rglob("*.py") if path.is_file())
    return files


def _resolve_from_module(current: str, is_package: bool, level: int, module: str | None) -> str:
    if level == 0:
        return module or ""
    package = current.split(".") if is_package else current.split(".")[:-1]
    keep = max(0, len(package) - (level - 1))
    prefix = package[:keep]
    if module:
        prefix.extend(module.split("."))
    return ".".join(prefix)


def _imported_modules(path: Path, repo_root: Path) -> set[str]:
    current = _module_name(path, repo_root)
    if not current:
        return set()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, UnicodeDecodeError, SyntaxError):
        return set()

    imported: set[str] = set()
    is_package = path.name == "__init__.py"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_from_module(current, is_package, node.level, node.module)
            if base:
                imported.add(base)
            for alias in node.names:
                if alias.name != "*" and base:
                    imported.add(f"{base}.{alias.name}")
    return imported


@functools.lru_cache(maxsize=4)
def _local_import_graph(repo_root: Path) -> tuple[dict[str, set[str]], dict[str, str]]:
    files = _python_files(repo_root)
    module_to_path: dict[str, str] = {}
    for path in files:
        module = _module_name(path, repo_root)
        if module:
            module_to_path[module] = path.relative_to(repo_root).as_posix()

    reverse: dict[str, set[str]] = defaultdict(set)
    for path in files:
        importer = path.relative_to(repo_root).as_posix()
        for imported in _imported_modules(path, repo_root):
            candidate = imported
            while candidate:
                dependency = module_to_path.get(candidate)
                if dependency:
                    reverse[dependency].add(importer)
                    break
                candidate = candidate.rpartition(".")[0]
    return reverse, module_to_path


def _dependent_tests(
    changed_file: str,
    reverse_graph: Mapping[str, set[str]],
    test_candidates: set[str],
) -> set[str]:
    selected: set[str] = set()
    queue: deque[str] = deque([changed_file])
    seen = {changed_file}
    while queue:
        dependency = queue.popleft()
        for importer in reverse_graph.get(dependency, ()):
            if importer in seen:
                continue
            seen.add(importer)
            if importer in test_candidates:
                selected.add(importer)
            queue.append(importer)
    return selected


def _family_from_path(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    markers = ("models", "llm_models", "other_models")
    for marker in markers:
        if marker in parts:
            index = parts.index(marker) + 1
            if index < len(parts):
                return parts[index]
    return None


def _family_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isalnum())


def _family_tests(changed_file: str, test_candidates: set[str]) -> set[str]:
    family = _family_from_path(changed_file)
    if not family:
        return set()
    key = _family_key(family)
    if not key:
        return set()
    return {test for test in test_candidates if key in _family_key(test)}


def load_policy(path: Path) -> dict:
    data = read_json(path)
    fields = {
        "always_tests", "frozen_patterns", "documentation_patterns",
        "guarded_patterns", "full_suite_patterns",
    }
    if set(data) != fields | {"version"} or type(data["version"]) is not int or data["version"] != 2:
        raise ValueError(f"Invalid CI policy schema in {path}; expected version 2 and safety policy only")
    for field in fields:
        validate_paths(data[field], field=f"{path}:{field}", allow_empty=field != "always_tests")
    return data


def _validate_policy_tests(policy: Mapping, repo_root: Path, test_candidates: set[str]) -> None:
    declared = set(policy["always_tests"])
    missing = sorted(test for test in declared if test not in test_candidates or not (repo_root / test).is_file())
    if missing:
        raise ValueError("CI policy references missing/non-test paths: " + ", ".join(missing))


def select_for_changes(
    changed_files: Sequence[str] | None,
    *,
    repo_root: Path = REPO_ROOT,
    policy_path: Path | None = None,
    force_all: bool = False,
) -> Selection:
    policy_path = policy_path or repo_root / POLICY_RELATIVE
    policy = load_policy(policy_path)
    candidates = _test_candidates(repo_root)
    _validate_policy_tests(policy, repo_root, candidates)
    rules = load_manifests(repo_root, SUITE_RELATIVE, candidates)
    changed = tuple(sorted({_normalized(path) for path in changed_files or () if str(path).strip()}))
    frozen = tuple(path for path in changed if _matches(path, policy["frozen_patterns"]))
    if frozen:
        joined = "\n  - ".join(frozen)
        raise FrozenPathError(
            "Changes under frozen legacy directories are not allowed:\n"
            f"  - {joined}\n"
            "Migrate the change to xhmodel_merak/, examples_merak/, or configs_merak/."
        )

    # Both incremental and full selection have the same hard execution
    # boundary. Declarations/imports/family inference cannot enroll tests from
    # tests/ or another benchmark lane into this CI lane.
    default_suite = _default_suite(repo_root)
    selected = set(policy["always_tests"])
    reasons: dict[str, set[str]] = defaultdict(set)
    for test in selected:
        reasons[test].add("always:ci-impact-policy")

    full_reasons: set[str] = set()

    def select_all(reason: str) -> None:
        full_reasons.add(reason)
        selected.update(default_suite)
        for test in default_suite:
            reasons[test].add(reason)

    if force_all:
        select_all("forced-full-suite")
    elif changed_files is None:
        select_all("unknown-diff-full-suite")

    reverse_graph = None
    uncovered: list[str] = []

    for changed_file in changed:
        covered = False
        matched_explicit_rule = False
        if changed_file in candidates:
            selected.add(changed_file)
            reasons[changed_file].add(f"changed-test:{changed_file}")
            covered = True
        elif _looks_like_test_path(changed_file):
            # A deleted test is absent from the current candidate set.  Its
            # removal must remain visible in the diff, but it does not need a
            # replacement impact mapping merely to satisfy fail-closed.
            covered = True

        owner = manifest_test(changed_file, SUITE_RELATIVE)
        if owner is not None:
            # Ownership is immutable and encoded in the filename, not in JSON.
            # Edits/removals always select the surviving owner, and a rename
            # selects both surviving owners when both paths are in the diff.
            # This does not need a base revision or an old manifest snapshot.
            covered = True
            if owner in candidates:
                selected.add(owner)
                reasons[owner].add(f"changed-manifest:{changed_file}")

        if _matches(changed_file, policy["documentation_patterns"]):
            covered = True

        if changed_file == POLICY_RELATIVE or _matches(changed_file, policy["full_suite_patterns"]):
            covered = True
            select_all(f"full-suite:{changed_file}")

        for rule in rules:
            if not _matches(changed_file, rule.patterns):
                continue
            covered = True
            matched_explicit_rule = True
            selected.add(rule.test)
            reasons[rule.test].add(f"rule:{rule.id}:{changed_file}")

        # A maintained rule is the authoritative boundary for its subsystem.
        # Import reachability is intentionally a fallback: shared helpers and
        # compatibility imports otherwise pull unrelated model families into a
        # focused change (for example Qwen3.5 -> Ling/Qwen3Next).
        if not matched_explicit_rule and changed_file.endswith(".py"):
            if reverse_graph is None:
                reverse_graph, _ = _local_import_graph(repo_root)
            for test in _dependent_tests(changed_file, reverse_graph, candidates):
                covered = True
                selected.add(test)
                reasons[test].add(f"python-import:{changed_file}")

        # Family co-location is the fallback for dynamic/config-driven code.
        # Explicit rules and import edges are more precise and must win when
        # available; otherwise a one-line adapter change would run an entire
        # model family's export/runtime suite.
        if not covered:
            for test in _family_tests(changed_file, candidates):
                covered = True
                selected.add(test)
                reasons[test].add(f"model-family:{changed_file}")

        if _matches(changed_file, policy["guarded_patterns"]) and not covered:
            uncovered.append(changed_file)

    result = Selection(
        changed,
        tuple(sorted(selected)),
        {key: tuple(sorted(value)) for key, value in reasons.items()},
        tuple(sorted(uncovered)),
        bool(full_reasons),
        tuple(sorted(full_reasons)),
    )
    if result.uncovered_files:
        joined = "\n  - ".join(result.uncovered_files)
        raise MissingCoverageError(
            "Guarded source/config changes have no affected CI test:\n"
            f"  - {joined}\n"
            "Add a focused test and its adjacent <test-stem>.impact.json declaration in the same change."
        )
    return result


def _run_git(repo_root: Path, *args: str) -> list[str]:
    process = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Git's human-readable output quotes non-ASCII paths according to
    # core.quotePath.  NUL-delimited output is unambiguous for every valid
    # repository filename, including spaces, newlines, and Chinese names.
    return [_normalized(os.fsdecode(item)) for item in process.stdout.split(b"\0") if item]


def discover_changed_files(repo_root: Path, base_ref: str | None, head_ref: str) -> list[str] | None:
    base_ref = base_ref or os.environ.get("CI_TEST_BASE_REF") or os.environ.get("CI_MERGE_REQUEST_DIFF_BASE_SHA")
    head_ref = os.environ.get("CI_TEST_HEAD_REF", head_ref)
    if base_ref:
        return _run_git(
            repo_root,
            "diff",
            "--name-only",
            "-z",
            "--no-renames",
            "--diff-filter=ACDMRTUXB",
            f"{base_ref}...{head_ref}",
        )

    # An exported CI source tree can have no Git metadata. It must use the
    # runner's path list or conservatively select the full registered suite.
    try:
        _run_git(repo_root, "rev-parse", "--verify", "HEAD")
    except subprocess.CalledProcessError:
        return None
    dirty = _run_git(repo_root, "diff", "--name-only", "-z", "--no-renames", "--diff-filter=ACDMRTUXB", "HEAD")
    untracked = _run_git(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    if dirty or untracked:
        return sorted(set(dirty + untracked))

    try:
        return _run_git(
            repo_root, "diff", "--name-only", "-z", "--no-renames", "--diff-filter=ACDMRTUXB", "HEAD^", "HEAD",
        )
    except subprocess.CalledProcessError:
        return None


def _report(selection: Selection, *, detailed: bool = True) -> str:
    lines = ["CI impact selection:"]
    mode = "full" if selection.full_suite else "incremental"
    lines.append(f"  mode: {mode}; {len(selection.changed_files)} changed paths; {len(selection.tests)} test files")
    for reason in selection.full_suite_reasons:
        lines.append(f"  full-suite reason: {reason}")
    if detailed:
        lines.append("  changed: " + (", ".join(selection.changed_files) if selection.changed_files else "<none>"))
    for test in selection.tests:
        lines.append(f"  test: {test}")
        test_reasons = selection.reasons.get(test, ())
        for reason in test_reasons if detailed else test_reasons[:3]:
            lines.append(f"    <- {reason}")
        if not detailed and len(test_reasons) > 3:
            lines.append(f"    ... {len(test_reasons) - 3} more reasons (use --format json/report)")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--policy", type=Path)
    parser.add_argument("--base-ref")
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--changed-files-file", type=Path)
    parser.add_argument("--all", action="store_true", dest="force_all")
    parser.add_argument("--format", choices=("paths", "json", "report"), default="report")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    try:
        # An explicitly supplied empty list is a known empty diff, not a
        # request to inspect some unrelated local/last-commit changes.
        changed = list(args.changed_file)
        if args.changed_files_file is not None:
            changed.extend(args.changed_files_file.read_text(encoding="utf-8").splitlines())
        elif not changed:
            changed = discover_changed_files(repo_root, args.base_ref, args.head_ref)
        selection = select_for_changes(
            changed,
            repo_root=repo_root,
            policy_path=args.policy.resolve() if args.policy else None,
            force_all=args.force_all,
        )
    except (FrozenPathError, MissingCoverageError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"ci-impact error: {exc}", file=sys.stderr)
        return 2

    if args.format == "paths":
        print("\n".join(selection.tests))
        print(_report(selection, detailed=False), file=sys.stderr)
    elif args.format == "json":
        print(json.dumps({
            "changed_files": selection.changed_files,
            "tests": selection.tests,
            "reasons": selection.reasons,
            "full_suite": selection.full_suite,
            "full_suite_reasons": selection.full_suite_reasons,
        }, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(_report(selection))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
