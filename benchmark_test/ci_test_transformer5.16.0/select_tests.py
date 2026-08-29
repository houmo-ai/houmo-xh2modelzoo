#!/usr/bin/env python3
"""Select CI tests affected by a set of changed files.

Selection is deterministic and auditable:

1. reject changes under frozen legacy directories;
2. apply explicit high-risk rules from ``impact_rules.json``;
3. follow reverse Python-import dependencies when no explicit rule owns the file;
4. use model-family co-location for existing ``tests/<family>`` suites;
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


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
SUITE_RELATIVE = SCRIPT_DIR.relative_to(REPO_ROOT).as_posix()
DEFAULT_RULES = SCRIPT_DIR / "impact_rules.json"
SOURCE_ROOTS = ("xhmodel_merak", "examples_merak", "benchmark_test", "tests")


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
    if not (
        path.startswith(f"{SUITE_RELATIVE}/")
        or path.startswith("tests/")
    ):
        return False
    return candidate.name.startswith("test_") or candidate.name.endswith("_test.py")


@functools.lru_cache(maxsize=4)
def _test_candidates(repo_root: Path) -> set[str]:
    candidates: set[str] = set()
    for root_name in (SUITE_RELATIVE, "tests"):
        root = repo_root / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.py"):
            name = path.name
            if name.startswith("test_") or name.endswith("_test.py"):
                candidates.add(path.relative_to(repo_root).as_posix())
    return candidates


@functools.lru_cache(maxsize=4)
def _default_suite(repo_root: Path) -> set[str]:
    root = repo_root / SUITE_RELATIVE
    return {
        path.relative_to(repo_root).as_posix()
        for path in root.glob("*.py")
        if path.is_file()
        and (path.name.startswith("test_") or path.name.endswith("_test.py"))
    }


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


def load_rules(path: Path = DEFAULT_RULES) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("version") != 1:
        raise ValueError(f"Unsupported impact rule version in {path}: {data.get('version')!r}")
    return data


def _validate_rule_tests(rules: Mapping, repo_root: Path, test_candidates: set[str]) -> None:
    declared = set(rules.get("always_tests", ()))
    for rule in rules.get("rules", ()):
        declared.update(rule.get("tests", ()))
    missing = sorted(test for test in declared if test not in test_candidates or not (repo_root / test).is_file())
    if missing:
        raise ValueError("Impact rules reference missing/non-test paths: " + ", ".join(missing))


def select_for_changes(
    changed_files: Sequence[str],
    *,
    repo_root: Path = REPO_ROOT,
    rules_path: Path = DEFAULT_RULES,
    force_all: bool = False,
) -> Selection:
    rules = load_rules(rules_path)
    candidates = _test_candidates(repo_root)
    _validate_rule_tests(rules, repo_root, candidates)
    changed = tuple(sorted({_normalized(path) for path in changed_files if _normalized(path)}))
    frozen = tuple(path for path in changed if _matches(path, rules.get("frozen_patterns", ())))
    if frozen:
        joined = "\n  - ".join(frozen)
        raise FrozenPathError(
            "Changes under frozen legacy directories are not allowed:\n"
            f"  - {joined}\n"
            "Migrate the change to xhmodel_merak/, examples_merak/, or configs_merak/."
        )

    default_suite = _default_suite(repo_root)
    selected = set(rules.get("always_tests", ()))
    reasons: dict[str, set[str]] = defaultdict(set)
    for test in selected:
        reasons[test].add("always:ci-impact-policy")

    ran_full_suite = False
    if force_all:
        selected.update(default_suite)
        for test in default_suite:
            reasons[test].add("forced-full-suite")
        ran_full_suite = True
    elif not changed:
        selected.update(default_suite)
        for test in default_suite:
            reasons[test].add("no-diff-full-suite")
        return Selection(
            changed,
            tuple(sorted(selected)),
            {key: tuple(sorted(value)) for key, value in reasons.items()},
            full_suite=True,
        )

    reverse_graph, _ = _local_import_graph(repo_root)
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

        if _matches(changed_file, rules.get("documentation_patterns", ())):
            covered = True

        if _matches(changed_file, rules.get("full_suite_patterns", ())):
            ran_full_suite = True
            covered = True
            selected.update(default_suite)
            for test in default_suite:
                reasons[test].add(f"full-suite:{changed_file}")

        for rule in rules.get("rules", ()):
            if not _matches(changed_file, rule.get("patterns", ())):
                continue
            covered = True
            matched_explicit_rule = True
            for test in rule.get("tests", ()):
                selected.add(test)
                reasons[test].add(f"rule:{rule['id']}:{changed_file}")

        # A maintained rule is the authoritative boundary for its subsystem.
        # Import reachability is intentionally a fallback: shared helpers and
        # compatibility imports otherwise pull unrelated model families into a
        # focused change (for example Qwen3.5 -> Ling/Qwen3Next).
        if not matched_explicit_rule:
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

        if _matches(changed_file, rules.get("guarded_patterns", ())) and not covered:
            uncovered.append(changed_file)

    result = Selection(
        changed,
        tuple(sorted(selected)),
        {key: tuple(sorted(value)) for key, value in reasons.items()},
        tuple(sorted(uncovered)),
        ran_full_suite,
    )
    if result.uncovered_files:
        joined = "\n  - ".join(result.uncovered_files)
        raise MissingCoverageError(
            "Guarded source/config changes have no affected CI test:\n"
            f"  - {joined}\n"
            "Add a focused test and/or an impact_rules.json mapping in the same change."
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


def discover_changed_files(repo_root: Path, base_ref: str | None, head_ref: str) -> list[str]:
    base_ref = base_ref or os.environ.get("CI_TEST_BASE_REF") or os.environ.get("CI_MERGE_REQUEST_DIFF_BASE_SHA")
    head_ref = os.environ.get("CI_TEST_HEAD_REF", head_ref)
    if base_ref:
        return _run_git(
            repo_root,
            "diff",
            "--name-only",
            "-z",
            "--diff-filter=ACDMRTUXB",
            f"{base_ref}...{head_ref}",
        )

    dirty = _run_git(repo_root, "diff", "--name-only", "-z", "--diff-filter=ACDMRTUXB", "HEAD")
    untracked = _run_git(repo_root, "ls-files", "--others", "--exclude-standard", "-z")
    if dirty or untracked:
        return sorted(set(dirty + untracked))

    try:
        return _run_git(repo_root, "diff", "--name-only", "-z", "--diff-filter=ACDMRTUXB", "HEAD^", "HEAD")
    except subprocess.CalledProcessError:
        return []


def _report(selection: Selection) -> str:
    lines = ["CI impact selection:"]
    lines.append("  changed: " + (", ".join(selection.changed_files) if selection.changed_files else "<none>"))
    for test in selection.tests:
        lines.append(f"  test: {test}")
        for reason in selection.reasons.get(test, ()):
            lines.append(f"    <- {reason}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--base-ref")
    parser.add_argument("--head-ref", default="HEAD")
    parser.add_argument("--changed-file", action="append", default=[])
    parser.add_argument("--changed-files-file", type=Path)
    parser.add_argument("--all", action="store_true", dest="force_all")
    parser.add_argument("--format", choices=("paths", "json", "report"), default="report")
    args = parser.parse_args(argv)

    repo_root = args.repo_root.resolve()
    changed = list(args.changed_file)
    if args.changed_files_file:
        changed.extend(args.changed_files_file.read_text(encoding="utf-8").splitlines())
    if not changed:
        changed = discover_changed_files(repo_root, args.base_ref, args.head_ref)

    try:
        selection = select_for_changes(
            changed,
            repo_root=repo_root,
            rules_path=args.rules.resolve(),
            force_all=args.force_all,
        )
    except (FrozenPathError, MissingCoverageError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"ci-impact error: {exc}", file=sys.stderr)
        return 2

    if args.format == "paths":
        print("\n".join(selection.tests))
        print(_report(selection), file=sys.stderr)
    elif args.format == "json":
        print(json.dumps({
            "changed_files": selection.changed_files,
            "tests": selection.tests,
            "reasons": selection.reasons,
            "full_suite": selection.full_suite,
        }, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(_report(selection))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
