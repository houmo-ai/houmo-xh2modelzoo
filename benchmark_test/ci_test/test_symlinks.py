import os
from pathlib import Path
import fnmatch
import pytest


def get_gitignore_patterns(repo_root: Path):
    gitignore_path = repo_root / ".gitignore"
    patterns = []
    if gitignore_path.exists():
        with open(gitignore_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    patterns.append(line)
    return patterns


def matches_gitignore(path: Path, patterns: list, repo_root: Path) -> bool:
    rel_path = path.relative_to(repo_root)
    path_str = str(rel_path)
    for pattern in patterns:
        if pattern.startswith('/'):
            if path_str == pattern.lstrip('/') or path_str.startswith(pattern.lstrip('/')):
                return True
        elif '/' in pattern:
            if pattern.endswith('/'):
                if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path_str, pattern.rstrip('/')):
                    return True
            else:
                if fnmatch.fnmatch(path_str, pattern):
                    return True
        else:
            if rel_path.name == pattern or fnmatch.fnmatch(path_str, f"**/{pattern}"):
                return True
    return False


def find_symlinks_not_in_gitignore(repo_root: Path):
    symlinks = []
    gitignore_patterns = get_gitignore_patterns(repo_root)

    skipped_top_level_dirs = {'models', 'data', 'third_party', 'golden', 'runs', 'wandb', 'logs'}

    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)

        if '.git' in root_path.parts:
            continue

        for name in dirs + files:
            path = root_path / name
            rel_parts = path.relative_to(repo_root).parts

            if rel_parts == ('.git',):
                continue

            if path.is_symlink():
                if len(rel_parts) > 1 and rel_parts[0] in skipped_top_level_dirs:
                    continue
                if not matches_gitignore(path, gitignore_patterns, repo_root):
                    symlinks.append(path)

    return symlinks


def test_symlinks_not_in_gitignore():
    repo_root = Path(__file__).parent.parent.parent.resolve()

    symlinks = find_symlinks_not_in_gitignore(repo_root)

    msg = f"发现 {len(symlinks)} 个软链接不在 .gitignore 中:\n"
    for link in symlinks:
        msg += f"  - {link.relative_to(repo_root)}\n"

    if symlinks:
        pytest.fail(msg)


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent.parent.resolve()
    print(f"检查仓库: {repo_root}")

    symlinks = find_symlinks_not_in_gitignore(repo_root)

    if symlinks:
        print(f"\n发现 {len(symlinks)} 个软链接不在 .gitignore 中:")
        for link in symlinks:
            print(f"  - {link.relative_to(repo_root)}")
    else:
        print("\n✅ 未发现违规软链接")
