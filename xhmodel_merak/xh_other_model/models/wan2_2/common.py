# Copyright 2026 HOUMO AI
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path


def ensure_wan2_2_repo(repo_root: str | Path = "/data01/home/xuchen/g_video/Wan2.2-main") -> Path:
    repo_root = Path(repo_root)
    repo_str = str(repo_root)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_root
