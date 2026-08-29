"""Pytest bootstrap for the active ModelZoo CI lane."""

from __future__ import annotations

import os
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _prepend_pythonpath(path: Path) -> None:
    text = str(path)
    if path.exists() and text not in sys.path:
        sys.path.insert(0, text)


GPTQMODEL_SOURCE_DIR = os.environ.get("GPTQMODEL_SOURCE_DIR")
if GPTQMODEL_SOURCE_DIR:
    gptqmodel_root = Path(GPTQMODEL_SOURCE_DIR).expanduser().resolve()
    _prepend_pythonpath(gptqmodel_root)
    _prepend_pythonpath(gptqmodel_root / "third_party" / "auto-round")
