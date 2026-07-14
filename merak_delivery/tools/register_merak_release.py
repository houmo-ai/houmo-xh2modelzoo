#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from merak_model_flow import main


if __name__ == "__main__":
    raise SystemExit(main(["register-release", *sys.argv[1:]]))
