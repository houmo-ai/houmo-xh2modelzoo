#!/usr/bin/env python3
import json
from pathlib import Path

ABS_PREFIX = "test/"
REL_PREFIX = "data/calib_data/test/"

def rewrite(obj):
    changed = False
    if isinstance(obj, dict):
        if obj.get("type") == "image":
            val = obj.get("value")
            if isinstance(val, str) and val.startswith(ABS_PREFIX):
                obj["value"] = REL_PREFIX + val[len(ABS_PREFIX):].lstrip("/")
                changed = True
        for v in obj.values():
            if rewrite(v):
                changed = True
    elif isinstance(obj, list):
        for item in obj:
            if rewrite(item):
                changed = True
    return changed

def process_file(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if rewrite(data):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"updated {path}")
    else:
        print(f"no changes {path}")

def main():
    base = Path("data/calib_data")
    for json_path in base.rglob("*.json"):
        process_file(json_path)

if __name__ == "__main__":
    main()
