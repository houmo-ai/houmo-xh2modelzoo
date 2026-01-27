from pathlib import Path


HEADER_TEMPLATE = """# Copyright 2025 HOUMO AI
#
# File: {filename}
# Description:
#   Example script: {relpath}
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""


def has_license_header(text: str) -> bool:
    """Return True if the file already appears to have a license header."""
    # Only look at the first ~40 lines to decide.
    lines = text.splitlines()
    head = "\n".join(lines[:40])

    if "SPDX-License-Identifier:" in head:
        return True
    if "Licensed under the Apache License" in head:
        return True
    if "Copyright" in head and "HOUMO AI" in head:
        return True

    return False


def add_header_to_file(path: Path, relpath: Path) -> bool:
    """Add the standard HOUMO AI header to one file if missing.

    Returns True if the file was modified, False otherwise.
    """
    original = path.read_text(encoding="utf-8")

    if has_license_header(original):
        return False

    header = HEADER_TEMPLATE.format(
        filename=path.name,
        relpath=str(relpath),
    )
    path.write_text(header + original, encoding="utf-8")
    return True


def main() -> None:
    # This script lives inside the `examples/` directory, so its parent is the
    # examples root.
    examples_root = Path(__file__).resolve().parent

    modified = 0
    total = 0

    for py_path in sorted(examples_root.rglob("*.py")):
        # Skip this utility script itself.
        if py_path.name == Path(__file__).name:
            continue

        total += 1
        relpath = py_path.relative_to(examples_root)
        changed = add_header_to_file(py_path, relpath)
        if changed:
            modified += 1
            print(f"Added header to: {py_path}")

    print(f"Processed {total} Python files; added headers to {modified} files.")


if __name__ == "__main__":
    main()

