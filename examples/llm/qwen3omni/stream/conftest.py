# Copyright 2025 HOUMO AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Root conftest for the stream package — ensures package is importable."""

import sys
from pathlib import Path

# Add the stream directory to sys.path so tests can import modules
_stream_dir = str(Path(__file__).resolve().parent)
if _stream_dir not in sys.path:
    sys.path.insert(0, _stream_dir)
