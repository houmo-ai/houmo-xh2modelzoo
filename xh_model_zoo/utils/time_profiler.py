# Copyright 2025 HOUMO AI
#
# File: time_profiler.py
# Description:
#   Time profiling utility for performance measurement.
#   This module provides the TimeProfiler context manager for measuring
#   execution time of code blocks.
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
from contextlib import AbstractContextManager, contextmanager
from time import perf_counter
from typing import Callable, Generator


class TimeProfiler(AbstractContextManager):
    def __init__(self, name: str, logger=None):
        self.name = name
        self.logger = logger

    def __enter__(self):
        self.start = perf_counter()
        return self

    def __exit__(self, type, value, traceback):
        self.time = perf_counter() - self.start

        # seconds = self.time
        # m, s = divmod(seconds, 60)
        # h, m = divmod(m, 60)
        self.readout = f"{self.name} time: {self.time:.4f} s"
        if self.logger is not None:
            self.logger.info(self.readout)
        else:
            print(self.readout)


from contextlib import contextmanager
from time import perf_counter


@contextmanager
def time_profiler() -> Generator[Callable[[], float], None, None]:
    start = perf_counter()
    yield lambda: perf_counter() - start
