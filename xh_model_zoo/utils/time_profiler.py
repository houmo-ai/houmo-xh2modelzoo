from contextlib import AbstractContextManager
from contextlib import contextmanager
from time import perf_counter
from typing import Callable
from typing import Generator


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
