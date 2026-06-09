from __future__ import annotations

from typing import Any, Callable

from .config import ParallelConfig
from .distributed import (
    WorkerContext,
    destroy_worker_distributed_environment,
    init_worker_distributed_environment,
)

WorkerFn = Callable[..., Any]


def worker_main(
    rank: int,
    local_rank: int,
    cfg: ParallelConfig,
    worker_fn: WorkerFn,
    worker_args: tuple[Any, ...],
    worker_kwargs: dict[str, Any],
) -> None:
    ctx = init_worker_distributed_environment(cfg, rank=rank, local_rank=local_rank)
    try:
        worker_fn(ctx, *worker_args, **worker_kwargs)
    finally:
        destroy_worker_distributed_environment()
