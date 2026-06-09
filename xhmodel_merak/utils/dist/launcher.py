from __future__ import annotations

import multiprocessing as py_mp
from typing import Optional

import torch.multiprocessing as mp

from .config import ParallelConfig
from .utils import find_free_port
from .worker import WorkerFn, worker_main


class TPLauncher: 

    def __init__(self, cfg: ParallelConfig):
        cfg.validate()
        if cfg.master_port is None:
            cfg = ParallelConfig(
                tensor_parallel_size=cfg.tensor_parallel_size, 
                distributed_executor_backend=cfg.distributed_executor_backend,
                master_addr=cfg.master_addr,
                master_port=find_free_port(cfg.master_addr),
                torch_dist_backend=cfg.torch_dist_backend,
                device_type=cfg.device_type,
            )
        self.cfg = cfg

    def run(self, worker_fn: WorkerFn, *args, **kwargs) -> None:
        if self.cfg.distributed_executor_backend == "mp":
            self._run_mp(worker_fn, *args, **kwargs) 
        else:
            raise ValueError(f"unsupported backend: {self.cfg.distributed_executor_backend}")

    def _run_mp(self, worker_fn: WorkerFn, *args, **kwargs) -> None:
        # spawn is the safest start method for CUDA.
        try:
            mp.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

        procs = []
        ctx = py_mp.get_context("spawn")
        for local_rank in range(self.cfg.world_size):
            rank = local_rank  # single-node simplification
            p = ctx.Process(
                target=worker_main,
                args=(rank, local_rank, self.cfg, worker_fn, args, kwargs),
                name=f"MiniTPWorker-{rank}",
                daemon=False,
            )
            p.start()
            procs.append(p)

        failed: Optional[int] = None
        for p in procs:
            p.join()
            if p.exitcode != 0 and failed is None:
                failed = p.exitcode

        if failed is not None:
            raise RuntimeError(f"at least one worker failed, first exitcode={failed}")
 