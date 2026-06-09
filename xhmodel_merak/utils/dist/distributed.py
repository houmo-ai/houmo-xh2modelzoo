from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

from .config import ParallelConfig
from .model_parallel import initialize_model_parallel


@dataclass(frozen=True)
class WorkerContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device


def init_worker_distributed_environment(
    cfg: ParallelConfig,
    *,
    rank: int,
    local_rank: int,
) -> WorkerContext:
    """vLLM-like worker init:

    1. bind device
    2. init torch.distributed
    3. initialize model parallel groups
    """
    cfg.validate()

    if cfg.device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("cfg.device_type='cuda' but CUDA is not available")
        if local_rank >= torch.cuda.device_count():
            raise RuntimeError(
                f"local_rank={local_rank} but only {torch.cuda.device_count()} CUDA devices exist"
            )
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        backend = cfg.torch_dist_backend or "nccl"
    else:
        device = torch.device("cpu")
        backend = cfg.torch_dist_backend or "gloo"

    init_method = f"tcp://{cfg.master_addr}:{cfg.master_port}"

    # Helpful for code that expects torchrun-like env vars.
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(cfg.world_size)
    os.environ["MASTER_ADDR"] = cfg.master_addr
    os.environ["MASTER_PORT"] = str(cfg.master_port)

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        rank=rank,
        world_size=cfg.world_size,
    )

    initialize_model_parallel(tensor_parallel_size=cfg.tensor_parallel_size)

    return WorkerContext(
        rank=rank,
        local_rank=local_rank,
        world_size=cfg.world_size,
        device=device,
    )


def destroy_worker_distributed_environment() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()
