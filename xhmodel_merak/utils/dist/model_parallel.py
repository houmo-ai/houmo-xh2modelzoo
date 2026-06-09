from __future__ import annotations

import os
from typing import Optional

import torch.distributed as dist


_TP_GROUP: Optional[dist.ProcessGroup] = None
_TP_RANK: Optional[int] = None
_TP_WORLD_SIZE: Optional[int] = None

def initialize_model_parallel(tensor_parallel_size: int) -> None:
    """Create a tensor-parallel group layout.

    This implementation supports TP only.
    """
    global _TP_GROUP, _TP_RANK, _TP_WORLD_SIZE

    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized first")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    expected_world_size = tensor_parallel_size
    if world_size != expected_world_size:
        raise ValueError(f"world_size={world_size} must equal tp_size={tensor_parallel_size}")
 
    # TP-only mode: TP group spans the whole world.
    tp_group = dist.group.WORLD

    _TP_GROUP = tp_group
    _TP_RANK = rank
    _TP_WORLD_SIZE = world_size

    # Torchrun-like env exposure for downstream code that reads rank info from env.
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)

def get_tp_group() -> dist.ProcessGroup:
    if _TP_GROUP is None:
        raise RuntimeError("TP group is not initialized")
    return _TP_GROUP


def get_tensor_model_parallel_rank() -> int:
    if _TP_RANK is None:
        raise RuntimeError("TP rank is not initialized")
    return _TP_RANK


def get_tensor_model_parallel_world_size() -> int:
    if _TP_WORLD_SIZE is None:
        raise RuntimeError("TP world size is not initialized")
    return _TP_WORLD_SIZE
