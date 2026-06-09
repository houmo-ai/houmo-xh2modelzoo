from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


Backend = Literal["mp", "ray"]
DeviceType = Literal["cuda", "cpu"]


@dataclass(frozen=True)
class ParallelConfig:
    """Minimal vLLM-like parallel config.

    world_size = tensor_parallel_size
    This mini package currently supports TP only.
    """

    tensor_parallel_size: int = 1
    distributed_executor_backend: Backend = "mp"
    master_addr: str = "127.0.0.1"
    master_port: Optional[int] = None
    torch_dist_backend: Optional[str] = None  # default: nccl for cuda, gloo for cpu
    device_type: DeviceType = "cuda"

    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size

    def validate(self) -> None:
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be > 0") 
        if self.distributed_executor_backend not in ("mp", "ray"):
            raise ValueError("distributed_executor_backend must be 'mp' or 'ray'")
        if self.device_type not in ("cuda", "cpu"):
            raise ValueError("device_type must be 'cuda' or 'cpu'")
