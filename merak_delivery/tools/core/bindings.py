from __future__ import annotations

from types import SimpleNamespace
from typing import Any


_services: SimpleNamespace | None = None


def configure(**services: Any) -> None:
    global _services
    _services = SimpleNamespace(**services)


def get_services() -> SimpleNamespace:
    if _services is None:
        raise RuntimeError("Merak delivery core services have not been configured")
    return _services
