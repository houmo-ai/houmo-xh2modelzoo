from __future__ import annotations

import socket
from contextlib import closing


def find_free_port(host: str = "127.0.0.1") -> int:
    """Find a currently free TCP port for single-node demos."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])
