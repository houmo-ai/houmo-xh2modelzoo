from __future__ import annotations

import argparse
from collections.abc import Callable

from .bindings import get_services


class MerakModelFlowCLI:
    """Build and dispatch the backward-compatible Merak delivery CLI."""

    def __init__(self, parser_factory: Callable[[], argparse.ArgumentParser] | None = None) -> None:
        self._parser_factory = parser_factory

    def build_parser(self) -> argparse.ArgumentParser:
        parser_factory = self._parser_factory or get_services().build_parser
        return parser_factory()

    def run(self, argv: list[str] | None = None) -> int:
        args = self.build_parser().parse_args(argv)
        return args.func(args)
