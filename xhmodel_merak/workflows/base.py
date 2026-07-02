from typing import Any, Protocol


class WorkflowProtocol(Protocol):
    def quant(self, output_dir: str, device: str, config_overrides: Any = None) -> Any:
        ...

    def export(self, quant_result: Any, output_dir: str, device: str, config_overrides: Any = None) -> Any:
        ...

    def dump_golden(self, export_result: Any, device: str, input_messages: Any) -> str:
        ...


__all__ = ["WorkflowProtocol"]
