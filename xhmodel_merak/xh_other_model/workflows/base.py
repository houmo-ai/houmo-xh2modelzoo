from collections.abc import Mapping
from typing import Any

from .config import WorkflowConfig
from .result import ExportResult, QuantResult


class BaseOtherModelWorkflow:
    def __init__(
        self,
        model_dir: str,
        config_path: str,
        seed: int = 1024,
        debug: bool = False,
    ):
        if not model_dir:
            raise ValueError("model_dir must be provided!")
        self.workflow_config = WorkflowConfig.from_file(config_path)
        self.model_dir = model_dir
        self.seed = seed
        self.debug = debug

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        raise NotImplementedError(f"{type(self).__name__}.quant() must be implemented")

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        raise NotImplementedError(f"{type(self).__name__}.export() must be implemented")

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any,
    ) -> str:
        raise NotImplementedError(f"{type(self).__name__}.dump_golden() must be implemented")


__all__ = ["BaseOtherModelWorkflow"]
