import os
from collections.abc import Mapping
from typing import Any

from .config import WorkflowConfig
from .result import ExportResult, QuantResult
from .utils import same_abs_path


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
        self.model_dir = os.path.abspath(os.path.normpath(str(model_dir)))
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

    def _resolve_export_model_dir(self, quant_result: QuantResult) -> str:
        if quant_result is None:
            raise ValueError("quant_result can't be None")
        if not same_abs_path(quant_result.raw_model_dir, self.model_dir):
            raise ValueError("QuantResult.raw_model_dir must be the same as self.model_dir")
        if quant_result.skipped:
            return self.model_dir
        if not quant_result.quanted_model_dir:
            raise ValueError("QuantResult.quanted_model_dir must be provided when quant is not skipped")
        return os.path.abspath(os.path.normpath(str(quant_result.quanted_model_dir)))


__all__ = ["BaseOtherModelWorkflow"]
