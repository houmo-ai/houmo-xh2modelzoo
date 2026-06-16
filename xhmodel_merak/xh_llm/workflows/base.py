import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import WorkflowConfig
from .result import ExportResult, QuantResult
from .utils import same_abs_path


class BaseHMONNXWorkflow:
    expected_model_config_cls_name: str | None = None
    expected_model_cls_name: str | None = None

    def __init__(
        self,
        hf_model_dir: str,
        config_path: str,
        seed: int = 1024,
        debug: bool = False,
    ):
        if not hf_model_dir:
            raise ValueError("hf_model_dir must be provided!")
        self.workflow_config = WorkflowConfig.from_file(config_path)
        self.hf_model_dir = os.path.abspath(os.path.normpath(str(hf_model_dir)))
        self.seed = seed
        self.debug = debug

    def quant(
        self,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> QuantResult:
        workflow_config = self.workflow_config.with_overrides(config_overrides)
        if workflow_config.quant is not None:
            raise NotImplementedError(f"{type(self).__name__}.quant() must implement model-specific quantization")
        return QuantResult(hf_model_dir=self.hf_model_dir, skipped=True)

    def export(
        self,
        quant_result: QuantResult,
        output_dir: str,
        device: str,
        config_overrides: Mapping[str, Any] | None = None,
    ) -> ExportResult:
        """
        绝大多数情况可复用基类export()方法导出HMONNX
        特殊情况例如mineru2_5需要导出多个vit，则由子类自己实现
        """
        import torch

        from xhmodel_merak.xh_llm import AutoLLMConfig, AutoLLMModel
        from xhquant.api import Config, get_xhquant_logger, set_random_seed, xhquant_init
        from xhquant.utils import MemoryTracker, TimeProfiler

        if quant_result is None:
            raise ValueError("quant_result can't be None")

        workflow_config = self.workflow_config.with_overrides(config_overrides)
        export_hf_model_dir = self._resolve_export_hf_model_dir(quant_result)
        export_cfg = workflow_config.build_export_dict(export_hf_model_dir)

        work_dir_path = Path(output_dir)
        if work_dir_path.exists():
            raise FileExistsError(f"{str(work_dir_path)} already exists!")
        work_dir_path.mkdir(parents=True)

        xhquant_init(str(work_dir_path / "export_hmonnx.log"), self.debug)
        set_random_seed(self.seed)
        logger = get_xhquant_logger()

        config_file = str(work_dir_path / f"{workflow_config.name}.yaml")
        workflow_config.dump(config_file)
        cfg = Config(export_cfg)

        logger.info(f"Workflow config: {workflow_config.name}")
        logger.info(f"Using device: {device}, cuda_available: {torch.cuda.is_available()}")
        logger.info(f"Config:\n{cfg.pretty_text}")

        model_cfg = AutoLLMConfig.from_pretrained(cfg.model)
        logger.info(f"Resolved model config type: {type(model_cfg).__name__}")
        if (
            self.expected_model_config_cls_name is not None
            and type(model_cfg).__name__ != self.expected_model_config_cls_name
        ):
            raise TypeError(
                f"Expected model config type {self.expected_model_config_cls_name}, "
                f"but got {type(model_cfg).__name__}"
            )
        logger.info(f"Model Config:\n{model_cfg.to_json_string()}")

        xh_model = AutoLLMModel.from_pretrained(config=model_cfg)
        logger.info(f"Resolved model type: {type(xh_model).__name__}")
        if self.expected_model_cls_name is not None and type(xh_model).__name__ != self.expected_model_cls_name:
            raise TypeError(f"Expected model type {self.expected_model_cls_name}, but got {type(xh_model).__name__}")
        # TODO work_dir是否必要？
        if hasattr(xh_model, "work_dir"):
            xh_model.work_dir = str(work_dir_path)

        with TimeProfiler("convert", logger), MemoryTracker(device, "convert2hmonnx", logger):
            meta = xh_model.export_hmonnx(str(work_dir_path))
        return ExportResult(work_dir=str(work_dir_path), config_file=config_file, meta=meta)

    def dump_golden(
        self,
        export_result: ExportResult,
        device: str,
        input_messages: Any,
    ) -> str:
        raise NotImplementedError(f"{type(self).__name__}.dump_golden() must be implemented")

    def _resolve_export_hf_model_dir(self, quant_result: QuantResult) -> str:
        if not same_abs_path(quant_result.hf_model_dir, self.hf_model_dir):
            raise ValueError("QuantResult.hf_model_dir must be the same as self.hf_model_dir")
        if quant_result.skipped:
            return self.hf_model_dir
        if not quant_result.quanted_model_dir:
            raise ValueError("QuantResult.quanted_model_dir must be provided when quant is not skipped")
        return quant_result.quanted_model_dir

    @staticmethod
    def _find_golden_meta_file(export_result: ExportResult) -> str:
        work_dir = Path(export_result.work_dir)
        if not work_dir.is_dir():
            raise FileNotFoundError(f"Export work_dir does not exist or is not a directory: {export_result.work_dir}")

        meta_files = []
        for path in work_dir.iterdir():
            if not path.is_dir() or not path.name.startswith("hmquant"):
                continue
            meta_file = path / "golden_meta_info.json"
            if meta_file.is_file():
                meta_files.append(meta_file)

        if not meta_files:
            raise FileNotFoundError(f"No golden_meta_info.json found under hmquant* directories in {work_dir}")
        if len(meta_files) > 1:
            meta_file_list = ", ".join(str(path) for path in meta_files)
            raise ValueError(f"Found multiple golden_meta_info.json files under {work_dir}: {meta_file_list}")
        return str(meta_files[0])
