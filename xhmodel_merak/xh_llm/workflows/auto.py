import importlib
import os

from xhmodel_merak.xh_llm.builder import get_model_class

from .base import BaseHMONNXWorkflow
from .config import WorkflowConfig


class AutoLLMWorkflow:
    @classmethod
    def from_config(
        cls,
        hf_model_dir: str,
        config_path: str,
        seed: int = 1024,
        debug: bool = False,
    ) -> BaseHMONNXWorkflow:
        workflow_config = WorkflowConfig.from_file(config_path)
        export_cfg = workflow_config.build_export_dict(os.path.abspath(os.path.normpath(str(hf_model_dir))))
        model_cls = get_model_class(export_cfg["model"])
        workflow_cls = cls._get_workflow_class(model_cls)
        return workflow_cls(
            hf_model_dir=hf_model_dir,
            config_path=config_path,
            seed=seed,
            debug=debug,
        )

    @staticmethod
    def _get_workflow_class(model_cls: type) -> type[BaseHMONNXWorkflow]:
        workflow_cls_path = getattr(model_cls, "WORKFLOW_CLS", None)
        if workflow_cls_path is None:
            return BaseHMONNXWorkflow
        if not isinstance(workflow_cls_path, str) or ":" not in workflow_cls_path:
            raise ValueError(
                f"{model_cls.__name__}.WORKFLOW_CLS must be a 'module:ClassName' string, "
                f"got {workflow_cls_path!r}"
            )
        module_name, class_name = workflow_cls_path.split(":", 1)
        module = importlib.import_module(module_name)
        workflow_cls = getattr(module, class_name)
        if not issubclass(workflow_cls, BaseHMONNXWorkflow):
            raise TypeError(f"{workflow_cls_path} must inherit BaseHMONNXWorkflow")
        return workflow_cls


__all__ = ["AutoLLMWorkflow"]
