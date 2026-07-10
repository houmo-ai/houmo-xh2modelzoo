import copy
import importlib

from xhmodel_merak.xh_other_model.builder import get_model_class

from .base import BaseOtherModelWorkflow
from .config import WorkflowConfig

class AutoOtherModelWorkflow:
    @classmethod
    def from_config(
        cls,
        model_dir: str | None = None,
        config_path: str | None = None,
        seed: int = 1024,
        debug: bool = False,
    ) -> BaseOtherModelWorkflow:
        if config_path is None:
            raise ValueError("config_path must be provided")
        if model_dir is None:
            raise ValueError("model_dir must be provided")
        workflow_config = WorkflowConfig.from_file(config_path)
        model_cfg = copy.deepcopy(workflow_config.export["model"])
        model_cls = get_model_class(model_cfg)
        workflow_cls = cls._get_workflow_class(model_cls)
        return workflow_cls(
            model_dir=model_dir,
            config_path=config_path,
            seed=seed,
            debug=debug,
        )

    @staticmethod
    def _get_workflow_class(model_cls: type) -> type[BaseOtherModelWorkflow]:
        workflow_cls_path = getattr(model_cls, "WORKFLOW_CLS", None)
        if workflow_cls_path is None:
            return BaseOtherModelWorkflow
        if not isinstance(workflow_cls_path, str) or ":" not in workflow_cls_path:
            raise ValueError(
                f"{model_cls.__name__}.WORKFLOW_CLS must be a 'module:ClassName' string, "
                f"got {workflow_cls_path!r}"
            )
        module_name, class_name = workflow_cls_path.split(":", 1)
        module = importlib.import_module(module_name)
        workflow_cls = getattr(module, class_name)
        if not issubclass(workflow_cls, BaseOtherModelWorkflow):
            raise TypeError(f"{workflow_cls_path} must inherit BaseOtherModelWorkflow")
        return workflow_cls


__all__ = ["AutoOtherModelWorkflow"]
