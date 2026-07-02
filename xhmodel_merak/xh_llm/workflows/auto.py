import copy
import importlib

from xhmodel_merak.xh_llm.builder import get_model_class

from .base import BaseLLMWorkflow
from .config import WorkflowConfig


class AutoLLMWorkflow:
    @classmethod
    def from_config(
        cls,
        model_dir: str,
        config_path: str,
        seed: int = 1024,
        debug: bool = False,
    ) -> BaseLLMWorkflow:
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
    def _get_workflow_class(model_cls: type) -> type[BaseLLMWorkflow]:
        workflow_cls_path = getattr(model_cls, "WORKFLOW_CLS", None)
        if workflow_cls_path is None:
            return BaseLLMWorkflow
        if not isinstance(workflow_cls_path, str) or ":" not in workflow_cls_path:
            raise ValueError(
                f"{model_cls.__name__}.WORKFLOW_CLS must be a 'module:ClassName' string, "
                f"got {workflow_cls_path!r}"
            )
        module_name, class_name = workflow_cls_path.split(":", 1)
        module = importlib.import_module(module_name)
        workflow_cls = getattr(module, class_name)
        if not issubclass(workflow_cls, BaseLLMWorkflow):
            raise TypeError(f"{workflow_cls_path} must inherit BaseLLMWorkflow")
        return workflow_cls


__all__ = ["AutoLLMWorkflow"]
