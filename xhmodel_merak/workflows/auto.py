from pathlib import Path
from typing import Any

import yaml


class AutoWorkflow:
    @classmethod
    def from_config(
        cls,
        model_dir: str | None = None,
        config_path: str | None = None,
        seed: int = 1024,
        debug: bool = False,
    ) -> Any:
        if config_path is None:
            raise ValueError("config_path must be provided")
        if model_dir is None:
            raise ValueError("model_dir must be provided")

        model_cfg = cls._read_model_config(config_path)
        model_type = cls._read_model_type(model_cfg, config_path)
        if model_cfg.get("type") == model_type and model_cfg.get("model_type") is None:
            from xhmodel_merak.xh_other_model.workflows import AutoOtherModelWorkflow

            return AutoOtherModelWorkflow.from_config(
                model_dir=model_dir,
                config_path=config_path,
                seed=seed,
                debug=debug,
            )

        supported_families = cls._supported_workflow_families(model_type)

        if len(supported_families) == 1:
            family = supported_families[0]
            if family == "other_model":
                from xhmodel_merak.xh_other_model.workflows import AutoOtherModelWorkflow

                return AutoOtherModelWorkflow.from_config(
                    model_dir=model_dir,
                    config_path=config_path,
                    seed=seed,
                    debug=debug,
                )
            from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

            return AutoLLMWorkflow.from_config(
                model_dir=model_dir,
                config_path=config_path,
                seed=seed,
                debug=debug,
            )

        if not supported_families:
            raise ValueError(f"Unsupported workflow model_type: {model_type}")

        family = cls._resolve_ambiguous_family(model_cfg, model_type)
        if family == "other_model":
            from xhmodel_merak.xh_other_model.workflows import AutoOtherModelWorkflow

            return AutoOtherModelWorkflow.from_config(
                model_dir=model_dir,
                config_path=config_path,
                seed=seed,
                debug=debug,
            )
        from xhmodel_merak.xh_llm.workflows import AutoLLMWorkflow

        return AutoLLMWorkflow.from_config(
            model_dir=model_dir,
            config_path=config_path,
            seed=seed,
            debug=debug,
        )

    @staticmethod
    def _read_model_config(config_path: str) -> dict[str, Any]:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as fin:
            data = yaml.safe_load(fin) or {}
        try:
            model = data["export"]["model"]
        except KeyError as exc:
            raise ValueError(f"workflow config {path} must contain export.model") from exc
        if not isinstance(model, dict) or not model:
            raise ValueError(f"workflow config {path} must contain non-empty export.model")
        return model

    @staticmethod
    def _read_model_type(model_cfg: dict[str, Any], config_path: str) -> str:
        model_type = model_cfg.get("type") or model_cfg.get("model_type")
        if not isinstance(model_type, str) or not model_type:
            raise ValueError(
                f"workflow config {config_path} field export.model.type or export.model.model_type "
                "must be a non-empty string"
            )
        return model_type

    @staticmethod
    def _supported_workflow_families(model_type: str) -> list[str]:
        from xhmodel_merak.xh_other_model.builder import is_model_type_supported
        from xhmodel_merak.xh_llm.builder import is_model_type_supported as is_llm_model_type_supported

        families = []
        if is_model_type_supported(model_type):
            families.append("other_model")
        if is_llm_model_type_supported(model_type):
            families.append("llm")
        return families

    @staticmethod
    def _resolve_ambiguous_family(model_cfg: dict[str, Any], model_type: str) -> str:
        has_other_type = model_cfg.get("type") == model_type
        has_llm_type = model_cfg.get("model_type") == model_type
        if has_other_type and not has_llm_type:
            return "other_model"
        if has_llm_type and not has_other_type:
            return "llm"
        raise ValueError(
            f"Ambiguous workflow model_type {model_type!r}: it is registered by both "
            "xh_other_model and xh_llm. Use only export.model.type for other_model or "
            "only export.model.model_type for xh_llm."
        )


__all__ = ["AutoWorkflow"]
