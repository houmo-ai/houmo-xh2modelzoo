from typing import Any


class AutoOtherModelWorkflow:
    @classmethod
    def from_config(
        cls,
        hf_model_dir: str,
        config_path: str,
        seed: int = 1024,
        debug: bool = False,
    ) -> Any:
        raise NotImplementedError(
            "xh_other_model auto workflow binding is not implemented yet; "
            "add an xh_other_model registry before routing non-LLM models."
        )


__all__ = ["AutoOtherModelWorkflow"]
