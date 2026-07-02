from typing import Any


class AutoWorkflow:
    @classmethod
    def from_config(
        cls,
        hf_model_dir: str,
        config_path: str,
        seed: int = 1024,
        debug: bool = False,
    ) -> Any:
        raise NotImplementedError("AutoWorkflow routing is not implemented yet.")


__all__ = ["AutoWorkflow"]
