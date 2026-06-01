from pathlib import Path

from transformers import AutoProcessor


class GlmOcrProcessor:
    @staticmethod
    def _resolve_local_path(pretrained_model_name_or_path: str) -> str:
        raw_path = Path(pretrained_model_name_or_path).expanduser()
        if raw_path.is_absolute() and raw_path.exists():
            return str(raw_path)

        cwd_path = Path.cwd() / raw_path
        if cwd_path.exists():
            return str(cwd_path.resolve())

        repo_root = Path(__file__).resolve().parents[3]
        repo_root_path = repo_root / raw_path
        if repo_root_path.exists():
            return str(repo_root_path.resolve())

        return pretrained_model_name_or_path

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        resolved_path = cls._resolve_local_path(pretrained_model_name_or_path)
        kwargs.setdefault("trust_remote_code", True)
        return AutoProcessor.from_pretrained(resolved_path, **kwargs)
