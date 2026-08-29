"""Keep Transformer CI assertions independent from the optional Allure reporter."""

from __future__ import annotations


try:
    import allure as allure
except ModuleNotFoundError as exc:
    if exc.name != "allure":
        raise

    class _AttachmentType:
        TEXT = "text/plain"

    class _AllureFallback:
        attachment_type = _AttachmentType()

        @staticmethod
        def title(_name: str):
            def decorator(function):
                return function

            return decorator

        @staticmethod
        def attach(_body, _name: str, *, attachment_type=None) -> None:
            return None

    allure = _AllureFallback()


__all__ = ["allure"]
