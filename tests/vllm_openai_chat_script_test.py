import importlib.util
import json
import sys
from pathlib import Path
from urllib import error

import pytest


MODULE_PATH = Path("examples_develop/llm/vllm_openai_chat.py")


def _load_module(monkeypatch):
    spec = importlib.util.spec_from_file_location("vllm_openai_chat_testmod", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_resolve_model_name_uses_first_server_model(monkeypatch):
    module = _load_module(monkeypatch)

    def _fake_urlopen(req, timeout):
        assert req.full_url == "http://127.0.0.1:7766/v1/models"
        assert timeout == 30
        return _FakeResponse(
            {
                "data": [
                    {"id": "Qwen3.6-35B-A3B"},
                    {"id": "fallback-model"},
                ]
            }
        )

    monkeypatch.setattr(module.request, "urlopen", _fake_urlopen)

    assert module.resolve_model_name("http://127.0.0.1:7766", 30, None) == "Qwen3.6-35B-A3B"


def test_extract_assistant_text_rejects_reasoning_only_response(monkeypatch):
    module = _load_module(monkeypatch)

    with pytest.raises(RuntimeError, match="only returned reasoning_content"):
        module.extract_assistant_text(
            {
                "choices": [
                    {
                        "message": {
                            "reasoning_content": "internal chain of thought",
                        }
                    }
                ]
            }
        )


def test_resolve_model_name_surfaces_http_error_details(monkeypatch):
    module = _load_module(monkeypatch)

    class _FakeHTTPError(error.HTTPError):
        def __init__(self):
            super().__init__(
                "http://127.0.0.1:7766/v1/models",
                503,
                "Service Unavailable",
                hdrs=None,
                fp=None,
            )

        def read(self):
            return b"{\"error\":\"warming up\"}"

    def _fake_urlopen(req, timeout):
        raise _FakeHTTPError()

    monkeypatch.setattr(module.request, "urlopen", _fake_urlopen)

    with pytest.raises(RuntimeError, match="HTTP 503"):
        module.resolve_model_name("http://127.0.0.1:7766", 30, None)
