import ast
from dataclasses import fields
from pathlib import Path

from xhmodel_merak.xh_llm.workflows.result import QuantResult


def test_gemma4_quant_result_constructors_match_public_contract():
    allowed_kwargs = {field.name for field in fields(QuantResult)}
    sources = [
        Path("xhmodel_merak/xh_llm/models/gemma4_series/workflow.py"),
        Path("xhmodel_merak/xh_llm/models/gemma4_series/quant_adapter.py"),
    ]

    unexpected: list[str] = []
    for source in sources:
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "QuantResult":
                continue
            for keyword in node.keywords:
                if keyword.arg is not None and keyword.arg not in allowed_kwargs:
                    unexpected.append(f"{source}:{keyword.lineno} {keyword.arg}")

    assert unexpected == []
