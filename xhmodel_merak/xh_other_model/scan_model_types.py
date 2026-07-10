import ast
from collections import OrderedDict
from pathlib import Path


def is_register_other_model_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    if isinstance(node.func, ast.Name):
        return node.func.id == "register_other_model"
    if isinstance(node.func, ast.Attribute):
        return node.func.attr == "register_other_model"
    return False


def extract_register_other_model_args(decorator: ast.Call) -> tuple[str | None, bool, bool]:
    name = None
    master = True
    force = False

    if decorator.args:
        first_arg = decorator.args[0]
        if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
            name = first_arg.value

    for keyword in decorator.keywords:
        if keyword.arg in {"model_type", "name"}:
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, str):
                name = keyword.value.value
        elif keyword.arg == "master":
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, bool):
                master = keyword.value.value
        elif keyword.arg == "force":
            if isinstance(keyword.value, ast.Constant) and isinstance(keyword.value.value, bool):
                force = keyword.value.value

    return name, master, force


def parse_register_other_models(py_file: Path) -> list[dict[str, object]]:
    package_dir = Path(__file__).parent
    rel = Path(py_file).relative_to(package_dir)
    if len(rel.parts) < 3 or rel.parts[0] != "models":
        return []
    module_name = rel.parts[1]
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    results: list[dict[str, object]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for decorator in node.decorator_list:
            if not is_register_other_model_call(decorator):
                continue
            name, master, force = extract_register_other_model_args(decorator)
            if name is None:
                continue
            results.append(
                {
                    "target": node.name,
                    "model_type": name,
                    "module_name": module_name,
                    "master": master,
                    "force": force,
                    "lineno": getattr(decorator, "lineno", None),
                    "py_file": py_file,
                }
            )

    return results


def _scan_registered_models() -> list[dict[str, object]]:
    models_dir = Path(__file__).parent / "models"
    results = []
    for py_file in sorted(models_dir.rglob("*.py")):
        results += parse_register_other_models(py_file)
    return results


def get_support_all_model_types() -> OrderedDict[str, str]:
    model_types: OrderedDict[str, str] = OrderedDict()
    model_forced = {}
    for result in _scan_registered_models():
        model_type = str(result["model_type"])
        force = bool(result.get("force"))
        if model_type not in model_types or force or not model_forced.get(model_type, False):
            model_types[model_type] = str(result["module_name"])
            model_forced[model_type] = force
    return model_types


def get_support_master_model_types() -> OrderedDict[str, str]:
    model_types: OrderedDict[str, str] = OrderedDict()
    model_forced = {}
    for result in _scan_registered_models():
        if not result["master"]:
            continue
        model_type = str(result["model_type"])
        force = bool(result.get("force"))
        if model_type not in model_types or force or not model_forced.get(model_type, False):
            model_types[model_type] = str(result["module_name"])
            model_forced[model_type] = force
    return model_types
