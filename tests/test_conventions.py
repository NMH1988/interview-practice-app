import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "app.py", *(ROOT / "src").rglob("*.py"), *(ROOT / "tests").rglob("*.py")]


def test_every_function_and_class_has_a_docstring():
    """Every function, method and class in app.py, src/ and tests/ has a docstring (CLAUDE.md)."""
    missing = []
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            is_def = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            if is_def and not ast.get_docstring(node):
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno} {node.name}")
    assert not missing, "Add a one-line docstring to: " + ", ".join(missing)
