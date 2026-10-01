import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SOURCES = [ROOT / "app.py", *(ROOT / "src").rglob("*.py"), *(ROOT / "tests").rglob("*.py")]


def missing_docstrings(paths: list[Path], root: Path) -> list[str]:
    """Return "file:line name" for every function, method or class in paths without a docstring."""
    missing = []
    for path in paths:
        tree = ast.parse(path.read_bytes(), filename=str(path))
        for node in ast.walk(tree):
            is_def = isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
            if is_def and not ast.get_docstring(node):
                missing.append(f"{path.relative_to(root)}:{node.lineno} {node.name}")
    return missing


def test_missing_docstrings_reports_each_undocumented_definition(tmp_path):
    """Every undocumented def or class is reported, incl. methods, nested and empty docstrings."""
    lines = [
        "def documented():",  # 1
        '    """Has one."""',
        "def bare():",  # 3
        "    pass",
        "class Bare:",  # 5
        "    def method(self):",  # 6
        "        def nested():",  # 7
        "            pass",
        "        return nested",
        "async def bare_async():",  # 10
        "    pass",
        "def empty_docstring():",  # 12
        '    ""',
        "class Documented:",
        '    """Has one."""',
        "    def method(self):",
        '        """Has one."""',
    ]
    path = tmp_path / "sample.py"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert sorted(missing_docstrings([path], tmp_path)) == sorted(
        [
            "sample.py:3 bare",
            "sample.py:5 Bare",
            "sample.py:6 method",
            "sample.py:7 nested",
            "sample.py:10 bare_async",
            "sample.py:12 empty_docstring",
        ]
    )


def test_missing_docstrings_handles_utf8_bom(tmp_path):
    """A file saved with a UTF-8 BOM (e.g. by Notepad) is parsed and still checked."""
    path = tmp_path / "bom.py"
    path.write_text("def bare():\n    pass\n", encoding="utf-8-sig")
    assert missing_docstrings([path], tmp_path) == ["bom.py:1 bare"]


def test_missing_docstrings_names_file_on_syntax_error(tmp_path):
    """A file that does not parse raises a SyntaxError naming that file."""
    path = tmp_path / "broken.py"
    path.write_text("def broken(:\n", encoding="utf-8")
    with pytest.raises(SyntaxError, match="broken.py"):
        missing_docstrings([path], tmp_path)


def test_every_function_and_class_has_a_docstring():
    """Every function, method and class in app.py, src/ and tests/ has a docstring (CLAUDE.md)."""
    missing = missing_docstrings(SOURCES, ROOT)
    assert not missing, "Add a one-line docstring to: " + ", ".join(missing)
