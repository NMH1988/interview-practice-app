import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_requirements(path: Path) -> list[str]:
    """Return the requirement specifiers in a requirements file, without comments."""
    requirements = []
    for line in path.read_text(encoding="utf-8").splitlines():
        spec = line.split("#", 1)[0].strip()
        if spec:
            requirements.append(spec)
    return requirements


def test_read_requirements_skips_comments(tmp_path):
    """Indented comments, inline comments and blank lines are not counted as dependencies."""
    path = tmp_path / "requirements.txt"
    path.write_text("# header\n  # indented\n\nstreamlit==1.64.0  # UI\n", encoding="utf-8")
    assert read_requirements(path) == ["streamlit==1.64.0"]


def test_pyproject_and_requirements_list_same_runtime_dependencies():
    """pyproject.toml and requirements.txt pin the same runtime dependencies."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    requirements = read_requirements(ROOT / "requirements.txt")
    assert sorted(pyproject["project"]["dependencies"]) == sorted(requirements)
