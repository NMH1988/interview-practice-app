import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_pyproject() -> dict:
    """Return the parsed contents of pyproject.toml."""
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_every_runtime_dependency_is_pinned_with_double_equals():
    """Each runtime dependency in pyproject.toml is pinned to one exact version with `==`."""
    dependencies = load_pyproject()["project"]["dependencies"]
    assert dependencies
    for spec in dependencies:
        name, _, version = spec.partition("==")
        assert name.strip() and version.strip(), f"{spec!r} is not pinned with =="
        assert not any(mark in version for mark in "<>=~!*,"), f"{spec!r} is not an exact pin"


def test_dev_tools_are_in_the_dev_extra():
    """pytest, pytest-cov and ruff are installed through the `dev` extra."""
    dev = load_pyproject()["project"]["optional-dependencies"]["dev"]
    names = {spec.split("==")[0].strip() for spec in dev}
    assert {"pytest", "pytest-cov", "ruff"} <= names


def test_package_discovery_is_limited_to_src():
    """Setuptools only packages `src`, so app.py, scripts/ and tests/ stay out of the install."""
    discovery = load_pyproject()["tool"]["setuptools"]["packages"]["find"]
    assert discovery["include"] == ["src*"]


def test_no_requirements_files_exist():
    """Dependencies live only in pyproject.toml: no requirements*.txt file is back."""
    assert list(ROOT.glob("requirements*.txt")) == []
