import tomllib
from pathlib import Path

from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parent.parent


def load_pyproject() -> dict:
    """Return the parsed contents of pyproject.toml."""
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def is_exact_pin(spec: str) -> bool:
    """Return True when the requirement has one specifier, `==`, with no wildcard."""
    specifiers = list(Requirement(spec).specifier)
    return (
        len(specifiers) == 1 and specifiers[0].operator == "==" and "*" not in specifiers[0].version
    )


def test_is_exact_pin_accepts_only_one_double_equals_version():
    """Exact pins pass; ranges, wildcards, combined specifiers and missing versions fail."""
    assert is_exact_pin("openai==3.22.1")
    for loose in ("openai", "openai>=3", "openai~=3.22", "openai==3.*", "openai<4,==3.22.1"):
        assert not is_exact_pin(loose), loose
    assert not is_exact_pin("openai==3.22.1,<4")


def test_every_runtime_dependency_is_pinned_with_double_equals():
    """Each runtime dependency in pyproject.toml is pinned to one exact version with `==`."""
    dependencies = load_pyproject()["project"]["dependencies"]
    assert dependencies
    for spec in dependencies:
        assert is_exact_pin(spec), f"{spec!r} is not pinned with =="


def test_dev_tools_are_in_the_dev_extra():
    """pytest, pytest-cov, ruff and packaging are installed through the `dev` extra."""
    dev = load_pyproject()["project"]["optional-dependencies"]["dev"]
    names = {Requirement(spec).name for spec in dev}
    assert {"pytest", "pytest-cov", "ruff", "packaging"} <= names


def test_package_discovery_is_limited_to_src():
    """Setuptools only packages `src`, so app.py, scripts/ and tests/ stay out of the install."""
    discovery = load_pyproject()["tool"]["setuptools"]["packages"]["find"]
    assert discovery["include"] == ["src*"]


def test_no_requirements_files_exist():
    """Dependencies live only in pyproject.toml: no requirements*.txt file is back."""
    assert list(ROOT.glob("requirements*.txt")) == []
