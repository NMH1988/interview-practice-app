from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_secret_files_are_gitignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".streamlit/secrets.toml" in ignored
    assert ".env" in ignored


def test_secrets_example_has_placeholder_key_only():
    example = (ROOT / ".streamlit" / "secrets.toml.example").read_text(encoding="utf-8")
    assert 'OPENROUTER_API_KEY = "sk-or-v1-your-key-here"' in example
