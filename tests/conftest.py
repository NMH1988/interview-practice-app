import pytest

from src.config import API_KEY_NAME


@pytest.fixture
def no_env_key(monkeypatch):
    """Remove the API key env var so tests only see the key they set up."""
    monkeypatch.delenv(API_KEY_NAME, raising=False)
