from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import API_KEY_NAME

APP = Path(__file__).resolve().parent.parent / "app.py"


@pytest.fixture(autouse=True)
def no_env_key(monkeypatch):
    monkeypatch.delenv(API_KEY_NAME, raising=False)


def test_app_renders_without_exception():
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = "sk-test-not-a-real-key"
    at.run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "Interview Practice"
    assert not at.error
    assert len(at.metric) == 3


def test_missing_api_key_shows_friendly_error():
    at = AppTest.from_file(str(APP))
    # A non-empty secrets dict replaces any local secrets.toml, so the key is truly absent.
    at.secrets["UNRELATED"] = "x"
    at.run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "Interview Practice"
    assert len(at.error) == 1
    assert API_KEY_NAME in at.error[0].value
    assert len(at.metric) == 0
