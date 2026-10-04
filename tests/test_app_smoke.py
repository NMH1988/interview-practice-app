from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import config
from src.config import API_KEY_NAME

APP = Path(__file__).resolve().parent.parent / "app.py"


# Each test controls where the key comes from (see tests/conftest.py).
pytestmark = pytest.mark.usefixtures("no_env_key")


def test_app_renders_without_exception():
    """With a key set, the app loads fully: title and no error."""
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = "sk-test-not-a-real-key"
    at.run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "Interview Practice"
    assert not at.error


def test_placeholder_dashboard_is_gone():
    """The made-up metrics, date filter and score chart of the old dashboard are not drawn."""
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = "sk-test-not-a-real-key"
    at.run(timeout=30)
    assert not at.exception
    assert len(at.metric) == 0
    assert len(at.date_input) == 0
    # AppTest has no chart accessor; st.line_chart is drawn as an element of this type.
    assert len(at.get("vega_lite_chart")) == 0
    assert "Filters" not in [header.value for header in at.sidebar.header]


def test_missing_api_key_shows_friendly_error():
    """Without a key, the app shows a setup message instead of crashing."""
    at = AppTest.from_file(str(APP))
    # A non-empty secrets dict replaces any local secrets.toml, so the key is truly absent.
    at.secrets["UNRELATED"] = "x"
    at.run(timeout=30)
    assert not at.exception
    assert at.title[0].value == "Interview Practice"
    assert len(at.error) == 1
    assert API_KEY_NAME in at.error[0].value
    assert len(at.metric) == 0


def test_unparseable_secrets_file_shows_its_own_error(monkeypatch):
    """A broken secrets.toml gets a "check the quotes" message, not "key missing"."""

    def broken_file():
        """Fail the way get_api_key does when secrets.toml cannot be parsed."""
        raise config.SecretsFileError(".streamlit/secrets.toml could not be parsed.")

    # app.py imports get_api_key on each run, so patching the module attribute is enough.
    monkeypatch.setattr(config, "get_api_key", broken_file)
    at = AppTest.from_file(str(APP))
    at.run(timeout=30)
    assert not at.exception
    assert len(at.error) == 1
    assert "could not be parsed" in at.error[0].value
    assert "missing" not in at.error[0].value
    assert len(at.metric) == 0
