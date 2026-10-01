import pytest

from src import config


def test_allowed_models():
    """The allowed model list is exactly gpt-5-mini."""
    assert config.ALLOWED_MODELS == ("openai/gpt-5-mini",)


def test_default_model_is_allowed():
    """The default model is gpt-5-mini and is in the allowed list."""
    assert config.DEFAULT_MODEL == "openai/gpt-5-mini"
    assert config.DEFAULT_MODEL in config.ALLOWED_MODELS


def test_default_temperature_in_range():
    """The default temperature sits inside the allowed range."""
    assert config.MIN_TEMPERATURE <= config.DEFAULT_TEMPERATURE <= config.MAX_TEMPERATURE


class _NoSecretsFile:
    """Mimics st.secrets when no secrets.toml exists."""

    def get(self, key, default=None):
        """Fail like Streamlit does when there is no secrets file."""
        raise FileNotFoundError("No secrets found.")


@pytest.fixture
def no_env_key(monkeypatch):
    """Remove the API key env var so tests only see what they set up."""
    monkeypatch.delenv(config.API_KEY_NAME, raising=False)


def test_api_key_from_secrets(monkeypatch, no_env_key):
    """The key is read from st.secrets."""
    monkeypatch.setattr(config.st, "secrets", {config.API_KEY_NAME: "sk-from-secrets"})
    assert config.get_api_key() == "sk-from-secrets"


def test_secrets_take_precedence_over_env(monkeypatch):
    """If both are set, st.secrets wins over the env var."""
    monkeypatch.setattr(config.st, "secrets", {config.API_KEY_NAME: "sk-from-secrets"})
    monkeypatch.setenv(config.API_KEY_NAME, "sk-from-env")
    assert config.get_api_key() == "sk-from-secrets"


def test_api_key_falls_back_to_env(monkeypatch):
    """If st.secrets has no key, the env var is used."""
    monkeypatch.setattr(config.st, "secrets", {})
    monkeypatch.setenv(config.API_KEY_NAME, "sk-from-env")
    assert config.get_api_key() == "sk-from-env"


def test_api_key_from_env_when_no_secrets_file(monkeypatch):
    """With no secrets file at all, the env var is still used."""
    monkeypatch.setattr(config.st, "secrets", _NoSecretsFile())
    monkeypatch.setenv(config.API_KEY_NAME, "sk-from-env")
    assert config.get_api_key() == "sk-from-env"


def test_api_key_is_stripped(monkeypatch, no_env_key):
    """Spaces and newlines around the key are removed."""
    monkeypatch.setattr(config.st, "secrets", {config.API_KEY_NAME: "  sk-padded \n"})
    assert config.get_api_key() == "sk-padded"


@pytest.mark.parametrize("secrets", [{}, {config.API_KEY_NAME: ""}, {config.API_KEY_NAME: "  "}])
def test_missing_api_key_raises(monkeypatch, no_env_key, secrets):
    """An absent, empty or blank key raises MissingAPIKeyError."""
    monkeypatch.setattr(config.st, "secrets", secrets)
    with pytest.raises(config.MissingAPIKeyError):
        config.get_api_key()


def test_missing_api_key_raises_when_no_secrets_file(monkeypatch, no_env_key):
    """No secrets file and no env var raises MissingAPIKeyError."""
    monkeypatch.setattr(config.st, "secrets", _NoSecretsFile())
    with pytest.raises(config.MissingAPIKeyError):
        config.get_api_key()
