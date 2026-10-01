import pytest

from src import config


def test_allowed_models():
    assert config.ALLOWED_MODELS == ("openai/gpt-5-mini",)


def test_default_model_is_allowed():
    assert config.DEFAULT_MODEL == "openai/gpt-5-mini"
    assert config.DEFAULT_MODEL in config.ALLOWED_MODELS


def test_default_temperature_in_range():
    assert config.MIN_TEMPERATURE <= config.DEFAULT_TEMPERATURE <= config.MAX_TEMPERATURE


class _NoSecretsFile:
    """Mimics st.secrets when no secrets.toml exists."""

    def get(self, key, default=None):
        raise FileNotFoundError("No secrets found.")


@pytest.fixture
def no_env_key(monkeypatch):
    monkeypatch.delenv(config.API_KEY_NAME, raising=False)


def test_api_key_from_secrets(monkeypatch, no_env_key):
    monkeypatch.setattr(config.st, "secrets", {config.API_KEY_NAME: "sk-from-secrets"})
    assert config.get_api_key() == "sk-from-secrets"


def test_secrets_take_precedence_over_env(monkeypatch):
    monkeypatch.setattr(config.st, "secrets", {config.API_KEY_NAME: "sk-from-secrets"})
    monkeypatch.setenv(config.API_KEY_NAME, "sk-from-env")
    assert config.get_api_key() == "sk-from-secrets"


def test_api_key_falls_back_to_env(monkeypatch):
    monkeypatch.setattr(config.st, "secrets", {})
    monkeypatch.setenv(config.API_KEY_NAME, "sk-from-env")
    assert config.get_api_key() == "sk-from-env"


def test_api_key_from_env_when_no_secrets_file(monkeypatch):
    monkeypatch.setattr(config.st, "secrets", _NoSecretsFile())
    monkeypatch.setenv(config.API_KEY_NAME, "sk-from-env")
    assert config.get_api_key() == "sk-from-env"


def test_api_key_is_stripped(monkeypatch, no_env_key):
    monkeypatch.setattr(config.st, "secrets", {config.API_KEY_NAME: "  sk-padded \n"})
    assert config.get_api_key() == "sk-padded"


@pytest.mark.parametrize("secrets", [{}, {config.API_KEY_NAME: ""}, {config.API_KEY_NAME: "  "}])
def test_missing_api_key_raises(monkeypatch, no_env_key, secrets):
    monkeypatch.setattr(config.st, "secrets", secrets)
    with pytest.raises(config.MissingAPIKeyError):
        config.get_api_key()


def test_missing_api_key_raises_when_no_secrets_file(monkeypatch, no_env_key):
    monkeypatch.setattr(config.st, "secrets", _NoSecretsFile())
    with pytest.raises(config.MissingAPIKeyError):
        config.get_api_key()
