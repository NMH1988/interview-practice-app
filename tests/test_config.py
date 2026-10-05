import os

import pytest
from streamlit import config as st_config
from streamlit.runtime.secrets import Secrets

from src import config
from src.prompts import SENIORITY_LEVELS, STRATEGIES


def test_allowed_models():
    """Allowed models are exactly the three chat models from the brief."""
    assert config.ALLOWED_MODELS == ("openai/gpt-5-mini", "openai/gpt-5-nano", "openai/gpt-5")


def test_default_model_is_allowed():
    """Default model is gpt-5-mini and is in the allowed list."""
    assert config.DEFAULT_MODEL == "openai/gpt-5-mini"
    assert config.DEFAULT_MODEL in config.ALLOWED_MODELS


def test_reasoning_efforts_and_default():
    """The four effort levels every gpt-5 model accepts, with OpenAI's default "medium"."""
    assert config.REASONING_EFFORTS == ("minimal", "low", "medium", "high")
    assert config.DEFAULT_REASONING_EFFORT == "medium"
    assert config.DEFAULT_REASONING_EFFORT in config.REASONING_EFFORTS


def test_temperature_settings_are_gone():
    """gpt-5 models ignore temperature, so config.py no longer offers one (T2.4)."""
    assert not [name for name in dir(config) if "TEMPERATURE" in name]


def test_default_role_limit_is_60():
    """The role limit agreed for T5.1 lives in config.py."""
    assert config.MAX_ROLE_CHARS == 60


def test_jd_limit_is_6000_and_above_the_message_limit():
    """Job-description mode allows 6,000 characters, more than a normal message (T5.3)."""
    assert config.MAX_JD_CHARS == 6000
    assert config.MAX_JD_CHARS > config.MAX_INPUT_CHARS


def test_default_role_fits_the_role_limit():
    """The default role is not blank and passes the role length limit."""
    assert config.DEFAULT_ROLE.strip()
    assert len(config.DEFAULT_ROLE) <= config.MAX_ROLE_CHARS


def test_default_max_tokens_fits_the_cap():
    """The default token budget is positive and not above the per-request cap."""
    assert 0 < config.DEFAULT_MAX_TOKENS <= config.MAX_TOKENS_CAP


def test_every_effort_has_a_token_budget():
    """Each reasoning effort has exactly one budget, and no other key is listed (T2.5)."""
    assert tuple(config.MAX_TOKENS_BY_EFFORT) == config.REASONING_EFFORTS


def test_only_high_effort_gets_the_larger_budget():
    """Minimal, low and medium keep the old 4,000; high gets 16,000 for its longer thinking."""
    assert config.DEFAULT_MAX_TOKENS == 4000
    assert config.HIGH_EFFORT_MAX_TOKENS == 16000
    assert dict(config.MAX_TOKENS_BY_EFFORT) == {
        "minimal": config.DEFAULT_MAX_TOKENS,
        "low": config.DEFAULT_MAX_TOKENS,
        "medium": config.DEFAULT_MAX_TOKENS,
        "high": config.HIGH_EFFORT_MAX_TOKENS,
    }


def test_cap_is_the_largest_effort_budget():
    """No budget is above the cap, and the cap allows no more than the largest budget needs."""
    budgets = config.MAX_TOKENS_BY_EFFORT.values()
    assert all(0 < budget <= config.MAX_TOKENS_CAP for budget in budgets)
    assert config.MAX_TOKENS_CAP == max(budgets)


def test_token_budgets_cannot_be_changed_at_runtime():
    """The budget table is read-only, so no caller can raise a budget by accident."""
    with pytest.raises(TypeError):
        config.MAX_TOKENS_BY_EFFORT["high"] = 1  # type: ignore[index]


def test_default_rate_limits():
    """The T4.3 limits (10 requests a minute, 50 a session) live in config.py."""
    assert config.RATE_LIMIT_PER_MINUTE == 10
    assert config.RATE_LIMIT_WINDOW_SECONDS == 60
    assert config.RATE_LIMIT_PER_SESSION == 50


def test_default_seniority_is_a_known_level():
    """The default seniority is one of the levels the user prompt accepts."""
    assert config.DEFAULT_SENIORITY in SENIORITY_LEVELS


def test_default_strategy_is_the_evaluation_winner_and_registered():
    """The default strategy exists in the registry and is the T3.3 winner, Structured output."""
    assert config.DEFAULT_STRATEGY in STRATEGIES
    assert config.DEFAULT_STRATEGY == "structured_output"


class _NoSecretsFile:
    """Mimics st.secrets when no secrets.toml exists."""

    def get(self, key, default=None):
        """Fail like Streamlit does when there is no secrets file."""
        raise FileNotFoundError("No secrets found.")


@pytest.fixture
def secrets_from_file(monkeypatch, tmp_path):
    """Return a helper that points a real st.secrets at a secrets.toml with the given text."""
    original = st_config.get_option("secrets.files")
    # Streamlit copies parsed string secrets into os.environ, so restore the key afterwards.
    saved_env = os.environ.get(config.API_KEY_NAME)

    def use(text):
        """Write `text` as secrets.toml and install a fresh Secrets object that reads it."""
        path = tmp_path / "secrets.toml"
        path.write_text(text, encoding="utf-8")
        st_config.set_option("secrets.files", [str(path)])
        monkeypatch.setattr(config.st, "secrets", Secrets())

    yield use
    st_config.set_option("secrets.files", original)
    if saved_env is None:
        os.environ.pop(config.API_KEY_NAME, None)
    else:
        os.environ[config.API_KEY_NAME] = saved_env


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
    """No secrets file and no env var raises MissingAPIKeyError, not SecretsFileError."""
    monkeypatch.setattr(config.st, "secrets", _NoSecretsFile())
    with pytest.raises(config.MissingAPIKeyError) as excinfo:
        config.get_api_key()
    assert not isinstance(excinfo.value, config.SecretsFileError)


@pytest.mark.parametrize("value", [123, {"nested": "table"}, ["a"], True])
def test_non_string_secret_is_treated_as_missing(monkeypatch, no_env_key, value):
    """A number, table or list under the key is not turned into a fake key string."""
    monkeypatch.setattr(config.st, "secrets", {config.API_KEY_NAME: value})
    with pytest.raises(config.MissingAPIKeyError):
        config.get_api_key()


def test_unparseable_secrets_file_raises_secrets_file_error(secrets_from_file, no_env_key):
    """An unquoted key makes a real secrets.toml unparseable, which gets its own error."""
    secrets_from_file(f"{config.API_KEY_NAME} = sk-or-v1-unquoted\n")
    with pytest.raises(config.SecretsFileError) as excinfo:
        config.get_api_key()
    # The raw parser message could echo file content, so it must not leak out.
    assert "sk-or-v1-unquoted" not in str(excinfo.value)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__


def test_unparseable_secrets_file_falls_back_to_env(secrets_from_file, monkeypatch):
    """A broken secrets.toml does not block a key set in the environment."""
    secrets_from_file(f"{config.API_KEY_NAME} = sk-or-v1-unquoted\n")
    monkeypatch.setenv(config.API_KEY_NAME, "sk-from-env")
    assert config.get_api_key() == "sk-from-env"


def test_valid_secrets_file_is_read(secrets_from_file, no_env_key):
    """A correctly quoted key in a real secrets.toml is returned."""
    secrets_from_file(f'{config.API_KEY_NAME} = "sk-from-file"\n')
    assert config.get_api_key() == "sk-from-file"
