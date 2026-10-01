"""App-wide settings: allowed models, model defaults and secrets loading."""

import os

import streamlit as st

ALLOWED_MODELS: tuple[str, ...] = ("openai/gpt-5-mini",)
DEFAULT_MODEL = "openai/gpt-5-mini"

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.5
DEFAULT_TEMPERATURE = 0.7

API_KEY_NAME = "OPENROUTER_API_KEY"


class MissingAPIKeyError(RuntimeError):
    """Raised when no OpenRouter API key is configured."""


class SecretsFileError(MissingAPIKeyError):
    """Raised when secrets.toml exists but cannot be parsed, and no env var is set."""


def get_api_key() -> str:
    """Return the OpenRouter key from `st.secrets`, falling back to the environment."""
    file_broken = False
    try:
        value = st.secrets.get(API_KEY_NAME)
    except FileNotFoundError as exc:
        # Streamlit raises this for a missing secrets.toml, and chains the TOML error
        # as the cause when the file exists but cannot be parsed.
        value = None
        file_broken = exc.__cause__ is not None
    secret = value.strip() if isinstance(value, str) else ""
    key = secret or os.environ.get(API_KEY_NAME, "").strip()
    if not key:
        if file_broken:
            # `from None` keeps the parser's message (it may quote file content) out of tracebacks.
            raise SecretsFileError(".streamlit/secrets.toml could not be parsed.") from None
        raise MissingAPIKeyError(f"{API_KEY_NAME} is not set.")
    return key
