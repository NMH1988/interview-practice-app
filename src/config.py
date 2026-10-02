"""App-wide settings: allowed models, model defaults and secrets loading."""

import os

import streamlit as st

# The three chat models allowed by the project brief.
ALLOWED_MODELS: tuple[str, ...] = ("openai/gpt-5-mini", "openai/gpt-5-nano", "openai/gpt-5")
DEFAULT_MODEL = "openai/gpt-5-mini"

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.5
DEFAULT_TEMPERATURE = 0.7

# gpt-5 models spend reasoning tokens from this budget too, so keep it generous; T2.3 caps it.
DEFAULT_MAX_TOKENS = 4000

# Longest user message (after cleaning) the guard lets through to the LLM.
MAX_INPUT_CHARS = 2000

# The role goes into the system prompt, so it is free text but kept short (see T3.1 follow-ups).
DEFAULT_ROLE = "Software Engineer"
MAX_ROLE_CHARS = 60
# Must be one of prompts.SENIORITY_LEVELS (a test checks this).
DEFAULT_SENIORITY = "Mid-level"

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
