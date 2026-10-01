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


def get_api_key() -> str:
    """Return the OpenRouter key from `st.secrets`, falling back to the environment."""
    try:
        value = st.secrets.get(API_KEY_NAME)
    except FileNotFoundError:
        # Streamlit raises this when there is no secrets.toml (or it cannot be parsed).
        value = None
    key = str(value or "").strip() or os.environ.get(API_KEY_NAME, "").strip()
    if not key:
        raise MissingAPIKeyError(f"{API_KEY_NAME} is not set.")
    return key
