"""App-wide settings: allowed models, model defaults and secrets loading."""

import os

import streamlit as st

# The three chat models allowed by the project brief.
ALLOWED_MODELS: tuple[str, ...] = ("openai/gpt-5-mini", "openai/gpt-5-nano", "openai/gpt-5")
DEFAULT_MODEL = "openai/gpt-5-mini"

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.5
DEFAULT_TEMPERATURE = 0.7

# gpt-5 models spend reasoning tokens from this budget too, so keep it generous: a low budget
# can be used up by thinking alone and return no text.
DEFAULT_MAX_TOKENS = 4000
# Every request is clamped to this, whatever a caller asks for, to bound the cost of one reply.
MAX_TOKENS_CAP = 4000

# Longest user message (after cleaning) the guard lets through to the LLM.
MAX_INPUT_CHARS = 2000
# The same limit in job-description mode, where a pasted JD is often 2,000-5,000 characters.
# About 1,500 tokens, re-sent with every later turn of the session.
MAX_JD_CHARS = 6000
# A system prompt (or one of its paragraphs) this long, repeated in a reply, counts as a leak.
# Shorter text (headings, the 1-5 scale, a tiny test prompt) may appear in a normal reply.
MIN_LEAK_CHARS = 80

# Requests to the LLM allowed per browser session: at most this many in any rolling window, and
# this many in all. "New session" does not reset them; reloading the page starts a new session.
RATE_LIMIT_PER_MINUTE = 10
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_PER_SESSION = 50

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
