"""App-wide settings: allowed models, model defaults and secrets loading."""

import os
from collections.abc import Mapping
from types import MappingProxyType

import streamlit as st

# The three chat models allowed by the project brief.
ALLOWED_MODELS: tuple[str, ...] = ("openai/gpt-5-mini", "openai/gpt-5-nano", "openai/gpt-5")
DEFAULT_MODEL = "openai/gpt-5-mini"

# How long a gpt-5 model thinks before it answers (OpenRouter's `reasoning.effort`). These models
# ignore `temperature`, so this is the setting the app offers. Higher effort spends more of
# max_tokens on thinking. "medium" is OpenAI's own default, so it changes nothing by itself.
REASONING_EFFORTS: tuple[str, ...] = ("minimal", "low", "medium", "high")
DEFAULT_REASONING_EFFORT = "medium"

# gpt-5 models spend reasoning tokens from this budget too, so keep it generous: a low budget
# can be used up by thinking alone and return no text. Enough for minimal, low and medium.
DEFAULT_MAX_TOKENS = 4000
# At "high", the sample-JD starter used 3,648 of 4,000 tokens thinking and was cut off (T2.5).
# 16,000 leaves room for twice that thinking plus a long reply. OpenRouter bills the tokens
# used, not this limit, so the cost only rises when the model really thinks that long.
HIGH_EFFORT_MAX_TOKENS = 16000
# The max_tokens the app sends for each reasoning effort (one entry per REASONING_EFFORTS level).
MAX_TOKENS_BY_EFFORT: Mapping[str, int] = MappingProxyType(
    {
        "minimal": DEFAULT_MAX_TOKENS,
        "low": DEFAULT_MAX_TOKENS,
        "medium": DEFAULT_MAX_TOKENS,
        "high": HIGH_EFFORT_MAX_TOKENS,
    }
)
# Every request is clamped to this, whatever a caller asks for, to bound the cost of one reply.
# The largest budget above, so "high" fits and nothing can ask for more.
MAX_TOKENS_CAP = 16000

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
# The prompt strategy the app starts with: the winner of the T3.3 evaluation
# (docs/PROMPT_EVALUATION.md). Must be a key of prompts.STRATEGIES (a test checks this).
DEFAULT_STRATEGY = "structured_output"

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
