"""OpenRouter client wrapper: one `complete()` call with model checks, errors and retries."""

import time
from typing import TYPE_CHECKING

import openai
from openai import OpenAI
from openai.types.chat import ChatCompletion

from src.config import (
    ALLOWED_MODELS,
    API_KEY_NAME,
    MissingAPIKeyError,
    SecretsFileError,
    get_api_key,
)

if TYPE_CHECKING:
    # The SDK's own HTTP library; only needed for the type hint, so never imported at runtime.
    import httpx2

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT = 30.0
# 429 and 5xx are retried this many times, waiting BACKOFF_SECONDS, then twice as long, etc.
MAX_RETRIES = 2
BACKOFF_SECONDS = 1.0
_RETRYABLE = (openai.RateLimitError, openai.InternalServerError)
_UNREADABLE = "The AI service sent an unreadable answer. Please try again."

# Indirection so tests can patch out the backoff wait.
_sleep = time.sleep


class LLMError(RuntimeError):
    """Base error for a failed LLM call; its message is safe to show to the user."""


class InvalidModelError(LLMError, ValueError):
    """Raised when a model outside `ALLOWED_MODELS` is requested."""


class LLMTimeoutError(LLMError):
    """Raised when OpenRouter does not answer within `REQUEST_TIMEOUT` seconds."""


class LLMAuthError(LLMError):
    """Raised when the API key is missing, unreadable or rejected by OpenRouter (HTTP 401)."""


class LLMRateLimitError(LLMError):
    """Raised when OpenRouter keeps answering HTTP 429 (too many requests)."""


class LLMServerError(LLMError):
    """Raised when OpenRouter keeps answering with an HTTP 5xx error."""


def _translate(exc: openai.APIError) -> LLMError:
    """Map an SDK error to our own exception with a fixed, user-readable message."""
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeoutError("The AI service took too long to respond. Please try again.")
    if isinstance(exc, openai.AuthenticationError):
        return LLMAuthError(
            f"The OpenRouter API key was rejected. Check {API_KEY_NAME} and reload the page."
        )
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimitError(
            "The AI service is receiving too many requests. Please wait a moment and try again."
        )
    if isinstance(exc, openai.InternalServerError):
        return LLMServerError(
            "The AI service is having problems right now. Please try again later."
        )
    return LLMError("The request to the AI service failed. Please try again.")


def _reply_text(response: object) -> str | None:
    """Return the first choice's text (None if it has none); raise LLMError for a non-chat reply."""
    # The SDK does not validate 200 responses: a non-JSON body comes back as a plain str,
    # and a JSON body may lack choices or message, so read every field defensively.
    if not isinstance(response, ChatCompletion):
        raise LLMError(_UNREADABLE)
    choices = response.choices
    if not isinstance(choices, list) or not choices:
        return None
    content = getattr(getattr(choices[0], "message", None), "content", None)
    return content if isinstance(content, str) else None


def make_client(api_key: str | None = None, http_client: httpx2.Client | None = None) -> OpenAI:
    """Return an OpenAI SDK client pointed at OpenRouter, with the SDK's own retries off."""
    if not api_key:
        try:
            api_key = get_api_key()
        except SecretsFileError as exc:
            raise LLMAuthError(
                ".streamlit/secrets.toml could not be parsed. Check the file's TOML syntax."
            ) from exc
        except MissingAPIKeyError as exc:
            raise LLMAuthError(
                f"No OpenRouter API key is set. Add {API_KEY_NAME} and reload the page."
            ) from exc
    if not api_key.isascii():
        # HTTP headers are ASCII; smart quotes or a zero-width space pasted with the key would
        # otherwise fail inside the SDK as a UnicodeEncodeError before any request is sent.
        raise LLMAuthError(
            f"The OpenRouter API key contains invalid characters. Check {API_KEY_NAME} "
            "and reload the page."
        )
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=REQUEST_TIMEOUT,
        # complete() runs its own retry loop, so the SDK must not retry as well.
        max_retries=0,
        http_client=http_client,
    )


def complete(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    *,
    client: OpenAI | None = None,
) -> str:
    """Send a chat request to OpenRouter and return the assistant's reply text."""
    if model not in ALLOWED_MODELS:
        raise InvalidModelError(f"Model {model!r} is not allowed. Choose one of the listed models.")
    client = client or make_client()
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            break
        except _RETRYABLE as exc:
            if attempt == MAX_RETRIES:
                raise _translate(exc) from exc
            _sleep(BACKOFF_SECONDS * 2**attempt)
        except openai.APIError as exc:
            raise _translate(exc) from exc
        except ValueError as exc:
            # Mostly a 200 body that does not parse (json.JSONDecodeError is not an APIError).
            # Request-side ValueErrors, e.g. a NaN temperature, also land here; a non-ASCII
            # key never does, because make_client() rejects it first.
            raise LLMError(_UNREADABLE) from exc
    text = _reply_text(response)
    if not text or not text.strip():
        raise LLMError("The AI service returned an empty answer. Please try again.")
    return text
