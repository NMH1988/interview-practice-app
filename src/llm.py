"""OpenRouter client wrapper: `complete()` and `stream()` with model checks, errors and retries."""

import logging
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import openai
from openai import OpenAI, Stream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from src.config import (
    ALLOWED_MODELS,
    API_KEY_NAME,
    MAX_TOKENS_CAP,
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
_EMPTY = "The AI service returned an empty answer. Please try again."
# gpt-5 models think before they write, and that thinking is paid from the same max_tokens.
_CUT_OFF_EMPTY = "The model used up its token limit before writing an answer. Please try again."

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class Usage:
    """Token counts OpenRouter reported for one request."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Part of completion_tokens that gpt-5 spent thinking; None when not reported.
    reasoning_tokens: int | None = None


@dataclass
class _StreamEnd:
    """What a stream reported by the time it ended: token usage and why the model stopped."""

    usage: Usage | None = None
    finish_reason: str | None = None


def _count(value: object) -> int | None:
    """Return `value` if it is a whole, non-negative token count, else None."""
    return value if isinstance(value, int) and value >= 0 else None


def _read_usage(source: object) -> Usage | None:
    """Return the token usage on a response or chunk, or None if it has none or it is malformed."""
    # Not validated by the SDK either: any field may be missing or of the wrong type.
    usage = getattr(source, "usage", None)
    prompt = _count(getattr(usage, "prompt_tokens", None))
    completion = _count(getattr(usage, "completion_tokens", None))
    total = _count(getattr(usage, "total_tokens", None))
    if prompt is None or completion is None or total is None:
        return None
    details = getattr(usage, "completion_tokens_details", None)
    return Usage(prompt, completion, total, _count(getattr(details, "reasoning_tokens", None)))


def _log_usage(model: str, usage: Usage) -> None:
    """Log one request's token counts (never the text) on this module's logger."""
    reasoning = "" if usage.reasoning_tokens is None else f" (reasoning {usage.reasoning_tokens})"
    logger.info(
        "Token usage (%s): prompt %d, completion %d%s, total %d",
        model,
        usage.prompt_tokens,
        usage.completion_tokens,
        reasoning,
        usage.total_tokens,
    )


def _finish_reason(source: object) -> str | None:
    """Return why the first choice stopped ("length" = hit max_tokens), or None if not given."""
    choices = getattr(source, "choices", None)
    if not isinstance(choices, list) or not choices:
        return None
    reason = getattr(choices[0], "finish_reason", None)
    return reason if isinstance(reason, str) else None


def _no_text_error(finish_reason: str | None) -> LLMError:
    """Return the error for a reply without text, saying so when the token limit caused it."""
    return LLMError(_CUT_OFF_EMPTY if finish_reason == "length" else _EMPTY)


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


def _chunk_text(chunk: object) -> str:
    """Return the text a streamed chunk adds ("" if none); raise LLMError for a non-chat chunk."""
    # Streamed chunks are not validated either, so read every field defensively.
    if not isinstance(chunk, ChatCompletionChunk):
        raise LLMError(_UNREADABLE)
    choices = chunk.choices
    if not isinstance(choices, list) or not choices:
        return ""
    content = getattr(getattr(choices[0], "delta", None), "content", None)
    return content if isinstance(content, str) else ""


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
    if not (api_key.isascii() and api_key.isprintable()):
        # HTTP header values must be printable ASCII. Smart quotes or a zero-width space pasted
        # with the key would fail inside the SDK as a UnicodeEncodeError; a newline, CR or NUL
        # would fail in h11 as a connection error whose message quotes the raw key.
        raise LLMAuthError(
            f"The OpenRouter API key contains invalid characters. Check {API_KEY_NAME} "
            "and reload the page."
        )
    return OpenAI(
        api_key=api_key,
        base_url=OPENROUTER_BASE_URL,
        timeout=REQUEST_TIMEOUT,
        # _send() runs its own retry loop, so the SDK must not retry as well.
        max_retries=0,
        http_client=http_client,
    )


def _check_model(model: str) -> None:
    """Raise InvalidModelError if `model` is not in ALLOWED_MODELS."""
    if model not in ALLOWED_MODELS:
        raise InvalidModelError(f"Model {model!r} is not allowed. Choose one of the listed models.")


def _send(
    client: OpenAI,
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    *,
    stream: bool = False,
) -> object:
    """Send one chat request, retrying 429/5xx with backoff, and return the SDK's response."""
    # Whatever the caller asks for, one reply never costs more than the cap.
    max_tokens = min(max_tokens, MAX_TOKENS_CAP)
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Arguments spelled out (no **kwargs), as T4.4's source scan requires.
            return client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                # Left out unless streaming, so complete() sends the same body as before.
                stream=True if stream else openai.omit,
                # Without this a stream reports no token usage (it comes in a last, textless chunk).
                stream_options={"include_usage": True} if stream else openai.omit,
            )
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
    raise AssertionError("unreachable: the last attempt returns or raises")


def complete(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    *,
    client: OpenAI | None = None,
) -> str:
    """Send a chat request to OpenRouter and return the assistant's reply text."""
    _check_model(model)
    client = client or make_client()
    response = _send(client, messages, model, temperature, max_tokens)
    text = _reply_text(response)
    usage = _read_usage(response)
    if usage is not None:
        _log_usage(model, usage)
    finish_reason = _finish_reason(response)
    if not text or not text.strip():
        raise _no_text_error(finish_reason)
    if finish_reason == "length":
        # complete() returns only text, so scripts (T3.3) can only see a cut-off reply here.
        logger.warning("Reply from %s was cut off by max_tokens", model)
    return text


class ReplyStream(Iterator[str]):
    """The reply's text pieces; `usage` and `finish_reason` are set once all of them are read."""

    def __init__(self, pieces: Iterator[str], end: _StreamEnd):
        """Wrap the piece generator and the record it fills in when the stream ends."""
        # The generator holds `end`, not this object, so there is no reference cycle and
        # refcounting still closes it as soon as nobody holds the stream.
        self._pieces = pieces
        self._end = end

    def __next__(self) -> str:
        """Return the next piece of text; the request goes out on the first call."""
        return next(self._pieces)

    def close(self) -> None:
        """Stop reading early and close the HTTP response."""
        self._pieces.close()

    @property
    def usage(self) -> Usage | None:
        """Token counts for the request, or None until the end or if OpenRouter sent none."""
        return self._end.usage

    @property
    def finish_reason(self) -> str | None:
        """Why the model stopped ("length" = cut off by max_tokens), or None if not given."""
        return self._end.finish_reason


def stream(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    *,
    client: OpenAI | None = None,
) -> ReplyStream:
    """Check the model and key now, and return the reply as a stream of text pieces."""
    # Not a generator itself, so a bad model or key fails here rather than on the first next().
    _check_model(model)
    client = client or make_client()
    end = _StreamEnd()
    return ReplyStream(_stream_pieces(client, messages, model, temperature, max_tokens, end), end)


def _stream_pieces(
    client: OpenAI,
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    end: _StreamEnd,
) -> Iterator[str]:
    """Send the streaming request on the first next() and yield each piece of text as it comes."""
    # Only the request is retried: once text is shown, a retry would repeat it.
    response = _send(client, messages, model, temperature, max_tokens, stream=True)
    if not isinstance(response, Stream):
        raise LLMError(_UNREADABLE)
    has_text = False
    try:
        for chunk in response:
            piece = _chunk_text(chunk)
            # The reason comes with the last text chunk, the usage in a chunk of its own after it.
            end.finish_reason = _finish_reason(chunk) or end.finish_reason
            end.usage = _read_usage(chunk) or end.usage
            if piece:
                has_text = has_text or bool(piece.strip())
                yield piece
    except openai.APIError as exc:
        # A timeout, dropped connection or error event after the reply has started.
        raise _translate(exc) from exc
    except ValueError as exc:
        # An event whose data is not JSON.
        raise LLMError(_UNREADABLE) from exc
    finally:
        # Also runs when the caller stops early (close()), so the connection is not kept open.
        response.close()
    if end.usage is not None:
        _log_usage(model, end.usage)
    if not has_text:
        raise _no_text_error(end.finish_reason)
