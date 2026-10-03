import gc
import json

import httpx2
import pytest

from src import config, llm
from src.config import DEFAULT_MODEL

FAKE_KEY = "sk-test-not-a-real-key"
MESSAGES = [{"role": "user", "content": "Ask me a behavioural question."}]


@pytest.fixture(autouse=True)
def sleeps(monkeypatch):
    """Replace the backoff sleep with a recorder so tests run fast; return the delays."""
    delays = []
    monkeypatch.setattr(llm, "_sleep", delays.append)
    return delays


def _reply(text, *, finish_reason="stop", usage=None, reasoning=None):
    """Return a minimal OpenAI-style chat completion response with the given text."""
    body = {
        "id": "gen-test",
        "object": "chat.completion",
        "created": 0,
        "model": DEFAULT_MODEL,
        "choices": [
            {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {"role": "assistant", "content": text},
            }
        ],
    }
    if usage is not None:
        body["usage"] = usage
    if reasoning is not None:
        # OpenRouter may return the model's thinking next to the answer once reasoning is asked for.
        body["choices"][0]["message"]["reasoning"] = reasoning
    return httpx2.Response(200, json=body)


def _error(status):
    """Return an OpenRouter-style error response whose body echoes the API key."""
    return httpx2.Response(status, json={"error": {"message": f"failed for key {FAKE_KEY}"}})


def _timeout():
    """Return a transport-level read timeout, as raised when OpenRouter is too slow."""
    return httpx2.ReadTimeout("timed out")


class FakeOpenRouter:
    """Mocked HTTP layer: replays queued responses and records every request it gets."""

    def __init__(self, *responses):
        """Queue the responses (or exceptions) to return, one per request."""
        self.responses = list(responses)
        self.requests = []

    def handle(self, request):
        """Record the request and return, or raise, the next queued response."""
        self.requests.append(request)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def client(self):
        """Return an llm client whose HTTP calls go to this fake instead of the network."""
        transport = httpx2.MockTransport(self.handle)
        return llm.make_client(FAKE_KEY, http_client=httpx2.Client(transport=transport))


def test_complete_returns_assistant_text():
    """A valid request returns the assistant's reply text."""
    fake = FakeOpenRouter(_reply("Tell me about a time you led a team."))
    text = llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert text == "Tell me about a time you led a team."


def test_complete_sends_arguments_as_given():
    """Model, messages, reasoning effort and max_tokens reach OpenRouter unchanged."""
    fake = FakeOpenRouter(_reply("ok"))
    llm.complete(MESSAGES, "openai/gpt-5-nano", "low", 128, client=fake.client())
    (request,) = fake.requests
    assert str(request.url) == f"{llm.OPENROUTER_BASE_URL}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-5-nano"
    assert body["messages"] == MESSAGES
    assert body["reasoning"] == {"effort": "low"}
    # gpt-5 models ignore it, so it is not sent at all (T2.4).
    assert "temperature" not in body
    assert body["max_tokens"] == 128
    assert "stream" not in body
    assert "stream_options" not in body


def _sent_max_tokens(call, requested):
    """Make one request asking for `requested` tokens and return the max_tokens actually sent."""
    fake = FakeOpenRouter(_reply("ok") if call == "complete" else _sse(_chunk("ok")))
    if call == "complete":
        llm.complete(MESSAGES, DEFAULT_MODEL, "medium", requested, client=fake.client())
    else:
        list(llm.stream(MESSAGES, DEFAULT_MODEL, "medium", requested, client=fake.client()))
    (request,) = fake.requests
    return json.loads(request.content)["max_tokens"]


@pytest.mark.parametrize("call", ["complete", "stream"])
def test_max_tokens_above_the_cap_is_clamped(call):
    """A request for more tokens than MAX_TOKENS_CAP sends the cap from config.py instead."""
    assert _sent_max_tokens(call, config.MAX_TOKENS_CAP + 1000) == config.MAX_TOKENS_CAP


@pytest.mark.parametrize("call", ["complete", "stream"])
def test_max_tokens_at_or_below_the_cap_is_sent_unchanged(call):
    """A request at or under the cap keeps the number it asked for."""
    assert _sent_max_tokens(call, config.MAX_TOKENS_CAP) == config.MAX_TOKENS_CAP
    assert _sent_max_tokens(call, 300) == 300


def test_cap_is_read_at_call_time(monkeypatch):
    """Changing llm's cap changes what is sent, so the limit is not fixed when llm is imported."""
    # llm imports MAX_TOKENS_CAP by name, so patching src.config at runtime would not reach it.
    monkeypatch.setattr(llm, "MAX_TOKENS_CAP", 500)
    assert _sent_max_tokens("stream", 4000) == 500


@pytest.mark.parametrize("model", ["openai/gpt-4o", "anthropic/claude-x", "", "OPENAI/GPT-5-MINI"])
def test_disallowed_model_raises_before_http_call(model):
    """A model outside the allowed list raises before any request is sent."""
    fake = FakeOpenRouter(_reply("should not be sent"))
    with pytest.raises(llm.InvalidModelError):
        llm.complete(MESSAGES, model, "medium", 256, client=fake.client())
    assert fake.requests == []


@pytest.mark.parametrize(
    ("make_failure", "error_class"),
    [
        (_timeout, llm.LLMTimeoutError),
        (lambda: _error(401), llm.LLMAuthError),
        (lambda: _error(429), llm.LLMRateLimitError),
        (lambda: _error(500), llm.LLMServerError),
        (lambda: _error(503), llm.LLMServerError),
    ],
    ids=["timeout", "401", "429", "500", "503"],
)
def test_failures_raise_their_own_readable_error(make_failure, error_class):
    """Timeout, 401, 429 and 5xx each raise a distinct error with a readable message."""
    fake = FakeOpenRouter(*[make_failure() for _ in range(3)])
    with pytest.raises(error_class) as excinfo:
        llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    message = str(excinfo.value)
    assert message.endswith(".") and len(message) > 20
    assert FAKE_KEY not in message


def test_error_classes_are_distinct():
    """No mapped error class is a subclass of another, so callers can tell them apart."""
    classes = [llm.LLMTimeoutError, llm.LLMAuthError, llm.LLMRateLimitError, llm.LLMServerError]
    for a in classes:
        for b in classes:
            assert a is b or not issubclass(a, b)
        assert issubclass(a, llm.LLMError)


@pytest.mark.parametrize(
    "failure",
    [_error(400), httpx2.ConnectError("no route")],
    ids=["400", "connection"],
)
def test_other_failures_raise_generic_llm_error(failure):
    """Errors without their own class still raise a readable LLMError."""
    fake = FakeOpenRouter(failure)
    with pytest.raises(llm.LLMError) as excinfo:
        llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert type(excinfo.value) is llm.LLMError
    assert FAKE_KEY not in str(excinfo.value)


@pytest.mark.parametrize("text", [None, "", "  \n\t"], ids=["none", "empty", "whitespace"])
def test_empty_reply_raises_llm_error(text):
    """A reply with no text, or only whitespace, raises LLMError instead of returning it."""
    fake = FakeOpenRouter(_reply(text))
    with pytest.raises(llm.LLMError, match="empty answer"):
        llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            httpx2.Response(200, text="<html>down</html>", headers={"content-type": "text/html"}),
            "unreadable answer",
        ),
        (
            httpx2.Response(
                200, content=b"{not json", headers={"content-type": "application/json"}
            ),
            "unreadable answer",
        ),
        (httpx2.Response(200, json={"choices": [{"index": 0}]}), "empty answer"),
        (httpx2.Response(200, json={"choices": "oops"}), "empty answer"),
        (
            httpx2.Response(200, json={"choices": [{"index": 0, "message": {"content": [1]}}]}),
            "empty answer",
        ),
    ],
    ids=["html", "bad-json", "no-message", "choices-not-list", "content-not-text"],
)
def test_malformed_200_raises_llm_error(response, message):
    """A 200 reply that is not a usable chat completion raises a readable LLMError."""
    fake = FakeOpenRouter(response)
    with pytest.raises(llm.LLMError, match=message) as excinfo:
        llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert type(excinfo.value) is llm.LLMError
    assert FAKE_KEY not in str(excinfo.value)
    assert len(fake.requests) == 1


@pytest.mark.parametrize(
    ("status", "error_class"),
    [
        (429, llm.LLMRateLimitError),
        (500, llm.LLMServerError),
        (502, llm.LLMServerError),
        (503, llm.LLMServerError),
    ],
)
def test_retryable_errors_are_retried_twice_then_raise(status, error_class, sleeps):
    """429 and 5xx are retried at most 2 times with growing backoff, then raise."""
    fake = FakeOpenRouter(*[_error(status) for _ in range(4)])
    with pytest.raises(error_class):
        llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert len(fake.requests) == 1 + llm.MAX_RETRIES == 3
    assert sleeps == [llm.BACKOFF_SECONDS, 2 * llm.BACKOFF_SECONDS]


@pytest.mark.parametrize("status", [429, 500])
def test_retry_succeeds_after_transient_error(status, sleeps):
    """A transient 429/5xx followed by a good reply returns the reply text."""
    fake = FakeOpenRouter(_error(status), _reply("Second time lucky."))
    text = llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert text == "Second time lucky."
    assert len(fake.requests) == 2
    assert sleeps == [llm.BACKOFF_SECONDS]


@pytest.mark.parametrize(
    "make_failure",
    [lambda: _error(401), _timeout, lambda: _error(400)],
    ids=["401", "timeout", "400"],
)
def test_non_retryable_errors_are_not_retried(make_failure, sleeps):
    """401, timeouts and other client errors fail after a single request, with no wait."""
    fake = FakeOpenRouter(*[make_failure() for _ in range(3)])
    with pytest.raises(llm.LLMError):
        llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert len(fake.requests) == 1
    assert sleeps == []


def test_sdk_retries_are_disabled():
    """The SDK's built-in retries are off, so only complete() decides how often to retry."""
    assert llm.make_client(FAKE_KEY).max_retries == 0


# Keys that cannot go in an HTTP header, written as escapes so the bad characters stay visible.
BAD_HEADER_KEYS = {
    "smart-quotes": "sk-test-\u201cnot-real\u201d",
    "zero-width-space": "sk-test-not-real\u200b",
    "newline": "sk-test-not\nreal",
    "carriage-return": "sk-test-not\rreal",
    "nul": "sk-test-not\x00real",
}


@pytest.mark.parametrize("key", list(BAD_HEADER_KEYS.values()), ids=list(BAD_HEADER_KEYS))
def test_bad_header_key_raises_llm_auth_error(key):
    """A key with characters that cannot go in an HTTP header is rejected with LLMAuthError."""
    with pytest.raises(llm.LLMAuthError, match="invalid characters") as excinfo:
        llm.make_client(key)
    assert config.API_KEY_NAME in str(excinfo.value)
    assert key not in str(excinfo.value)


@pytest.mark.parametrize("key", list(BAD_HEADER_KEYS.values()), ids=list(BAD_HEADER_KEYS))
def test_bad_header_key_from_secrets_fails_before_any_request(key, no_env_key, monkeypatch):
    """Through complete(), a key unfit for a header raises LLMAuthError before any client exists."""
    monkeypatch.setattr(config.st, "secrets", {config.API_KEY_NAME: key})
    monkeypatch.setattr(llm, "OpenAI", lambda **kwargs: pytest.fail("client was built"))
    with pytest.raises(llm.LLMAuthError, match="invalid characters") as excinfo:
        llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256)
    assert key not in str(excinfo.value)


def test_missing_api_key_raises_llm_auth_error(no_env_key, monkeypatch):
    """With no key in st.secrets or the environment, complete() raises a readable LLMAuthError."""
    monkeypatch.setattr(config.st, "secrets", {})
    with pytest.raises(llm.LLMAuthError) as excinfo:
        llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256)
    assert config.API_KEY_NAME in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, config.MissingAPIKeyError)


def test_unparseable_secrets_raises_llm_auth_error(monkeypatch):
    """A secrets.toml that cannot be parsed raises LLMAuthError pointing at that file."""

    def broken_secrets():
        """Fail the way get_api_key() does for an unparseable secrets.toml."""
        raise config.SecretsFileError(".streamlit/secrets.toml could not be parsed.")

    monkeypatch.setattr(llm, "get_api_key", broken_secrets)
    with pytest.raises(llm.LLMAuthError, match="secrets.toml"):
        llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256)


@pytest.mark.parametrize("effort", ["none", "xhigh", "Medium", "", "0.7"])
def test_disallowed_effort_raises_before_http_call(effort):
    """A reasoning effort outside the allowed list raises before any request is sent."""
    fake = FakeOpenRouter(_reply("should not be sent"))
    with pytest.raises(llm.InvalidEffortError, match="Reasoning effort"):
        llm.complete(MESSAGES, DEFAULT_MODEL, effort, 256, client=fake.client())
    assert fake.requests == []


@pytest.mark.parametrize("effort", config.REASONING_EFFORTS)
def test_every_allowed_effort_is_sent(effort):
    """Each allowed level reaches OpenRouter as `reasoning.effort`."""
    fake = FakeOpenRouter(_reply("ok"))
    llm.complete(MESSAGES, DEFAULT_MODEL, effort, 256, client=fake.client())
    (request,) = fake.requests
    assert json.loads(request.content)["reasoning"] == {"effort": effort}


@pytest.mark.parametrize("call", [llm.complete, llm.stream], ids=["complete", "stream"])
def test_disallowed_effort_does_not_need_api_key(call, no_env_key, monkeypatch):
    """The effort check runs before the API key is looked up or a client is built."""
    monkeypatch.setattr(llm, "make_client", lambda: pytest.fail("client was built"))
    with pytest.raises(llm.InvalidEffortError):
        call(MESSAGES, DEFAULT_MODEL, "xhigh", 256)


def test_complete_leaves_out_returned_reasoning():
    """Reasoning text OpenRouter returns next to the answer never ends up in the reply."""
    fake = FakeOpenRouter(_reply("Use the STAR method.", reasoning="thinking about STAR..."))
    text = llm.complete(MESSAGES, DEFAULT_MODEL, "high", 256, client=fake.client())
    assert text == "Use the STAR method."


def test_disallowed_model_does_not_need_api_key(no_env_key, monkeypatch):
    """The model check runs before the API key is looked up or a client is built."""
    monkeypatch.setattr(llm, "make_client", lambda: pytest.fail("client was built"))
    with pytest.raises(llm.InvalidModelError):
        llm.complete(MESSAGES, "openai/gpt-4o", "medium", 256)


# --- stream() -------------------------------------------------------------------------------


def _chunk(content=None, *, role=None, choices=True, finish_reason=None):
    """Return one OpenAI-style streamed chat chunk carrying `content` (or none)."""
    delta = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    return {
        "id": "gen-test",
        "object": "chat.completion.chunk",
        "created": 0,
        "model": DEFAULT_MODEL,
        "choices": (
            [{"index": 0, "delta": delta, "finish_reason": finish_reason}] if choices else []
        ),
    }


def _sse_bytes(*events):
    """Encode events (dicts as JSON, strings as raw data) as server-sent event bytes."""
    lines = []
    for event in events:
        data = event if isinstance(event, str) else json.dumps(event)
        lines.append(f"data: {data}\n\n".encode())
    return b"".join(lines)


def _sse(*events, done=True):
    """Return a 200 event-stream response holding `events`, ending with [DONE] if asked."""
    body = _sse_bytes(*events) + (b"data: [DONE]\n\n" if done else b"")
    return httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})


def _sse_then_fail(error, *events):
    """Return an event-stream response that sends `events`, then raises `error` mid-body."""

    def body():
        """Yield the events, then fail the way a dropped or slow connection does."""
        yield _sse_bytes(*events)
        raise error

    return httpx2.Response(200, content=body(), headers={"content-type": "text/event-stream"})


def _stream(fake, model=DEFAULT_MODEL):
    """Start a stream through `fake` and return its pieces as a list."""
    return list(llm.stream(MESSAGES, model, "medium", 256, client=fake.client()))


def test_stream_yields_pieces_in_order():
    """The text pieces come out in the order sent, skipping chunks without text."""
    fake = FakeOpenRouter(
        _sse(
            _chunk(role="assistant"),
            _chunk("Tell me "),
            _chunk(choices=False),
            _chunk(""),
            _chunk("about a time "),
            _chunk("you led a team."),
        )
    )
    assert _stream(fake) == ["Tell me ", "about a time ", "you led a team."]


def test_stream_sends_arguments_and_asks_for_a_stream():
    """Model, messages, reasoning effort and max_tokens are sent unchanged, with stream on."""
    fake = FakeOpenRouter(_sse(_chunk("ok")))
    list(llm.stream(MESSAGES, "openai/gpt-5-nano", "low", 128, client=fake.client()))
    (request,) = fake.requests
    body = json.loads(request.content)
    assert body["stream"] is True
    assert body["model"] == "openai/gpt-5-nano"
    assert body["messages"] == MESSAGES
    assert body["reasoning"] == {"effort": "low"}
    # gpt-5 models ignore it, so it is not sent at all (T2.4).
    assert "temperature" not in body
    assert body["max_tokens"] == 128


def test_stream_sends_nothing_until_first_piece_is_asked_for():
    """The request goes out on the first next(), not when stream() is called."""
    fake = FakeOpenRouter(_sse(_chunk("ok")))
    pieces = llm.stream(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert fake.requests == []
    assert next(pieces) == "ok"
    assert len(fake.requests) == 1


def test_stream_disallowed_model_raises_at_call_time():
    """A model outside the allowed list raises when stream() is called, before any request."""
    fake = FakeOpenRouter(_sse(_chunk("should not be sent")))
    with pytest.raises(llm.InvalidModelError):
        llm.stream(MESSAGES, "openai/gpt-4o", "medium", 256, client=fake.client())
    assert fake.requests == []


def test_stream_disallowed_effort_raises_at_call_time():
    """An effort outside the allowed list raises when stream() is called, before any request."""
    fake = FakeOpenRouter(_sse(_chunk("should not be sent")))
    with pytest.raises(llm.InvalidEffortError):
        llm.stream(MESSAGES, DEFAULT_MODEL, "xhigh", 256, client=fake.client())
    assert fake.requests == []


def test_stream_missing_key_raises_at_call_time(no_env_key, monkeypatch):
    """With no API key, stream() raises LLMAuthError at once, not on the first next()."""
    monkeypatch.setattr(config.st, "secrets", {})
    with pytest.raises(llm.LLMAuthError):
        llm.stream(MESSAGES, DEFAULT_MODEL, "medium", 256)


def test_stream_leaves_out_streamed_reasoning():
    """Chunks carrying only the model's thinking yield nothing; the pieces are the answer alone."""
    thinking = _chunk()
    thinking["choices"][0]["delta"] = {"reasoning": "thinking about STAR...", "content": None}
    fake = FakeOpenRouter(_sse(_chunk(role="assistant"), thinking, _chunk("Use the STAR method.")))
    assert _stream(fake) == ["Use the STAR method."]


def test_stream_error_event_mid_reply_raises_readable_llm_error():
    """An error event after some text raises LLMError with a fixed message, not the raw one."""
    fake = FakeOpenRouter(
        _sse(_chunk("Partial "), {"error": {"message": f"boom for key {FAKE_KEY}"}}, done=False)
    )
    pieces = llm.stream(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert next(pieces) == "Partial "
    with pytest.raises(llm.LLMError) as excinfo:
        next(pieces)
    assert type(excinfo.value) is llm.LLMError
    assert "boom" not in str(excinfo.value)
    assert FAKE_KEY not in str(excinfo.value)


@pytest.mark.parametrize(
    ("error", "error_class"),
    [
        (httpx2.ReadTimeout("timed out"), llm.LLMTimeoutError),
        (httpx2.ReadError("connection reset"), llm.LLMError),
    ],
    ids=["timeout", "dropped"],
)
def test_stream_transport_failure_mid_reply_raises_mapped_error(error, error_class):
    """A timeout or dropped connection after some text raises the mapped LLMError."""
    fake = FakeOpenRouter(_sse_then_fail(error, _chunk("Partial ")))
    pieces = llm.stream(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert next(pieces) == "Partial "
    with pytest.raises(error_class) as excinfo:
        next(pieces)
    assert type(excinfo.value) is error_class
    assert len(fake.requests) == 1


def test_stream_unparseable_event_raises_unreadable():
    """An event whose data is not JSON raises LLMError saying the answer was unreadable."""
    fake = FakeOpenRouter(_sse(_chunk("Partial "), "{not json", done=False))
    with pytest.raises(llm.LLMError, match="unreadable answer"):
        _stream(fake)


@pytest.mark.parametrize(
    "events",
    [(), (_chunk(role="assistant"),), (_chunk("  "), _chunk("\n"))],
    ids=["no-chunks", "role-only", "whitespace"],
)
def test_stream_without_text_raises_empty_answer(events):
    """A stream that ends without any real text raises LLMError saying the answer was empty."""
    fake = FakeOpenRouter(_sse(*events))
    with pytest.raises(llm.LLMError, match="empty answer"):
        _stream(fake)


@pytest.mark.parametrize("status", [429, 500])
def test_stream_retries_before_the_first_piece(status, sleeps):
    """A 429/5xx when the request is sent is retried, and the next try's stream comes through."""
    fake = FakeOpenRouter(_error(status), _sse(_chunk("Second time lucky.")))
    assert _stream(fake) == ["Second time lucky."]
    assert len(fake.requests) == 2
    assert sleeps == [llm.BACKOFF_SECONDS]


@pytest.mark.parametrize(
    ("make_failure", "error_class", "requests"),
    [
        (lambda: _error(401), llm.LLMAuthError, 1),
        (_timeout, llm.LLMTimeoutError, 1),
        (lambda: _error(503), llm.LLMServerError, 1 + llm.MAX_RETRIES),
    ],
    ids=["401", "timeout", "503"],
)
def test_stream_request_failures_raise_mapped_error(make_failure, error_class, requests):
    """Failures when sending the request raise the same mapped errors as complete()."""
    fake = FakeOpenRouter(*[make_failure() for _ in range(3)])
    with pytest.raises(error_class):
        _stream(fake)
    assert len(fake.requests) == requests


def test_stream_closed_early_closes_the_response():
    """Stopping the generator after one piece closes the HTTP response."""
    # A body fed from an iterator stays open until read or closed (a bytes body never is).
    body = iter([_sse_bytes(_chunk("One ")), _sse_bytes(_chunk("two."), "[DONE]")])
    response = httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})
    fake = FakeOpenRouter(response)
    pieces = llm.stream(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert next(pieces) == "One "
    assert not response.is_closed
    pieces.close()
    assert response.is_closed


# --- token usage and finish_reason ----------------------------------------------------------

USAGE = {
    "prompt_tokens": 120,
    "completion_tokens": 45,
    "total_tokens": 165,
    "completion_tokens_details": {"reasoning_tokens": 20},
}


def _llm_records(caplog):
    """Return only the log records from src.llm (the HTTP library logs its requests at INFO)."""
    return [record for record in caplog.records if record.name == "src.llm"]


def _usage_chunk(usage=USAGE):
    """Return the last, textless chunk a stream sends when include_usage is on."""
    return {**_chunk(choices=False), "usage": usage}


def test_stream_asks_for_usage():
    """A streaming request turns on include_usage, or OpenRouter would report no token counts."""
    fake = FakeOpenRouter(_sse(_chunk("ok")))
    _stream(fake)
    (request,) = fake.requests
    assert json.loads(request.content)["stream_options"] == {"include_usage": True}


def test_stream_reads_usage_from_the_last_chunk():
    """The token counts in the final usage chunk are available once the stream has ended."""
    fake = FakeOpenRouter(
        _sse(_chunk("Hi "), _chunk("there.", finish_reason="stop"), _usage_chunk())
    )
    stream = llm.stream(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert stream.usage is None
    assert list(stream) == ["Hi ", "there."]
    assert stream.usage == llm.Usage(
        prompt_tokens=120, completion_tokens=45, total_tokens=165, reasoning_tokens=20
    )
    assert stream.finish_reason == "stop"


def test_stream_without_usage_does_not_crash():
    """A stream that reports no usage still gives its text, and usage stays None."""
    fake = FakeOpenRouter(_sse(_chunk("Hi.")))
    stream = llm.stream(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert list(stream) == ["Hi."]
    assert stream.usage is None
    assert stream.finish_reason is None


@pytest.mark.parametrize(
    "usage",
    [
        None,
        "lots",
        {"prompt_tokens": 1, "completion_tokens": 2},
        # The SDK turns "1" or True into 1 itself, so only values it cannot convert get here.
        {"prompt_tokens": "many", "completion_tokens": 2, "total_tokens": 3},
        {"prompt_tokens": 2.5, "completion_tokens": 2, "total_tokens": 3},
        {"prompt_tokens": -1, "completion_tokens": 2, "total_tokens": 3},
    ],
    ids=["null", "string", "missing-total", "word-count", "fraction", "negative"],
)
def test_malformed_usage_is_ignored(usage):
    """Usage that is missing a count or holds a wrong type is treated as not reported."""
    fake = FakeOpenRouter(_sse(_chunk("Hi."), _usage_chunk(usage)))
    stream = llm.stream(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert list(stream) == ["Hi."]
    assert stream.usage is None


def test_usage_without_reasoning_details_has_no_reasoning_count():
    """Usage without completion_tokens_details gives the counts, with reasoning_tokens None."""
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    fake = FakeOpenRouter(_sse(_chunk("Hi."), _usage_chunk(usage)))
    stream = llm.stream(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    list(stream)
    assert stream.usage == llm.Usage(10, 5, 15, None)


def test_usage_log_leaves_out_an_unreported_reasoning_count(caplog):
    """Without a reasoning count the log line just gives the three totals."""
    usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
    fake = FakeOpenRouter(_sse(_chunk("Hi."), _usage_chunk(usage)))
    with caplog.at_level("INFO", logger="src.llm"):
        _stream(fake)
    (record,) = _llm_records(caplog)
    assert record.getMessage() == (
        f"Token usage ({DEFAULT_MODEL}): prompt 10, completion 5, total 15"
    )


def test_stream_logs_usage_without_the_text(caplog):
    """The token counts are logged once at INFO, and neither the messages nor the reply are."""
    fake = FakeOpenRouter(_sse(_chunk("Secret reply."), _usage_chunk()))
    with caplog.at_level("INFO", logger="src.llm"):
        _stream(fake)
    (record,) = _llm_records(caplog)
    assert record.levelname == "INFO"
    message = record.getMessage()
    assert "prompt 120" in message
    assert "completion 45 (reasoning 20)" in message
    assert "total 165" in message
    assert "Secret reply" not in message
    assert MESSAGES[0]["content"] not in message


def test_complete_logs_usage(caplog):
    """complete() logs the token counts from its response too."""
    fake = FakeOpenRouter(_reply("ok", usage=USAGE))
    with caplog.at_level("INFO", logger="src.llm"):
        assert llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client()) == "ok"
    (record,) = _llm_records(caplog)
    assert "total 165" in record.getMessage()


def test_complete_without_usage_logs_nothing(caplog):
    """A response without usage does not crash and logs no usage line."""
    fake = FakeOpenRouter(_reply("ok"))
    with caplog.at_level("INFO", logger="src.llm"):
        assert llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client()) == "ok"
    assert _llm_records(caplog) == []


def test_stream_cut_off_by_the_token_limit_keeps_its_text():
    """A reply stopped by max_tokens still yields its text, with finish_reason "length"."""
    fake = FakeOpenRouter(_sse(_chunk("Half an "), _chunk("answer", finish_reason="length")))
    stream = llm.stream(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert list(stream) == ["Half an ", "answer"]
    assert stream.finish_reason == "length"


def test_stream_cut_off_before_any_text_says_so():
    """No text plus finish_reason "length" (all tokens spent thinking) gets its own message."""
    fake = FakeOpenRouter(_sse(_chunk(role="assistant"), _chunk(finish_reason="length")))
    with pytest.raises(llm.LLMError, match="used up its token limit"):
        _stream(fake)


def test_stream_without_text_still_logs_its_usage(caplog):
    """A reply that spent its whole budget thinking is still logged: it is the costliest case."""
    events = (_chunk(role="assistant"), _chunk(finish_reason="length"), _usage_chunk())
    fake = FakeOpenRouter(_sse(*events))
    with caplog.at_level("INFO", logger="src.llm"), pytest.raises(llm.LLMError):
        _stream(fake)
    (record,) = _llm_records(caplog)
    assert record.levelname == "INFO"
    assert "total 165" in record.getMessage()


def test_complete_logs_a_reply_cut_off_with_text(caplog):
    """complete() returns a cut-off reply's text but logs a warning that it was cut off."""
    fake = FakeOpenRouter(_reply("Half an answer", finish_reason="length"))
    with caplog.at_level("INFO", logger="src.llm"):
        text = llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert text == "Half an answer"
    (record,) = _llm_records(caplog)
    assert record.levelname == "WARNING"
    assert "cut off" in record.getMessage()
    assert "Half an answer" not in record.getMessage()


def test_complete_cut_off_before_any_text_says_so():
    """complete() gives the same token-limit message for an empty reply stopped by max_tokens."""
    fake = FakeOpenRouter(_reply(None, finish_reason="length"))
    with pytest.raises(llm.LLMError, match="used up its token limit"):
        llm.complete(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())


def test_closed_stream_has_no_reference_cycle():
    """Dropping an unfinished stream closes its response at once, without waiting for the GC."""
    body = iter([_sse_bytes(_chunk("One ")), _sse_bytes(_chunk("two."), "[DONE]")])
    response = httpx2.Response(200, content=body, headers={"content-type": "text/event-stream"})
    fake = FakeOpenRouter(response)
    stream = llm.stream(MESSAGES, DEFAULT_MODEL, "medium", 256, client=fake.client())
    assert next(stream) == "One "
    gc_was_on = gc.isenabled()
    gc.disable()
    try:
        del stream
        assert response.is_closed
    finally:
        if gc_was_on:
            gc.enable()
