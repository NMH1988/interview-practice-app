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


def _reply(text):
    """Return a minimal OpenAI-style chat completion response with the given text."""
    return httpx2.Response(
        200,
        json={
            "id": "gen-test",
            "object": "chat.completion",
            "created": 0,
            "model": DEFAULT_MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": text},
                }
            ],
        },
    )


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
    text = llm.complete(MESSAGES, DEFAULT_MODEL, 0.7, 256, client=fake.client())
    assert text == "Tell me about a time you led a team."


def test_complete_sends_arguments_as_given():
    """Model, messages, temperature and max_tokens reach OpenRouter unchanged."""
    fake = FakeOpenRouter(_reply("ok"))
    llm.complete(MESSAGES, "openai/gpt-5-nano", 0.3, 128, client=fake.client())
    (request,) = fake.requests
    assert str(request.url) == f"{llm.OPENROUTER_BASE_URL}/chat/completions"
    assert request.headers["authorization"] == f"Bearer {FAKE_KEY}"
    body = json.loads(request.content)
    assert body["model"] == "openai/gpt-5-nano"
    assert body["messages"] == MESSAGES
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 128


@pytest.mark.parametrize("model", ["openai/gpt-4o", "anthropic/claude-x", "", "OPENAI/GPT-5-MINI"])
def test_disallowed_model_raises_before_http_call(model):
    """A model outside the allowed list raises before any request is sent."""
    fake = FakeOpenRouter(_reply("should not be sent"))
    with pytest.raises(llm.InvalidModelError):
        llm.complete(MESSAGES, model, 0.7, 256, client=fake.client())
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
        llm.complete(MESSAGES, DEFAULT_MODEL, 0.7, 256, client=fake.client())
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
        llm.complete(MESSAGES, DEFAULT_MODEL, 0.7, 256, client=fake.client())
    assert type(excinfo.value) is llm.LLMError
    assert FAKE_KEY not in str(excinfo.value)


@pytest.mark.parametrize("text", [None, "", "  \n\t"], ids=["none", "empty", "whitespace"])
def test_empty_reply_raises_llm_error(text):
    """A reply with no text, or only whitespace, raises LLMError instead of returning it."""
    fake = FakeOpenRouter(_reply(text))
    with pytest.raises(llm.LLMError, match="empty answer"):
        llm.complete(MESSAGES, DEFAULT_MODEL, 0.7, 256, client=fake.client())


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
        llm.complete(MESSAGES, DEFAULT_MODEL, 0.7, 256, client=fake.client())
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
        llm.complete(MESSAGES, DEFAULT_MODEL, 0.7, 256, client=fake.client())
    assert len(fake.requests) == 1 + llm.MAX_RETRIES == 3
    assert sleeps == [llm.BACKOFF_SECONDS, 2 * llm.BACKOFF_SECONDS]


@pytest.mark.parametrize("status", [429, 500])
def test_retry_succeeds_after_transient_error(status, sleeps):
    """A transient 429/5xx followed by a good reply returns the reply text."""
    fake = FakeOpenRouter(_error(status), _reply("Second time lucky."))
    text = llm.complete(MESSAGES, DEFAULT_MODEL, 0.7, 256, client=fake.client())
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
        llm.complete(MESSAGES, DEFAULT_MODEL, 0.7, 256, client=fake.client())
    assert len(fake.requests) == 1
    assert sleeps == []


def test_sdk_retries_are_disabled():
    """The SDK's built-in retries are off, so only complete() decides how often to retry."""
    assert llm.make_client(FAKE_KEY).max_retries == 0


def test_missing_api_key_raises_llm_auth_error(no_env_key, monkeypatch):
    """With no key in st.secrets or the environment, complete() raises a readable LLMAuthError."""
    monkeypatch.setattr(config.st, "secrets", {})
    with pytest.raises(llm.LLMAuthError) as excinfo:
        llm.complete(MESSAGES, DEFAULT_MODEL, 0.7, 256)
    assert config.API_KEY_NAME in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, config.MissingAPIKeyError)


def test_unparseable_secrets_raises_llm_auth_error(monkeypatch):
    """A secrets.toml that cannot be parsed raises LLMAuthError pointing at that file."""

    def broken_secrets():
        """Fail the way get_api_key() does for an unparseable secrets.toml."""
        raise config.SecretsFileError(".streamlit/secrets.toml could not be parsed.")

    monkeypatch.setattr(llm, "get_api_key", broken_secrets)
    with pytest.raises(llm.LLMAuthError, match="secrets.toml"):
        llm.complete(MESSAGES, DEFAULT_MODEL, 0.7, 256)


def test_disallowed_model_does_not_need_api_key(no_env_key, monkeypatch):
    """The model check runs before the API key is looked up or a client is built."""
    monkeypatch.setattr(llm, "make_client", lambda: pytest.fail("client was built"))
    with pytest.raises(llm.InvalidModelError):
        llm.complete(MESSAGES, "openai/gpt-4o", 0.7, 256)
