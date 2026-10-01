import pytest

from src import llm
from src.config import API_KEY_NAME


@pytest.fixture
def no_env_key(monkeypatch):
    """Remove the API key env var so tests only see the key they set up."""
    monkeypatch.delenv(API_KEY_NAME, raising=False)


class FakeLLM:
    """Stand-in for `llm.complete` that records each call and returns a fixed reply."""

    reply = "Fake coach reply."

    def __init__(self):
        """Start with no recorded calls."""
        self.calls: list[dict] = []

    def __call__(self, messages, model, temperature, max_tokens, **kwargs):
        """Record the call's arguments and return the fixed reply."""
        self.calls.append(
            {
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                **kwargs,
            }
        )
        return self.reply


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace `llm.complete` with a FakeLLM so no test ever reaches OpenRouter."""
    fake = FakeLLM()
    # app.py looks up `llm.complete` on the module when it calls it, so patching the attribute
    # is enough.
    monkeypatch.setattr(llm, "complete", fake)
    return fake
