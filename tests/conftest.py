import re

import pytest

from src import llm
from src.config import API_KEY_NAME


@pytest.fixture
def no_env_key(monkeypatch):
    """Remove the API key env var so tests only see the key they set up."""
    monkeypatch.delenv(API_KEY_NAME, raising=False)


class FakeLLM:
    """Stand-in for `llm.complete` and `llm.stream`: records each call, gives a fixed reply."""

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

    def stream(self, messages, model, temperature, max_tokens, **kwargs):
        """Record the call like `__call__` and return the fixed reply in word-sized pieces."""
        reply = self(messages, model, temperature, max_tokens, **kwargs)
        return self._pieces(reply)

    @staticmethod
    def _pieces(reply):
        """Yield the reply word by word, or fail like `llm.stream` if it has no text."""
        # Several pieces, so the app really has to join them; spaces stay with their word.
        pieces = re.findall(r"\s*\S+\s*", reply)
        if not pieces:
            raise llm.LLMError("The AI service returned an empty answer. Please try again.")
        yield from pieces


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace `llm.complete` and `llm.stream` with a FakeLLM so no test reaches OpenRouter."""
    fake = FakeLLM()
    # app.py looks up `llm.stream` on the module when it calls it, so patching the attribute
    # is enough.
    monkeypatch.setattr(llm, "complete", fake)
    monkeypatch.setattr(llm, "stream", fake.stream)
    return fake
