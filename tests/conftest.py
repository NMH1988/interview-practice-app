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
    # What the stream reports once all pieces are read, like llm.ReplyStream.
    usage = llm.Usage(
        prompt_tokens=1234, completion_tokens=567, total_tokens=1801, reasoning_tokens=320
    )
    finish_reason = "stop"

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
        end = llm._StreamEnd()
        return llm.ReplyStream(self._pieces(reply, end), end)

    def _pieces(self, reply, end):
        """Yield the reply word by word, then report usage; fail like `llm.stream` if no text."""
        # Several pieces, so the app really has to join them; spaces stay with their word.
        pieces = re.findall(r"\s*\S+\s*", reply)
        if not pieces:
            # The same message llm.stream gives, which depends on why the reply had no text.
            raise llm._no_text_error(self.finish_reason)
        yield from pieces
        # Only now, as the real stream gets them from its last chunks.
        end.usage = self.usage
        end.finish_reason = self.finish_reason


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace `llm.complete` and `llm.stream` with a FakeLLM so no test reaches OpenRouter."""
    fake = FakeLLM()
    # app.py looks up `llm.stream` on the module when it calls it, so patching the attribute
    # is enough.
    monkeypatch.setattr(llm, "complete", fake)
    monkeypatch.setattr(llm, "stream", fake.stream)
    return fake
