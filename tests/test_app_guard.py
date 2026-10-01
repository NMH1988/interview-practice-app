from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import llm
from src.config import API_KEY_NAME, DEFAULT_MODEL, MAX_INPUT_CHARS

APP = Path(__file__).resolve().parent.parent / "app.py"
FAKE_KEY = "sk-test-not-a-real-key"

pytestmark = pytest.mark.usefixtures("no_env_key")


def send(text: str) -> AppTest:
    """Run the app, submit `text` in the chat input and return the finished run."""
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = FAKE_KEY
    at.run(timeout=30)
    at.chat_input[0].set_value(text).run(timeout=30)
    return at


@pytest.mark.parametrize(
    ("text", "expected"),
    [("   \n\t ", "type a message"), ("a" * (MAX_INPUT_CHARS + 1), "too long")],
    ids=["whitespace-only", "too-long"],
)
def test_rejected_input_shows_message_and_skips_llm(fake_llm, text, expected):
    """Whitespace-only or too-long input shows the guard's warning and never calls the LLM."""
    at = send(text)
    assert not at.exception
    assert len(at.warning) == 1
    assert expected in at.warning[0].value
    assert fake_llm.calls == []
    assert len(at.chat_message) == 0


def test_valid_input_reaches_llm_once_with_cleaned_text(fake_llm):
    """A valid message is cleaned, sent to the LLM once, and the reply is shown."""
    at = send("  I led the" + chr(0x07) + " migration.\r\nIt went well.  ")
    assert not at.exception
    assert not at.warning
    assert len(fake_llm.calls) == 1
    call = fake_llm.calls[0]
    assert call["messages"] == [{"role": "user", "content": "I led the migration.\nIt went well."}]
    assert call["model"] == DEFAULT_MODEL
    assert at.chat_message[1].markdown[0].value == fake_llm.reply


def test_llm_error_shows_only_its_message(monkeypatch):
    """An LLMError from the client is shown as its fixed text, without crashing the app."""

    def failing_complete(*args, **kwargs):
        """Fail the way llm.complete does when OpenRouter times out."""
        raise llm.LLMTimeoutError("The AI service took too long to answer. Please try again.")

    monkeypatch.setattr(llm, "complete", failing_complete)
    at = send("Tell me about yourself.")
    assert not at.exception
    assert len(at.error) == 1
    assert "took too long" in at.error[0].value
    assert FAKE_KEY not in at.error[0].value
