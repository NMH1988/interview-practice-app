import logging
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import guard, llm
from src.config import (
    API_KEY_NAME,
    DEFAULT_MODEL,
    DEFAULT_ROLE,
    DEFAULT_SENIORITY,
    MAX_INPUT_CHARS,
)
from src.prompts import INTERVIEW_TYPES, build_user_prompt
from tests.injection_samples import ATTACKS

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
    clean = "I led the migration.\nIt went well."
    # The cleaned text reaches the LLM inside the built user prompt; the chat shows it as is.
    assert call["messages"][-1] == {
        "role": "user",
        "content": build_user_prompt(DEFAULT_ROLE, INTERVIEW_TYPES[0], DEFAULT_SENIORITY, clean),
    }
    assert call["model"] == DEFAULT_MODEL
    assert at.chat_message[0].markdown[0].value == clean
    assert at.chat_message[1].markdown[0].value == fake_llm.reply


def test_any_guard_error_shows_warning_and_skips_llm(monkeypatch, fake_llm):
    """Every GuardError subclass (e.g. T4.2's blocks) is shown as a warning, not a crash."""

    class OtherBlock(guard.GuardError):
        """A guard block that is not an InvalidInputError."""

    def blocking_validate(text, max_chars=guard.MAX_INPUT_CHARS):
        """Block every message the way a future guard check would."""
        raise OtherBlock("That message was blocked.")

    # app.py re-imports validate_input from the module on every run, so this patch reaches it.
    monkeypatch.setattr(guard, "validate_input", blocking_validate)
    at = send("Tell me about yourself.")
    assert not at.exception
    assert len(at.warning) == 1
    assert "was blocked" in at.warning[0].value
    assert fake_llm.calls == []


def test_llm_error_shows_only_its_message(monkeypatch):
    """An LLMError is shown as its fixed text; the chained SDK error (key, body) never is."""
    raw_body = f'{{"error": "upstream timeout", "auth": "Bearer {FAKE_KEY}"}}'

    def failing_stream(*args, **kwargs):
        """Fail the way llm.stream does on a timeout, with the raw SDK error chained."""
        raise llm.LLMTimeoutError(
            "The AI service took too long to respond. Please try again."
        ) from RuntimeError(raw_body)

    monkeypatch.setattr(llm, "stream", failing_stream)
    at = send("Tell me about yourself.")
    assert not at.exception
    assert len(at.error) == 1
    shown = at.error[0].value
    assert "took too long to respond" in shown
    assert FAKE_KEY not in shown
    assert "upstream timeout" not in shown


def test_injection_attempt_shows_neutral_refusal_and_skips_llm(fake_llm, caplog):
    """An attack string shows the neutral refusal, is logged without the key, and is never sent."""
    with caplog.at_level(logging.WARNING, logger="src.guard"):
        at = send(ATTACKS[0])
    assert not at.exception
    assert len(at.warning) == 1
    assert at.warning[0].value == guard.INJECTION_REFUSAL
    assert fake_llm.calls == []
    assert len(at.chat_message) == 0
    # Only the guard's records: Streamlit may log its own warnings while AppTest runs.
    guard_logs = [r.getMessage() for r in caplog.records if r.name == "src.guard"]
    assert guard_logs == [f"Blocked message: patterns=role_override length={len(ATTACKS[0])}"]
    assert FAKE_KEY not in caplog.text
