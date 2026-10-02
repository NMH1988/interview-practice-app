from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import llm, rate_limit
from src.config import API_KEY_NAME, RATE_LIMIT_PER_MINUTE

APP = Path(__file__).resolve().parent.parent / "app.py"
FAKE_KEY = "sk-test-not-a-real-key"
# What an SDK error chains under ours: the raw response body, here holding the key.
RAW_BODY = f'{{"error": {{"message": "upstream exploded"}}, "auth": "Bearer {FAKE_KEY}"}}'
NOT_SENT = "Your message was not sent. Copy it from here to keep it:"
NO_ANSWER = "Your message got no answer. Copy it from here to keep it:"

# One of each LLMError the app can get, with the fixed text llm.py gives it.
LLM_ERRORS = [
    llm.LLMError("The request to the AI service failed. Please try again."),
    llm.InvalidModelError("Model 'x' is not allowed. Choose one of the listed models."),
    llm.LLMTimeoutError("The AI service took too long to respond. Please try again."),
    llm.LLMAuthError(
        f"The OpenRouter API key was rejected. Check {API_KEY_NAME} and reload the page."
    ),
    llm.LLMRateLimitError(
        "The AI service is receiving too many requests. Please wait a moment and try again."
    ),
    llm.LLMServerError("The AI service is having problems right now. Please try again later."),
]

pytestmark = pytest.mark.usefixtures("no_env_key")


def start(request_times: list[float] | None = None) -> AppTest:
    """Run the app once with a fake key, optionally with earlier requests already counted."""
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = FAKE_KEY
    if request_times is not None:
        at.session_state["request_times"] = request_times
    at.run(timeout=30)
    assert not at.exception
    return at


def say(at: AppTest, text: str) -> None:
    """Submit `text` in the chat input and wait for the run (and its rerun) to finish."""
    at.chat_input[0].set_value(text).run(timeout=30)
    assert not at.exception


def notice_style(at: AppTest) -> tuple[str, str, str]:
    """Return the one notice in the main area as (element type, icon, title)."""
    alerts = [*at.main.warning, *at.main.error]
    assert len(alerts) == 1
    return alerts[0].type, alerts[0].icon, alerts[0].proto.title


def everything_shown(at: AppTest) -> str:
    """Return all the text the run put on screen, alert titles included, as one string."""
    alerts = [*at.error, *at.warning, *at.info]
    texts = [element.value for element in [*alerts, *at.code, *at.caption, *at.markdown]]
    return "\n".join(texts + [alert.proto.title for alert in alerts])


def raising_stream(error: llm.LLMError, pieces_before: list[str]):
    """Return an llm.stream stand-in that sends some pieces, then raises `error` over RAW_BODY."""

    def stream(*args, **kwargs):
        """Yield the given pieces, then fail with the raw SDK error chained, as llm.py does."""
        yield from pieces_before
        raise error from RuntimeError(RAW_BODY)

    return stream


def test_guard_block_rate_limit_llm_error_and_interruption_each_look_different(
    monkeypatch, fake_llm
):
    """Each kind of notice has its own element type, icon and title, so they can be told apart."""
    styles = {}

    at = start()
    say(at, "   ")
    styles["guard"] = notice_style(at)

    monkeypatch.setattr(rate_limit, "clock", lambda: 1000.0)
    at = start([1000.0] * RATE_LIMIT_PER_MINUTE)
    say(at, "One too many.")
    styles["rate_limit"] = notice_style(at)

    monkeypatch.setattr(llm, "stream", raising_stream(LLM_ERRORS[0], []))
    at = start()
    say(at, "My answer.")
    styles["llm"] = notice_style(at)

    def interrupted_stream(*args, **kwargs):
        """Send one piece, then request a rerun the way a sidebar click does mid-stream."""
        yield "Half of "
        st.rerun()

    monkeypatch.setattr(llm, "stream", interrupted_stream)
    at = start()
    say(at, "My answer.")
    styles["interrupted"] = notice_style(at)

    assert styles == {
        "guard": ("warning", "✋", "Message not sent"),
        "rate_limit": ("warning", "⏳", "Message limit reached"),
        "llm": ("error", "⚠️", "AI service problem"),
        "interrupted": ("error", "⏹️", "Answer interrupted"),
    }
    assert len(set(styles.values())) == len(styles)


@pytest.mark.parametrize("pieces_before", [[], ["Half of ", "a reply"]], ids=["first", "mid"])
@pytest.mark.parametrize("error", LLM_ERRORS, ids=lambda exc: type(exc).__name__)
def test_each_llm_error_shows_only_its_own_text(monkeypatch, fake_llm, error, pieces_before):
    """Every LLMError shows as an error with its fixed text, never the key or the raw body."""
    monkeypatch.setattr(llm, "stream", raising_stream(error, pieces_before))
    at = start()
    say(at, "My answer.")
    assert notice_style(at) == ("error", "⚠️", "AI service problem")
    assert at.error[0].value == str(error)
    assert at.caption[0].value == NO_ANSWER
    assert at.code[0].value == "My answer."
    shown = everything_shown(at)
    assert FAKE_KEY not in shown
    assert "upstream exploded" not in shown
    # The half reply is gone too: only the error remains.
    assert "Half of" not in shown
    assert len(at.chat_message) == 0


def test_rate_limit_caption_says_the_message_was_not_sent(monkeypatch, fake_llm):
    """A rate-limited message never left the app, so its copy box says it was not sent."""
    monkeypatch.setattr(rate_limit, "clock", lambda: 1000.0)
    at = start([1000.0] * RATE_LIMIT_PER_MINUTE)
    say(at, "One too many.")
    assert at.caption[0].value == NOT_SENT
    assert at.code[0].value == "One too many."
