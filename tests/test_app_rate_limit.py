from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import llm, rate_limit
from src.config import API_KEY_NAME, RATE_LIMIT_PER_MINUTE, RATE_LIMIT_PER_SESSION
from tests.injection_samples import ATTACKS

APP = Path(__file__).resolve().parent.parent / "app.py"

pytestmark = pytest.mark.usefixtures("no_env_key")


class FakeClock:
    """A clock the test moves by hand, so no test waits for real time to pass."""

    def __init__(self, now: float = 1000.0):
        """Start the clock at `now` seconds."""
        self.now = now

    def __call__(self) -> float:
        """Return the current fake time."""
        return self.now


@pytest.fixture
def clock(monkeypatch):
    """Replace `rate_limit.clock` with a FakeClock that the app reads through the module."""
    fake = FakeClock()
    monkeypatch.setattr(rate_limit, "clock", fake)
    return fake


def start(request_times: list[float] | None = None) -> AppTest:
    """Run the app once with a fake key, optionally with earlier requests already counted."""
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = "sk-test-not-a-real-key"
    if request_times is not None:
        at.session_state["request_times"] = request_times
    at.run(timeout=30)
    assert not at.exception
    return at


def say(at: AppTest, text: str) -> None:
    """Submit `text` in the chat input and wait for the run (and its rerun) to finish."""
    at.chat_input[0].set_value(text).run(timeout=30)
    assert not at.exception


def test_eleventh_message_in_a_minute_is_blocked_until_the_window_passes(fake_llm, clock):
    """10 messages in a minute get replies; the 11th shows the wait and skips the LLM."""
    at = start()
    for i in range(RATE_LIMIT_PER_MINUTE):
        say(at, f"Answer {i}.")
    assert len(fake_llm.calls) == RATE_LIMIT_PER_MINUTE
    assert not at.warning
    assert at.session_state.request_times == [clock.now] * RATE_LIMIT_PER_MINUTE

    clock.now += 15
    say(at, "One too many.")
    assert len(fake_llm.calls) == RATE_LIMIT_PER_MINUTE
    assert len(at.warning) == 1
    assert "Please wait 45 seconds" in at.warning[0].value
    # The message was fine, so it is kept for copying, and it is not in the chat or counted.
    assert at.code[0].value == "One too many."
    assert len(at.chat_message) == 2 * RATE_LIMIT_PER_MINUTE
    assert len(at.session_state.request_times) == RATE_LIMIT_PER_MINUTE

    clock.now += 45
    say(at, "After the wait.")
    assert len(fake_llm.calls) == RATE_LIMIT_PER_MINUTE + 1
    assert not at.warning
    assert at.chat_message[-2].markdown[0].value == "After the wait."


def test_used_up_session_locks_the_input(fake_llm, clock):
    """After 50 requests, spread over the session, the input is locked and the cap is explained."""
    earlier = [clock.now - 120.0 * (i + 1) for i in range(RATE_LIMIT_PER_SESSION)]
    at = start(earlier)
    assert at.chat_input[0].disabled
    assert [(info.icon, info.proto.title, info.value) for info in at.info] == [
        ("⏳", "Session limit reached", "You have used all 50 messages for this session.")
    ]
    # The empty chat's example prompts are locked too, so nothing can be sent.
    starters = [b for b in at.main.button if (b.key or "").startswith("example_")]
    assert starters and all(button.disabled for button in starters)
    assert not at.warning


def test_message_queued_before_the_cap_is_still_refused(fake_llm, clock):
    """A message already pending when the cap is hit is refused on send and kept for copying."""
    earlier = [clock.now - 120.0 * (i + 1) for i in range(RATE_LIMIT_PER_SESSION)]
    at = start(earlier)
    at.session_state["pending"] = "My answer."
    at.run(timeout=30)
    assert not at.exception
    assert fake_llm.calls == []
    assert len(at.warning) == 1
    assert at.warning[0].value == "You have used all 50 messages for this session."
    assert at.code[0].value == "My answer."
    assert len(at.chat_message) == 0


def test_last_message_of_the_session_is_still_sent(fake_llm, clock):
    """The 50th request of a session is allowed, and only then is the input locked."""
    earlier = [clock.now - 120.0 * (i + 1) for i in range(RATE_LIMIT_PER_SESSION - 1)]
    at = start(earlier)
    assert not at.chat_input[0].disabled
    assert not at.info
    say(at, "My answer.")
    assert len(fake_llm.calls) == 1
    assert not at.warning
    assert len(at.session_state.request_times) == RATE_LIMIT_PER_SESSION
    assert at.chat_input[0].disabled
    assert at.info[0].proto.title == "Session limit reached"


def test_new_session_does_not_unlock_a_used_up_session(fake_llm, clock):
    """New session clears the chat but keeps the count, so the input stays locked."""
    at = start([clock.now - 120.0 * (i + 1) for i in range(RATE_LIMIT_PER_SESSION)])
    new_session = next(button for button in at.button if button.label == "New session")
    new_session.click().run(timeout=30)
    assert at.chat_input[0].disabled
    assert len(at.info) == 1


@pytest.mark.parametrize("text", ["   ", ATTACKS[0]], ids=["blank", "injection"])
def test_message_blocked_by_the_guard_is_not_counted(fake_llm, clock, text):
    """A message the guard blocks shows the guard's reason, not the limit, and is not counted."""
    at = start([clock.now] * RATE_LIMIT_PER_MINUTE)
    say(at, text)
    assert fake_llm.calls == []
    assert len(at.warning) == 1
    assert "wait" not in at.warning[0].value
    # The injected message is kept for copying like any block; a blank one has nothing to keep.
    assert [code.value for code in at.code] == ([] if text.isspace() else [text])

    at = start()
    say(at, text)
    assert at.session_state.request_times == []


def test_failed_request_is_counted(monkeypatch, fake_llm, clock):
    """A request that fails after it went out still counts: it may have spent tokens."""

    def failing_stream(*args, **kwargs):
        """Fail on the first piece, the way `llm.stream` does when the service is down."""
        raise llm.LLMError("The AI service is having problems. Please try again.")
        yield  # Makes this a generator, so the error comes on next(), after the request.

    monkeypatch.setattr(llm, "stream", failing_stream)
    at = start()
    say(at, "My answer.")
    assert len(at.error) == 1
    assert at.session_state.request_times == [clock.now]


def test_failure_before_sending_is_not_counted(monkeypatch, fake_llm, clock):
    """A key that fails when `llm.stream` is called sends nothing, so it uses no turn."""

    def bad_key_stream(*args, **kwargs):
        """Fail at call time, the way `llm.stream` does for a key it cannot use."""
        raise llm.LLMAuthError("The OpenRouter API key was rejected.")

    monkeypatch.setattr(llm, "stream", bad_key_stream)
    at = start()
    say(at, "My answer.")
    assert len(at.error) == 1
    assert at.session_state.request_times == []


def test_new_session_keeps_the_count(fake_llm, clock):
    """The New session button clears the chat but not the request count, so it skips no limit."""
    at = start([clock.now] * (RATE_LIMIT_PER_MINUTE - 1))
    say(at, "Last one this minute.")
    new_session = next(button for button in at.button if button.label == "New session")
    new_session.click().run(timeout=30)
    assert at.session_state.history == []
    assert len(at.session_state.request_times) == RATE_LIMIT_PER_MINUTE
    say(at, "One too many.")
    assert len(fake_llm.calls) == 1
    assert "Please wait 60 seconds" in at.warning[0].value
