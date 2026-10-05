from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import guard, llm, rate_limit
from src.config import (
    API_KEY_NAME,
    MAX_INPUT_CHARS,
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_PER_SESSION,
)
from src.prompts import EXAMPLE_PROMPTS, INTERVIEW_TYPES
from tests.injection_samples import ATTACKS

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


def raising_stream(error: llm.LLMError, pieces_before: list[str] | None):
    """Return an llm.stream stand-in that raises `error` over RAW_BODY at the given point."""

    def at_call(*args, **kwargs):
        """Fail when called, as llm.stream does for a bad model or key (nothing is sent)."""
        raise error from RuntimeError(RAW_BODY)

    def stream(*args, **kwargs):
        """Yield the given pieces, then fail with the raw SDK error chained, as llm.py does."""
        yield from pieces_before
        raise error from RuntimeError(RAW_BODY)

    return at_call if pieces_before is None else stream


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
    # The interrupted request went out, so the copy box must not claim it was not sent.
    assert at.caption[0].value == NO_ANSWER
    assert at.code[0].value == "My answer."

    assert styles == {
        "guard": ("warning", "✋", "Message not sent"),
        "rate_limit": ("warning", "⏳", "Message limit reached"),
        "llm": ("error", "⚠️", "AI service problem"),
        "interrupted": ("error", "⏹️", "Answer interrupted"),
    }
    assert len(set(styles.values())) == len(styles)


@pytest.mark.parametrize(
    "pieces_before", [None, [], ["Half of ", "a reply"]], ids=["at-call", "first", "mid"]
)
@pytest.mark.parametrize("error", LLM_ERRORS, ids=lambda exc: type(exc).__name__)
def test_each_llm_error_shows_only_its_own_text(monkeypatch, fake_llm, error, pieces_before):
    """Every LLMError shows as an error with its fixed text, never the key or the raw body."""
    monkeypatch.setattr(llm, "stream", raising_stream(error, pieces_before))
    at = start()
    say(at, "My answer.")
    # Failing at the call sends nothing, so it uses no turn; failing later may have spent tokens.
    assert len(at.session_state.request_times) == (0 if pieces_before is None else 1)
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


@pytest.mark.parametrize(
    "text",
    ["a" * (MAX_INPUT_CHARS + 1), ATTACKS[0]],
    ids=["too-long", "injection"],
)
def test_message_the_guard_blocks_is_kept_in_a_copy_box(fake_llm, text):
    """A blocked message is shown back in a copy box, so a long answer need not be retyped."""
    at = start()
    say(at, text)
    assert fake_llm.calls == []
    assert notice_style(at) == ("warning", "✋", "Message not sent")
    assert at.caption[0].value == NOT_SENT
    assert [code.value for code in at.code] == [text]


def test_any_guard_block_keeps_the_message(monkeypatch, fake_llm):
    """Every GuardError keeps the message, not only the length and injection checks."""

    def blocking_validate(message, max_chars=guard.MAX_INPUT_CHARS):
        """Block every message the way a future guard check would."""
        raise guard.GuardError("That message was blocked.")

    # app.py re-imports validate_input from the module on every run, so this patch reaches it.
    monkeypatch.setattr(guard, "validate_input", blocking_validate)
    at = start()
    say(at, "Line one.\nLine two.")
    assert notice_style(at) == ("warning", "✋", "Message not sent")
    assert [code.value for code in at.code] == ["Line one.\nLine two."]


def test_copy_box_holds_the_cleaned_message(fake_llm):
    """The copy box shows the message as the guard saw it: trimmed, control characters gone."""
    at = start()
    say(at, "  " + "a" * MAX_INPUT_CHARS + chr(0x07) + "b \r\n")
    assert [code.value for code in at.code] == ["a" * MAX_INPUT_CHARS + "b"]


@pytest.mark.parametrize(
    "text", ["   ", "\n\t ", chr(0x200B) + " " + chr(0x3164)], ids=["spaces", "breaks", "fillers"]
)
def test_blank_message_gets_no_copy_box(fake_llm, text):
    """A blank message is refused with no copy box: there is nothing to keep."""
    at = start()
    say(at, text)
    assert notice_style(at) == ("warning", "✋", "Message not sent")
    assert not at.code
    # The only caption left is the empty chat's starter line, not a copy-box one.
    assert NOT_SENT not in [caption.value for caption in at.caption]


def test_notice_of_an_unknown_kind_is_shown_as_an_error_not_a_crash(fake_llm):
    """A notice saved with an old or unknown kind is shown once as an error, without crashing."""
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = FAKE_KEY
    at.session_state["notice"] = {"kind": "warning", "text": "Saved by an older version."}
    at.run(timeout=30)
    assert not at.exception
    assert notice_style(at) == ("error", "⚠️", "AI service problem")
    assert at.error[0].value == "Saved by an older version."
    assert at.session_state.notice is None


def last_in_main(at: AppTest):
    """Return the last node of the main area: the one just above the chat input."""
    # The chat input lives in the bottom area, not in main, so this is whatever comes last
    # after the chat history.
    return list(at.main.children.values())[-1]


def loose_in_main(at: AppTest) -> list[str]:
    """Return notice and starter elements drawn straight into the main area, outside a block."""
    # Every element type the notice/starter block draws. Add a type here when the block gains
    # one (e.g. st.markdown, st.success, st.link_button), or a loose one would slip through.
    loose = {"Warning", "Error", "Info", "Caption", "Code", "Button"}
    return [
        type(node).__name__ for node in at.main.children.values() if type(node).__name__ in loose
    ]


def kinds(node) -> list[str]:
    """Return the element types directly inside a block, in order."""
    return [type(child).__name__ for child in node.children.values()]


# The empty chat's caption and one button per example. Counted from the data so this layout
# check does not repeat the number; each mode's count is pinned in test_prompts.py.
STARTERS = ["Caption", *["Button"] * len(EXAMPLE_PROMPTS[INTERVIEW_TYPES[0]])]


def assert_one_block_below_the_chat(at: AppTest, expected: list[str]) -> None:
    """Check that the area above the chat input is exactly one block holding `expected`."""
    assert loose_in_main(at) == []
    assert kinds(last_in_main(at)) == expected


def test_notice_and_starters_share_one_container(fake_llm):
    """A notice, its copy box and the starters are one block, so a send run replaces all of it."""
    at = start()
    say(at, "a" * (MAX_INPUT_CHARS + 1))
    assert_one_block_below_the_chat(at, ["Warning", "Caption", "Code", *STARTERS])


def test_notice_after_earlier_turns_is_one_container(monkeypatch, fake_llm):
    """A failed later turn's error and copy box form one block below the chat."""
    at = start()
    say(at, "First answer.")
    monkeypatch.setattr(llm, "stream", raising_stream(LLM_ERRORS[0], []))
    say(at, "Second answer.")
    assert_one_block_below_the_chat(at, ["Error", "Caption", "Code"])


def test_session_cap_info_shares_the_starters_container(fake_llm):
    """The cap notice sits in the same block as the (locked) starters."""
    at = start([0.0] * RATE_LIMIT_PER_SESSION)
    assert_one_block_below_the_chat(at, ["Info", *STARTERS])


def test_empty_chat_starters_are_one_container(fake_llm):
    """With nothing else to show, the starters alone still form one block."""
    at = start()
    assert_one_block_below_the_chat(at, STARTERS)


def test_nothing_is_drawn_below_a_finished_turn(fake_llm):
    """After a reply with no notice, no empty container is left between the chat and the input."""
    at = start()
    say(at, "My answer.")
    assert loose_in_main(at) == []
    assert getattr(last_in_main(at), "type", None) == "chat_message"
