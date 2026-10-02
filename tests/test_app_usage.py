from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import llm
from src.config import API_KEY_NAME

APP = Path(__file__).resolve().parent.parent / "app.py"
CUT_OFF = "The answer was cut off because it reached the token limit."

pytestmark = pytest.mark.usefixtures("no_env_key")


def start() -> AppTest:
    """Run the app once with a fake key and return it, ready for interaction."""
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = "sk-test-not-a-real-key"
    at.run(timeout=30)
    assert not at.exception
    return at


def say(at: AppTest, text: str) -> None:
    """Submit `text` in the chat input and wait for the run (and its rerun) to finish."""
    at.chat_input[0].set_value(text).run(timeout=30)
    assert not at.exception


def usage_shown(message) -> str | None:
    """Return the text in a chat message's "Token usage" expander, or None if it has none."""
    expanders = [e for e in message.expander if e.label == "Token usage"]
    if not expanders:
        return None
    (expander,) = expanders
    return expander.caption[0].value


def test_reply_shows_its_token_usage(fake_llm):
    """After a reply, the expander under it shows the prompt, completion and total counts."""
    at = start()
    say(at, "I led the migration.")
    user, assistant = at.chat_message
    assert usage_shown(assistant) == (
        "Prompt 1,234 · Completion 567 (reasoning 320) · Total 1,801 tokens"
    )
    assert usage_shown(user) is None


def test_each_reply_keeps_its_own_usage(fake_llm):
    """An earlier reply still shows its own counts after a later reply comes in."""
    at = start()
    fake_llm.usage = llm.Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
    say(at, "First answer.")
    fake_llm.usage = llm.Usage(prompt_tokens=40, completion_tokens=8, total_tokens=48)
    say(at, "Second answer.")
    shown = [usage_shown(message) for message in at.chat_message]
    assert shown == [
        None,
        "Prompt 10 · Completion 5 · Total 15 tokens",
        None,
        "Prompt 40 · Completion 8 · Total 48 tokens",
    ]


def test_reply_without_usage_shows_no_expander(fake_llm):
    """A reply whose usage was not reported is kept and shown, just without the expander."""
    fake_llm.usage = None
    at = start()
    say(at, "I led the migration.")
    assert at.chat_message[-1].markdown[0].value == fake_llm.reply
    assert usage_shown(at.chat_message[-1]) is None
    assert at.session_state.history[-1]["usage"] is None


def test_reply_cut_off_by_the_token_limit_is_kept_with_a_warning(fake_llm):
    """A reply stopped by max_tokens is saved as it is, with a warning under it that stays."""
    fake_llm.finish_reason = "length"
    at = start()
    say(at, "Tell me everything.")
    assert at.session_state.history[-1]["content"] == fake_llm.reply
    assert at.session_state.history[-1]["cut_off"] is True
    assert [w.value for w in at.chat_message[-1].warning] == [CUT_OFF]
    # Still there on the next turn, under the reply it belongs to and nowhere else.
    fake_llm.finish_reason = "stop"
    say(at, "Shorter, please.")
    assert [[w.value for w in m.warning] for m in at.chat_message] == [[], [CUT_OFF], [], []]


def test_complete_reply_has_no_cut_off_warning(fake_llm):
    """A reply that ended normally gets no warning."""
    at = start()
    say(at, "I led the migration.")
    assert at.session_state.history[-1]["cut_off"] is False
    assert not at.chat_message[-1].warning


def test_usage_is_not_sent_back_to_the_llm(fake_llm):
    """The next request carries only role and text for the earlier reply, not its usage."""
    fake_llm.finish_reason = "length"
    at = start()
    say(at, "First answer.")
    say(at, "Second answer.")
    earlier_reply = fake_llm.calls[1]["messages"][2]
    assert earlier_reply == {"role": "assistant", "content": fake_llm.reply}


def test_reply_with_no_text_before_the_token_limit_says_so(fake_llm):
    """When the model spends the whole limit thinking, the error says so and nothing is saved."""
    fake_llm.reply = ""
    fake_llm.finish_reason = "length"
    at = start()
    say(at, "Tell me everything.")
    assert [e.value for e in at.error] == [
        "The model used up its token limit before writing an answer. Please try again."
    ]
    assert at.session_state.history == []
