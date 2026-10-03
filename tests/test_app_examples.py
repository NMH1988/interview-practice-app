from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import guard, rate_limit
from src.config import API_KEY_NAME, DEFAULT_ROLE, DEFAULT_SENIORITY, RATE_LIMIT_PER_MINUTE
from src.prompts import EXAMPLE_CAPTIONS, EXAMPLE_PROMPTS, INTERVIEW_TYPES, build_user_prompt

APP = Path(__file__).resolve().parent.parent / "app.py"
STARTER_CAPTION = EXAMPLE_CAPTIONS[INTERVIEW_TYPES[0]]
# Every (interview type, starter index) pair, so each starter is clicked once.
EVERY_STARTER = [
    (interview_type, index)
    for interview_type, examples in EXAMPLE_PROMPTS.items()
    for index in range(len(examples))
]

pytestmark = pytest.mark.usefixtures("no_env_key")


def start(request_times: list[float] | None = None) -> AppTest:
    """Run the app once with a fake key, optionally with earlier requests already counted."""
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = "sk-test-not-a-real-key"
    if request_times is not None:
        at.session_state["request_times"] = request_times
    at.run(timeout=30)
    assert not at.exception
    return at


def examples(at: AppTest) -> list:
    """Return the example-prompt buttons in the main area, in the order they are shown."""
    return [button for button in at.main.button if (button.key or "").startswith("example_")]


def test_empty_chat_offers_the_modes_examples(fake_llm):
    """With no history, the chosen mode's example prompts are shown as buttons, by label."""
    at = start()
    assert STARTER_CAPTION in [caption.value for caption in at.caption]
    labels = [example.label for example in EXAMPLE_PROMPTS[INTERVIEW_TYPES[0]]]
    assert [button.label for button in examples(at)] == labels
    assert not any(button.disabled for button in examples(at))
    assert fake_llm.calls == []


@pytest.mark.parametrize(("interview_type", "index"), EVERY_STARTER)
def test_clicking_an_example_sends_its_text_like_a_typed_message(fake_llm, interview_type, index):
    """A clicked starter sends its full text (not its label) through the built user prompt."""
    at = start()
    at.sidebar.selectbox(key="interview_type").set_value(interview_type).run(timeout=30)
    example = EXAMPLE_PROMPTS[interview_type][index]
    examples(at)[index].click().run(timeout=30)
    assert not at.exception
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["messages"][-1] == {
        "role": "user",
        "content": build_user_prompt(DEFAULT_ROLE, interview_type, DEFAULT_SENIORITY, example.text),
    }
    assert [msg.markdown[0].value for msg in at.chat_message] == [example.text, fake_llm.reply]
    # The chat is no longer empty, so the starters and their caption are gone.
    assert examples(at) == []
    assert EXAMPLE_CAPTIONS[interview_type] not in [caption.value for caption in at.caption]


@pytest.mark.parametrize("interview_type", INTERVIEW_TYPES)
def test_examples_follow_the_interview_type(fake_llm, interview_type):
    """Choosing another interview type shows that type's caption and example prompts."""
    at = start()
    at.sidebar.selectbox(key="interview_type").set_value(interview_type).run(timeout=30)
    assert not at.exception
    labels = [example.label for example in EXAMPLE_PROMPTS[interview_type]]
    assert [button.label for button in examples(at)] == labels
    assert EXAMPLE_CAPTIONS[interview_type] in [caption.value for caption in at.caption]


def test_examples_are_hidden_once_the_chat_has_a_turn(fake_llm):
    """A typed message fills the chat, so the starters are no longer offered."""
    at = start()
    at.chat_input[0].set_value("My answer.").run(timeout=30)
    assert len(at.chat_message) == 2
    assert examples(at) == []


def test_examples_are_locked_while_the_role_is_blank(fake_llm):
    """Like the chat input, the examples cannot be used while the role is missing."""
    at = start()
    at.sidebar.text_input(key="role").set_value("   ").run(timeout=30)
    assert at.chat_input[0].disabled
    assert examples(at) and all(button.disabled for button in examples(at))


def test_clicked_example_still_meets_the_rate_limit(monkeypatch, fake_llm):
    """An example takes no shortcut: over the limit it is refused and kept, like a typed one."""
    monkeypatch.setattr(rate_limit, "clock", lambda: 1000.0)
    at = start([1000.0] * RATE_LIMIT_PER_MINUTE)
    example = EXAMPLE_PROMPTS[INTERVIEW_TYPES[0]][0].text
    examples(at)[0].click().run(timeout=30)
    assert not at.exception
    assert fake_llm.calls == []
    assert at.main.warning[0].proto.title == "Message limit reached"
    assert [code.value for code in at.code] == [example]


def test_clicked_example_still_meets_the_guard(monkeypatch, fake_llm):
    """An example takes no shortcut past the guard: a block refuses it like a typed message."""

    def blocking_validate(message):
        """Block every message the way any guard check would."""
        raise guard.GuardError("That message was blocked.")

    # app.py re-imports validate_input from the module on every run, so this patch reaches it.
    monkeypatch.setattr(guard, "validate_input", blocking_validate)
    at = start()
    examples(at)[0].click().run(timeout=30)
    assert not at.exception
    assert fake_llm.calls == []
    assert at.main.warning[0].proto.title == "Message not sent"
    assert [code.value for code in at.code] == [EXAMPLE_PROMPTS[INTERVIEW_TYPES[0]][0].text]
