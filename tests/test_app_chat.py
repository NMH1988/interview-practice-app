from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src import llm, prompts
from src.config import API_KEY_NAME, DEFAULT_ROLE, DEFAULT_SENIORITY
from src.prompts import INTERVIEW_TYPES, STRATEGIES, build_user_prompt

APP = Path(__file__).resolve().parent.parent / "app.py"

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


def shown(at: AppTest) -> list[tuple[str, str]]:
    """Return the chat as (author, text) pairs, in the order they are shown."""
    return [(msg.name, msg.markdown[0].value) for msg in at.chat_message]


def test_two_turns_are_kept_in_order(fake_llm):
    """After two turns the chat shows both questions and both replies, in order."""
    at = start()
    fake_llm.reply = "First reply."
    say(at, "First answer.")
    fake_llm.reply = "Second reply."
    say(at, "Second answer.")
    assert shown(at) == [
        ("user", "First answer."),
        ("assistant", "First reply."),
        ("user", "Second answer."),
        ("assistant", "Second reply."),
    ]
    assert [turn["content"] for turn in at.session_state.history] == [
        "First answer.",
        "First reply.",
        "Second answer.",
        "Second reply.",
    ]
    assert len(fake_llm.calls) == 2


def test_llm_gets_chosen_strategy_and_delimited_message(fake_llm):
    """The LLM gets the chosen strategy's system prompt and the tagged, built user message."""
    at = start()
    at.sidebar.selectbox(key="strategy").set_value("persona")
    at.sidebar.selectbox(key="interview_type").set_value("Technical")
    at.sidebar.text_input(key="role").set_value("Data Analyst")
    at.sidebar.selectbox(key="seniority").set_value("Senior")
    at.run(timeout=30)
    say(at, "I would use a window function.")
    assert fake_llm.calls[0]["messages"] == [
        {"role": "system", "content": STRATEGIES["persona"]("Data Analyst", "Technical")},
        {
            "role": "user",
            "content": build_user_prompt(
                "Data Analyst", "Technical", "Senior", "I would use a window function."
            ),
        },
    ]


def test_second_turn_resends_earlier_turns_inside_tags(fake_llm):
    """The second call sends the earlier turns too, with the old user message still tagged."""
    at = start()
    say(at, "First answer.")
    say(at, "Second answer.")
    system, *turns = fake_llm.calls[1]["messages"]
    assert system["role"] == "system"

    def tagged(text):
        """Build the user prompt the default settings give for `text`."""
        return build_user_prompt(DEFAULT_ROLE, INTERVIEW_TYPES[0], DEFAULT_SENIORITY, text)

    assert turns == [
        {"role": "user", "content": tagged("First answer.")},
        {"role": "assistant", "content": fake_llm.reply},
        {"role": "user", "content": tagged("Second answer.")},
    ]


def test_role_is_cleaned_before_it_reaches_the_prompts(fake_llm):
    """Extra spaces and line breaks in the role are folded before both prompts are built."""
    at = start()
    at.sidebar.text_input(key="role").set_value("  Data \n  Analyst ").run(timeout=30)
    say(at, "Hello.")
    system, user = fake_llm.calls[0]["messages"]
    assert system["content"] == STRATEGIES[next(iter(STRATEGIES))](
        "Data Analyst", INTERVIEW_TYPES[0]
    )
    assert user["content"].startswith("Role: Data Analyst\n")


def test_chat_shows_raw_text_not_the_built_prompt(fake_llm):
    """The chat shows what the user typed, not the escaped, tagged prompt sent to the LLM."""
    at = start()
    say(at, "if a < b && c:")
    assert shown(at)[0] == ("user", "if a < b && c:")
    assert "&lt;" in fake_llm.calls[0]["messages"][-1]["content"]


def test_chat_input_is_unlocked_after_the_reply(fake_llm):
    """The input is locked only while a reply is pending, so the next message can be sent."""
    at = start()
    say(at, "First answer.")
    assert not at.chat_input[0].disabled
    assert at.session_state.pending is None


@pytest.mark.parametrize(
    ("module", "name"),
    [(llm, "complete"), (prompts, "build_user_prompt")],
    ids=["crash-in-llm-call", "crash-building-prompt"],
)
def test_chat_input_is_locked_during_the_turn_and_freed_after_a_crash(
    monkeypatch, fake_llm, module, name
):
    """The input is drawn locked before the turn runs; a crash in it does not leave it locked."""

    def crash(*args, **kwargs):
        """Fail with an error the app does not handle, so the run ends right here."""
        raise RuntimeError("unexpected bug")

    # app.py looks these up again on every run, so the patch reaches it.
    monkeypatch.setattr(module, name, crash)
    at = start()
    at.chat_input[0].set_value("First answer.").run(timeout=30)
    # AppTest keeps what was drawn before the crash: the input was already locked.
    assert at.exception
    assert at.chat_input[0].disabled
    assert at.session_state.pending is None
    at.run(timeout=30)
    assert not at.exception
    assert not at.chat_input[0].disabled


def test_settings_changed_mid_session_keep_history(fake_llm):
    """A strategy and role changed after a turn keep the history; the next call uses them."""
    at = start()
    say(at, "First answer.")
    at.sidebar.selectbox(key="strategy").set_value("persona")
    at.sidebar.text_input(key="role").set_value("Data Analyst")
    at.run(timeout=30)
    assert len(at.chat_message) == 2
    say(at, "Second answer.")
    system, first_user, first_reply, second_user = fake_llm.calls[1]["messages"]
    assert system["content"] == STRATEGIES["persona"]("Data Analyst", INTERVIEW_TYPES[0])
    assert first_user["content"] == build_user_prompt(
        DEFAULT_ROLE, INTERVIEW_TYPES[0], DEFAULT_SENIORITY, "First answer."
    )
    assert first_reply == {"role": "assistant", "content": fake_llm.reply}
    assert second_user["content"].startswith("Role: Data Analyst\n")


def test_failed_turn_stays_out_of_history(monkeypatch, fake_llm):
    """An LLM error shows its message and the unsent text, and the turn is not kept or resent."""
    at = start()
    say(at, "First answer.")

    def failing_complete(*args, **kwargs):
        """Fail the way llm.complete does when OpenRouter keeps answering 5xx."""
        raise llm.LLMServerError("The AI service is having problems right now.")

    monkeypatch.setattr(llm, "complete", failing_complete)
    say(at, "Answer that fails.")
    assert len(at.error) == 1
    assert "having problems" in at.error[0].value
    assert at.code[0].value == "Answer that fails."
    assert shown(at) == [("user", "First answer."), ("assistant", fake_llm.reply)]
    assert not at.chat_input[0].disabled

    monkeypatch.setattr(llm, "complete", fake_llm)
    say(at, "Third answer.")
    sent = [msg["content"] for msg in fake_llm.calls[-1]["messages"]]
    assert not any("Answer that fails." in text for text in sent)
    assert not at.error


def test_new_session_clears_history(fake_llm):
    """The "New session" button empties the chat, and the next call has no earlier turns."""
    at = start()
    say(at, "First answer.")
    assert len(at.chat_message) == 2
    button = at.sidebar.button[0]
    assert button.label == "New session"
    button.click().run(timeout=30)
    assert not at.exception
    assert len(at.chat_message) == 0
    assert at.session_state.history == []
    say(at, "Fresh start.")
    assert [msg["role"] for msg in fake_llm.calls[-1]["messages"]] == ["system", "user"]
