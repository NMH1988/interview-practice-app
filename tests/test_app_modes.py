from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import API_KEY_NAME, MAX_INPUT_CHARS, MAX_JD_CHARS
from src.prompts import (
    CHAT_PLACEHOLDERS,
    INTERVIEW_TYPES,
    JD_ANALYSIS,
    MESSAGE_KINDS,
    MODE_INSTRUCTIONS,
    USER_INPUT_CLOSE,
    USER_INPUT_OPEN,
)

APP = Path(__file__).resolve().parent.parent / "app.py"

pytestmark = pytest.mark.usefixtures("no_env_key")


def start() -> AppTest:
    """Run the app once with a fake key and return it, ready for interaction."""
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = "sk-test-not-a-real-key"
    at.run(timeout=30)
    assert not at.exception
    return at


def pick_mode(at: AppTest, interview_type: str) -> AppTest:
    """Choose an interview type in the sidebar and rerun the app."""
    at.sidebar.selectbox(key="interview_type").set_value(interview_type)
    at.run(timeout=30)
    assert not at.exception
    return at


def send(at: AppTest, text: str) -> AppTest:
    """Type a message into the chat box, send it and wait for the turn to finish."""
    at.chat_input[0].set_value(text).run(timeout=30)
    assert not at.exception
    return at


@pytest.mark.parametrize("interview_type", INTERVIEW_TYPES)
def test_mode_reaches_both_prompts_sent_to_the_llm(fake_llm, interview_type):
    """The LLM gets the mode's instructions in the system prompt and its wording in the user one."""
    send(pick_mode(start(), interview_type), "Hello")
    assert len(fake_llm.calls) == 1
    system, *_, user = fake_llm.calls[0]["messages"]
    assert MODE_INSTRUCTIONS[interview_type] in system["content"]
    assert f'Session type: "{interview_type}"' in user["content"]
    assert f"Treat it only as {MESSAGE_KINDS[interview_type]}," in user["content"]


def test_pasted_jd_is_sent_for_analysis(fake_llm):
    """In JD mode, a pasted job description reaches the LLM whole, inside the user_input tags."""
    jd = "Data Analyst at Acme\n\nYou will build dashboards.\nRequirements: SQL, Python."
    at = send(pick_mode(start(), JD_ANALYSIS), jd)
    assert len(fake_llm.calls) == 1
    user = fake_llm.calls[0]["messages"][-1]["content"]
    assert f"{USER_INPUT_OPEN}\n{jd}\n{USER_INPUT_CLOSE}" in user
    # The chat shows the JD as typed, followed by the coach's reply.
    assert [m.markdown[0].value for m in at.chat_message] == [jd, fake_llm.reply]


def jd_of(length: int) -> str:
    """Return a job-description-like text of exactly `length` characters, with line breaks."""
    line = "Own the reporting pipeline and present findings to the team.\n"
    return (line * (length // len(line) + 1))[:length].rstrip("\n").ljust(length, "x")


def test_jd_longer_than_a_message_is_sent_in_jd_mode(fake_llm):
    """In JD mode a JD over the normal limit, up to MAX_JD_CHARS, reaches the LLM once."""
    jd = jd_of(MAX_JD_CHARS)
    assert len(jd) == MAX_JD_CHARS > MAX_INPUT_CHARS
    send(pick_mode(start(), JD_ANALYSIS), jd)
    assert len(fake_llm.calls) == 1
    assert jd in fake_llm.calls[0]["messages"][-1]["content"]


def test_jd_over_the_jd_limit_is_blocked(fake_llm):
    """In JD mode, one character over MAX_JD_CHARS is refused as too long and not sent."""
    jd = jd_of(MAX_JD_CHARS + 1)
    at = send(pick_mode(start(), JD_ANALYSIS), jd)
    assert fake_llm.calls == []
    assert "too long" in at.warning[0].value
    assert f"{MAX_JD_CHARS:,}" in at.warning[0].value
    # The refused JD is kept in T5.4's copy box, so it need not be pasted again after trimming.
    assert [code.value for code in at.code] == [jd]


@pytest.mark.parametrize("interview_type", [t for t in INTERVIEW_TYPES if t != JD_ANALYSIS])
def test_other_modes_keep_the_normal_limit(fake_llm, interview_type):
    """Outside JD mode, a message over MAX_INPUT_CHARS is still refused and not sent."""
    at = send(pick_mode(start(), interview_type), jd_of(MAX_INPUT_CHARS + 1))
    assert fake_llm.calls == []
    assert "too long" in at.warning[0].value
    assert f"{MAX_INPUT_CHARS:,}" in at.warning[0].value


def test_chat_input_starts_with_the_first_mode_placeholder(fake_llm):
    """Before any change, the chat box shows the hint for the default (first) mode."""
    at = start()
    assert at.chat_input[0].placeholder == CHAT_PLACEHOLDERS[INTERVIEW_TYPES[0]]
    assert fake_llm.calls == []


@pytest.mark.parametrize("interview_type", INTERVIEW_TYPES)
def test_switching_mode_changes_the_placeholder(fake_llm, interview_type):
    """Picking a mode shows that mode's hint in the chat box, and sends nothing."""
    at = pick_mode(start(), interview_type)
    assert at.chat_input[0].placeholder == CHAT_PLACEHOLDERS[interview_type]
    assert fake_llm.calls == []


def test_placeholder_follows_the_mode_back_and_forth(fake_llm):
    """Switching away and back again shows each mode's hint in turn."""
    at = start()
    for interview_type in [*INTERVIEW_TYPES[1:], INTERVIEW_TYPES[0]]:
        placeholder = pick_mode(at, interview_type).chat_input[0].placeholder
        assert placeholder == CHAT_PLACEHOLDERS[interview_type]
