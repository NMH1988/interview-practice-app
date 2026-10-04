import logging
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import (
    ALLOWED_MODELS,
    API_KEY_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    DEFAULT_ROLE,
    DEFAULT_SENIORITY,
    HIGH_EFFORT_MAX_TOKENS,
    MAX_ROLE_CHARS,
    MAX_TOKENS_BY_EFFORT,
    MAX_TOKENS_CAP,
    REASONING_EFFORTS,
)
from src.guard import INJECTION_REFUSAL
from src.prompts import INTERVIEW_TYPES, SENIORITY_LEVELS
from tests.injection_samples import ROLE_ATTACK

APP = Path(__file__).resolve().parent.parent / "app.py"

pytestmark = pytest.mark.usefixtures("no_env_key")


def start() -> AppTest:
    """Run the app once with a fake key and return it, ready for interaction."""
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = "sk-test-not-a-real-key"
    at.run(timeout=30)
    assert not at.exception
    return at


def test_model_select_lists_exactly_the_allowed_models(fake_llm):
    """The sidebar model select offers exactly the allowed models, gpt-5-mini by default."""
    select = start().sidebar.selectbox(key="model")
    assert select.label == "Model"
    assert select.options == list(ALLOWED_MODELS)
    assert select.value == DEFAULT_MODEL == "openai/gpt-5-mini"
    assert fake_llm.calls == []


def test_reasoning_effort_select_levels_and_default(fake_llm):
    """The sidebar reasoning effort select offers the four levels and starts at medium."""
    select = start().sidebar.selectbox(key="reasoning_effort")
    assert select.label == "Reasoning effort"
    assert select.options == ["Minimal", "Low", "Medium", "High"]
    assert select.value == DEFAULT_REASONING_EFFORT == "medium"
    assert select.help and "token limit" in select.help
    assert fake_llm.calls == []


def test_reasoning_effort_help_names_both_token_limits(fake_llm):
    """The help says thinking uses the token limit and that High gets the larger one (T2.5)."""
    help_text = start().sidebar.selectbox(key="reasoning_effort").help
    assert "thinking counts against the token limit" in help_text
    assert f"High gets a larger limit ({HIGH_EFFORT_MAX_TOKENS:,} tokens" in help_text
    assert f"instead of {DEFAULT_MAX_TOKENS:,})" in help_text


def test_no_temperature_slider_remains(fake_llm):
    """The temperature slider is gone: the allowed gpt-5 models ignore temperature (T2.4)."""
    at = start()
    assert len(at.sidebar.slider) == 0
    assert "temperature" not in at.session_state


def test_default_model_and_reasoning_effort_reach_the_llm(fake_llm):
    """Without changes, the LLM call gets the default model and reasoning effort."""
    at = start()
    at.chat_input[0].set_value("Tell me about yourself.").run(timeout=30)
    assert not at.exception
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["model"] == DEFAULT_MODEL
    assert fake_llm.calls[0]["reasoning_effort"] == DEFAULT_REASONING_EFFORT


@pytest.mark.parametrize("effort", REASONING_EFFORTS)
def test_changed_model_and_reasoning_effort_reach_the_llm(fake_llm, effort):
    """A model and reasoning effort picked in the sidebar are the ones sent to the LLM."""
    at = start()
    at.sidebar.selectbox(key="model").set_value("openai/gpt-5-nano")
    at.sidebar.selectbox(key="reasoning_effort").set_value(effort)
    at.run(timeout=30)
    at.chat_input[0].set_value("Tell me about yourself.").run(timeout=30)
    assert not at.exception
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["model"] == "openai/gpt-5-nano"
    assert fake_llm.calls[0]["reasoning_effort"] == effort


@pytest.mark.parametrize("effort", REASONING_EFFORTS)
def test_token_budget_follows_the_reasoning_effort(fake_llm, effort):
    """The LLM call gets the chosen effort's own token budget, never more than the cap (T2.5)."""
    at = start()
    at.sidebar.selectbox(key="reasoning_effort").set_value(effort)
    at.run(timeout=30)
    at.chat_input[0].set_value("Tell me about yourself.").run(timeout=30)
    assert not at.exception
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["max_tokens"] == MAX_TOKENS_BY_EFFORT[effort]
    assert fake_llm.calls[0]["max_tokens"] <= MAX_TOKENS_CAP


def test_interview_type_role_and_seniority_inputs(fake_llm):
    """The sidebar shows interview type, a length-limited role and seniority with their defaults."""
    sidebar = start().sidebar
    interview_type = sidebar.selectbox(key="interview_type")
    assert interview_type.label == "Interview type"
    assert interview_type.options == list(INTERVIEW_TYPES)
    assert interview_type.value == INTERVIEW_TYPES[0]
    role = sidebar.text_input(key="role")
    assert role.label == "Role"
    assert role.value == DEFAULT_ROLE
    assert role.max_chars == MAX_ROLE_CHARS
    seniority = sidebar.selectbox(key="seniority")
    assert seniority.label == "Seniority"
    assert seniority.options == list(SENIORITY_LEVELS)
    assert seniority.value == DEFAULT_SENIORITY
    assert fake_llm.calls == []


def test_blank_role_warns_in_sidebar_and_locks_chat_input(fake_llm):
    """A blank role shows a sidebar warning at once and locks the chat input until it is fixed."""
    at = start()
    assert not at.chat_input[0].disabled
    at.sidebar.text_input(key="role").set_value("   ").run(timeout=30)
    assert not at.exception
    assert len(at.sidebar.warning) == 1
    assert "enter the role" in at.sidebar.warning[0].value
    # AppTest refuses to type into a disabled widget, just as a browser would.
    assert at.chat_input[0].disabled
    at.sidebar.text_input(key="role").set_value("Data Analyst").run(timeout=30)
    assert not at.sidebar.warning
    assert not at.chat_input[0].disabled
    assert fake_llm.calls == []


def test_role_with_an_injection_is_refused_and_locks_chat_input(fake_llm, caplog):
    """A role holding an instruction shows the neutral refusal in the sidebar and locks the chat."""
    at = start()
    with caplog.at_level(logging.DEBUG, logger="src.guard"):
        at.sidebar.text_input(key="role").set_value(ROLE_ATTACK).run(timeout=30)
        at.run(timeout=30)
    # The sidebar check runs on every rerun, so it does not log; nothing was sent.
    # Only the guard's records: Streamlit may log its own warnings while AppTest runs.
    assert [r for r in caplog.records if r.name == "src.guard"] == []
    assert not at.exception
    assert at.session_state.role == ROLE_ATTACK
    assert len(at.sidebar.warning) == 1
    assert at.sidebar.warning[0].value == INJECTION_REFUSAL
    assert at.chat_input[0].disabled
    assert fake_llm.calls == []


def test_message_queued_before_an_injected_role_is_refused_and_logged_once(fake_llm, caplog):
    """A message already pending when the role turns into an attack is blocked on send, once."""
    at = start()
    # The message is queued and the role changed in the same run, as when the user sends a
    # message and edits the role before that run reaches the send step.
    at.session_state["pending"] = "Tell me about yourself."
    at.sidebar.text_input(key="role").set_value(ROLE_ATTACK)
    with caplog.at_level(logging.DEBUG, logger="src.guard"):
        at.run(timeout=30)
    assert not at.exception
    assert fake_llm.calls == []
    assert at.session_state.pending is None
    assert len(at.chat_message) == 0
    # The refusal shows in the chat area (from the send step) and in the sidebar.
    assert [w.value for w in at.main.warning] == [INJECTION_REFUSAL]
    assert [w.value for w in at.sidebar.warning] == [INJECTION_REFUSAL]
    # The message was fine, so it is kept for copying once the role is fixed.
    assert [code.value for code in at.code] == ["Tell me about yourself."]
    # Logged by the send step only: the sidebar check runs with log=False.
    guard_logs = [r.getMessage() for r in caplog.records if r.name == "src.guard"]
    assert guard_logs == [f"Blocked role: patterns=mode_override length={len(ROLE_ATTACK)}"]


def test_too_long_role_is_cut_to_the_limit(fake_llm):
    """Streamlit cuts a role over the limit to MAX_ROLE_CHARS before the app sees it."""
    at = start()
    at.sidebar.text_input(key="role").set_value("a" * (MAX_ROLE_CHARS + 20)).run(timeout=30)
    assert not at.exception
    assert at.session_state.role == "a" * MAX_ROLE_CHARS
    at.chat_input[0].set_value("Tell me about yourself.").run(timeout=30)
    assert not at.warning
    assert len(fake_llm.calls) == 1
