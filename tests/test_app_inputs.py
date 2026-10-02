from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import (
    ALLOWED_MODELS,
    API_KEY_NAME,
    DEFAULT_MODEL,
    DEFAULT_ROLE,
    DEFAULT_SENIORITY,
    DEFAULT_TEMPERATURE,
    MAX_ROLE_CHARS,
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
)
from src.prompts import INTERVIEW_TYPES, SENIORITY_LEVELS

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


def test_temperature_slider_range_and_default(fake_llm):
    """The sidebar temperature slider runs 0.0-1.5 in 0.1 steps and starts at 0.7."""
    slider = start().sidebar.slider(key="temperature")
    assert slider.label == "Temperature"
    assert (slider.min, slider.max) == (MIN_TEMPERATURE, MAX_TEMPERATURE) == (0.0, 1.5)
    assert slider.step == pytest.approx(0.1)
    assert slider.value == DEFAULT_TEMPERATURE == 0.7
    assert fake_llm.calls == []


def test_default_model_and_temperature_reach_the_llm(fake_llm):
    """Without changes, the LLM call gets the default model and temperature."""
    at = start()
    at.chat_input[0].set_value("Tell me about yourself.").run(timeout=30)
    assert not at.exception
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["model"] == DEFAULT_MODEL
    assert fake_llm.calls[0]["temperature"] == DEFAULT_TEMPERATURE


def test_changed_model_and_temperature_reach_the_llm(fake_llm):
    """A model and temperature picked in the sidebar are the ones sent to the LLM."""
    at = start()
    at.sidebar.selectbox(key="model").set_value("openai/gpt-5-nano")
    at.sidebar.slider(key="temperature").set_value(1.2)
    at.run(timeout=30)
    at.chat_input[0].set_value("Tell me about yourself.").run(timeout=30)
    assert not at.exception
    assert len(fake_llm.calls) == 1
    assert fake_llm.calls[0]["model"] == "openai/gpt-5-nano"
    assert fake_llm.calls[0]["temperature"] == pytest.approx(1.2)


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


def test_blank_role_shows_warning_and_skips_llm(fake_llm):
    """A blank role blocks the message with a warning and no LLM call."""
    at = start()
    at.sidebar.text_input(key="role").set_value("   ").run(timeout=30)
    at.chat_input[0].set_value("Tell me about yourself.").run(timeout=30)
    assert not at.exception
    assert len(at.warning) == 1
    assert "enter the role" in at.warning[0].value
    assert fake_llm.calls == []
    assert len(at.chat_message) == 0


def test_too_long_role_is_cut_to_the_limit(fake_llm):
    """Streamlit cuts a role over the limit to MAX_ROLE_CHARS before the app sees it."""
    at = start()
    at.sidebar.text_input(key="role").set_value("a" * (MAX_ROLE_CHARS + 20)).run(timeout=30)
    assert not at.exception
    assert at.session_state.role == "a" * MAX_ROLE_CHARS
    at.chat_input[0].set_value("Tell me about yourself.").run(timeout=30)
    assert not at.warning
    assert len(fake_llm.calls) == 1
