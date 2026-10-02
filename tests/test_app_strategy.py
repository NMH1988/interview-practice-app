from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import API_KEY_NAME
from src.prompts import STRATEGIES, STRATEGY_LABELS

APP = Path(__file__).resolve().parent.parent / "app.py"

pytestmark = pytest.mark.usefixtures("no_env_key")


def test_strategy_select_lists_every_strategy_by_label(fake_llm):
    """The sidebar select shows every registered strategy by its technique label, in order."""
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = "sk-test-not-a-real-key"
    at.run(timeout=30)
    assert not at.exception
    select = at.sidebar.selectbox(key="strategy")
    assert select.label == "Prompt strategy"
    assert select.options == [STRATEGY_LABELS[key] for key in STRATEGIES]
    assert select.value == next(iter(STRATEGIES))
    assert fake_llm.calls == []


def test_chosen_strategy_is_stored_by_its_key(fake_llm):
    """Picking a label stores the strategy's key in session state, where T5.2 will read it."""
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = "sk-test-not-a-real-key"
    at.run(timeout=30)
    at.sidebar.selectbox(key="strategy").set_value("persona").run(timeout=30)
    assert not at.exception
    assert at.session_state.strategy == "persona"
    assert at.session_state.strategy in STRATEGIES
    assert fake_llm.calls == []
