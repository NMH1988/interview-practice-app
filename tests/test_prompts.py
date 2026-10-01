import pytest

from src.prompts import INTERVIEW_TYPES, STRATEGIES


def test_at_least_five_strategies_registered():
    """The registry holds at least five strategies, all callable."""
    assert len(STRATEGIES) >= 5
    assert all(callable(strategy) for strategy in STRATEGIES.values())


@pytest.mark.parametrize("interview_type", INTERVIEW_TYPES)
@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_strategy_returns_non_empty_prompt(name, interview_type):
    """Each strategy returns a non-empty prompt naming the role and interview type."""
    prompt = STRATEGIES[name]("Data Analyst", interview_type)
    assert isinstance(prompt, str)
    assert prompt.strip()
    assert "Data Analyst" in prompt
    assert interview_type in prompt
