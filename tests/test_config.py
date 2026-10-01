from src import config


def test_allowed_models():
    """Allowed models are exactly the three chat models from the brief."""
    assert config.ALLOWED_MODELS == ("openai/gpt-5-mini", "openai/gpt-5-nano", "openai/gpt-5")


def test_default_model_is_allowed():
    """Default model is gpt-5-mini and is in the allowed list."""
    assert config.DEFAULT_MODEL == "openai/gpt-5-mini"
    assert config.DEFAULT_MODEL in config.ALLOWED_MODELS


def test_default_temperature_in_range():
    """Default temperature lies within the allowed temperature range."""
    assert config.MIN_TEMPERATURE <= config.DEFAULT_TEMPERATURE <= config.MAX_TEMPERATURE
