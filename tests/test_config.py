from src import config


def test_allowed_models():
    assert config.ALLOWED_MODELS == ("openai/gpt-5-mini",)


def test_default_model_is_allowed():
    assert config.DEFAULT_MODEL == "openai/gpt-5-mini"
    assert config.DEFAULT_MODEL in config.ALLOWED_MODELS


def test_default_temperature_in_range():
    assert config.MIN_TEMPERATURE <= config.DEFAULT_TEMPERATURE <= config.MAX_TEMPERATURE
