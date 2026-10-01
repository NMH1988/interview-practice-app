"""App-wide settings: allowed models and model defaults."""

ALLOWED_MODELS: tuple[str, ...] = ("openai/gpt-5-mini",)
DEFAULT_MODEL = "openai/gpt-5-mini"

MIN_TEMPERATURE = 0.0
MAX_TEMPERATURE = 1.5
DEFAULT_TEMPERATURE = 0.7
