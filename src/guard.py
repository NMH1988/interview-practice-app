"""Security guard: checks user input before any tokens are spent on the LLM."""

import unicodedata

from src.config import MAX_INPUT_CHARS

# Control characters (Unicode category Cc) that are still normal text and must be kept.
_KEPT_CONTROLS = frozenset("\n\t")


class GuardError(ValueError):
    """Raised when the guard blocks a message; `str(exc)` is safe to show the user."""


class InvalidInputError(GuardError):
    """Raised when a message is empty or too long."""


def clean_input(text: str) -> str:
    """Return `text` with line breaks unified to `\\n` and other control characters removed."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(ch for ch in text if ch in _KEPT_CONTROLS or unicodedata.category(ch) != "Cc")


def validate_input(text: str | None, max_chars: int = MAX_INPUT_CHARS) -> str:
    """Return the cleaned, trimmed message, or raise `InvalidInputError` if it cannot be sent."""
    cleaned = clean_input(text or "").strip()
    # Format characters (zero-width space, BOM) are kept inside text but alone look blank.
    if all(ch.isspace() or unicodedata.category(ch) == "Cf" for ch in cleaned):
        raise InvalidInputError("Please type a message before sending.")
    if len(cleaned) > max_chars:
        raise InvalidInputError(
            f"Your message is too long ({len(cleaned):,} characters). "
            f"Please shorten it to {max_chars:,} characters or fewer."
        )
    return cleaned
