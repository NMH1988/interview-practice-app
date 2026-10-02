"""Security guard: checks user input before any tokens are spent on the LLM."""

import unicodedata

from src.config import MAX_INPUT_CHARS, MAX_ROLE_CHARS

# Control characters (Unicode category Cc) that are still normal text and must be kept.
_KEPT_CONTROLS = frozenset("\n\t")
# Kept inside text, but a message made only of these looks blank: format characters
# (zero-width space, BOM) and combining marks / variation selectors.
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Mn"})
# Letters and symbols that render as empty space: Hangul fillers and the blank Braille pattern.
_BLANK_LOOKING = frozenset(map(chr, (0x115F, 0x1160, 0x3164, 0xFFA0, 0x2800)))


class GuardError(ValueError):
    """Raised when the guard blocks a message; `str(exc)` is safe to show the user."""


class InvalidInputError(GuardError):
    """Raised when a message or the role is empty or too long."""


def clean_input(text: str) -> str:
    """Return `text` with line breaks unified to `\\n` and other control characters removed."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return "".join(ch for ch in text if ch in _KEPT_CONTROLS or unicodedata.category(ch) != "Cc")


def _looks_blank(ch: str) -> bool:
    """Return True if `ch` shows as nothing on screen when it stands alone."""
    return ch.isspace() or ch in _BLANK_LOOKING or unicodedata.category(ch) in _INVISIBLE_CATEGORIES


def _validated(text: str | None, max_chars: int, blank_msg: str, what: str) -> str:
    """Return `text` cleaned and trimmed, or raise `InvalidInputError` if blank or too long."""
    cleaned = clean_input(text or "").strip()
    if all(_looks_blank(ch) for ch in cleaned):
        raise InvalidInputError(blank_msg)
    if len(cleaned) > max_chars:
        raise InvalidInputError(
            f"{what} is too long ({len(cleaned):,} characters). "
            f"Please shorten it to {max_chars:,} characters or fewer."
        )
    return cleaned


def validate_input(text: str | None, max_chars: int = MAX_INPUT_CHARS) -> str:
    """Return the cleaned, trimmed message, or raise `InvalidInputError` if it cannot be sent."""
    return _validated(text, max_chars, "Please type a message before sending.", "Your message")


def validate_role(role: str | None, max_chars: int = MAX_ROLE_CHARS) -> str:
    """Return the cleaned role on one line, or raise `InvalidInputError` if blank or too long."""
    # Fold line breaks and tabs first, so the limit measures the text that goes into the prompt.
    one_line = " ".join(clean_input(role or "").split())
    return _validated(
        one_line, max_chars, "Please enter the role you are practising for.", "The role"
    )
