"""Security guard: checks user input before any tokens are spent, and the LLM's reply after."""

import html
import unicodedata

from src.config import MAX_INPUT_CHARS, MAX_ROLE_CHARS, MIN_LEAK_CHARS
from src.prompts import FEW_SHOT_EXAMPLES

# Shown instead of a reply that repeats the system prompt.
REFUSAL_MESSAGE = "Sorry, I can't share my instructions. Let's get back to your interview practice."

# Control characters (Unicode category Cc) that are still normal text and must be kept.
_KEPT_CONTROLS = frozenset("\n\t")
# Kept inside text, but a message made only of these looks blank: format characters
# (zero-width space, BOM) and combining marks / variation selectors.
_INVISIBLE_CATEGORIES = frozenset({"Cf", "Mn"})
# Letters and symbols that render as empty space: Hangul fillers and the blank Braille pattern.
_BLANK_LOOKING = frozenset(map(chr, (0x115F, 0x1160, 0x3164, 0xFFA0, 0x2800)))
# Curly quotes a model may write where the prompt has straight ones.
_STRAIGHT_QUOTES = str.maketrans(
    {chr(0x2018): "'", chr(0x2019): "'", chr(0x201C): '"', chr(0x201D): '"'}
)


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


def _normalise(text: str) -> str:
    """Return `text` in a form where case, spacing and look-alike characters do not matter."""
    # NFKC first, so a full-width "&lt;" is unescaped too.
    text = html.unescape(unicodedata.normalize("NFKC", text)).translate(_STRAIGHT_QUOTES)
    text = "".join(ch for ch in text if unicodedata.category(ch) not in _INVISIBLE_CATEGORIES)
    return " ".join(text.split()).casefold()


# Few-shot example paragraphs are made to be imitated, so repeating one is not a leak.
_PUBLIC_PARAGRAPHS = frozenset(
    _normalise(p) for example in FEW_SHOT_EXAMPLES for p in example.split("\n\n")
)


def check_output(reply: str, system_prompt: str) -> str:
    """Return `reply`, or `REFUSAL_MESSAGE` if it repeats the system prompt or a long part of it."""
    prompt = _normalise(system_prompt)
    if not prompt:
        return reply
    seen = _normalise(reply)
    # The whole prompt, plus each long paragraph, since a leak usually copies only parts.
    paragraphs = {_normalise(p) for p in system_prompt.split("\n\n")} - _PUBLIC_PARAGRAPHS
    leaked = [prompt, *(p for p in paragraphs if len(p) >= MIN_LEAK_CHARS)]
    return REFUSAL_MESSAGE if any(part in seen for part in leaked) else reply
