import pytest

from src.config import MAX_INPUT_CHARS
from src.guard import GuardError, InvalidInputError, clean_input, validate_input

# Invisible characters are built with chr() so they cannot be lost or mangled in the source.
NUL, BEL, ESC, DEL, NEL = chr(0x00), chr(0x07), chr(0x1B), chr(0x7F), chr(0x85)
ZWSP, BOM = chr(0x200B), chr(0xFEFF)


def test_default_limit_is_2000():
    """The ticket's default limit of 2000 characters lives in config.py."""
    assert MAX_INPUT_CHARS == 2000


@pytest.mark.parametrize(
    "text",
    ["", "   ", "\n\t \n", f"{NUL}{BEL} {ESC}", f"{ZWSP} {ZWSP}", BOM, None],
    ids=["empty", "spaces", "whitespace-mix", "only-controls", "zero-width", "bom", "none"],
)
def test_empty_or_whitespace_input_is_rejected(text):
    """Nothing left after cleaning and trimming is rejected with a clear message."""
    with pytest.raises(InvalidInputError, match="type a message"):
        validate_input(text)


@pytest.mark.parametrize("length", [MAX_INPUT_CHARS + 1, MAX_INPUT_CHARS * 5])
def test_over_limit_input_is_rejected(length):
    """Input longer than the limit is rejected, and the message names both sizes."""
    with pytest.raises(InvalidInputError, match="too long") as excinfo:
        validate_input("a" * length)
    assert f"{length:,}" in str(excinfo.value)
    assert f"{MAX_INPUT_CHARS:,}" in str(excinfo.value)


def test_input_of_exactly_the_limit_is_accepted():
    """A message of exactly the limit passes unchanged."""
    text = "a" * MAX_INPUT_CHARS
    assert validate_input(text) == text


def test_custom_limit_is_respected():
    """A caller-supplied max_chars overrides the default."""
    assert validate_input("abcde", max_chars=5) == "abcde"
    with pytest.raises(InvalidInputError):
        validate_input("abcdef", max_chars=5)


def test_invalid_input_error_is_a_guard_error():
    """Callers can catch every guard block through the GuardError base class."""
    assert issubclass(InvalidInputError, GuardError)


@pytest.mark.parametrize("char", [NUL, BEL, ESC, DEL, NEL], ids=["NUL", "BEL", "ESC", "DEL", "NEL"])
def test_control_characters_are_stripped(char):
    """C0, DEL and C1 control characters are removed from the message."""
    assert validate_input(f"I led{char} the team") == "I led the team"


def test_newlines_and_tabs_are_kept():
    """Normal line breaks and tabs inside the message survive cleaning."""
    text = "Situation: outage\nTask: fix it\n\tAction: rolled back"
    assert validate_input(text) == text


def test_windows_and_old_mac_line_breaks_become_newlines():
    """CRLF and lone CR turn into a single newline instead of being dropped."""
    assert clean_input("one\r\ntwo\rthree") == "one\ntwo\nthree"


def test_leading_and_trailing_whitespace_is_trimmed():
    """Surrounding whitespace is removed; inner whitespace is not."""
    assert validate_input("  \n hello  world \n ") == "hello  world"


def test_control_characters_do_not_count_towards_the_limit():
    """The limit applies to the cleaned text, so stripped characters are not counted."""
    text = "a" * MAX_INPUT_CHARS + NUL * 10
    assert validate_input(text) == "a" * MAX_INPUT_CHARS


def test_non_ascii_text_is_kept():
    """Accented letters and emoji are ordinary text, not control characters."""
    text = "Tôi đã dẫn dắt nhóm " + chr(0x1F680)
    assert validate_input(text) == text


def test_zero_width_characters_inside_text_are_kept():
    """Format characters are not stripped, so emoji joined with ZWJ stay intact."""
    family = chr(0x1F468) + chr(0x200D) + chr(0x1F469) + chr(0x200D) + chr(0x1F467)
    assert validate_input(f"My team {family}") == f"My team {family}"
