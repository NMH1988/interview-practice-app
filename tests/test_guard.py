import pytest

from src.config import MAX_INPUT_CHARS, MAX_ROLE_CHARS
from src.guard import (
    GuardError,
    InvalidInputError,
    clean_input,
    is_blank,
    validate_input,
    validate_role,
)

# Invisible characters are built with chr() so they cannot be lost or mangled in the source.
NUL, BEL, ESC, DEL, NEL = chr(0x00), chr(0x07), chr(0x1B), chr(0x7F), chr(0x85)
ZWSP, BOM = chr(0x200B), chr(0xFEFF)
HANGUL_FILLER, BRAILLE_BLANK = chr(0x3164), chr(0x2800)
COMBINING_GRAPHEME_JOINER, VARIATION_SELECTOR_16 = chr(0x034F), chr(0xFE0F)
COMBINING_ACUTE = chr(0x0301)


def test_default_limit_is_2000():
    """The ticket's default limit of 2000 characters lives in config.py."""
    assert MAX_INPUT_CHARS == 2000


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "\n\t \n",
        f"{NUL}{BEL} {ESC}",
        f"{ZWSP} {ZWSP}",
        BOM,
        f"{HANGUL_FILLER}{HANGUL_FILLER}",
        f"{BRAILLE_BLANK} {BRAILLE_BLANK}",
        f"{COMBINING_GRAPHEME_JOINER}{VARIATION_SELECTOR_16}",
        None,
    ],
    ids=[
        "empty",
        "spaces",
        "whitespace-mix",
        "only-controls",
        "zero-width",
        "bom",
        "hangul-filler",
        "braille-blank",
        "only-marks",
        "none",
    ],
)
def test_empty_or_blank_looking_input_is_rejected(text):
    """Nothing visible left after cleaning and trimming is rejected with a clear message."""
    with pytest.raises(InvalidInputError, match="type a message"):
        validate_input(text)


@pytest.mark.parametrize(
    "code_point",
    [0x115F, 0x1160, 0x3164, 0xFFA0, 0x2800],
    ids=["U+115F", "U+1160", "U+3164", "U+FFA0", "U+2800"],
)
def test_each_blank_looking_filler_is_rejected(code_point):
    """Each filler that renders as empty space is rejected; listed here so guard typos fail."""
    with pytest.raises(InvalidInputError, match="type a message"):
        validate_input(chr(code_point) * 3)


@pytest.mark.parametrize(
    "text",
    ["", "   ", f"{ZWSP} {BOM}", f"{HANGUL_FILLER}{BRAILLE_BLANK}", COMBINING_GRAPHEME_JOINER],
    ids=["empty", "spaces", "zero-width", "fillers", "only-mark"],
)
def test_is_blank_is_true_for_text_that_shows_nothing(text):
    """Text that shows as nothing on screen counts as blank, the same as validate_input sees it."""
    assert is_blank(text)


@pytest.mark.parametrize("text", ["a", f"{ZWSP}a{ZWSP}", f"e{COMBINING_ACUTE}", "  ."])
def test_is_blank_is_false_once_anything_shows(text):
    """One visible character, even among invisible ones, makes the text not blank."""
    assert not is_blank(text)


def test_combining_marks_inside_words_are_kept():
    """Marks only count as blank on their own; on a letter they are normal text."""
    text = f"Cafe{COMBINING_ACUTE} and more"
    assert validate_input(text) == text


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


@pytest.mark.parametrize(
    "role",
    ["", "   ", f"{ZWSP}{BOM}", f"{NUL}{BEL}", None],
    ids=["empty", "spaces", "zero-width", "only-controls", "none"],
)
def test_blank_role_is_rejected(role):
    """A role with nothing visible in it is rejected with the 'enter the role' message."""
    with pytest.raises(InvalidInputError, match="enter the role"):
        validate_role(role)


def test_role_over_the_limit_is_rejected():
    """A role one character over the limit is rejected and the message says how long it is."""
    with pytest.raises(InvalidInputError, match="role is too long") as info:
        validate_role("a" * (MAX_ROLE_CHARS + 1))
    assert f"{MAX_ROLE_CHARS + 1}" in str(info.value)


def test_role_of_exactly_the_limit_is_accepted():
    """A role of exactly the limit passes unchanged."""
    role = "a" * MAX_ROLE_CHARS
    assert validate_role(role) == role


def test_role_is_cleaned_and_trimmed():
    """Control characters and surrounding spaces are removed from the role."""
    assert validate_role(f"  Data{BEL} Engineer  ") == "Data Engineer"


def test_role_is_folded_onto_one_line():
    """Line breaks and tabs inside the role become single spaces, so it cannot add prompt lines."""
    assert validate_role("Engineer\r\n\nfor\tcloud teams") == "Engineer for cloud teams"


def test_role_limit_counts_the_folded_text():
    """The limit measures the role after folding, so extra line breaks do not push it over."""
    role = "a" * 30 + "\n\n\n" + "b" * 29
    assert validate_role(role) == "a" * 30 + " " + "b" * 29


def test_blank_role_message_does_not_mention_the_ui():
    """The guard's message works outside the app, so it does not point to a sidebar."""
    with pytest.raises(InvalidInputError) as info:
        validate_role("")
    assert "sidebar" not in str(info.value).lower()


def test_custom_role_limit_is_respected():
    """The role limit can be passed in, so the check does not depend on the config value."""
    with pytest.raises(InvalidInputError):
        validate_role("abcdef", max_chars=5)
    assert validate_role("abcde", max_chars=5) == "abcde"
