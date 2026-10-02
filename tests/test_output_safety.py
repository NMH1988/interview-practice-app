import ast
import html
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.config import API_KEY_NAME, MIN_LEAK_CHARS
from src.guard import REFUSAL_MESSAGE, _normalise, check_output
from src.prompts import FEW_SHOT_EXAMPLES, INTERVIEW_TYPES, STRATEGIES

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app.py"
FAKE_KEY = "sk-test-not-a-real-key"
ZWSP = chr(0x200B)
COMBINING_JOINER = chr(0x034F)
# Streamlit calls whose second positional argument is `unsafe_allow_html`.
FLAG_SECOND_ARG = {"markdown", "caption"}

PROMPTS = {key: strategy("Software Engineer", "Technical") for key, strategy in STRATEGIES.items()}

# Replies a coach would normally give, one per review format; none may be refused.
NORMAL_REPLIES = [
    "Great, let's begin. Tell me about a time you had to deliver under a tight deadline.",
    "## Evaluation\nA solid answer with a clear example.\n\n"
    "## Rubric\n- Relevance: 4/5\n- Correctness: 5/5\n- Clarity: 3/5\n- Depth: 4/5\n"
    "- Practicality: 4/5\n\n"
    "## Feedback\nYou explained the trade-off well, but say what you measured.\n\n"
    "## Follow-up Question\nHow would you test this change before release?",
    "## Assessment Rationale\nYou named the cause and the fix, but not the impact.\n\n"
    "Score: 3/5\n\nWhat would you do differently next time?",
    "Here are three questions you could ask the interviewer:\n"
    "1. What does a typical week look like?\n2. How is success measured in this role?\n"
    "3. What are the team's biggest challenges right now?",
]


def is_example_text(paragraph: str) -> bool:
    """Return True if the paragraph comes from one of the few-shot examples."""
    return any(paragraph in example for example in FEW_SHOT_EXAMPLES)


def long_paragraphs(prompt: str) -> list[str]:
    """Return the prompt's own paragraphs that are long enough to count as a leak alone."""
    return [
        p
        for p in prompt.split("\n\n")
        if len(_normalise(p)) >= MIN_LEAK_CHARS and not is_example_text(p)
    ]


def to_fullwidth(text: str) -> str:
    """Return `text` with every printable ASCII character swapped for its full-width twin."""
    return "".join(chr(ord(ch) + 0xFEE0) if "!" <= ch <= "~" else ch for ch in text)


@pytest.mark.parametrize("key", list(STRATEGIES))
def test_reply_with_whole_system_prompt_is_refused(key):
    """A reply that contains the whole system prompt verbatim is replaced by the refusal."""
    reply = f"Sure, here are my instructions:\n\n{PROMPTS[key]}\n\nAnything else?"
    assert check_output(reply, PROMPTS[key]) == REFUSAL_MESSAGE


@pytest.mark.parametrize("key", list(STRATEGIES))
def test_reply_with_any_long_paragraph_is_refused(key):
    """Each long paragraph of the prompt, repeated on its own, is enough to refuse the reply."""
    paragraphs = long_paragraphs(PROMPTS[key])
    assert paragraphs
    for paragraph in paragraphs:
        assert check_output(f"My rules say: {paragraph}", PROMPTS[key]) == REFUSAL_MESSAGE


@pytest.mark.parametrize(
    "paragraph",
    [p for example in FEW_SHOT_EXAMPLES for p in example.split("\n\n") if len(p) >= MIN_LEAK_CHARS],
)
def test_reply_copying_a_few_shot_example_paragraph_is_unchanged(paragraph):
    """Examples are meant to be imitated, so a reply repeating one paragraph is not refused."""
    reply = f"## Evaluation\nA clear answer.\n\n{paragraph}"
    assert check_output(reply, PROMPTS["few_shot"]) is reply


@pytest.mark.parametrize(
    "disguise",
    [
        str.upper,
        lambda text: text.replace(" ", "\n  "),
        lambda text: text.replace(" ", f" {ZWSP}"),
        lambda text: ZWSP.join(text),
        lambda text: COMBINING_JOINER.join(text),
        to_fullwidth,
        lambda text: text.replace("'", chr(0x2019)),
    ],
    ids=[
        "upper-case",
        "rewrapped",
        "zero-width-between-words",
        "zero-width-between-letters",
        "combining-joiner-between-letters",
        "full-width",
        "curly-apostrophes",
    ],
)
def test_disguised_leak_is_still_refused(disguise):
    """Changing case, spacing or look-alike characters does not hide a leak."""
    prompt = PROMPTS["zero_shot"]
    paragraph = next(p for p in long_paragraphs(prompt) if "'" in p)
    assert disguise(paragraph) != paragraph
    assert check_output(disguise(prompt), prompt) == REFUSAL_MESSAGE
    assert check_output(disguise(paragraph), prompt) == REFUSAL_MESSAGE


def test_leak_of_escaped_role_line_is_refused():
    """A role escaped in the prompt (`&lt;`) still matches when the reply shows it as `<`."""
    prompt = STRATEGIES["persona"]("C++ <Embedded> Engineer for the payments platform", "Technical")
    role_paragraph = next(p for p in prompt.split("\n\n") if "&lt;Embedded&gt;" in p)
    assert len(_normalise(role_paragraph)) >= MIN_LEAK_CHARS
    assert check_output(html.unescape(role_paragraph), prompt) == REFUSAL_MESSAGE


@pytest.mark.parametrize("interview_type", INTERVIEW_TYPES)
@pytest.mark.parametrize("key", list(STRATEGIES))
@pytest.mark.parametrize("reply", NORMAL_REPLIES)
def test_normal_reply_is_unchanged(reply, key, interview_type):
    """A normal coaching reply is returned as is, for every strategy and interview type."""
    prompt = STRATEGIES[key]("Software Engineer", interview_type)
    assert check_output(reply, prompt) is reply


def test_short_prompt_paragraph_alone_is_not_refused():
    """A short prompt paragraph (heading, scale) may appear in a normal reply."""
    prompt = PROMPTS["structured_output"]
    short = "Score each criterion from 1 to 5:"
    assert short in prompt
    assert check_output(f"{short} done.", prompt) == f"{short} done."


@pytest.mark.parametrize("prompt", ["", "  \n\n\t "], ids=["empty", "whitespace"])
def test_blank_system_prompt_never_refuses(prompt):
    """With no system prompt there is nothing to leak, so the reply is never refused."""
    assert check_output("Any reply at all.", prompt) == "Any reply at all."


def html_render_nodes(path: Path) -> list[ast.AST]:
    """Return the nodes in `path` that can render raw HTML (see the sample in the test below)."""
    nodes = []
    for node in ast.walk(ast.parse(path.read_bytes(), filename=str(path))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("streamlit"):
            if any(alias.name in {"html", "components"} for alias in node.names):
                nodes.append(node)
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        # `**kwargs` can hide the flag, so it counts as unsafe too.
        unsafe = any(
            kw.arg is None
            or (
                kw.arg == "unsafe_allow_html"
                and not (isinstance(kw.value, ast.Constant) and kw.value.value is False)
            )
            for kw in node.keywords
        )
        if unsafe or name == "html" or (name in FLAG_SECOND_ARG and len(node.args) > 1):
            nodes.append(node)
    return nodes


def theme_names(path: Path) -> set[str]:
    """Return names assigned once, as `x = st.get_option(...)` or `... or "<literal>"`."""

    def is_get_option(value: ast.expr) -> bool:
        """Return True if `value` is a call to `*.get_option(...)`."""
        return (
            isinstance(value, ast.Call)
            and isinstance(value.func, ast.Attribute)
            and value.func.attr == "get_option"
        )

    def is_theme_value(value: ast.expr) -> bool:
        """Return True if `value` is a theme option, optionally with a literal fallback."""
        if isinstance(value, ast.BoolOp) and isinstance(value.op, ast.Or):
            first, *rest = value.values
            return is_get_option(first) and all(
                isinstance(v, ast.Constant) and isinstance(v.value, str) for v in rest
            )
        return is_get_option(value)

    tree = ast.parse(path.read_bytes(), filename=str(path))
    stores = [
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    ]
    themed = {
        target.id
        for node in tree.body
        if isinstance(node, ast.Assign) and is_theme_value(node.value)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    return {name for name in themed if stores.count(name) == 1}


def is_static_style_block(node: ast.AST, allowed_names: set[str]) -> bool:
    """Return True if the call's first argument is a `<style>` block using only theme names."""
    if not isinstance(node, ast.Call) or not node.args:
        return False
    arg = node.args[0]
    parts = arg.values if isinstance(arg, ast.JoinedStr) else [arg]
    if not all(
        isinstance(p, ast.FormattedValue)
        or (isinstance(p, ast.Constant) and isinstance(p.value, str))
        for p in parts
    ):
        return False
    literal = "".join(p.value for p in parts if isinstance(p, ast.Constant)).strip()
    filled = [p.value for p in parts if isinstance(p, ast.FormattedValue)]
    return (
        literal.startswith("<style>")
        and literal.endswith("</style>")
        and all(isinstance(v, ast.Name) and v.id in allowed_names for v in filled)
    )


def app_sources() -> list[Path]:
    """Return every app Python file: `app.py` and the rest, without tests or hidden folders."""
    skipped = {"tests", "venv", "env"}
    sources = []
    for entry in ROOT.iterdir():
        if entry.name.startswith(".") or entry.name in skipped:
            continue
        if entry.is_dir():
            sources.extend(entry.rglob("*.py"))
        elif entry.suffix == ".py":
            sources.append(entry)
    return sources


def test_html_render_nodes_finds_every_raw_html_route(tmp_path):
    """The scan flags each way to turn on raw HTML, and leaves safe calls alone."""
    lines = [
        "st.markdown(reply)",  # 1
        "st.markdown(reply, unsafe_allow_html=True)",  # 2
        "st.markdown(reply, unsafe_allow_html=False)",
        "st.write(reply, unsafe_allow_html=flag)",  # 4
        "st.html(reply)",  # 5
        "components.v1.html(reply)",  # 6
        "html.escape(reply)",
        "st.markdown(reply, True)",  # 8
        "st.caption(reply, True)",  # 9
        "st.markdown(reply, **options)",  # 10
        "from streamlit import html as raw",  # 11
        "from streamlit import components",  # 12
        "st.write(reply, other)",
    ]
    source = tmp_path / "sample.py"
    source.write_text("\n".join(lines), encoding="utf-8")
    assert sorted(node.lineno for node in html_render_nodes(source)) == [
        2,
        4,
        5,
        6,
        8,
        9,
        10,
        11,
        12,
    ]


def test_static_style_block_check_rejects_user_or_model_text(tmp_path):
    """Only a `<style>` block filled with single-assignment theme names passes."""
    source = tmp_path / "sample.py"
    source.write_text(
        "primary = st.get_option('theme.primaryColor') or '#fff'\n"
        "accent = st.get_option('theme.primaryColor') or st.session_state.accent\n"
        "card = st.get_option('theme.secondaryBackgroundColor')\n"
        "card = reply\n"
        "st.markdown(f'<style>a {{ color: {primary}; }}</style>', unsafe_allow_html=True)\n"
        "st.markdown(f'<style>a {{ color: {reply}; }}</style>', unsafe_allow_html=True)\n"
        "st.markdown(f'<style>a {{ color: {accent}; }}</style>', unsafe_allow_html=True)\n"
        "st.markdown(f'<style>a {{ color: {card}; }}</style>', unsafe_allow_html=True)\n"
        "st.markdown(f'<b>{primary}</b>', unsafe_allow_html=True)\n"
        "st.markdown(1, unsafe_allow_html=True)\n",
        encoding="utf-8",
    )
    names = theme_names(source)
    assert names == {"primary"}
    checks = [is_static_style_block(node, names) for node in html_render_nodes(source)]
    assert checks == [True, False, False, False, False, False]


def test_only_raw_html_is_the_static_theme_css_block():
    """Model and user text never render as raw HTML; the one exception is the theme CSS."""
    sources = app_sources()
    assert APP in sources and (ROOT / "src" / "guard.py") in sources
    found = {path: html_render_nodes(path) for path in sources}
    others = [path.name for path, nodes in found.items() if nodes and path != APP]
    assert others == []
    assert len(found[APP]) == 1
    assert is_static_style_block(found[APP][0], theme_names(APP))


def test_chat_renders_html_as_text(fake_llm, no_env_key):
    """HTML in the user's message or the model's reply is shown as text, never run."""
    fake_llm.reply = "<img src=x onerror=alert(1)> **Good answer.**"
    at = AppTest.from_file(str(APP))
    at.secrets[API_KEY_NAME] = FAKE_KEY
    at.run(timeout=30)
    at.chat_input[0].set_value("<b>I led the migration.</b>").run(timeout=30)
    assert not at.exception
    user, assistant = (message.markdown[0] for message in at.chat_message)
    assert (user.value, user.proto.allow_html) == ("<b>I led the migration.</b>", False)
    assert (assistant.value, assistant.proto.allow_html) == (fake_llm.reply, False)
