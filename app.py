from collections.abc import Iterator
from contextlib import closing
from datetime import date, timedelta
from itertools import chain

import pandas as pd
import streamlit as st

from src import llm
from src.config import (
    ALLOWED_MODELS,
    API_KEY_NAME,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_ROLE,
    DEFAULT_SENIORITY,
    DEFAULT_TEMPERATURE,
    MAX_ROLE_CHARS,
    MAX_TEMPERATURE,
    MIN_TEMPERATURE,
    MissingAPIKeyError,
    SecretsFileError,
    get_api_key,
)
from src.guard import GuardError, check_output, validate_input, validate_role
from src.prompts import (
    INTERVIEW_TYPES,
    SENIORITY_LEVELS,
    STRATEGIES,
    STRATEGY_LABELS,
    build_messages,
    build_user_prompt,
)

st.set_page_config(page_title="Interview Practice", layout="wide")

# Theme colours come from .streamlit/config.toml so the cards always match the app.
primary = st.get_option("theme.primaryColor") or "#4f46e5"
card_bg = st.get_option("theme.secondaryBackgroundColor") or "#f1f2f9"
text = st.get_option("theme.textColor") or "#1f2430"

st.markdown(
    f"""
    <style>
    [data-testid="stMetric"] {{
        background: {card_bg};
        border-left: 4px solid {primary};
        border-radius: 8px;
        padding: 12px 16px;
    }}
    [data-testid="stMetricLabel"] {{ color: {text}; opacity: 0.75; }}
    [data-testid="stMetricValue"] {{ color: {primary}; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Interview Practice")

try:
    get_api_key()
except SecretsFileError:
    st.error(
        "**`.streamlit/secrets.toml` could not be parsed.** Check that the key is in quotes, "
        f'for example `{API_KEY_NAME} = "sk-or-v1-..."`, then reload this page.',
        icon="🔑",
    )
    st.stop()
except MissingAPIKeyError:
    st.error(
        f"**OpenRouter API key missing.** Add `{API_KEY_NAME}` to `.streamlit/secrets.toml` "
        "(copy `.streamlit/secrets.toml.example`) or set it as an environment variable, "
        "then reload this page. See the README for setup steps.",
        icon="🔑",
    )
    st.stop()


@st.cache_data
def load_sessions() -> pd.DataFrame:
    """Return 60 days of made-up practice sessions for the dashboard."""
    # Placeholder data: replace with your real practice-session records.
    days = pd.date_range(end=date.today(), periods=60, freq="D")
    return pd.DataFrame(
        {
            "date": days,
            "questions": [3 + (i * 7) % 6 for i in range(60)],
            "score": [60 + (i * 11) % 40 for i in range(60)],
        }
    )


df = load_sessions()

# Chat state. Each history turn is {"role", "content" (shown in the chat), "sent" (sent to the
# LLM)}. "pending" holds a submitted message until its reply is in; "notice" is a warning or
# error to show once, since the run that sets it ends with st.rerun().
st.session_state.setdefault("history", [])
st.session_state.setdefault("pending", None)
st.session_state.setdefault("notice", None)

INTERRUPTED = "The answer was interrupted before it finished."


def queue_message() -> None:
    """Keep the submitted chat message as pending, so this run can lock the input first."""
    st.session_state.pending = st.session_state.chat_box


def new_session() -> None:
    """Forget the chat history and anything still waiting to be sent or shown."""
    st.session_state.history = []
    st.session_state.pending = None
    st.session_state.notice = None


def reply_pieces(
    messages: list[dict],
    system_prompt: str,
    model: str,
    temperature: float,
    clean: str,
    user_prompt: str,
) -> Iterator[str]:
    """Stream the reply's pieces, saving the turn's outcome before Streamlit gets control back."""
    # The request goes out on this first next(), so from here a stopped run must not resend the
    # message; until the reply is in or has failed, it counts as interrupted.
    st.session_state.pending = None
    st.session_state.notice = {"kind": "error", "text": INTERRUPTED, "unsent": clean}
    received = []
    try:
        # closing(): when this generator is closed mid-stream, close the inner one (and its
        # connection) too, rather than leaving that to garbage collection.
        with closing(llm.stream(messages, model, temperature, DEFAULT_MAX_TOKENS)) as stream:
            for piece in stream:
                received.append(piece)
                yield piece
    except llm.LLMError as exc:
        # Only the fixed message: the chained SDK error holds the raw response body. The turn
        # stays out of the history, so it alternates user/assistant.
        st.session_state.notice = {"kind": "error", "text": str(exc), "unsent": clean}
        raise
    # Saved before st.write_stream draws the final text, where a requested rerun could stop it.
    # The whole reply is checked before it is stored, so a leaked prompt never stays in the chat
    # or reaches the next request; the streamed text is replaced by the rerun that follows.
    reply = check_output("".join(received), system_prompt)
    st.session_state.notice = None
    st.session_state.history.extend(
        [
            {"role": "user", "content": clean, "sent": user_prompt},
            {"role": "assistant", "content": reply, "sent": reply},
        ]
    )


with st.sidebar:
    st.header("Session settings")
    model = st.selectbox(
        "Model", ALLOWED_MODELS, index=ALLOWED_MODELS.index(DEFAULT_MODEL), key="model"
    )
    # Shows each strategy by its technique label; the key picks the system prompt.
    strategy = st.selectbox(
        "Prompt strategy",
        list(STRATEGIES),
        format_func=STRATEGY_LABELS.__getitem__,
        key="strategy",
    )
    temperature = st.slider(
        "Temperature",
        MIN_TEMPERATURE,
        MAX_TEMPERATURE,
        DEFAULT_TEMPERATURE,
        step=0.1,
        key="temperature",
    )
    interview_type = st.selectbox("Interview type", INTERVIEW_TYPES, key="interview_type")
    # Streamlit cuts the value to max_chars on the server too; validate_role checks it again.
    role = st.text_input("Role", DEFAULT_ROLE, max_chars=MAX_ROLE_CHARS, key="role")
    # Checked here, not on send, so the chat input is locked before anything is typed.
    try:
        validate_role(role)
        role_ok = True
    except GuardError as exc:
        st.warning(str(exc), icon="✋")
        role_ok = False
    seniority = st.selectbox(
        "Seniority",
        SENIORITY_LEVELS,
        index=SENIORITY_LEVELS.index(DEFAULT_SENIORITY),
        key="seniority",
    )
    st.button("New session", on_click=new_session, icon="🔄")
    st.header("Filters")
    picked = st.date_input(
        "Date range",
        value=(date.today() - timedelta(days=29), date.today()),
        min_value=df["date"].min().date(),
        max_value=df["date"].max().date(),
    )

# While the user is mid-selection Streamlit returns a single date.
if len(picked) == 2:
    start, end = picked
    filtered = df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)]
else:
    filtered = df.iloc[0:0]

c1, c2, c3 = st.columns(3)
c1.metric("Sessions", len(filtered))
c2.metric("Questions answered", int(filtered["questions"].sum()))
c3.metric("Average score", f"{filtered['score'].mean():.0f}" if len(filtered) else "-")

st.line_chart(filtered.set_index("date")["score"])

for turn in st.session_state.history:
    st.chat_message(turn["role"]).markdown(turn["content"])

notice = st.session_state.notice
if notice is not None:
    if notice["kind"] == "warning":
        st.warning(notice["text"], icon="✋")
    else:
        st.error(notice["text"])
        st.caption("Your message was not sent. Copy it from here to try again:")
        st.code(notice["unsent"], language=None, wrap_lines=True)
    # Cleared only once drawn, so a run stopped mid-draw shows it on the next run instead.
    st.session_state.notice = None

# Drawn before the LLM call and locked while a reply is pending, so a second message cannot
# be sent (and cut this run short) while the first one is waiting.
st.chat_input(
    "Type your answer or question",
    key="chat_box",
    on_submit=queue_message,
    disabled=not role_ok or st.session_state.pending is not None,
)

# Chat turn: guard -> prompts -> streamed LLM reply. Every path clears "pending" and ends in
# st.rerun(), which unlocks the input. A rerun requested before the request is sent (e.g. a
# sidebar click) stops the run with the message still pending, so the next run sends it. One
# requested later stops the run at the next piece drawn; reply_pieces has already saved the
# outcome by then (reply, error or "interrupted"), so the message is never sent twice.
message = st.session_state.pending
if message is not None:
    try:
        try:
            clean = validate_input(message)
            # Also gives the cleaned role, and blocks a message that was queued before the role
            # was blanked (locking the input does not stop it).
            clean_role = validate_role(role)
        except GuardError as exc:
            st.session_state.notice = {"kind": "warning", "text": str(exc)}
            st.session_state.pending = None
        else:
            st.chat_message("user").markdown(clean)
            user_prompt = build_user_prompt(clean_role, interview_type, seniority, clean)
            system_prompt = STRATEGIES[strategy](clean_role, interview_type)
            messages = build_messages(system_prompt, st.session_state.history, user_prompt)
            pieces = reply_pieces(messages, system_prompt, model, temperature, clean, user_prompt)
            try:
                with st.chat_message("assistant"):
                    # The spinner covers the wait for the first piece (gpt-5 thinks first).
                    with st.spinner("Thinking..."):
                        first = next(pieces)
                    st.write_stream(chain([first], pieces), cursor="▌")
            except llm.LLMError:
                pass  # reply_pieces has saved the error; the rerun shows it.
            finally:
                # A run stopped mid-stream leaves the generator open; close it (and the
                # connection) now rather than whenever it is garbage-collected.
                pieces.close()
    except Exception:
        # A real bug, not Streamlit's rerun signal (a BaseException): without this the message
        # would stay pending, so every later run would lock the input and crash again.
        st.session_state.pending = None
        raise
    st.rerun()
